"""
Server-side hand landmark detection and sign classification.

Each WebSocket connection owns one MediaPipe HandLandmarker in VIDEO mode:
  * VIDEO mode tracks hands between frames (cheaper than re-detecting).
  * MediaPipe graphs are not safe to share across threads, and frames are
    processed in a worker thread so the event loop never blocks.

Frames arrive from the browser already mirrored (selfie view), matching the
desktop runner, so handedness labels match the training convention.
"""

import base64
import binascii
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend import config
from training.dataset_generator import CLASSES
from training.normalize import correct_aspect, normalize_dual_hands

log = logging.getLogger("signbridge.vision")

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    MEDIAPIPE_AVAILABLE = os.path.exists(config.TASK_PATH)
except Exception as exc:  # pragma: no cover - depends on platform wheels
    log.warning("MediaPipe unavailable: %s", exc)
    MEDIAPIPE_AVAILABLE = False


class FrameError(ValueError):
    """The client sent a frame we refuse to process."""


def decode_image(data_url: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL) JPEG/PNG into an RGB array."""
    encoded = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    # base64 inflates by 4/3; reject oversize payloads before decoding.
    if len(encoded) * 3 // 4 > config.WS_MAX_IMAGE_BYTES:
        raise FrameError("Frame too large")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FrameError("Frame is not valid base64") from exc
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FrameError("Frame is not a decodable image")
    if bgr.shape[0] > 1080 or bgr.shape[1] > 1920:
        raise FrameError("Frame dimensions too large")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class Classifier:
    def __init__(self, model: Any):
        self.model = model

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, features: List[float]) -> Tuple[str, float, List[Dict[str, Any]]]:
        if self.model is None or not any(features):
            return "NONE", 0.0, []
        probs = self.model.predict_proba(np.asarray(features, dtype=np.float32).reshape(1, -1))[0]
        order = np.argsort(probs)[::-1][:3]
        top3 = [{"sign": CLASSES[i], "confidence": round(float(probs[i]), 4)} for i in order]
        return CLASSES[int(order[0])], float(probs[order[0]]), top3


class HandTracker:
    """Per-connection landmark tracker. Not thread-safe; use from one worker at a time."""

    def __init__(self) -> None:
        self._landmarker = None
        self._t0 = time.monotonic()
        self._last_ts = -1

    @staticmethod
    def available() -> bool:
        return MEDIAPIPE_AVAILABLE

    def _ensure(self) -> None:
        if self._landmarker is None:
            opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=config.TASK_PATH),
                num_hands=2,
                running_mode=mp_vision.RunningMode.VIDEO,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure()
        ts = int((time.monotonic() - self._t0) * 1000)
        ts = max(ts, self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        hands: List[Dict[str, Any]] = []
        for idx, landmarks in enumerate(result.hand_landmarks or []):
            category = result.handedness[idx][0]
            hands.append({
                "label": category.category_name,
                "score": round(float(category.score), 3),
                "landmarks": [
                    {"x": round(lm.x, 5), "y": round(lm.y, 5), "z": round(lm.z, 5)} for lm in landmarks
                ],
            })
        return hands

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            finally:
                self._landmarker = None


def count_hands_in_images(data_urls: List[str]) -> int:
    """Max hands found in any image (still-image mode, private landmarker).
    Used to ground the AI vision call: no hands -> nothing to identify."""
    if not MEDIAPIPE_AVAILABLE:
        return -1  # unknown; caller decides
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.TASK_PATH),
        num_hands=2,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    best = 0
    with mp_vision.HandLandmarker.create_from_options(opts) as landmarker:
        for url in data_urls:
            rgb = decode_image(url)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            best = max(best, len(result.hand_landmarks or []))
    return best


def validate_client_hands(hands: Any) -> List[Dict[str, Any]]:
    """Accept landmarks computed client-side, but only in the expected shape."""
    if not isinstance(hands, list) or len(hands) > 2:
        raise FrameError("hands must be a list of at most 2 items")
    clean: List[Dict[str, Any]] = []
    for hand in hands:
        if not isinstance(hand, dict):
            raise FrameError("hand must be an object")
        lms = hand.get("landmarks")
        if not isinstance(lms, list) or len(lms) != 21:
            raise FrameError("each hand needs 21 landmarks")
        points = []
        for lm in lms:
            try:
                points.append({"x": float(lm["x"]), "y": float(lm["y"]), "z": float(lm.get("z", 0.0))})
            except (TypeError, KeyError, ValueError) as exc:
                raise FrameError("invalid landmark") from exc
        label = hand.get("label")
        clean.append({"label": label if label in ("Left", "Right") else "", "landmarks": points})
    return clean


class FrameResult:
    __slots__ = ("hands", "features", "sign", "confidence", "top3", "source", "ms")

    def __init__(self, hands, features, sign, confidence, top3, source, ms):
        self.hands, self.features, self.sign = hands, features, sign
        self.confidence, self.top3, self.source, self.ms = confidence, top3, source, ms


def process_frame(
    tracker: Optional[HandTracker],
    classifier: Classifier,
    image: Optional[str],
    hands: Optional[Any],
    custom: Any = None,
) -> FrameResult:
    """Blocking work for one frame: detect -> features -> built-in model vs taught signs."""
    started = time.perf_counter()
    aspect = 1.0
    if image:
        if tracker is None:
            raise FrameError("Server-side detection unavailable")
        rgb = decode_image(image)
        aspect = rgb.shape[1] / rgb.shape[0]
        detected = tracker.detect(rgb)
    else:
        detected = validate_client_hands(hands or [])

    features = normalize_dual_hands(correct_aspect(detected, aspect)) if detected else [0.0] * 126
    sign, conf, top3 = classifier.predict(features)
    source = "model"
    if custom is not None and detected:
        name, custom_conf = custom.match(features)
        if name and custom_conf > conf:
            sign, conf, source = name, custom_conf, "taught"
            top3 = [{"sign": name, "confidence": round(custom_conf, 4)}] + top3[:2]
    return FrameResult(detected, features, sign, conf, top3, source, (time.perf_counter() - started) * 1000)
