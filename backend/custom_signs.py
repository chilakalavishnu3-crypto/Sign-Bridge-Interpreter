"""
User-taught signs ("Teach a sign").

Any sign from any sign language can be taught at the counter: the camera
captures a few seconds of real hand landmarks, staff type what the sign means,
and from then on it is recognized by nearest-neighbour matching on the same
126-feature normalized vector the main model uses — and spoken aloud.

Storage is a small JSON file (landmark features only, never images).
"""

import json
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend import config

NAME_RE = re.compile(r"^[\w][\w \-'.?!]{0,39}$", re.UNICODE)
MAX_SIGNS = 200
MAX_SAMPLES_PER_SIGN = 300
MIN_SAMPLES = 12
SLOT = 63


class CustomSignError(ValueError):
    pass


def _presence(features: np.ndarray) -> Tuple[bool, bool]:
    return bool(np.any(features[..., :SLOT] != 0)), bool(np.any(features[..., SLOT:] != 0))


class CustomSignStore:
    def __init__(self, path: str, reserved_names: Optional[set] = None):
        self.path = path
        self.reserved = {n.upper() for n in (reserved_names or set())}
        self._lock = threading.Lock()
        self._signs: List[Dict[str, Any]] = []
        self._index = (np.zeros((0, 126), np.float32), np.zeros((0,), np.int32), np.zeros((0, 2), bool))
        self._load()

    # Persistence ---------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._signs = json.load(f).get("signs", [])
            except (OSError, ValueError):
                self._signs = []
        self._rebuild()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "signs": self._signs}, f, ensure_ascii=False)
        os.replace(tmp, self.path)  # atomic on the same volume

    def _rebuild(self) -> None:
        rows, labels = [], []
        for idx, sign in enumerate(self._signs):
            for sample in sign.get("samples", []):
                rows.append(sample)
                labels.append(idx)
        X = np.asarray(rows, dtype=np.float32).reshape(-1, 126)
        presence = np.stack([np.any(X[:, :SLOT] != 0, 1), np.any(X[:, SLOT:] != 0, 1)], 1) if len(X) else np.zeros((0, 2), bool)
        self._index = (X, np.asarray(labels, dtype=np.int32), presence)  # swapped atomically

    # Queries ----------------------------------------------------------------

    @staticmethod
    def public(sign: Dict[str, Any]) -> Dict[str, Any]:
        samples = sign.get("samples", [])
        # Median recorded hand shape: lets the signing avatar reproduce the
        # taught sign (126 normalized numbers; no image data).
        pose = np.round(np.median(np.asarray(samples, np.float32), axis=0), 3).tolist() if samples else None
        return {k: sign[k] for k in ("id", "name", "text", "text_lang", "translations", "created_at")} | {
            "samples": len(samples),
            "pose": pose,
        }

    def list(self) -> List[Dict[str, Any]]:
        return [self.public(s) for s in self._signs]

    def names(self) -> set:
        return {s["name"] for s in self._signs}

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return next((s for s in self._signs if s["name"] == name), None)

    def match(self, features: List[float]) -> Tuple[Optional[str], float]:
        """Nearest-neighbour match. Returns (name, confidence 0..1)."""
        X, labels, presence = self._index
        if not len(X):
            return None, 0.0
        q = np.asarray(features, dtype=np.float32)
        q_presence = np.array(_presence(q))
        if not q_presence.any():
            return None, 0.0
        candidates = np.all(presence == q_presence, axis=1)  # same hands visible
        if not candidates.any():
            return None, 0.0
        mask = np.concatenate([np.full(SLOT, q_presence[0]), np.full(SLOT, q_presence[1])])
        diff = (X[candidates][:, mask] - q[mask])
        dist = np.sqrt((diff ** 2).mean(axis=1))
        cand_labels = labels[candidates]

        best = int(np.argmin(dist))
        d1 = float(dist[best])
        label = int(cand_labels[best])
        others = dist[cand_labels != label]
        d2 = float(others.min()) if len(others) else float("inf")

        scale = config.CUSTOM_MATCH_SCALE
        conf = 1.0 / (1.0 + (d1 / scale) ** 2)
        if d2 < 1.3 * d1:  # another taught sign is almost as close: ambiguous
            conf *= 0.75
        return self._signs[label]["name"], conf

    # Mutations --------------------------------------------------------------

    def add(self, name: str, text: str, text_lang: str, translations: Dict[str, str],
            samples: List[List[float]], allowed_langs: List[str]) -> Dict[str, Any]:
        name = (name or "").strip()
        text = (text or "").strip()
        if not NAME_RE.match(name):
            raise CustomSignError("Name must be 1–40 letters, numbers or spaces.")
        if name.upper() in self.reserved:
            raise CustomSignError("That name is already a built-in sign.")
        if not text or len(text) > 200:
            raise CustomSignError("Meaning must be 1–200 characters.")
        if text_lang not in allowed_langs:
            raise CustomSignError("Unsupported language.")
        clean_tr = {
            lang: str(value).strip()[:200]
            for lang, value in (translations or {}).items()
            if lang in allowed_langs and str(value).strip()
        }
        if len(samples) < MIN_SAMPLES:
            raise CustomSignError(f"Need at least {MIN_SAMPLES} frames with hands visible — record again.")

        rounded = [[round(float(v), 4) for v in s] for s in samples[-MAX_SAMPLES_PER_SIGN:]]
        with self._lock:
            existing = self.get(name)
            if existing:  # teaching an existing name adds more examples
                existing["samples"] = (existing["samples"] + rounded)[-MAX_SAMPLES_PER_SIGN:]
                existing.update(text=text, text_lang=text_lang, translations=clean_tr)
                sign = existing
            else:
                if len(self._signs) >= MAX_SIGNS:
                    raise CustomSignError("Too many taught signs. Delete some first.")
                sign = {
                    "id": uuid.uuid4().hex[:10],
                    "name": name,
                    "text": text,
                    "text_lang": text_lang,
                    "translations": clean_tr,
                    "created_at": int(time.time()),
                    "samples": rounded,
                }
                self._signs.append(sign)
            self._save()
            self._rebuild()
        return self.public(sign)

    def delete(self, sign_id: str) -> bool:
        with self._lock:
            before = len(self._signs)
            self._signs = [s for s in self._signs if s["id"] != sign_id]
            if len(self._signs) == before:
                return False
            self._save()
            self._rebuild()
            return True

    # Sentences -----------------------------------------------------------

    def text_for(self, name: str, lang: str) -> Optional[Tuple[str, str]]:
        """(text, language of that text) for speaking a taught sign in `lang`."""
        sign = self.get(name)
        if not sign:
            return None
        if lang in sign["translations"]:
            return sign["translations"][lang], lang
        return sign["text"], sign["text_lang"]
