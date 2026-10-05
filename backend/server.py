"""
SignBridge backend (FastAPI)

  GET  /api/health       liveness + readiness detail
  GET  /api/vocabulary   classes, packs, languages, regions, templates
  POST /v1/sentence      signs -> sentence (template -> Groq LLM -> gloss)
  POST /api/predict      single-frame classification
  GET  /api/tts          speech audio (pre-generated cache -> gTTS)
  POST /api/ai/identify-sign  opt-in OpenAI vision guess (citizen consent required)
  GET/DELETE /api/custom-signs  taught signs
  WS   /ws/live          live recognition session (see PROTOCOL below)

PROTOCOL (/ws/live, JSON messages)
  client -> server
    {"action": "frame", "image": "<data-url jpeg>"}   server-side MediaPipe
    {"action": "frame", "hands": [...]}               client-side landmarks
    {"action": "reset" | "finalize"}
    {"action": "remove_word", "index": n}
    {"action": "add_word", "sign": "TICKET"}           tap-to-sign fallback
    {"action": "configure", "pack": "railway", "lang": "ta"}
    {"action": "teach_start"}                           record TEACH_SECONDS of samples
    {"action": "teach_save", "name", "text", "text_lang", "translations"}
    {"action": "teach_discard"}
  server -> client
    hello, frame_result, word_removed, words_updated, reset_ack,
    sentence_pending, sentence_ready, teach_started, teach_recorded,
    teach_saved, error
  The client sends its next frame after receiving frame_result, which gives
  natural back-pressure: the frame rate adapts to server capacity.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend import config
from backend.ai_vision import SignVisionService, VisionError
from backend.custom_signs import CustomSignError, CustomSignStore
from backend.logging_setup import setup_logging
from backend.sentences import SentenceService
from backend.session import SignSession
from backend.vision import (Classifier, FrameError, HandTracker, count_hands_in_images, process_frame,
                            validate_client_hands)
from training.dataset_generator import CLASSES
from training.normalize import normalize_dual_hands

setup_logging(config.LOG_LEVEL)
log = logging.getLogger("signbridge.server")

SERVER_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Static data & model loading
# ---------------------------------------------------------------------------

def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        log.warning("missing data file", extra={"path": path})
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


templates_data: Dict[str, Dict[str, str]] = _load_json(os.path.join(config.PACKS_DIR, "templates.json"), {})
packs_data: Dict[str, Any] = _load_json(os.path.join(config.PACKS_DIR, "packs.json"), {})
languages_data: List[Dict[str, str]] = _load_json(os.path.join(config.PACKS_DIR, "languages.json"), [])
regions_data: Dict[str, Any] = _load_json(os.path.join(config.PACKS_DIR, "regions.json"), {})
lexicon_data: Dict[str, Any] = _load_json(os.path.join(config.PACKS_DIR, "lexicon.json"), {"words": {}, "notes": {}})
audio_index_raw: Dict[str, str] = _load_json(os.path.join(config.AUDIO_DIR, "index.json"), {})

LANG_CODES: List[str] = [l["code"] for l in languages_data] or ["en"]
PACK_IDS: List[str] = list(packs_data.keys()) or ["railway"]


def _audio_key(lang: str, text: str) -> str:
    return f"{lang}:{text.strip().casefold()}"


# The index keeps original casing; lookups are case-insensitive.
audio_index: Dict[str, str] = {}
for _key, _file in audio_index_raw.items():
    _lang, _, _text = _key.partition(":")
    audio_index[_audio_key(_lang, _text)] = _file

_model = None
if os.path.exists(config.MODEL_PATH):
    try:
        _model = joblib.load(config.MODEL_PATH)
        log.info("classifier loaded", extra={"path": config.MODEL_PATH})
    except Exception as exc:
        log.error("classifier failed to load: %s", exc)
classifier = Classifier(_model)

custom_store = CustomSignStore(config.CUSTOM_SIGNS_PATH, reserved_names=set(CLASSES))
sentences = SentenceService(templates_data, LANG_CODES, custom=custom_store)
vision_ai = SignVisionService()


def is_known_sign(sign: Any) -> bool:
    return isinstance(sign, str) and (sign in CLASSES or sign in custom_store.names())


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await sentences.close()
    await vision_ai.close()


app = FastAPI(
    title="SignBridge Civic Counter API",
    description="Two-way Indian Sign Language communication for public service counters",
    version=config.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Request-ID"],
)

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data: blob:",
    "media-src 'self' blob:",
    "connect-src 'self' ws: wss:",
    "frame-ancestors 'self'",
    "base-uri 'self'",
    "form-action 'self'",
])


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error", extra={"request_id": request_id, "path": request.url.path})
        response = JSONResponse(
            {"detail": "Internal server error", "request_id": request_id},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    duration_ms = (time.perf_counter() - started) * 1000.0

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY

    if request.url.path.startswith("/api") or request.url.path.startswith("/v1"):
        log.info(
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(duration_ms, 1),
            },
        )
    return response


# ---------------------------------------------------------------------------
# Rate limiting (sliding window, in-memory; one process per kiosk)
# ---------------------------------------------------------------------------

rate_limit_records: Dict[str, Deque[float]] = {}
_last_rate_prune = time.monotonic()


def check_rate_limit(request: Request) -> None:
    global _last_rate_prune
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = config.RATE_LIMIT_WINDOW_SECONDS

    if now - _last_rate_prune > window:
        for key in [k for k, v in rate_limit_records.items() if not v or now - v[-1] > window]:
            rate_limit_records.pop(key, None)
        _last_rate_prune = now

    records = rate_limit_records.setdefault(ip, deque())
    while records and now - records[0] >= window:
        records.popleft()
    if len(records) >= config.RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment.",
            headers={"Retry-After": str(int(window))},
        )
    records.append(now)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SentenceRequest(BaseModel):
    signs: List[str] = Field(..., max_length=8, description="Ordered list of up to 8 ISL signs")
    pack: Optional[str] = Field("railway", description="Vocabulary pack id")
    languages: Optional[List[str]] = Field(None, max_length=10, description="Target language codes")

    @field_validator("pack")
    @classmethod
    def _known_pack(cls, v: Optional[str]) -> str:
        if v is None:
            return "railway"
        if v not in PACK_IDS:
            raise ValueError(f"Unknown pack. Expected one of {PACK_IDS}")
        return v

    @field_validator("languages")
    @classmethod
    def _known_languages(cls, v: Optional[List[str]]) -> List[str]:
        if not v:
            return ["en", "ta"]
        unknown = [c for c in v if c not in LANG_CODES]
        if unknown:
            raise ValueError(f"Unsupported languages: {unknown}")
        return list(dict.fromkeys(v))


class IdentifyRequest(BaseModel):
    images: List[str] = Field(..., min_length=1, max_length=4)
    consent: bool = Field(..., description="Citizen agreed to send camera frames to the AI service")
    pack: Optional[str] = "railway"


class PredictRequest(BaseModel):
    features: Optional[List[float]] = Field(None, description="126 normalized features")
    hands: Optional[List[Dict[str, Any]]] = Field(None, max_length=2, description="Raw hand landmarks")


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    ready = classifier.loaded
    return {
        "status": "healthy" if ready else "degraded",
        "version": config.APP_VERSION,
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 2),
        "model_loaded": classifier.loaded,
        "classes_count": len(CLASSES),
        "server_mediapipe_available": HandTracker.available(),
        "llm_enabled": sentences.llm_enabled,
        "llm_provider": (sentences.provider() or {}).get("name"),
        "vision_enabled": vision_ai.enabled,
        "audio_cached_count": len(audio_index),
        "packs": PACK_IDS,
        "supported_languages": LANG_CODES,
    }


@app.get("/api/vocabulary")
async def get_vocabulary():
    return {
        "classes": CLASSES,
        "packs": packs_data,
        "languages": languages_data,
        "regions": regions_data,
        "templates": templates_data,
        "custom_signs": custom_store.list(),
        "lexicon": lexicon_data,
        "threshold": config.CONFIDENCE_THRESHOLD,
        "timing": {
            "hold_seconds": config.HOLD_SECONDS,
            "pause_seconds": config.PAUSE_SECONDS,
        },
    }


@app.post("/v1/sentence")
async def generate_sentence(req: SentenceRequest, request: Request):
    check_rate_limit(request)
    if not req.signs:
        raise HTTPException(status_code=400, detail="Sign list cannot be empty.")
    unknown = [s for s in req.signs if not is_known_sign(s)]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown sign: {unknown[0]}")
    result = await sentences.compose(req.signs, req.pack, req.languages, request.state.request_id)
    speech_langs = result.pop("_speech_langs", None)
    if speech_langs:
        result["speech_langs"] = speech_langs
    return result


@app.post("/api/predict")
async def predict_gesture(req: PredictRequest):
    if not classifier.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if req.features is not None:
        if len(req.features) != 126:
            raise HTTPException(status_code=422, detail="features must contain exactly 126 values.")
        feat = np.asarray(req.features, dtype=np.float32).reshape(1, -1)
    elif req.hands is not None:
        try:
            hands = validate_client_hands(req.hands)
        except FrameError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        feat = np.asarray(normalize_dual_hands(hands), dtype=np.float32).reshape(1, -1)
    else:
        raise HTTPException(status_code=400, detail="Must provide 126 features or hands landmarks.")

    probs = classifier.model.predict_proba(feat)[0]
    order = np.argsort(probs)[::-1]
    pred_sign = CLASSES[int(order[0])]
    conf = float(probs[order[0]])
    return {
        "predicted_sign": pred_sign,
        "confidence": round(conf, 4),
        "accepted": conf >= config.CONFIDENCE_THRESHOLD and pred_sign != "NONE",
        "top3": [{"sign": CLASSES[i], "confidence": round(float(probs[i]), 4)} for i in order[:3]],
    }


@app.post("/api/ai/identify-sign")
async def identify_sign(req: IdentifyRequest, request: Request):
    check_rate_limit(request)
    if not req.consent:
        raise HTTPException(status_code=400, detail="The citizen's consent is required before sending camera images.")
    pack = req.pack if req.pack in PACK_IDS else PACK_IDS[0]
    vocabulary = list(packs_data.get(pack, {}).get("signs", [])) + sorted(custom_store.names())
    if not vision_ai.enabled:
        raise HTTPException(status_code=503, detail="AI identification is not configured (set OPENAI_API_KEY).")
    # Ground the AI: if our own detector sees no hand, don't ask (avoids
    # confident made-up answers and saves cost).
    try:
        hands_seen = await asyncio.to_thread(count_hands_in_images, req.images)
    except FrameError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if hands_seen == 0:
        log.info("vision_skipped_no_hands", extra={"request_id": request.state.request_id})
        return {"sign": "UNCLEAR", "in_vocabulary": False, "confidence": 0.0, "alternatives": [],
                "reason": "No hand was visible in the pictures.", "model": None, "hands_seen": 0}
    try:
        result = await vision_ai.identify(req.images, vocabulary, pack, request.state.request_id)
        result["hands_seen"] = hands_seen
        return result
    except VisionError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@app.get("/api/custom-signs")
async def list_custom_signs():
    return {"signs": custom_store.list()}


@app.delete("/api/custom-signs/{sign_id}")
async def delete_custom_sign(sign_id: str, request: Request):
    check_rate_limit(request)
    if not await asyncio.to_thread(custom_store.delete, sign_id[:32]):
        raise HTTPException(status_code=404, detail="Taught sign not found.")
    log.info("custom_sign_deleted", extra={"request_id": request.state.request_id, "sign_id": sign_id[:32]})
    return {"deleted": sign_id}


def _synthesize(text: str, lang: str) -> bytes:
    import io
    from gtts import gTTS

    fp = io.BytesIO()
    gTTS(text=text, lang=lang, slow=False).write_to_fp(fp)
    return fp.getvalue()


@app.get("/api/tts")
async def text_to_speech(
    request: Request,
    text: str = Query(..., min_length=1, max_length=config.TTS_MAX_CHARS),
    lang: str = Query("en"),
):
    if lang not in LANG_CODES:
        raise HTTPException(status_code=422, detail=f"Unsupported language: {lang}")

    cached = audio_index.get(_audio_key(lang, text))
    if cached:
        path = os.path.join(config.AUDIO_DIR, os.path.basename(cached))
        if os.path.exists(path):
            return FileResponse(
                path, media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=86400"}
            )

    # Uncached speech goes to an external service: rate-limit it.
    check_rate_limit(request)
    try:
        audio = await asyncio.wait_for(asyncio.to_thread(_synthesize, text.strip(), lang), timeout=8.0)
    except Exception as exc:
        log.warning("tts_failed", extra={"request_id": request.state.request_id, "error": type(exc).__name__})
        raise HTTPException(status_code=503, detail="Speech synthesis is unavailable (offline?).")
    return Response(content=audio, media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# Live recognition WebSocket
# ---------------------------------------------------------------------------

class LiveConnection:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.id = uuid.uuid4().hex[:8]
        self.session = SignSession()
        self.tracker: Optional[HandTracker] = HandTracker() if HandTracker.available() else None
        self.pack = PACK_IDS[0]
        self.lang = "ta" if "ta" in LANG_CODES else LANG_CODES[0]
        self.send_lock = asyncio.Lock()
        self.last_frame_at = 0.0
        self.tasks: set = set()
        self.stats = {"frames": 0, "with_hands": 0, "confirmed": 0, "errors": 0, "ms": 0.0}
        self.stats_since = time.monotonic()
        self.teach_until: Optional[float] = None
        self.teach_samples: List[List[float]] = []

    def record(self, hands: int, ms: float, confirmed: Optional[str], sign: str = "NONE", conf: float = 0.0) -> None:
        """Aggregate per-connection activity; logged every 5 s (no images or landmarks)."""
        st = self.stats
        st["frames"] += 1
        st["with_hands"] += 1 if hands else 0
        st["confirmed"] += 1 if confirmed else 0
        st["ms"] += ms
        if hands:
            best = st.setdefault("guesses", {})
            best[sign] = max(best.get(sign, 0.0), round(conf, 2))
        now = time.monotonic()
        if now - self.stats_since >= 5.0:
            log.info("ws_stats", extra={
                "connection": self.id,
                "fps": round(st["frames"] / (now - self.stats_since), 1),
                "frames": st["frames"],
                "frames_with_hands": st["with_hands"],
                "signs_confirmed": st["confirmed"],
                "frame_errors": st["errors"],
                "avg_processing_ms": round(st["ms"] / max(1, st["frames"]), 1),
                "words": list(self.session.words),
                # Best confidence per predicted sign this window (diagnostics).
                "top_guesses": dict(sorted(st.get("guesses", {}).items(), key=lambda kv: -kv[1])[:4]),
            })
            self.stats = {"frames": 0, "with_hands": 0, "confirmed": 0, "errors": 0, "ms": 0.0}
            self.stats_since = now

    async def send(self, payload: Dict[str, Any]) -> None:
        async with self.send_lock:
            await self.ws.send_json(payload)

    async def error(self, code: str, message: str) -> None:
        await self.send({"type": "error", "code": code, "message": message})

    def configure(self, data: Dict[str, Any]) -> None:
        if data.get("pack") in PACK_IDS:
            self.pack = data["pack"]
        if data.get("lang") in LANG_CODES:
            self.lang = data["lang"]

    def emit_sentence(self, words: List[str]) -> None:
        task = asyncio.create_task(self._compose_and_send(words))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _compose_and_send(self, words: List[str]) -> None:
        try:
            await self.send({"type": "sentence_pending", "words": words})
            result = await sentences.compose(words, self.pack, LANG_CODES, f"ws-{self.id}")
            source = result.pop("source")
            speech_langs = result.pop("_speech_langs", {})
            await self.send({
                "type": "sentence_ready",
                "words": words,
                "translations": result,
                "sentence": result.get(self.lang, result.get("en", "")),
                "english": result.get("en", ""),
                "lang": self.lang,
                "source": source,
                "speech_langs": speech_langs,
            })
        except Exception:
            log.exception("sentence emit failed", extra={"connection": self.id})

    async def close(self) -> None:
        for task in list(self.tasks):
            task.cancel()
        if self.tracker is not None:
            await asyncio.to_thread(self.tracker.close)


@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    await websocket.accept()
    conn = LiveConnection(websocket)
    log.info("ws_open", extra={"connection": conn.id})
    await conn.send({
        "type": "hello",
        "connection_id": conn.id,
        "model_loaded": classifier.loaded,
        "server_detection": conn.tracker is not None,
        "llm_enabled": sentences.llm_enabled,
        "vision_enabled": vision_ai.enabled,
        "hold_seconds": config.HOLD_SECONDS,
        "pause_seconds": config.PAUSE_SECONDS,
    })
    min_interval = 1.0 / config.WS_MAX_FPS if config.WS_MAX_FPS > 0 else 0.0

    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > config.WS_MAX_IMAGE_BYTES * 2:
                await conn.error("payload_too_large", "Message too large.")
                continue
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError
            except ValueError:
                await conn.error("bad_json", "Messages must be JSON objects.")
                continue

            action = data.get("action", "frame")
            conn.configure(data)

            if action == "configure":
                continue

            if action == "reset":
                conn.session.reset()
                await conn.send({"type": "reset_ack", "words": []})
                continue

            if action == "remove_word":
                idx = data.get("index")
                if isinstance(idx, int) and conn.session.remove_word(idx):
                    await conn.send({"type": "word_removed", "words": list(conn.session.words)})
                continue

            if action == "add_word":
                sign = data.get("sign")
                if is_known_sign(sign) and conn.session.add_word(sign):
                    await conn.send({"type": "words_updated", "words": list(conn.session.words), "added": sign})
                else:
                    await conn.error("invalid_sign", "That sign cannot be added.")
                continue

            if action == "finalize":
                words = conn.session.finalize()
                if words:
                    await conn.send({"type": "words_updated", "words": []})
                    conn.emit_sentence(words)
                continue

            if action == "teach_start":
                if conn.tracker is None:
                    await conn.error("teach_unavailable", "Teaching needs the camera and server hand detection.")
                    continue
                conn.teach_until = time.monotonic() + config.TEACH_SECONDS
                await conn.send({"type": "teach_started", "seconds": config.TEACH_SECONDS,
                                 "samples": len(conn.teach_samples)})
                continue

            if action == "teach_discard":
                conn.teach_until = None
                conn.teach_samples = []
                await conn.send({"type": "teach_recorded", "samples": 0})
                continue

            if action == "teach_save":
                try:
                    saved = await asyncio.to_thread(
                        custom_store.add,
                        str(data.get("name", ""))[:60],
                        str(data.get("text", ""))[:400],
                        data.get("text_lang") if data.get("text_lang") in LANG_CODES else "en",
                        data.get("translations") if isinstance(data.get("translations"), dict) else {},
                        conn.teach_samples,
                        LANG_CODES,
                    )
                except CustomSignError as exc:
                    await conn.error("teach_invalid", str(exc))
                    continue
                conn.teach_samples = []
                log.info("custom_sign_saved", extra={"connection": conn.id, "sign": saved["name"],
                                                     "samples": saved["samples"]})
                await conn.send({"type": "teach_saved", "sign": saved, "signs": custom_store.list()})
                continue

            if action != "frame":
                await conn.error("unknown_action", f"Unknown action: {str(action)[:32]}")
                continue

            # Throttle (never silently drop: the client waits for each reply).
            wait = min_interval - (time.monotonic() - conn.last_frame_at)
            if wait > 0:
                await asyncio.sleep(wait)
            conn.last_frame_at = time.monotonic()

            image = data.get("image")
            if image is not None and not isinstance(image, str):
                await conn.error("bad_frame", "image must be a string.")
                continue
            try:
                res = await asyncio.to_thread(
                    process_frame, conn.tracker, classifier, image, data.get("hands"), custom_store
                )
            except FrameError as exc:
                conn.stats["errors"] += 1
                await conn.error("bad_frame", str(exc))
                continue
            hands, sign, conf, top3, ms = res.hands, res.sign, res.confidence, res.top3, res.ms

            # Teaching: collect samples instead of recognizing.
            if conn.teach_until is not None:
                if hands and len(conn.teach_samples) < 600:
                    conn.teach_samples.append(res.features)
                remaining = conn.teach_until - time.monotonic()
                await conn.send({
                    "type": "frame_result",
                    "hands_detected": len(hands),
                    "teaching": True,
                    "teach_progress": round(min(1.0, 1 - remaining / config.TEACH_SECONDS), 3),
                    "teach_samples": len(conn.teach_samples),
                    "words": list(conn.session.words),
                    "extracted_landmarks": hands if image else None,
                    "processing_ms": round(ms, 1),
                })
                if remaining <= 0:
                    conn.teach_until = None
                    await conn.send({"type": "teach_recorded", "samples": len(conn.teach_samples)})
                continue

            outcome = conn.session.process(len(hands), sign, conf)
            conn.record(len(hands), ms, outcome.confirmed_word, sign, conf)
            if outcome.confirmed_word:
                log.info("sign_confirmed", extra={"connection": conn.id, "sign": outcome.confirmed_word,
                                                  "confidence": round(conf, 3)})
            await conn.send({
                "type": "frame_result",
                "hands_detected": outcome.hands_detected,
                "top_prediction": outcome.top_prediction,
                "confidence": round(outcome.confidence, 4),
                "top3": top3,
                "candidate": outcome.candidate,
                "hold_progress": outcome.hold_progress,
                "pause_progress": outcome.pause_progress,
                "words": outcome.words,
                "confirmed_word": outcome.confirmed_word,
                "prediction_source": res.source,
                "extracted_landmarks": hands if image else None,
                "processing_ms": round(ms, 1),
            })
            if outcome.sentence_words:
                conn.emit_sentence(outcome.sentence_words)

    except WebSocketDisconnect:
        pass
    finally:
        await conn.close()
        log.info("ws_close", extra={"connection": conn.id})


# ---------------------------------------------------------------------------
# Static files (kiosk UI)
# ---------------------------------------------------------------------------

os.makedirs(config.APP_DIR, exist_ok=True)
app.mount("/assets", StaticFiles(directory=config.ASSETS_DIR), name="assets")
app.mount("/", StaticFiles(directory=config.APP_DIR, html=True), name="app")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server:app", host=config.HOST, port=config.PORT, log_config=None)
