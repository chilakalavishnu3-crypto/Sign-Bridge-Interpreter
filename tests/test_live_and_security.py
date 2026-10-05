"""
Integration tests: live WebSocket protocol, sentence service, and the
security hardening on HTTP endpoints.
"""

import asyncio
import base64
import json

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import config, server
from backend.sentences import SentenceService
from backend.server import app

client = TestClient(app)


def real_hands(sign: str, frame: int = 10):
    """Landmarks for one frame of a recorded sign (slot 0 = Left, 1 = Right)."""
    data = np.load(f"data/raw/Signer_A/{sign}.npz")["landmarks"][frame]
    hands = []
    for slot, label in ((0, "Left"), (1, "Right")):
        pts = data[slot]
        if np.all(np.abs(pts) < 1e-6):
            continue
        hands.append({"label": label, "landmarks": [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in pts]})
    return hands


# --- WebSocket ------------------------------------------------------------------

def test_ws_hello_and_frame_result():
    with client.websocket_connect("/ws/live") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["model_loaded"] is True

        ws.send_json({"action": "frame", "hands": real_hands("TICKET")})
        msg = ws.receive_json()
        assert msg["type"] == "frame_result"
        assert msg["hands_detected"] >= 1
        assert msg["top_prediction"] == "TICKET"
        assert len(msg["top3"]) == 3


def test_ws_confirms_sign_and_finalizes_sentence(monkeypatch):
    monkeypatch.setattr(config, "WS_MAX_FPS", 0)
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        hands = real_hands("TICKET")
        confirmed = None
        # Hold is time-based; stream frames until the server confirms.
        for _ in range(400):
            ws.send_json({"action": "frame", "hands": hands})
            msg = ws.receive_json()
            if msg.get("confirmed_word"):
                confirmed = msg["confirmed_word"]
                break
        assert confirmed == "TICKET"

        ws.send_json({"action": "finalize"})
        assert ws.receive_json() == {"type": "words_updated", "words": []}
        pending = ws.receive_json()
        assert pending["type"] == "sentence_pending"
        ready = ws.receive_json()
        assert ready["type"] == "sentence_ready"
        assert ready["words"] == ["TICKET"]
        assert set(ready["translations"]) >= {"en", "ta", "hi"}
        assert ready["source"] == "template"


def test_ws_tap_to_sign_and_remove():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "add_word", "sign": "WATER"})
        assert ws.receive_json()["words"] == ["WATER"]
        ws.send_json({"action": "add_word", "sign": "NOT_A_SIGN"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"action": "remove_word", "index": 0})
        assert ws.receive_json() == {"type": "word_removed", "words": []}


def test_ws_rejects_malformed_messages_without_disconnecting():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_text("not json")
        assert ws.receive_json()["code"] == "bad_json"
        ws.send_json({"action": "frame", "hands": [{"landmarks": [1, 2]}]})
        assert ws.receive_json()["code"] == "bad_frame"
        ws.send_json({"action": "frame", "image": "data:image/jpeg;base64,@@@"})
        assert ws.receive_json()["code"] == "bad_frame"
        ws.send_json({"action": "explode"})
        assert ws.receive_json()["code"] == "unknown_action"
        # Connection still usable
        ws.send_json({"action": "reset"})
        assert ws.receive_json()["type"] == "reset_ack"


@pytest.mark.skipif(not server.HandTracker.available(), reason="MediaPipe not installed")
def test_ws_server_side_detection_with_image():
    blank = np.full((240, 320, 3), 200, np.uint8)
    ok, jpg = cv2.imencode(".jpg", blank)
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
    with client.websocket_connect("/ws/live") as ws:
        hello = ws.receive_json()
        assert hello["server_detection"] is True
        ws.send_json({"action": "frame", "image": data_url})
        msg = ws.receive_json()
        assert msg["type"] == "frame_result"
        assert msg["hands_detected"] == 0
        assert msg["extracted_landmarks"] == []


# --- Sentence service --------------------------------------------------------------

def _service_with_transport(handler):
    svc = SentenceService({"TRAIN WHEN": {"en": "When does my train leave?"}}, ["en", "ta"])
    svc._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return svc


def test_sentence_service_llm_validates_and_fills(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")

    def handler(request):
        content = json.dumps({"en": "Where is the water?"})  # 'ta' missing
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {}})

    svc = _service_with_transport(handler)
    out = asyncio.run(svc.compose(["WATER", "WHERE", "HELP"], languages=["en", "ta"]))
    # Missing Tamil is filled with English — and flagged to be spoken in English.
    assert out == {"en": "Where is the water?", "ta": "Where is the water?", "source": "llm",
                   "_speech_langs": {"ta": "en"}}


def test_sentence_service_prefers_template_and_falls_back(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500)

    svc = _service_with_transport(handler)
    out = asyncio.run(svc.compose(["TRAIN", "WHEN"], languages=["en"]))
    assert out["source"] == "template" and not calls

    out = asyncio.run(svc.compose(["TRAIN", "MONEY"], languages=["en"]))
    assert out == {"en": "TRAIN MONEY", "source": "fallback"}
    assert calls


def test_sentence_rejects_unknown_language_and_pack():
    assert client.post("/v1/sentence", json={"signs": ["HELP"], "languages": ["xx"]}).status_code == 422
    assert client.post("/v1/sentence", json={"signs": ["HELP"], "pack": "casino"}).status_code == 422


# --- HTTP hardening ----------------------------------------------------------------

def test_security_headers_present():
    resp = client.get("/api/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in resp.headers["Content-Security-Policy"]
    assert resp.headers["X-Request-ID"]
    assert resp.json()["status"] == "healthy"


def test_cors_does_not_reflect_arbitrary_origins():
    resp = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert resp.headers.get("access-control-allow-origin") is None


def test_tts_uses_case_insensitive_cache_and_validates_input():
    # Cached phrase with different casing must be served from disk (no network).
    resp = client.get("/api/tts", params={"text": "when does my train leave?", "lang": "en"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert "max-age" in resp.headers.get("cache-control", "")

    assert client.get("/api/tts", params={"text": "hi", "lang": "zz"}).status_code == 422
    assert client.get("/api/tts", params={"text": "x" * 301, "lang": "en"}).status_code == 422


def test_predict_rejects_wrong_feature_length():
    assert client.post("/api/predict", json={"features": [0.0] * 10}).status_code == 422


def test_predict_accepts_hands():
    resp = client.post("/api/predict", json={"hands": real_hands("TRAIN")})
    assert resp.status_code == 200
    assert resp.json()["predicted_sign"] == "TRAIN"
