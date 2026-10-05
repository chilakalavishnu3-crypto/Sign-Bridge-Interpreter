"""OpenAI integration tests (fully mocked — no key or network needed)."""

import asyncio
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import config, server
from backend.sentences import SentenceService, script_ok

client = TestClient(server.app)
JPEG = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff\xe0fakejpeg").decode()


def mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def completion(content, usage=None):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}], "usage": usage or {}})


def test_openai_preferred_for_sentences(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk-test")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return completion({"en": "I need water and help.", "ta": "எனக்கு தண்ணீர் மற்றும் உதவி வேண்டும்."})

    svc = SentenceService({}, ["en", "ta"])
    svc._client = mock_client(handler)
    out = asyncio.run(svc.compose(["WATER", "HELP"], languages=["en", "ta"]))
    assert out["source"] == "llm" and out["en"] == "I need water and help."
    assert seen["url"].startswith("https://api.openai.com/v1/chat/completions")
    assert seen["auth"] == "Bearer sk-test"
    assert "max_completion_tokens" in seen["body"] and "temperature" not in seen["body"]
    assert svc.provider()["name"] == "openai"


def test_identify_requires_key_and_consent(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    r = client.post("/api/ai/identify-sign", json={"images": [JPEG], "consent": True})
    assert r.status_code == 503

    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    r = client.post("/api/ai/identify-sign", json={"images": [JPEG], "consent": False})
    assert r.status_code == 400 and "consent" in r.json()["detail"]

    r = client.post("/api/ai/identify-sign", json={"images": ["https://evil.example/x.jpg"], "consent": True})
    assert r.status_code == 400

    r = client.post("/api/ai/identify-sign", json={"images": [JPEG] * 5, "consent": True})
    assert r.status_code == 422


@pytest.fixture
def hands_visible(monkeypatch):
    monkeypatch.setattr(server, "count_hands_in_images", lambda images: 1)


def test_identify_validates_model_answer(monkeypatch, hands_visible):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    sent = {}

    def handler(request):
        body = json.loads(request.content)
        sent["images"] = [c for c in body["messages"][0]["content"] if c["type"] == "image_url"]
        sent["model"] = body["model"]
        # Model claims in_vocabulary wrongly and uses lower case; we verify ourselves.
        return completion({"sign": "ticket", "in_vocabulary": False, "confidence": 1.7,
                           "alternatives": ["money", "COFFEE"], "reason": "two fingers pinching"})

    monkeypatch.setattr(server.vision_ai, "_client", mock_client(handler))
    r = client.post("/api/ai/identify-sign", json={"images": [JPEG, JPEG], "consent": True, "pack": "railway"})
    assert r.status_code == 200
    data = r.json()
    assert data["sign"] == "TICKET" and data["in_vocabulary"] is True
    assert data["confidence"] == 1.0
    assert data["alternatives"] == [{"sign": "MONEY", "in_vocabulary": True}, {"sign": "COFFEE", "in_vocabulary": False}]
    assert len(sent["images"]) == 2 and sent["images"][0]["image_url"]["detail"] == "low"
    assert sent["model"] == config.OPENAI_VISION_MODEL


def test_identify_handles_bad_ai_responses(monkeypatch, hands_visible):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(server.vision_ai, "_client", mock_client(lambda r: httpx.Response(401, json={})))
    assert client.post("/api/ai/identify-sign", json={"images": [JPEG], "consent": True}).status_code == 502

    monkeypatch.setattr(server.vision_ai, "_client", mock_client(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})))
    assert client.post("/api/ai/identify-sign", json={"images": [JPEG], "consent": True}).status_code == 502


def test_health_reports_ai_flags(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    h = client.get("/api/health").json()
    assert h["vision_enabled"] is True and h["llm_provider"] == "openai"
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    h = client.get("/api/health").json()
    assert h["vision_enabled"] is False and h["llm_provider"] is None


def test_identify_skips_ai_when_no_hand_visible(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(server, "count_hands_in_images", lambda images: 0)
    called = []
    monkeypatch.setattr(server.vision_ai, "_client", mock_client(lambda r: called.append(1) or completion({"sign": "5"})))
    r = client.post("/api/ai/identify-sign", json={"images": [JPEG], "consent": True})
    assert r.status_code == 200
    assert r.json()["sign"] == "UNCLEAR" and r.json()["hands_seen"] == 0
    assert not called  # no money spent, no invented answer


def test_identify_rejects_undecodable_image(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    r = client.post("/api/ai/identify-sign", json={"images": [JPEG], "consent": True})
    if server.HandTracker.available():
        assert r.status_code == 400


def test_script_check():
    assert script_ok("ta", "எனக்கு தண்ணீர் வேண்டும்.")
    assert not script_ok("ta", "நాకు தண்ணீர் வேண்டும்")  # Telugu word mixed in
    assert not script_ok("ta", "I need water")
    assert script_ok("te", "నాకు నీరు కావాలి")
    assert script_ok("en", "I need water.") and not script_ok("en", "नमस्ते")


def test_mixed_script_is_retried_then_falls_back(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return completion({"en": "I need water.", "ta": "నాకు தண்ணீர்", "te": "నాకు నీరు కావాలి"})
        return completion({"en": "I need water.", "ta": "எனக்கு தண்ணீர் வேண்டும்.", "te": "నాకు నీరు కావాలి"})

    svc = SentenceService({}, ["en", "ta", "te"])
    svc._client = mock_client(handler)
    out = asyncio.run(svc.compose(["WATER", "MONEY"], languages=["en", "ta", "te"]))
    assert len(calls) == 2
    assert out["ta"] == "எனக்கு தண்ணீர் வேண்டும்." and "_speech_langs" not in out

    calls.clear()
    svc._client = mock_client(lambda r: calls.append(1) or completion({"en": "I need water.", "ta": "నాకు", "te": "నాకు"}))
    out = asyncio.run(svc.compose(["WATER", "MONEY"], languages=["en", "ta", "te"]))
    assert out["ta"] == "I need water." and out["_speech_langs"] == {"ta": "en"}
