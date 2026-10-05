"""
Automated Integration Tests for FastAPI Backend Endpoints
Matches B5 Acceptance Criteria.
"""

import pytest
from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)


def test_sentence_template_fallback():
    """Verify POST /v1/sentence builds expected sentence from templates."""
    payload = {
        "signs": ["TRAIN", "WHEN"],
        "pack": "railway",
        "languages": ["en", "ta"]
    }
    resp = client.post("/v1/sentence", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "en" in data
    assert "ta" in data
    assert data["en"] == "When does my train leave?"
    assert data["source"] in ["template", "llm"]


def test_sentence_unknown_sign():
    """Verify unknown signs are rejected with 400."""
    payload = {
        "signs": ["INVALID_SIGN_NAME"],
        "pack": "railway"
    }
    resp = client.post("/v1/sentence", json=payload)
    assert resp.status_code == 400


def test_sentence_max_signs_limit():
    """Verify more than 8 signs are rejected with 422."""
    payload = {
        "signs": ["TRAIN"] * 9,
        "pack": "railway"
    }
    resp = client.post("/v1/sentence", json=payload)
    assert resp.status_code == 422


def test_predict_endpoint():
    """Verify POST /api/predict returns classification."""
    features = [0.0] * 126
    resp = client.post("/api/predict", json={"features": features})
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_sign" in data
    assert "confidence" in data
    assert "top3" in data
    assert len(data["top3"]) == 3


def test_vocabulary_endpoint():
    """Verify GET /api/vocabulary returns classes, packs, languages."""
    resp = client.get("/api/vocabulary")
    assert resp.status_code == 200
    data = resp.json()
    assert "classes" in data
    assert len(data["classes"]) >= 20
    assert "packs" in data
    assert "railway" in data["packs"]
    assert "hospital" in data["packs"]
    assert "languages" in data
    assert len(data["languages"]) == 6


def test_tts_endpoint():
    """Verify GET /api/tts returns audio stream."""
    resp = client.get("/api/tts?text=Ticket&lang=en")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert len(resp.content) > 100


def test_lexicon_covers_every_sign_in_every_language():
    """The sign guide shows sign names in the selected language."""
    data = client.get("/api/vocabulary").json()
    langs = [l["code"] for l in data["languages"]]
    words = data["lexicon"]["words"]
    for sign in data["classes"]:
        if sign == "NONE":
            continue
        assert sign in words, f"missing lexicon entry for {sign}"
        for lang in langs:
            assert words[sign].get(lang, "").strip(), f"{sign} has no {lang} name"
    for sign, note in data["lexicon"]["notes"].items():
        assert set(langs) <= set(note), f"note for {sign} missing a language"
