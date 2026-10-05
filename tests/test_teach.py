"""Tests for user-taught signs (store, matching, aspect correction, WS teach flow)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import config, server
from backend.custom_signs import CustomSignError, CustomSignStore
from training.dataset_generator import CLASSES
from training.normalize import correct_aspect, normalize_dual_hands

LANGS = ["en", "ta", "hi", "te", "kn", "ml"]


def odd_hand(seed=0, jitter=0.0):
    """A hand pose unlike the built-in vocabulary (crossed, bent fingers)."""
    rng = np.random.default_rng(seed)
    base = np.array([[0.5, 0.8, 0]] + [[0.5 + 0.03 * np.sin(i * 1.7), 0.8 - 0.012 * i * (1 + (i % 4)), 0.01 * (i % 3)] for i in range(1, 21)])
    base += rng.normal(0, jitter, base.shape)
    return [{"label": "Right", "landmarks": [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in base]}]


def feats(hands):
    return normalize_dual_hands(hands)


@pytest.fixture
def store(tmp_path):
    return CustomSignStore(str(tmp_path / "custom.json"), reserved_names=set(CLASSES))


def test_store_add_match_persist_delete(store, tmp_path):
    samples = [feats(odd_hand(i, 0.002)) for i in range(20)]
    sign = store.add("COFFEE", "I would like a coffee.", "en", {"ta": "எனக்கு காபி வேண்டும்."}, samples, LANGS)
    assert sign["samples"] == 20 and "samples" not in [k for k, v in sign.items() if isinstance(v, list)]

    name, conf = store.match(feats(odd_hand(99, 0.002)))
    assert name == "COFFEE" and conf > 0.8

    # Different hand configuration (two hands) never matches a one-hand sign.
    two = odd_hand(1) + [{**odd_hand(2)[0], "label": "Left"}]
    assert store.match(feats(two)) == (None, 0.0)

    reloaded = CustomSignStore(str(tmp_path / "custom.json"))
    assert reloaded.match(feats(odd_hand(5, 0.002)))[0] == "COFFEE"
    assert reloaded.text_for("COFFEE", "ta") == ("எனக்கு காபி வேண்டும்.", "ta")
    assert reloaded.text_for("COFFEE", "hi") == ("I would like a coffee.", "en")

    assert store.delete(sign["id"])
    assert store.match(feats(odd_hand(3)))[0] is None


def test_store_validation(store):
    samples = [feats(odd_hand(i)) for i in range(20)]
    with pytest.raises(CustomSignError):
        store.add("TICKET", "x", "en", {}, samples, LANGS)  # built-in name
    with pytest.raises(CustomSignError):
        store.add("<script>", "x", "en", {}, samples, LANGS)
    with pytest.raises(CustomSignError):
        store.add("TEA", "", "en", {}, samples, LANGS)
    with pytest.raises(CustomSignError):
        store.add("TEA", "Tea please", "en", {}, samples[:3], LANGS)  # too few frames


def test_correct_aspect_restores_isotropy():
    # A square in pixel space on a 4:3 frame: normalized x spans 3/4 of y.
    hand = [{"label": "Right", "landmarks": [{"x": 0.5, "y": 0.5, "z": 0.0}] * 9 + [{"x": 0.5 + 0.075, "y": 0.5 - 0.1, "z": 0.0}] * 12}]
    fixed = correct_aspect(hand, 4 / 3)[0]["landmarks"]
    dx = fixed[9]["x"] - fixed[0]["x"]
    dy = fixed[0]["y"] - fixed[9]["y"]
    assert dx == pytest.approx(dy)
    assert correct_aspect(hand, 1.0) is hand


def test_ws_teach_then_recognize_and_speak(store, monkeypatch):
    monkeypatch.setattr(server, "custom_store", store)
    monkeypatch.setattr(server.sentences, "custom", store)
    monkeypatch.setattr(config, "TEACH_SECONDS", 0.6)
    monkeypatch.setattr(config, "WS_MAX_FPS", 0)
    client = TestClient(server.app)

    with client.websocket_connect("/ws/live") as ws:
        hello = ws.receive_json()
        if not hello["server_detection"]:
            pytest.skip("teaching requires MediaPipe")

        ws.send_json({"action": "teach_start"})
        assert ws.receive_json()["type"] == "teach_started"
        recorded = None
        for i in range(400):
            ws.send_json({"action": "frame", "hands": odd_hand(i, 0.002)})
            msg = ws.receive_json()
            assert msg["type"] == "frame_result" and msg.get("teaching") is True
            if msg["teach_progress"] >= 1:
                recorded = ws.receive_json()  # follows the final teaching frame
                assert recorded["type"] == "teach_recorded"
                break
        assert recorded and recorded["samples"] >= 12

        ws.send_json({"action": "teach_save", "name": "COFFEE", "text": "I would like a coffee.", "text_lang": "en"})
        saved = ws.receive_json()
        assert saved["type"] == "teach_saved" and saved["sign"]["name"] == "COFFEE"

        # Now signing it is recognized as the taught sign and confirmed.
        confirmed = None
        for i in range(400):
            ws.send_json({"action": "frame", "hands": odd_hand(1000 + i, 0.002)})
            msg = ws.receive_json()
            if msg.get("confirmed_word"):
                confirmed = msg
                break
        assert confirmed["confirmed_word"] == "COFFEE"
        assert confirmed["prediction_source"] == "taught"

        ws.send_json({"action": "finalize"})
        ws.receive_json()  # words_updated
        assert ws.receive_json()["type"] == "sentence_pending"
        ready = ws.receive_json()
        assert ready["source"] == "taught"
        assert ready["translations"]["en"] == "I would like a coffee."
        # No Tamil translation was given: the UI must speak it as English.
        assert ready["speech_langs"]["ta"] == "en"

    resp = client.get("/api/custom-signs")
    assert [s["name"] for s in resp.json()["signs"]] == ["COFFEE"]
    sign_id = resp.json()["signs"][0]["id"]
    assert client.delete(f"/api/custom-signs/{sign_id}").status_code == 200
    assert client.delete(f"/api/custom-signs/{sign_id}").status_code == 404
