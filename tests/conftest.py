"""Test isolation: never use real API keys from .env or the environment."""

import pytest

from backend import config


@pytest.fixture(autouse=True)
def _no_real_api_keys(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
