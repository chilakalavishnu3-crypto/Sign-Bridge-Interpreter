"""
SignBridge configuration.

Every tunable can be overridden through an environment variable so the same
build runs on a dev laptop, a counter kiosk, or a container. Secrets (the Groq
key) are only ever read from the environment — never commit them.
"""

import os
from typing import List


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path: str) -> None:
    """Minimal .env loader (KEY=value lines). Real environment variables win."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


_load_dotenv(os.path.join(BASE_DIR, ".env"))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PACKS_DIR = os.path.join(ASSETS_DIR, "packs")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
MODEL_DIR = os.path.join(ASSETS_DIR, "model")
APP_DIR = os.path.join(BASE_DIR, "app")

APP_VERSION = "2.1.0"

# Server
HOST: str = os.getenv("SIGNBRIDGE_HOST", "127.0.0.1")
PORT: int = _env_int("SIGNBRIDGE_PORT", 8080)
DEBUG: bool = os.getenv("SIGNBRIDGE_DEBUG", "false").lower() == "true"
LOG_LEVEL: str = os.getenv("SIGNBRIDGE_LOG_LEVEL", "INFO").upper()
# The kiosk UI is served by this same server, so cross-origin access is only
# needed for explicitly listed origins. Never combine "*" with credentials.
CORS_ORIGINS: List[str] = _env_list(
    "SIGNBRIDGE_CORS_ORIGINS",
    [f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"],
)

# AI / Groq LLM (optional — offline templates are used when absent)
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
GROQ_URL: str = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT: float = _env_float("GROQ_TIMEOUT", 2.0)
PROMPT_VERSION = "sentence-v2"

# OpenAI (optional). When set, it is preferred over Groq for sentence
# phrasing (text only) and enables the opt-in "Ask AI" sign identification
# (sends a few hand-cropped camera frames — requires citizen consent).
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_VISION_MODEL: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip()
OPENAI_TIMEOUT: float = _env_float("OPENAI_TIMEOUT", 4.0)
OPENAI_VISION_TIMEOUT: float = _env_float("OPENAI_VISION_TIMEOUT", 20.0)
VISION_PROMPT_VERSION = "identify-sign-v2"
VISION_MAX_IMAGES = 4
VISION_MAX_IMAGE_BYTES = 350_000

# Gesture recognition
MODEL_PATH: str = os.path.join(MODEL_DIR, "isl_model.joblib")
TASK_PATH: str = os.path.join(MODEL_DIR, "hand_landmarker.task")
CONFIDENCE_THRESHOLD: float = _env_float("SIGNBRIDGE_CONFIDENCE", 0.80)

# Time-based hold rule. The original rule was "25 of 30 frames at 30 fps";
# server-side detection runs at a variable 15-25 fps, so we express it in
# seconds and keep a small tolerance for single-frame dropouts.
HOLD_SECONDS: float = _env_float("SIGNBRIDGE_HOLD_SECONDS", 0.85)
HOLD_DROPOUT_SECONDS: float = _env_float("SIGNBRIDGE_HOLD_DROPOUT_SECONDS", 0.25)
PAUSE_SECONDS: float = _env_float("SIGNBRIDGE_PAUSE_SECONDS", 1.5)
# Hands leaving the frame this long lets the same sign be confirmed again
# (e.g. "2 0 0" for 200 rupees).
REPEAT_RESET_SECONDS: float = _env_float("SIGNBRIDGE_REPEAT_RESET_SECONDS", 0.4)
MAX_WORDS: int = 8

# Taught signs ("Teach a sign"): nearest-neighbour distance at which a match
# scores 50% confidence. Lower = stricter.
CUSTOM_SIGNS_PATH: str = os.getenv("SIGNBRIDGE_CUSTOM_SIGNS_PATH", os.path.join(BASE_DIR, "data", "custom_signs.json"))
CUSTOM_MATCH_SCALE: float = _env_float("SIGNBRIDGE_CUSTOM_MATCH_SCALE", 0.5)
TEACH_SECONDS: float = _env_float("SIGNBRIDGE_TEACH_SECONDS", 3.0)

# Legacy frame-count settings, kept for the desktop runner and older tooling.
HOLD_REQUIRED_FRAMES: int = 25
HOLD_BUFFER_SIZE: int = 30
PAUSE_REQUIRED_FRAMES: int = 45

# WebSocket limits
WS_MAX_IMAGE_BYTES: int = _env_int("SIGNBRIDGE_WS_MAX_IMAGE_BYTES", 400_000)
WS_MAX_FPS: float = _env_float("SIGNBRIDGE_WS_MAX_FPS", 30.0)

# Rate limiting (HTTP)
RATE_LIMIT_MAX_REQUESTS: int = _env_int("SIGNBRIDGE_RATE_LIMIT", 60)
RATE_LIMIT_WINDOW_SECONDS: float = 60.0

# Text-to-speech
TTS_MAX_CHARS: int = 300
