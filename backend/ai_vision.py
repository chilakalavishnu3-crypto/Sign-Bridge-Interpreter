"""
Opt-in "Ask AI" sign identification (OpenAI vision).

Used only when staff press "Ask AI" *and* the citizen has consented: a few
camera frames (cropped around the hands by the browser when possible) are
sent to a vision model, which proposes the sign. The answer is a suggestion
for staff to accept or reject — never added automatically.

General vision models are not trained sign-language recognizers and cannot
see motion between frames, so results must be treated as guesses.
Images are never stored or logged.
"""

import base64
import binascii
import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from backend import config

log = logging.getLogger("signbridge.ai_vision")


class VisionError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _check_image(data_url: str) -> str:
    if not isinstance(data_url, str) or not data_url.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
        raise VisionError("Images must be JPEG or PNG data URLs.")
    encoded = data_url.split(",", 1)[1]
    if len(encoded) * 3 // 4 > config.VISION_MAX_IMAGE_BYTES:
        raise VisionError("Image too large.")
    try:
        base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise VisionError("Image is not valid base64.")
    return data_url


class SignVisionService:
    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def enabled(self) -> bool:
        return bool(config.OPENAI_API_KEY)

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=config.OPENAI_VISION_TIMEOUT)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def identify(self, images: List[str], vocabulary: List[str], pack: str, request_id: str = "-") -> Dict[str, Any]:
        if not self.enabled:
            raise VisionError("AI identification is not configured (set OPENAI_API_KEY).", 503)
        if not images or len(images) > config.VISION_MAX_IMAGES:
            raise VisionError(f"Send 1–{config.VISION_MAX_IMAGES} images.")
        images = [_check_image(i) for i in images]

        prompt = (
            f"These {len(images)} images are consecutive camera frames (mirrored, selfie view) of a person "
            f"signing at a public {pack} service counter in India, most likely in Indian Sign Language (ISL). "
            "Identify the single sign being made. Consider hand shape, palm orientation, and position; "
            "motion between frames may be implied by differences between them.\n"
            f"Prefer one of these known signs if one fits: {json.dumps(vocabulary)}.\n"
            "If none fits, give your best short English gloss in capital letters.\n"
            "First check that a human hand is clearly visible and forming a deliberate sign. "
            "If not, or if you cannot tell, answer sign \"UNCLEAR\" with confidence 0. "
            "Do not guess from vague shapes; be conservative with confidence.\n"
            'Reply ONLY with JSON: {"sign": str, "in_vocabulary": bool, "confidence": number 0-1, '
            '"alternatives": [str, ...up to 3], "reason": short str}'
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": img, "detail": "low"}} for img in images]
        payload = {
            "model": config.OPENAI_VISION_MODEL,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 300,
        }

        started = time.perf_counter()
        outcome, usage = "error", {}
        try:
            resp = await self._http().post(
                f"{config.OPENAI_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                json=payload,
            )
            if resp.status_code != 200:
                outcome = f"http_{resp.status_code}"
                raise VisionError("The AI service rejected the request. Check the API key and model.", 502)
            body = resp.json()
            usage = body.get("usage") or {}
            result = self._validate(json.loads(body["choices"][0]["message"]["content"]), vocabulary)
            outcome = "ok"
            return result
        except httpx.TimeoutException:
            outcome = "timeout"
            raise VisionError("The AI service took too long. Try again.", 504)
        except httpx.HTTPError:
            outcome = "network"
            raise VisionError("Couldn't reach the AI service (offline?).", 502)
        except (ValueError, KeyError, IndexError, TypeError):
            outcome = "invalid_response"
            raise VisionError("The AI gave an unreadable answer. Try again.", 502)
        finally:
            log.info("vision_call", extra={
                "request_id": request_id,
                "model": config.OPENAI_VISION_MODEL,
                "prompt_version": config.VISION_PROMPT_VERSION,
                "images": len(images),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "outcome": outcome,
            })

    @staticmethod
    def _validate(parsed: Any, vocabulary: List[str]) -> Dict[str, Any]:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("sign"), str):
            raise ValueError("missing sign")
        by_upper = {v.upper(): v for v in vocabulary}
        sign = parsed["sign"].strip()[:40] or "UNCLEAR"
        canonical = by_upper.get(sign.upper())
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        alternatives = []
        for alt in parsed.get("alternatives") or []:
            if isinstance(alt, str) and alt.strip() and len(alternatives) < 3:
                a = alt.strip()[:40]
                alternatives.append({"sign": by_upper.get(a.upper(), a), "in_vocabulary": a.upper() in by_upper})
        return {
            "sign": canonical or sign.upper(),
            "in_vocabulary": canonical is not None,  # verified here, not trusted from the model
            "confidence": round(confidence, 2),
            "alternatives": alternatives,
            "reason": str(parsed.get("reason", ""))[:200],
            "model": config.OPENAI_VISION_MODEL,
        }
