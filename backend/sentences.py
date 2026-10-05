"""
Sentence composition: ISL sign sequence -> natural sentences in N languages.

Resolution order
  1. Curated template (exact sign sequence)  -> source "template"
     Verified translations, instant, offline. Always preferred when present.
  2. LLM: OpenAI if OPENAI_API_KEY is set, else Groq if GROQ_API_KEY is
     set                                     -> source "llm"
     Async, short timeout, JSON-validated; missing languages are filled from
     English so the response shape is always complete.
  3. Raw sign gloss                          -> source "fallback"
     e.g. "TRAIN TICKET" — honest about not being a sentence.

Taught signs carry their own meaning text. A sequence containing them is
composed from those meanings (LLM-smoothed when available) -> source
"taught". Results may include "_speech_langs": the language each text is
actually written in, when a meaning had no translation for that language.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from backend import config

log = logging.getLogger("signbridge.sentences")

# Unicode blocks of the Indian scripts we support. A sentence for one of
# these languages must use its own script and no other Indian script
# (LLMs sometimes mix e.g. Telugu words into Tamil).
SCRIPTS = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
}
_INDIC_RANGE = (0x0900, 0x0DFF)


def script_ok(lang: str, text: str) -> bool:
    lo_all, hi_all = _INDIC_RANGE
    indic = [ord(ch) for ch in text if lo_all <= ord(ch) <= hi_all]
    if lang not in SCRIPTS:
        return not indic  # e.g. English must not contain Indian script
    lo, hi = SCRIPTS[lang]
    own = [c for c in indic if lo <= c <= hi]
    return bool(own) and len(own) == len(indic)


class SentenceService:
    def __init__(self, templates: Dict[str, Dict[str, str]], languages: List[str], custom=None):
        self.templates = templates
        self.languages = languages
        self.custom = custom
        self._client: Optional[httpx.AsyncClient] = None

    @staticmethod
    def provider() -> Optional[Dict[str, Any]]:
        """Active text LLM provider (OpenAI preferred), or None for offline."""
        if config.OPENAI_API_KEY:
            return {"name": "openai", "url": f"{config.OPENAI_BASE_URL}/chat/completions",
                    "key": config.OPENAI_API_KEY, "model": config.OPENAI_MODEL, "timeout": config.OPENAI_TIMEOUT}
        if config.GROQ_API_KEY:
            return {"name": "groq", "url": config.GROQ_URL, "key": config.GROQ_API_KEY,
                    "model": config.GROQ_MODEL, "timeout": config.GROQ_TIMEOUT}
        return None

    @property
    def llm_enabled(self) -> bool:
        return self.provider() is not None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)  # per-request timeouts below
        return self._client

    async def compose(
        self,
        signs: List[str],
        pack: str = "railway",
        languages: Optional[List[str]] = None,
        request_id: str = "-",
    ) -> Dict[str, str]:
        langs = languages or self.languages
        clean = [s for s in signs if s != "NONE"]
        if not clean:
            return {**{lang: "..." for lang in langs}, "source": "template"}

        key = " ".join(clean)
        template = self.templates.get(key)
        if template:
            out = {lang: template.get(lang) or template.get("en", key) for lang in langs}
            out["source"] = "template"
            return out

        taught = self.custom is not None and any(self.custom.get(s) for s in clean)
        if taught:
            return await self._compose_taught(clean, pack, langs, request_id)

        if self.llm_enabled:
            llm = await self._compose_llm(clean, pack, langs, request_id)
            if llm is not None:
                return llm

        out = {lang: key for lang in langs}
        out["source"] = "fallback"
        return out

    def _word_text(self, sign: str, lang: str):
        """(text, language it is written in) for one sign."""
        custom = self.custom.get(sign) if self.custom is not None else None
        if custom:
            return self.custom.text_for(sign, lang)
        single = self.templates.get(sign)
        if single:
            return single.get(lang) or single.get("en", sign), lang if single.get(lang) else "en"
        return sign, "en"

    async def _compose_taught(self, signs: List[str], pack: str, langs: List[str], request_id: str) -> Dict[str, str]:
        if len(signs) > 1 and self.llm_enabled:
            meanings = [self._word_text(s, "en")[0] for s in signs]
            llm = await self._compose_llm(meanings, pack, langs, request_id)
            if llm is not None:
                llm["source"] = "taught"
                return llm

        out: Dict[str, Any] = {}
        speech_langs: Dict[str, str] = {}
        for lang in langs:
            parts = [self._word_text(s, lang) for s in signs]
            out[lang] = " ".join(text.rstrip(".") if i < len(parts) - 1 else text for i, (text, _) in enumerate(parts))
            written = {pl for _, pl in parts}
            if written != {lang}:
                # Mixed or untranslated: speak with the meaning's own language.
                speech_langs[lang] = parts[0][1] if len(written) == 1 else "en"
        out["source"] = "taught"
        if speech_langs:
            out["_speech_langs"] = speech_langs
        return out

    async def _compose_llm(
        self, signs: List[str], pack: str, langs: List[str], request_id: str
    ) -> Optional[Dict[str, str]]:
        system_prompt = (
            f"You are SignBridge's Indian Sign Language grammar interpreter at a public {pack} service counter. "
            "Turn the given ISL sign glosses into one short, natural, polite sentence a citizen would say. "
            "ISL places question words last (TRAIN WHEN -> 'When does my train leave?'). "
            "Never add facts that are not in the signs. "
            "Write each language ONLY in its own native script (Tamil in Tamil script, Telugu in Telugu script, "
            "Hindi in Devanagari, etc.) and never mix scripts. "
            f"Reply ONLY with a JSON object whose keys are exactly {json.dumps(langs)} "
            "and whose values are the sentence in that language (ISO 639-1 codes)."
        )
        provider = self.provider()
        if provider is None:
            return None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Signs: {json.dumps(signs)}"},
        ]
        result = await self._llm_once(provider, messages, signs, langs, request_id)
        if result is None:
            return None
        bad = [lang for lang in langs if not script_ok(lang, result[lang])]
        if bad:
            # One corrective retry, then fall back to English for what is still wrong.
            retry = messages + [
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                {"role": "user", "content": f"The sentences for {bad} use the wrong script or mix scripts. "
                                            "Rewrite them entirely in each language's own script. Same JSON format."},
            ]
            second = await self._llm_once(provider, retry, signs, langs, request_id, attempt=2)
            if second is not None:
                result.update({lang: second[lang] for lang in bad if script_ok(lang, second[lang])})
            still_bad = [lang for lang in langs if not script_ok(lang, result[lang])]
            if still_bad:
                english = result.get("en") if script_ok("en", result.get("en", "x")) else " ".join(signs)
                for lang in still_bad:
                    result[lang] = english
                result["_speech_langs"] = {lang: "en" for lang in still_bad}
        result["source"] = "llm"
        return result

    async def _llm_once(self, provider, messages, signs, langs, request_id, attempt=1):
        payload = {
            "model": provider["model"],
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if provider["name"] == "openai":
            payload["max_completion_tokens"] = 400
        else:
            payload.update(temperature=0.2, max_tokens=300)
        started = time.perf_counter()
        outcome = "error"
        usage: Dict[str, int] = {}
        try:
            resp = await self._http().post(
                provider["url"],
                headers={"Authorization": f"Bearer {provider['key']}"},
                json=payload,
                timeout=provider["timeout"],
            )
            if resp.status_code != 200:
                outcome = f"http_{resp.status_code}"
                return None
            body = resp.json()
            usage = body.get("usage") or {}
            parsed = json.loads(body["choices"][0]["message"]["content"])
            result = self._validate(parsed, langs)
            if result is None:
                outcome = "invalid_schema"
                return None
            outcome = "ok"
            return result
        except httpx.TimeoutException:
            outcome = "timeout"
            return None
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            outcome = f"error:{type(exc).__name__}"
            return None
        finally:
            # Observability without logging user content.
            log.info(
                "llm_call",
                extra={
                    "request_id": request_id,
                    "provider": provider["name"],
                    "model": provider["model"],
                    "prompt_version": config.PROMPT_VERSION,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "sign_count": len(signs),
                    "attempt": attempt,
                    "outcome": outcome,
                },
            )

    @staticmethod
    def _validate(parsed: object, langs: List[str]) -> Optional[Dict[str, str]]:
        if not isinstance(parsed, dict):
            return None
        english = parsed.get("en")
        out: Dict[str, str] = {}
        for lang in langs:
            value = parsed.get(lang)
            if isinstance(value, str) and value.strip():
                out[lang] = value.strip()[:300]
            elif isinstance(english, str) and english.strip():
                out[lang] = english.strip()[:300]
            else:
                return None
        return out
