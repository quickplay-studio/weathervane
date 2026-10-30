"""Free-tier LLM router: Groq, Gemini and OpenRouter over plain HTTPS. No Anthropic SDK, no Claude API."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date

import requests

from . import config

log = logging.getLogger("weathervane.llm")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_TOKENS = 8192
KEEP_DAYS = 45


class LLMError(Exception):
    """No provider produced an answer."""


@dataclass(frozen=True)
class Provider:
    provider: str
    model: str
    env_key: str
    daily_cap: int

    @property
    def key(self) -> str | None:
        return os.environ.get(self.env_key) or None

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


def providers_from(cfg: dict) -> list[Provider]:
    out = []
    for p in cfg["llm"]["providers"]:
        model = os.environ.get(f"WV_{p['provider'].upper()}_MODEL") or p["model"]
        out.append(Provider(p["provider"], model, p["env_key"], int(p.get("daily_cap", 1000))))
    return out


def redact(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


def parse_json_reply(text: str) -> dict:
    """Lenient JSON object parse: strips code fences and surrounding prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        lo, hi = cleaned.find("{"), cleaned.rfind("}")
        if lo < 0 or hi <= lo:
            raise ValueError("reply has no JSON object")
        obj = json.loads(cleaned[lo: hi + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


class Router:
    """Tries providers in order, skips those without a key or past their daily cap, records usage per day."""

    def __init__(self, cfg: dict, session: requests.Session | None = None, today: str | None = None):
        self.cfg = cfg["llm"]
        self.providers = providers_from(cfg)
        self.http = session or requests.Session()
        self.today = today or date.today().isoformat()
        self.usage_path = config.path("results", "llm_usage.json")
        self.usage = config.read_json(self.usage_path, {})
        self.usage.setdefault(self.today, {})

    def _used(self, label: str) -> int:
        return self.usage[self.today].get(label, {}).get("requests", 0)

    def _total(self) -> int:
        return sum(v.get("requests", 0) for v in self.usage[self.today].values())

    def usable(self) -> list[Provider]:
        if self._total() >= self.cfg["daily_requests"]:
            return []
        return [p for p in self.providers if p.key and self._used(p.label) < p.daily_cap]

    def available(self) -> bool:
        return bool(self.usable())

    def _record(self, label: str, ok: bool) -> None:
        row = self.usage[self.today].setdefault(label, {"requests": 0, "errors": 0})
        row["requests"] += 1
        row["errors"] += 0 if ok else 1
        for day in sorted(self.usage)[:-KEEP_DAYS]:
            del self.usage[day]
        config.write_json(self.usage_path, self.usage)

    def complete(self, system: str, user: str) -> tuple[str, str]:
        """Return (reply text, provider label). Raises LLMError when every usable provider fails."""
        errors = []
        for p in self.usable():
            try:
                text = self._ask(p, system, user)
                self._record(p.label, True)
                return text, p.label
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                self._record(p.label, False)
                errors.append(f"{p.label}: {redact(f'{type(exc).__name__} {exc}', [p.key or ''])[:160]}")
                log.warning("LLM provider failed: %s", errors[-1])
        raise LLMError("; ".join(errors) or "no provider has a key and budget left")

    def _ask(self, p: Provider, system: str, user: str) -> str:
        timeout, temp = self.cfg["timeout_s"], self.cfg["temperature"]
        if p.provider == "gemini":
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": temp, "responseMimeType": "application/json",
                                         "maxOutputTokens": MAX_TOKENS}}
            resp = self.http.post(GEMINI_URL.format(model=p.model), params={"key": p.key}, json=body, timeout=timeout)
            _raise_for_status(resp)
            parts = resp.json()["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts)
        url = GROQ_URL if p.provider == "groq" else OPENROUTER_URL
        body = {"model": p.model, "temperature": temp, "max_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {p.key}"}
        if p.provider == "groq":
            body["response_format"] = {"type": "json_object"}
        resp = self.http.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code == 400 and "response_format" in body:      # model without JSON mode: ask again without it
            del body["response_format"]
            resp = self.http.post(url, headers=headers, json=body, timeout=timeout)
        _raise_for_status(resp)
        return resp.json()["choices"][0]["message"]["content"] or ""


def _raise_for_status(resp: requests.Response) -> None:
    """Any error status (429 and 5xx included) sends the router on to the next provider."""
    if resp.status_code >= 400:
        raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:120]}")
