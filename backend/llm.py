"""LLM access layer.

Provider chain: OpenRouter (DeepSeek v4 flash by default) -> OpenAI -> offline.
All calls request JSON output, extract it defensively, validate it against a
pydantic schema, and retry once with the validation error fed back to the model.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from . import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# OpenRouter slugs look like "vendor/model-name"; used to sanitize overrides.
_MODEL_SLUG_RE = re.compile(r"^[a-z0-9-]+/[a-z0-9._:-]+$", re.I)


def resolve_model_override(requested: Optional[str], provider: str) -> Optional[str]:
    """Validate a client-requested model. Only honored on OpenRouter."""
    if not requested or provider != "openrouter":
        return None
    slug = str(requested).strip()
    return slug if _MODEL_SLUG_RE.fullmatch(slug) else None

_client: Optional[OpenAI] = None
_client_provider: Optional[str] = None


class LLMUnavailable(RuntimeError):
    """No LLM provider is configured or the provider call failed."""


def reset_client() -> None:
    """Drop the cached client, e.g. after the API key changed at runtime."""
    global _client, _client_provider
    _client = None
    _client_provider = None


def get_client() -> tuple[OpenAI, str, str]:
    """Return (client, provider, model) for the active provider."""
    global _client, _client_provider
    provider = config.active_provider()
    if provider == "offline":
        raise LLMUnavailable("No API key configured (set OPENROUTER_API_KEY in .env).")
    if _client is None or _client_provider != provider:
        if provider == "openrouter":
            _client = OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
                timeout=config.LLM_TIMEOUT_SECONDS,
                max_retries=1,
                default_headers={
                    "HTTP-Referer": "https://github.com/stvsever/LearnLanguage",
                    "X-Title": "AI-Studio for Learning Languages",
                },
            )
        else:
            _client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=config.LLM_TIMEOUT_SECONDS,
                max_retries=1,
            )
        _client_provider = provider
    model = config.OPENROUTER_MODEL if provider == "openrouter" else config.OPENAI_MODEL
    return _client, provider, model


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, tolerating fences and prose."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty model response.")
    fence = _JSON_BLOCK_RE.search(raw)
    if fence:
        raw = fence.group(1).strip()
    if not raw.startswith("{"):
        start = raw.find("{")
        if start == -1:
            raise ValueError("No JSON object found in model response.")
        raw = raw[start:]
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(raw)
    if not isinstance(obj, dict):
        raise ValueError("Model response was not a JSON object.")
    return obj


def schema_hint(response_model: Type[BaseModel]) -> str:
    """Compact JSON-schema description embedded in the prompt."""
    return json.dumps(response_model.model_json_schema(), ensure_ascii=False)


def generate_structured(system_prompt: str, user_prompt: str, response_model: Type[T],
                        model_override: Optional[str] = None) -> T:
    """Chat completion -> JSON extraction -> pydantic validation, with one repair retry."""
    client, provider, model = get_client()
    override = resolve_model_override(model_override, provider)
    if override:
        model = override
    schema = schema_hint(response_model)
    system = (
        f"{system_prompt}\n\n"
        "Respond with a single JSON object only - no markdown, no commentary.\n"
        "The object must be an INSTANCE conforming to the JSON Schema below - "
        "actual data values, never the schema itself, never 'properties'/'type' keys.\n"
        "Never use the em dash character (\u2014) in any text you produce; "
        "use a comma, colon, or plain hyphen instead.\n"
        f"JSON Schema:\n{schema}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    # Hybrid-reasoning models (e.g. DeepSeek V4) can spend minutes "thinking"
    # about large JSON schemas; content generation needs none of that. Some
    # models (GPT-5 family) refuse to disable reasoning - fall back to minimal
    # effort for those.
    def create(msgs):
        if provider != "openrouter":
            return client.chat.completions.create(
                model=model, messages=msgs,
                response_format={"type": "json_object"}, temperature=0.7,
            )
        try:
            return client.chat.completions.create(
                model=model, messages=msgs,
                response_format={"type": "json_object"}, temperature=0.7,
                extra_body={"reasoning": {"enabled": False}},
            )
        except Exception as exc:  # noqa: BLE001 - inspect for mandatory-reasoning 400
            if "easoning" not in str(exc):
                raise
            return client.chat.completions.create(
                model=model, messages=msgs,
                response_format={"type": "json_object"}, temperature=0.7,
                extra_body={"reasoning": {"effort": "low"}},
            )

    last_error: Optional[Exception] = None
    for attempt in (1, 2):
        start = time.perf_counter()
        try:
            response = create(messages)
            content = response.choices[0].message.content or ""
            parsed = response_model.model_validate(extract_json(content))
            logger.info(
                "%s/%s generated %s in %.1fs (attempt %d)",
                provider, model, response_model.__name__, time.perf_counter() - start, attempt,
            )
            return parsed
        except Exception as exc:  # noqa: BLE001 - single repair loop, then surface
            last_error = exc
            logger.warning("Generation attempt %d failed: %s", attempt, exc)
            if attempt == 1:
                messages.append({"role": "assistant", "content": "(previous invalid response)"})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was invalid: "
                        f"{str(exc)[:600]}\n"
                        "Return ONLY a corrected JSON object with real data values "
                        "conforming to the schema (not the schema itself)."
                    ),
                })
    raise LLMUnavailable(f"Generation failed after retry: {last_error}")
