"""Central configuration: paths, environment, and model/provider settings."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parents[1]

STATIC_DIR = APP_DIR / "static"
SEED_DIR = APP_DIR / "backend" / "seed"
RUNTIME_DIR = APP_DIR / "runtime"
AUDIO_DIR = RUNTIME_DIR / "audio"
CACHE_DIR = RUNTIME_DIR / "cache"

load_dotenv(APP_DIR / ".env")

# Preferred provider: OpenRouter (free-tier friendly, model-agnostic).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("LEARNLANGUAGE_MODEL", "deepseek/deepseek-v4-flash-0731")

# Curated OpenRouter model choices surfaced in Settings (any valid slug works too).
MODEL_CHOICES = [
    {"id": "deepseek/deepseek-v4-flash-0731", "label": "DeepSeek V4 Flash · default"},
    {"id": "deepseek/deepseek-v4-pro-0813", "label": "DeepSeek V4 Pro"},
    {"id": "openai/gpt-5-mini", "label": "GPT-5 Mini"},
    {"id": "openai/gpt-5.2", "label": "GPT-5.2"},
    {"id": "anthropic/claude-haiku-4.5", "label": "Claude Haiku 4.5"},
    {"id": "anthropic/claude-sonnet-5", "label": "Claude Sonnet 5"},
    {"id": "google/gemini-3.5-flash", "label": "Gemini 3.5 Flash"},
    {"id": "moonshotai/kimi-k2.5", "label": "Kimi K2.5"},
]

# Fallback provider: direct OpenAI, used only when OpenRouter is not configured.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("LEARNLANGUAGE_OPENAI_MODEL", "gpt-4o-mini")

LLM_TIMEOUT_SECONDS = float(os.getenv("LEARNLANGUAGE_LLM_TIMEOUT", "120"))

DEFAULT_LANGUAGE = os.getenv("LEARNLANGUAGE_DEFAULT_LANGUAGE", "fr")

APP_NAME = "Language Learning Studio"
APP_VERSION = "2.1.0"


def ensure_runtime_dirs() -> None:
    for path in (AUDIO_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def active_provider() -> str:
    """Which LLM provider will serve generation requests."""
    if OPENROUTER_API_KEY:
        return "openrouter"
    if OPENAI_API_KEY:
        return "openai"
    return "offline"


ENV_PATH = APP_DIR / ".env"


def masked_key() -> str:
    """Display form of the stored OpenRouter key, safe to send to the UI."""
    if not OPENROUTER_API_KEY:
        return ""
    return f"{OPENROUTER_API_KEY[:9]}...{OPENROUTER_API_KEY[-4:]}"


def set_openrouter_key(key: str, env_path: Optional[Path] = None) -> None:
    """Persist a new OpenRouter key to .env and activate it immediately.

    Called from the in-app key setup so no restart is needed. Preserves any
    other lines already present in the .env file.
    """
    global OPENROUTER_API_KEY
    key = (key or "").strip().strip('"').strip("'")
    if not key.startswith("sk-or-") or len(key) < 24 or any(c.isspace() for c in key):
        raise ValueError("That does not look like an OpenRouter key (they start with sk-or-).")
    path = env_path or ENV_PATH
    lines = []
    if path.exists():
        lines = [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("OPENROUTER_API_KEY")
        ]
    lines.append(f'OPENROUTER_API_KEY="{key}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    OPENROUTER_API_KEY = key
    os.environ["OPENROUTER_API_KEY"] = key


def active_model() -> str:
    provider = active_provider()
    if provider == "openrouter":
        return OPENROUTER_MODEL
    if provider == "openai":
        return OPENAI_MODEL
    return "seed-content"
