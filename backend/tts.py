"""Text-to-speech layer.

Default provider is Microsoft Edge neural TTS (free, no key) via edge-tts,
with a content-addressed on-disk cache. The frontend additionally falls back
to the browser's Web Speech API when this endpoint is unreachable.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import edge_tts

try:
    from mutagen.mp3 import MP3 as MP3Info
except Exception:  # pragma: no cover - mutagen is optional
    MP3Info = None

from .config import AUDIO_DIR, ensure_runtime_dirs
from .languages import get_language

logger = logging.getLogger(__name__)

# UI speed presets -> edge-tts rate strings.
RATE_PRESETS = {
    "slow": "-30%",
    "study": "-12%",
    "natural": "+0%",
    "fast": "+15%",
}


def resolve_rate(rate: Optional[str]) -> str:
    raw = (rate or "study").strip().lower()
    if raw in RATE_PRESETS:
        return RATE_PRESETS[raw]
    if raw.endswith("%") and (raw.startswith("+") or raw.startswith("-")):
        return raw
    return RATE_PRESETS["study"]


def cache_key(text: str, voice: str, rate: str) -> str:
    return hashlib.sha1(json.dumps([text, voice, rate], ensure_ascii=False).encode("utf-8")).hexdigest()


def audio_duration_seconds(path: Path) -> Optional[float]:
    if MP3Info is None:
        return None
    try:
        return round(float(MP3Info(str(path)).info.length), 2)
    except Exception:
        return None


def synthesize(text: str, language_code: str, voice: Optional[str] = None, rate: Optional[str] = None) -> dict:
    ensure_runtime_dirs()
    clean = " ".join((text or "").split())
    if not clean:
        raise ValueError("Text is required for TTS.")
    if len(clean) > 2400:
        clean = clean[:2400]
    language = get_language(language_code)
    voice_id = voice if voice in {v.id for v in language.voices} else language.default_voice
    edge_rate = resolve_rate(rate)
    key = cache_key(clean, voice_id, edge_rate)
    out_path = AUDIO_DIR / f"{language.code}-{key}.mp3"
    cached = out_path.exists() and out_path.stat().st_size > 0
    if not cached:
        async def _run() -> None:
            communicate = edge_tts.Communicate(text=clean, voice=voice_id, rate=edge_rate)
            await communicate.save(str(out_path))

        asyncio.run(_run())
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError("TTS produced no audio.")
    return {
        "language": language.code,
        "voice": voice_id,
        "rate": edge_rate,
        "url": f"/audio/{out_path.name}",
        "durationSeconds": audio_duration_seconds(out_path),
        "cached": cached,
    }
