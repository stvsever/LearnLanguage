"""Language profiles: prompts, scripts, neural voices, and speech-recognition locales.

Supported set (v2.1): French (default), Spanish, Russian, Mandarin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import DEFAULT_LANGUAGE


@dataclass(frozen=True)
class Voice:
    id: str
    label: str
    gender: str


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    display: str
    native_name: str
    flag: str
    prompt_name: str
    script_hint: str
    recognition_locale: str
    default_voice: str
    voices: List[Voice]
    font_stack: str = (
        'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif'
    )


LANGUAGES: Dict[str, LanguageProfile] = {
    "fr": LanguageProfile(
        code="fr",
        display="French",
        native_name="Français",
        flag="🇫🇷",
        prompt_name="French",
        script_hint="Natural French with correct accents, elision, and typography (thin spaces before ?!;: are optional, apostrophes for elision are not).",
        recognition_locale="fr-FR",
        default_voice="fr-FR-DeniseNeural",
        voices=[
            Voice("fr-FR-DeniseNeural", "Denise · France", "f"),
            Voice("fr-FR-HenriNeural", "Henri · France", "m"),
            Voice("fr-FR-VivienneMultilingualNeural", "Vivienne · France", "f"),
            Voice("fr-FR-RemyMultilingualNeural", "Rémy · France", "m"),
            Voice("fr-CA-SylvieNeural", "Sylvie · Québec", "f"),
            Voice("fr-CA-AntoineNeural", "Antoine · Québec", "m"),
        ],
    ),
    "es": LanguageProfile(
        code="es",
        display="Spanish",
        native_name="Español",
        flag="🇪🇸",
        prompt_name="Spanish",
        script_hint="Natural Spanish with correct accents and inverted punctuation (¿…? ¡…!).",
        recognition_locale="es-ES",
        default_voice="es-ES-ElviraNeural",
        voices=[
            Voice("es-ES-ElviraNeural", "Elvira · Spain", "f"),
            Voice("es-ES-AlvaroNeural", "Álvaro · Spain", "m"),
            Voice("es-MX-DaliaNeural", "Dalia · Mexico", "f"),
            Voice("es-MX-JorgeNeural", "Jorge · Mexico", "m"),
        ],
    ),
    "ru": LanguageProfile(
        code="ru",
        display="Russian",
        native_name="Русский",
        flag="🇷🇺",
        prompt_name="Russian",
        script_hint="Natural Russian in Cyrillic, writing ё where standard.",
        recognition_locale="ru-RU",
        default_voice="ru-RU-SvetlanaNeural",
        voices=[
            Voice("ru-RU-SvetlanaNeural", "Svetlana · Russia", "f"),
            Voice("ru-RU-DmitryNeural", "Dmitry · Russia", "m"),
        ],
    ),
    "zh": LanguageProfile(
        code="zh",
        display="Mandarin",
        native_name="普通话",
        flag="🇨🇳",
        prompt_name="Mandarin Chinese",
        script_hint="Simplified Chinese characters. Provide pinyin with tone marks in pronunciation fields.",
        recognition_locale="zh-CN",
        default_voice="zh-CN-XiaoxiaoNeural",
        voices=[
            Voice("zh-CN-XiaoxiaoNeural", "Xiaoxiao · Mainland", "f"),
            Voice("zh-CN-YunxiNeural", "Yunxi · Mainland", "m"),
            Voice("zh-CN-YunjianNeural", "Yunjian · Mainland", "m"),
            Voice("zh-TW-HsiaoChenNeural", "HsiaoChen · Taiwan", "f"),
        ],
        font_stack='"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
    ),
}


def normalize_language_code(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if raw in LANGUAGES:
        return raw
    aliases = {
        "french": "fr", "français": "fr", "francais": "fr",
        "spanish": "es", "español": "es", "espanol": "es",
        "russian": "ru", "русский": "ru",
        "mandarin": "zh", "chinese": "zh", "zh-cn": "zh", "中文": "zh", "普通话": "zh",
    }
    return aliases.get(raw, DEFAULT_LANGUAGE)


def get_language(value: Optional[str]) -> LanguageProfile:
    return LANGUAGES[normalize_language_code(value)]


def public_language_payload() -> List[dict]:
    return [
        {
            "code": p.code,
            "display": p.display,
            "nativeName": p.native_name,
            "flag": p.flag,
            "recognitionLocale": p.recognition_locale,
            "defaultVoice": p.default_voice,
            "voices": [{"id": v.id, "label": v.label, "gender": v.gender} for v in p.voices],
            "fontStack": p.font_stack,
        }
        for p in LANGUAGES.values()
    ]
