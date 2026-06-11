from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Voice:
    label: str
    id: str


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    display: str
    native_name: str
    flag: str
    prompt_name: str
    script_name: str
    script_sample: str
    sentence_hint: str
    default_voice: str
    voices: List[Voice]
    font_stack: str


LANGUAGES: Dict[str, LanguageProfile] = {
    "es": LanguageProfile(
        code="es",
        display="Spanish",
        native_name="Español",
        flag="🇪🇸",
        prompt_name="Spanish",
        script_name="Latin alphabet with accents",
        script_sample="A B C Ñ á é í ó ú ü ¿ ¡",
        sentence_hint="Use natural Spanish with correct accents and punctuation.",
        default_voice="es-ES-AlvaroNeural",
        voices=[
            Voice("Spain, Alvaro", "es-ES-AlvaroNeural"),
            Voice("Spain, Elvira", "es-ES-ElviraNeural"),
            Voice("Mexico, Dalia", "es-MX-DaliaNeural"),
            Voice("US, Paloma", "es-US-PalomaNeural"),
        ],
        font_stack='Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    ),
    "ru": LanguageProfile(
        code="ru",
        display="Russian",
        native_name="Русский",
        flag="🇷🇺",
        prompt_name="Russian",
        script_name="Cyrillic",
        script_sample="А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ы Э Ю Я",
        sentence_hint="Use natural Russian in Cyrillic, including Ё where it is normally written.",
        default_voice="ru-RU-DmitryNeural",
        voices=[
            Voice("Russia, Dmitry", "ru-RU-DmitryNeural"),
            Voice("Russia, Svetlana", "ru-RU-SvetlanaNeural"),
        ],
        font_stack='Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Arial Unicode MS", sans-serif',
    ),
    "fr": LanguageProfile(
        code="fr",
        display="French",
        native_name="Français",
        flag="🇫🇷",
        prompt_name="French",
        script_name="Latin alphabet with French accents",
        script_sample="A B C Ç à â æ é è ê ë î ï ô œ ù û ü ÿ",
        sentence_hint="Use natural French with correct accents and typography.",
        default_voice="fr-FR-HenriNeural",
        voices=[
            Voice("France, Henri", "fr-FR-HenriNeural"),
            Voice("France, Denise", "fr-FR-DeniseNeural"),
            Voice("Canada, Sylvie", "fr-CA-SylvieNeural"),
        ],
        font_stack='Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    ),
    "zh": LanguageProfile(
        code="zh",
        display="Mandarin Chinese",
        native_name="普通话",
        flag="🇨🇳",
        prompt_name="Mandarin Chinese",
        script_name="Simplified Chinese characters",
        script_sample="我 你 他 她 学 语 文 中 国 普 通 话",
        sentence_hint="Use Simplified Chinese characters. Do not add pinyin unless explicitly requested.",
        default_voice="zh-CN-YunxiNeural",
        voices=[
            Voice("Mainland, Yunxi", "zh-CN-YunxiNeural"),
            Voice("Mainland, Xiaoxiao", "zh-CN-XiaoxiaoNeural"),
            Voice("Taiwan, YunJhe", "zh-TW-YunJheNeural"),
            Voice("Taiwan, HsiaoChen", "zh-TW-HsiaoChenNeural"),
        ],
        font_stack='"PingFang SC", "Hiragino Sans GB", "Heiti SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif',
    ),
}


LANGUAGE_ALIASES = {
    "spanish": "es",
    "español": "es",
    "espanol": "es",
    "es": "es",
    "russian": "ru",
    "русский": "ru",
    "ru": "ru",
    "french": "fr",
    "français": "fr",
    "francais": "fr",
    "fr": "fr",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "chinese": "zh",
    "中文": "zh",
    "普通话": "zh",
    "zh": "zh",
    "zh-cn": "zh",
}


def normalize_language_code(value: Optional[str]) -> str:
    if not value:
        return "es"
    raw = str(value).strip()
    if raw in LANGUAGES:
        return raw
    return LANGUAGE_ALIASES.get(raw.lower(), "es")


def get_language(value: Optional[str]) -> LanguageProfile:
    return LANGUAGES[normalize_language_code(value)]


def public_language_payload() -> List[dict]:
    return [
        {
            "code": profile.code,
            "display": profile.display,
            "nativeName": profile.native_name,
            "flag": profile.flag,
            "scriptName": profile.script_name,
            "scriptSample": profile.script_sample,
            "defaultVoice": profile.default_voice,
            "voices": [{"label": voice.label, "id": voice.id} for voice in profile.voices],
            "fontStack": profile.font_stack,
        }
        for profile in LANGUAGES.values()
    ]
