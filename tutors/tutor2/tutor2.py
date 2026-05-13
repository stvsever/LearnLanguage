# multilingual_scenario_trainer.py
"""
AI Language Scenario Generator & Trainer (macOS-ready, English UI)

Supports Spanish, Russian, French, and Mandarin Chinese with:
1) Target-language generation plus English mirrors.
2) UTF-8 JSON import/export with non-Latin scripts preserved.
3) Script-aware fonts and aligned English translations.
4) Edge-TTS voices per language, playback speed control, and a single OS audio player.
5) One-shot test flow with locked answers and an English review summary.

Install
  pip install -r ../../requirements.txt
  # or:
  pip install openai python-dotenv pydantic pygame mutagen edge-tts

.env
  Add your OpenAI API key as an environment variable named OPENAI_API_KEY.

Run
  python tutor2.py
"""

import os
import re
import sys
import json
import time
import asyncio
import argparse
import logging
import warnings
from typing import List, Literal, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import unicodedata
import tempfile
import random
import threading
import shutil
import subprocess

# --- UI (tkinter) ---
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext, filedialog
    import tkinter.font as tkfont
except Exception:
    print("tkinter is required (ships with macOS Python).")
    raise

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)
try:
    import pygame
    from mutagen.mp3 import MP3 as MP3Info
except Exception:
    print("Please install pygame and mutagen: pip install pygame mutagen")
    raise

# Optional fallback for speed control if no ffmpeg
try:
    from pydub import AudioSegment, effects as pydub_effects
    _PYDUB_AVAILABLE = True
except Exception:
    _PYDUB_AVAILABLE = False

# --- OpenAI + dotenv ---
import openai
from dotenv import load_dotenv

# --- Pydantic v2 ---
from pydantic import BaseModel, Field, field_validator

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# -------------------------------
# OpenAI key
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.critical("No OpenAI API key found in environment variables.")
    raise ValueError("No OpenAI API key found in environment variables.")
openai.api_key = api_key
logger.info("OpenAI API key loaded successfully.")

# ==========================================================
#                 L A N G U A G E   S U P P O R T
# ==========================================================
TargetLanguage = Literal["Spanish", "Russian", "French", "Mandarin Chinese"]
DEFAULT_LANGUAGE: TargetLanguage = "Spanish"


@dataclass(frozen=True)
class LanguageProfile:
    display: TargetLanguage
    native_name: str
    prompt_name: str
    short_code: str
    script_name: str
    script_sample: str
    sentence_hint: str
    default_voice_label: str
    voices: Dict[str, str]
    preferred_fonts: Tuple[str, ...]


LANGUAGE_PROFILES: Dict[str, LanguageProfile] = {
    "Spanish": LanguageProfile(
        display="Spanish",
        native_name="Español",
        prompt_name="Spanish",
        short_code="ES",
        script_name="Latin alphabet with accents",
        script_sample="A B C Ñ á é í ó ú ü ¿ ¡",
        sentence_hint="Use standard Spanish punctuation and accents.",
        default_voice_label="Spain · Alvaro",
        voices={
            "Spain · Alvaro": "es-ES-AlvaroNeural",
            "Spain · Elvira": "es-ES-ElviraNeural",
            "Mexico · Dalia": "es-MX-DaliaNeural",
            "US · Paloma": "es-US-PalomaNeural",
        },
        preferred_fonts=("Helvetica", "Arial", "TkDefaultFont"),
    ),
    "Russian": LanguageProfile(
        display="Russian",
        native_name="Русский",
        prompt_name="Russian",
        short_code="RU",
        script_name="Cyrillic",
        script_sample="А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ы Э Ю Я",
        sentence_hint="Use natural Russian in Cyrillic, including Ё when it is normally written.",
        default_voice_label="Russia · Dmitry",
        voices={
            "Russia · Dmitry": "ru-RU-DmitryNeural",
            "Russia · Svetlana": "ru-RU-SvetlanaNeural",
        },
        preferred_fonts=("Helvetica", "Arial", "Arial Unicode MS", "TkDefaultFont"),
    ),
    "French": LanguageProfile(
        display="French",
        native_name="Français",
        prompt_name="French",
        short_code="FR",
        script_name="Latin alphabet with French accents",
        script_sample="A B C Ç à â æ é è ê ë î ï ô œ ù û ü ÿ",
        sentence_hint="Use natural French typography and accents.",
        default_voice_label="France · Henri",
        voices={
            "France · Henri": "fr-FR-HenriNeural",
            "France · Denise": "fr-FR-DeniseNeural",
            "Canada · Sylvie": "fr-CA-SylvieNeural",
        },
        preferred_fonts=("Helvetica", "Arial", "TkDefaultFont"),
    ),
    "Mandarin Chinese": LanguageProfile(
        display="Mandarin Chinese",
        native_name="普通话",
        prompt_name="Mandarin Chinese",
        short_code="ZH",
        script_name="Simplified Chinese characters",
        script_sample="我 你 他 她 学 语 文 中 国 普 通 话",
        sentence_hint="Use Simplified Chinese characters and natural Mandarin. Do not romanize unless the topic explicitly asks for pinyin.",
        default_voice_label="Mainland · Yunxi",
        voices={
            "Mainland · Yunxi": "zh-CN-YunxiNeural",
            "Mainland · Xiaoxiao": "zh-CN-XiaoxiaoNeural",
            "Taiwan · YunJhe": "zh-TW-YunJheNeural",
            "Taiwan · HsiaoChen": "zh-TW-HsiaoChenNeural",
        },
        preferred_fonts=("PingFang SC", "Songti SC", "Heiti SC", "Arial Unicode MS", "TkDefaultFont"),
    ),
}

LANGUAGE_LABELS = list(LANGUAGE_PROFILES.keys())
LANGUAGE_ALIASES = {
    "spanish": "Spanish",
    "es": "Spanish",
    "español": "Spanish",
    "russian": "Russian",
    "ru": "Russian",
    "русский": "Russian",
    "french": "French",
    "fr": "French",
    "français": "French",
    "francais": "French",
    "mandarin": "Mandarin Chinese",
    "mandarin chinese": "Mandarin Chinese",
    "chinese": "Mandarin Chinese",
    "zh": "Mandarin Chinese",
    "zh-cn": "Mandarin Chinese",
    "中文": "Mandarin Chinese",
    "普通话": "Mandarin Chinese",
}


def normalize_language_label(value: Optional[str]) -> TargetLanguage:
    if not value:
        return DEFAULT_LANGUAGE
    raw = str(value).strip()
    if raw in LANGUAGE_PROFILES:
        return raw  # type: ignore[return-value]
    return LANGUAGE_ALIASES.get(raw.lower(), DEFAULT_LANGUAGE)  # type: ignore[return-value]


def get_language_profile(language: Optional[str]) -> LanguageProfile:
    return LANGUAGE_PROFILES[normalize_language_label(language)]


def detect_language_from_text(text: str) -> TargetLanguage:
    """Best-effort fallback for legacy imports that have no language metadata."""
    if re.search(r"[\u4e00-\u9fff]", text or ""):
        return "Mandarin Chinese"
    if re.search(r"[\u0400-\u04ff]", text or ""):
        return "Russian"
    if re.search(r"[¿¡ñÑ]", text or ""):
        return "Spanish"
    if re.search(r"[çÇœŒàâêëîïôùûüÿ]", text or ""):
        return "French"
    return DEFAULT_LANGUAGE

# ==========================================================
#                  P Y D A N T I C   S C H E M A
# ==========================================================
class QAItem(BaseModel):
    # Target language
    question: str = Field(..., description="Target-language question requiring non-trivial inference.")
    choices: List[str] = Field(..., min_items=4, max_items=4, description="Exactly 4 plausible options in the target language.")
    reasoning_note: str = Field(..., description="1-2 sentences explaining correctness in the target language.")
    # English mirror for UI toggle
    question_en: str = Field(..., description="English version of the question.")
    choices_en: List[str] = Field(..., min_items=4, max_items=4, description="Exactly 4 plausible options in English.")
    reasoning_note_en: str = Field(..., description="1–2 sentences explaining correctness in English.")
    # Shared
    correct_choice: int = Field(..., ge=0, le=3, description="Index (0..3) of the correct option.")
    accepted_answers: List[str] = Field(
        ...,
        min_items=1,
        description="Short correct strings in the target language (canonical + paraphrases)."
    )

    @field_validator("question")
    @classmethod
    def v_question_target(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Question is too short/trivial.")
        return v

    @field_validator("question_en")
    @classmethod
    def v_question_en(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("English question is too short/trivial.")
        return v

    @field_validator("choices")
    @classmethod
    def v_choices_target(cls, v: List[str]) -> List[str]:
        lowered = [c.strip().lower() for c in v]
        banned = {
            "todas las anteriores", "ninguna de las anteriores", "ambas a y b", "ambas b y c",
            "toutes les réponses ci-dessus", "aucune des réponses ci-dessus",
            "все вышеперечисленное", "ничего из вышеперечисленного",
            "以上皆是", "以上都不是",
            "all of the above", "none of the above", "both a and b", "both b and c"
        }
        if any(c in banned for c in lowered):
            raise ValueError("Choices contain banned meta options.")
        for c in v:
            if len(c) > 140:
                raise ValueError("Each choice must be concise (<140 chars).")
        return v

    @field_validator("choices_en")
    @classmethod
    def v_choices_en(cls, v: List[str]) -> List[str]:
        lowered = [c.strip().lower() for c in v]
        banned = {"all of the above", "none of the above", "both a and b", "both b and c"}
        if any(c in banned for c in lowered):
            raise ValueError("English choices contain banned meta options.")
        for c in v:
            if len(c) > 140:
                raise ValueError("Each choice must be concise (<140 chars).")
        return v

    @field_validator("accepted_answers")
    @classmethod
    def v_accepteds(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError("At least one accepted answer required.")
        return v


class ScenarioPack(BaseModel):
    target_language: TargetLanguage = Field(DEFAULT_LANGUAGE, description="Target language for the passage, questions, choices, and target-language explanations.")
    scenario_title: str = Field(..., description="Short scenario title.")
    difficulty: Literal["easy", "medium", "hard"]
    passage: str = Field(..., description="Target-language passage (target: 3-5 sentences, specific, self-contained).")
    passage_en_lines: List[str] = Field(..., description="English translations aligned 1:1 with target-language sentences.")
    items: List[QAItem] = Field(..., min_items=5, max_items=5)

# ==========================================================
#                     O P E N A I   C A L L
# ==========================================================
def call_GPT(
    system_prompt: str,
    user_query: str,
    pydantic_model: BaseModel,
    model: str = "gpt-5-nano",
) -> BaseModel:
    try:
        response = openai.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            response_format=pydantic_model,
        )
        logger.info("LLM response generated successfully")
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        raise

def build_system_prompt(target_language: str) -> str:
    profile = get_language_profile(target_language)
    return f"""You are a rigorous educational content generator.
Goal: produce ONE language-learning scenario package.

Target language: {profile.prompt_name} ({profile.native_name})
Script and orthography: {profile.sentence_hint}

You MUST:
- Set `target_language` exactly to "{profile.display}".
- Write the passage, original questions, original choices, target-language reasoning notes, and accepted answers in {profile.prompt_name}.
- Write a passage of EXACTLY 3-5 sentences about the user's topic. Make it specific and information-dense: entities, dates, quantities, restrictions, conditions, and trade-offs.
- Return `passage_en_lines`: English translations line by line, with the same number of sentences and the same order as the target-language passage.
- Write 5 demanding comprehension questions that require:
  - inference across multiple sentences,
  - temporal order, quantities, cause vs. correlation,
  - conditions/exceptions,
  - implicit intention or trade-offs.
- For each question:
  - Provide EXACTLY 4 plausible choices, only one correct.
  - Set `correct_choice` to the correct index from 0 to 3.
  - Include `accepted_answers`, and include the exact correct target-language choice text plus at least 2-3 short target-language paraphrases.
- Also return English mirrors for every question:
  - `question_en`: English version of the question.
  - `choices_en`: 4 English choices aligned by index with the target-language choices.
  - `reasoning_note_en`: brief English explanation of why the answer is correct.
- Vary the correct answer index; do not make A correct every time.

Forbidden:
- Meta answers such as "all of the above", "none of the above", "both A and B", or equivalents in the target language.
- Choices longer than 140 characters.
- Markdown, commentary, or text outside the JSON object.

Output:
- Must conform exactly to the provided Pydantic JSON schema.
- Original language: {profile.prompt_name}.
- English mirror fields: English only.
"""


def build_user_query(topic: str, difficulty: str, target_language: str) -> str:
    profile = get_language_profile(target_language)
    return f"""Topic: {topic.strip()}
Target language: {profile.display}
Difficulty: {difficulty.lower().strip()}
Additional instructions:
- The passage must have EXACTLY 3-5 target-language sentences.
- Return `passage_en_lines` as English translations, one per target-language sentence, in the same order.
- In every `accepted_answers`, include the exact correct target-language choice plus short target-language paraphrases.
- For every question, include `question_en`, `choices_en` aligned with the target-language choices, and `reasoning_note_en`.
- Return JSON only."""

# ==========================================================
#        F I X U P   /   P O S T - P R O C E S S I N G
# ==========================================================
_SENTENCE_RE = re.compile(r".+?(?:[.!?。！？]+[\"'”’»）)]*|$)", flags=re.S)


def split_sentences(s: str) -> List[str]:
    text = (s or "").strip()
    if not text:
        return []
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    source = " ".join(lines)
    parts = [p.strip() for p in _SENTENCE_RE.findall(source) if p.strip()]
    return parts or [source]


def count_sentences(s: str) -> int:
    return len(split_sentences(s))


def count_sentences_spanish(s: str) -> int:
    return count_sentences(s)


def split_spanish_sentences(s: str) -> List[str]:
    return split_sentences(s)


def fixup_pack(sp: ScenarioPack, target_language: str = DEFAULT_LANGUAGE) -> ScenarioPack:
    sp.target_language = normalize_language_label(getattr(sp, "target_language", target_language))
    changed = False
    for it in sp.items:
        idx = it.correct_choice
        if 0 <= idx < len(it.choices):
            correct_text = it.choices[idx].strip()
            if not any(a.strip().lower() == correct_text.lower() for a in it.accepted_answers):
                it.accepted_answers = [correct_text] + it.accepted_answers
                changed = True
        # Guard: ensure choices_en length is 4
        if not it.choices_en or len(it.choices_en) != 4:
            # Fallback: mirror target-language choices to avoid UI crash.
            it.choices_en = list(it.choices)[:4]

    target_lines = split_sentences(sp.passage)
    if len(sp.passage_en_lines) != len(target_lines):
        logger.warning("Translation line count mismatch; attempting to harmonize by truncating/expanding.")
        if len(sp.passage_en_lines) > len(target_lines):
            sp.passage_en_lines = sp.passage_en_lines[:len(target_lines)]
        else:
            sp.passage_en_lines = sp.passage_en_lines + [""] * (len(target_lines) - len(sp.passage_en_lines))

    ns = count_sentences(sp.passage)
    if ns < 3 or ns > 5:
        logger.warning(f"Passage has {ns} sentences (expected 3-5). You can regenerate if needed.")

    if changed:
        logger.info("Auto-fixed accepted_answers to include exact correct choice.")
    return sp

def generate_with_retries(topic: str, difficulty: str, model: str, target_language: str, attempts: int = 3) -> ScenarioPack:
    last_err: Optional[str] = None
    language = normalize_language_label(target_language)
    sys_prompt = build_system_prompt(language)
    user_query = build_user_query(topic, difficulty, language)

    for attempt in range(1, attempts + 1):
        try:
            sp: ScenarioPack = call_GPT(sys_prompt, user_query, ScenarioPack, model=model)
            sp = fixup_pack(sp, language)
            n_sent = count_sentences(sp.passage)
            if (n_sent < 3 or n_sent > 5) and attempt < attempts:
                logger.warning(f"Retrying due to sentence count ({n_sent}). Attempt {attempt+1}/{attempts}.")
                user_query += "\n\nIMPORTANT: The passage must have exactly 3-5 target-language sentences. Follow this strictly."
                continue
            return sp
        except Exception as e:
            last_err = str(e)
            logger.warning(f"Generation attempt {attempt} failed: {last_err}")
    raise RuntimeError(f"Failed to generate a valid ScenarioPack after {attempts} attempts. Last error: {last_err}")

# ==========================================================
#        C O N V E R S I O N   &   F I L E   H E L P E R S
# ==========================================================
def to_trainer_payload(sp: ScenarioPack) -> Dict[str, List[Dict[str, Any]]]:
    """
    Trainer payload (legacy UI format) repeats the passage inside each item to simplify UI logic.
    The generator itself can be compact; this conversion is only for the trainer GUI.
    """
    items = []
    target_language = normalize_language_label(getattr(sp, "target_language", DEFAULT_LANGUAGE))
    for q in sp.items:
        items.append({
            "target_language": target_language,
            "sentence": sp.passage,
            "sentence_en_lines": sp.passage_en_lines,  # aligned translations
            "question": q.question,
            "question_en": q.question_en,
            "choices": q.choices,
            "choices_en": q.choices_en,
            "correct_index": q.correct_choice,
            "explain": q.reasoning_note,
            "explain_en": q.reasoning_note_en,
            "answers": q.accepted_answers
        })
    return {sp.scenario_title: items}

def slugify(s: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in s.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_") or "scenario"

def save_bundle(sp: ScenarioPack, out_path: str):
    raw_dict = json.loads(sp.json())
    trainer_dict = to_trainer_payload(sp)
    bundle = {"raw": raw_dict, "trainer": trainer_dict}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {out_path}")

# -------------- Robust import normalization ---------------
def _item_with_passage(item: Dict[str, Any],
                       passage: str,
                       passage_en_lines: List[str],
                       target_language: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """Normalize a single compact item into the trainer item shape."""
    out = dict(item)  # shallow copy
    item_language = out.get("target_language") or out.get("language") or target_language
    out["target_language"] = normalize_language_label(item_language)
    # Normalize keys expected by trainer
    if "correct_index" not in out and "correct_choice" in out:
        out["correct_index"] = out.pop("correct_choice")
    if "answers" not in out and "accepted_answers" in out:
        out["answers"] = out.pop("accepted_answers")
    # Inject passage (legacy UI expects it per item)
    out["sentence"] = passage
    out["sentence_en_lines"] = passage_en_lines
    # Ensure required fields exist even if empty
    out.setdefault("question_en", out.get("question", ""))
    out.setdefault("choices_en", out.get("choices", ["", "", "", ""]))
    out.setdefault("explain_en", out.get("explain", ""))
    return out

def _convert_compact_block_to_trainer(title: str, block: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert {passage, passage_en_lines, items:[...]} into (title, [trainer_items]).
    Pass-through extra top-level keys (e.g., words_es, rules) is deliberately ignored
    for trainer compatibility.
    """
    passage = block.get("passage") or block.get("sentence") or ""
    pel = block.get("passage_en_lines") or block.get("sentence_en_lines") or []
    target_language = normalize_language_label(
        block.get("target_language") or block.get("language") or detect_language_from_text(passage)
    )
    items = block.get("items", [])
    trainer_items = [_item_with_passage(it, passage, pel, target_language) for it in items]
    return title, trainer_items


def _normalize_trainer_mapping(
    trainer_dict: Dict[str, List[Dict[str, Any]]],
    fallback_language: str = DEFAULT_LANGUAGE
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for title, items in trainer_dict.items():
        if not isinstance(items, list):
            continue
        normalized_items = []
        detected = fallback_language
        if items and isinstance(items[0], dict):
            detected = items[0].get("target_language") or items[0].get("language") or detect_language_from_text(items[0].get("sentence", ""))
        for item in items:
            if not isinstance(item, dict):
                continue
            item_copy = dict(item)
            item_copy["target_language"] = normalize_language_label(
                item_copy.get("target_language") or item_copy.get("language") or detected
            )
            item_copy.setdefault("question_en", item_copy.get("question", ""))
            item_copy.setdefault("choices_en", item_copy.get("choices", ["", "", "", ""]))
            item_copy.setdefault("explain_en", item_copy.get("explain", ""))
            normalized_items.append(item_copy)
        if normalized_items:
            out[title] = normalized_items
    return out

def _is_trainer_mapping(d: Dict[str, Any]) -> bool:
    # title -> list of items (each item should be a dict)
    return all(isinstance(v, list) for v in d.values())

def _is_compact_mapping(d: Dict[str, Any]) -> bool:
    # title -> {passage, items:[...]}
    return all(isinstance(v, dict) and "items" in v for v in d.values())

def load_trainer_from_bundle(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads many possible JSON shapes and returns the **trainer mapping**:
        { "<title>": [ {sentence, sentence_en_lines, question, choices, choices_en, correct_index, explain, explain_en, answers}, ... ] }
    Accepted inputs:
      • A file previously saved by this app: {"trainer": {...}}  (preferred)
      • A compact difficulty file from the generator: {title: {passage, passage_en_lines, items:[...]}, ...}
      • A single compact ScenarioPack dict: {"scenario_title": "...", "target_language": "...", "passage": "...", "passage_en_lines": [...], "items":[...]}
      • Legacy trainer mapping (already in target format).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: bundle with "trainer"
    if isinstance(data, dict) and "trainer" in data and isinstance(data["trainer"], dict):
        logger.info("Detected app bundle with 'trainer' field. Using it directly.")
        raw_language = DEFAULT_LANGUAGE
        if isinstance(data.get("raw"), dict):
            raw_language = data["raw"].get("target_language", DEFAULT_LANGUAGE)
        return _normalize_trainer_mapping(data["trainer"], raw_language)

    # Case 2: legacy trainer mapping
    if isinstance(data, dict) and _is_trainer_mapping(data):
        logger.info("Detected legacy trainer mapping (title -> list).")
        return _normalize_trainer_mapping(data)

    # Case 3: compact mapping (title -> {passage, items})
    if isinstance(data, dict) and _is_compact_mapping(data):
        logger.info("Detected compact scenario mapping (title -> {passage, items}). Converting to trainer format.")
        out: Dict[str, List[Dict[str, Any]]] = {}
        for title, block in data.items():
            t, items = _convert_compact_block_to_trainer(title, block)
            out[t] = items
        return out

    # Case 4: app bundle with "raw" single pack
    if isinstance(data, dict) and "raw" in data and isinstance(data["raw"], dict):
        raw = data["raw"]
        title = raw.get("scenario_title", "Imported Scenario")
        logger.info("Detected app bundle with 'raw' single ScenarioPack. Converting to trainer format.")
        _, items = _convert_compact_block_to_trainer(title, raw)
        return {title: items}

    # Case 5: single compact ScenarioPack (no wrapping)
    if isinstance(data, dict) and "scenario_title" in data and "items" in data:
        title = data.get("scenario_title") or "Imported Scenario"
        logger.info("Detected single compact ScenarioPack dict. Converting to trainer format.")
        _, items = _convert_compact_block_to_trainer(title, data)
        return {title: items}

    # Case 6: list of packs?
    if isinstance(data, list):
        logger.info("Detected list at top-level; attempting to collect compact packs.")
        out: Dict[str, List[Dict[str, Any]]] = {}
        for pack in data:
            if isinstance(pack, dict) and "scenario_title" in pack and "items" in pack:
                title = pack.get("scenario_title") or "Imported Scenario"
                _, items = _convert_compact_block_to_trainer(title, pack)
                out[title] = items
        if out:
            return out

    # Fallback: unknown shape
    raise ValueError("Unrecognized file format. Expected trainer mapping, app bundle, or compact scenario mapping.")

# ==========================================================
#              A U D I O   ( E d g e - T T S  +  P y g a m e )
# ==========================================================
import edge_tts  # pip install edge-tts

class Synthesizer:
    """Edge-TTS cache with built-in rate control (no ffmpeg pitch issues)."""
    def __init__(self, lang="es"):
        self.lang = lang
        self.base_cache = {}  # (text, voice, rate) -> (path, length)
        self._neutral_bias_pct = -15  # tweak: 1.0 speed feels natural for study playback

    def _rate_from_speed(self, speed: float) -> str:
        # Map 0.25..2.0 to Edge's percentage-based rate control.
        pct = self._neutral_bias_pct + (speed - 1.0) * 100.0
        pct = int(round(max(-75, min(100, pct))))  # clamp to a safe range
        return f"{pct:+d}%"

    def get_audio(self, text: str, tld_or_voice: str, speed: float):
        voice = tld_or_voice
        rate = self._rate_from_speed(speed)
        key = (text, voice, rate)
        if key in self.base_cache and os.path.exists(self.base_cache[key][0]):
            p, l = self.base_cache[key]
            return p, l, None

        fd, out_path = tempfile.mkstemp(prefix="tts_", suffix=".mp3"); os.close(fd)

        async def _synth():
            comm = edge_tts.Communicate(text, voice=voice, rate=rate)
            await comm.save(out_path)

        asyncio.run(_synth())
        length = float(MP3Info(out_path).info.length)
        self.base_cache[key] = (out_path, length)
        return out_path, length, None


class AudioController:
    """Single global player using pygame.mixer.music with pause/resume and seeking."""
    def __init__(self) -> None:
        pygame.mixer.init()
        self.loaded_path: Optional[str] = None
        self.total_len: float = 0.0
        self.playing: bool = False
        self.paused: bool = False
        self.current_start_sec: float = 0.0

    def load(self, path: str, total_len: float) -> None:
        if self.loaded_path != path:
            pygame.mixer.music.load(path)
            self.loaded_path = path
            self.total_len = total_len
            self.playing = False
            self.paused = False
            self.current_start_sec = 0.0
        else:
            # Keep duration in sync even if reloading same file
            self.total_len = total_len

    def play(self, start_sec: float = 0.0) -> None:
        if self.loaded_path is None:
            return
        if start_sec < 0.0:
            start_sec = 0.0
        if self.total_len > 0.0 and start_sec > self.total_len:
            start_sec = max(0.0, self.total_len - 0.05)
        pygame.mixer.music.play(loops=0, start=float(start_sec))
        self.current_start_sec = float(start_sec)
        self.playing = True
        self.paused = False

    def pause(self) -> None:
        if self.playing and not self.paused:
            pygame.mixer.music.pause()
            self.paused = True

    def resume(self) -> None:
        if self.playing and self.paused:
            pos = self.get_pos_sec()
            pygame.mixer.music.play(loops=0, start=float(pos))
            self.current_start_sec = float(pos)
            self.paused = False

    def stop(self) -> None:
        pygame.mixer.music.stop()
        self.playing = False
        self.paused = False

    def get_pos_sec(self) -> float:
        if not self.playing:
            return 0.0
        ms = pygame.mixer.music.get_pos()
        if ms < 0:
            ms = 0
        pos = self.current_start_sec + (ms / 1000.0)
        if self.total_len > 0.0 and pos > self.total_len:
            pos = self.total_len
        return float(pos)

# ==========================================================
#                 N O R M A L I Z A T I O N
# ==========================================================
NUMBER_WORDS = {
    "cero": "0", "uno": "1", "una": "1", "dos": "2", "tres": "3", "cuatro": "4", "cinco": "5",
    "seis": "6", "siete": "7", "ocho": "8", "nueve": "9", "diez": "10",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "zéro": "0", "un": "1", "une": "1", "deux": "2", "trois": "3", "quatre": "4", "cinq": "5",
    "six": "6", "sept": "7", "huit": "8", "neuf": "9", "dix": "10",
    "ноль": "0", "один": "1", "одна": "1", "два": "2", "две": "2", "три": "3",
    "четыре": "4", "пять": "5", "шесть": "6", "семь": "7", "восемь": "8",
    "девять": "9", "десять": "10",
    "零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"
}

def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = strip_accents(s)
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("p.m.", "pm").replace("a.m.", "am")
    s = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", s, flags=re.UNICODE)
    tokens = [NUMBER_WORDS.get(tok, tok) for tok in s.split()]
    s = " ".join(tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def answer_matches(user_answer: str, accepted_answers: List[str]) -> bool:
    ua = normalize_text(user_answer)
    if not ua:
        return False
    for ans in accepted_answers:
        a = normalize_text(ans)
        if ua == a:
            return True
        if len(a) >= 5 and (a in ua or ua in a):
            return True
    return False

# ==========================================================
#                 S C E N A R I O   S T O R E
# ==========================================================
def default_spanish_seed() -> Dict[str, List[Dict[str, Any]]]:
    passage_es = [
        "La doctora Ruiz le pidió a Marta suspender el antihistamínico durante 48 horas antes de las pruebas, porque puede falsear los resultados.",
        "Aun así, le dijo que llevara su inhalador por si notaba opresión en el pecho.",
        "La cita quedó para el jueves a las 16:20, y le enviaron por correo unas instrucciones con una lista de alimentos que debía evitar desde el martes por la noche."
    ]
    passage_en = [
        "Dr. Ruiz asked Marta to stop the antihistamine 48 hours before the tests because it can skew the results.",
        "Even so, she told her to bring her inhaler in case she felt chest tightness.",
        "The appointment was set for Thursday at 16:20, and she was emailed instructions with a list of foods to avoid starting Tuesday night."
    ]
    passage = " ".join(passage_es)
    return {
        "Allergy check-up (DEFAULT)": [
            {
                "sentence": passage,
                "sentence_en_lines": passage_en,
                "question": "¿Por qué debía dejar de tomar el antihistamínico antes de las pruebas?",
                "question_en": "Why did she need to stop taking the antihistamine before the tests?",
                "answers": ["porque puede falsear los resultados", "para no alterar los resultados", "porque altera los resultados"],
                "choices": [
                    "Porque puede falsear los resultados",
                    "Para ahorrar medicamento",
                    "Porque no tenía receta válida",
                    "Para dormir mejor la noche anterior"
                ],
                "choices_en": [
                    "Because it can skew the results",
                    "To save medication",
                    "Because she didn't have a valid prescription",
                    "To sleep better the night before"
                ],
                "correct_index": 0,
                "explain": "Los antihistamínicos pueden interferir con las pruebas y producir falsos negativos.",
                "explain_en": "Antihistamines can interfere with the tests and produce false negatives."
            },
            {
                "sentence": passage,
                "sentence_en_lines": passage_en,
                "question": "¿Qué debía llevar Marta a la cita como precaución?",
                "question_en": "What was Marta supposed to bring to the appointment as a precaution?",
                "answers": ["su inhalador", "el inhalador", "inhalador"],
                "choices": ["Su inhalador", "Un antibiótico", "Una crema hidratante", "Un informe laboral"],
                "choices_en": ["Her inhaler", "An antibiotic", "A moisturizer", "A work report"],
                "correct_index": 0,
                "explain": "Se indicó llevar el inhalador por si aparecía opresión torácica.",
                "explain_en": "She was told to bring the inhaler in case chest tightness occurred."
            },
            {
                "sentence": passage,
                "sentence_en_lines": passage_en,
                "question": "¿Desde cuándo debía evitar ciertos alimentos Marta?",
                "question_en": "Since when was Marta supposed to avoid certain foods?",
                "answers": ["desde el martes por la noche", "a partir del martes por la noche", "desde martes por la noche"],
                "choices": ["Desde el martes por la noche", "Desde el lunes por la mañana", "Solo el mismo jueves", "Durante dos semanas"],
                "choices_en": ["Starting Tuesday night", "Since Monday morning", "Only on Thursday", "For two weeks"],
                "correct_index": 0,
                "explain": "El correo especificaba esa fecha.",
                "explain_en": "The email specified that date."
            },
            {
                "sentence": passage,
                "sentence_en_lines": passage_en,
                "question": "¿Cuánto tiempo antes debía suspender el antihistamínico?",
                "question_en": "How long in advance did she need to stop the antihistamine?",
                "answers": ["48 horas", "dos dias", "2 dias", "2 días"],
                "choices": ["24 horas", "48 horas", "72 horas", "Una semana"],
                "choices_en": ["24 hours", "48 hours", "72 hours", "One week"],
                "correct_index": 1,
                "explain": "El texto menciona 48 horas.",
                "explain_en": "The passage mentions 48 hours."
            },
            {
                "sentence": passage,
                "sentence_en_lines": passage_en,
                "question": "¿A qué hora se fijó la cita del jueves?",
                "question_en": "At what time was Thursday’s appointment scheduled?",
                "answers": ["16:20", "a las 16 20", "16 20"],
                "choices": ["15:45", "16:20", "17:30", "18:00"],
                "choices_en": ["3:45 PM", "4:20 PM", "5:30 PM", "6:00 PM"],
                "correct_index": 1,
                "explain": "La cita se programó para las 16:20.",
                "explain_en": "The appointment was scheduled for 16:20."
            },
        ]
    }


def _with_language(items: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    return [{**item, "target_language": normalize_language_label(language)} for item in items]


def default_multilingual_seed() -> Dict[str, List[Dict[str, Any]]]:
    french_passage_lines = [
        "Le docteur Martin a demandé à Claire d'arrêter l'antihistaminique 48 heures avant les tests, car il peut fausser les résultats.",
        "Elle lui a tout de même dit d'apporter son inhalateur au cas où elle ressentirait une oppression dans la poitrine.",
        "Le rendez-vous a été fixé au jeudi à 16 h 20, et la clinique lui a envoyé un courriel avec une liste d'aliments à éviter dès mardi soir."
    ]
    french_passage_en = [
        "Dr. Martin asked Claire to stop the antihistamine 48 hours before the tests because it can skew the results.",
        "Even so, she told her to bring her inhaler in case she felt chest tightness.",
        "The appointment was set for Thursday at 16:20, and the clinic emailed her a list of foods to avoid starting Tuesday night."
    ]
    french_passage = " ".join(french_passage_lines)
    french_items = _with_language([
        {
            "sentence": french_passage,
            "sentence_en_lines": french_passage_en,
            "question": "Pourquoi Claire devait-elle arrêter l'antihistaminique avant les tests ?",
            "question_en": "Why did Claire need to stop the antihistamine before the tests?",
            "answers": ["parce qu'il peut fausser les résultats", "pour ne pas fausser les résultats", "parce qu'il modifie les résultats"],
            "choices": [
                "Parce qu'il peut fausser les résultats",
                "Pour économiser le médicament",
                "Parce que son ordonnance n'était plus valable",
                "Pour mieux dormir la veille"
            ],
            "choices_en": [
                "Because it can skew the results",
                "To save medication",
                "Because her prescription was no longer valid",
                "To sleep better the night before"
            ],
            "correct_index": 0,
            "explain": "L'antihistaminique peut interférer avec les tests et fausser leur interprétation.",
            "explain_en": "The antihistamine can interfere with the tests and distort their interpretation."
        },
        {
            "sentence": french_passage,
            "sentence_en_lines": french_passage_en,
            "question": "Que devait apporter Claire par précaution ?",
            "question_en": "What was Claire supposed to bring as a precaution?",
            "answers": ["son inhalateur", "l'inhalateur", "un inhalateur"],
            "choices": ["Son inhalateur", "Un antibiotique", "Une crème hydratante", "Un rapport de travail"],
            "choices_en": ["Her inhaler", "An antibiotic", "A moisturizer", "A work report"],
            "correct_index": 0,
            "explain": "Le médecin lui a demandé d'apporter son inhalateur en cas d'oppression dans la poitrine.",
            "explain_en": "The doctor asked her to bring her inhaler in case she felt chest tightness."
        },
        {
            "sentence": french_passage,
            "sentence_en_lines": french_passage_en,
            "question": "À partir de quand devait-elle éviter certains aliments ?",
            "question_en": "Starting when was she supposed to avoid certain foods?",
            "answers": ["dès mardi soir", "à partir de mardi soir", "depuis mardi soir"],
            "choices": ["Dès mardi soir", "Dès lundi matin", "Seulement le jeudi même", "Pendant deux semaines"],
            "choices_en": ["Starting Tuesday night", "Starting Monday morning", "Only on Thursday itself", "For two weeks"],
            "correct_index": 0,
            "explain": "Le courriel précisait que la restriction commençait mardi soir.",
            "explain_en": "The email specified that the restriction started Tuesday night."
        },
        {
            "sentence": french_passage,
            "sentence_en_lines": french_passage_en,
            "question": "Combien de temps à l'avance devait-elle arrêter l'antihistaminique ?",
            "question_en": "How long in advance did she need to stop the antihistamine?",
            "answers": ["48 heures", "quarante huit heures", "deux jours"],
            "choices": ["24 heures", "48 heures", "72 heures", "Une semaine"],
            "choices_en": ["24 hours", "48 hours", "72 hours", "One week"],
            "correct_index": 1,
            "explain": "Le texte indique clairement une durée de 48 heures avant les tests.",
            "explain_en": "The passage clearly states a duration of 48 hours before the tests."
        },
        {
            "sentence": french_passage,
            "sentence_en_lines": french_passage_en,
            "question": "À quelle heure le rendez-vous du jeudi était-il fixé ?",
            "question_en": "At what time was Thursday's appointment scheduled?",
            "answers": ["16 h 20", "16:20", "seize heures vingt"],
            "choices": ["15 h 45", "16 h 20", "17 h 30", "18 h 00"],
            "choices_en": ["3:45 PM", "4:20 PM", "5:30 PM", "6:00 PM"],
            "correct_index": 1,
            "explain": "Le rendez-vous était fixé au jeudi à 16 h 20.",
            "explain_en": "The appointment was scheduled for Thursday at 16:20."
        },
    ], "French")

    russian_passage_lines = [
        "Врач Иванова попросила Анну прекратить принимать антигистаминный препарат за 48 часов до тестов, потому что он может исказить результаты.",
        "При этом она сказала взять с собой ингалятор на случай, если появится стеснение в груди.",
        "Прием назначили на четверг в 16:20, а клиника отправила ей по электронной почте список продуктов, которых нужно избегать с вечера вторника."
    ]
    russian_passage_en = [
        "Dr. Ivanova asked Anna to stop taking the antihistamine 48 hours before the tests because it can skew the results.",
        "At the same time, she told her to bring an inhaler in case chest tightness appeared.",
        "The appointment was scheduled for Thursday at 16:20, and the clinic emailed her a list of foods to avoid starting Tuesday evening."
    ]
    russian_passage = " ".join(russian_passage_lines)
    russian_items = _with_language([
        {
            "sentence": russian_passage,
            "sentence_en_lines": russian_passage_en,
            "question": "Почему Анне нужно было прекратить принимать антигистаминный препарат перед тестами?",
            "question_en": "Why did Anna need to stop taking the antihistamine before the tests?",
            "answers": ["потому что он может исказить результаты", "чтобы не исказить результаты", "потому что препарат влияет на результаты"],
            "choices": [
                "Потому что он может исказить результаты",
                "Чтобы сэкономить лекарство",
                "Потому что рецепт закончился",
                "Чтобы лучше спать накануне"
            ],
            "choices_en": [
                "Because it can skew the results",
                "To save medication",
                "Because the prescription had expired",
                "To sleep better the night before"
            ],
            "correct_index": 0,
            "explain": "В тексте сказано, что препарат может исказить результаты тестов.",
            "explain_en": "The passage says that the medication can skew the test results."
        },
        {
            "sentence": russian_passage,
            "sentence_en_lines": russian_passage_en,
            "question": "Что Анна должна была взять с собой на прием как меру предосторожности?",
            "question_en": "What was Anna supposed to bring to the appointment as a precaution?",
            "answers": ["ингалятор", "свой ингалятор", "взять ингалятор"],
            "choices": ["Ингалятор", "Антибиотик", "Увлажняющий крем", "Рабочий отчет"],
            "choices_en": ["An inhaler", "An antibiotic", "A moisturizer", "A work report"],
            "correct_index": 0,
            "explain": "Врач сказала взять ингалятор на случай стеснения в груди.",
            "explain_en": "The doctor told her to bring an inhaler in case of chest tightness."
        },
        {
            "sentence": russian_passage,
            "sentence_en_lines": russian_passage_en,
            "question": "С какого времени Анне нужно было избегать некоторых продуктов?",
            "question_en": "Starting when did Anna need to avoid certain foods?",
            "answers": ["с вечера вторника", "начиная с вечера вторника", "со вторника вечером"],
            "choices": ["С вечера вторника", "С утра понедельника", "Только в сам четверг", "В течение двух недель"],
            "choices_en": ["Starting Tuesday evening", "Starting Monday morning", "Only on Thursday itself", "For two weeks"],
            "correct_index": 0,
            "explain": "Список продуктов нужно было соблюдать начиная с вечера вторника.",
            "explain_en": "The food list had to be followed starting Tuesday evening."
        },
        {
            "sentence": russian_passage,
            "sentence_en_lines": russian_passage_en,
            "question": "За сколько времени до тестов нужно было прекратить прием препарата?",
            "question_en": "How long before the tests did she need to stop the medication?",
            "answers": ["за 48 часов", "48 часов", "за двое суток"],
            "choices": ["За 24 часа", "За 48 часов", "За 72 часа", "За неделю"],
            "choices_en": ["24 hours before", "48 hours before", "72 hours before", "A week before"],
            "correct_index": 1,
            "explain": "В первом предложении указано: за 48 часов до тестов.",
            "explain_en": "The first sentence states: 48 hours before the tests."
        },
        {
            "sentence": russian_passage,
            "sentence_en_lines": russian_passage_en,
            "question": "На какое время был назначен прием в четверг?",
            "question_en": "At what time was the Thursday appointment scheduled?",
            "answers": ["16:20", "на 16:20", "в 16 20"],
            "choices": ["15:45", "16:20", "17:30", "18:00"],
            "choices_en": ["3:45 PM", "4:20 PM", "5:30 PM", "6:00 PM"],
            "correct_index": 1,
            "explain": "Прием назначили на четверг в 16:20.",
            "explain_en": "The appointment was scheduled for Thursday at 16:20."
        },
    ], "Russian")

    chinese_passage_lines = [
        "李医生让马悦在检查前48小时停用抗组胺药，因为它可能影响结果。",
        "尽管如此，她还是提醒马悦带上吸入器，以防胸口发紧。",
        "预约定在星期四16:20，诊所还通过电子邮件发来了一份从星期二晚上开始需要避免的食物清单。"
    ]
    chinese_passage_en = [
        "Dr. Li told Ma Yue to stop using the antihistamine 48 hours before the tests because it could affect the results.",
        "Even so, she reminded Ma Yue to bring an inhaler in case her chest felt tight.",
        "The appointment was set for Thursday at 16:20, and the clinic also emailed a list of foods to avoid starting Tuesday evening."
    ]
    chinese_passage = "".join(chinese_passage_lines)
    chinese_items = _with_language([
        {
            "sentence": chinese_passage,
            "sentence_en_lines": chinese_passage_en,
            "question": "为什么马悦需要在检查前停用抗组胺药？",
            "question_en": "Why did Ma Yue need to stop using the antihistamine before the tests?",
            "answers": ["因为它可能影响结果", "避免影响结果", "因为药会影响检查结果"],
            "choices": [
                "因为它可能影响结果",
                "为了节省药物",
                "因为处方已经过期",
                "为了前一晚睡得更好"
            ],
            "choices_en": [
                "Because it could affect the results",
                "To save medication",
                "Because the prescription had expired",
                "To sleep better the night before"
            ],
            "correct_index": 0,
            "explain": "文章说明抗组胺药可能影响检查结果，所以需要提前停用。",
            "explain_en": "The passage states that the antihistamine could affect the test results, so it needed to be stopped ahead of time."
        },
        {
            "sentence": chinese_passage,
            "sentence_en_lines": chinese_passage_en,
            "question": "马悦需要带什么作为预防？",
            "question_en": "What did Ma Yue need to bring as a precaution?",
            "answers": ["吸入器", "带上吸入器", "她的吸入器"],
            "choices": ["吸入器", "抗生素", "保湿霜", "工作报告"],
            "choices_en": ["An inhaler", "An antibiotic", "A moisturizer", "A work report"],
            "correct_index": 0,
            "explain": "李医生提醒她带上吸入器，以防胸口发紧。",
            "explain_en": "Dr. Li reminded her to bring an inhaler in case her chest felt tight."
        },
        {
            "sentence": chinese_passage,
            "sentence_en_lines": chinese_passage_en,
            "question": "她从什么时候开始需要避免某些食物？",
            "question_en": "Starting when did she need to avoid certain foods?",
            "answers": ["从星期二晚上开始", "星期二晚上开始", "从周二晚上开始"],
            "choices": ["从星期二晚上开始", "从星期一早上开始", "只在星期四当天", "连续两周"],
            "choices_en": ["Starting Tuesday evening", "Starting Monday morning", "Only on Thursday itself", "For two weeks"],
            "correct_index": 0,
            "explain": "邮件中的清单说明从星期二晚上开始需要避免这些食物。",
            "explain_en": "The emailed list said those foods had to be avoided starting Tuesday evening."
        },
        {
            "sentence": chinese_passage,
            "sentence_en_lines": chinese_passage_en,
            "question": "她需要提前多久停用抗组胺药？",
            "question_en": "How long in advance did she need to stop the antihistamine?",
            "answers": ["48小时", "提前48小时", "两天"],
            "choices": ["24小时", "48小时", "72小时", "一周"],
            "choices_en": ["24 hours", "48 hours", "72 hours", "One week"],
            "correct_index": 1,
            "explain": "第一句明确说是在检查前48小时停用。",
            "explain_en": "The first sentence explicitly says to stop it 48 hours before the tests."
        },
        {
            "sentence": chinese_passage,
            "sentence_en_lines": chinese_passage_en,
            "question": "星期四的预约定在几点？",
            "question_en": "At what time was Thursday's appointment scheduled?",
            "answers": ["16:20", "下午4点20分", "十六点二十分"],
            "choices": ["15:45", "16:20", "17:30", "18:00"],
            "choices_en": ["3:45 PM", "4:20 PM", "5:30 PM", "6:00 PM"],
            "correct_index": 1,
            "explain": "文中写明预约定在星期四16:20。",
            "explain_en": "The passage states that the appointment was set for Thursday at 16:20."
        },
    ], "Mandarin Chinese")

    return {
        "Allergy check-up (FRENCH DEFAULT)": french_items,
        "Allergy check-up (RUSSIAN DEFAULT)": russian_items,
        "Allergy check-up (MANDARIN DEFAULT)": chinese_items,
    }


def default_seed_scenarios() -> Dict[str, List[Dict[str, Any]]]:
    seeds = {
        title: _with_language(items, "Spanish")
        for title, items in default_spanish_seed().items()
    }
    seeds.update(default_multilingual_seed())
    return seeds


class ScenarioStore:
    def __init__(self):
        self._scenarios: Dict[str, List[Dict[str, Any]]] = default_seed_scenarios()

    def all_titles(self) -> List[str]:
        return list(self._scenarios.keys())

    def get_items(self, title: str) -> List[Dict[str, Any]]:
        items = self._scenarios[title]
        if isinstance(items, dict):
            # Safety guard: if a compact block slipped in somehow, convert on the fly.
            logger.warning("Scenario '%s' stored as dict; converting to trainer items on the fly.", title)
            _, trainer_items = _convert_compact_block_to_trainer(title, items)
            self._scenarios[title] = trainer_items
            return trainer_items
        return items

    def merge_trainer_dict(self, trainer_dict: Dict[str, List[Dict[str, Any]]]):
        # Expecting already-normalized mapping title -> list[items]
        added = 0
        for k, v in _normalize_trainer_mapping(trainer_dict).items():
            if not isinstance(v, list):
                logger.warning("Skipping scenario '%s' because value is not a list after normalization.", k)
                continue
            if len(v) == 0:
                logger.warning("Skipping scenario '%s' because it has zero items.", k)
                continue
            self._scenarios[k] = v
            added += 1
        logger.info("Merged %d scenario(s) into store. Total now: %d", added, len(self._scenarios))

    def add_scenario_pack(self, sp: ScenarioPack):
        self.merge_trainer_dict(to_trainer_payload(sp))

# ==========================================================
#                U I   C O M P O N E N T S
# ==========================================================
def mmss(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    return f"{m:02d}:{s:02d}"

_SPEED_CHOICES = [round(x * 0.25, 2) for x in range(1, 9)]  # 0.25..2.0

class AudioPlayer(ttk.Frame):
    """
    Single audio player (Play/Pause/Stop, -5/+5, seek bar, time).
    Source toggle: Passage / Question.
    Speed: 0.25x .. 2.0x.
    """
    def __init__(self, master, synthesizer: Synthesizer, controller: AudioController, get_text_callable):
        super().__init__(master, padding=(8,8,8,8))
        self.synth = synthesizer
        self.ctrl = controller
        self.get_text = get_text_callable  # fn(source_str) -> target-language text
        self.language = DEFAULT_LANGUAGE
        default_profile = get_language_profile(DEFAULT_LANGUAGE)
        self.voice_choice = tk.StringVar(value=default_profile.default_voice_label)
        self.source = tk.StringVar(value="Passage")
        self.speed_var = tk.StringVar(value="1.0")
        self.total_len = 0.0
        self._updater_job = None
        self._user_dragging = False

        # Top row: Source + Voice + Speed
        top = ttk.Frame(self)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Audio source:").pack(side=tk.LEFT)
        ttk.OptionMenu(top, self.source, "Passage", "Passage", "Question").pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="Voice:").pack(side=tk.LEFT, padx=(16,0))
        self.voice_combo = ttk.Combobox(
            top,
            values=list(default_profile.voices.keys()),
            textvariable=self.voice_choice,
            width=24,
            state="readonly",
        )
        self.voice_combo.pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="Speed:").pack(side=tk.LEFT, padx=(16,0))
        sp_vals = [f"{v:.2f}" for v in _SPEED_CHOICES]
        self.speed_combo = ttk.Combobox(top, values=sp_vals, textvariable=self.speed_var, width=5, state="readonly")
        self.speed_combo.pack(side=tk.LEFT, padx=6)
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Controls row
        ctrls = ttk.Frame(self)
        ctrls.pack(fill=tk.X, pady=(8,4))

        self.play_btn = ttk.Button(ctrls, text="▶ Play", command=self.on_play)
        self.play_btn.pack(side=tk.LEFT)
        self.pause_btn = ttk.Button(ctrls, text="⏸ Pause/Resume", command=self.on_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=(6,0))
        self.stop_btn = ttk.Button(ctrls, text="■ Stop", command=self.on_stop)
        self.stop_btn.pack(side=tk.LEFT, padx=(6,0))
        self.rew_btn = ttk.Button(ctrls, text="⏪ -5s", command=lambda: self.nudge(-5))
        self.rew_btn.pack(side=tk.LEFT, padx=(12,0))
        self.ffw_btn = ttk.Button(ctrls, text="+5s ⏩", command=lambda: self.nudge(5))
        self.ffw_btn.pack(side=tk.LEFT, padx=(6,0))

        # Seek bar row
        seek = ttk.Frame(self)
        seek.pack(fill=tk.X)
        self.cur_lbl = ttk.Label(seek, text="00:00")
        self.cur_lbl.pack(side=tk.LEFT)
        self.scale = ttk.Scale(seek, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=self._on_scale_move)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.tot_lbl = ttk.Label(seek, text="00:00")
        self.tot_lbl.pack(side=tk.LEFT)

        # Bind drag start/end
        self.scale.bind("<ButtonPress-1>", lambda e: self._set_dragging(True))
        self.scale.bind("<ButtonRelease-1>", self._on_scale_release)

        # Note label for fallbacks
        self.note_lbl = ttk.Label(self, text="", foreground="#a66")
        self.note_lbl.pack(anchor="w", pady=(4,0))

    def set_language(self, language: str):
        profile = get_language_profile(language)
        self.language = profile.display
        labels = list(profile.voices.keys())
        self.voice_combo.configure(values=labels, width=max(18, min(30, max(len(v) for v in labels) + 2)))
        if self.voice_choice.get() not in labels:
            self.voice_choice.set(profile.default_voice_label)

    def current_voice_name(self) -> str:
        profile = get_language_profile(self.language)
        return profile.voices.get(self.voice_choice.get(), profile.voices[profile.default_voice_label])

    def _set_dragging(self, val: bool):
        self._user_dragging = val

    def _on_scale_move(self, _val):
        if not self._user_dragging:
            return
        if self.total_len > 0:
            pos = float(self.scale.get()) * self.total_len
            self.cur_lbl.config(text=mmss(pos))

    def _on_scale_release(self, _event):
        self._set_dragging(False)
        if self.total_len <= 0:
            return
        target = float(self.scale.get()) * self.total_len
        was_playing = self.ctrl.playing and not self.ctrl.paused
        self.ctrl.play(start_sec=target) if was_playing else self.ctrl.play(start_sec=target) or self.ctrl.pause()

    def nudge(self, delta: int):
        if self.total_len <= 0:
            return
        cur = self.ctrl.get_pos_sec() if self.ctrl.playing else (float(self.scale.get()) * self.total_len)
        target = max(0.0, min(self.total_len-0.05, cur + delta))
        self.ctrl.play(start_sec=target)

    def _current_speed(self) -> float:
        try:
            return float(self.speed_var.get())
        except Exception:
            return 1.0

    def _prepare_audio(self, start_relative: float = 0.0):
        src = self.source.get()
        text = self.get_text(src)
        if not text:
            messagebox.showinfo("Nothing to play", f"No text found for {src}.")
            return None
        voice_name = self.current_voice_name()
        speed = self._current_speed()
        try:
            path, length, note = self.synth.get_audio(text, voice_name, speed)
        except Exception as e:
            messagebox.showerror("TTS error", str(e))
            return None
        self.ctrl.load(path, length)
        self.total_len = length
        self.scale.configure(to=1.0)
        self.tot_lbl.config(text=mmss(length))
        self.note_lbl.config(text=note or "")
        start_sec = min(max(0.0, start_relative) * length, max(0.0, length - 0.05))
        return start_sec

    def _on_speed_change(self, _e=None):
        rel = 0.0
        if self.total_len > 0:
            rel = self.ctrl.get_pos_sec() / self.total_len if self.ctrl.playing else float(self.scale.get())
        start_sec = self._prepare_audio(start_relative=rel)
        if start_sec is None:
            return
        self.ctrl.play(start_sec=start_sec)
        self._start_updater()

    def on_play(self):
        rel = float(self.scale.get())
        start_sec = self._prepare_audio(start_relative=rel)
        if start_sec is None:
            return
        self.ctrl.play(start_sec=start_sec)
        self._start_updater()

    def on_pause(self):
        if self.ctrl.playing and not self.ctrl.paused:
            self.ctrl.pause()
        elif self.ctrl.playing and self.ctrl.paused:
            self.ctrl.resume()
        self._start_updater()

    def on_stop(self):
        self.ctrl.stop()
        self._start_updater(reset_to_zero=True)

    def _start_updater(self, reset_to_zero: bool=False):
        if self._updater_job:
            self.after_cancel(self._updater_job)
            self._updater_job = None

        def tick():
            if reset_to_zero or not self.ctrl.playing:
                self.cur_lbl.config(text="00:00")
                if not self._user_dragging:
                    self.scale.set(0.0)
            else:
                pos = min(self.ctrl.get_pos_sec(), self.total_len if self.total_len>0 else 0.0)
                if not self._user_dragging and self.total_len>0:
                    self.scale.set(pos / self.total_len)
                self.cur_lbl.config(text=mmss(pos))
            self._updater_job = self.after(200, tick)

        tick()

# ---------------------- Option Card -----------------------
class OptionCard(ttk.Frame):
    def __init__(self, master, index: int, text: str, variable: tk.IntVar, on_select, font: tkfont.Font):
        super().__init__(master, padding=(8,8,8,8), style="Card.TFrame")
        self.index = index
        self.text = text
        self.variable = variable
        self.on_select = on_select

        self.bind("<Button-1>", self._select)
        self.badge = ttk.Label(self, text=f"{'ABCD'[index]}", style="Badge.TLabel", width=3, anchor="center")
        self.badge.pack(side=tk.LEFT)
        self.badge.bind("<Button-1>", self._select)

        self.text_label = ttk.Label(self, text=text, wraplength=820, justify="left")
        self.text_label.pack(side=tk.LEFT, padx=(10,0), fill=tk.X, expand=True)
        self.text_label.bind("<Button-1>", self._select)
        self.bind("<Configure>", self._on_resize)
        self.set_font(font)

    def _on_resize(self, event):
        self.text_label.configure(wraplength=max(280, event.width - 86))

    def _select(self, _event=None):
        self.variable.set(self.index)
        self.on_select(self.index)

    def set_font(self, font: tkfont.Font):
        try:
            self.text_label.configure(font=font)
        except Exception:
            pass

    def set_state(self, state: str):
        style = {
            "normal": "Card.TFrame",
            "selected": "CardSelected.TFrame",
            "correct": "CardCorrect.TFrame",
            "wrong": "CardWrong.TFrame",
            "disabled": "CardDisabled.TFrame"
        }.get(state, "Card.TFrame")
        self.configure(style=style)

# ==========================================================
#                M A I N   A P P
# ==========================================================
@dataclass
class UIState:
    model: str = "gpt-5-nano"

# --- 1) Add this class somewhere above `class App` (e.g., right after OptionCard) ---
class ScrollableFrame(ttk.Frame):
    """A vertical scroll container that holds widgets in `self.body`."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#f5f7fb")
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.body = ttk.Frame(self.canvas)
        self._body_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        # Keep scrollregion and width in sync
        self.body.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel support (Win/macOS/Linux)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)        # Win & macOS
        self.canvas.bind_all("<Button-4>", self._on_wheel_linux_up)      # Linux
        self.canvas.bind_all("<Button-5>", self._on_wheel_linux_down)    # Linux

    def _on_frame_configure(self, _e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        # Make inner frame match the canvas width
        self.canvas.itemconfigure(self._body_id, width=e.width)

    def _on_mousewheel(self, e):
        # e.delta is ±120 on Windows, small multiples on macOS
        step = int(-e.delta / 120) if e.delta else 0
        if step != 0:
            self.canvas.yview_scroll(step, "units")

    def _on_wheel_linux_up(self, _e):
        self.canvas.yview_scroll(-1, "units")

    def _on_wheel_linux_down(self, _e):
        self.canvas.yview_scroll(1, "units")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Language Scenario Tutor")
        self.geometry("1220x900")
        self.minsize(1040, 760)
        self.configure(background="#f5f7fb")

        # Styles
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
            self.tk.call("tk", "scaling", 1.2)
        except Exception:
            pass
        style.configure(".", font=("Helvetica", 12), background="#f5f7fb", foreground="#1f2937")
        style.configure("TNotebook", background="#f5f7fb", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8), font=("Helvetica", 12, "bold"))
        style.configure("TLabelframe", background="#f5f7fb", bordercolor="#cbd5e1")
        style.configure("TLabelframe.Label", background="#f5f7fb", foreground="#334155", font=("Helvetica", 12, "bold"))
        style.configure("TButton", padding=(12, 7))
        style.configure("Primary.TButton", padding=(14, 8), font=("Helvetica", 12, "bold"))
        style.configure("Card.TFrame", relief="solid", borderwidth=1, background="#ffffff")
        style.configure("CardSelected.TFrame", relief="solid", borderwidth=2, background="#e0f2fe")
        style.configure("CardCorrect.TFrame", relief="solid", borderwidth=2, background="#dcfce7")
        style.configure("CardWrong.TFrame", relief="solid", borderwidth=2, background="#fee2e2")
        style.configure("CardDisabled.TFrame", relief="solid", borderwidth=1, background="#f8fafc")
        style.configure("Badge.TLabel", background="#e2e8f0", foreground="#0f172a", padding=5, font=("Helvetica", 11, "bold"))
        style.configure("Subtle.TLabel", background="#f5f7fb", foreground="#475569")
        style.configure("Status.TLabel", background="#f5f7fb", foreground="#475569")

        # Fonts + size control
        self.text_size_var = tk.IntVar(value=16)
        initial_font = self._choose_font_family(DEFAULT_LANGUAGE)
        self.font_body = tkfont.Font(family=initial_font, size=self.text_size_var.get())
        self.font_body_bold = tkfont.Font(family=initial_font, size=self.text_size_var.get(), weight="bold")
        self.font_body_italic = tkfont.Font(family=initial_font, size=self.text_size_var.get(), slant="italic")

        self.store = ScenarioStore()
        self.synth = Synthesizer(lang=DEFAULT_LANGUAGE)
        self.audio = AudioController()

        # Trainer state
        self.current_scenario = tk.StringVar(value=self.store.all_titles()[0])
        self.gen_language = tk.StringVar(value=DEFAULT_LANGUAGE)
        self.hide_sentence = tk.BooleanVar(value=False)
        self.randomized = tk.BooleanVar(value=False)
        self.show_translation = tk.BooleanVar(value=False)
        self.qa_lang = tk.StringVar(value=DEFAULT_LANGUAGE)  # target language or English

        # Per-scenario state
        self.order: Dict[str, List[int]] = {}
        self.index: Dict[str, int] = {}
        self.correct_count: Dict[str, int] = {}
        self.answered: Dict[str, Dict[int, Dict[str, Any]]] = {}  # scen -> {global_idx: {"selected": int, "correct": bool}}
        for scen in self.store.all_titles():
            total = len(self.store.get_items(scen))
            self.order[scen] = list(range(total))
            self.index[scen] = 0
            self.correct_count[scen] = 0
            self.answered[scen] = {}

        self._build_ui()
        self._refresh_all()
        self._bind_shortcuts()

    def _choose_font_family(self, language: str) -> str:
        profile = get_language_profile(language)
        try:
            installed = set(tkfont.families(self))
        except Exception:
            installed = set()
        for family in profile.preferred_fonts:
            if not installed or family in installed:
                return family
        return "TkDefaultFont"

    def _apply_language_fonts(self, language: str):
        family = self._choose_font_family(language)
        self.font_body.configure(family=family)
        self.font_body_bold.configure(family=family)
        self.font_body_italic.configure(family=family)
        if hasattr(self, "passage_text"):
            self.passage_text.configure(font=self.font_body)
        if hasattr(self, "translation_text"):
            self.translation_text.configure(font=self.font_body)
            self.translation_text.tag_configure("target", font=self.font_body_bold)
            self.translation_text.tag_configure("en", font=self.font_body_italic, foreground="#334155")
        if hasattr(self, "question_label"):
            self.question_label.configure(font=self.font_body_bold)
        for card in getattr(self, "option_cards", []):
            card.set_font(self.font_body)

    def _item_language(self, item: Optional[Dict[str, Any]] = None) -> TargetLanguage:
        try:
            item = item or self._current_item()
            return normalize_language_label(item.get("target_language") or detect_language_from_text(item.get("sentence", "")))
        except Exception:
            return DEFAULT_LANGUAGE

    # ---------------- UI Layout ----------------
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Generator
        self.tab_gen = ttk.Frame(nb)
        nb.add(self.tab_gen, text="Generate (LLM)")

        # Tab 2: Trainer
        self.tab_train = ttk.Frame(nb)
        nb.add(self.tab_train, text="Practice")

        self._build_generator_tab(self.tab_gen)
        self._build_trainer_tab(self.tab_train)

    # -------- Generator Tab --------
    def _build_generator_tab(self, parent):
        root = ttk.Frame(parent, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root, text="Scenario topic/prompt").pack(anchor="w")
        self.topic_txt = scrolledtext.ScrolledText(root, height=4, wrap=tk.WORD)
        self.topic_txt.configure(font=("Helvetica", 14), background="#ffffff", foreground="#111827", insertbackground="#111827")
        self.topic_txt.pack(fill=tk.X, pady=(4, 10))

        row = ttk.Frame(root)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text="Target language").pack(side=tk.LEFT)
        self.gen_language_combo = ttk.Combobox(
            row,
            values=LANGUAGE_LABELS,
            textvariable=self.gen_language,
            state="readonly",
            width=18,
        )
        self.gen_language_combo.pack(side=tk.LEFT, padx=(6, 18))

        ttk.Label(row, text="Difficulty").pack(side=tk.LEFT)
        self.diff_var = tk.StringVar(value="medium")
        ttk.OptionMenu(row, self.diff_var, "medium", "easy", "medium", "hard").pack(side=tk.LEFT, padx=6)

        ttk.Label(row, text="Model").pack(side=tk.LEFT, padx=(18, 0))
        self.model_var = tk.StringVar(value="gpt-5-nano")
        ttk.Entry(row, textvariable=self.model_var, width=18).pack(side=tk.LEFT, padx=6)

        btns = ttk.Frame(root)
        btns.pack(fill=tk.X, pady=(6, 0))
        self.gen_btn = ttk.Button(btns, text="Generate", style="Primary.TButton", command=self._on_generate)
        self.gen_btn.pack(side=tk.LEFT)
        self.send_btn = ttk.Button(btns, text="Send to Trainer", state=tk.DISABLED, command=self._on_send_to_trainer)
        self.send_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.save_btn = ttk.Button(btns, text="Save JSON…", state=tk.DISABLED, command=self._on_save_gen)
        self.save_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="Import JSON to Trainer…", command=self._on_import_to_trainer).pack(side=tk.RIGHT)

        ttk.Label(root, text="Preview").pack(anchor="w", pady=(12, 4))
        self.out_txt = scrolledtext.ScrolledText(root, height=24, wrap=tk.WORD)
        self.out_txt.configure(font=("Helvetica", 13), background="#ffffff", foreground="#111827")
        self.out_txt.pack(fill=tk.BOTH, expand=True)

        self.status_gen = ttk.Label(root, text="", anchor="w", style="Status.TLabel")
        self.status_gen.pack(fill=tk.X, pady=(6, 0))

    def _busy(self, on: bool):
        try:
            self.configure(cursor="watch" if on else "")
            self.update_idletasks()
        except Exception:
            pass

    def _on_generate(self):
        topic = self.topic_txt.get("1.0", tk.END).strip()
        difficulty = self.diff_var.get().strip().lower()
        model = self.model_var.get().strip() or "gpt-5-nano"
        target_language = normalize_language_label(self.gen_language.get())
        profile = get_language_profile(target_language)
        preview_font = self._choose_font_family(target_language)
        self.out_txt.configure(font=(preview_font, 13))

        if not topic:
            messagebox.showwarning("Missing topic", "Please enter a topic for the scenario.")
            return

        try:
            self._busy(True)
            self.gen_btn.config(state=tk.DISABLED)
            self.status_gen.config(text=f"Generating {profile.display} scenario...")
            self.update_idletasks()
            self.current_pack: ScenarioPack = generate_with_retries(topic, difficulty, model, target_language, attempts=3)
        except Exception as e:
            messagebox.showerror("Generation error", str(e))
            self.status_gen.config(text="Generation failed.")
            self.gen_btn.config(state=tk.NORMAL)
            self._busy(False)
            return
        finally:
            self._busy(False)
            self.gen_btn.config(state=tk.NORMAL)

        sp = self.current_pack
        self.out_txt.delete("1.0", tk.END)
        self.out_txt.insert(tk.END, f"Title: {sp.scenario_title}\n")
        self.out_txt.insert(tk.END, f"Target language: {sp.target_language}\n")
        self.out_txt.insert(tk.END, f"Difficulty: {sp.difficulty}\n\n")
        self.out_txt.insert(tk.END, f"Passage ({sp.target_language}):\n{sp.passage}\n\n")
        if sp.passage_en_lines:
            self.out_txt.insert(tk.END, "English translation (aligned):\n")
            for i, line in enumerate(sp.passage_en_lines, 1):
                self.out_txt.insert(tk.END, f"  {i}. {line}\n")
            self.out_txt.insert(tk.END, "\n")
        for i, item in enumerate(sp.items, start=1):
            self.out_txt.insert(tk.END, f"Q{i} ({get_language_profile(sp.target_language).short_code}): {item.question}\n")
            for j, choice in enumerate(item.choices):
                label = "ABCD"[j]
                self.out_txt.insert(tk.END, f"   {label}. {choice}\n")
            self.out_txt.insert(tk.END, f"Q{i} (EN): {item.question_en}\n")
            for j, choice in enumerate(item.choices_en):
                label = "ABCD"[j]
                self.out_txt.insert(tk.END, f"   {label}. {choice}\n")
            self.out_txt.insert(tk.END, f"   Correct: {item.choices_en[item.correct_choice]}\n")
            self.out_txt.insert(tk.END, f"   Why (EN): {item.reasoning_note_en}\n\n")

        self.status_gen.config(text="Generation complete.")
        self.send_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)

    def _on_send_to_trainer(self):
        if not hasattr(self, "current_pack"):
            return
        self.store.add_scenario_pack(self.current_pack)
        title = self.current_pack.scenario_title
        if title not in self.order:
            total = len(self.store.get_items(title))
            self.order[title] = list(range(total))
            self.index[title] = 0
            self.correct_count[title] = 0
            self.answered[title] = {}
        self.current_scenario.set(title)
        messagebox.showinfo("Sent", f"'{title}' is now available in the Practice tab.")
        self.status_gen.config(text=f"Sent to trainer: {title}")
        self._refresh_all()

    def _on_save_gen(self):
        if not hasattr(self, "current_pack"):
            return
        sp = self.current_pack
        suggested = f"{slugify(sp.scenario_title)}_{sp.difficulty}.json"
        path = filedialog.asksaveasfilename(
            title="Save JSON",
            defaultextension=".json",
            initialfile=suggested,
            filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        try:
            save_bundle(sp, path)
            messagebox.showinfo("Saved", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _on_import_to_trainer(self):
        path = filedialog.askopenfilename(
            title="Import JSON",
            filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        try:
            trainer = load_trainer_from_bundle(path)  # ← robust loader: handles compact + legacy
            self.store.merge_trainer_dict(trainer)
            for title in trainer.keys():
                if title not in self.order:
                    total = len(self.store.get_items(title))
                    self.order[title] = list(range(total))
                    self.index[title] = 0
                    self.correct_count[title] = 0
                    self.answered[title] = {}
            # Select the first imported scenario
            if trainer:
                self.current_scenario.set(list(trainer.keys())[0])
            messagebox.showinfo("Imported", "Scenario(s) imported to Practice.")
            self._refresh_all()
        except Exception as e:
            messagebox.showerror("Import error", str(e))

    # -------- Trainer Tab --------
    def _build_trainer_tab(self, parent):
        top = ttk.Frame(parent, padding=(12, 12, 12, 0))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Scenario:").pack(side=tk.LEFT)
        self.scen_combo = ttk.Combobox(
            top, textvariable=self.current_scenario, values=self.store.all_titles(),
            state="readonly", width=30
        )
        self.scen_combo.pack(side=tk.LEFT, padx=(6, 12))
        self.scen_combo.bind("<<ComboboxSelected>>", lambda e: self._on_scenario_change())

        ttk.Checkbutton(top, text="Hide passage", variable=self.hide_sentence, command=self._refresh_sentence).pack(side=tk.LEFT, padx=(6, 6))
        #ttk.Checkbutton(top, text="Randomize order", variable=self.randomized, command=self._on_randomize_toggle).pack(side=tk.LEFT, padx=(6, 6))
        ttk.Checkbutton(top, text="Show EN translation", variable=self.show_translation, command=self._refresh_translation).pack(side=tk.LEFT, padx=(6, 6))

        ttk.Label(top, text="Q&A language:").pack(side=tk.LEFT, padx=(16, 0))
        self.qa_lang_combo = ttk.Combobox(top, values=[DEFAULT_LANGUAGE, "English"], textvariable=self.qa_lang, state="readonly", width=18)
        self.qa_lang_combo.pack(side=tk.LEFT, padx=(6, 6))
        self.qa_lang_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_question())

        ttk.Label(top, text="Text size:").pack(side=tk.LEFT, padx=(16, 0))
        size_vals = [str(s) for s in (12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40)]
        self.size_combo = ttk.Combobox(
            top,
            values=size_vals,
            state="readonly",
            width=4
        )
        self.size_combo.set(str(self.text_size_var.get()))
        self.size_combo.bind("<<ComboboxSelected>>", self._on_text_size_change)
        self.size_combo.pack(side=tk.LEFT, padx=(6, 6))

        #ttk.Button(top, text="Import JSON…", command=self._on_import_to_trainer).pack(side=tk.RIGHT)

        self.progress_label = ttk.Label(top, text="", style="Status.TLabel")
        self.progress_label.pack(side=tk.RIGHT, padx=(0, 10))

        # Make everything below the toolbar scrollable
        self.trainer_scroll = ScrollableFrame(parent)
        self.trainer_scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        container = self.trainer_scroll.body


        # Passage + Audio + Translation
        upper = ttk.Frame(container, padding=(12, 8, 12, 8))
        upper.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        self.language_info_label = ttk.Label(upper, text="", style="Subtle.TLabel", wraplength=1040, justify="left")
        self.language_info_label.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        self.passage_frame = ttk.LabelFrame(upper, text=f"Passage ({DEFAULT_LANGUAGE})", padding=10)
        self.passage_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.passage_text = tk.Text(self.passage_frame, height=6, wrap=tk.WORD, font=self.font_body)
        self.passage_text.configure(state=tk.DISABLED, background="#ffffff", foreground="#111827", relief=tk.FLAT, padx=10, pady=10)
        self.passage_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Translation area (hidden until toggled)
        self.translation_frame = ttk.LabelFrame(upper, text="English translation (aligned)", padding=10)
        self.translation_text = tk.Text(self.translation_frame, height=6, wrap=tk.WORD, font=self.font_body)
        self.translation_text.configure(state=tk.DISABLED, background="#ffffff", foreground="#111827", relief=tk.FLAT, padx=10, pady=10)
        self.translation_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.translation_text.tag_configure("target", font=self.font_body_bold)
        self.translation_text.tag_configure("en", font=self.font_body_italic, foreground="#334155")

        # Unified audio player below passage
        self.audio_player = AudioPlayer(
            container, synthesizer=self.synth, controller=self.audio,
            get_text_callable=self._get_audio_text
        )
        self.audio_player.pack(fill=tk.X, padx=12, pady=(0, 6))

        # Q&A area
        self.qa_frame = ttk.LabelFrame(container, text="Question + Multiple Choice", padding=12)
        self.qa_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.qa_frame.bind("<Configure>", self._on_qa_resize)

        self.question_label = ttk.Label(self.qa_frame, text="", anchor="w", justify="left", wraplength=1000)
        self.question_label.configure(font=self.font_body_bold)
        self.question_label.pack(side=tk.TOP, anchor="w")

        self.choice_var = tk.IntVar(value=-1)
        self.option_cards = []
        self.mc_container = ttk.Frame(self.qa_frame)
        self.mc_container.pack(side=tk.TOP, fill=tk.X, pady=(8, 6))

        actions = ttk.Frame(self.qa_frame)
        actions.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        self.check_btn = ttk.Button(actions, text="Check", style="Primary.TButton", command=self._on_check)
        self.check_btn.pack(side=tk.LEFT)
        self.reveal_btn = ttk.Button(actions, text="Reveal", command=self._on_reveal)
        self.reveal_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.feedback_label = ttk.Label(self.qa_frame, text="", anchor="w", justify="left", wraplength=1000)
        self.feedback_label.pack(side=tk.TOP, anchor="w", pady=(8, 0))

        # Bottom nav
        bottom = ttk.Frame(container, padding=(12, 0, 12, 12))
        bottom.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bottom, text="◀ Previous", command=self._prev_item).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Next ▶", command=self._next_item).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(bottom, text="Reset scenario", command=self._reset_scenario).pack(side=tk.RIGHT)

        self.score_label = ttk.Label(bottom, text="")
        self.score_label.pack(side=tk.RIGHT, padx=(0, 16))

    def _on_qa_resize(self, event):
        wrap = max(360, event.width - 48)
        self.question_label.configure(wraplength=wrap)
        self.feedback_label.configure(wraplength=wrap)

    # ---------- Helpers ----------
    def _get_audio_text(self, source: str) -> str:
        # Always target-language audio (passage or target-language question).
        item = self._current_item()
        return item["sentence"] if source == "Passage" else item["question"]

    def _scenario_items(self, scen: str) -> List[Dict[str, Any]]:
        return self.store.get_items(scen)

    def _current_item(self) -> Dict[str, Any]:
        scen = self.current_scenario.get()
        items = self._scenario_items(scen)
        if not isinstance(items, list) or len(items) == 0:
            raise KeyError(f"Scenario '{scen}' has no items or is not a list after import.")
        idx = self.order[scen][self.index[scen]]
        return items[idx]

    def _current_global_index(self) -> int:
        scen = self.current_scenario.get()
        return self.order[scen][self.index[scen]]

    # ------------- Refresh -------------
    def _refresh_all(self):
        self.scen_combo["values"] = self.store.all_titles()
        if self.current_scenario.get() not in self.store.all_titles():
            self.current_scenario.set(self.store.all_titles()[0])

        self._refresh_language_context()
        self._refresh_progress()
        self._refresh_sentence()
        self._refresh_translation()
        self._refresh_question()
        self._refresh_score()

    def _refresh_language_context(self):
        item = self._current_item()
        language = self._item_language(item)
        profile = get_language_profile(language)
        previous = self.qa_lang.get()
        allowed = [profile.display, "English"]
        self.qa_lang_combo.configure(values=allowed, width=max(12, min(24, len(profile.display) + 4)))
        if previous not in allowed:
            self.qa_lang.set("English" if previous == "English" else profile.display)
        self.audio_player.set_language(profile.display)
        self._apply_language_fonts(profile.display)
        self.passage_frame.configure(text=f"Passage ({profile.display})")
        self.qa_frame.configure(text=f"Question + Multiple Choice ({self.qa_lang.get()})")
        self.language_info_label.config(
            text=f"{profile.display} · {profile.native_name} · {profile.script_name}: {profile.script_sample}"
        )

    def _refresh_progress(self):
        scen = self.current_scenario.get()
        i = self.index[scen]
        total = len(self.order[scen])
        self.progress_label.config(text=f"{scen} | {i+1} of {total}")

    def _refresh_sentence(self):
        item = self._current_item()
        text = "" if self.hide_sentence.get() else item["sentence"]
        self.passage_text.configure(state=tk.NORMAL)
        self.passage_text.delete("1.0", tk.END)
        self.passage_text.insert(tk.END, text)
        self.passage_text.configure(state=tk.DISABLED)

    def _refresh_translation(self):
        if self.show_translation.get():
            if not self.translation_frame.winfo_ismapped():
                self.translation_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
            item = self._current_item()
            target_lines = split_sentences(item.get("sentence", ""))
            en_lines = item.get("sentence_en_lines", [])
            n = min(len(target_lines), len(en_lines)) if en_lines else len(target_lines)
            self.translation_text.configure(state=tk.NORMAL)
            self.translation_text.delete("1.0", tk.END)
            if n == 0:
                self.translation_text.insert(tk.END, "(No aligned translations available in this scenario bundle.)")
            else:
                for i in range(n):
                    target = target_lines[i].strip()
                    en = en_lines[i].strip() if i < len(en_lines) else ""
                    self.translation_text.insert(tk.END, target + "\n", ("target",))
                    self.translation_text.insert(tk.END, en + "\n\n", ("en",))
            self.translation_text.configure(state=tk.DISABLED)
        else:
            if self.translation_frame.winfo_ismapped():
                self.translation_frame.forget()

    def _render_mc(self, item: Dict[str, Any]):
        for child in self.mc_container.winfo_children():
            child.destroy()
        self.option_cards.clear()
        self.choice_var.set(-1)

        # Choose language
        use_en = self.qa_lang.get() == "English"
        choices = item.get("choices_en" if use_en else "choices")
        if not (choices and isinstance(choices, list) and len(choices) == 4):
            lbl = ttk.Label(self.mc_container, text="(This item has no choices; import a bundle with 4 choices.)")
            lbl.pack(anchor="w")
            return
        for i, ch in enumerate(choices):
            card = OptionCard(self.mc_container, i, ch, self.choice_var, self._on_select_card, font=self.font_body)
            card.pack(fill=tk.X, pady=4)
            self.option_cards.append(card)

    def _refresh_question(self):
        item = self._current_item()
        use_en = self.qa_lang.get() == "English"
        q = item.get("question_en" if use_en else "question", "")
        if hasattr(self, "qa_frame"):
            self.qa_frame.configure(text=f"Question + Multiple Choice ({self.qa_lang.get()})")
        self.question_label.config(text=q)
        self.feedback_label.config(text="")
        self._render_mc(item)

        # Lock UI if already answered
        scen = self.current_scenario.get()
        gidx = self._current_global_index()
        answered = self.answered[scen].get(gidx)
        if answered:
            corr_idx = item.get("correct_index")
            wrong_idx = None if answered["correct"] else answered["selected"]
            self._apply_card_states("feedback", wrong_index=wrong_idx, correct_index=corr_idx)
            self.check_btn.config(state=tk.DISABLED)
        else:
            self._apply_card_states("normal")
            self.check_btn.config(state=tk.NORMAL)

    def _apply_card_states(self, mode: str, wrong_index: Optional[int]=None, correct_index: Optional[int]=None):
        for idx, card in enumerate(self.option_cards):
            if mode == "normal":
                state = "selected" if self.choice_var.get() == idx else "normal"
            elif mode == "feedback":
                if correct_index is not None and idx == correct_index:
                    state = "correct"
                elif wrong_index is not None and idx == wrong_index:
                    state = "wrong"
                else:
                    state = "disabled"
            else:
                state = "normal"
            card.set_state(state)

    # ------------- Text size -------------
    def _on_text_size_change(self, _e=None):
        try:
            size = int(self.size_combo.get())
        except Exception:
            return
        self.text_size_var.set(size)
        self.font_body.configure(size=size)
        self.font_body_bold.configure(size=size)
        self.font_body_italic.configure(size=size)
        # Apply to widgets
        self.passage_text.configure(font=self.font_body)
        self.translation_text.configure(font=self.font_body)
        self.translation_text.tag_configure("target", font=self.font_body_bold)
        self.translation_text.tag_configure("en", font=self.font_body_italic)
        self.question_label.configure(font=self.font_body_bold)
        for card in self.option_cards:
            card.set_font(self.font_body)

    # ------------- Actions -------------
    def _on_select_card(self, idx: int):
        # Ignore selections on answered items
        scen = self.current_scenario.get()
        gidx = self._current_global_index()
        if self.answered[scen].get(gidx):
            return
        # Update selection visuals
        self._apply_card_states("normal")

        # Instant TTS only if Q&A is shown in the target language.
        if self.qa_lang.get() == "English":
            return
        try:
            item = self._current_item()
            choice_text = item["choices"][idx]
            voice_name = self.audio_player.current_voice_name()
            try:
                speed = float(self.audio_player.speed_var.get())
            except Exception:
                speed = 1.0
            path, length, note = self.synth.get_audio(choice_text, voice_name, speed)
            self.audio.load(path, length)
            self.audio.play(0.0)
            # Sync audio UI display
            self.audio_player.total_len = length
            self.audio_player.tot_lbl.config(text=mmss(length))
            self.audio_player.note_lbl.config(text=note or "")
            self.audio_player.scale.set(0.0)
            self.audio_player._start_updater()
        except Exception as e:
            messagebox.showerror("Audio error", str(e))

    def _on_check(self):
        scen = self.current_scenario.get()
        gidx = self._current_global_index()
        # Prevent re-check
        if self.answered[scen].get(gidx):
            self.check_btn.config(state=tk.DISABLED)
            return

        item = self._current_item()
        corr_idx = item.get("correct_index")
        sel = self.choice_var.get()
        if sel == -1:
            messagebox.showinfo("Select an option", "Please select one choice.")
            return

        is_correct = (sel == corr_idx)
        # Record answer (lock)
        self.answered[scen][gidx] = {"selected": sel, "correct": is_correct}
        if is_correct:
            self.correct_count[scen] += 1
            self.feedback_label.config(text="Correct.")
            self._apply_card_states("feedback", correct_index=corr_idx)
        else:
            self.feedback_label.config(text="Incorrect. See Reveal or continue.")
            self._apply_card_states("feedback", wrong_index=sel, correct_index=corr_idx)

        # Lock check button for this item
        self.check_btn.config(state=tk.DISABLED)
        self._refresh_score()

        # If all answered, show summary
        if len(self.answered[scen]) == len(self.order[scen]):
            self._show_summary()

    def _show_summary(self):
        scen = self.current_scenario.get()
        items = self._scenario_items(scen)
        total = len(self.order[scen])
        correct = self.correct_count[scen]

        win = tk.Toplevel(self)
        win.title("Test Summary")
        win.geometry("760x520")
        ttk.Label(win, text=f"Score: {correct}/{total}", font=self.font_body_bold).pack(anchor="w", padx=12, pady=(12, 6))

        # Detail box
        box = scrolledtext.ScrolledText(win, wrap=tk.WORD, height=24)
        box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        if correct == total:
            box.insert(tk.END, "Great job — all answers were correct!\n")
        else:
            box.insert(tk.END, "Review of incorrect items:\n\n")
            for i, global_idx in enumerate(self.order[scen], 1):
                rec = self.answered[scen].get(global_idx)
                if rec and not rec["correct"]:
                    it = items[global_idx]
                    q_en = it.get("question_en", it.get("question", ""))
                    ch_en = it.get("choices_en", it.get("choices", []))
                    sel = rec["selected"]
                    corr = it.get("correct_index", 0)
                    why = it.get("explain_en", it.get("explain", ""))
                    your = ch_en[sel] if 0 <= sel < len(ch_en) else "(?)"
                    corr_text = ch_en[corr] if 0 <= corr < len(ch_en) else "(?)"
                    box.insert(tk.END, f"Q{i}: {q_en}\n")
                    box.insert(tk.END, f"  Your answer: {your}\n")
                    box.insert(tk.END, f"  Correct answer: {corr_text}\n")
                    box.insert(tk.END, f"  Why: {why}\n\n")
        box.configure(state=tk.DISABLED)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 12))

    def _on_reveal(self):
        item = self._current_item()
        use_en = self.qa_lang.get() == "English"
        corr_idx = item.get("correct_index", 0)
        choices = item.get("choices_en" if use_en else "choices", [])
        ans = f"{'ABCD'[corr_idx]}. {choices[corr_idx] if 0 <= corr_idx < len(choices) else '(?)'}"
        explanation = item.get("explain_en" if use_en else "explain")
        msg = f"Answer: {ans}"
        if explanation:
            msg += f"\n\nWhy: {explanation}"
        self.feedback_label.config(text=msg)
        self._apply_card_states("feedback", correct_index=corr_idx)

    def _prev_item(self):
        scen = self.current_scenario.get()
        if self.index[scen] > 0:
            self.index[scen] -= 1
            self._refresh_all()

    def _next_item(self):
        scen = self.current_scenario.get()
        if self.index[scen] < len(self.order[scen]) - 1:
            self.index[scen] += 1
            self._refresh_all()

    def _reset_scenario(self):
        scen = self.current_scenario.get()
        self.order[scen] = list(range(len(self._scenario_items(scen))))
        #if self.randomized.get():
        #    random.shuffle(self.order[scen])
        self.index[scen] = 0
        self.correct_count[scen] = 0
        self.answered[scen] = {}
        self._refresh_all()

    def _on_randomize_toggle(self):
        scen = self.current_scenario.get()
        if self.randomized.get():
            random.shuffle(self.order[scen])
        else:
            self.order[scen] = list(range(len(self._scenario_items(scen))))
        self.index[scen] = 0
        self._refresh_all()

    def _on_scenario_change(self):
        scen = self.current_scenario.get()
        if scen not in self.order:
            total = len(self._scenario_items(scen))
            self.order[scen] = list(range(total))
            self.index[scen] = 0
            self.correct_count[scen] = 0
            self.answered[scen] = {}
        if self.randomized.get():
            random.shuffle(self.order[scen])
        self._refresh_all()

    def _refresh_score(self):
        scen = self.current_scenario.get()
        correct = self.correct_count[scen]
        total = len(self.order[scen])
        self.score_label.config(text=f"Score: {correct}/{total}")

    def _bind_shortcuts(self):
        for i in range(4):
            self.bind(str(i+1), lambda e, idx=i: self._select_by_key(idx))
        self.bind("<space>", lambda e: self.audio_player.on_pause())
        self.bind("<Right>", lambda e: self._next_item())
        self.bind("<Left>", lambda e: self._prev_item())

    def _select_by_key(self, idx: int):
        if 0 <= idx < len(self.option_cards):
            # Ignore if answered
            scen = self.current_scenario.get()
            gidx = self._current_global_index()
            if self.answered[scen].get(gidx):
                return
            self.choice_var.set(idx)
            self._apply_card_states("normal")
            self._on_select_card(idx)

    def destroy(self):
        try:
            self.audio.stop()
            pygame.mixer.quit()
        finally:
            super().destroy()

# ==========================================================
#                         M A I N
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="AI Language Scenario Generator & Trainer.")
    parser.add_argument("--no-gui", action="store_true", help="Reserved. GUI only.")
    args = parser.parse_args()

    if args.no_gui:
        print("This script is designed for GUI use. Run without --no-gui.")
        return

    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
