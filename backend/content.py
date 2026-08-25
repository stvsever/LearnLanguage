"""Content generation service.

Prompts are designed around second-language-acquisition research:
- lessons teach high-frequency *chunks* in sentence context (usage-based learning),
- compositions are comprehensible input pitched slightly above the learner's
  level (i+1), with the format (dialogue / monologue / story / article)
  CLASSIFIED BY THE MODEL from the learner's free-form request in the same
  single call that writes the content,
- every generation is grammar-aware: the language's grammar roadmap
  (backend/grammar.py) tells the model which structures the learner owns and
  which to target, and the response reports which features it actually used -
  feeding local progress tracking.

When no LLM is available, curated seed content keeps the app fully usable.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from . import config
from .grammar import feature_index, prompt_brief
from .languages import get_language
from .llm import LLMUnavailable, generate_structured
from .models import CompositionPack, Gloss, LessonPack

logger = logging.getLogger(__name__)

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")

LEVEL_GUIDANCE = {
    "A1": "Absolute beginner. Very high-frequency words and fixed chunks. Simple present-oriented sentences under 8 words.",
    "A2": "Elementary. Everyday topics, first past-tense forms, short compound sentences.",
    "B1": "Intermediate. Opinions, plans, narration across time frames. Natural connectors.",
    "B2": "Upper-intermediate. Abstract topics, idiomatic collocations, register contrasts.",
    "C1": "Advanced. Nuanced idiom, low-frequency vocabulary, complex syntax.",
    "C2": "Mastery. Native-like idiom, subtle register, literary and technical range.",
}


def normalize_level(value: Optional[str]) -> str:
    raw = str(value or "A2").strip().upper()
    return raw if raw in CEFR_LEVELS else "A2"


def _load_seed(name: str) -> Optional[dict]:
    path = config.SEED_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read seed file %s", name)
        return None


def seed_lesson(language_code: str, count: int) -> Optional[dict]:
    seed = _load_seed(f"{language_code}_core.json")
    if not seed:
        return None
    pack = dict(seed)
    pack["items"] = seed.get("items", [])[: max(4, count)]
    pack["source"] = "seed"
    if config.active_provider() == "offline":
        pack["notice"] = (
            "Built-in starter deck. Add an OPENROUTER_API_KEY to .env to generate "
            "unlimited lessons on any topic."
        )
    return pack


def seed_composition(language_code: str) -> Optional[dict]:
    seed = _load_seed(f"{language_code}_composition.json")
    if not seed:
        return None
    pack = dict(seed)
    pack["source"] = "seed"
    if config.active_provider() == "offline":
        pack["notice"] = (
            "Built-in starter piece. Add an OPENROUTER_API_KEY to .env to compose "
            "anything you can describe."
        )
    return pack


def _valid_feature_list(language_code: str, level: str) -> List[str]:
    """Feature menu offered to the model: ids at or one level below the target."""
    index = feature_index(language_code)
    position = CEFR_LEVELS.index(level)
    allowed_levels = set(CEFR_LEVELS[max(0, position - 1): position + 1])
    return [fid for fid, meta in index.items() if meta["level"] in allowed_levels]


def _filter_features(features: List[str], language_code: str) -> List[str]:
    index = feature_index(language_code)
    return [f for f in features if f in index]


def generate_lesson(topic: str, language_code: str, level: str, count: int,
                    known_words: Optional[list] = None, model: Optional[str] = None) -> dict:
    language = get_language(language_code)
    level = normalize_level(level)
    count = max(4, min(int(count or 12), 24))
    topic = (topic or "everyday conversation").strip()[:300]
    system_prompt = (
        "You are an expert language-course designer applying second-language-acquisition research. "
        "You create vocabulary lessons made of high-frequency, immediately usable items."
    )
    payload = {
        "task": f"Create {count} learning items in {language.prompt_name} for an English speaker.",
        "topic": topic,
        "learner_level": f"CEFR {level}. {LEVEL_GUIDANCE[level]}",
        "orthography": language.script_hint,
        "grammar_brief": prompt_brief(language.code, level),
        "grammar_feature_menu": _valid_feature_list(language.code, level),
        "rules": [
            "Prefer multi-word chunks and collocations over isolated words when natural (usage-based learning).",
            "Every item must be genuinely high-frequency and useful for this topic and level.",
            "pronunciation: IPA (or pinyin with tone marks for Mandarin) for the target item.",
            "example: one short, natural sentence using the item, pitched at the learner's level.",
            "example_en: faithful natural English translation of the example.",
            "note: include ONLY when there is a real trap - false friend, irregular form, register, or grammar point. One short sentence. Otherwise empty string.",
            "tags: 1-3 lowercase tags (part of speech or theme).",
            "grammar_features: ids from grammar_feature_menu that the examples genuinely exercise (0-4 ids).",
            "No duplicates. No numbering. No romanization in 'target' beyond the standard script.",
        ],
    }
    if known_words:
        payload["avoid_these_already_known_items"] = list(known_words)[:120]
    try:
        pack = generate_structured(system_prompt, json.dumps(payload, ensure_ascii=False), LessonPack, model_override=model)
        result = pack.model_dump()
        result.update({"language": language.code, "topic": topic, "level": level, "source": config.active_provider()})
        result["items"] = result["items"][:count]
        result["grammar_features"] = _filter_features(result.get("grammar_features", []), language.code)
        return result
    except LLMUnavailable as exc:
        logger.warning("Lesson generation unavailable: %s", exc)
        seeded = seed_lesson(language.code, count)
        if seeded:
            return seeded
        raise


def generate_composition(prompt: str, language_code: str, level: str, length: str = "medium",
                         model: Optional[str] = None) -> dict:
    """One call: classify the right format for the learner's request AND write it."""
    language = get_language(language_code)
    level = normalize_level(level)
    prompt = (prompt or "an everyday situation with a small, satisfying twist").strip()[:500]
    segment_range = {"short": "6 to 9", "medium": "10 to 16", "long": "18 to 26"}.get(length, "10 to 16")
    system_prompt = (
        "You are an expert author of graded learning texts (comprehensible input, i+1) and a "
        "careful applied linguist. The learner describes what they want in free form. You must:\n"
        "1) CLASSIFY the best presentation format for that request - dialogue (spoken exchange, "
        "2-3 named speakers), monologue (one voice: speech, voicemail, inner thoughts, vlog), "
        "story (narrated fiction), or article (expository/informational prose). Respect any "
        "explicit format wish in the request; otherwise choose what serves the content best.\n"
        "2) WRITE it at exactly the learner's level: mostly language they own, a thin layer of "
        "inferable new material.\n"
        "3) Weave in the TARGET grammar structures listed in the grammar brief, and report which "
        "feature ids you actually used as grammar_spotlights with exact excerpts."
    )
    payload = {
        "learner_request": prompt,
        "target_language": language.prompt_name,
        "learner_level": f"CEFR {level}. {LEVEL_GUIDANCE[level]}",
        "orthography": language.script_hint,
        "grammar_brief": prompt_brief(language.code, level),
        "grammar_feature_menu": _valid_feature_list(language.code, level),
        "rules": [
            f"segments: {segment_range} segments. For dialogues: one speaker turn per segment (speaker = the name). "
            "For prose: one sentence per segment (speaker = empty string).",
            "text_en: exactly one faithful, natural English translation per segment.",
            "Dialogues: 2-3 participants with simple, culturally fitting names; list them in participants.",
            "Make it concrete and alive: names, places, small tension or insight, a satisfying ending.",
            "scene: one short English line setting the scene.",
            "glossary: 8-14 words/chunks from the text a learner at this level may not know, with concise contextual glosses.",
            "grammar_spotlights: 2-4 entries; feature must be an id from grammar_feature_menu; excerpt must be copied verbatim from a segment.",
            "questions: 4 comprehension questions in the target language requiring real understanding (inference welcome), "
            "4 plausible choices each, exactly one correct, one-sentence English explanation. Vary the position of the correct choice.",
            "title: short and evocative, in the target language.",
        ],
    }
    try:
        pack = generate_structured(system_prompt, json.dumps(payload, ensure_ascii=False), CompositionPack, model_override=model)
        result = pack.model_dump()
        result.update({"language": language.code, "level": level, "source": config.active_provider()})
        result["grammar_spotlights"] = [
            s for s in result.get("grammar_spotlights", []) if s.get("feature") in feature_index(language.code)
        ]
        if result["format"] != "dialogue":
            result["participants"] = []
        return result
    except LLMUnavailable as exc:
        logger.warning("Composition generation unavailable: %s", exc)
        seeded = seed_composition(language.code)
        if seeded:
            return seeded
        raise


def generate_gloss(text: str, context: str, language_code: str, model: Optional[str] = None) -> dict:
    language = get_language(language_code)
    text = (text or "").strip()[:120]
    if not text:
        raise ValueError("Text is required.")
    system_prompt = (
        "You are a concise bilingual dictionary that explains words as they are used in context."
    )
    payload = {
        "task": f"Explain this {language.prompt_name} word/phrase for an English-speaking learner.",
        "selection": text,
        "sentence_context": (context or "").strip()[:400],
        "rules": [
            "gloss: the contextual English meaning, max 8 words.",
            "lemma: dictionary form if the selection is inflected, else empty string.",
            "pronunciation: IPA (or pinyin with tone marks for Mandarin) of the selection.",
            "note: one short grammar/usage note only if genuinely helpful, else empty string.",
        ],
    }
    result = generate_structured(system_prompt, json.dumps(payload, ensure_ascii=False), Gloss, model_override=model)
    payload_out = result.model_dump()
    payload_out["text"] = text
    payload_out["source"] = config.active_provider()
    return payload_out
