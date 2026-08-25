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

from . import config, curriculum
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


def strip_em_dashes(value):
    """House style bans em dashes everywhere, including generated content.
    Walks any JSON-like structure and replaces them with plain hyphens."""
    if isinstance(value, str):
        return value.replace(" \u2014 ", " - ").replace("\u2014", "-")
    if isinstance(value, list):
        return [strip_em_dashes(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_em_dashes(item) for key, item in value.items()}
    return value


def _load_seed(name: str) -> Optional[dict]:
    path = config.SEED_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read seed file %s", name)
        return None


def seed_lesson(language_code: str, count: int = 24) -> Optional[dict]:
    """The curated cross-domain starter set for a language, no LLM involved."""
    language = get_language(language_code)
    pack = curriculum.starter_pack(language.code, max(4, count))
    if not pack:
        return None
    pack["notice"] = (
        "Curated starter set. Browse Topics for the full library, or connect a "
        "key to generate lessons on anything else."
    )
    return pack


def curriculum_lesson(language_code: str, unit_id: str) -> Optional[dict]:
    """One curriculum unit, shaped exactly like a generated lesson pack."""
    language = get_language(language_code)
    return curriculum.unit_pack(language.code, unit_id)


def _curriculum_fallback(language_code: str, topic: str, count: int) -> Optional[dict]:
    """Best curated material for a free-form topic when generation is unavailable.

    Tries a topic search first so "ordering coffee" lands on the cafe unit
    rather than on the generic starter set.
    """
    unit_id = curriculum.best_unit_for_topic(language_code, topic) if topic else None
    if unit_id:
        pack = curriculum.unit_pack(language_code, unit_id)
        if pack:
            pack["notice"] = (
                f"Generation is unavailable, so this is the curated "
                f"\u201c{pack['topic']}\u201d unit from the library instead."
            )
            return pack
    hits = curriculum.search(language_code, topic) if topic else {"units": [], "items": []}
    if hits["items"]:
        return {
            "language": language_code,
            "topic": topic,
            "level": hits["items"][0].get("level", "A2"),
            "items": hits["items"][:count],
            "grammar_features": [],
            "source": "curriculum",
            "notice": "Generation is unavailable; these are matching items from the curated library.",
        }
    return seed_lesson(language_code, count)


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
                    known_words: Optional[list] = None, model: Optional[str] = None,
                    unit: Optional[str] = None) -> dict:
    """Build a lesson pack.

    Three sources, in order of preference:
      1. a curriculum unit, when the learner picked one and generation is off
         (or when they explicitly asked for the curated version),
      2. the LLM, grounded in the unit or free-form topic,
      3. curated fallback content, so this never returns nothing.
    """
    language = get_language(language_code)
    level = normalize_level(level)
    count = max(4, min(int(count or 12), 24))
    unit_meta = curriculum.unit_meta(unit) if unit else None

    if unit_meta:
        # A picked unit always has curated content behind it; without an LLM
        # that IS the lesson, and with one it becomes the grounding brief.
        curated = curriculum_lesson(language.code, unit)
        if config.active_provider() == "offline":
            if curated:
                return curated
            raise LLMUnavailable("No curated content for this unit yet, and no API key configured.")
        topic = topic or f"{unit_meta['title']} ({unit_meta['goal']})"
        level = unit_meta["level"]
    else:
        curated = None

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
    if unit_meta:
        payload["curriculum_unit"] = {
            "title": unit_meta["title"],
            "area": unit_meta["domainTitle"],
            "can_do_goal": unit_meta["goal"],
            "keywords": unit_meta["keywords"],
        }
        payload["rules"].append(
            "This lesson EXTENDS a curated unit: stay strictly inside its scope and "
            "produce material the learner does not have yet (see the avoid list)."
        )
    avoid = list(known_words or [])
    if curated:
        avoid = [item["target"] for item in curated["items"]] + avoid
    if avoid:
        payload["avoid_these_already_known_items"] = avoid[:160]
    try:
        pack = generate_structured(system_prompt, json.dumps(payload, ensure_ascii=False), LessonPack, model_override=model)
        result = pack.model_dump()
        result.update({"language": language.code, "topic": topic, "level": level, "source": config.active_provider()})
        result["items"] = result["items"][:count]
        result["grammar_features"] = _filter_features(result.get("grammar_features", []), language.code)
        if unit:
            result["unit"] = unit
        return strip_em_dashes(result)
    except LLMUnavailable as exc:
        logger.warning("Lesson generation unavailable: %s", exc)
        fallback = curated or _curriculum_fallback(language.code, topic, count)
        if fallback:
            return fallback
        raise


COMPOSITION_FORMATS = ("dialogue", "monologue", "story", "article")

REGISTERS = {
    "casual": "Relaxed spoken register: contractions, everyday idiom, the informal address form.",
    "neutral": "Standard everyday register, neither slangy nor stiff.",
    "formal": "Careful, polite register: the formal address form, full forms, no slang.",
}

LENGTH_SEGMENTS = {"short": "6 to 9", "medium": "10 to 16", "long": "18 to 26"}


def normalize_format(value: Optional[str]) -> Optional[str]:
    """Explicit format wish, or None for 'let the model decide'."""
    raw = str(value or "").strip().lower()
    return raw if raw in COMPOSITION_FORMATS else None


def generate_composition(prompt: str, language_code: str, level: str, length: str = "medium",
                         model: Optional[str] = None, fmt: Optional[str] = None,
                         register: Optional[str] = None, speakers: Optional[int] = None,
                         focus: Optional[list] = None, vocabulary: Optional[list] = None,
                         unit: Optional[str] = None) -> dict:
    """Write a graded text.

    Every knob is optional. With none of them set the model classifies the best
    format from the free-form request, exactly as before. With them set, the
    learner is in charge: format, register, number of speakers, the grammar
    structures to exercise, and the vocabulary to weave in.
    """
    language = get_language(language_code)
    level = normalize_level(level)
    chosen_format = normalize_format(fmt)
    register_key = str(register or "neutral").strip().lower()
    if register_key not in REGISTERS:
        register_key = "neutral"
    segment_range = LENGTH_SEGMENTS.get(str(length or "medium").lower(), LENGTH_SEGMENTS["medium"])
    speaker_count = max(2, min(int(speakers or 2), 4)) if chosen_format == "dialogue" else None

    unit_meta = curriculum.unit_meta(unit) if unit else None
    if unit_meta and not (prompt or "").strip():
        prompt = f"A scene that naturally uses the language of {unit_meta['title'].lower()}: {unit_meta['goal']}"
    prompt = (prompt or "an everyday situation with a small, satisfying twist").strip()[:500]

    valid_features = _valid_feature_list(language.code, level)
    focus_ids = [f for f in (focus or []) if f in feature_index(language.code)][:4]

    if chosen_format:
        classify_line = (
            f"1) The learner has REQUESTED the format explicitly: write a {chosen_format}. "
            "Set format to exactly that value. Do not substitute another format.\n"
        )
    else:
        classify_line = (
            "1) CLASSIFY the best presentation format for that request - dialogue (spoken exchange, "
            "2-3 named speakers), monologue (one voice: speech, voicemail, inner thoughts, vlog), "
            "story (narrated fiction), or article (expository/informational prose). Respect any "
            "explicit format wish in the request; otherwise choose what serves the content best.\n"
        )
    system_prompt = (
        "You are an expert author of graded learning texts (comprehensible input, i+1) and a "
        "careful applied linguist. The learner describes what they want in free form. You must:\n"
        + classify_line +
        "2) WRITE it at exactly the learner's level: mostly language they own, a thin layer of "
        "inferable new material.\n"
        "3) Weave in the TARGET grammar structures listed in the grammar brief, and report which "
        "feature ids you actually used as grammar_spotlights with exact excerpts."
    )
    rules = [
        f"segments: {segment_range} segments. For dialogues: one speaker turn per segment (speaker = the name). "
        "For prose: one sentence per segment (speaker = empty string).",
        "text_en: exactly one faithful, natural English translation per segment.",
        "Make it concrete and alive: names, places, small tension or insight, a satisfying ending.",
        "scene: one short English line setting the scene.",
        "glossary: 8-14 words/chunks from the text a learner at this level may not know, with concise contextual glosses.",
        "grammar_spotlights: 2-4 entries; feature must be an id from grammar_feature_menu; excerpt must be copied verbatim from a segment.",
        "questions: 4 comprehension questions in the target language requiring real understanding (inference welcome), "
        "4 plausible choices each, exactly one correct, one-sentence English explanation. Vary the position of the correct choice.",
        "title: short and evocative, in the target language.",
    ]
    if speaker_count:
        rules.insert(2, f"Exactly {speaker_count} participants with simple, culturally fitting names; "
                        "list them in participants, in order of first appearance.")
    else:
        rules.insert(2, "Dialogues: 2-3 participants with simple, culturally fitting names; list them in participants.")

    payload = {
        "learner_request": prompt,
        "target_language": language.prompt_name,
        "learner_level": f"CEFR {level}. {LEVEL_GUIDANCE[level]}",
        "register": REGISTERS[register_key],
        "orthography": language.script_hint,
        "grammar_brief": prompt_brief(language.code, level),
        "grammar_feature_menu": valid_features,
        "rules": rules,
    }
    if chosen_format:
        payload["required_format"] = chosen_format
    if focus_ids:
        payload["must_exercise_these_features"] = focus_ids
        rules.append("The features in must_exercise_these_features have to appear in the text "
                     "AND in grammar_spotlights.")
    if vocabulary:
        payload["must_include_vocabulary"] = [str(v)[:60] for v in vocabulary][:24]
        rules.append("Every entry in must_include_vocabulary has to appear naturally in the text, "
                     "inflected as the sentence requires.")
    if unit_meta:
        payload["curriculum_unit"] = {
            "title": unit_meta["title"],
            "area": unit_meta["domainTitle"],
            "can_do_goal": unit_meta["goal"],
            "keywords": unit_meta["keywords"],
        }
    try:
        pack = generate_structured(system_prompt, json.dumps(payload, ensure_ascii=False), CompositionPack, model_override=model)
        result = pack.model_dump()
        result.update({"language": language.code, "level": level, "source": config.active_provider()})
        if chosen_format and result["format"] != chosen_format:
            # The learner asked for a format; honour the request over the model's taste.
            logger.info("Model returned %s, learner asked for %s - relabelling.", result["format"], chosen_format)
            result["format"] = chosen_format
        result["grammar_spotlights"] = [
            s for s in result.get("grammar_spotlights", []) if s.get("feature") in feature_index(language.code)
        ]
        if result["format"] != "dialogue":
            result["participants"] = []
        elif not result["participants"]:
            # Recover speaker list from the segments so per-speaker voices work.
            seen = []
            for segment in result["segments"]:
                name = (segment.get("speaker") or "").strip()
                if name and name not in seen:
                    seen.append(name)
            result["participants"] = seen
        if unit:
            result["unit"] = unit
        result["controls"] = {
            "format": chosen_format or "auto", "register": register_key,
            "length": length, "speakers": speaker_count, "focus": focus_ids,
        }
        return strip_em_dashes(result)
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
    return strip_em_dashes(payload_out)
