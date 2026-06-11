from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import edge_tts
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

try:
    from mutagen.mp3 import MP3 as MP3Info
except Exception:
    MP3Info = None

from .languages import LANGUAGES, get_language, normalize_language_code, public_language_payload
from .models import ScenarioPack, TranslationBatch, VocabularyItem, VocabularyPack


logger = logging.getLogger(__name__)

TUTOR3_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = TUTOR3_DIR / "static"
RUNTIME_DIR = TUTOR3_DIR / "runtime"
AUDIO_DIR = RUNTIME_DIR / "audio"
CACHE_DIR = RUNTIME_DIR / "cache"
SESSION_DIR = RUNTIME_DIR / "sessions"
VOCAB_SOURCE_PATH = REPO_ROOT / "tutors" / "tutor1" / "data" / "vocabulary_es.json"
load_dotenv(REPO_ROOT / ".env")
TRANSLATION_CACHE_PATH = CACHE_DIR / "topic_translations.json"
MODEL = os.getenv("LEARNLANGUAGE_MODEL", "gpt-4o-mini")
T = TypeVar("T", bound=BaseModel)


def ensure_runtime_dirs() -> None:
    for path in (AUDIO_DIR, CACHE_DIR, SESSION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def serialize_model(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return json.loads(model.json())


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "item"


def difficulty_label(value: str) -> str:
    labels = {
        "beginner": "Beginner, A1",
        "elementary": "Elementary, A2",
        "intermediate": "Intermediate, B1 to B2",
        "advanced": "Advanced, C1",
        "expert": "Expert, C2",
    }
    return labels.get((value or "").lower(), "Intermediate, B1 to B2")


def scenario_difficulty(value: str) -> str:
    mapping = {
        "beginner": "easy",
        "elementary": "easy",
        "intermediate": "medium",
        "advanced": "hard",
        "expert": "hard",
        "easy": "easy",
        "medium": "medium",
        "hard": "hard",
    }
    return mapping.get((value or "").lower(), "medium")


def openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, timeout=45.0, max_retries=0)


def parse_with_openai(system_prompt: str, user_prompt: str, response_model: Type[T]) -> T:
    client = openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    start = time.perf_counter()
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=response_model,
    )
    logger.info("OpenAI generation finished in %.2fs with model %s.", time.perf_counter() - start, MODEL)
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI returned no parsed response.")
    return parsed


def log_generation_failure(label: str, exc: Exception) -> None:
    message = str(exc)
    if "OPENAI_API_KEY is not configured" in message:
        logger.warning("%s skipped because OPENAI_API_KEY is not configured.", label)
    elif "timed out" in message.lower() or exc.__class__.__name__ in {"APITimeoutError", "ReadTimeout"}:
        logger.warning("%s timed out and returned local demo content.", label)
    elif "validation error" in message.lower():
        logger.warning("%s returned invalid structured content and used local demo content.", label)
    else:
        logger.exception("%s failed: %s", label, exc)


def config_payload() -> dict:
    return {
        "appName": "LearnLanguage",
        "model": MODEL,
        "languages": public_language_payload(),
        "difficulties": [
            {"id": "beginner", "label": "Beginner", "cefr": "A1"},
            {"id": "elementary", "label": "Elementary", "cefr": "A2"},
            {"id": "intermediate", "label": "Intermediate", "cefr": "B1 to B2"},
            {"id": "advanced", "label": "Advanced", "cefr": "C1"},
            {"id": "expert", "label": "Expert", "cefr": "C2"},
        ],
        "hasOpenAIKey": bool(os.getenv("OPENAI_API_KEY")),
    }


def load_vocabulary_topics() -> List[dict]:
    if not VOCAB_SOURCE_PATH.exists():
        return []
    raw = json.loads(VOCAB_SOURCE_PATH.read_text(encoding="utf-8"))
    topics = raw.get("topics", [])
    payload = []
    for index, topic in enumerate(topics):
        names = topic.get("name", ["Untitled", "Sin título"])
        english_name = names[0] if names else "Untitled"
        spanish_name = names[1] if len(names) > 1 else english_name
        entries = topic.get("entries", [])
        payload.append(
            {
                "id": f"{index}-{slugify(english_name)}",
                "index": index,
                "nameEn": english_name,
                "nameEs": spanish_name,
                "count": len(entries),
            }
        )
    return payload


def topic_by_id(topic_id: str) -> Tuple[dict, List[List[str]], str]:
    raw = json.loads(VOCAB_SOURCE_PATH.read_text(encoding="utf-8"))
    topics = raw.get("topics", [])
    index_text = str(topic_id).split("-", 1)[0]
    if not index_text.isdigit():
        raise ValueError("Invalid topic id.")
    index = int(index_text)
    if index < 0 or index >= len(topics):
        raise ValueError("Topic not found.")
    topic = topics[index]
    names = topic.get("name", ["Untitled", "Sin título"])
    topic_name = names[0] if names else "Untitled"
    return topic, topic.get("entries", []), topic_name


def load_translation_cache() -> dict:
    if not TRANSLATION_CACHE_PATH.exists():
        return {"schemaVersion": 1, "topics": {}}
    try:
        return json.loads(TRANSLATION_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Translation cache was unreadable. Starting with an empty cache.")
        return {"schemaVersion": 1, "topics": {}}


def save_translation_cache(cache: dict) -> None:
    ensure_runtime_dirs()
    TRANSLATION_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def demo_vocabulary(language_code: str, concept: str, count: int, difficulty: str) -> dict:
    samples = {
        "es": [
            ("Where is the metro station?", "¿Dónde está la estación de metro?"),
            ("I need a ticket.", "Necesito un billete."),
            ("Does this bus go downtown?", "¿Este autobús va al centro?"),
            ("The platform is on the left.", "El andén está a la izquierda."),
            ("We arrive in ten minutes.", "Llegamos en diez minutos."),
        ],
        "ru": [
            ("Where is the metro station?", "Где находится станция метро?"),
            ("I need a ticket.", "Мне нужен билет."),
            ("Does this bus go downtown?", "Этот автобус едет в центр?"),
            ("The platform is on the left.", "Платформа находится слева."),
            ("We arrive in ten minutes.", "Мы приедем через десять минут."),
        ],
        "fr": [
            ("Where is the metro station?", "Où est la station de métro ?"),
            ("I need a ticket.", "J'ai besoin d'un billet."),
            ("Does this bus go downtown?", "Ce bus va-t-il au centre-ville ?"),
            ("The platform is on the left.", "Le quai est à gauche."),
            ("We arrive in ten minutes.", "Nous arrivons dans dix minutes."),
        ],
        "zh": [
            ("Where is the metro station?", "地铁站在哪里？"),
            ("I need a ticket.", "我需要一张票。"),
            ("Does this bus go downtown?", "这辆公交车去市中心吗？"),
            ("The platform is on the left.", "站台在左边。"),
            ("We arrive in ten minutes.", "我们十分钟后到。"),
        ],
    }
    rows = samples.get(language_code, samples["es"])
    items = [
        VocabularyItem(
            english=english,
            target=target,
            note="Demo item. Add OPENAI_API_KEY for live generation.",
            tags=[difficulty, "demo"],
        )
        for english, target in rows[: max(1, min(count, len(rows)))]
    ]
    return {
        "source": "offline-demo",
        "warning": "OpenAI generation was unavailable, so LearnLanguage returned a built-in demo set.",
        **serialize_model(VocabularyPack(target_language=language_code, concept=concept, difficulty=difficulty, items=items)),
    }


def generate_vocabulary(concept: str, count: int, language_code: str, difficulty: str) -> dict:
    language = get_language(language_code)
    count = max(3, min(int(count or 12), 50))
    system_prompt = (
        "You are a precise language-learning content generator. "
        "Return only JSON matching the supplied schema. Do not add markdown or commentary."
    )
    user_prompt = json.dumps(
        {
            "task": "Create aligned English to target-language learning items.",
            "concept": concept,
            "item_count": count,
            "target_language": language.prompt_name,
            "native_name": language.native_name,
            "script": language.script_name,
            "orthography": language.sentence_hint,
            "difficulty": difficulty_label(difficulty),
            "rules": [
                "Items may be words, phrases, or short sentences depending on the concept.",
                "Keep every English item aligned with the target-language item at the same index.",
                "Use natural target-language wording, not literal word-for-word translation.",
                "For Mandarin Chinese, use Simplified Chinese characters.",
                "Include a short note only when it helps with grammar, register, morphology, or usage.",
                "Include concise tags such as phrase, noun, verb, travel, register, or grammar point.",
            ],
        },
        ensure_ascii=False,
    )
    try:
        pack = parse_with_openai(system_prompt, user_prompt, VocabularyPack)
        pack.target_language = language.code
        pack.concept = concept
        pack.difficulty = difficulty
        pack.items = pack.items[:count]
        return {"source": "openai", **serialize_model(pack)}
    except Exception as exc:
        log_generation_failure("Vocabulary generation", exc)
        return demo_vocabulary(language.code, concept, count, difficulty)


def translate_entries(topic_id: str, topic_name: str, english_entries: List[str], language_code: str, difficulty: str) -> List[str]:
    cache = load_translation_cache()
    topic_cache = cache.setdefault("topics", {}).setdefault(topic_id, {})
    language_cache = topic_cache.setdefault(language_code, {})
    missing = [item for item in english_entries if item not in language_cache]
    if missing:
        language = get_language(language_code)
        for start in range(0, len(missing), 40):
            chunk = missing[start : start + 40]
            system_prompt = (
                "You translate English vocabulary-learning items into the requested target language. "
                "Return only JSON matching the schema and preserve order exactly."
            )
            user_prompt = json.dumps(
                {
                    "topic": topic_name,
                    "difficulty": difficulty_label(difficulty),
                    "target_language": language.prompt_name,
                    "native_name": language.native_name,
                    "orthography": language.sentence_hint,
                    "items": chunk,
                    "rules": [
                        "Return exactly one natural equivalent per source item.",
                        "Do not add explanations, numbering, pinyin, or punctuation unless it belongs to the phrase.",
                        "Use Simplified Chinese characters for Mandarin Chinese.",
                    ],
                },
                ensure_ascii=False,
            )
            translated = parse_with_openai(system_prompt, user_prompt, TranslationBatch)
            if len(translated.translations) != len(chunk):
                raise ValueError("Translation count did not match source count.")
            for source, target in zip(chunk, translated.translations):
                language_cache[source] = target
            save_translation_cache(cache)
    return [language_cache.get(item, item) for item in english_entries]


def topic_vocabulary(topic_id: str, language_code: str, count: int, difficulty: str) -> dict:
    language = get_language(language_code)
    topic, entries, topic_name = topic_by_id(topic_id)
    count = max(3, min(int(count or 20), len(entries) or 20))
    selected = entries[:count]
    english_entries = [row[0] for row in selected]
    if language.code == "es":
        target_entries = [row[1] for row in selected]
        source = "topic-source"
    else:
        try:
            target_entries = translate_entries(topic_id, topic_name, english_entries, language.code, difficulty)
            source = "topic-openai-cache"
        except Exception as exc:
            log_generation_failure("Topic translation", exc)
            demo = demo_vocabulary(language.code, topic_name, count, difficulty)
            demo["concept"] = topic_name
            return demo
    items = [
        VocabularyItem(english=english, target=target, tags=["topic", topic_name])
        for english, target in zip(english_entries, target_entries)
    ]
    return {
        "source": source,
        "topic": {
            "id": topic_id,
            "nameEn": topic.get("name", [topic_name])[0],
            "nameEs": topic.get("name", [topic_name, topic_name])[1],
        },
        **serialize_model(VocabularyPack(target_language=language.code, concept=topic_name, difficulty=difficulty, items=items)),
    }


_SENTENCE_RE = re.compile(r".+?(?:[.!?。！？]+[\"'”’»）)]*|$)", flags=re.S)


def split_sentences(text: str) -> List[str]:
    source = " ".join(line.strip() for line in (text or "").splitlines() if line.strip())
    parts = [part.strip() for part in _SENTENCE_RE.findall(source) if part.strip()]
    return parts or ([source] if source else [])


def fix_scenario_pack(pack: ScenarioPack, language_code: str) -> ScenarioPack:
    pack.target_language = language_code
    for item in pack.items:
        if 0 <= item.correct_choice < len(item.choices):
            correct = item.choices[item.correct_choice].strip()
            if correct and not any(answer.strip().lower() == correct.lower() for answer in item.accepted_answers):
                item.accepted_answers = [correct, *item.accepted_answers]
        if len(item.choices_en) != 4:
            item.choices_en = list(item.choices[:4])
    sentence_count = len(split_sentences(pack.passage))
    if len(pack.passage_en_lines) != sentence_count:
        if len(pack.passage_en_lines) > sentence_count:
            pack.passage_en_lines = pack.passage_en_lines[:sentence_count]
        else:
            pack.passage_en_lines = [*pack.passage_en_lines, *[""] * (sentence_count - len(pack.passage_en_lines))]
    return pack


def demo_scenario(language_code: str, topic: str, difficulty: str) -> dict:
    demo = {
        "es": {
            "title": "Retraso en la estación",
            "passage": "Marta llega a la estación a las 8:10 porque su autobús se retrasa. El tren directo sale a las 8:15, pero el andén cambia al número cuatro. Si pierde ese tren, tendrá que esperar treinta minutos y llamar a su cliente.",
            "lines": [
                "Marta arrives at the station at 8:10 because her bus is delayed.",
                "The direct train leaves at 8:15, but the platform changes to number four.",
                "If she misses that train, she will have to wait thirty minutes and call her client.",
            ],
            "question": "¿Por qué Marta debe moverse rápidamente al andén cuatro?",
            "choices": ["Porque el tren directo sale pronto.", "Porque su cliente ya llegó.", "Porque el autobús vuelve a salir.", "Porque la estación cierra."],
            "reason": "El tren directo sale a las 8:15 y ella llega a las 8:10.",
        },
        "ru": {
            "title": "Задержка на станции",
            "passage": "Марта приезжает на станцию в 8:10, потому что автобус задержался. Прямой поезд отправляется в 8:15, но платформу меняют на четвертую. Если она опоздает на этот поезд, ей придется ждать тридцать минут и звонить клиенту.",
            "lines": [
                "Marta arrives at the station at 8:10 because her bus is delayed.",
                "The direct train leaves at 8:15, but the platform changes to number four.",
                "If she misses that train, she will have to wait thirty minutes and call her client.",
            ],
            "question": "Почему Марте нужно быстро перейти на четвертую платформу?",
            "choices": ["Потому что прямой поезд скоро отправляется.", "Потому что клиент уже приехал.", "Потому что автобус снова уезжает.", "Потому что станция закрывается."],
            "reason": "Прямой поезд отправляется в 8:15, а она приезжает в 8:10.",
        },
        "fr": {
            "title": "Retard à la gare",
            "passage": "Marta arrive à la gare à 8 h 10 parce que son bus est en retard. Le train direct part à 8 h 15, mais le quai passe au numéro quatre. Si elle manque ce train, elle devra attendre trente minutes et appeler son client.",
            "lines": [
                "Marta arrives at the station at 8:10 because her bus is delayed.",
                "The direct train leaves at 8:15, but the platform changes to number four.",
                "If she misses that train, she will have to wait thirty minutes and call her client.",
            ],
            "question": "Pourquoi Marta doit-elle se déplacer rapidement vers le quai quatre ?",
            "choices": ["Parce que le train direct part bientôt.", "Parce que son client est déjà arrivé.", "Parce que le bus repart.", "Parce que la gare ferme."],
            "reason": "Le train direct part à 8 h 15 et elle arrive à 8 h 10.",
        },
        "zh": {
            "title": "车站延误",
            "passage": "玛尔塔因为公交车晚点，八点十分才到车站。直达列车八点十五分开，但站台改到了四号。 如果她错过这趟车，就必须等三十分钟并给客户打电话。",
            "lines": [
                "Marta arrives at the station at 8:10 because her bus is delayed.",
                "The direct train leaves at 8:15, but the platform changes to number four.",
                "If she misses that train, she will have to wait thirty minutes and call her client.",
            ],
            "question": "玛尔塔为什么必须快点去四号站台？",
            "choices": ["因为直达列车很快就要开了。", "因为客户已经到了。", "因为公交车又要出发。", "因为车站要关门。"],
            "reason": "直达列车八点十五分开，而她八点十分才到。",
        },
    }[language_code]
    item = {
        "question": demo["question"],
        "choices": demo["choices"],
        "reasoning_note": demo["reason"],
        "question_en": "Why does Marta need to move quickly to platform four?",
        "choices_en": [
            "Because the direct train leaves soon.",
            "Because her client has already arrived.",
            "Because the bus is leaving again.",
            "Because the station is closing.",
        ],
        "reasoning_note_en": "The direct train leaves at 8:15, and she arrives at 8:10.",
        "correct_choice": 0,
        "accepted_answers": [demo["choices"][0]],
    }
    pack = ScenarioPack(
        target_language=language_code,
        scenario_title=demo["title"],
        difficulty=scenario_difficulty(difficulty),
        passage=demo["passage"],
        passage_en_lines=demo["lines"],
        items=[item, item, item, item, item],
    )
    return {
        "source": "offline-demo",
        "warning": "OpenAI generation was unavailable, so LearnLanguage returned a built-in scenario.",
        **serialize_model(pack),
    }


def generate_scenario(topic: str, language_code: str, difficulty: str) -> dict:
    language = get_language(language_code)
    scenario_level = scenario_difficulty(difficulty)
    system_prompt = f"""
You are a rigorous educational content generator for a multilingual AI tutor.
Return only JSON matching the supplied schema.

Target language: {language.prompt_name} ({language.native_name})
Script and orthography: {language.sentence_hint}

Requirements:
- Write the passage, questions, choices, target-language explanations, and accepted answers in {language.prompt_name}.
- Write passage_en_lines in English, aligned one-to-one with target-language sentences.
- Create exactly 5 inference-oriented multiple-choice questions.
- Each question has exactly 4 plausible choices, exactly one correct choice, and English mirror fields.
- Include the exact correct target-language choice in accepted_answers.
- Avoid meta answers such as all of the above or none of the above.
- Keep choices under 140 characters.
""".strip()
    user_prompt = json.dumps(
        {
            "topic": topic,
            "difficulty": scenario_level,
            "target_language": language.display,
            "passage_rules": [
                "3 to 5 target-language sentences.",
                "Information-dense with names, times, quantities, restrictions, causes, or trade-offs.",
                "Self-contained and useful for comprehension practice.",
            ],
        },
        ensure_ascii=False,
    )
    last_error: Optional[Exception] = None
    for attempt in range(1, 3):
        try:
            prompt = user_prompt
            if attempt > 1:
                prompt += "\n\nThe previous attempt was invalid. Do not use all of the above, none of the above, both A and B, or any equivalent in any language."
            pack = parse_with_openai(system_prompt, prompt, ScenarioPack)
            pack = fix_scenario_pack(pack, language.code)
            return {"source": "openai", **serialize_model(pack)}
        except Exception as exc:
            last_error = exc
            if attempt == 1:
                logger.warning("Scenario generation attempt failed, retrying once: %s", exc)

    log_generation_failure("Scenario generation", last_error or RuntimeError("Unknown scenario generation error."))
    return demo_scenario(language.code, topic, difficulty)


def audio_duration_seconds(path: Path) -> Optional[float]:
    if MP3Info is None:
        return None
    try:
        return round(float(MP3Info(str(path)).info.length), 2)
    except Exception:
        return None


def synthesize_speech(text: str, language_code: str, voice: Optional[str], rate: str = "-10%") -> dict:
    ensure_runtime_dirs()
    if not text.strip():
        raise ValueError("Text is required for TTS.")
    language = get_language(language_code)
    voice_id = voice or language.default_voice
    if voice_id not in {item.id for item in language.voices}:
        voice_id = language.default_voice
    key = hashlib.sha1(json.dumps([text, language.code, voice_id, rate], ensure_ascii=False).encode("utf-8")).hexdigest()
    out_path = AUDIO_DIR / f"{language.code}-{key}.mp3"
    if not out_path.exists() or out_path.stat().st_size == 0:
        async def _run() -> None:
            communicate = edge_tts.Communicate(text=text, voice=voice_id, rate=rate)
            await communicate.save(str(out_path))

        asyncio.run(_run())
    return {
        "language": language.code,
        "voice": voice_id,
        "url": f"/audio/{out_path.name}",
        "durationSeconds": audio_duration_seconds(out_path),
        "cached": True,
    }


def save_session(payload: Dict[str, Any]) -> dict:
    ensure_runtime_dirs()
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    filename = f"session-{created_at.replace(':', '').replace('+', 'Z')}.json"
    path = SESSION_DIR / filename
    clean_payload = {
        "createdAt": created_at,
        "app": "LearnLanguage",
        "payload": payload,
    }
    path.write_text(json.dumps(clean_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": True, "file": filename}
