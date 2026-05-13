import asyncio
import json
import logging
import os
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)

import edge_tts
import openai
import pygame
from colorama import Fore, init
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

init(autoreset=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
AUDIO_DIR = APP_DIR / "audio"
SPANISH_AUDIO_DIR = AUDIO_DIR / "spanish"
TEST_RESULTS_DIR = APP_DIR / "test_results"
VOCAB_SOURCE_PATH = DATA_DIR / "vocabulary_es.json"
VOCAB_CACHE_PATH = DATA_DIR / "vocabulary_multilingual_cache.json"
MAX_TTS_WORKERS = 20

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.critical("No OpenAI API key found in environment variables.")
    raise ValueError("No OpenAI API key found in environment variables.")

openai.api_key = api_key
logger.info("OpenAI API key loaded successfully.")


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    display: str
    native_name: str
    prompt_name: str
    script_name: str
    sentence_hint: str
    default_voice_label: str
    voices: Dict[str, str]
    preferred_fonts: tuple[str, ...]


LANGUAGE_PROFILES: Dict[str, LanguageProfile] = {
    "es": LanguageProfile(
        code="es",
        display="Spanish",
        native_name="Español",
        prompt_name="Spanish",
        script_name="Latin alphabet with accents",
        sentence_hint="Use natural Spanish with correct accents and punctuation.",
        default_voice_label="Spain · Alvaro",
        voices={
            "Spain · Alvaro": "es-ES-AlvaroNeural",
            "Spain · Elvira": "es-ES-ElviraNeural",
            "Mexico · Dalia": "es-MX-DaliaNeural",
            "US · Paloma": "es-US-PalomaNeural",
        },
        preferred_fonts=("Helvetica", "Arial", "TkDefaultFont"),
    ),
    "ru": LanguageProfile(
        code="ru",
        display="Russian",
        native_name="Русский",
        prompt_name="Russian",
        script_name="Cyrillic",
        sentence_hint="Use natural Russian in Cyrillic, including Ё where it is normally written.",
        default_voice_label="Russia · Dmitry",
        voices={
            "Russia · Dmitry": "ru-RU-DmitryNeural",
            "Russia · Svetlana": "ru-RU-SvetlanaNeural",
        },
        preferred_fonts=("Helvetica", "Arial", "Arial Unicode MS", "TkDefaultFont"),
    ),
    "fr": LanguageProfile(
        code="fr",
        display="French",
        native_name="Français",
        prompt_name="French",
        script_name="Latin alphabet with French accents",
        sentence_hint="Use natural French with correct accents and typography.",
        default_voice_label="France · Henri",
        voices={
            "France · Henri": "fr-FR-HenriNeural",
            "France · Denise": "fr-FR-DeniseNeural",
            "Canada · Sylvie": "fr-CA-SylvieNeural",
        },
        preferred_fonts=("Helvetica", "Arial", "TkDefaultFont"),
    ),
    "zh": LanguageProfile(
        code="zh",
        display="Mandarin Chinese",
        native_name="普通话",
        prompt_name="Mandarin Chinese",
        script_name="Simplified Chinese characters",
        sentence_hint="Use Simplified Chinese characters. Do not add pinyin unless the user asks for it.",
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

LANGUAGE_ALIASES = {
    "spanish": "es",
    "español": "es",
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
}


def normalize_language_code(value: Optional[str]) -> str:
    if not value:
        return "es"
    raw = str(value).strip()
    if raw in LANGUAGE_PROFILES:
        return raw
    return LANGUAGE_ALIASES.get(raw.lower(), "es")


def get_language_profile(value: Optional[str]) -> LanguageProfile:
    return LANGUAGE_PROFILES[normalize_language_code(value)]


class BilingualDict(BaseModel):
    untranslated_words: List[str] = Field(..., description="English source items.")
    translated_words: List[str] = Field(..., description="Target-language equivalents aligned by index.")

    @field_validator("translated_words")
    @classmethod
    def v_nonempty_translations(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("translated_words cannot be empty")
        return v


class TranslationBatch(BaseModel):
    translations: List[str] = Field(..., description="Target-language equivalents aligned by index with the supplied English items.")


def call_GPT(
    system_prompt: str,
    user_query: str,
    pydantic_model: type[BaseModel],
    model: str = "gpt-5-nano",
) -> BaseModel:
    try:
        start = time.perf_counter()
        logger.info("LLM request started (model=%s).", model)
        response = openai.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            response_format=pydantic_model,
        )
        elapsed = time.perf_counter() - start
        logger.info("LLM response generated successfully in %.2fs.", elapsed)
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error("Error generating LLM response: %s", e)
        raise


def _load_cache() -> dict:
    if not VOCAB_CACHE_PATH.exists():
        return {"schema_version": 1, "topics": {}}
    try:
        return json.loads(VOCAB_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Vocabulary cache could not be read; starting with an empty cache.")
        return {"schema_version": 1, "topics": {}}


def _save_cache(cache: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VOCAB_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


class Tutor:
    """
    Backend for the vocabulary tutor.

    It produces English-to-target-language lists with OpenAI and uses Edge-TTS for
    language-specific audio in Spanish, Russian, French, and Mandarin Chinese.
    """

    def __init__(self):
        logger.info("Tutor initialized.")
        self.audio_files_en: Dict[str, str] = {}
        self.audio_files_target: Dict[str, str] = {}
        self.target_language = "es"
        self.audio_lock = threading.Lock()

        try:
            pygame.mixer.init()
            logger.info("Pygame mixer initialized successfully.")
        except Exception as e:
            logger.error("Error initializing pygame mixer: %s", e)

    def map_difficulty_to_level(self, difficulty: str) -> str:
        mapping = {
            "beginner": "Beginner",
            "elementary": "Elementary",
            "intermediate": "Intermediate",
            "advanced": "Advanced",
            "expert": "Expert",
        }
        return mapping.get(str(difficulty).lower(), "Intermediate")

    def request_concept(
        self,
        concept: str,
        num_items: int,
        target_language: str,
        difficulty: str = "intermediate",
    ) -> BilingualDict:
        profile = get_language_profile(target_language)
        difficulty_label = self.map_difficulty_to_level(difficulty)
        logger.info(
            "Generating %d item(s) for concept=%r in %s (%s).",
            num_items,
            concept,
            profile.display,
            difficulty_label,
        )
        system_prompt = (
            "You are a precise language-learning content generator. Return only JSON that matches the schema. "
            "Do not include private context, markdown, or commentary."
        )
        user_query = (
            f"Create exactly {num_items} useful language-learning items for the concept: {concept!r}.\n"
            f"Return `untranslated_words` in English and `translated_words` in {profile.prompt_name}.\n"
            f"Target language: {profile.display} ({profile.native_name}).\n"
            f"Orthography: {profile.sentence_hint}\n"
            f"Difficulty: {difficulty_label}.\n"
            "Items may be words, phrases, or short sentences depending on the concept. "
            "If the concept asks for an alphabet, include letters or characters as items. "
            "Keep each pair aligned by index and avoid duplicates."
        )
        try:
            return call_GPT(system_prompt, user_query, BilingualDict)
        except Exception as e:
            logger.error("Failed to retrieve bilingual content: %s", e)
            return BilingualDict(untranslated_words=[], translated_words=[])

    def translate_entries(
        self,
        topic_id: str,
        topic_name_en: str,
        english_entries: List[str],
        target_language: str,
        difficulty: str = "intermediate",
    ) -> List[str]:
        code = normalize_language_code(target_language)
        if code == "es":
            raise ValueError("Spanish source entries should be read directly from the vocabulary file.")

        cache = _load_cache()
        topic_cache = cache.setdefault("topics", {}).setdefault(topic_id, {})
        language_cache = topic_cache.setdefault(code, {})
        translations: List[Optional[str]] = [language_cache.get(item) for item in english_entries]
        missing = [item for item, translated in zip(english_entries, translations) if not translated]

        if missing:
            profile = get_language_profile(code)
            logger.info("Generating %d %s vocabulary equivalent(s) for topic '%s'.", len(missing), profile.display, topic_name_en)
            for start in range(0, len(missing), 40):
                chunk = missing[start:start + 40]
                translated = self._translate_chunk(chunk, profile, topic_name_en, difficulty)
                for src, dst in zip(chunk, translated):
                    language_cache[src] = dst
                _save_cache(cache)

        return [language_cache.get(item, item) for item in english_entries]

    def _translate_chunk(
        self,
        english_items: List[str],
        profile: LanguageProfile,
        topic_name_en: str,
        difficulty: str,
    ) -> List[str]:
        system_prompt = (
            "You translate English vocabulary-learning items into the requested target language. "
            "Return only JSON matching the schema. Preserve order exactly."
        )
        user_query = json.dumps(
            {
                "target_language": profile.display,
                "native_name": profile.native_name,
                "orthography": profile.sentence_hint,
                "topic": topic_name_en,
                "difficulty": self.map_difficulty_to_level(difficulty),
                "items": english_items,
                "rules": [
                    "Return exactly one natural equivalent per source item.",
                    "Keep technical terms precise.",
                    "Use Simplified Chinese for Mandarin Chinese.",
                    "Do not add explanations, pinyin, numbering, or punctuation unless it belongs to the phrase.",
                ],
            },
            ensure_ascii=False,
        )
        parsed: TranslationBatch = call_GPT(system_prompt, user_query, TranslationBatch)  # type: ignore[assignment]
        if len(parsed.translations) != len(english_items):
            raise ValueError("Translation count did not match source count.")
        return parsed.translations

    def text_to_speech(self, text: str, language: str = "es", voice_name: Optional[str] = None) -> str:
        profile = get_language_profile(language)
        voice = voice_name or profile.voices[profile.default_voice_label]
        out_path = ""
        try:
            fd, out_path = tempfile.mkstemp(prefix="language_tutor_", suffix=".mp3")
            os.close(fd)

            async def _synth() -> None:
                communicate = edge_tts.Communicate(text=text, voice=voice, rate="-10%")
                await communicate.save(out_path)

            asyncio.run(_synth())
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                raise ValueError("No audio data received.")
            logger.info("Text-to-speech conversion successful for %s.", profile.display)
            return out_path
        except Exception as e:
            if out_path:
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            logger.error("Error in text-to-speech conversion: %s", e)
            return ""

    def text_to_speech_batch(
        self,
        items: List[str],
        language: str = "es",
        voice_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_rounds: int = 3,
        retry_delay_s: float = 0.6,
    ) -> Dict[str, str]:
        if not items:
            return {}

        profile = get_language_profile(language)
        unique_items = list(dict.fromkeys(items))
        total = len(unique_items)
        results: Dict[str, str] = {}
        start = time.perf_counter()
        progress_step = 1 if total <= 20 else 5 if total <= 100 else 10

        def _synth(item: str) -> tuple[str, str]:
            return item, self.text_to_speech(item, language, voice_name)

        if progress_callback:
            try:
                progress_callback(0, total)
            except Exception:
                pass

        remaining = unique_items
        for round_idx in range(1, max_rounds + 1):
            if not remaining:
                break

            max_workers = min(MAX_TTS_WORKERS, len(remaining))
            if round_idx == 1:
                logger.info(
                    "Starting TTS generation for %d item(s) in %s using up to %d worker(s).",
                    total,
                    profile.display,
                    max_workers,
                )
            else:
                logger.warning(
                    "Retry round %d/%d for %d item(s) in %s (max %d worker(s)).",
                    round_idx,
                    max_rounds,
                    len(remaining),
                    profile.display,
                    max_workers,
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_synth, item): item for item in remaining}
                completed = 0
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        _, path = future.result()
                    except Exception as e:
                        logger.error("TTS task failed for '%s': %s", item, e)
                        path = ""

                    if path:
                        results[item] = path
                    completed += 1
                    if completed % progress_step == 0 or completed == len(remaining):
                        logger.info("TTS progress: %d/%d completed.", len(results), total)
                    if progress_callback:
                        try:
                            progress_callback(len(results), total)
                        except Exception:
                            pass

            remaining = [item for item in remaining if item not in results]
            if remaining and round_idx < max_rounds:
                time.sleep(retry_delay_s)
                retry_delay_s *= 2

        elapsed = time.perf_counter() - start
        if remaining:
            logger.warning("TTS missing for %d/%d item(s) after %d round(s).", len(remaining), total, max_rounds)
        logger.info("TTS generation finished: %d/%d succeeded in %.2fs.", len(results), total, elapsed)
        return results

    def play_audio(self, file_path: str) -> None:
        try:
            with self.audio_lock:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                logger.info("Playing audio.")

                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                pygame.mixer.music.unload()
                logger.info("Finished playing audio.")
        except Exception as e:
            logger.error("Error playing audio: %s", e)
            print(Fore.RED + "Error playing audio. Check logs for details.")
