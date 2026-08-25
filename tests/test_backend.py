"""Backend unit tests (no network calls).

Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from backend import config, content, tts  # noqa: E402
from backend.grammar import CEFR_ORDER, GRAMMAR, feature_index, grammar_profile, prompt_brief  # noqa: E402
from backend.languages import get_language, normalize_language_code, LANGUAGES  # noqa: E402
from backend.llm import extract_json, resolve_model_override  # noqa: E402
from backend.models import CompositionPack, LessonPack  # noqa: E402


class TestLanguages(unittest.TestCase):
    def test_default_language_is_french(self):
        self.assertEqual(config.DEFAULT_LANGUAGE, "fr")
        self.assertEqual(normalize_language_code(None), "fr")
        self.assertEqual(normalize_language_code("nonsense"), "fr")

    def test_aliases(self):
        self.assertEqual(normalize_language_code("French"), "fr")
        self.assertEqual(normalize_language_code("ESPAÑOL"), "es")
        self.assertEqual(normalize_language_code("zh-CN"), "zh")

    def test_supported_set(self):
        self.assertEqual(set(LANGUAGES), {"fr", "es", "ru", "zh"})

    def test_profiles_complete(self):
        for code, profile in LANGUAGES.items():
            self.assertTrue(profile.voices, f"{code} has voices")
            self.assertIn(profile.default_voice, {v.id for v in profile.voices})
            self.assertTrue(profile.recognition_locale)
            genders = {v.gender for v in profile.voices}
            self.assertEqual(genders, {"f", "m"}, f"{code} needs both voice genders for dialogues")


class TestGrammar(unittest.TestCase):
    def test_every_language_has_full_architecture(self):
        for code in LANGUAGES:
            profile = grammar_profile(code)
            self.assertTrue(profile["overview"])
            self.assertEqual(len(profile["pillars"]), 6, code)
            self.assertEqual(set(profile["roadmap"]), set(CEFR_ORDER), code)
            for level, features in profile["roadmap"].items():
                self.assertGreaterEqual(len(features), 4, f"{code} {level}")
                for feature in features:
                    for key in ("id", "name", "tip", "example", "example_en"):
                        self.assertTrue(feature[key], f"{code} {feature.get('id')} missing {key}")
            self.assertTrue(profile["challenges"])
            self.assertTrue(profile["phonology"])

    def test_feature_ids_unique_and_prefixed(self):
        for code in LANGUAGES:
            index = feature_index(code)
            total = sum(len(v) for v in GRAMMAR[code]["roadmap"].values())
            self.assertEqual(len(index), total, f"{code} has duplicate feature ids")
            for fid in index:
                self.assertTrue(fid.startswith(f"{code}-"), fid)

    def test_prompt_brief_mentions_targets(self):
        brief = prompt_brief("fr", "B1")
        self.assertIn("TARGET structures", brief)
        self.assertIn("fr-imparfait-vs-pc", brief)
        self.assertIn("Passé composé", brief)  # known structure listed as usable

    def test_prompt_brief_bad_level_defaults(self):
        self.assertIn("TARGET structures", prompt_brief("zh", "weird"))


class TestJsonExtraction(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_object(self):
        self.assertEqual(extract_json('```json\n{"a": "é"}\n```'), {"a": "é"})

    def test_prose_wrapped(self):
        self.assertEqual(extract_json('Here you go: {"items": []} hope it helps'), {"items": []})

    def test_trailing_garbage(self):
        self.assertEqual(extract_json('{"a": 1}}}'), {"a": 1})

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            extract_json("[1, 2]")
        with self.assertRaises(ValueError):
            extract_json("no json here")
        with self.assertRaises(ValueError):
            extract_json("")


class TestContent(unittest.TestCase):
    def test_normalize_level(self):
        self.assertEqual(content.normalize_level("b1"), "B1")
        self.assertEqual(content.normalize_level("weird"), "A2")
        self.assertEqual(content.normalize_level(None), "A2")

    def test_seed_lesson_validates(self):
        pack = content.seed_lesson("fr", 12)
        self.assertIsNotNone(pack)
        validated = LessonPack.model_validate(pack)
        self.assertEqual(len(validated.items), 12)
        for item in validated.items:
            self.assertTrue(item.target and item.english)

    def test_seed_composition_validates(self):
        pack = content.seed_composition("fr")
        self.assertIsNotNone(pack)
        validated = CompositionPack.model_validate(pack)
        self.assertIn(validated.format, ("dialogue", "monologue", "story", "article"))
        index = feature_index("fr")
        for spotlight in validated.grammar_spotlights:
            self.assertIn(spotlight.feature, index)
            self.assertTrue(any(spotlight.excerpt in s.text for s in validated.segments),
                            f"excerpt not found verbatim: {spotlight.excerpt}")
        for question in validated.questions:
            self.assertEqual(len(question.choices), 4)
            self.assertTrue(0 <= question.correct_choice <= 3)

    def test_seed_missing_language(self):
        self.assertIsNone(content.seed_lesson("es", 10))
        self.assertIsNone(content.seed_composition("ru"))


class TestTTS(unittest.TestCase):
    def test_rate_presets(self):
        self.assertEqual(tts.resolve_rate("slow"), "-30%")
        self.assertEqual(tts.resolve_rate("study"), "-12%")
        self.assertEqual(tts.resolve_rate(None), "-12%")
        self.assertEqual(tts.resolve_rate("+20%"), "+20%")
        self.assertEqual(tts.resolve_rate("garbage"), "-12%")

    def test_cache_key_stable_and_distinct(self):
        a = tts.cache_key("bonjour", "fr-FR-DeniseNeural", "-12%")
        b = tts.cache_key("bonjour", "fr-FR-DeniseNeural", "-12%")
        c = tts.cache_key("bonjour", "fr-FR-HenriNeural", "-12%")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            tts.synthesize("   ", "fr")

    def test_voice_fallback(self):
        language = get_language("fr")
        self.assertIn(language.default_voice, {v.id for v in language.voices})


class TestEmDashPolicy(unittest.TestCase):
    def test_strip_em_dashes_walks_structures(self):
        dirty = {"a": "one \u2014 two", "b": ["x\u2014y", {"c": "clean"}], "d": 5}
        clean = content.strip_em_dashes(dirty)
        self.assertEqual(clean["a"], "one - two")
        self.assertEqual(clean["b"][0], "x-y")
        self.assertEqual(clean["b"][1]["c"], "clean")
        self.assertEqual(clean["d"], 5)

    def test_no_em_dashes_in_seed_content(self):
        for name in ("fr_core.json", "fr_composition.json"):
            text = (config.SEED_DIR / name).read_text(encoding="utf-8")
            self.assertNotIn("\u2014", text, name)


class TestKeySetup(unittest.TestCase):
    def setUp(self):
        self._original = config.OPENROUTER_API_KEY

    def tearDown(self):
        config.OPENROUTER_API_KEY = self._original

    def test_valid_key_written_and_other_lines_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text('OTHER_SETTING="keep me"\nOPENROUTER_API_KEY="sk-or-old"\n')
            config.set_openrouter_key("sk-or-v1-" + "a" * 32, env_path=env_path)
            text = env_path.read_text()
            self.assertIn('OTHER_SETTING="keep me"', text)
            self.assertNotIn("sk-or-old", text)
            self.assertIn("sk-or-v1-" + "a" * 32, text)
            self.assertEqual(config.active_provider(), "openrouter")
            self.assertTrue(config.masked_key().startswith("sk-or-v1-"))
            self.assertNotIn("a" * 32, config.masked_key())

    def test_invalid_keys_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            for bad in ("", "hello", "sk-something", "sk-or-short", "sk-or- with space " + "x" * 30):
                with self.assertRaises(ValueError):
                    config.set_openrouter_key(bad, env_path=env_path)
            self.assertFalse(env_path.exists())


class TestConfig(unittest.TestCase):
    def test_provider_chain_values(self):
        self.assertIn(config.active_provider(), {"openrouter", "openai", "offline"})

    def test_openrouter_model_configured(self):
        self.assertTrue(config.OPENROUTER_MODEL)

    def test_model_choices_curated(self):
        ids = [c["id"] for c in config.MODEL_CHOICES]
        self.assertIn("deepseek/deepseek-v4-flash-0731", ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_model_override_validation(self):
        self.assertEqual(resolve_model_override("openai/gpt-5-mini", "openrouter"), "openai/gpt-5-mini")
        self.assertIsNone(resolve_model_override("openai/gpt-5-mini", "openai"))
        self.assertIsNone(resolve_model_override("rm -rf /", "openrouter"))
        self.assertIsNone(resolve_model_override("", "openrouter"))
        self.assertIsNone(resolve_model_override(None, "openrouter"))


if __name__ == "__main__":
    unittest.main()
