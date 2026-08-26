"""Backend unit tests (no network calls).

Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from backend import config, content, curriculum, levels, tts  # noqa: E402
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

    def test_starter_lesson_validates_for_every_language(self):
        for code in LANGUAGES:
            pack = content.seed_lesson(code, 12)
            self.assertIsNotNone(pack, code)
            validated = LessonPack.model_validate(pack)
            self.assertEqual(len(validated.items), 12, code)
            for item in validated.items:
                self.assertTrue(item.target and item.english)

    def test_seed_composition_validates_for_every_language(self):
        for code in LANGUAGES:
            pack = content.seed_composition(code)
            self.assertIsNotNone(pack, code)
            validated = CompositionPack.model_validate(pack)
            self.assertIn(validated.format, ("dialogue", "monologue", "story", "article"))
            index = feature_index(code)
            for spotlight in validated.grammar_spotlights:
                self.assertIn(spotlight.feature, index)
                self.assertTrue(any(spotlight.excerpt in s.text for s in validated.segments),
                                f"{code}: excerpt not found verbatim: {spotlight.excerpt}")
            for question in validated.questions:
                self.assertEqual(len(question.choices), 4)
                self.assertTrue(0 <= question.correct_choice <= 3)

    def test_unknown_language_falls_back_to_the_default(self):
        pack = content.seed_lesson("klingon", 8)
        self.assertIsNotNone(pack)
        self.assertEqual(pack["language"], config.DEFAULT_LANGUAGE)

    def test_curriculum_lesson_shapes_like_a_generated_pack(self):
        pack = content.curriculum_lesson("fr", "cafe-bar")
        self.assertIsNotNone(pack)
        LessonPack.model_validate(pack)
        self.assertEqual(pack["unit"], "cafe-bar")
        self.assertEqual(pack["source"], "curriculum")

    def test_curriculum_lesson_unknown_unit(self):
        self.assertIsNone(content.curriculum_lesson("fr", "no-such-unit"))

    def test_offline_lesson_always_returns_material(self):
        """The learner must never be left with nothing, key or no key."""
        original = (config.OPENROUTER_API_KEY, config.OPENAI_API_KEY)
        config.OPENROUTER_API_KEY = ""
        config.OPENAI_API_KEY = ""
        try:
            logging.disable(logging.WARNING)
            self.assertEqual(config.active_provider(), "offline")
            for topic, expected in (("ordering a coffee at a cafe", "Cafe & bar"),
                                    ("total nonsense xyzzy", "Core starter set")):
                pack = content.generate_lesson(topic, "es", "A2", 12)
                self.assertEqual(pack["topic"], expected)
                self.assertEqual(len(pack["items"]), 12)
                LessonPack.model_validate(pack)
            unit_pack = content.generate_lesson("", "ru", "A2", 12, unit="doctor-pharmacy")
            self.assertEqual(unit_pack["unit"], "doctor-pharmacy")
            self.assertEqual(unit_pack["source"], "curriculum")
        finally:
            config.OPENROUTER_API_KEY, config.OPENAI_API_KEY = original
            logging.disable(logging.NOTSET)

    def test_normalize_format(self):
        self.assertEqual(content.normalize_format("dialogue"), "dialogue")
        self.assertEqual(content.normalize_format("STORY"), "story")
        self.assertIsNone(content.normalize_format("auto"))
        self.assertIsNone(content.normalize_format(None))


class TestCurriculum(unittest.TestCase):
    def test_every_language_covers_the_whole_taxonomy(self):
        for code in LANGUAGES:
            summary = curriculum.summary(code)
            self.assertEqual(summary["units"], len(curriculum.UNIT_INDEX), code)
            self.assertEqual(summary["domains"], len(curriculum.DOMAINS), code)
            self.assertGreaterEqual(summary["items"], 200, code)

    def test_items_are_complete_and_card_ready(self):
        for code in LANGUAGES:
            for unit_id in curriculum.UNIT_INDEX:
                items = curriculum.unit_items(code, unit_id)
                self.assertTrue(items, f"{code}/{unit_id} has no items")
                for item in items:
                    for field in ("target", "english", "pronunciation", "example", "example_en"):
                        self.assertTrue(item[field], f"{code}/{unit_id}: {field} empty on {item['target']}")
                    self.assertEqual(item["unit"], unit_id)

    def test_targets_are_unique_within_a_language(self):
        """Cards are keyed by target, so a repeat across units would silently
        collapse into one card and break per-unit progress."""
        for code in LANGUAGES:
            seen = {}
            for unit_id in curriculum.UNIT_INDEX:
                for item in curriculum.unit_items(code, unit_id):
                    key = item["target"].lower().strip()
                    self.assertNotIn(key, seen, f"{code}: {key!r} in both {unit_id} and {seen.get(key)}")
                    seen[key] = unit_id

    def test_unit_grammar_ids_exist(self):
        for code in LANGUAGES:
            index = feature_index(code)
            for unit_id in curriculum.UNIT_INDEX:
                detail = curriculum.unit_detail(code, unit_id)
                for fid in detail["grammar"]:
                    self.assertIn(fid, index, f"{code}/{unit_id}")

    def test_learning_path_is_level_ordered_and_complete(self):
        path = curriculum.learning_path()
        self.assertEqual(sorted(path), sorted(curriculum.UNIT_INDEX))
        levels = [curriculum.UNIT_INDEX[u]["level"] for u in path]
        order = [curriculum.CEFR_ORDER.index(lv) for lv in levels]
        self.assertEqual(order, sorted(order), "path must not go backwards in level")

    def test_tree_reports_availability(self):
        tree = curriculum.tree("fr")
        self.assertEqual(len(tree), len(curriculum.DOMAINS))
        for domain in tree:
            self.assertTrue(domain["units"])
            for unit in domain["units"]:
                self.assertTrue(unit["available"], unit["id"])
                self.assertGreater(unit["itemCount"], 0)

    def test_search_finds_units_and_items(self):
        results = curriculum.search("fr", "cafe")
        self.assertTrue(any(u["id"] == "cafe-bar" for u in results["units"]))
        self.assertTrue(results["items"])

    def test_search_is_accent_insensitive(self):
        with_accent = curriculum.search("fr", "café")["items"]
        without = curriculum.search("fr", "cafe")["items"]
        self.assertEqual([i["target"] for i in with_accent], [i["target"] for i in without])

    def test_search_ignores_one_character_queries(self):
        self.assertEqual(curriculum.search("fr", "a"), {"query": "a", "units": [], "items": []})

    def test_unknown_language_has_no_content(self):
        self.assertEqual(curriculum.unit_items("xx", "greetings"), [])
        self.assertIsNone(curriculum.unit_detail("xx", "greetings"))
        self.assertIsNone(curriculum.starter_pack("xx"))

    def test_topic_matching_routes_to_the_right_unit(self):
        cases = [
            ("es", "ordering a coffee at a cafe", "cafe-bar"),
            ("fr", "how to book a hotel room and complain", "travel-stay"),
            ("ru", "talking about the weather in winter", "weather-nature"),
            ("zh", "job interview and office work", "jobs"),
            ("fr", "asking for directions in town", "directions"),
        ]
        for code, topic, expected in cases:
            self.assertEqual(curriculum.best_unit_for_topic(code, topic), expected, topic)

    def test_topic_matching_rejects_noise(self):
        self.assertIsNone(curriculum.best_unit_for_topic("fr", "xyzzy nonsense"))
        self.assertIsNone(curriculum.best_unit_for_topic("fr", ""))
        # Stopwords alone must never be enough to claim a unit.
        self.assertIsNone(curriculum.best_unit_for_topic("fr", "the and with about"))

    def test_starter_pack_spans_domains(self):
        for code in LANGUAGES:
            pack = curriculum.starter_pack(code, 24)
            self.assertEqual(len(pack["items"]), 24, code)
            self.assertEqual({item["level"] for item in pack["items"]}, {"A1"}, code)
            domains = {curriculum.UNIT_INDEX[i["unit"]]["domain"] for i in pack["items"]}
            self.assertGreaterEqual(len(domains), 4, f"{code} starter set is too narrow")


class TestLevels(unittest.TestCase):
    def test_every_level_has_a_name_a_blurb_and_guidance(self):
        self.assertEqual(tuple(levels.LEVELS), levels.CEFR_ORDER)
        for code, meta in levels.LEVELS.items():
            for key in ("name", "blurb", "guidance"):
                self.assertTrue(meta[key], f"{code} missing {key}")
            self.assertNotEqual(meta["name"], code, f"{code} name must be words, not the code")

    def test_one_table_feeds_every_module(self):
        """The order lived in three modules and drifted; assert it cannot again."""
        from backend import grammar
        self.assertIs(grammar.CEFR_ORDER, levels.CEFR_ORDER)
        self.assertIs(curriculum.CEFR_ORDER, levels.CEFR_ORDER)
        self.assertIs(content.CEFR_LEVELS, levels.CEFR_ORDER)
        self.assertEqual(set(content.LEVEL_GUIDANCE), set(levels.CEFR_ORDER))

    def test_normalisation_and_ordering(self):
        self.assertEqual(levels.normalize_level("b1"), "B1")
        self.assertEqual(levels.normalize_level("nonsense"), levels.DEFAULT_LEVEL)
        self.assertEqual(levels.normalize_level(None), levels.DEFAULT_LEVEL)
        self.assertEqual(levels.level_index("A1"), 0)
        self.assertLess(levels.level_index("A2"), levels.level_index("C1"))

    def test_public_payload_shape(self):
        payload = levels.public_level_payload()
        self.assertEqual([row["code"] for row in payload], list(levels.CEFR_ORDER))
        for row in payload:
            self.assertEqual(set(row), {"code", "name", "blurb"})

    def test_every_curriculum_unit_sits_on_the_scale(self):
        for unit_id, meta in curriculum.UNIT_INDEX.items():
            self.assertIn(meta["level"], levels.CEFR_ORDER, unit_id)


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
        for path in sorted(config.SEED_DIR.glob("*.json")):
            self.assertNotIn("\u2014", path.read_text(encoding="utf-8"), path.name)

    def test_no_em_dashes_in_curriculum_content(self):
        for path in sorted(curriculum.DATA_DIR.glob("*/*.json")):
            self.assertNotIn("\u2014", path.read_text(encoding="utf-8"), str(path))


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
