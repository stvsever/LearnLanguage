<div align="center">

# AI-Studio for Learning Languages

[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red)](https://github.com/stvsever/LearnLanguage)
[![Local First](https://img.shields.io/badge/Local-First-2ea44f)](https://github.com/stvsever/LearnLanguage#-privacy)
[![LLM Agnostic](https://img.shields.io/badge/LLM-Agnostic-8A2BE2)](https://openrouter.ai/models)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)

**A local-first, science-based language tutor that runs on your machine.** 🧠

A curated topic library of 1,200 hand-written items, spaced repetition, active recall, listening, speaking, free-form composed input, and an explicit grammar map: one continuous learning loop, with all progress stored privately in your browser.

Built for 🇫🇷 **French** (default), 🇪🇸 **Spanish**, 🇷🇺 **Russian**, and 🇨🇳 **Mandarin**. The interface is in English.

![AI-Studio for Learning Languages dashboard](docs/images/learnlanguage-interface.png)

</div>

## 🚀 Quickstart

```bash
git clone https://github.com/stvsever/LearnLanguage.git
cd LearnLanguage
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py --open
```

The app opens at `http://127.0.0.1:8765`. A first-run walkthrough shows you around, and the app itself guides you through connecting a free [OpenRouter](https://openrouter.ai/keys) key: paste it once in the interface and it saves automatically, no restart needed.

**No key is required to learn.** All four languages ship a complete, hand-written curriculum: 24 topics, 300 items each, with pronunciation, example sentences, and usage traps. A key adds unlimited *extra* generation on any topic you can describe. Audio always works: free neural voices with a browser fallback.

🐳 Prefer Docker?

```bash
docker build -t learnlanguage .
docker run -p 8765:8765 learnlanguage
```

## 🔁 How you learn with it

A session follows one loop, and the dashboard always points at the next step:

1. **Review** clears the cards the scheduler predicts you are about to forget.
2. **Learn** introduces new vocabulary through a guided ladder: see and hear the item in context, recognize its meaning, then type it from memory.
3. **Topics** is where that vocabulary comes from: pick a topic from the curated library, or generate one.
4. **Compose, Listen, or Speak** turns knowledge into skill with real input and output.

Ten focused minutes a day beats a weekend marathon; the streak and heatmap keep you honest. 🔥

## 🗂️ The rooms

| Room | What happens there |
| --- | --- |
| 🏠 **Home** | The single best next action, streak, weekly activity, deck pipeline, and a rotating "today's structure" nudge |
| 🗂️ **Topics** | The curated library: 8 areas of life, 24 topics, 300 items per language, browsable and addable without any AI key |
| ✨ **Learn** | The guided encoding ladder, plus lesson generation on any topic at your CEFR level |
| 🔄 **Review** | The FSRS spaced-repetition queue with four-grade rating, real interval previews, and exercises that get harder as memories get stronger |
| 🎧 **Listen** | Dictation and sound-discrimination drills built from your own deck |
| 🎙️ **Speak** | Pronunciation practice scored word by word with free on-device speech recognition |
| ✍️ **Compose** | Describe anything; one model call picks the best format and writes it at your level (see below) |
| 📖 **Grammar** | The structural map of your language: pillars, a CEFR roadmap, transfer traps, phonology |
| 📈 **Progress** | Vocabulary growth, live memory strength, grammar coverage, study mix, consistency, leeches |

## 🗂️ Topics: a taxonomy of what there is to learn

The library is a four-tier tree, and the tiers above the leaf are the same in every language, so learning the map once pays off four times:

```text
Topics
└── Domain          8 broad areas of life        "Food & Drink"
    └── Unit        24 teachable topics, CEFR    "Restaurant & ordering" (A2)
        └── Group   slices inside the unit       "Getting a table"
            └── Item  one card                   "la carte" (the menu, not le menu)
```

Every one of the 1,200 items is hand-written, with IPA (or pinyin with tone marks), a natural example sentence and its translation, and a note only where there is a real trap: a false friend, an irregular form, a register clash, a grammar point. Units also declare the grammar features they exercise, which links the library straight into the Grammar map.

Browse by area, filter by CEFR level, search across every topic and word at once, tick exactly the items you want, and they flow into the same spaced-repetition pipeline as anything the model writes. Per-unit progress rings show how much of each topic is already in your deck. With a key connected, any unit can also be extended with fresh generated items that stay inside its scope and skip everything you already have.

## ✍️ Compose: describe it, get it

Type any idea: a scene, a rant, a voicemail, a news piece. Leave everything on Auto and a single LLM call classifies the best presentation format (dialogue, monologue, story, or article) and writes it at your level, deliberately weaving in the grammar structures you are currently learning. Or take control: pin the format, the register (casual, neutral, formal), the number of speakers, up to four grammar structures that must appear, and the vocabulary to weave in, drawn from the cards due for review. While it works, the interface shows every underlying step, from format classification to validation.

![Compose: a generated dialogue](docs/images/learnlanguage-compose.png)

Every composition arrives with per-segment audio (dialogues get a distinct neural voice per speaker), tap-any-word contextual glosses, aligned translations, grammar spotlights (verbatim excerpts showing your target structures in action), and comprehension questions. One tap adds any word to your spaced-repetition deck. Pieces are saved to a local library, because rereading familiar input is some of the highest-value practice there is.

## 📖 Grammar as a first-class object

Each language ships a structured model of what there is to learn and in what order, grounded in its typology: six pillars (the mental model), a CEFR-staged roadmap of about thirty features with playable examples, the classic traps for English speakers, and the sound system.

![Grammar roadmap](docs/images/learnlanguage-grammar.png)

This one model does three jobs:

1. **Steers generation.** Lessons and compositions receive a grammar brief: which structures you own, which to target. Input lands just above your level by construction.
2. **Is the reference.** The Grammar view renders the whole map, and every feature has a one-tap "Practice" that pre-fills Compose.
3. **Tracks coverage.** Generated content reports which features it actually used; Progress shows how much of each level you have genuinely met.

The four models in brief: French (fusional Romance: gender and agreement, the verb engine, clitic pronouns, liaison), Spanish (pro-drop, ser/estar, two past aspects, the living subjunctive), Russian (six cases, aspect pairs, motion verbs, word order as information structure), Mandarin (tones, topic-prominence, aspect particles, measure words, complements).

## 🧠 The learning science inside

| Principle | Implementation |
| --- | --- |
| Spaced repetition | An FSRS scheduler models every card's memory state and times reviews to a configurable target retention (default 90%) |
| Active recall | Every exposure is a test; the modality escalates with card maturity: recognition, typed production, listening transcription, cloze in context |
| Rich encoding | New items pair orthography, pronunciation, audio, and context, then get retrieved twice within the first minute |
| Comprehensible input | Compositions are written from a brief of exactly what you know, plus a thin layer of inferable new material |
| Output and noticing | Speaking practice surfaces the gap between what you meant to say and what a recognizer heard |
| Nuanced feedback | Diacritic-aware verdicts, typo tolerance, and character-level diffs at the moment of error |
| Consolidation | Daily streaks, sane default limits, and a consistency heatmap, because memories consolidate during sleep |

## 🏗️ Architecture

```text
app.py                  # zero-framework HTTP server (Python stdlib) + JSON API
backend/
├── config.py           # env, paths, provider selection, in-app key setup
├── languages.py        # language profiles: voices, scripts, recognition locales
├── grammar.py          # per-language grammar model -> prompts + UI + progress
├── llm.py              # provider chain, schema-validated JSON generation with
│                       #   repair retry, reasoning-mode handling, model override
├── content.py          # lessons, format-classified compositions, glosses
├── models.py           # pydantic schemas; all LLM output is validated
├── tts.py              # neural text-to-speech with a content-addressed disk cache
├── curriculum/
│   ├── taxonomy.py     # the language-neutral tree: domains, units, CEFR, path
│   └── data/<lang>/    # 1,200 hand-written items, one file per domain
└── seed/               # a curated composition per language (offline mode)
static/
├── css/                # design system: tokens, components, views (light + dark)
└── js/
    ├── srs.js          # FSRS scheduler
    ├── grading.js      # normalization, edit distance, diffs, verdicts
    ├── store.js        # localStorage state, migrations, grammar tracking
    ├── audio.js        # TTS client, per-speaker voices, Web Speech fallback
    ├── keysetup.js     # in-app OpenRouter key onboarding (paste, auto-save)
    └── views/          # home, topics, learn, review, listen, speak,
                        #   compose, grammar, progress, settings
tests/                  # Python unit tests + Node tests for the JS engines
```

Design decisions:

- **No build step.** The frontend is native ES modules: open the app, edit a file, refresh.
- **Local-first.** All learner data lives in `localStorage`, exportable as JSON from Settings. The server only generates content and audio.
- **The LLM is optional.** The curriculum is the spine; generation is the extension. When a key is missing or a call fails, a free-form topic is matched to the closest curated unit rather than returning nothing. [OpenRouter](https://openrouter.ai) first (default model `deepseek/deepseek-v4-flash-0731`, switchable in Settings to a curated list or any custom slug), OpenAI as fallback. Every response is validated against pydantic schemas with one repair retry.
- **Free TTS.** Microsoft Edge neural voices via `edge-tts`, cached on disk, with the browser's Web Speech API as an offline fallback.
- **Honest feedback.** Long-running generation shows its underlying steps and then reports exactly what landed in your deck: added, already present, skipped. Generation failures and storage failures are separate paths, so a text that arrived is never lost because one item was malformed.
- **Routable state.** Every view is a hash route, and the library carries its position (`#/topics/food/restaurant`), so the back button, bookmarks, and deep links all work.

## ⚙️ Configuration

The easiest path is in-app: **Settings → AI model** connects your OpenRouter key (it saves to the local `.env` automatically) and lets you pick any model. Environment variables for advanced setups:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | none | Enables lesson, composition, and gloss generation |
| `LEARNLANGUAGE_MODEL` | `deepseek/deepseek-v4-flash-0731` | Server-default model; users can override in Settings |
| `OPENAI_API_KEY` | none | Fallback provider when no OpenRouter key is set |
| `LEARNLANGUAGE_OPENAI_MODEL` | `gpt-4o-mini` | Model for the OpenAI fallback |
| `LEARNLANGUAGE_DEFAULT_LANGUAGE` | `fr` | Language preselected on first run |
| `LEARNLANGUAGE_LLM_TIMEOUT` | `120` | Per-request LLM timeout in seconds |

## 🧪 Tests

```bash
# Backend: grammar model, curriculum integrity, schemas, offline fallbacks,
# key setup (47 tests, no network)
python -m unittest discover -s tests -v

# Frontend engines: FSRS scheduler, grading, deck store (27 tests, Node 20+)
node --test tests/*.test.mjs
```

The curriculum tests are the load-bearing ones: they assert that every language
covers the whole taxonomy, that no item is missing a field, that no card target
repeats across units (which would silently collapse two cards into one), and
that every declared grammar id exists in the grammar model.

## 🔒 Privacy

- Your API key is stored only in the local `.env` on your machine; `.env` is gitignored.
- Learner data never leaves your machine except for the text sent to the LLM and TTS providers you configure.
- Generated audio and caches live in `runtime/`, which is gitignored.

## 📄 License

MIT, see [LICENSE](LICENSE).
