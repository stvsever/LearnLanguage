"""The CEFR level scale: one table, used by every layer.

The order lived in three modules (content, grammar, curriculum.taxonomy) and the
prompt guidance in a fourth place, so a level could be described one way to the
model and another way in the interface. This is now the single source.

Each level carries:
  ``name``      a plain-English adjective, because "A2" tells a learner nothing
                on its own. This is what the interface shows next to the code.
  ``blurb``     one line on what you can actually do at that level.
  ``guidance``  the instruction handed to the model when generating at it.
"""
from __future__ import annotations

from typing import List, Optional

CEFR_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")

LEVELS = {
    "A1": {
        "name": "Beginner",
        "blurb": "First words and fixed phrases: greet, order, ask where things are.",
        "guidance": "Absolute beginner. Very high-frequency words and fixed chunks. Simple present-oriented sentences under 8 words.",
    },
    "A2": {
        "name": "Elementary",
        "blurb": "Everyday exchanges: shopping, travel, describing your routine and your past.",
        "guidance": "Elementary. Everyday topics, first past-tense forms, short compound sentences.",
    },
    "B1": {
        "name": "Intermediate",
        "blurb": "Opinions, plans, and stories across past, present, and future.",
        "guidance": "Intermediate. Opinions, plans, narration across time frames. Natural connectors.",
    },
    "B2": {
        "name": "Upper intermediate",
        "blurb": "Abstract subjects and real argument, with idiom and register under control.",
        "guidance": "Upper-intermediate. Abstract topics, idiomatic collocations, register contrasts.",
    },
    "C1": {
        "name": "Advanced",
        "blurb": "Nuance, implication, and low-frequency vocabulary in complex syntax.",
        "guidance": "Advanced. Nuanced idiom, low-frequency vocabulary, complex syntax.",
    },
    "C2": {
        "name": "Mastery",
        "blurb": "Native-like range: subtle register, literary and technical alike.",
        "guidance": "Mastery. Native-like idiom, subtle register, literary and technical range.",
    },
}

# Kept as a mapping for the generation prompts, which only need the instruction.
LEVEL_GUIDANCE = {code: meta["guidance"] for code, meta in LEVELS.items()}

DEFAULT_LEVEL = "A2"


def normalize_level(value: Optional[str]) -> str:
    raw = str(value or "").strip().upper()
    return raw if raw in LEVELS else DEFAULT_LEVEL


def level_index(value: Optional[str]) -> int:
    return CEFR_ORDER.index(normalize_level(value))


def level_name(value: Optional[str]) -> str:
    return LEVELS[normalize_level(value)]["name"]


def public_level_payload() -> List[dict]:
    """Levels as the client needs them: code plus the human-readable half."""
    return [
        {"code": code, "name": LEVELS[code]["name"], "blurb": LEVELS[code]["blurb"]}
        for code in CEFR_ORDER
    ]
