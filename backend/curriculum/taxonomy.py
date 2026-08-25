"""The content taxonomy: what there is to learn, independent of any LLM.

Four tiers, deliberately shallow enough to browse and deep enough to be honest:

    Domain  (8)   broad area of life        e.g. "Food & Drink"
      Unit  (24)  a teachable topic         e.g. "Restaurant & ordering"
      Group (2-3) a slice inside the unit   e.g. "Ordering", "Paying"
      Item        one learnable card        e.g. "l'addition, s'il vous plait"

The tiers above the item are LANGUAGE-NEUTRAL: the same map of human life is
offered for French, Spanish, Russian, and Mandarin, so a learner who knows the
map in one language already knows it in the next. Only the leaves differ, and
those live in ``data/<lang>/<domain>.json``.

Units carry a CEFR level and the grammar feature ids (backend/grammar.py) they
exercise, which makes the taxonomy usable three ways:
  1. as a browsable library the learner picks from (no API key needed),
  2. as an ordered path (``learning_path``) when the learner just wants "next",
  3. as topic grounding for LLM generation when a key IS configured.
"""
from __future__ import annotations

from typing import Dict, List, Optional

CEFR_ORDER = ("A1", "A2", "B1", "B2", "C1", "C2")


def _u(uid: str, title: str, level: str, goal: str, keywords: List[str]) -> dict:
    """One unit: the leaf of the browsable tree, holder of the item groups."""
    return {"id": uid, "title": title, "level": level, "goal": goal, "keywords": keywords}


DOMAINS: List[dict] = [
    {
        "id": "foundations",
        "title": "Foundations",
        "blurb": "The first hundred words that make every later sentence possible.",
        "icon": "sparkles",
        "accent": "amber",
        "units": [
            _u("greetings", "Greetings & politeness", "A1",
               "Open and close a conversation with a stranger without stumbling.",
               ["hello", "please", "thank you", "goodbye", "sorry"]),
            _u("self-intro", "Introducing yourself", "A1",
               "Say who you are, where you are from, and what you do.",
               ["name", "nationality", "age", "job", "languages"]),
            _u("numbers-time", "Numbers, dates & time", "A1",
               "Handle prices, clock times, days, and dates in the wild.",
               ["numbers", "clock", "days", "months", "dates"]),
        ],
    },
    {
        "id": "people",
        "title": "People & Relationships",
        "blurb": "Talk about the humans in your life and how they make you feel.",
        "icon": "globe",
        "accent": "rose",
        "units": [
            _u("family", "Family & relationships", "A1",
               "Describe your family and who belongs to whom.",
               ["family", "partner", "children", "friends"]),
            _u("describing-people", "Describing people", "A2",
               "Describe appearance and character precisely enough to be recognised.",
               ["appearance", "character", "age", "adjectives"]),
            _u("feelings", "Feelings & small talk", "A2",
               "Say how you actually feel, and keep a light conversation alive.",
               ["emotions", "moods", "small talk", "reactions"]),
        ],
    },
    {
        "id": "daily-life",
        "title": "Daily Life",
        "blurb": "Home, routine, and the everyday transactions of an ordinary week.",
        "icon": "home",
        "accent": "sky",
        "units": [
            _u("home", "Home & living space", "A1",
               "Name the rooms and objects around you, and describe where you live.",
               ["rooms", "furniture", "housing"]),
            _u("routines", "Daily routine & chores", "A2",
               "Narrate an ordinary day, including the boring parts.",
               ["routine", "chores", "frequency", "housework"]),
            _u("shopping-money", "Shopping, clothes & money", "A2",
               "Buy things, ask for another size, and talk about what it costs.",
               ["shopping", "clothes", "prices", "payment"]),
        ],
    },
    {
        "id": "food",
        "title": "Food & Drink",
        "blurb": "The most useful domain in any language: eating and drinking well.",
        "icon": "star",
        "accent": "orange",
        "units": [
            _u("cafe-bar", "Cafe & bar", "A1",
               "Order a drink and a snack, and pay, in under a minute.",
               ["coffee", "drinks", "snacks", "counter"]),
            _u("restaurant", "Restaurant & ordering", "A2",
               "Book a table, read a menu, order a full meal, and ask for the bill.",
               ["menu", "courses", "booking", "bill"]),
            _u("cooking-groceries", "Groceries & cooking", "A2",
               "Shop for ingredients and describe how a dish is made.",
               ["market", "ingredients", "quantities", "recipes"]),
        ],
    },
    {
        "id": "getting-around",
        "title": "Getting Around",
        "blurb": "Movement through a city and a country, from directions to hotels.",
        "icon": "arrowRight",
        "accent": "teal",
        "units": [
            _u("directions", "Directions & the city", "A1",
               "Ask for the way, understand the answer, and name what you pass.",
               ["directions", "places", "city", "prepositions"]),
            _u("transport", "Transport & tickets", "A2",
               "Buy a ticket, catch the right train, and survive a delay.",
               ["train", "bus", "tickets", "delays"]),
            _u("travel-stay", "Travel & accommodation", "B1",
               "Book a room, check in, complain politely, and check out.",
               ["hotel", "booking", "luggage", "complaints"]),
        ],
    },
    {
        "id": "health-body",
        "title": "Health & Body",
        "blurb": "The vocabulary you hope not to need, and are grateful to have.",
        "icon": "target",
        "accent": "emerald",
        "units": [
            _u("body-basics", "The body & basic needs", "A1",
               "Name body parts and say what hurts or what you need.",
               ["body", "pain", "needs", "tired"]),
            _u("doctor-pharmacy", "Doctor & pharmacy", "B1",
               "Describe symptoms accurately and understand the instructions you get.",
               ["symptoms", "medicine", "appointment", "prescription"]),
            _u("wellbeing-sport", "Wellbeing & sport", "B1",
               "Talk about exercise, sleep, stress, and staying in shape.",
               ["sport", "exercise", "sleep", "stress"]),
        ],
    },
    {
        "id": "work-study",
        "title": "Work & Study",
        "blurb": "The register of offices, classrooms, screens, and formal writing.",
        "icon": "book",
        "accent": "indigo",
        "units": [
            _u("jobs", "Jobs & workplaces", "A2",
               "Say what you do for a living and ask others about theirs.",
               ["jobs", "workplace", "colleagues", "schedule"]),
            _u("office-email", "Meetings, phone & email", "B1",
               "Run a call, write a clean email, and use the polite formulas that matter.",
               ["email", "phone", "meetings", "formal"]),
            _u("study-tech", "Study & technology", "B1",
               "Talk about learning, devices, software, and everything going wrong online.",
               ["study", "computer", "internet", "problems"]),
        ],
    },
    {
        "id": "world-ideas",
        "title": "World & Ideas",
        "blurb": "Weather, culture, and the abstract language of opinion and argument.",
        "icon": "zap",
        "accent": "violet",
        "units": [
            _u("weather-nature", "Weather & nature", "A2",
               "Discuss weather, seasons, and landscape like a local.",
               ["weather", "seasons", "nature", "climate"]),
            _u("culture-media", "Culture, media & free time", "B1",
               "Recommend a film, describe a book, and talk about what you do for fun.",
               ["film", "music", "books", "hobbies"]),
            _u("opinions-debate", "Opinions & argument", "B2",
               "Agree, disagree, concede a point, and structure a real argument.",
               ["opinion", "agreement", "argument", "nuance"]),
        ],
    },
]

DOMAIN_INDEX: Dict[str, dict] = {d["id"]: d for d in DOMAINS}

UNIT_INDEX: Dict[str, dict] = {
    unit["id"]: {**unit, "domain": domain["id"], "domainTitle": domain["title"]}
    for domain in DOMAINS
    for unit in domain["units"]
}


def unit_meta(unit_id: str) -> Optional[dict]:
    return UNIT_INDEX.get(unit_id)


def learning_path() -> List[str]:
    """Unit ids in recommended study order: by CEFR level, then by domain order.

    Within a level the domains stay in taxonomy order, so a learner following
    the path alternates between areas of life instead of grinding one domain.
    """
    return [
        unit_id
        for level in CEFR_ORDER
        for unit_id, meta in UNIT_INDEX.items()
        if meta["level"] == level
    ]


def taxonomy_payload() -> List[dict]:
    """The tree without item content, for the client-side library browser."""
    return [
        {
            "id": domain["id"],
            "title": domain["title"],
            "blurb": domain["blurb"],
            "icon": domain["icon"],
            "accent": domain["accent"],
            "units": [dict(unit) for unit in domain["units"]],
        }
        for domain in DOMAINS
    ]
