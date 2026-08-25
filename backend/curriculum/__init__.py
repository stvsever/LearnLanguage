"""Curated curriculum: the app's offline spine.

Everything here works with no API key and no network. The LLM, when present,
adds unlimited *extra* material on top; it is never required to have something
to learn.

Content lives in ``data/<language>/<domain>.json`` keyed by unit id:

    {
      "greetings": {
        "grammar": ["fr-articles-gender"],
        "groups": [
          {"title": "Meeting someone", "items": [{...}, {...}]}
        ]
      }
    }

Each item mirrors the LessonItem schema so curriculum content and generated
content flow through exactly the same card pipeline:
``target, english, pronunciation, example, example_en, note, tags``.
"""
from __future__ import annotations

import json
import logging

import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .taxonomy import (  # noqa: F401 - re-exported as the module's public tree
    CEFR_ORDER,
    DOMAINS,
    DOMAIN_INDEX,
    UNIT_INDEX,
    learning_path,
    taxonomy_payload,
    unit_meta,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"

ITEM_FIELDS = ("target", "english", "pronunciation", "example", "example_en", "note", "tags")


def _clean_item(raw: dict, unit_id: str, level: str) -> Optional[dict]:
    """Normalize one authored item; drop anything without the two required halves."""
    target = str(raw.get("target") or "").strip()
    english = str(raw.get("english") or "").strip()
    if not target or not english:
        return None
    tags = [str(t).strip() for t in (raw.get("tags") or []) if str(t).strip()]
    return {
        "target": target,
        "english": english,
        "pronunciation": str(raw.get("pronunciation") or "").strip(),
        "example": str(raw.get("example") or "").strip(),
        "example_en": str(raw.get("example_en") or "").strip(),
        "note": str(raw.get("note") or "").strip(),
        "tags": tags,
        "unit": unit_id,
        "level": level,
    }


@lru_cache(maxsize=8)
def _language_content(language_code: str) -> Dict[str, dict]:
    """All units authored for one language, keyed by unit id. Cached per process."""
    root = DATA_DIR / language_code
    if not root.is_dir():
        return {}
    units: Dict[str, dict] = {}
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Curriculum file %s is not readable JSON", path)
            continue
        for unit_id, body in payload.items():
            meta = UNIT_INDEX.get(unit_id)
            if not meta:
                logger.warning("%s defines unknown unit '%s' - skipped", path.name, unit_id)
                continue
            groups = []
            for group in body.get("groups") or []:
                items = [
                    item for item in
                    (_clean_item(raw, unit_id, meta["level"]) for raw in group.get("items") or [])
                    if item
                ]
                if items:
                    groups.append({"title": str(group.get("title") or "Items").strip(), "items": items})
            if not groups:
                continue
            units[unit_id] = {
                "grammar": [str(g) for g in body.get("grammar") or []],
                "groups": groups,
            }
    return units


def clear_cache() -> None:
    """Drop the parsed-content cache (used by tests that write fixtures)."""
    _language_content.cache_clear()


def available_languages() -> List[str]:
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and _language_content(p.name))


def unit_items(language_code: str, unit_id: str) -> List[dict]:
    """Flat item list for one unit, in authored order."""
    unit = _language_content(language_code).get(unit_id)
    if not unit:
        return []
    return [item for group in unit["groups"] for item in group["items"]]


def unit_detail(language_code: str, unit_id: str) -> Optional[dict]:
    """One unit with its metadata and grouped items, ready for the UI."""
    meta = UNIT_INDEX.get(unit_id)
    unit = _language_content(language_code).get(unit_id)
    if not meta or not unit:
        return None
    return {
        **meta,
        "language": language_code,
        "grammar": unit["grammar"],
        "groups": unit["groups"],
        "itemCount": sum(len(g["items"]) for g in unit["groups"]),
    }


def tree(language_code: str) -> List[dict]:
    """The full browsable taxonomy annotated with what this language actually has."""
    content = _language_content(language_code)
    result = []
    for domain in taxonomy_payload():
        units = []
        for unit in domain["units"]:
            body = content.get(unit["id"])
            units.append({
                **unit,
                "domain": domain["id"],
                "domainTitle": domain["title"],
                "itemCount": sum(len(g["items"]) for g in body["groups"]) if body else 0,
                "groupCount": len(body["groups"]) if body else 0,
                "grammar": body["grammar"] if body else [],
                "available": bool(body),
            })
        domain["units"] = units
        domain["itemCount"] = sum(u["itemCount"] for u in units)
        result.append(domain)
    return result


def summary(language_code: str) -> dict:
    content = _language_content(language_code)
    units = len(content)
    items = sum(len(g["items"]) for body in content.values() for g in body["groups"])
    return {"language": language_code, "units": units, "items": items,
            "domains": len({UNIT_INDEX[u]["domain"] for u in content})}


def _fold(text: str) -> str:
    """Accent-insensitive, case-insensitive key for search."""
    stripped = unicodedata.normalize("NFD", str(text or "").lower())
    return "".join(ch for ch in stripped if unicodedata.category(ch) != "Mn")


# Words too common to say anything about which unit a topic belongs to.
_STOPWORDS = frozenset("""
a about all an and any are as at be by can do for from get go have how i in into is it
me my of on or our so some that the their them there they this to up us we what when
where which who with you your
""".split())


def best_unit_for_topic(language_code: str, topic: str) -> Optional[str]:
    """Which curated unit best covers a free-form topic phrase.

    Whole-phrase search never matches ("ordering a coffee at a cafe" is nobody's
    unit title), so score each unit by how many meaningful words of the topic it
    contains, across its title, goal, keywords, and its items.
    """
    words = {w for w in (_fold(topic).replace("-", " ").split()) if len(w) > 2 and w not in _STOPWORDS}
    if not words:
        return None
    content = _language_content(language_code)
    best, best_score = None, 0
    for unit_id, body in content.items():
        meta = UNIT_INDEX[unit_id]
        # Unit metadata is a strong signal, item text a weaker but broader one.
        meta_hay = _fold(" ".join([meta["title"], meta["goal"], *meta["keywords"]]))
        item_hay = _fold(" ".join(
            f"{item['english']} {' '.join(item['tags'])}"
            for group in body["groups"] for item in group["items"]
        ))
        score = sum(3 for w in words if w in meta_hay) + sum(1 for w in words if w in item_hay)
        if score > best_score:
            best, best_score = unit_id, score
    # One incidental word match is noise, not a topic match.
    return best if best_score >= 3 else None


def search(language_code: str, query: str, limit: int = 40) -> dict:
    """Search the curriculum. Units and items are returned as separate lists so
    the UI can offer "jump to this topic" above "here are matching words"."""
    needle = _fold(query).strip()
    if len(needle) < 2:
        return {"query": query, "units": [], "items": []}
    content = _language_content(language_code)
    unit_hits: List[tuple] = []
    item_hits: List[tuple] = []
    for unit_id, body in content.items():
        meta = UNIT_INDEX[unit_id]
        title = _fold(meta["title"])
        haystack = _fold(" ".join([meta["title"], meta["goal"], *meta["keywords"]]))
        if needle in haystack:
            unit_hits.append((0 if title.startswith(needle) else 1 if needle in title else 2, {
                **meta,
                "itemCount": sum(len(g["items"]) for g in body["groups"]),
            }))
        for group in body["groups"]:
            for item in group["items"]:
                target, english = _fold(item["target"]), _fold(item["english"])
                if needle not in target and needle not in english:
                    continue
                rank = 0 if target.startswith(needle) or english.startswith(needle) else 1
                item_hits.append((rank, len(item["target"]), {
                    **item,
                    "unitTitle": meta["title"],
                    "domain": meta["domain"],
                    "domainTitle": meta["domainTitle"],
                    "group": group["title"],
                }))
    unit_hits.sort(key=lambda h: h[0])
    item_hits.sort(key=lambda h: (h[0], h[1]))
    return {
        "query": query,
        "units": [h[1] for h in unit_hits[:8]],
        "items": [h[2] for h in item_hits[:limit]],
    }


def starter_items(language_code: str, count: int = 24) -> List[dict]:
    """A cross-domain A1 starter set, used by the one-tap starter deck."""
    picks: List[dict] = []
    content = _language_content(language_code)
    for unit_id in learning_path():
        if unit_id not in content:
            continue
        if UNIT_INDEX[unit_id]["level"] != "A1":
            continue
        # Two per unit, breadth first, so the starter deck spans daily life.
        picks.extend(unit_items(language_code, unit_id)[:3])
        if len(picks) >= count:
            break
    return picks[:count]


def starter_pack(language_code: str, count: int = 24) -> Optional[dict]:
    items = starter_items(language_code, count)
    if not items:
        return None
    return {
        "language": language_code,
        "topic": "Core starter set",
        "level": "A1",
        "items": items,
        "grammar_features": [],
        "source": "curriculum",
        "unit": "starter",
    }


def unit_pack(language_code: str, unit_id: str) -> Optional[dict]:
    """A unit shaped exactly like a generated lesson pack, for the same code path."""
    detail = unit_detail(language_code, unit_id)
    if not detail:
        return None
    return {
        "language": language_code,
        "topic": detail["title"],
        "level": detail["level"],
        "items": [item for group in detail["groups"] for item in group["items"]],
        "grammar_features": detail["grammar"],
        "source": "curriculum",
        "unit": unit_id,
    }
