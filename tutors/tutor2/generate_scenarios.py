#!/usr/bin/env python3
"""
Bulk scenario generator for the Spanish Scenario Trainer
========================================================

End-to-end pipeline that:
1) Loads your vocabulary (EN↔ES) and builds semantic groups (7–15 words, ≥7 minimum).
2) (Optional) Attaches **3 grammar rules** to every group (ensuring global coverage if requested).
3) Uses an LLM to generate a compact ScenarioPack per group (ES + EN items, MCQs).

Key files (defaults):
- Vocabulary JSON: ../tutor1/data/vocabulary_es.json
- Grammar JSON:    grammar_rules.json
- Group+Rules:     group_rules_attached.json

Notes
-----
•  **Compact output**: The passage appears **once** per scenario; items do **not** repeat it.
•  LLM prompt explicitly forbids redundant restatement of the passage in items/options.
•  On-disk embedding cache (~/.cache/spanish_scen_embeddings.json).
•  Fast deterministic grouping **7–15 words** (vectorised similarity matrix) with ≥min safeguard.
•  Redundancy guards (unique questions + choice de-duplication).
•  Reuse an existing group_rules_attached.json with --reuse-groups, or rebuild with --regen-groups.

Changelog for this version
--------------------------
1. Output schema is compact (no repeated passage in items).
2. Prompts emphasize **concisión** and **no redundancy** to minimize tokens.
3. Detailed logging at each stage; per-group progress and counts.
4. NEW: WITH_RULES toggle (default False). When off, rules are ignored everywhere.
5. FIX: capacity-aware merging that guarantees groups stay within [min_size, max_size].

MODE (no CLI needed)
--------------------
Set the MODE variable at the bottom to run common flows without bash flags:
  "1": Full pipeline (rebuild groups+rules, then LLM)
  "2": Full pipeline (reuse existing groups+rules, then LLM)
  "3": Only build/attach groups+rules (no LLM), rebuild
  "4": Quick test on 2 groups (reuse if present)
  "5": Rebuild + save dendrogram (then LLM)
  "off": Ignore MODE and use CLI flags normally
"""

from __future__ import annotations
import argparse, concurrent.futures, json, logging, os, random, sys, threading, time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import openai
from dotenv import load_dotenv
from difflib import SequenceMatcher
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────
# 0.  CONFIG, API KEY & LOGGING
# ──────────────────────────────────────────────────────────────

# Global toggle: default is OFF (no rules anywhere)
WITH_RULES_DEFAULT = False

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or sys.exit("❌  Set OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("generator")
progress_lock = threading.Lock()

# Will be set from CLI in main()
WITH_RULES = WITH_RULES_DEFAULT

# ══════════════════════════════════════════════════════════════
# 1.  Pydantic SCHEMATA  (LLM → compact → expanded)
# ══════════════════════════════════════════════════════════════
class CompactQAItem(BaseModel):
    question: str
    choices: List[str] = Field(..., min_items=4, max_items=4)
    reasoning_note: str
    question_en: str
    choices_en: List[str] = Field(..., min_items=4, max_items=4)
    reasoning_note_en: str
    correct_choice: int = Field(..., ge=0, le=3)
    accepted_answers: List[str]

class CompactScenarioPack(BaseModel):
    scenario_title: str
    difficulty: str
    passage: str
    passage_en_lines: List[str]
    items: List[CompactQAItem] = Field(..., min_items=5, max_items=5)

class QAItem(BaseModel):
    question: str
    choices: List[str]
    reasoning_note: str
    question_en: str
    choices_en: List[str]
    reasoning_note_en: str
    correct_choice: int
    accepted_answers: List[str]

class ScenarioPack(BaseModel):
    scenario_title: str
    difficulty: str
    passage: str
    passage_en_lines: List[str]
    items: List[QAItem]

# ══════════════════════════════════════════════════════════════
# 2.  GPT HELPER (throttled, extra logging)
# ══════════════════════════════════════════════════════════════
_GPT_THROTTLE = 0.10  # 100 ms between calls

def gpt_call(system_msg: str, user_msg: str, model: str) -> CompactScenarioPack:
    time.sleep(_GPT_THROTTLE)
    resp = openai.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user",   "content": user_msg}],
        response_format=CompactScenarioPack
    )
    parsed = resp.choices[0].message.parsed
    return parsed

# ══════════════════════════════════════════════════════════════
# 3.  EMBEDDINGS (disk cache  + parallel download)
# ══════════════════════════════════════════════════════════════
EMB_MODEL   = "text-embedding-3-small"
CACHE_PATH  = Path.home() / ".cache/spanish_scen_embeddings.json"
_EMB: Dict[str, List[float]] = {}
_EMB_THROTTLE = 0.05  # 50 ms / embedding thread

def _load_cache():
    if CACHE_PATH.exists():
        try:
            _EMB.update(json.loads(CACHE_PATH.read_text("utf-8")))
            log.info("🔹  Loaded %d vectors from cache", len(_EMB))
        except Exception as e:
            log.warning("Cache unreadable – rebuilding (%s)", e)

def _save_cache():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_EMB))
    log.info("🔹  Saved cache → %s", CACHE_PATH)

def _embed_one(word: str) -> Tuple[str, List[float]]:
    time.sleep(_EMB_THROTTLE)
    vec = openai.embeddings.create(input=[word], model=EMB_MODEL).data[0].embedding
    return word, vec

def precompute_embeddings(words: List[str], workers: int = 50):
    missing = [w for w in words if w not in _EMB]
    if not missing:
        log.info("✅  No new embeddings needed.")
        return
    total = len(missing)
    log.info("⬇️  Downloading %d new embeddings (≤%d threads)…", total, workers)
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for idx, (tok, vec) in enumerate(ex.map(_embed_one, missing), 1):
            _EMB[tok] = vec
            if idx % 200 == 0 or idx == total:
                log.info("   E[%d/%d]", idx, total)
    _save_cache()

def vec(word: str) -> np.ndarray:
    return np.asarray(_EMB[word], dtype=np.float32)

# ══════════════════════════════════════════════════════════════
# 4.  GROUPING (capacity-aware bounds + optional dendrogram)
# ══════════════════════════════════════════════════════════════
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import math

def build_groups(
    pairs: List[Tuple[str, str]],
    min_size: int = 6,
    max_size: int = 10,
    visualize: bool = False,
    dendro_path: Optional[str] = None
) -> List[List[Tuple[str, str]]]:
    """
    Cluster vocabulary (Spanish side) with agglomerative clustering on cosine-distance,
    split large clusters so every group is ≤ max_size, then merge undersized clusters
    (<min_size items) without ever exceeding max_size. If no neighbor has capacity,
    perform a balanced repartition with the nearest neighbor so all resulting chunks are
    within [min_size, max_size]. Optionally saves a dendrogram PNG.

    Returns a list of groups: each is a list of (en, es) tuples. Every group has min_size–max_size items.
    """
    # 1 ── vectors & distance matrix
    vecs = np.vstack([vec(es) for _, es in pairs])          # shape (N, D)
    dist = 1 - cosine_similarity(vecs)                      # cosine → distance

    # 2 ── decide number of clusters ≈ len(words) / midpoint
    avg_target = max(min_size, (min_size + max_size) // 2)
    n_clusters = max(1, math.ceil(len(pairs) / avg_target))

    # 3 ── run agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
        compute_distances=True
    ).fit(dist)

    # 4 ── collect indices per cluster label
    label_to_idx: Dict[int, List[int]] = {}
    for idx, label in enumerate(clustering.labels_):
        label_to_idx.setdefault(label, []).append(idx)

    # 5 ── split oversize clusters into chunks of ≤ max_size
    groups: List[List[Tuple[str, str]]] = []
    for idxs in label_to_idx.values():
        for start in range(0, len(idxs), max_size):
            sub = idxs[start:start + max_size]
            groups.append([pairs[i] for i in sub])

    # ───────────────── helpers for capacity-aware merging ─────────────────
    def group_vec(g: List[Tuple[str, str]]) -> np.ndarray:
        return np.mean([vec(es) for _, es in g], axis=0)

    def split_with_minmax(seq: List[Tuple[str, str]], max_sz: int, min_sz: int = min_size):
        """
        Split seq into chunks so each chunk size ∈ [min_sz, max_sz].
        Assumes len(seq) ≥ min_sz. Uses greedy forward split with final-chunk rebalance.
        """
        n = len(seq)
        chunks = []
        i = 0
        while i < n:
            remaining = n - i
            take = min(max_sz, remaining)
            # ensure remainder won't be < min_sz unless this is the last chunk
            remainder = remaining - take
            if remainder != 0 and remainder < min_sz:
                deficit = min_sz - remainder
                take -= deficit
            chunks.append(seq[i:i+take])
            i += take
        if any(len(c) < min_sz for c in chunks):
            return [seq]
        return chunks

    def nearest_group_index(source_gv: np.ndarray, candidates: List[List[Tuple[str, str]]]) -> int:
        sims = [cosine_similarity([source_gv], [group_vec(h)])[0, 0] for h in candidates]
        return int(np.argmax(sims)) if sims else -1

    # 6 ── merge small groups (<min_size) without violating max_size
    big: List[List[Tuple[str, str]]] = []
    small: List[List[Tuple[str, str]]] = []
    for g in groups:
        (big if len(g) >= min_size else small).append(g)

    if not small:
        merged = big
    else:
        merged = list(big)
        while small:
            g = small.pop()
            gv = group_vec(g)

            caps = [idx for idx, h in enumerate(merged) if len(h) < max_size]
            if caps:
                whole_fit = [idx for idx in caps if len(merged[idx]) + len(g) <= max_size]
                if whole_fit:
                    cand_groups = [merged[idx] for idx in whole_fit]
                    j = nearest_group_index(gv, cand_groups)
                    target_idx = whole_fit[j]
                    merged[target_idx].extend(g)
                else:
                    cand_groups = [merged[idx] for idx in caps]
                    sims = [cosine_similarity([gv], [group_vec(h)])[0, 0] for h in cand_groups]
                    order = [caps[k] for k in np.argsort(sims)[::-1]]
                    remaining = list(g)
                    for idx in order:
                        room = max_size - len(merged[idx])
                        if room <= 0:
                            continue
                        take = min(room, len(remaining))
                        if take == 0:
                            break
                        merged[idx].extend(remaining[:take])
                        remaining = remaining[take:]
                    if len(remaining) >= min_size:
                        merged.append(remaining)
                    elif len(remaining) == 0:
                        pass
                    else:
                        if merged:
                            k = nearest_group_index(group_vec(remaining), merged)
                            combo = merged[k] + remaining
                            chunks = split_with_minmax(combo, max_size, min_size)
                            merged[k] = chunks[0]
                            for ch in chunks[1:]:
                                merged.append(ch)
                        else:
                            merged.append(remaining)
            else:
                if not merged:
                    if small:
                        h = small.pop()
                        combo = h + g
                        chunks = split_with_minmax(combo, max_size, min_size)
                        merged.extend(chunks)
                    else:
                        merged.append(g)
                else:
                    k = nearest_group_index(gv, merged)
                    combo = merged[k] + g
                    chunks = split_with_minmax(combo, max_size, min_size)
                    merged[k] = chunks[0]
                    for ch in chunks[1:]:
                        merged.append(ch)

    # 7 ── Optional: save dendrogram
    if visualize:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy.cluster.hierarchy import dendrogram, linkage

            Z = linkage(vecs, method="average", metric="cosine")
            fig_w = max(14, min(40, len(pairs) * 0.15))
            plt.figure(figsize=(fig_w, 6))
            dendrogram(
                Z,
                labels=[es for _, es in pairs],
                leaf_rotation=90,
                leaf_font_size=8,
                color_threshold=0.7 * max(Z[:, 2]),
            )
            plt.title("Hierarchical Clustering Dendrogram", fontsize=16, weight="bold")
            plt.xlabel("Vocabulary Items (Spanish)", fontsize=12)
            plt.ylabel("Cosine Distance", fontsize=12)
            plt.subplots_adjust(bottom=0.3, top=0.9)
            out = dendro_path or "dendrogram.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            log.info("📊  Dendrogram saved → %s", out)
        except Exception as e:
            log.warning("Could not render dendrogram (%s)", e)

    # 8 ── Final safety: enforce bounds & log
    for i, g in enumerate(merged):
        if not (min_size <= len(g) <= max_size):
            raise AssertionError(f"Group {i} violates size constraint: {len(g)} (bounds {min_size}..{max_size})")

    log.info("🔍  Clustering done → %d groups, all sizes in [%d, %d].", len(merged), min_size, max_size)
    return merged

# ══════════════════════════════════════════════════════════════
# 5.  REDUNDANCY GUARDS
# ══════════════════════════════════════════════════════════════
def _lev(a: str, b: str) -> int:
    return int((1 - SequenceMatcher(None, a, b).ratio()) * max(len(a), len(b)))

def dedupe_choices(item: CompactQAItem):
    seen = []
    for i, ch in enumerate(item.choices):
        if any(_lev(ch.lower(), s.lower()) <= 5 for s in seen):
            item.choices[i] = ch + " (var.)"
        seen.append(item.choices[i])

def ensure_unique_items(pack: CompactScenarioPack) -> CompactScenarioPack | None:
    uniq: Dict[str, CompactQAItem] = {}
    for it in pack.items:
        key = it.question.lower().strip()
        if key not in uniq:
            dedupe_choices(it)
            uniq[key] = it
    if len(uniq) < 5:
        return None
    pack.items = list(uniq.values())[:5]
    return pack

# ══════════════════════════════════════════════════════════════
# 6.  PROMPTS (concise, no redundancy, compact-only schema)
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """
Eres un diseñador de materiales de ELE (Español como Lengua Extranjera).

OBJETIVO → genera **UN** paquete en ESPAÑOL basado en:
• Palabras clave (ES) → cohesión semántica.

📏 Reglas de economía y formato (OBLIGATORIAS)
• Concisión del relato: **110–170 palabras** (no relleno).
• **NO** repitas el pasaje dentro de las preguntas, opciones o notas.
• **NO** incluyas glosarios, listas de palabras o las reglas en la salida.
• Devuelve **exclusivamente** un JSON válido que respete el **esquema** proporcionado por el sistema.
• El pasaje aparece **una sola vez** en el campo `passage`. Los ítems **no** deben contener el pasaje.

🟣 FASE 1 – MICRO-RELATO  
• 5–10 frases, mismo escenario y personajes, verosímil y didáctico.  
• Usa la mayoría de las palabras clave (ES), no es preciso usar todas.

🟢 FASE 2 – PREGUNTAS  
• 5 preguntas, cada una sobre un aspecto distinto que requiera **razonamiento**.  
• 4 opciones plausibles (≤120 caracteres); **solo 1 correcta** (posiciones variadas).  
• Incluye versión EN, `accepted_answers`, y notas de razonamiento breves (ES + EN).

Nivel de dificultad: {difficulty}
"""

def build_user_query(words_es: List[str],
                     glossary_es2en: Dict[str, List[str]],
                     rule_strings: List[str]) -> str:
    """Builds the per-group user prompt. Includes rules only if WITH_RULES is True."""
    gloss_lines = []
    for es in words_es:
        ens = glossary_es2en.get(es, [])
        ens = list(dict.fromkeys([e for e in ens if e]))  # unique, non-empty
        if not ens:
            gloss_lines.append(f"- {es}: (no EN in vocab; infer from context)")
        else:
            shown = "; ".join(ens[:2]) + ("" if len(ens) <= 2 else "; …")
            gloss_lines.append(f"- {es}: {shown}")

    parts = [
        f"Palabras ES (usa la mayoría, no es preciso usar todas): {', '.join(words_es)}",
        "Glosario ES→EN (referencia; múltiples traducciones posibles):",
        "\n".join(gloss_lines),
        ""
    ]
    if WITH_RULES and rule_strings:
        rules_block = "\n".join([f"- {r}" for r in rule_strings])
        parts += [
            "🎯 Reglas gramaticales a incorporar (usa **todas** de forma natural):",
            rules_block,
            ""
        ]

    parts.append("- Devuelve SOLO JSON del esquema; los ítems **no** llevan el pasaje.")
    return "\n".join(parts)

# ══════════════════════════════════════════════════════════════
# 7.  EXPANSION & PAYLOAD (compact; no repeated passage)
# ══════════════════════════════════════════════════════════════
def expand(p: CompactScenarioPack) -> ScenarioPack:
    return ScenarioPack(
        scenario_title=p.scenario_title,
        difficulty=p.difficulty,
        passage=p.passage,
        passage_en_lines=p.passage_en_lines,
        items=[QAItem(**it.model_dump()) for it in p.items]
    )

def to_payload(sp: ScenarioPack,
               meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compact payload:
      { "<title>": {
          "passage": "...",
          "passage_en_lines": [...],
          "words_es": [...],           # from meta
          "rules": [...],              # only when WITH_RULES=True and provided
          "items": [ {question,...}, ... ]
        }}
    """
    payload = {
        sp.scenario_title: {
            "passage": sp.passage,
            "passage_en_lines": sp.passage_en_lines,
            "items": [
                {
                    "question": q.question,
                    "question_en": q.question_en,
                    "choices": q.choices,
                    "choices_en": q.choices_en,
                    "correct_index": q.correct_choice,
                    "explain": q.reasoning_note,
                    "explain_en": q.reasoning_note_en,
                    "answers": q.accepted_answers
                }
                for q in sp.items
            ]
        }
    }
    if meta:
        if "words_es" in meta:
            payload[sp.scenario_title]["words_es"] = meta["words_es"]
        if WITH_RULES and "rules" in meta and meta["rules"]:
            payload[sp.scenario_title]["rules"] = meta["rules"]
    return payload

# ══════════════════════════════════════════════════════════════
# 8.  GROUPS + (optional) RULES: Build or Reuse
# ══════════════════════════════════════════════════════════════
def load_vocab_pairs(vocab_path: str) -> Tuple[List[Tuple[str, str]], Dict[str, List[str]]]:
    """
    Returns:
      - pairs: unique list of (en, es)
      - es2en: dict ES -> list[EN] (multi-translation aware)
    """
    with open(vocab_path, encoding="utf-8") as f:
        raw = json.load(f)
    all_pairs = []
    for t in raw.get("topics", []):
        for en, es in t.get("entries", []):
            en_s = str(en).strip()
            es_s = str(es).strip()
            if en_s and es_s:
                all_pairs.append((en_s, es_s))

    # Deduplicate by (en, es) and also build ES→EN multi-map
    uniq_pairs = list(dict.fromkeys(all_pairs))
    es2en: Dict[str, List[str]] = {}
    for en, es in uniq_pairs:
        es2en.setdefault(es, [])
        if en not in es2en[es]:
            es2en[es].append(en)
    return uniq_pairs, es2en

def _build_rule_pool(grammar: Dict[str, Any], seed: Optional[int]) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    persons = grammar["persons"]["order"]
    g = grammar["verbs"]["regular_endings"]

    pool = []
    for tense in ["present", "preterite", "imperfect", "future", "conditional"]:
        bucket = g["indicative"][tense]
        inf_classes = ["-ar", "-er", "-ir"] if "-ar" in bucket else ["all"]
        for ic in inf_classes:
            k = rng.randint(1, 3)
            sel = rng.sample(persons, k=k)
            nice = ", ".join(sel)
            prompt = (f"Conjugate **regular verbs** in **Indicative {tense.capitalize()}** (persons: {nice})."
                      if ic == "all" else
                      f"Conjugate **regular {ic} verbs** in **Indicative {tense.capitalize()}** (persons: {nice}).")
            pool.append({"id": f"verbs.ind.{tense}.{ic}", "macro": "verbs", "prompt": prompt})

    for ic in ["-ar", "-er/-ir"]:
        k = rng.randint(1, 3)
        sel = rng.sample(persons, k=k)
        pool.append({"id": f"verbs.subj.present.{ic}", "macro": "verbs",
                     "prompt": f"Conjugate **regular {ic} verbs** in **Subjunctive Present** (persons: {', '.join(sel)})."})

    for ic in ["-ar", "-er", "-ir"]:
        for p in ["tú", "Ud.", "nosotros", "vosotros", "Uds."]:
            pool.append({"id": f"verbs.imp.aff.{ic}.{p}", "macro": "verbs",
                         "prompt": f"Use **affirmative imperative** for **{ic}** (person: {p}); attach pronouns if needed (dímelo)."})

    pool += [
        {"id":"verbs.perfect.present.auto","macro":"verbs",
         "prompt":"Use **haber (presente) + participio** (pretérito perfecto) with at least two different participles."},
        {"id":"verbs.progressive.present.auto","macro":"verbs",
         "prompt":"Use **estar (presente) + gerundio**; include one irregular gerund if natural."},
        {"id":"verbs.pret.spelling.auto","macro":"verbs",
         "prompt":"Include a preterite **yo** form with correct spelling ( -car→qué, -gar→gué, -zar→cé )."},
        {"id":"verbs.present.stemchange.auto","macro":"verbs",
         "prompt":"Include at least one **present stem-changing verb** (e→ie, o→ue, e→i) respecting boot patterns."}
    ]

    # You can also extend with grammar["rules"] here when WITH_RULES is True (not needed now).
    # We keep the pool minimal for determinism even if rules are turned on later.

    # De-duplicate
    seen = set()
    uniq = []
    for r in pool:
        key = (r["id"], r["prompt"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    rng.shuffle(uniq)
    return uniq

def attach_rules_coverage(groups_words: List[List[str]],
                          grammar_path: str,
                          k_rules: int,
                          seed: Optional[int],
                          strict_coverage: bool = True) -> List[Dict[str, Any]]:
    """Attach k rules per group with coverage; NO-OP when WITH_RULES=False."""
    if not WITH_RULES:
        # Return groups without rules
        return [{"words_es": words} for words in groups_words]

    with open(grammar_path, encoding="utf-8") as f:
        grammar = json.load(f)
    pool = _build_rule_pool(grammar, seed)
    num_groups = len(groups_words)
    capacity = num_groups * k_rules

    if len(pool) == 0:
        sys.exit("❌  No rules available in grammar JSON.")
    if len(pool) > capacity:
        msg = (f"Rules in pool: {len(pool)} > capacity (groups × k): {capacity}.\n"
               f"Increase --rules-per-group or reduce the rule pool.")
        if strict_coverage:
            sys.exit("❌  Not enough slots to use all rules at least once.\n" + msg)
        else:
            log.warning("⚠️  Coverage relaxed: %s", msg)
            rng = random.Random(seed)
            pool = rng.sample(pool, capacity)

    # Empty bundles
    bundles: List[List[Dict[str, str]]] = [[] for _ in range(num_groups)]
    # Round-robin place each rule once
    gi = 0
    for rule in pool:
        placed = False
        for _ in range(num_groups):
            idx = gi % num_groups
            if len(bundles[idx]) < k_rules and all(rule["macro"] != r["macro"] for r in bundles[idx]):
                bundles[idx].append(rule); gi += 1; placed = True; break
            gi += 1
        if not placed:
            for idx in range(num_groups):
                if len(bundles[idx]) < k_rules:
                    bundles[idx].append(rule); placed = True; break
        if not placed:
            sys.exit("❌  Internal error while placing rules.")

    # Top-up each bundle to k using diversity then any
    rng = random.Random(seed)
    flat = list(pool)
    rng.shuffle(flat)
    for b in bundles:
        used_ids = { (r["id"], r["prompt"]) for r in b }
        used_macros = { r["macro"] for r in b }
        for r in flat:
            if len(b) >= k_rules: break
            key = (r["id"], r["prompt"])
            if key not in used_ids and r["macro"] not in used_macros:
                b.append(r); used_ids.add(key); used_macros.add(r["macro"])
        for r in flat:
            if len(b) >= k_rules: break
            key = (r["id"], r["prompt"])
            if key not in used_ids:
                b.append(r); used_ids.add(key)
        while len(b) < k_rules:
            b.append(rng.choice(flat))

    attached = []
    for words, rules in zip(groups_words, bundles):
        attached.append({
            "words_es": words,
            "rules": [{"id": r["id"], "macro": r["macro"], "rule_string": r["prompt"]} for r in rules]
        })
    return attached

def build_or_reuse_group_rules(vocab_path: str,
                               grammar_path: str,
                               group_rules_json: str,
                               min_group: int,
                               max_group: int,
                               emb_workers: int,
                               visualize: bool,
                               dendro_path: Optional[str],
                               regen: bool,
                               rules_per_group: int,
                               strict_coverage: bool,
                               seed: Optional[int]) -> List[Dict[str, Any]]:
    """
    Returns list of dicts with keys:
      - always: words_es (List[str])
      - when WITH_RULES=True: rules (List[dict])
    Writes to group_rules_json if building anew or when regen is True.
    """
    target = Path(group_rules_json)
    if target.exists() and not regen:
        log.info("📦  Reusing existing groups+rules → %s", target)
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            ok = isinstance(data, list) and data and "words_es" in data[0]
            if WITH_RULES:
                ok = ok and "rules" in data[0]
            if ok:
                log.info("✅  groups JSON validated (len=%d).", len(data))
                return data
            else:
                log.warning("Existing JSON not in expected format for WITH_RULES=%s. Will rebuild.", WITH_RULES)
        except Exception as e:
            log.warning("Could not read existing JSON (%s). Will rebuild.", e)

    # Build from vocab
    pairs_all, _ = load_vocab_pairs(vocab_path)
    es_seen = set()
    pairs_unique_es: List[Tuple[str, str]] = []
    for en, es in pairs_all:
        if es not in es_seen:
            es_seen.add(es)
            pairs_unique_es.append((en, es))

    log.info("📖  Unique Spanish tokens for clustering: %d", len(pairs_unique_es))
    _load_cache()
    precompute_embeddings([es for _, es in pairs_unique_es], workers=emb_workers)
    log.info("🔧  Forming groups (%d–%d words)…", min_group, max_group)
    groups_pairs = build_groups(pairs_unique_es, min_size=min_group, max_size=max_group,
                                visualize=visualize, dendro_path=dendro_path)
    groups_words = [[es for _, es in g] for g in groups_pairs]
    log.info("📦  Groups formed: %d", len(groups_words))

    attached = attach_rules_coverage(groups_words, grammar_path,
                                     k_rules=rules_per_group,
                                     seed=seed, strict_coverage=strict_coverage)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(attached, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("💾  Saved groups JSON → %s", target)
    return attached

# ══════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    global WITH_RULES

    script_dir = Path(__file__).resolve().parent
    DEF_VOCAB = str(script_dir.parent / "tutor1" / "data" / "vocabulary_es.json")
    DEF_GRAM  = str(script_dir / "grammar_rules.json")
    DEF_GRPJS = str(script_dir / "group_rules_attached.json")

    ap = argparse.ArgumentParser()
    # Inputs / outputs
    ap.add_argument("--vocab", default=DEF_VOCAB, help="Vocabulary JSON (topics[].entries[[en, es], ...])")
    ap.add_argument("--grammar", default=DEF_GRAM, help="Spanish grammar JSON knowledge base")
    ap.add_argument("--group-rules-json", default=DEF_GRPJS, help="Path for the groups(+rules) JSON")
    ap.add_argument("--out-dir", default="scenarios_out", help="Directory for scenario JSON outputs")
    ap.add_argument("--model", default="gpt-5-mini")

    # Grouping controls
    ap.add_argument("--min-group", type=int, default=6)
    ap.add_argument("--max-group", type=int, default=11)
    ap.add_argument("--emb-workers", type=int, default=50)
    ap.add_argument("--visualize-dendrogram", action="store_true", default=False)
    ap.add_argument("--dendrogram-path", default=None)

    # Rules
    ap.add_argument("--with-rules", action="store_true", default=WITH_RULES_DEFAULT,
                    help="Include grammar rules end-to-end (prompting + output). Default off.")
    ap.add_argument("--rules-per-group", type=int, default=3)
    ap.add_argument("--strict-coverage", action="store_true", default=True,
                    help="Fail if total rules > number_of_groups * rules_per_group (only used when --with-rules).")

    # Flow control
    ap.add_argument("--reuse-groups", action="store_true", default=False,
                    help="Reuse existing group_rules_attached.json (skip regroup/reattach if present)")
    ap.add_argument("--regen-groups", action="store_true", default=False,
                    help="Force regeneration of groups JSON")
    ap.add_argument("--no-llm", action="store_true", default=False,
                    help="Only build/attach (or reuse) groups JSON; do not invoke LLM")
    ap.add_argument("--max-groups", type=int, default=None,
                    help="Cut off processing after N groups for quick runs")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--test", action="store_true", default=False, help="Quick smoke test on 2 groups")
    ap.add_argument("--temperature", type=float, default=0.05, help="(kept for compatibility; not used in clustering)")
    args = ap.parse_args()

    WITH_RULES = bool(args.with_rules)

    rng = random.Random(args.seed)
    log.info("🛠️  Config → model=%s, workers=%d, WITH_RULES=%s, rules/group=%d, reuse=%s, regen=%s, no_llm=%s",
             args.model, args.workers, WITH_RULES, args.rules_per_group, args.reuse_groups, args.regen_groups, args.no_llm)

    # 1) Build or reuse groups JSON
    attached = build_or_reuse_group_rules(
        vocab_path=args.vocab,
        grammar_path=args.grammar,
        group_rules_json=args.group_rules_json,
        min_group=args.min_group,
        max_group=args.max_group,
        emb_workers=args.emb_workers,
        visualize=args.visualize_dendrogram,
        dendro_path=args.dendrogram_path,
        regen=args.regen_groups or (not args.reuse_groups),
        rules_per_group=args.rules_per_group,
        strict_coverage=args.strict_coverage,
        seed=args.seed
    )

    total_groups = len(attached)
    if args.test:
        attached = attached[:2]
        log.info("🧪  TEST MODE – generating %d/%d scenarios", len(attached), total_groups)
    elif args.max_groups:
        original = len(attached)
        attached = attached[:args.max_groups]
        log.info("✂️  Limiting to first %d of %d groups for this run.", len(attached), original)
    else:
        log.info("📦  Groups to process: %d", len(attached))

    # 2) If only building/attaching requested
    if args.no_llm:
        line = "─" * 78
        for i, entry in enumerate(attached, start=1):
            words = entry["words_es"]
            preview = ", ".join(words[:12]) + (", …" if len(words) > 12 else "")
            print(line)
            print(f"GROUP {i:03d}  |  size={len(words)}")
            print(f"Words (ES): {preview}")
            if WITH_RULES and "rules" in entry:
                print("Rules:")
                for j, r in enumerate(entry["rules"], start=1):
                    print(f"  {j}. [{r.get('macro','?')}] {r.get('rule_string','')}")
        print(line)
        log.info("✅  Built/Reused %d groups%s. LLM generation skipped (--no-llm).",
                 len(attached), "" if not WITH_RULES else f" with {args.rules_per_group} rules/group")
        return

    # 3) Load vocab again to build ES→EN glossary map for prompts
    _, es2en = load_vocab_pairs(args.vocab)

    # 4) Generate with LLM
    DIFFS = ["elementary", "intermediate", "upper-intermediate", "advanced"]
    per_diff: Dict[str, Dict[str, Any]] = {d:{} for d in DIFFS}
    diff_idx: Dict[str, str] = {}
    total, done = len(attached), 0

    def _unique_title(dct: Dict[str, Any], title: str) -> str:
        """Avoid collisions if the model reuses a title."""
        if title not in dct:
            return title
        k = 2
        while f"{title} ({k})" in dct:
            k += 1
        return f"{title} ({k})"

    def worker(idx_and_entry):
        idx, entry = idx_and_entry
        nonlocal done
        words_es: List[str] = entry.get("words_es", [])
        words_es = [w for w in words_es if isinstance(w, str) and w.strip()]
        if len(words_es) < args.min_group:
            log.warning("🚫  Skipping group %d (<%d tokens): %s", idx+1, args.min_group, words_es)
            return

        # Gather EN translations (multi-translation aware)
        glossary_es2en: Dict[str, List[str]] = {}
        for es in words_es:
            ens = es2en.get(es, [])
            ens = [str(x).strip() for x in ens if isinstance(x, (str, int, float)) and str(x).strip()]
            glossary_es2en[es] = ens

        # Rules (optional)
        rule_strings: List[str] = []
        if WITH_RULES:
            rule_strings = [r.get("rule_string","").strip() for r in entry.get("rules", []) if r.get("rule_string")]

        lvl = rng.choice(DIFFS)
        sys_msg = SYSTEM_PROMPT.format(difficulty=lvl)
        usr_msg = build_user_query(words_es, glossary_es2en, rule_strings)

        try:
            for attempt in range(1, 4):
                try:
                    compact = gpt_call(sys_msg, usr_msg, args.model)
                    compact.difficulty = lvl
                    compact = ensure_unique_items(compact)
                    if not compact:
                        raise ValueError("duplicate questions or insufficient unique items")
                    full = expand(compact)
                    meta: Dict[str, Any] = {"words_es": words_es}
                    if WITH_RULES and rule_strings:
                        meta["rules"] = rule_strings
                    payload = to_payload(full, meta=meta)
                    with progress_lock:
                        title = list(payload.keys())[0]
                        title_u = _unique_title(per_diff[lvl], title)
                        if title_u != title:
                            payload[title_u] = payload.pop(title)
                        per_diff[lvl].update(payload)
                        diff_idx[title_u] = lvl
                        done += 1
                        log.info("   ✅  [%3d/%3d] %s", done, total, title_u)
                    return
                except Exception as e:
                    log.warning("   🔄  group %d attempt %d/3 failed: %s", idx+1, attempt, e)
                    time.sleep(0.4)
            log.error("   🚫  group %d failed after retries.", idx+1)
        except Exception as e:
            log.exception("   💥  Unexpected error in group %d: %s", idx+1, e)

    log.info("🤖  Generating with %d GPT threads…", args.workers)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(worker, enumerate(attached)))

    # 5) Write outputs (compact, per difficulty)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0
    for diff, data in per_diff.items():
        if data:
            path = out_dir / f"scenarios_{diff}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            files_written += 1
            log.info("📝  Wrote %d scenarios to %s", len(data), path)
    (out_dir / "difficulty_index.json").write_text(
        json.dumps(diff_idx, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("🎉  Finished – %d/%d scenarios written across %d files. Index saved.",
             sum(len(d) for d in per_diff.values()), done, files_written)

# ──────────────────────────────────────────────────────────────
# MODE SWITCH (no bash flags needed)
# ──────────────────────────────────────────────────────────────
# Set MODE to one of: "1","2","3","4","5","off"
#   "1": Full pipeline (rebuild groups+rules, then LLM)
#   "2": Full pipeline (reuse existing groups+rules, then LLM)
#   "3": Only build/attach groups+rules (no LLM), rebuild
#   "4": Quick test on 2 groups (reuse if present)
#   "5": Rebuild + save dendrogram (then LLM)
#   "off": Ignore MODE and use CLI flags normally
MODE = "2"   # ← change this value if you want to run a mode directly

if __name__ == "__main__":
    if MODE and MODE.lower() != "off":
        mode_map = {
            "1": ["--regen-groups"],  # rebuild + LLM (respects --with-rules if added)
            "2": ["--reuse-groups"],  # reuse + LLM
            "3": ["--no-llm", "--regen-groups"],  # rebuild, no LLM
            "4": ["--test", "--reuse-groups"],    # quick test (2 groups)
            "5": ["--regen-groups", "--visualize-dendrogram", "--dendrogram-path", "dendrogram.png"],  # rebuild + dendro + LLM
        }
        args_for_mode = mode_map.get(str(MODE).lower())
        if args_for_mode is None:
            print(f"❌  Unknown MODE='{MODE}'. Use one of: 1,2,3,4,5,off")
            sys.exit(2)
        log.info("🚦 MODE='%s' → argv = %s", MODE, " ".join(args_for_mode))
        sys.argv = [sys.argv[0]] + args_for_mode
    main()

# ──────────────────────────────────────────────────────────────
# USAGE MODES (if you prefer CLI)
# ──────────────────────────────────────────────────────────────
# 1) 🚀 Full pipeline (group+rules build OR reuse, then LLM):
#    python3 generate_scenarios.py
#
#    (Rebuild groups JSON, then generate)
#    python3 generate_scenarios.py --regen-groups
#
#    (Reuse existing /tutor2/group_rules_attached.json)
#    python3 generate_scenarios.py --reuse-groups
#
# 2) 🧪 Quick test on 2 groups:
#    python3 generate_scenarios.py --test
#
# 3) 🧰 Only build/attach (skip LLM):
#    python3 generate_scenarios.py --no-llm --regen-groups
#
# 4) 🔧 Tune grouping & coverage:
#    python3 generate_scenarios.py --min-group 7 --max-group 15
#
# 5) 📊 Save dendrogram of the clustering:
#    python3 generate_scenarios.py --visualize-dendrogram --dendrogram-path dendro.png --regen-groups
#
# To enable rules end-to-end on demand:
#    python3 generate_scenarios.py --with-rules
