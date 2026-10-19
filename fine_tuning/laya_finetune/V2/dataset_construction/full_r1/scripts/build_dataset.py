#!/usr/bin/env python3
"""Laya Tool Selection V4 dataset builder (smoke / full).

This script is intentionally self-contained and writes ONLY inside its run
directory. It reuses production, read-only components for the two things that
must stay byte-identical to inference:

  * capability cards  -> core...cap_router.card_renderer.render_card (B_noprov)
  * recall SQL shape  -> the same two SQL lanes the production retriever uses
                         (one ANN pool of 64 per lane), with business filters
                         deliberately disabled for this dataset experiment.

The 0.82 / 0.85 / 0.88 versions are all derived from ONE shared recall pass.
Scores are stored unrounded; thresholds are applied later.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import numpy as np

# ── paths / frozen experiment config ────────────────────────────────────────
HERE = Path(__file__).resolve().parent
RUN_DIR = HERE.parent
REPO_ROOT = HERE.parents[4]
SOURCE_CSV = REPO_ROOT / "scripts/laya_finetune/data_v3/laya_tool_selection_v2.csv"
DB_DSN = os.environ.get(
    "DELVETA_DB_DSN", "postgresql://delveta:delveta@localhost:15432/delveta")
EMBED_URL = os.environ.get("DELVETA_EMBED_URL", "http://localhost:18080/embed")

THRESHOLDS = (0.82, 0.85, 0.88)
THRESHOLD_TAGS = {0.82: "threshold_082", 0.85: "threshold_085",
                  0.88: "threshold_088"}
K_MAX = 3
MIN_OPTION_SLOTS = 1
MAX_OPTION_SLOTS = 4
DYNAMIC_OPTION_SLOTS = {"rule": "K+1", "min": MIN_OPTION_SLOTS,
                        "max": MAX_OPTION_SLOTS}
ANN_POOL = 64
EVIDENCE_FLOOR = 0.80          # 0.02 below the lowest threshold; see report
SEED = 424242
# V2 construction constants, reused verbatim. The per-row shuffle is keyed by
# the row id (source_query_id + "-p2"), exactly as in v2_generate_dataset.py.
SHUFFLE_SEED_PREFIX = "reject-shuffle-v1"
ROW_SEED = "reject-rows-v1"
CALIB_SEED = 20260922
CALIB_MAX = 400
REJECT_LABEL = "REJECT"
V2_PART = "p2"
SPLITS = ("train", "test", "val")

# The frozen V2 training row: exactly these 14 keys, in any order.
V2_ROW_KEYS = (
    "id", "source_query_id", "split", "part", "lang", "state", "order",
    "options", "gold_index", "target_kind", "gold_capability_id",
    "reject_index", "shuffle", "src_sha",
)

# Construction metadata lives ONLY in construction_audit.jsonl. It is joined
# back by id when auditing; it never enters a training row.
AUDIT_KEYS = (
    "id", "source_query_id", "split", "part", "lang", "lang_raw", "state",
    "source_gold_capability_id", "is_action", "label", "label_reason",
    "label_reason_detail", "threshold", "candidate_count", "option_count",
    "candidate_capability_ids", "retrieved_capabilities", "retrieval_scores",
    "gold_score", "gold_retrieval_rank", "decision_state", "parameters",
    "source", "review_status", "query_family", "query_family_id", "csv_line",
    "source_row_id", "source_file", "source_row",
)

# production modules (read-only)
sys.path.insert(0, str(REPO_ROOT / "packages" / "core"))
from core.application.chat.intent_funnel.cap_router.card_renderer import (  # noqa: E402
    INSTRUCTIONS,
    REJECT_CARD,
    render_card,
)
from core.application.chat.intent_funnel.registry.entry import (  # noqa: E402
    CapabilityEntry,
)

LOG = logging.getLogger("v4build")


class Row(dict):
    """dict with attribute access, matching what ``CapabilityEntry.from_rows``
    expects from DB rows while staying usable as a mapping everywhere else."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


_JSON_CAP_COLUMNS = ("patterns", "aliases", "request_query_examples",
                     "parameters", "arg_slots")


def capability_row(raw: Any) -> Row:
    """asyncpg returns json/jsonb columns as text because this script does not
    register a codec; decode exactly the columns CapabilityEntry consumes."""
    d = dict(raw)
    for key in _JSON_CAP_COLUMNS:
        if isinstance(d.get(key), str):
            try:
                d[key] = json.loads(d[key])
            except json.JSONDecodeError:
                pass
    return Row(d)


# ── generic helpers ─────────────────────────────────────────────────────────
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:.2f}%" if d else "n/a (denominator 0)"


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


# ── text normalisation / language / family ──────────────────────────────────
_QUOTE_RE = re.compile(
    r'(".*?"|\'.*?\'|“.*?”|‘.*?’|「.*?」|『.*?』|`.*?`)', re.S)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def normalize_query(text: Any) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).strip().casefold()
    return re.sub(r"\s+", " ", t)


def has_han(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf"
               for c in text)


def has_latin(text: str) -> bool:
    low = text.casefold()
    return any("a" <= c <= "z" for c in low)


def resolve_language(raw: Any, text: str) -> str:
    """Explicit source language wins; blank language is derived.

    Derived buckets: zh / en / mixed (Han+Latin) / unknown.  Mixed and unknown
    are NEVER folded into zh/en in this build.
    """
    r = str(raw or "").strip().lower()
    if r in {"zh", "en"}:
        return r
    h, l = has_han(text), has_latin(text)
    if h and l:
        return "mixed"
    if h:
        return "zh"
    if l:
        return "en"
    return "unknown"


def query_family(text: Any) -> tuple[str, str]:
    """Deterministic, documented family heuristic.

    NFKC + casefold + whitespace collapse, quoted spans -> <q>, digit runs ->
    <n>.  Queries with >= 3 whitespace tokens use the first 5 tokens; others
    use the first 12 characters.  The family key is the signature hash.
    """
    n = normalize_query(text)
    n = _QUOTE_RE.sub(" <q> ", n)
    n = _NUM_RE.sub(" <n> ", n)
    n = re.sub(r"\s+", " ", n).strip()
    toks = n.split()
    sig = " ".join(toks[:5]) if len(toks) >= 3 else n[:12]
    sig = sig or "<empty>"
    return sig, stable_key("family", sig)[:16]


def parse_bool(value: Any) -> bool | None:
    v = str(value or "").strip().lower()
    if v in {"true", "1", "yes"}:
        return True
    if v in {"false", "0", "no"}:
        return False
    return None


def parse_parameters(raw: Any) -> tuple[Any, str | None]:
    s = str(raw or "").strip()
    if not s:
        return None, None
    try:
        return json.loads(s), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


# ── source load / audit ─────────────────────────────────────────────────────
def load_source() -> tuple[list[dict], list[str], str]:
    if not SOURCE_CSV.exists():
        raise SystemExit(f"source CSV missing: {SOURCE_CSV}")
    src_sha = sha256_file(SOURCE_CSV)
    rows: list[dict] = []
    with open(SOURCE_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for idx, raw in enumerate(reader, start=1):
            r = dict(raw)
            r["_row_id"] = idx
            r["_csv_line"] = reader.line_num
            r["_query"] = str(r.get("query") or "")
            r["_norm"] = normalize_query(r["_query"])
            r["_lang_raw"] = str(r.get("language") or "").strip().lower()
            r["_lang"] = resolve_language(r["_lang_raw"], r["_query"])
            r["_family_sig"], r["_family_key"] = query_family(r["_query"])
            act = parse_bool(r.get("is_action"))
            r["_is_action"] = act
            r["_capability_id"] = str(r.get("capability_id") or "").strip()
            r["_source"] = str(r.get("source") or "")
            r["_review_status"] = str(r.get("review_status") or "")
            r["_source_file"] = str(r.get("source_file") or "")
            r["_source_row"] = str(r.get("source_row") or "")
            r["_source_record_key"] = (
                f"{r['_source_file']}:{r['_source_row']}:{r['_row_id']}")
            r["_parameters"], r["_parameters_error"] = parse_parameters(
                r.get("parameters"))
            rows.append(r)
    return rows, fields, src_sha


async def load_db(conn: asyncpg.Connection) -> dict:
    caps = await conn.fetch("SELECT * FROM capabilities ORDER BY capability_id")
    std = await conn.fetch(
        "SELECT id::text AS id, capability_id, query, language, enabled, position "
        "FROM capability_standard_queries ORDER BY id")
    sim = await conn.fetch(
        "SELECT q.id::text AS id, sq.capability_id, "
        "q.standard_query_id::text AS standard_query_id, "
        "q.query, q.language, q.enabled, q.position "
        "FROM capability_similar_queries q "
        "JOIN capability_standard_queries sq ON sq.id = q.standard_query_id "
        "ORDER BY q.id")
    neg = await conn.fetch(
        "SELECT capability_id, query, enabled FROM capability_negatives "
        "ORDER BY capability_id, query")
    std_vec = await conn.fetch(
        "SELECT id::text AS id, capability_id, query, language, "
        "embedding::text AS embedding "
        "FROM capability_standard_queries WHERE embedding IS NOT NULL ORDER BY id")
    sim_vec = await conn.fetch(
        "SELECT q.id::text AS id, sq.capability_id, q.query, q.language, "
        "q.embedding::text AS embedding "
        "FROM capability_similar_queries q "
        "JOIN capability_standard_queries sq ON sq.id = q.standard_query_id "
        "WHERE q.embedding IS NOT NULL ORDER BY q.id")
    profile_raw = await conn.fetchval(
        "SELECT value FROM app_settings WHERE key = 'embedding_profile'")
    profile = json.loads(profile_raw) if isinstance(profile_raw, str) else profile_raw
    db_version = await conn.fetchval("SELECT version()")
    indexes = await conn.fetch(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE tablename IN ('capability_standard_queries','capability_similar_queries') "
        "ORDER BY tablename, indexname")
    return {
        "caps": [capability_row(r) for r in caps],
        "std": [Row(r) for r in std],
        "sim": [Row(r) for r in sim],
        "neg": [Row(r) for r in neg],
        "std_vec": std_vec, "sim_vec": sim_vec,
        "profile": profile or {}, "db_version": db_version,
        "indexes": [dict(r) for r in indexes],
    }


def build_entries(db: dict) -> dict[str, CapabilityEntry]:
    std_by_cap: dict[str, list] = defaultdict(list)
    sim_by_cap: dict[str, list] = defaultdict(list)
    neg_by_cap: dict[str, list] = defaultdict(list)
    for r in db["std"]:
        std_by_cap[r["capability_id"]].append(r)
    for r in db["sim"]:
        # capability_id is joined from the standard row by the query above.
        # asyncpg rows from a join still expose capability_id only when selected;
        # the sim fetch selects sq.capability_id, so it is present.
        sim_by_cap[r["capability_id"]].append(r)
    for r in db["neg"]:
        neg_by_cap[r["capability_id"]].append(r)
    entries: dict[str, CapabilityEntry] = {}
    for row in db["caps"]:
        cid = row["capability_id"]
        entries[cid] = CapabilityEntry.from_rows(
            row, std_by_cap[cid], sim_by_cap[cid], neg_by_cap[cid])
    return entries


def corpus_struct(db: dict) -> dict:
    std_norm: dict[str, list[dict]] = defaultdict(list)
    sim_norm: dict[str, list[dict]] = defaultdict(list)
    for r in db["std"]:
        std_norm[normalize_query(r["query"])].append(dict(r))
    for r in db["sim"]:
        sim_norm[normalize_query(r["query"])].append(dict(r))
    lines = []
    for kind, rows in (("standard", db["std_vec"]), ("similar", db["sim_vec"])):
        for r in rows:
            lines.append(f"{kind}\t{r['id']}\t{r['capability_id']}\t"
                         f"{r['query']}\t{r['language']}\t{r['embedding']}")
    corpus_sha = sha256_bytes("\n".join(sorted(lines)).encode("utf-8"))
    return {
        "std_norm": std_norm, "sim_norm": sim_norm,
        "corpus_sha": corpus_sha,
        "std_count": len(db["std"]), "sim_count": len(db["sim"]),
        "std_vec_count": len(db["std_vec"]),
        "sim_vec_count": len(db["sim_vec"]),
    }


def classify_rows(rows: list[dict], corpus: dict,
                  cap_ids: set[str]) -> tuple[list[dict], Counter]:
    """Attach eligibility + exclusion evidence to every source row."""
    reasons = Counter()
    # exact/normalised duplicates among all rows
    norm_counts = Counter(r["_norm"] for r in rows if r["_norm"])
    seen_dup: dict[str, int] = {}
    for r in rows:
        r["_is_standard_query"] = bool(r["_norm"] and r["_norm"] in corpus["std_norm"])
        r["_is_similar_query"] = bool(r["_norm"] and r["_norm"] in corpus["sim_norm"])
        r["_duplicate_group_size"] = norm_counts.get(r["_norm"], 0)
        r["_duplicate_index"] = seen_dup.get(r["_norm"], 0)
        if r["_norm"]:
            seen_dup[r["_norm"]] = r["_duplicate_index"] + 1

        why: list[str] = []
        if not r["_query"].strip():
            why.append("empty_query")
        if r["_is_standard_query"] and r["_is_similar_query"]:
            why.append("standard_query_and_similar")
        elif r["_is_standard_query"]:
            why.append("standard_query")
        elif r["_is_similar_query"]:
            why.append("standard_query_similar")
        if r["_is_action"] is None:
            why.append("invalid_is_action")
        elif r["_is_action"]:
            if not r["_capability_id"]:
                why.append("action_missing_capability")
            elif r["_capability_id"] not in cap_ids:
                why.append("unknown_capability")
        else:
            if r["_capability_id"] not in {"", "reject"}:
                why.append("no_action_with_capability")
        if r["_duplicate_group_size"] > 1 and r["_duplicate_index"] > 0:
            why.append("duplicate_query")
        r["_exclude_reason"] = why[0] if why else ""
        r["_exclude_reasons"] = why
        r["_eligible"] = not why
        for w in why:
            reasons[w] += 1
    eligible = [r for r in rows if r["_eligible"]]
    return eligible, reasons


# ── 50-row deterministic smoke selection ────────────────────────────────────
def select_smoke(eligible: list[dict], cap_ids: list[str],
                 limit: int = 50) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in eligible:
        gold = r["_capability_id"] if r["_is_action"] else REJECT_LABEL
        groups[(gold, r["_lang"])].append(r)
    for g in groups:
        groups[g].sort(key=lambda r: stable_key(
            SEED, "select", r["_source_file"], r["_source_row"], r["_row_id"],
            r["_query"]))

    selected: list[dict] = []
    used: set[int] = set()
    families: set[str] = set()

    def add(r: dict, reason: str) -> bool:
        if r["_row_id"] in used:
            return False
        r["_selection_reason"] = reason
        selected.append(r)
        used.add(r["_row_id"])
        families.add(r["_family_key"])
        return True

    def pick(group_key: tuple[str, str], reason: str,
             prefer=None) -> bool:
        rows = groups.get(group_key, [])
        for r in rows:
            if r["_row_id"] in used:
                continue
            if prefer is not None and not prefer(r):
                continue
            if r["_family_key"] in families:
                continue
            if add(r, reason):
                return True
        for r in rows:
            if r["_row_id"] in used:
                continue
            if prefer is not None and not prefer(r):
                continue
            if add(r, reason):
                return True
        return False

    # 1) every capability in zh and en
    for cap in cap_ids:
        pick((cap, "zh"), f"coverage_cap_zh:{cap}")
        pick((cap, "en"), f"coverage_cap_en:{cap}")

    # 2) no-action coverage in both languages, preferring distinct review status
    used_review: set[str] = set()
    for lang in ("zh", "en"):
        pick(("REJECT", lang), f"coverage_reject_{lang}",
             prefer=lambda r, ur=used_review: r["_review_status"] not in ur)
        for r in selected:
            if r["_capability_id"] == "reject" or not r["_is_action"]:
                used_review.add(r["_review_status"])

    # 3) mixed-language breadth: six distinct capabilities
    mixed_groups = [g for g in groups
                    if g[1] == "mixed" and g[0] != REJECT_LABEL]
    mixed_groups.sort(key=lambda g: stable_key(SEED, "mixed", g[0]))
    for g in mixed_groups[:6]:
        pick(g, f"coverage_mixed:{g[0]}")

    # 4) three 3-row groups so train/test/val all get real smoke samples
    extra_groups: list[tuple[str, str]] = []
    reject_groups = [g for g in groups
                     if g[0] == REJECT_LABEL and len(groups[g]) >= 3]
    reject_groups.sort(key=lambda g: stable_key(SEED, "extra-reject", g[0], g[1]))
    if reject_groups:
        extra_groups.append(reject_groups[0])
    action_groups = [g for g in groups
                     if g[0] != REJECT_LABEL and len(groups[g]) >= 3]
    action_groups.sort(key=lambda g: stable_key(SEED, "extra-action", g[0], g[1]))
    for g in action_groups[:2]:
        extra_groups.append(g)
    for g in extra_groups:
        picked = 0
        for r in groups[g]:
            if r["_row_id"] in used:
                continue
            if r["_family_key"] in families and picked == 0:
                # try for a distinct family first; fall back below
                continue
            if add(r, f"split_extra:{g[0]}:{g[1]}:{picked + 2}"):
                picked += 1
            if picked == 2:
                break
        if picked < 2:
            for r in groups[g]:
                if r["_row_id"] in used:
                    continue
                if add(r, f"split_extra:{g[0]}:{g[1]}:{picked + 2}"):
                    picked += 1
                if picked == 2:
                    break

    # 5) fill deterministically if anything was unavailable
    if len(selected) < limit:
        all_groups = sorted(groups, key=lambda g: stable_key(SEED, "fill", g[0], g[1]))
        guard = 0
        while len(selected) < limit and guard < 10000:
            guard += 1
            progressed = False
            for g in all_groups:
                for r in groups[g]:
                    if r["_row_id"] in used:
                        continue
                    if r["_family_key"] in families:
                        continue
                    if add(r, f"fill:{g[0]}:{g[1]}"):
                        progressed = True
                        break
                if len(selected) >= limit:
                    break
            if not progressed:
                for g in all_groups:
                    for r in groups[g]:
                        if r["_row_id"] not in used:
                            add(r, f"fill_family_reuse:{g[0]}:{g[1]}")
                            progressed = True
                            break
                    if len(selected) >= limit:
                        break
            if not progressed:
                break
    return selected[:limit]


# ── split allocation (family-aware) ─────────────────────────────────────────
def allocate_counts(n: int) -> dict[str, int]:
    """75/20/5 with minimum representation for tiny groups.

    n=1 -> train; n=2 -> train+test; n>=3 -> at least one val and one test.
    Large groups converge to 75/20/5.  Deviations are reported.
    """
    if n <= 0:
        return {"train": 0, "test": 0, "val": 0}
    if n == 1:
        return {"train": 1, "test": 0, "val": 0}
    if n == 2:
        return {"train": 1, "test": 1, "val": 0}
    n_val = max(1, int(math.floor(n * 0.05 + 0.5)))
    n_test = max(1, int(math.floor(n * 0.20 + 0.5)))
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train = 1
        # n=3 naturally lands here as 1/1/1.
        overflow = n_train + n_test + n_val - n
        while overflow > 0 and n_test > 0:
            n_test -= 1
            overflow -= 1
        while overflow > 0 and n_val > 0:
            n_val -= 1
            overflow -= 1
    return {"train": n_train, "test": n_test, "val": n_val}


def assign_splits(selected: list[dict]) -> dict[int, str]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in selected:
        gold = r["_capability_id"] if r["_is_action"] else REJECT_LABEL
        groups[(gold, r["_lang"])].append(r)
    targets = {g: allocate_counts(len(rows)) for g, rows in groups.items()}
    assigned = {g: Counter() for g in groups}
    families: dict[str, list[dict]] = defaultdict(list)
    for r in selected:
        families[r["_family_key"]].append(r)
    split_priority = {"val": 0, "test": 1, "train": 2}

    for fam_key in sorted(families,
                          key=lambda k: stable_key(SEED, "split-family", k)):
        rows = families[fam_key]
        per_group = Counter(
            (r["_capability_id"] if r["_is_action"] else REJECT_LABEL, r["_lang"])
            for r in rows)
        best: tuple[float, int, str] | None = None
        for split in SPLITS:
            penalty = 0.0
            for g, count in per_group.items():
                cur = assigned[g][split]
                tgt = targets[g][split]
                after = cur + count
                penalty += (after - tgt) ** 2 - (cur - tgt) ** 2
                if after > tgt:
                    penalty += 10.0 + 2.0 * (after - tgt)
            cand = (penalty, split_priority[split], split)
            if best is None or cand < best:
                best = cand
        split = best[2]
        for r in rows:
            r["_split"] = split
        for g, count in per_group.items():
            assigned[g][split] += count

    return {
        "targets": {f"{g[0]}|{g[1]}": t for g, t in targets.items()},
        "actual": {f"{g[0]}|{g[1]}": dict(a) for g, a in assigned.items()},
    }


# ── embedding / recall ──────────────────────────────────────────────────────
async def embed_rows(rows: list[dict]) -> np.ndarray:
    out_path = RUN_DIR / "data" / "query_embeddings.npy"
    ids_path = RUN_DIR / "data" / "query_embedding_ids.json"
    ids = [r["_source_record_key"] for r in rows]
    if out_path.exists() and ids_path.exists():
        cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if cached_ids == ids:
            LOG.info("embedding cache hit: %s", out_path)
            return np.load(out_path)
    texts = [r["_query"] for r in rows]
    batch, conc = 16, 4
    sem = asyncio.Semaphore(conc)

    async def one(chunk: list[str]) -> np.ndarray:
        async with sem:
            last: Exception | None = None
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=180) as client:
                        resp = await client.post(
                            EMBED_URL,
                            json={"inputs": chunk, "normalize": True})
                        resp.raise_for_status()
                        arr = np.asarray(resp.json(), dtype=np.float32)
                    if arr.shape != (len(chunk), 1024):
                        raise RuntimeError(
                            f"embedding shape {arr.shape} != ({len(chunk)}, 1024)")
                    return arr
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    await asyncio.sleep(1 + attempt)
            raise RuntimeError(f"embedding failed: {last}")

    t0 = time.time()
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    parts = await asyncio.gather(*[one(c) for c in chunks])
    mat = np.vstack(parts)
    norms = np.linalg.norm(mat, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise RuntimeError(f"embeddings not L2-normalised: "
                           f"min={norms.min():.6f} max={norms.max():.6f}")
    np.save(out_path, mat)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    LOG.info("embedded %d queries in %.2fs", len(rows), time.time() - t0)
    return mat


def vec_to_pg(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.9g}" for x in vec) + "]"


RECALL_SQL = """
WITH std AS (
    SELECT s.capability_id,
           s.id::text AS corpus_id,
           s.query AS corpus_query,
           s.language AS corpus_language,
           'standard'::text AS lane,
           1 - (s.embedding <=> $1::vector) AS score
      FROM capability_standard_queries s
     WHERE s.embedding IS NOT NULL
     ORDER BY s.embedding <=> $1::vector
     LIMIT 64
), sim AS (
    SELECT sq.capability_id,
           q.id::text AS corpus_id,
           q.query AS corpus_query,
           q.language AS corpus_language,
           'similar'::text AS lane,
           1 - (q.embedding <=> $1::vector) AS score
      FROM capability_similar_queries q
      JOIN capability_standard_queries sq ON sq.id = q.standard_query_id
     WHERE q.embedding IS NOT NULL
     ORDER BY q.embedding <=> $1::vector
     LIMIT 64
)
SELECT capability_id, corpus_id, corpus_query, corpus_language, lane, score
  FROM std
UNION ALL
SELECT capability_id, corpus_id, corpus_query, corpus_language, lane, score
  FROM sim
"""


async def recall_rows(rows: list[dict], embeddings: np.ndarray) -> list[dict]:
    sem = asyncio.Semaphore(8)
    pool = await asyncpg.create_pool(DB_DSN, min_size=2, max_size=8)

    async def one(i: int) -> list[dict]:
        async with sem:
            raw = await pool.fetch(RECALL_SQL, vec_to_pg(embeddings[i]))
        hits = [dict(r) for r in raw]
        # deterministic lane ranks
        by_lane: dict[str, list[dict]] = defaultdict(list)
        for h in hits:
            by_lane[h["lane"]].append(h)
        for lane, items in by_lane.items():
            items.sort(key=lambda h: (-float(h["score"]), h["corpus_id"]))
            for rank, h in enumerate(items, start=1):
                h["lane_rank"] = rank
        hits.sort(key=lambda h: (-float(h["score"]), h["lane"], h["corpus_id"]))
        for rank, h in enumerate(hits, start=1):
            h["global_rank"] = rank
        return hits

    try:
        t0 = time.time()
        results = await asyncio.gather(*[one(i) for i in range(len(rows))])
        LOG.info("recall: %d queries in %.2fs", len(rows), time.time() - t0)
    finally:
        await pool.close()
    return results


def corpus_matrix(corpus: dict) -> tuple[np.ndarray, list[dict]]:
    vectors = []
    meta = []
    for kind, rows in (("standard", corpus["std_vec_rows"]),
                       ("similar", corpus["sim_vec_rows"])):
        for r in rows:
            vectors.append(np.asarray(json.loads(r["embedding"]), dtype=np.float32))
            meta.append({"kind": kind, "id": r["id"],
                         "capability_id": r["capability_id"],
                         "query": r["query"], "language": r["language"]})
    return np.vstack(vectors), meta


def exact_coverage_audit(query_emb: np.ndarray, corpus_emb: np.ndarray,
                         corpus_meta: list[dict],
                         sql_hits: list[list[dict]]) -> dict:
    exact = query_emb @ corpus_emb.T  # cosine (both L2-normalised)
    caps = sorted({m["capability_id"] for m in corpus_meta})
    cap_index = {c: i for i, c in enumerate(caps)}
    cap_of_row = np.array([cap_index[m["capability_id"]] for m in corpus_meta])

    rows = []
    total_missing_082 = 0
    total_exact_hits_082 = 0
    total_sql_hits_082 = 0
    max_cap_gap = 0.0
    for i, hits in enumerate(sql_hits):
        sql_cap_best = np.full(len(caps), -np.inf)
        for h in hits:
            j = cap_index[h["capability_id"]]
            sql_cap_best[j] = max(sql_cap_best[j], float(h["score"]))
        exact_cap_best = np.full(len(caps), -np.inf)
        for j in range(len(caps)):
            vals = exact[i][cap_of_row == j]
            if len(vals):
                exact_cap_best[j] = float(vals.max())
        gap = float(np.max(np.abs(sql_cap_best - exact_cap_best)))
        max_cap_gap = max(max_cap_gap, gap)
        exact_hits = int((exact[i] >= 0.82).sum())
        sql_hits_n = int(sum(1 for h in hits if float(h["score"]) >= 0.82))
        # exact rows >=0.82 not present in the SQL 128-row pool
        sql_ids = {h["corpus_id"] for h in hits}
        missing = 0
        for row_j, score in enumerate(exact[i]):
            if score >= 0.82 and corpus_meta[row_j]["id"] not in sql_ids:
                missing += 1
        total_exact_hits_082 += exact_hits
        total_sql_hits_082 += sql_hits_n
        total_missing_082 += missing
        rows.append({
            "query_index": i,
            "exact_hits_ge_082": exact_hits,
            "sql_pool_hits_ge_082": sql_hits_n,
            "exact_hits_missing_from_pool_ge_082": missing,
            "max_capability_max_score_gap": gap,
        })
    return {
        "rows": rows,
        "max_capability_max_score_gap": max_cap_gap,
        "exact_hits_ge_082": total_exact_hits_082,
        "sql_pool_hits_ge_082": total_sql_hits_082,
        "exact_hits_missing_from_pool_ge_082": total_missing_082,
    }


# ── threshold construction ──────────────────────────────────────────────────
def aggregate_capability_maxima(hits: list[dict],
                                cap_ids: list[str]) -> dict[str, float]:
    """Production aggregation: one maximum score per capability per query.

    Duplicate corpus hits for the same capability (standard and similar lanes,
    multiple rows) never produce duplicate candidate cards.
    """
    best = {c: -math.inf for c in cap_ids}
    for h in hits:
        cid = h["capability_id"]
        if cid in best:
            best[cid] = max(best[cid], float(h["score"]))
    return best


def make_record(row: dict, threshold: float, cards: dict[str, str],
                cap_ids: list[str], cap_scores: np.ndarray,
                src_sha: str) -> dict:
    tag = THRESHOLD_TAGS[threshold]
    gold_cap = row["_capability_id"] if row["_is_action"] else None
    cap_score = {c: float(cap_scores[i]) for i, c in enumerate(cap_ids)}
    retrieved = [c for c in cap_ids if cap_score[c] >= threshold]
    retrieved.sort(key=lambda c: (-cap_score[c], c))
    top3 = retrieved[:K_MAX]
    k = len(top3)

    label: str
    reason: str
    detail: str
    if not row["_is_action"]:
        label, reason = REJECT_LABEL, "no_action"
        detail = (row["_review_status"] or "source_no_action_unverified")
    elif k == 0:
        label, reason = REJECT_LABEL, "no_candidates"
        detail = "no_capability_above_threshold"
    elif gold_cap in top3:
        label, reason = gold_cap, "gold_in_candidates"
        detail = "gold_in_top3"
    else:
        label, reason = REJECT_LABEL, "gold_not_retrieved"
        detail = ("gold_below_threshold" if cap_score.get(gold_cap, -1.0) < threshold
                  else "gold_outside_top3")

    gold_rank = None
    if gold_cap is not None and cap_score.get(gold_cap, -1.0) >= threshold:
        gold_rank = retrieved.index(gold_cap) + 1

    base_order = top3 + [REJECT_LABEL]
    base_options = [cards[c] for c in top3] + [REJECT_CARD]
    # V2 rule: append REJECT after the capability cards, then shuffle the whole
    # option list with a per-row deterministic seed keyed by the V2 row id.
    out_id = f"{row['_source_record_key']}-{V2_PART}"
    perm = list(range(len(base_order)))
    random.Random(f"{SHUFFLE_SEED_PREFIX}:{out_id}").shuffle(perm)
    order = [base_order[i] for i in perm]
    options = [base_options[i] for i in perm]
    gold_index = order.index(label)
    reject_index = order.index(REJECT_LABEL)

    rec = {
        "id": out_id,
        "source_query_id": row["_source_record_key"],
        "split": row["_split"],
        "part": V2_PART,
        "lang": row["_lang"],
        "lang_raw": row["_lang_raw"],
        "state": row["_query"],
        "order": order,
        "options": options,
        "gold_index": gold_index,
        "target_kind": "capability" if label != REJECT_LABEL else "reject",
        "gold_capability_id": gold_cap,
        "label": label,
        "label_reason": reason,
        "label_reason_detail": detail,
        "reject_index": reject_index,
        "shuffle": {
            "seed_prefix": SHUFFLE_SEED_PREFIX,
            "permutation": perm,
        },
        "src_sha": src_sha,
        "threshold": threshold,
        "candidate_count": k,
        "option_count": k + 1,
        "candidate_capability_ids": top3,
        "retrieved_capabilities": retrieved,
        "retrieval_scores": {c: cap_score[c] for c in retrieved},
        "gold_score": (cap_score.get(gold_cap) if gold_cap is not None else None),
        "gold_retrieval_rank": gold_rank,
        "is_action": bool(row["_is_action"]),
        "decision_state": "direct_reject" if k == 0 else "choose",
        "parameters": row["_parameters"],
        "source": row["_source"],
        "review_status": row["_review_status"],
        "query_family": row["_family_sig"],
        "query_family_id": row["_family_key"],
        "csv_line": row["_csv_line"],
        "source_row_id": row["_row_id"],
        "source_file": row["_source_file"],
        "source_row": row["_source_row"],
    }
    return rec


def to_v2_row(rec: dict) -> dict:
    """Project one construction record onto the frozen V2 14-key training row.

    The only V4 difference is the option count: K capability cards + one REJECT
    (K+1 total, 1..4). Label semantics are V2's: a reject row carries
    ``gold_capability_id = None`` and points at the REJECT slot.
    """
    reject = rec["target_kind"] == "reject"
    row = {
        "id": rec["id"],
        "source_query_id": rec["source_query_id"],
        "split": rec["split"],
        "part": rec["part"],
        "lang": rec["lang"],
        "state": rec["state"],
        "order": list(rec["order"]),
        "options": list(rec["options"]),
        "gold_index": int(rec["gold_index"]),
        "target_kind": rec["target_kind"],
        "gold_capability_id": None if reject else rec["label"],
        "reject_index": int(rec["reject_index"]),
        "shuffle": {
            "seed_prefix": rec["shuffle"]["seed_prefix"],
            "permutation": list(rec["shuffle"]["permutation"]),
        },
        "src_sha": rec["src_sha"],
    }
    if tuple(row) != V2_ROW_KEYS:
        raise RuntimeError(f"{rec['id']}: V2 projection key drift")
    return row


def to_audit_row(rec: dict) -> dict:
    """Construction-only metadata; never part of the training row."""
    source = {"source_gold_capability_id": rec["gold_capability_id"]}
    out = {k: rec[k] for k in AUDIT_KEYS if k != "source_gold_capability_id"}
    out.update(source)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


CSV_FIELDS = [
    "id", "source_query_id", "split", "part", "lang", "lang_raw", "state",
    "order", "options", "gold_index", "target_kind", "gold_capability_id",
    "label", "label_reason", "label_reason_detail", "reject_index", "shuffle",
    "src_sha", "threshold", "candidate_count", "option_count",
    "candidate_capability_ids", "retrieved_capabilities", "retrieval_scores",
    "gold_score", "gold_retrieval_rank", "is_action", "decision_state",
    "parameters", "source", "review_status", "query_family", "query_family_id",
    "csv_line", "source_row_id", "source_file", "source_row",
]


def csv_value(key: str, value: Any) -> Any:
    if key in {"order", "options", "candidate_capability_ids",
               "retrieved_capabilities", "retrieval_scores", "parameters",
               "shuffle"}:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: csv_value(k, r[k]) for k in CSV_FIELDS})


# ── stats ───────────────────────────────────────────────────────────────────
def dist(counter: Counter, denominator: int) -> dict:
    return {str(k): {"count": int(counter.get(k, 0)),
                     "pct": (counter.get(k, 0) / denominator * 100.0
                             if denominator else None)}
            for k in sorted(counter, key=lambda x: str(x))}


def k_dist(rows: list[dict]) -> dict:
    """K=0/1/2/3 always present, including zero counts."""
    counts = Counter(r["candidate_count"] for r in rows)
    return {str(k): {"count": int(counts.get(k, 0)),
                     "pct": (counts.get(k, 0) / len(rows) * 100.0
                             if rows else None)}
            for k in range(K_MAX + 1)}


def group_metrics(rows: list[dict]) -> dict:
    action = [r for r in rows if r["is_action"]]
    reject = [r for r in rows if not r["is_action"]]
    gold_in_cand = sum(1 for r in action
                       if r["gold_retrieval_rank"] is not None)
    gold_top1 = sum(1 for r in action if r["gold_retrieval_rank"] == 1)
    gold_top3 = sum(1 for r in action
                    if r["gold_retrieval_rank"] is not None
                    and r["gold_retrieval_rank"] <= 3)
    reasons = Counter(r["label_reason"] for r in rows)
    no_action_by_status = Counter(
        r["review_status"] or "unverified" for r in reject)
    return {
        "rows": len(rows),
        "action_rows": len(action),
        "no_action_rows": len(reject),
        "k_distribution": k_dist(rows),
        "mean_k": (sum(r["candidate_count"] for r in rows) / len(rows)
                   if rows else None),
        "label_distribution": dist(Counter(r["label"] for r in rows), len(rows)),
        "label_reason_distribution": dist(reasons, len(rows)),
        "gold_in_candidates": gold_in_cand,
        "gold_in_candidates_pct": (gold_in_cand / len(action) * 100.0
                                   if action else None),
        "gold_top1": gold_top1,
        "gold_top1_pct": (gold_top1 / len(action) * 100.0 if action else None),
        "gold_top3": gold_top3,
        "gold_top3_pct": (gold_top3 / len(action) * 100.0 if action else None),
        "gold_not_retrieved": reasons.get("gold_not_retrieved", 0),
        "gold_not_retrieved_pct": (reasons.get("gold_not_retrieved", 0)
                                   / len(action) * 100.0 if action else None),
        "no_candidates": reasons.get("no_candidates", 0),
        "no_candidates_pct": (reasons.get("no_candidates", 0)
                              / len(action) * 100.0 if action else None),
        "no_action": len(reject),
        "no_action_pct": (len(reject) / len(rows) * 100.0 if rows else None),
        "no_action_by_review_status": dict(no_action_by_status),
        "k_by_split": {s: k_dist([r for r in rows if r["split"] == s])
                       for s in SPLITS},
        "split_counts": {s: sum(1 for r in rows if r["split"] == s)
                         for s in SPLITS},
    }


def per_key_metrics(rows: list[dict], key_fn) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[str(key_fn(r))].append(r)
    return {k: group_metrics(v) for k, v in sorted(buckets.items())}


def per_cap_metrics(rows: list[dict], cap_ids: list[str]) -> dict:
    out = {}
    for cap in cap_ids:
        sub = [r for r in rows if r["gold_capability_id"] == cap]
        m = group_metrics(sub)
        m["source_rows"] = len(sub)
        m["split_counts"] = {s: sum(1 for r in sub if r["split"] == s)
                             for s in SPLITS}
        m["lang_counts"] = {lang: sum(1 for r in sub if r["lang"] == lang)
                            for lang in ("zh", "en", "mixed", "unknown")}
        m["k0"] = sum(1 for r in sub if r["candidate_count"] == 0)
        m["reject_label"] = sum(1 for r in sub
                                if r["label"] == REJECT_LABEL)
        out[cap] = m
    # no-action bucket
    sub = [r for r in rows if not r["is_action"]]
    m = group_metrics(sub)
    m["source_rows"] = len(sub)
    m["split_counts"] = {s: sum(1 for r in sub if r["split"] == s)
                         for s in SPLITS}
    m["lang_counts"] = {lang: sum(1 for r in sub if r["lang"] == lang)
                        for lang in ("zh", "en", "mixed", "unknown")}
    m["k0"] = sum(1 for r in sub if r["candidate_count"] == 0)
    m["reject_label"] = len(sub)
    out[REJECT_LABEL] = m
    return out


def comparison(records: dict[float, list[dict]], low: float,
               high: float) -> dict:
    by_id_low = {r["source_row_id"]: r for r in records[low]}
    by_id_high = {r["source_row_id"]: r for r in records[high]}
    ids = sorted(set(by_id_low) & set(by_id_high))
    added_candidates = 0
    added_cards = 0
    q_added_candidates = 0
    q_added_cards = 0
    added_by_cap = Counter()
    added_by_lang = Counter()
    for i in ids:
        a, b = by_id_low[i], by_id_high[i]
        ca, cb = len(a["retrieved_capabilities"]), len(b["retrieved_capabilities"])
        ka, kb = a["candidate_count"], b["candidate_count"]
        added_candidates += ca - cb
        added_cards += ka - kb
        if ca > cb:
            q_added_candidates += 1
        if ka > kb:
            q_added_cards += 1
        extra = set(a["retrieved_capabilities"]) - set(b["retrieved_capabilities"])
        for c in extra:
            added_by_cap[c] += 1
            added_by_lang[a["lang"]] += 1

    def pct_change(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return a - b

    low_m = group_metrics(records[low])
    high_m = group_metrics(records[high])
    return {
        "low_threshold": low,
        "high_threshold": high,
        "same_sample_count": len(ids),
        "added_capability_candidates_total": added_candidates,
        "queries_with_added_capability_candidate": q_added_candidates,
        "added_card_slots_total": added_cards,
        "queries_with_added_card": q_added_cards,
        "added_candidates_by_capability": dict(added_by_cap),
        "added_candidates_by_language": dict(added_by_lang),
        "gold_top1_delta_pp": pct_change(low_m["gold_top1_pct"],
                                        high_m["gold_top1_pct"]),
        "gold_top3_delta_pp": pct_change(low_m["gold_top3_pct"],
                                        high_m["gold_top3_pct"]),
        "gold_not_retrieved_delta_pp": pct_change(
            low_m["gold_not_retrieved_pct"], high_m["gold_not_retrieved_pct"]),
        "no_candidates_delta_pp": pct_change(
            low_m["no_candidates_pct"], high_m["no_candidates_pct"]),
        "mean_k_delta": pct_change(low_m["mean_k"], high_m["mean_k"]),
        "k_distribution_low": low_m["k_distribution"],
        "k_distribution_high": high_m["k_distribution"],
    }


# ── report helpers ──────────────────────────────────────────────────────────
def write_source_audit(ctx: dict) -> None:
    rows = ctx["rows"]
    cap_counts = Counter(r["_capability_id"] if r["_is_action"] else "REJECT"
                         for r in rows)
    raw_lang = Counter(r["_lang_raw"] or "(blank)" for r in rows)
    lang = Counter(r["_lang"] for r in rows)
    src = Counter(r["_source"] for r in rows)
    review = Counter(r["_review_status"] or "(blank)" for r in rows)
    act = sum(1 for r in rows if r["_is_action"])
    rej = len(rows) - act
    param_errors = sum(1 for r in rows if r["_parameters_error"])
    lines = [
        "# Source Audit",
        "",
        f"- Source file: `{SOURCE_CSV.relative_to(REPO_ROOT)}`",
        f"- SHA-256: `{ctx['src_sha']}`",
        f"- Bytes: {SOURCE_CSV.stat().st_size}",
        f"- Rows: {len(rows)}",
        f"- Fields: {', '.join(ctx['fields'])}",
        f"- Action rows: {act}; No-Action/reject rows: {rej}",
        f"- Parameter JSON parse errors: {param_errors}",
        "",
        "## Capability distribution (source labels)",
        "",
        md_table(["Label", "Rows", "Share"],
                 [[k, v, pct(v, len(rows))] for k, v in sorted(cap_counts.items())]),
        "",
        "## Language",
        "",
        md_table(["Raw language", "Rows"],
                 [[k, v] for k, v in sorted(raw_lang.items())]),
        "",
        md_table(["Resolved language", "Rows", "Share"],
                 [[k, v, pct(v, len(rows))] for k, v in sorted(lang.items())]),
        "",
        "Resolution rule: explicit `zh`/`en` is kept; blank language is derived "
        "from script (Han only -> zh, Latin only -> en, Han+Latin -> mixed, "
        "neither -> unknown). Mixed/unknown are never folded into zh/en.",
        "",
        "## Source and review status",
        "",
        md_table(["Source", "Rows"], [[k, v] for k, v in sorted(src.items())]),
        "",
        md_table(["Review status", "Rows"],
                 [[k, v] for k, v in sorted(review.items())]),
        "",
        "## No-Action / REJECT provenance",
        "",
        "No row in the source is marked as a human-confirmed real No-Action. "
        "The reject rows split into:",
        "",
        md_table(["Review status", "Rows"],
                 [[k, v] for k, v in sorted(
                     Counter(r["_review_status"] or "unverified"
                             for r in rows if not r["_is_action"]).items())]),
        "",
        "Synthetic/rule-checked rows are reported separately and are never "
        "described as real user No-Action.",
        "",
        "## Standard / similar corpus exclusion",
        "",
        f"- Corpus: {ctx['corpus']['std_count']} standard + "
        f"{ctx['corpus']['sim_count']} similar rows "
        f"({ctx['corpus']['std_vec_count']} + "
        f"{ctx['corpus']['sim_vec_count']} embedded).",
        f"- Corpus SHA-256: `{ctx['corpus']['corpus_sha']}`",
        f"- Source rows matching a standard query: "
        f"{sum(1 for r in rows if r['_is_standard_query'])}",
        f"- Source rows matching a similar query: "
        f"{sum(1 for r in rows if r['_is_similar_query'])}",
        f"- Source rows matching both: "
        f"{sum(1 for r in rows if r['_is_standard_query'] and r['_is_similar_query'])}",
        f"- Unique standard corpus rows matched in source: "
        f"{len(set(ctx['corpus']['std_norm']) & {r['_norm'] for r in rows})}"
        f" / {ctx['corpus']['std_count']}",
        f"- Unique similar corpus rows matched in source: "
        f"{len(set(ctx['corpus']['sim_norm']) & {r['_norm'] for r in rows})}"
        f" / {ctx['corpus']['sim_count']}",
        f"- Eligible evaluation rows after all exclusions: {len(ctx['eligible'])}",
        "",
        md_table(["Exclusion reason", "Rows"],
                 [[k, v] for k, v in sorted(ctx["exclusion_reasons"].items())]),
        "",
        "Exclusion reasons are not additive when several apply to one row; "
        "the unique excluded total is "
        f"{len(rows) - len(ctx['eligible'])}.",
        "",
        "No parent/derived-record ID exists in the source CSV or the live "
        "corpus tables, so only exact/normalised equality can be used to "
        "identify standard/similar records. Rows that are merely derived from "
        "a corpus query (but not textually identical) are not silently "
        "removed; they remain eligible and are flagged as a limitation.",
        "",
        "## Database / embedding environment",
        "",
        f"- DB: `{ctx['db_summary']['db_version']}`",
        f"- Embedding profile: `{json.dumps(ctx['db_summary']['profile'], ensure_ascii=False)}`",
        f"- Embedding service: `{EMBED_URL}`",
        f"- Model SHA-256: `{ctx['db_summary'].get('model_sha', 'unavailable')}`",
        f"- ANN pool per lane: {ANN_POOL}",
        f"- Thresholds: {', '.join(str(t) for t in THRESHOLDS)}",
        "",
        "## Known source-quality caveats",
        "",
        "- The source `language` column is blank for 29,241 rows; those rows "
        "are handled by the explicit resolution rule above.",
        "- 1,000 reject rows are `synthetic_needs_validation`; 2,000 are "
        "`rule_checked_pending_human_review`. Neither category is a "
        "human-confirmed real No-Action.",
        "- Source labels include 18 capabilities, including `cap-research`, "
        "which production chat-plane filtering would exclude. This dataset "
        "experiment deliberately includes all 18 capabilities.",
    ]
    (RUN_DIR / "reports" / "source_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_logic_report(ctx: dict) -> None:
    lines = [
        "# Dataset Generation Logic",
        "",
        "## Stage A - source load",
        "",
        f"- Loaded `{SOURCE_CSV.relative_to(REPO_ROOT)}` with `utf-8-sig`.",
        f"- SHA-256 `{ctx['src_sha']}`.",
        f"- Assigned stable `source_row_id` = CSV data-row index and "
        f"`csv_line` = physical line number.",
        f"- Parsed `is_action`, `capability_id`, `language`, `parameters`, "
        f"`source`, `review_status`, `source_file`, `source_row`.",
        "",
        "## Stage B - eligible evaluation queries",
        "",
        "1. Normalise query text with NFKC, strip, casefold and whitespace "
        "collapse.",
        "2. Compare each normalised query against the live standard-query and "
        "similar-query corpora.",
        "3. Exclude rows that are a standard query, a similar query, or both.",
        "4. Exclude empty queries, invalid `is_action`, Action rows without a "
        "valid capability, No-Action rows carrying a capability, and "
        "normalised duplicate queries (keep the first deterministic row).",
        "5. Standard/similar corpus rows remain in the database and are still "
        "used for recall; only their use as evaluation samples is prohibited.",
        "",
        "No source row is silently dropped: every row is written to "
        "`data/sample_manifest.csv` with `eligible` and `exclude_reason`.",
        "",
        "## Stage C - split assignment",
        "",
        "Stratification key: `source gold label x resolved language`, where "
        "the gold label is the source capability id for Action rows and "
        "`REJECT` for No-Action rows.",
        "",
        "Integer allocation:",
        "",
        "- n=1 -> train=1; n=2 -> train=1, test=1;",
        "- n>=3 -> val=max(1, round(0.05n)), test=max(1, round(0.20n)), "
        "train=remainder.",
        "",
        "This guarantees non-empty test/val for small groups and converges to "
        "75/20/5 for large groups. Actual counts and deviations are reported.",
        "",
        "Query families are computed with a fixed heuristic: NFKC/casefold, "
        "quoted spans -> `<q>`, digit runs -> `<n>`; queries with >=3 "
        "whitespace tokens use the first 5 tokens, otherwise the first 12 "
        "characters. Families are assigned as indivisible units by a "
        "deterministic greedy allocator that minimises squared deviation from "
        "the group targets. A family never crosses train/test/val.",
        "",
        "One split assignment is computed once and reused by all three "
        "threshold versions.",
        "",
        "## Stage D - shared real recall",
        "",
        "For every selected query the script calls the live embedding service "
        f"`{EMBED_URL}` (model from `app_settings.embedding_profile`) and then "
        "executes the real two-lane SQL/ANN recall against PostgreSQL/pgvector:",
        "",
        "- standard lane: `capability_standard_queries`, ANN pool 64;",
        "- similar lane: `capability_similar_queries` joined to its standard "
        "query, ANN pool 64;",
        "- no `enabled`, `status`, `intent_kind` or UI-hidden filter is applied "
        "(all 18 capabilities participate by explicit experiment rule);",
        "- scores are `1 - cosine_distance`, stored unrounded.",
        "",
        f"Hits with score >= {EVIDENCE_FLOOR} are written to "
        "`data/shared_retrieval_evidence.jsonl`. Hits below that floor cannot "
        "affect any threshold >= 0.82; this floor is stated explicitly.",
        "",
        "Per query and capability, the maximum score is kept. This is the same "
        "aggregation the production candidate aggregation uses.",
        "",
        "## Stage E - three threshold versions",
        "",
        "For each threshold t in 0.82, 0.85, 0.88:",
        "",
        "1. Keep capability maxima with `score >= t`.",
        "2. Sort deterministically by `(-score, capability_id)`.",
        "3. Take at most Top 3; never pad.",
        "4. K = number of selected capabilities (REJECT excluded).",
        "5. Build K capability cards + exactly one REJECT card.",
        "6. Label:",
        "   - No-Action source -> REJECT / `no_action`;",
        "   - K=0 -> REJECT / `no_candidates`;",
        "   - gold capability in Top 3 -> that capability / "
        "`gold_in_candidates`;",
        "   - otherwise -> REJECT / `gold_not_retrieved` "
        "(`gold_below_threshold` or `gold_outside_top3`).",
        "7. Shuffle the K+1 options deterministically with seed prefix "
        f"`{SHUFFLE_SEED_PREFIX}`, recomputing `gold_index` and "
        "`reject_index`.",
        "",
        "Label precedence: source No-Action is evaluated before K=0, so a "
        "No-Action row that also has K=0 is recorded as `no_action`, not "
        "`no_candidates`. The three REJECT provenances are never merged.",
        "",
        "## Stage F - V2-format output",
        "",
        "`v2_train.jsonl`, `v2_val.jsonl` and `v2_test.jsonl` are the training "
        "files. Each row has EXACTLY the frozen V2 14 keys: `id`, "
        "`source_query_id`, `split`, `part`, `lang`, `state`, `order`, "
        "`options`, `gold_index`, `target_kind`, `gold_capability_id`, "
        "`reject_index`, `shuffle`, `src_sha`. No threshold, candidate-count "
        "or label-provenance field enters a training row.",
        "",
        "The only contract difference from frozen V2 is the option count: "
        "`order`/`options` carry K capability cards plus exactly one `REJECT` "
        "(K+1 options, 1..4). `part` is `p2` (real recall) for every row. "
        "`gold_index` points at the label; a reject row has "
        "`gold_capability_id = null`. The per-row shuffle uses the V2 seed "
        "prefix `reject-shuffle-v1` over the row id, and REJECT is shuffled "
        "with the capability cards exactly as in V2.",
        "",
        "Threshold, candidate scores, label provenance and the original source "
        "gold are written to `construction_audit.jsonl`, joined by `id`. "
        "`{split}.csv` is a human-readable audit export, not a training input.",
        "",
        "K=0 rows carry one REJECT option and are also written to "
        "`direct_reject.jsonl` in each threshold directory for a separate "
        "direct-reject path.",
        "",
        "## Stage G - statistics and validation",
        "",
        "Statistics are computed from the generated files after all three "
        "threshold versions are written. An independent validator re-reads "
        "the files, re-derives candidates from shared evidence, checks "
        "labels/options/indexes, split consistency, family isolation and "
        "schema integrity. Boundary tests are separate from real-sample "
        "counts.",
    ]
    (RUN_DIR / "reports" / "dataset_generation_logic.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_distribution_report(ctx: dict) -> None:
    stats = ctx["stats"]
    lines = [
        "# Dataset Distribution Report",
        "",
        f"Run: `{ctx['run_id']}`; mode: `{ctx['mode']}`; "
        f"selected evaluation rows: {ctx['n_selected']}.",
        "",
        "All percentages name their denominator. K counts only capability "
        "candidates; REJECT is never counted in K.",
        "",
        "## Source eligibility",
        "",
        f"- Source rows: {len(ctx['rows'])}",
        f"- Eligible rows: {len(ctx['eligible'])}",
        f"- Rows used in this run: {ctx['n_selected']} "
        f"({'50-row smoke selection' if ctx['mode'] == 'smoke' else 'full'}).",
        f"- Excluded rows: {len(ctx['rows']) - len(ctx['eligible'])}",
        "",
        "## Overall K distribution",
        "",
        md_table(
            ["Threshold", "K", "Samples", "Share (denominator = all rows at "
             "this threshold)"],
            [[t, k, v["count"], pct(v["count"], stats[t]["rows"])]
             for t in THRESHOLDS
             for k in ["0", "1", "2", "3"]
             for v in [stats[t]["k_distribution"].get(k, {"count": 0})]]),
        "",
        "## K distribution by split",
        "",
        md_table(
            ["Threshold", "Split", "K", "Samples",
             "Share (denominator = split rows)"],
            [[t, s, k, v["count"],
              pct(v["count"], stats[t]["split_counts"][s])]
             for t in THRESHOLDS for s in SPLITS
             for k in ["0", "1", "2", "3"]
             for v in [stats[t]["k_by_split"][s].get(k, {"count": 0})]]),
        "",
        "## Labels and REJECT provenance",
        "",
        md_table(
            ["Threshold", "Label / reason", "Count",
             "Share (denominator = all rows)"],
            [[t, k, v["count"], pct(v["count"], stats[t]["rows"])]
             for t in THRESHOLDS
             for k, v in stats[t]["label_reason_distribution"].items()]),
        "",
        "`no_action` is source No-Action; `no_candidates` is an Action query "
        "with K=0; `gold_not_retrieved` is an Action query with K>0 whose gold "
        "capability is not in Top 3. The three are never merged.",
        "",
        "## Gold retrieval metrics (Action rows only)",
        "",
        md_table(
            ["Threshold", "Action rows", "Gold in candidates", "Gold Top-1",
             "Gold Top-3", "gold_not_retrieved", "no_candidates"],
            [[t, stats[t]["action_rows"],
              f"{stats[t]['gold_in_candidates']} "
              f"({pct(stats[t]['gold_in_candidates'], stats[t]['action_rows'])})",
              f"{stats[t]['gold_top1']} "
              f"({pct(stats[t]['gold_top1'], stats[t]['action_rows'])})",
              f"{stats[t]['gold_top3']} "
              f"({pct(stats[t]['gold_top3'], stats[t]['action_rows'])})",
              f"{stats[t]['gold_not_retrieved']} "
              f"({pct(stats[t]['gold_not_retrieved'], stats[t]['action_rows'])})",
              f"{stats[t]['no_candidates']} "
              f"({pct(stats[t]['no_candidates'], stats[t]['action_rows'])})"]
             for t in THRESHOLDS]),
        "",
        "Denominator for every percentage in this table is the number of "
        "Action rows at that threshold.",
        "",
        "## Per-capability distribution",
        "",
        "For each threshold, denominators are the source Action rows of that "
        "capability (No-Action is reported as a separate `REJECT` bucket).",
        "",
    ]
    for t in THRESHOLDS:
        lines += [f"### Threshold {t}", "",
                  md_table(
                      ["Gold capability", "Source rows", "zh", "en", "mixed",
                       "Train", "Test", "Val",
                       "K=0", "K=1", "K=2", "K=3",
                       "Gold Top-3", "gold_not_retrieved", "REJECT label"],
                      [[cap, m["source_rows"], m["lang_counts"]["zh"],
                        m["lang_counts"]["en"], m["lang_counts"]["mixed"],
                        m["split_counts"]["train"],
                        m["split_counts"]["test"], m["split_counts"]["val"],
                        m["k_distribution"]["0"]["count"],
                        m["k_distribution"]["1"]["count"],
                        m["k_distribution"]["2"]["count"],
                        m["k_distribution"]["3"]["count"],
                        m["gold_top3"], m["gold_not_retrieved"],
                        m["reject_label"]]
                       for cap, m in stats[t]["per_cap"].items()]),
                  ""]
    lines += [
        "## Per-language distribution",
        "",
    ]
    for t in THRESHOLDS:
        lines += [f"### Threshold {t}", "",
                  md_table(
                      ["Language", "Rows", "K=0", "K=1", "K=2", "K=3",
                       "Gold Top-3 (Action)", "gold_not_retrieved"],
                      [[lang, m["rows"],
                        m["k_distribution"].get("0", {}).get("count", 0),
                        m["k_distribution"].get("1", {}).get("count", 0),
                        m["k_distribution"].get("2", {}).get("count", 0),
                        m["k_distribution"].get("3", {}).get("count", 0),
                        f"{m['gold_top3']} "
                        f"({pct(m['gold_top3'], m['action_rows'])})",
                        m["gold_not_retrieved"]]
                       for lang, m in stats[t]["per_lang"].items()]),
                  ""]
    lines += [
        "## Per-source distribution",
        "",
    ]
    for t in THRESHOLDS:
        lines += [f"### Threshold {t}", "",
                  md_table(
                      ["Source", "Rows", "Action rows", "K=0", "K=1", "K=2",
                       "K=3", "Gold Top-3"],
                      [[s, m["rows"], m["action_rows"],
                        m["k_distribution"].get("0", {}).get("count", 0),
                        m["k_distribution"].get("1", {}).get("count", 0),
                        m["k_distribution"].get("2", {}).get("count", 0),
                        m["k_distribution"].get("3", {}).get("count", 0),
                        m["gold_top3"]]
                       for s, m in stats[t]["per_source"].items()]),
                  ""]
    lines += [
        "## Split ratio check",
        "",
        "Targets are per `gold label x language` group. Because the smoke run "
        "has mostly singleton groups, the global smoke ratio deviates from "
        "75/20/5; per-group targets and actual counts are listed in "
        "`reports/split_integrity_report.md`.",
        "",
        md_table(
            ["Threshold", "Train", "Test", "Val", "Actual ratio"],
            [[t, stats[t]["split_counts"]["train"],
              stats[t]["split_counts"]["test"],
              stats[t]["split_counts"]["val"],
              f"{pct(stats[t]['split_counts']['train'], stats[t]['rows'])} / "
              f"{pct(stats[t]['split_counts']['test'], stats[t]['rows'])} / "
              f"{pct(stats[t]['split_counts']['val'], stats[t]['rows'])}"]
             for t in THRESHOLDS]),
        "",
    ]
    (RUN_DIR / "reports" / "dataset_distribution_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(ctx: dict) -> None:
    comp = ctx["comparison"]
    lines = [
        "# Threshold Comparison Report",
        "",
        "All three versions use the same eligible sample list and the same "
        "split assignment. Thresholds only change filtering/aggregation/Top-3 "
        "and therefore candidates and labels.",
        "",
        "## Overall comparison",
        "",
        md_table(
            ["Metric", "0.82", "0.85", "0.88", "Denominator"],
            [["Rows", *[ctx["stats"][t]["rows"] for t in THRESHOLDS],
              "same selected sample set"],
             ["Mean K", *[round(ctx["stats"][t]["mean_k"], 4)
                         for t in THRESHOLDS], "selected rows"],
             ["Gold Top-1", *[f"{ctx['stats'][t]['gold_top1']} "
                              f"({pct(ctx['stats'][t]['gold_top1'], ctx['stats'][t]['action_rows'])})"
                              for t in THRESHOLDS], "Action rows"],
             ["Gold Top-3", *[f"{ctx['stats'][t]['gold_top3']} "
                              f"({pct(ctx['stats'][t]['gold_top3'], ctx['stats'][t]['action_rows'])})"
                              for t in THRESHOLDS], "Action rows"],
             ["gold_not_retrieved", *[ctx["stats"][t]["gold_not_retrieved"]
                                      for t in THRESHOLDS], "Action rows"],
             ["no_candidates", *[ctx["stats"][t]["no_candidates"]
                                 for t in THRESHOLDS], "Action rows"],
             ["no_action", *[ctx["stats"][t]["no_action"]
                             for t in THRESHOLDS], "All rows"]],
        ),
        "",
        "## K distribution comparison",
        "",
        md_table(
            ["Threshold", "K=0", "K=1", "K=2", "K=3"],
            [[t, *[f"{v['count']} ({pct(v['count'], ctx['stats'][t]['rows'])})"
                   for k in ["0", "1", "2", "3"]
                   for v in [ctx["stats"][t]["k_distribution"].get(k, {"count": 0})]]]
             for t in THRESHOLDS]),
        "",
        "## Pairwise deltas",
        "",
    ]
    for key, c in comp.items():
        lines += [
            f"### {c['low_threshold']} vs {c['high_threshold']}",
            "",
            md_table(["Metric", "Value"], [
                ["Same sample rows", c["same_sample_count"]],
                ["Added capability candidates (sum over queries)",
                 c["added_capability_candidates_total"]],
                ["Queries with at least one added capability candidate",
                 c["queries_with_added_capability_candidate"]],
                ["Added card slots (K_low - K_high, sum)",
                 c["added_card_slots_total"]],
                ["Queries with at least one added card",
                 c["queries_with_added_card"]],
                ["Gold Top-1 delta (percentage points)",
                 round(c["gold_top1_delta_pp"], 4)
                 if c["gold_top1_delta_pp"] is not None else None],
                ["Gold Top-3 delta (percentage points)",
                 round(c["gold_top3_delta_pp"], 4)
                 if c["gold_top3_delta_pp"] is not None else None],
                ["gold_not_retrieved delta (percentage points)",
                 round(c["gold_not_retrieved_delta_pp"], 4)
                 if c["gold_not_retrieved_delta_pp"] is not None else None],
                ["no_candidates delta (percentage points)",
                 round(c["no_candidates_delta_pp"], 4)
                 if c["no_candidates_delta_pp"] is not None else None],
                ["Mean K delta", round(c["mean_k_delta"], 4)
                 if c["mean_k_delta"] is not None else None],
            ]),
            "",
            "Added candidates by capability:",
            "",
            md_table(["Capability", "Added count"],
                     [[k, v] for k, v in sorted(
                         c["added_candidates_by_capability"].items())] or
                     [["(none)", 0]]),
            "",
            "Added candidates by language:",
            "",
            md_table(["Language", "Added count"],
                     [[k, v] for k, v in sorted(
                         c["added_candidates_by_language"].items())] or
                     [["(none)", 0]]),
            "",
        ]
    lines += [
        "## Data bias and representativeness",
        "",
        "- The 50-row smoke sample is coverage-driven, not random: it takes one "
        "Chinese and one English row for every capability plus mixed-language "
        "and split-exercise rows. It therefore does NOT mirror source "
        "frequencies and must not be used to estimate production prevalence.",
        "- Added candidates at lower thresholds by language:",
        "",
        md_table(
            ["Pair", "en", "mixed", "zh", "Total"],
            [[f"{c['low_threshold']} vs {c['high_threshold']}",
              c["added_candidates_by_language"].get("en", 0),
              c["added_candidates_by_language"].get("mixed", 0),
              c["added_candidates_by_language"].get("zh", 0),
              c["added_capability_candidates_total"]]
             for c in comp.values()]),
        "",
        "- Source capability/language/source frequencies are reported in "
        "`source_audit.md`; the full run's selected-sample frequencies belong "
        "in `dataset_distribution_report.md`.",
        "",
        "## Findings",
        "",
        "- Lower thresholds mechanically add capability candidates and can "
        "increase K, but the smoke sample is not large enough to treat these "
        "deltas as a production threshold recommendation.",
        "- Any gold-recall gain must be weighed against added REJECT ambiguity, "
        "card length and model cost; no production threshold is changed here.",
    ]
    (RUN_DIR / "reports" / "threshold_comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_preliminary_reports(ctx: dict) -> None:
    lines = [
        "# Data Quality Report",
        "",
        "This build runs the checks below and writes the independent "
        "validation results in `validation_results.json`. The final "
        "authoritative section is regenerated by `validate_dataset.py`.",
        "",
        "## In-build checks",
        "",
        md_table(["Check", "Result", "Detail"],
                 [[k, "PASS" if v["ok"] else "FAIL", v["detail"]]
                  for k, v in ctx["checks"].items()]),
        "",
        "## Limitations",
        "",
        "- No row is a human-confirmed real No-Action; reject sources are "
        "synthetic/rule-checked.",
        "- Only exact/normalised corpus overlap can be excluded; the source "
        "has no parent/derived-record ID.",
        "- HNSW ANN is approximate. The exact-cosine audit reports any "
        ">=0.82 hit absent from the production-shaped 128-row pool.",
        "- Smoke distribution is not representative of the full dataset.",
    ]
    (RUN_DIR / "reports" / "data_quality_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    split_lines = [
        "# Split Integrity Report (preliminary)",
        "",
        "The independent validator regenerates this report from the output "
        "files.",
        "",
        md_table(["Group", "Target train/test/val", "Actual train/test/val"],
                 [[g, t, ctx["split_actual"].get(g)]
                  for g, t in sorted(ctx["split_targets"].items())]),
    ]
    (RUN_DIR / "reports" / "split_integrity_report.md").write_text(
        "\n".join(split_lines) + "\n", encoding="utf-8")

    (RUN_DIR / "reports" / "boundary_test_report.md").write_text(
        "# Boundary Test Report\n\nPending `test_boundary_cases.py`.\n",
        encoding="utf-8")
    (RUN_DIR / "reports" / "acceptance_report.md").write_text(
        "# Acceptance Report\n\nStatus: `PENDING_VALIDATION`\n\n"
        "The independent validator and boundary suite must complete before "
        "this status is final.\n",
        encoding="utf-8")


# ── manifests ───────────────────────────────────────────────────────────────
def write_config(ctx: dict) -> None:
    cfg = {
        "run_id": ctx["run_id"],
        "mode": ctx["mode"],
        "build_time_utc": ctx["started"],
        "source": {
            "path": str(SOURCE_CSV.relative_to(REPO_ROOT)),
            "bytes": SOURCE_CSV.stat().st_size,
            "sha256": ctx["src_sha"],
            "fields": ctx["fields"],
            "rows": len(ctx["rows"]),
        },
        "database": ctx["db_summary"],
        "corpus": {
            "standard_rows": ctx["corpus"]["std_count"],
            "similar_rows": ctx["corpus"]["sim_count"],
            "embedded_standard_rows": ctx["corpus"]["std_vec_count"],
            "embedded_similar_rows": ctx["corpus"]["sim_vec_count"],
            "sha256": ctx["corpus"]["corpus_sha"],
        },
        "exclusion_rules": {
            "standard_query": "exact/normalised query equality",
            "standard_query_similar": "exact/normalised query equality",
            "derived_from_standard_query": "not determinable: no parent/derived ID",
            "duplicate_query": "first deterministic row kept",
        },
        "recall": {
            "sql_lanes": ["standard", "similar"],
            "ann_pool_per_lane": ANN_POOL,
            "business_filters_applied": False,
            "capabilities_included": "all 18, including cap-research and hidden caps",
            "score": "1 - cosine_distance",
            "evidence_floor": EVIDENCE_FLOOR,
            "shared_across_thresholds": True,
        },
        "thresholds": list(THRESHOLDS),
        "aggregation": {
            "per_capability": "max score",
            "sort": "score DESC, capability_id ASC (deterministic tie-break)",
            "top_k": K_MAX,
            "padding": False,
            "reject": "one uppercase REJECT card appended after capability cards",
        },
        "labels": {
            "no_action": "REJECT",
            "no_candidates": "REJECT",
            "gold_in_top3": "gold capability",
            "gold_not_retrieved": "REJECT",
        },
        "language_rule": {
            "explicit": "keep source zh/en",
            "blank": "Han only -> zh; Latin only -> en; Han+Latin -> mixed; "
                     "neither -> unknown",
        },
        "query_family_rule": {
            "normalise": "NFKC, strip, casefold, whitespace collapse",
            "placeholders": "quoted spans -> <q>; digit runs -> <n>",
            "signature": "first 5 whitespace tokens if >=3 tokens, else first 12 chars",
        },
        "split": {
            "stratification_key": "gold label x resolved language",
            "ratios": {"train": 0.75, "test": 0.20, "val": 0.05},
            "integer_rule": "minimum representation for n<3; largest-remainder "
                            "style remainder with max(1, round()) for n>=3",
            "family_grouping": "whole family assigned to one split",
            "seed": SEED,
            "same_assignment_across_thresholds": True,
        },
        "schema": {
            "version": "V4-dynamic",
            "training_schema": "LayaChoice-v2 14-key JSONL",
            "training_keys": list(V2_ROW_KEYS),
            "part": V2_PART,
            "threshold_in_training_row": False,
            "construction_audit_file": "construction_audit.jsonl",
            "compatible_with_v2_loader": True,
            "v2_loader_modified": (
                "option-count/permutation/gold-index validation generalized "
                "from fixed 4 to the row's actual K+1 (1..4); schema, labels, "
                "REJECT construction and target semantics unchanged"),
            "option_slots": "K+1, K in {0,1,2,3} (1..4 options)",
            "reject_label": REJECT_LABEL,
            "card_source": (
                "production cap_router.card_renderer.render_card over the live "
                "Registry; byte-identical to V2/data/aligned, not V2/data base"),
        },
        "scripts": {
            "build_dataset.py": sha256_file(HERE / "build_dataset.py"),
            "validate_dataset.py": (sha256_file(HERE / "validate_dataset.py")
                                    if (HERE / "validate_dataset.py").exists()
                                    else None),
            "test_boundary_cases.py": (
                sha256_file(HERE / "test_boundary_cases.py")
                if (HERE / "test_boundary_cases.py").exists() else None),
        },
        "python": sys.version,
        "rebuild_command": (
            f"PYTHONPATH={REPO_ROOT / 'packages' / 'core'} "
            f".venv/Scripts/python.exe {HERE / 'build_dataset.py'} "
            f"--mode {ctx['mode']}"
        ),
    }
    (RUN_DIR / "config" / "generation_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cfg


def write_source_manifest(ctx: dict) -> None:
    rows = ctx["rows"]
    manifest = {
        "run_id": ctx["run_id"],
        "mode": ctx["mode"],
        "input": {
            "path": str(SOURCE_CSV.relative_to(REPO_ROOT)),
            "bytes": SOURCE_CSV.stat().st_size,
            "sha256": ctx["src_sha"],
            "rows": len(rows),
            "fields": ctx["fields"],
        },
        "counts": {
            "capabilities": dict(Counter(
                r["_capability_id"] if r["_is_action"] else REJECT_LABEL
                for r in rows)),
            "language_raw": dict(Counter(r["_lang_raw"] or "(blank)"
                                         for r in rows)),
            "language_resolved": dict(Counter(r["_lang"] for r in rows)),
            "source": dict(Counter(r["_source"] for r in rows)),
            "review_status": dict(Counter(r["_review_status"] or "(blank)"
                                          for r in rows)),
            "action": sum(1 for r in rows if r["_is_action"]),
            "no_action": sum(1 for r in rows if not r["_is_action"]),
            "parameter_parse_errors": sum(
                1 for r in rows if r["_parameters_error"]),
        },
        "no_action_note": (
            "No human-confirmed real No-Action rows exist in the source. "
            "Reject rows are synthetic_needs_validation or "
            "rule_checked_pending_human_review."),
        "corpus": {
            "standard_rows": ctx["corpus"]["std_count"],
            "similar_rows": ctx["corpus"]["sim_count"],
            "embedded_standard_rows": ctx["corpus"]["std_vec_count"],
            "embedded_similar_rows": ctx["corpus"]["sim_vec_count"],
            "sha256": ctx["corpus"]["corpus_sha"],
            "standard_rows_matched_in_source": len(
                set(ctx["corpus"]["std_norm"]) &
                {r["_norm"] for r in rows}),
            "similar_rows_matched_in_source": len(
                set(ctx["corpus"]["sim_norm"]) &
                {r["_norm"] for r in rows}),
        },
        "exclusion": {
            "eligible_rows": len(ctx["eligible"]),
            "selected_rows": ctx["n_selected"],
            "unique_excluded_rows": len(rows) - len(ctx["eligible"]),
            "reason_counts_non_additive": dict(ctx["exclusion_reasons"]),
        },
        "database": ctx["db_summary"],
        "selected_source_row_ids": [r["_row_id"] for r in ctx["selected"]],
        "selected_source_record_keys": [
            r["_source_record_key"] for r in ctx["selected"]],
    }
    (RUN_DIR / "manifests" / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def build_v2_manifest(ctx: dict, threshold: float, split_rows: dict[str, list[dict]],
                      row_order: dict[str, list[int]], data_dir: Path) -> dict:
    """A V2-compatible manifest for one threshold bundle.

    Everything V2's loader reads (bench, option contract, rows, calibration,
    source hashes) is present. The threshold and V4 audit pointers are extra
    metadata in this manifest, never in a training row.
    """
    train_rows = split_rows["train"]
    n_train = len(train_rows)
    order = list(range(n_train))
    random.Random(CALIB_SEED).shuffle(order)
    n_calib = min(CALIB_MAX, n_train // 10)
    calib_idx = sorted(order[:n_calib])
    train_idx = sorted(order[n_calib:])

    gold_slot = {}
    language = {}
    part_rows = {}
    for split in SPLITS:
        rows = split_rows[split]
        gold_slot[split] = {str(k): v for k, v in sorted(
            Counter(r["gold_index"] for r in rows).items())}
        language[split] = dict(Counter(r["lang"] for r in rows))
        part_rows[split] = dict(Counter(r["part"] for r in rows))

    source_files = {"source_csv": ctx["src_sha"]}
    for split in SPLITS:
        source_files[f"v2_{split}.jsonl"] = sha256_file(
            data_dir / f"v2_{split}.jsonl")
    source_files["construction_audit.jsonl"] = sha256_file(
        data_dir / "construction_audit.jsonl")

    return {
        "bench": "LayaChoice-v2",
        "dataset_variant": "V4-dynamic",
        "run_id": ctx["run_id"],
        "mode": ctx["mode"],
        "threshold": threshold,
        "formal_card_view": "B_noprov",
        "instructions": INSTRUCTIONS,
        "option_slots": {
            **DYNAMIC_OPTION_SLOTS,
            "observed": sorted({len(r["order"]) for r in train_rows}),
        },
        "training_schema_keys": list(V2_ROW_KEYS),
        "part": V2_PART,
        "construction_audit_file": "construction_audit.jsonl",
        "reject": {
            "label": REJECT_LABEL,
            "id": REJECT_LABEL,
            "card": REJECT_CARD,
            "meaning": "no listed capability correctly handles the request; "
                       "present on every row as exactly one option",
        },
        "option_order_rule": "per-row deterministic shuffle; seed prefix "
                             f"{SHUFFLE_SEED_PREFIX} over the row id; "
                             "option count = capability candidates + 1",
        "gold_slot_distribution": gold_slot,
        "source_files": source_files,
        "rows": {split: len(split_rows[split]) for split in SPLITS},
        "part_rows": part_rows,
        "language": language,
        "calibration": {
            "seed": CALIB_SEED, "source_train_rows": n_train,
            "calibration_rows": n_calib,
            "effective_train_rows": len(train_idx),
            "rule": "min(400, n_train // 10) over the FINAL (post-shuffle) "
                    "train order",
            "indices": calib_idx,
            "ids": [train_rows[i]["id"] for i in calib_idx],
        },
        "row_order": {
            "seed_prefix": ROW_SEED,
            "permutation": row_order,
            "note": "output slot p holds the pre-shuffle row permutation[p]",
        },
        "base_checkpoint": {
            "repo_id": "convaiinnovations/laya",
            "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
            "subfolder": "multilingual",
        },
        "smoke": ctx["mode"] == "smoke",
    }


def write_generation_manifest(ctx: dict, config: dict) -> None:
    out_files = []
    for p in RUN_DIR.rglob("*"):
        if p.is_file() and p.name not in {"generation_manifest.json",
                                          "output_hashes.json"}:
            out_files.append(str(p.relative_to(RUN_DIR)))
    manifest = {
        "run_id": ctx["run_id"],
        "mode": ctx["mode"],
        "started_utc": ctx["started"],
        "finished_utc": utc_now(),
        "duration_seconds": round(time.time() - ctx["t0"], 3),
        "seed": SEED,
        "thresholds": list(THRESHOLDS),
        "selected_rows": ctx["n_selected"],
        "eligible_rows": len(ctx["eligible"]),
        "source_rows": len(ctx["rows"]),
        "split_targets": ctx["split_targets"],
        "split_actual": ctx["split_actual"],
        "threshold_stats": ctx["stats"],
        "threshold_comparison": ctx["comparison"],
        "ann_coverage_audit": ctx["ann_audit"],
        "checks": ctx["checks"],
        "embedding": {
            "url": EMBED_URL,
            "profile": ctx["db_summary"]["profile"],
            "model_sha256": ctx["db_summary"].get("model_sha"),
            "batch": 16,
            "concurrency": 4,
            "dim": 1024,
        },
        "output_files_preliminary": sorted(out_files),
        "config": config,
        "scripts": config["scripts"],
    }
    (RUN_DIR / "manifests" / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


# ── main ────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    log_path = RUN_DIR / "logs" / "generation.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S"))
    logging.Formatter.converter = time.gmtime
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(handler.formatter)
    LOG.setLevel(logging.INFO)
    LOG.addHandler(handler)
    LOG.addHandler(console)


async def get_model_sha() -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "delveta-embedding", "python", "-c",
            "import hashlib;print(hashlib.sha256(open('/model/model.onnx','rb')"
            ".read()).hexdigest())",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode == 0:
            return out.decode().strip()
    except Exception:  # noqa: BLE001
        pass
    return "unavailable"


def validate_in_build(records: dict[float, list[dict]],
                      cap_ids: list[str], cards: dict[str, str]) -> dict:
    checks = {}

    def check(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    all_rows = [r for rows in records.values() for r in rows]

    bad_v2 = []
    for r in all_rows:
        v = to_v2_row(r)
        n = len(v["order"])
        if set(v) != set(V2_ROW_KEYS):
            bad_v2.append(r["id"])
            continue
        reject = v["target_kind"] == "reject"
        problems = (
            v["part"] != V2_PART,
            not (MIN_OPTION_SLOTS <= n <= MAX_OPTION_SLOTS),
            sorted(v["shuffle"]["permutation"]) != list(range(n)),
            reject and (v["gold_capability_id"] is not None
                        or v["gold_index"] != v["reject_index"]),
            not reject and (v["gold_capability_id"]
                            != v["order"][v["gold_index"]]
                            or v["gold_index"] == v["reject_index"]),
        )
        if any(problems):
            bad_v2.append(r["id"])
    check("v2_training_row_projection", not bad_v2,
          f"{len(bad_v2)} violations")
    check("all_18_capabilities_in_cards",
          len(cap_ids) == 18 and all(
              cards.get(cid, "").startswith(f"### {cid}") for cid in cap_ids),
          f"{len(cap_ids)} capability cards rendered")

    bad_dynamic = [r["id"] for r in all_rows
                   if r["option_count"] != r["candidate_count"] + 1
                   or len(r["order"]) != r["option_count"]
                   or len(r["options"]) != r["option_count"]]
    check("dynamic_option_count", not bad_dynamic,
          f"{len(bad_dynamic)} violations")

    bad_reject = [r["id"] for r in all_rows
                  if r["order"].count(REJECT_LABEL) != 1
                  or r["order"][r["reject_index"]] != REJECT_LABEL]
    check("exactly_one_reject", not bad_reject,
          f"{len(bad_reject)} violations")

    bad_label = [r["id"] for r in all_rows
                 if r["order"][r["gold_index"]] != r["label"]]
    check("gold_index_points_to_label", not bad_label,
          f"{len(bad_label)} violations")

    bad_k0 = [r["id"] for r in all_rows
              if r["candidate_count"] == 0 and
              (r["decision_state"] != "direct_reject"
               or r["order"] != [REJECT_LABEL])]
    check("k0_direct_reject", not bad_k0, f"{len(bad_k0)} violations")

    bad_k = [r["id"] for r in all_rows
             if r["candidate_count"] != len(
                 [x for x in r["order"] if x != REJECT_LABEL])]
    check("k_excludes_reject", not bad_k, f"{len(bad_k)} violations")

    bad_dup = [r["id"] for r in all_rows
               if len(set(x for x in r["order"] if x != REJECT_LABEL)) !=
               len([x for x in r["order"] if x != REJECT_LABEL])]
    check("no_duplicate_capability_option", not bad_dup,
          f"{len(bad_dup)} violations")

    bad_floor = []
    for r in all_rows:
        for cap, score in r["retrieval_scores"].items():
            if score < r["threshold"]:
                bad_floor.append(r["id"])
                break
    check("no_candidate_below_threshold", not bad_floor,
          f"{len(bad_floor)} violations")

    bad_gold = []
    for r in all_rows:
        if r["label_reason"] == "gold_in_candidates":
            if r["label"] != r["gold_capability_id"]:
                bad_gold.append(r["id"])
        elif r["label_reason"] in {"gold_not_retrieved", "no_candidates"}:
            if not r["is_action"] or r["label"] != REJECT_LABEL:
                bad_gold.append(r["id"])
        elif r["label_reason"] == "no_action":
            if r["is_action"] or r["label"] != REJECT_LABEL:
                bad_gold.append(r["id"])
    check("label_rules", not bad_gold, f"{len(bad_gold)} violations")

    return checks


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--run-id", default=RUN_DIR.name)
    args = ap.parse_args()

    setup_logging()
    ctx: dict[str, Any] = {
        "run_id": args.run_id,
        "mode": args.mode,
        "started": utc_now(),
        "t0": time.time(),
    }
    LOG.info("run_id=%s mode=%s", ctx["run_id"], ctx["mode"])

    data_dir = RUN_DIR / "data"
    for t in THRESHOLDS:
        (data_dir / THRESHOLD_TAGS[t]).mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "reports").mkdir(exist_ok=True)
    (RUN_DIR / "manifests").mkdir(exist_ok=True)

    rows, fields, src_sha = load_source()
    ctx.update({"rows": rows, "fields": fields, "src_sha": src_sha})
    LOG.info("source rows=%d sha=%s", len(rows), src_sha)

    conn = await asyncpg.connect(DB_DSN)
    try:
        db = await load_db(conn)
    finally:
        await conn.close()
    entries = build_entries(db)
    cap_ids = sorted(entries)
    if len(cap_ids) != 18:
        raise SystemExit(f"expected 18 capabilities, got {len(cap_ids)}")
    cards = {cid: render_card(entries[cid]) for cid in cap_ids}
    corpus = corpus_struct(db)
    corpus["std_vec_rows"] = db["std_vec"]
    corpus["sim_vec_rows"] = db["sim_vec"]
    eligible, exclusion_reasons = classify_rows(rows, corpus, set(cap_ids))

    model_sha = await get_model_sha()
    db_summary = {
        "db_version": db["db_version"],
        "profile": db["profile"],
        "model_sha": model_sha,
        "capabilities": len(cap_ids),
        "standard_rows": corpus["std_count"],
        "similar_rows": corpus["sim_count"],
        "indexes": db["indexes"],
    }
    ctx.update({"eligible": eligible, "exclusion_reasons": exclusion_reasons,
                "corpus": corpus, "db_summary": db_summary,
                "cap_ids": cap_ids, "cards": cards})
    LOG.info("eligible=%d excluded=%d", len(eligible),
             len(rows) - len(eligible))

    if args.mode == "smoke":
        selected = select_smoke(eligible, cap_ids, limit=args.limit)
    else:
        selected = list(eligible)
    if not selected:
        raise SystemExit("no selected rows")
    ctx["selected"] = selected
    ctx["n_selected"] = len(selected)
    LOG.info("selected=%d", len(selected))

    split_meta = assign_splits(selected)
    ctx["split_targets"] = split_meta["targets"]
    ctx["split_actual"] = split_meta["actual"]

    # ── shared embedding + real SQL recall ──
    embeddings = await embed_rows(selected)
    sql_hits = await recall_rows(selected, embeddings)

    # exact-cosine audit (does not feed candidates)
    corpus_emb, corpus_meta = corpus_matrix(corpus)
    ann_audit = exact_coverage_audit(embeddings, corpus_emb, corpus_meta,
                                     sql_hits)
    (data_dir / "ann_coverage_audit.json").write_text(
        json.dumps(ann_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    ctx["ann_audit"] = ann_audit
    LOG.info("ANN exact audit: missing>=0.82=%d, max cap gap=%.8f",
             ann_audit["exact_hits_missing_from_pool_ge_082"],
             ann_audit["max_capability_max_score_gap"])

    # shared evidence + per-capability matrix
    evidence_path = data_dir / "shared_retrieval_evidence.jsonl"
    with open(evidence_path, "w", encoding="utf-8", newline="\n") as f:
        for row, hits in zip(selected, sql_hits):
            for h in hits:
                if float(h["score"]) < EVIDENCE_FLOOR:
                    continue
                rec = {
                    "source_row_id": row["_row_id"],
                    "source_query_id": row["_source_record_key"],
                    "csv_line": row["_csv_line"],
                    "query": row["_query"],
                    "threshold_query_index": row["_row_id"],
                    "corpus_type": h["lane"],
                    "corpus_id": h["corpus_id"],
                    "corpus_query": h["corpus_query"],
                    "corpus_language": h["corpus_language"],
                    "capability_id": h["capability_id"],
                    "score": float(h["score"]),
                    "lane_rank": int(h["lane_rank"]),
                    "global_rank": int(h["global_rank"]),
                    "self_match_excluded": False,
                    "evidence_floor": EVIDENCE_FLOOR,
                }
                f.write(json.dumps(rec, ensure_ascii=False,
                                   separators=(",", ":")) + "\n")

    cap_index = {c: i for i, c in enumerate(cap_ids)}
    cap_matrix = np.full((len(selected), len(cap_ids)), -np.inf, dtype=np.float64)
    for i, hits in enumerate(sql_hits):
        row_best = aggregate_capability_maxima(hits, cap_ids)
        for c, j in cap_index.items():
            cap_matrix[i, j] = row_best[c]
    np.save(data_dir / "capability_max_scores.npy", cap_matrix)
    (data_dir / "capability_ids.json").write_text(
        json.dumps(cap_ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if np.isinf(cap_matrix).any():
        raise SystemExit("a capability has no score; standard lane should cover all")

    # ── three threshold versions ──
    # Each threshold directory is a standalone V2-readable bundle:
    #   v2_{split}.jsonl      exact 14-key training rows (K+1 options)
    #   manifest.json         V2 manifest (+ separate threshold metadata)
    #   construction_audit.jsonl  threshold/candidate/label provenance
    #   {split}.csv           human-readable audit export (never a training input)
    records: dict[float, list[dict]] = {}
    for t in THRESHOLDS:
        recs = [make_record(row, t, cards, cap_ids, cap_matrix[i], src_sha)
                for i, row in enumerate(selected)]
        records[t] = recs
        tag = THRESHOLD_TAGS[t]
        tdir = data_dir / tag
        split_rows: dict[str, list[dict]] = {}
        row_order: dict[str, list[int]] = {}
        for split in SPLITS:
            sub = [r for r in recs if r["split"] == split]
            perm = list(range(len(sub)))
            random.Random(f"{ROW_SEED}:{split}").shuffle(perm)
            sub = [sub[i] for i in perm]
            row_order[split] = perm
            split_rows[split] = sub
            write_jsonl(tdir / f"v2_{split}.jsonl",
                        [to_v2_row(r) for r in sub])
            write_csv(tdir / f"{split}.csv", sub)
        audit = [to_audit_row(r) for split in SPLITS
                 for r in split_rows[split]]
        write_jsonl(tdir / "construction_audit.jsonl", audit)
        direct = [r for split in SPLITS for r in split_rows[split]
                  if r["candidate_count"] == 0]
        write_jsonl(tdir / "direct_reject.jsonl",
                    [to_v2_row(r) for r in direct])
        manifest = build_v2_manifest(ctx, t, split_rows, row_order, tdir)
        (tdir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    LOG.info("wrote three threshold versions")

    # ── statistics ──
    stats = {}
    for t in THRESHOLDS:
        recs = records[t]
        stats[t] = group_metrics(recs)
        stats[t]["per_cap"] = per_cap_metrics(recs, cap_ids)
        stats[t]["per_lang"] = per_key_metrics(recs, lambda r: r["lang"])
        stats[t]["per_source"] = per_key_metrics(recs, lambda r: r["source"])
        stats[t]["per_review_status"] = per_key_metrics(
            recs, lambda r: r["review_status"] or "(blank)")
    ctx["stats"] = stats
    ctx["comparison"] = {
        "082_vs_085": comparison(records, 0.82, 0.85),
        "082_vs_088": comparison(records, 0.82, 0.88),
        "085_vs_088": comparison(records, 0.85, 0.88),
    }

    # ── boundary candidates (real examples; synthetic suite is separate) ──
    boundary = []
    for t in THRESHOLDS:
        for k in range(4):
            ex = next((r for r in records[t] if r["candidate_count"] == k), None)
            if ex:
                boundary.append({
                    "case_type": f"real_k{k}", "synthetic": False,
                    "threshold": t, "id": ex["id"],
                    "source_row_id": ex["source_row_id"],
                    "query": ex["state"], "candidate_count": ex["candidate_count"],
                    "order": ex["order"], "label": ex["label"],
                    "label_reason": ex["label_reason"],
                })
    for t in THRESHOLDS:
        ex = next((r for r in records[t]
                   if r["label_reason"] == "gold_not_retrieved"), None)
        if ex:
            boundary.append({"case_type": "real_gold_not_retrieved",
                             "synthetic": False, "threshold": t,
                             "id": ex["id"], "query": ex["state"],
                             "gold_capability_id": ex["gold_capability_id"],
                             "candidate_capability_ids":
                                 ex["candidate_capability_ids"],
                             "label": ex["label"]})
        ex = next((r for r in records[t] if not r["is_action"]), None)
        if ex:
            boundary.append({"case_type": "real_no_action", "synthetic": False,
                             "threshold": t, "id": ex["id"],
                             "query": ex["state"],
                             "review_status": ex["review_status"],
                             "label": ex["label"]})
    for r in rows:
        if r["_is_standard_query"]:
            boundary.append({"case_type": "real_excluded_standard_query",
                             "synthetic": False, "source_row_id": r["_row_id"],
                             "query": r["_query"],
                             "exclude_reason": r["_exclude_reason"]})
            break
    for r in rows:
        if r["_is_similar_query"]:
            boundary.append({"case_type": "real_excluded_similar_query",
                             "synthetic": False, "source_row_id": r["_row_id"],
                             "query": r["_query"],
                             "exclude_reason": r["_exclude_reason"]})
            break
    write_jsonl(data_dir / "boundary_cases.jsonl", boundary)

    # ── sample manifest (all source rows) ──
    selected_ids = {r["_row_id"] for r in selected}
    manifest_fields = [
        "source_row_id", "csv_line", "source_file", "source_row", "query",
        "is_action", "gold_capability_id", "source", "review_status",
        "language_raw", "language_resolved", "query_family",
        "query_family_id", "normalized_query_sha256", "duplicate_group_size",
        "is_standard_query", "is_standard_query_similar", "derived_relation",
        "eligible", "exclude_reason", "smoke_selected", "selection_reason",
        "split",
    ]
    with open(data_dir / "sample_manifest.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "source_row_id": r["_row_id"],
                "csv_line": r["_csv_line"],
                "source_file": r["_source_file"],
                "source_row": r["_source_row"],
                "query": r["_query"],
                "is_action": r["_is_action"],
                "gold_capability_id": (
                    r["_capability_id"] if r["_is_action"] else ""),
                "source": r["_source"],
                "review_status": r["_review_status"],
                "language_raw": r["_lang_raw"],
                "language_resolved": r["_lang"],
                "query_family": r["_family_sig"],
                "query_family_id": r["_family_key"],
                "normalized_query_sha256": stable_key("norm", r["_norm"]),
                "duplicate_group_size": r["_duplicate_group_size"],
                "is_standard_query": r["_is_standard_query"],
                "is_standard_query_similar": r["_is_similar_query"],
                "derived_relation": "none_determinable",
                "eligible": r["_eligible"],
                "exclude_reason": r["_exclude_reason"],
                "smoke_selected": r["_row_id"] in selected_ids,
                "selection_reason": r.get("_selection_reason", ""),
                "split": r.get("_split", ""),
            })

    # ── in-build checks + reports ──
    ctx["checks"] = validate_in_build(records, cap_ids, cards)
    write_source_audit(ctx)
    write_logic_report(ctx)
    write_distribution_report(ctx)
    write_comparison_report(ctx)
    write_preliminary_reports(ctx)
    config = write_config(ctx)
    write_source_manifest(ctx)
    ctx["finished"] = utc_now()
    write_generation_manifest(ctx, config)

    # threshold comparison JSON (supplement to the markdown report)
    (RUN_DIR / "reports" / "threshold_comparison_082_085_088.json").write_text(
        json.dumps(ctx["comparison"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    failed = [k for k, v in ctx["checks"].items() if not v["ok"]]
    LOG.info("build complete; checks failed=%s", failed or "none")
    print(json.dumps({
        "run_id": ctx["run_id"],
        "mode": ctx["mode"],
        "selected_rows": ctx["n_selected"],
        "eligible_rows": len(eligible),
        "checks_failed": failed,
        "thresholds": {str(t): {
            "k_distribution": stats[t]["k_distribution"],
            "split_counts": stats[t]["split_counts"],
            "gold_top3": stats[t]["gold_top3"],
            "action_rows": stats[t]["action_rows"],
        } for t in THRESHOLDS},
    }, ensure_ascii=False, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
