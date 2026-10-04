#!/usr/bin/env python3
"""Step 1 audit for the BGE-M3 Query -> Capability fine-tuning datasets.

Read-only. Reports, for scripts/bge_m3_finetune/data/:
  * schema / field contract
  * per-capability x language distribution (train + test)
  * source composition (which rows come from the real retrieval corpus908 vs synthetic)
  * Train vs Test exact + normalized leakage
  * THE core question: is ``pos`` a capability retrieval text, or just another query?
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
TRAIN = DATA / "bge_m3_train.jsonl"
TEST = DATA / "bge_m3_test.jsonl"
MANIFEST_TRAIN = DATA / "bge_m3_train_manifest.jsonl"
MANIFEST_TEST = DATA / "bge_m3_test_manifest.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path.name}:{i} bad JSON: {exc}") from exc
    return rows


def norm(text: str) -> str:
    """Normalized form for leakage: NFKC, lower, strip punctuation + whitespace."""
    s = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[^\w\u4e00-\u9fff ]", "", s, flags=re.UNICODE)
    return s.strip()


def keyset(rows: list[dict]) -> set[str]:
    return {norm(r["query"]) for r in rows}


def main() -> int:
    train = read_jsonl(TRAIN)
    test = read_jsonl(TEST)
    mtrain = read_jsonl(MANIFEST_TRAIN)
    mtest = read_jsonl(MANIFEST_TEST)

    problems: list[str] = []
    print(f"train rows: {len(train):,}   manifest: {len(mtrain):,}")
    print(f"test  rows: {len(test):,}   manifest: {len(mtest):,}")
    print()

    # ---- 1. schema contract ------------------------------------------------
    for name, rows in (("train", train), ("test", test)):
        bad = [i for i, r in enumerate(rows, 1)
               if set(r.keys()) != {"query", "pos", "neg"}
               or not isinstance(r["query"], str)
               or not isinstance(r["pos"], list) or not r["pos"]
               or not isinstance(r["neg"], list) or not r["neg"]]
        print(f"[{name}] schema {'OK' if not bad else f'FAIL {len(bad)} rows'}"
              f" (expected exactly query/pos/neg, non-empty lists)")
        if bad:
            problems.append(f"{name}: {len(bad)} rows violate query/pos/neg schema")
        kw = Counter(tuple(sorted(r.keys())) for r in rows)
        print(f"[{name}] key-shapes: {dict(kw)}")

    # ---- 2. THE core semantic question: pos == another query? --------------
    # A capability retrieval text is a definitional sentence ("Add a word, term,
    # or expression to ..."). A wrong pos is a plain user request.
    desc_markers = [
        "Use it when the user", "does:", "intent_kind", "tool:",
        "it does not apply", "中文：",
    ]
    for name, rows in (("train", train), ("test", test)):
        pos_looks_query = 0
        pos_looks_desc = 0
        for r in rows:
            for p in r["pos"]:
                s = str(p)
                if any(m in s for m in desc_markers) or len(s) > 120:
                    pos_looks_desc += 1
                else:
                    pos_looks_query += 1
        tot = pos_looks_query + pos_looks_desc
        print(f"[{name}] pos entries: {tot:,}  "
              f"definitional-looking={pos_looks_desc:,}  query-looking={pos_looks_query:,}")
        # CORE CONSTRAINT: pos must be a capability retrieval text, never another
        # user query (task: "不能把 Query -> Query similarity 错当成 Query ->
        # Capability retrieval"). A row whose pos is still a request sentence fails.
        if pos_looks_desc == 0:
            problems.append(
                f"{name}: ALL {tot:,} pos entries look like user queries, not "
                f"capability retrieval texts (Query->Query, forbidden by spec)")

    # ---- 3. distribution ---------------------------------------------------
    def dist(rows, capkey):
        d = defaultdict(Counter)
        for r in rows:
            d[r.get(capkey, "?")][r.get("lang", "?")] += 1
        return d

    dtr = dist(mtrain, "capability_id")
    dte = dist(mtest, "capability_id")
    caps = sorted(dtr.keys(), key=lambda c: (-sum(dtr[c].values()), c))
    print()
    print(f"capabilities train={len(caps)} test={len(set(dte))}")
    print(f"{'capability':<24}{'tr_zh':>6}{'tr_en':>6}{'te_zh':>6}{'te_en':>6}")
    for c in caps:
        print(f"{c:<24}{dtr[c]['zh']:>6}{dtr[c]['en']:>6}{dte[c]['zh']:>6}{dte[c]['en']:>6}")

    # ---- 4. source composition --------------------------------------------
    src = Counter(r.get("source", "?") for r in mtrain)
    print()
    print(f"train sources: {dict(src)}")
    srct = Counter(r.get("source", "?") for r in mtest)
    print(f"test  sources: {dict(srct)}")

    # ---- 5. leakage --------------------------------------------------------
    tr_exact = Counter(r["query"] for r in train)
    te_exact = {r["query"] for r in test}
    exact_overlap = [q for q in te_exact if tr_exact[q]]
    tr_norm = keyset(train)
    te_norm = keyset(test)
    norm_overlap = tr_norm & te_norm
    # dup-pos across cap (a failure mode: same pos text reused)
    pos_counter = Counter(p for r in train for p in r["pos"])
    print()
    print(f"leakage exact overlap (train∩test): {len(exact_overlap)}")
    for q in exact_overlap[:10]:
        print(f"    {q!r}")
    print(f"leakage normalized overlap: {len(norm_overlap)}")
    for q in list(norm_overlap)[:10]:
        print(f"    {q!r}")
    print(f"train distinct pos texts: {len(pos_counter):,}")

    if exact_overlap:
        problems.append(f"train/test exact query overlap = {len(exact_overlap)} (spec requires 0)")
    if norm_overlap:
        problems.append(f"train/test normalized overlap = {len(norm_overlap)} (spec requires 0)")

    # ---- 6. verdict on the core constraint --------------------------------
    print()
    print("=" * 72)
    if problems:
        print("STEP 1 AUDIT: FAIL")
        for p in problems:
            print(f"  - {p}")
    else:
        print("STEP 1 AUDIT: PASS")
    print("=" * 72)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
