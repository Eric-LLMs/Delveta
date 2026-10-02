#!/usr/bin/env python3
"""V2 data adjustment (2026-10-02): move 400 Test rows into Train.

Data-only, no V3. Semantics:
  * Train 1712 -> 2112, Val 300 (unchanged), Test 1800 -> 1400.
  * The only field that changes on any row is `split`.
  * The move unit is the SOURCE QUERY: every frozen source query owns exactly one
    p1 and one p2, both in the same split; splits share zero source queries. So we
    move whole source queries (200 x 2 rows = 400) to keep p1/p2 co-located and to
    keep Train and Test source-query-disjoint (no leakage).
  * Deterministic, stratified by (lang, capability-of-p1) with the largest-remainder
    method; `part` is auto-balanced (each source contributes one p1 and one p2).

Writes the frozen 1800-row Test to a durable backup BEFORE touching anything, then
rewrites v2_train.jsonl / v2_test.jsonl / manifest.json. v2_val.jsonl is untouched.

Default is a DRY RUN (prints the plan). Pass --write to mutate.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"   # V2 root -> V2/data (portable)
BACKUP = DATA.parent / "frozen_backup"
AUDIT = DATA / "v2_construction_audit.jsonl"

MOVE_SEED = "test-to-train-v1-20261002"
ADJUST_DATE = "2026-10-02"
TARGET_SOURCES = 200          # 200 sources x 2 rows == 400 rows
CALIB_SEED = 20260922
CALIB_MAX = 400


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def dump_jsonl(rows: list[dict], p: Path) -> None:
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def cap_of(r: dict) -> str:
    return r["gold_capability_id"] or "REJECT"


def largest_remainder(sizes: dict, total: int) -> dict:
    """Allocate `total` over strata proportionally; largest remainders get the rest.
    Ties broken by the (sorted) stratum key so the result is deterministic."""
    n = sum(sizes.values())
    quota = {k: v * total / n for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    short = total - sum(alloc.values())
    order = sorted(sizes, key=lambda k: (-(quota[k] - alloc[k]), k))
    for k in order[:short]:
        alloc[k] += 1
    return alloc


def calibrate(train_rows: list[dict]) -> dict:
    n = len(train_rows)
    order = list(range(n))
    random.Random(CALIB_SEED).shuffle(order)
    n_calib = min(CALIB_MAX, n // 10)
    calib_idx = sorted(order[:n_calib])
    train_idx = sorted(order[n_calib:])
    return {
        "seed": CALIB_SEED, "source_train_rows": n,
        "calibration_rows": n_calib, "effective_train_rows": len(train_idx),
        "rule": "min(400, n_train // 10) over the FINAL (post-shuffle) train order",
        "indices": calib_idx,
        "ids": [train_rows[i]["id"] for i in calib_idx],
    }


def stats(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "gold_slot": {str(k): v for k, v in sorted(Counter(r["gold_index"] for r in rows).items())},
        "part_rows": {"part1": sum(r["part"] == "p1" for r in rows),
                      "part2": sum(r["part"] == "p2" for r in rows)},
        "language": dict(Counter(r["lang"] for r in rows)),
        "kind": dict(Counter(r["target_kind"] for r in rows)),
        "per_cap": dict(Counter(cap_of(r) for r in rows)),
        "src_sha": sorted({r["src_sha"] for r in rows}),
    }


def main() -> None:
    write = "--write" in sys.argv

    pre_sha = {p.name: sha256_file(p) for p in sorted(DATA.glob("*.json*"))}
    print("== pre-adjustment SHA256 ==")
    for n, h in pre_sha.items():
        print(f"   {h}  {n}")

    train = load(DATA / "v2_train.jsonl")
    val = load(DATA / "v2_val.jsonl")
    test = load(DATA / "v2_test.jsonl")
    audit = {json.loads(l)["id"]: json.loads(l) for l in AUDIT.open(encoding="utf-8") if l.strip()}

    # ── group Test by source query ───────────────────────────────────────────
    src: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in test:
        src[r["source_query_id"]][r["part"]] = r
    assert all(len(d) == 2 and set(d) == {"p1", "p2"} for d in src.values()), \
        "a test source query must own exactly p1+p2"
    assert len(src) == 900, len(src)

    # ── stratify sources by (lang, capability-of-p1) ─────────────────────────
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid, d in src.items():
        strata[(d["p1"]["lang"], cap_of(d["p1"]))].append(sid)
    sizes = {k: len(v) for k, v in strata.items()}
    alloc = largest_remainder(sizes, TARGET_SOURCES)
    assert sum(alloc.values()) == TARGET_SOURCES

    moved_sources: list[str] = []
    for key in sorted(strata):
        sid_pool = sorted(strata[key])
        k = alloc[key]
        chosen = random.Random(f"{MOVE_SEED}:{key[0]}|{key[1]}").sample(sid_pool, k)
        moved_sources.extend(chosen)
    moved_sources = sorted(moved_sources)
    assert len(moved_sources) == TARGET_SOURCES

    moved_set = set(moved_sources)
    moved_rows = [r for r in test if r["source_query_id"] in moved_set]          # frozen test order
    kept_rows = [r for r in test if r["source_query_id"] not in moved_set]
    assert len(moved_rows) == 400 and len(kept_rows) == 1400

    # ── retag ONLY the split field on the moved rows ─────────────────────────
    moved_retagged = [dict(r, split="train") for r in moved_rows]
    new_train = train + moved_retagged                                          # frozen train order ++ moved
    new_test = kept_rows
    assert len(new_train) == 2112 and len(new_test) == 1400

    # ── calibration on the NEW train order ───────────────────────────────────
    calib = calibrate(new_train)

    # ── plan report ──────────────────────────────────────────────────────────
    print("\n== stratification (lang | cap_p1) ==")
    for key in sorted(strata):
        print(f"   {key[0]:2s} {key[1]:22s} pool={sizes[key]:3d} moved={alloc[key]:2d}")
    print(f"\n sources moved: {len(moved_sources)}  rows moved: {len(moved_rows)}")
    print(f" new splits: train={len(new_train)} val={len(val)} test={len(new_test)}")

    def cov(rows, label):
        s = stats(rows)
        print(f" [{label}] n={s['n']} part={s['part_rows']} lang={s['language']} kind={s['kind']}")
        print(f"        gold_slot={s['gold_slot']} src_sha={len(s['src_sha'])} value(s)")
    print()
    cov(train, "old train")
    cov(test, "old test")
    cov(new_train, "NEW train")
    cov(new_test, "NEW test")

    # moved-row coverage (the 400)
    ms = stats(moved_rows)
    print(f"\n moved 400 coverage: part={ms['part_rows']} lang={ms['language']} kind={ms['kind']}")
    print(f"        per_cap={dict(sorted(ms['per_cap'].items()))}")
    msyn = sum(1 for r in moved_rows if r["target_kind"] == "reject" and audit[r["id"]].get("synthetic"))
    mnat = sum(1 for r in moved_rows if r["target_kind"] == "reject" and audit[r["id"]].get("gold_absent_from_candidates"))
    print(f"        moved rejects: {ms['kind'].get('reject', 0)} (synthetic={msyn}, gold_absent={mnat})")

    # calibration summary
    print(f"\n calibration: n_train={calib['source_train_rows']} calib={calib['calibration_rows']} "
          f"effective={calib['effective_train_rows']} seed={calib['seed']}")
    upt = -(-calib["effective_train_rows"] // 32)
    print(f" scheduler: updates/epoch=ceil({calib['effective_train_rows']}/32)={upt} "
          f"total_updates={upt*4}")

    if not write:
        print("\n[dry-run] no files written; pass --write to apply")
        return

    # ── backup BEFORE mutating ───────────────────────────────────────────────
    BACKUP.mkdir(parents=True, exist_ok=True)
    bpath = BACKUP / "v2_test_1800_pre_adjustment.jsonl"
    if bpath.exists():
        raise SystemExit(f"backup already exists: {bpath} (refusing to overwrite)")
    shutil.copyfile(DATA / "v2_test.jsonl", bpath)
    (BACKUP / "FROZEN_SHA256.json").write_text(
        json.dumps({"note": "SHA256 of the frozen V2 bundle BEFORE the 2026-10-02 Test->Train move",
                    "at": ADJUST_DATE, "files": pre_sha}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    assert sha256_file(bpath) == pre_sha["v2_test.jsonl"], "backup sha mismatch!"

    # ── write splits (val untouched) ─────────────────────────────────────────
    dump_jsonl(new_train, DATA / "v2_train.jsonl")
    dump_jsonl(new_test, DATA / "v2_test.jsonl")

    # ── rewrite manifest (current-state blocks recomputed) ───────────────────
    man = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    ns = {s: stats(rows) for s, rows in (("train", new_train), ("val", val), ("test", new_test))}
    man["rows"] = {s: ns[s]["n"] for s in ("train", "val", "test")}
    man["gold_slot_distribution"] = {s: ns[s]["gold_slot"] for s in ns}
    man["part_rows"] = {s: ns[s]["part_rows"] for s in ns}
    man["language"] = {s: ns[s]["language"] for s in ns}
    man["split_provenance"] = {s: ns[s]["src_sha"] for s in ns}
    man["calibration"] = calib
    man["row_order"]["note"] = (
        (man["row_order"].get("note", "") +
         " | PRE-adjustment construction only: the train/test permutations describe the "
         "frozen build before the 2026-10-02 Test->Train move; see `adjustment` for the "
         "current file order.").strip(" |"))
    man["adjustment"] = {
        "date": ADJUST_DATE,
        "rule": ("move 400 Test rows into Train by moving 200 whole source queries "
                 "(each source query owns one p1 + one p2, both in the same split); the "
                 "only field changed on any row is `split`"),
        "seed": MOVE_SEED,
        "unit": "source_query (p1+p2 together)",
        "moved_sources": len(moved_sources),
        "moved_rows": len(moved_rows),
        "strata": {f"{k[0]}|{k[1]}": {"pool": sizes[k], "moved": alloc[k]} for k in sorted(strata)},
        "order_rule": ("train = frozen train order ++ the moved rows in frozen test order; "
                       "test = the non-moved rows in frozen test order"),
        "pre_adjustment_sha256": pre_sha,
        "pre_adjustment_test_backup": str(bpath.relative_to(DATA.parent.parent)),
        "moved_source_query_ids": moved_sources,
        "moved_row_ids": sorted(r["id"] for r in moved_rows),
    }
    (DATA / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    post_sha = {p.name: sha256_file(p) for p in sorted(DATA.glob("*.json*"))}
    print("\n== post-adjustment SHA256 ==")
    for n, h in post_sha.items():
        tag = "" if pre_sha.get(n) == h else "  <- changed"
        print(f"   {h}  {n}{tag}")
    print(f"\nbackup -> {bpath}")


if __name__ == "__main__":
    main()
