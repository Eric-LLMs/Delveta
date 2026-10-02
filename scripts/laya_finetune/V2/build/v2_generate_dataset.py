#!/usr/bin/env python3
"""V2 (REJECT) dataset — full corrected scope, one build.

V1 is read-only. Nothing under scripts/laya_finetune/ or production is touched.

Per split (train/val/test) = Part1 + Part2, merged, then ROW-shuffled.

Part1  (copy of the V1 bundle split; 4 options each):
  * deterministic draw of 10% of the split's rows -> synthetic REJECT rows
  * drawn rows : keep query; keep the two original distractors; REPLACE gold with
                 a real wrong capability drawn from the 17 caps minus {gold,d1,d2};
                 add REJECT; target = REJECT
  * other rows : keep query, gold and the two distractors; add REJECT; target = gold
  * per-row deterministic 4-way shuffle; gold_index recomputed

Part2  (real Recall candidates, from v2_final/; 4 options each):
  * Recall candidates already carry the locked pipeline
      (min_score=0.0, no threshold -> remove self-hit -> cap dedup max -> sort DESC -> Top-3)
  * take Top-3 + REJECT
  * gold in Top-3 -> target = gold slot ; gold absent (natural miss) -> target = REJECT
  * per-row deterministic 4-way shuffle; gold_index recomputed

Merge step (rule 5):
  * Part1 rows and Part2 rows are MERGED first, then the whole split is
    deterministically row-shuffled (NOT part1-then-part2 concatenation).
  * ids are part-tagged (src_id + "-p1"/"-p2") so the two rows of one source
    query stay distinguishable.
  * the same source query's p1/p2 must NOT be adjacent -> deterministic
    anti-adjacency repair after the base shuffle.
  * row_order_permutation is saved to the report and the manifest.

Calibration manifest mirrors V1's structure on the FINAL (post-shuffle) train
rows: seed 20260922, n_calib = min(400, n_train // 10) = 171.

Output (v2_dataset/): v2_{train,val,test}.jsonl, manifest.json,
                     v2_build_report.json, SHA256SUMS

Deterministic constants are fixed below and recorded in the report.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DEST = OUT / "v2_dataset"
BUNDLE = REPO / "V1" / "data"   # upstream V1 query bundles; NOT V2 training data
V2FINAL = OUT / "v2_final"

SPLITS = ("train", "val", "test")
SPLIT_FILE = {"train": "v2_train2.jsonl", "val": "v2_val2.jsonl", "test": "v2_test2.jsonl"}

# ── frozen deterministic constants (recorded in the report) ──────────────────
DRAW_SEED = 20261002          # which 10% of each split's rows become synthetic-REJECT
DRAW_FRACTION = 0.10
SHUFFLE_SEED = "reject-shuffle-v1"   # per-row 4-way shuffle seed prefix
WRONG_SEED = "reject-wrong-v1"       # per-row wrong-tool choice seed prefix
ROW_SEED = "reject-rows-v1"          # per-split row-order shuffle seed prefix
CALIB_SEED = 20260922                # same seed as the V1 calibration holdout
CALIB_MAX = 400                      # same cap as V1: min(400, n_train // 10)

# ── REJECT option (ASSUMPTION - single point of change) ──────────────────────
REJECT_LABEL = "REJECT"
REJECT_ID = "REJECT"
REJECT_CARD = (
    "### REJECT\n"
    "tool: none\n"
    "does: No listed capability correctly handles the user's request. Choose this "
    "option only when none of the other options is the right capability.\n\n"
    "中文：以上列出的能力都不适用于该请求。只有当其它选项都不是正确的能力时才选它。"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bundle(split: str) -> list[dict]:
    rows = []
    with gzip.open(BUNDLE / f"layachoice_v1_{split}_b_noprov.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def build_card_map() -> dict[str, str]:
    """Canonical B_noprov card per capability id (must be unique)."""
    cards = defaultdict(set)
    for split in SPLITS:
        for r in load_bundle(split):
            for cap, card in zip(r["order"], r["options"]):
                cards[cap].add(card)
    conflicts = {c: len(s) for c, s in cards.items() if len(s) > 1}
    if conflicts:
        raise RuntimeError(f"non-canonical cards: {conflicts}")
    return {c: next(iter(s)) for c, s in cards.items()}


def shuffled_order(pre_items: list[dict], row_key: str) -> tuple[list[dict], list[int]]:
    """Deterministic 4-way shuffle. Returns (output_items, permutation) where
    output_items[p] came from pre_items[permutation[p]]."""
    perm = list(range(len(pre_items)))
    random.Random(f"{SHUFFLE_SEED}:{row_key}").shuffle(perm)
    return [pre_items[i] for i in perm], perm


def row_shuffle_order(sample_ids: list[str], split: str) -> tuple[list[int], list[str]]:
    """Deterministic row-order shuffle over the merged split, then a deterministic
    anti-adjacency repair so the two rows of one source query (p1/p2) are never
    directly adjacent. Returns (permutation, sid_in_output_order) where
    output slot p holds the pre-shuffle row permutation[p]."""
    n = len(sample_ids)
    perm = list(range(n))
    random.Random(f"{ROW_SEED}:{split}").shuffle(perm)
    sid = [sample_ids[p] for p in perm]

    for _pass in range(200):
        changed = False
        i = 0
        while i < n - 1:
            if sid[i] == sid[i + 1]:
                a = sid[i + 1]
                for j in range(i + 2, n):
                    b = sid[j]
                    if b == a:
                        continue
                    if b == sid[i]:                       # new left neighbour of i+1
                        continue
                    if i + 2 < n and b == sid[i + 2]:     # new right neighbour of i+1
                        continue
                    if sid[j - 1] == a:                   # new left neighbour of j
                        continue
                    if j + 1 < n and sid[j + 1] == a:     # new right neighbour of j
                        continue
                    sid[i + 1], sid[j] = b, a
                    perm[i + 1], perm[j] = perm[j], perm[i + 1]
                    changed = True
                    break
            i += 1
        if not changed:
            break

    bad = [i for i in range(n - 1) if sid[i] == sid[i + 1]]
    if bad:
        raise RuntimeError(f"{split}: anti-adjacency repair failed at {bad[:5]}")
    return perm, sid


def main():
    canon = build_card_map()
    caps = sorted(canon)
    if len(caps) != 17:
        raise RuntimeError(f"capability set is {len(caps)}, expected 17")

    DEST.mkdir(parents=True, exist_ok=True)

    # ── per-split provenance + the merged, pre-shuffle row build ─────────────
    merged: dict[str, list[dict]] = {}
    src_index: dict[str, list[str]] = {}        # output order -> source query id
    part_counts = {}
    reject_hist = defaultdict(Counter)          # split -> target_kind Counter
    gold_idx_hist = defaultdict(Counter)
    reject_idx_hist = defaultdict(Counter)
    role_pos = defaultdict(lambda: defaultdict(Counter))   # split -> role -> position
    wrong_hist = Counter()
    drawn_report = {}
    part1_synth = part2_natural = 0

    def emit(bucket: list[dict], row, split, part, pre_items, target_pre_idx, meta):
        out_id = f"{row['_src_id']}-{part}"
        order_items, perm = shuffled_order(pre_items, out_id)
        gold_index = perm.index(target_pre_idx)
        is_reject = order_items[gold_index]["cap"] == REJECT_ID
        out_row = {
            "id": out_id,
            "source_query_id": row["_src_id"],
            "split": split,
            "part": part,
            "lang": row["lang"],
            "state": row["state"],
            "order": [it["cap"] for it in order_items],
            "options": [it["card"] for it in order_items],
            "gold_index": gold_index,
            "target_kind": "reject" if is_reject else "capability",
            "gold_capability_id": None if is_reject else order_items[gold_index]["cap"],
            "reject_index": next(i for i, it in enumerate(order_items)
                                 if it["cap"] == REJECT_ID),
            "shuffle": {"seed_prefix": SHUFFLE_SEED, "permutation": perm},
            "src_sha": row["_src_sha"],
            "_audit": dict(meta),          # construction metadata -> audit file only
        }
        bucket.append(out_row)
        reject_hist[split][out_row["target_kind"]] += 1
        gold_idx_hist[split][gold_index] += 1
        reject_idx_hist[split][out_row["reject_index"]] += 1
        for i, it in enumerate(order_items):
            role = ("reject" if it["cap"] == REJECT_ID
                    else "gold" if i == gold_index else "distractor")
            role_pos[split][role][i] += 1

    for split in SPLITS:
        bundle = load_bundle(split)
        src_sha = {r["id"]: r.get("src_sha") for r in bundle}
        n = len(bundle)

        # ── Part1: copy of the V1 bundle split, 4 options ────────────────────
        p1: list[dict] = []
        n_synth = round(DRAW_FRACTION * n)
        drawn_idx = sorted(random.Random(f"{DRAW_SEED}:{split}").sample(range(n), n_synth))
        drawn = set(drawn_idx)
        reject_item = {"cap": REJECT_ID, "card": REJECT_CARD}
        for i, r in enumerate(bundle):
            gold_cap = r["order"][r["gold"]]
            d1_cap = r["order"][r["d1"]]
            d2_cap = r["order"][r["d2"]]
            r["_src_id"] = r["id"]
            r["_src_sha"] = src_sha[r["id"]]
            if i in drawn:
                pool = [c for c in caps if c not in (gold_cap, d1_cap, d2_cap)]
                wrong = random.Random(f"{WRONG_SEED}:{r['id']}").choice(pool)
                pre = [{"cap": wrong, "card": canon[wrong]},
                       {"cap": d1_cap, "card": canon[d1_cap]},
                       {"cap": d2_cap, "card": canon[d2_cap]},
                       reject_item]
                emit(p1, r, split, "p1", pre, 3,
                     {"synthetic": True, "wrong_tool_id": wrong,
                      "original_gold_tool": gold_cap})
                wrong_hist[wrong] += 1
                part1_synth += 1
            else:
                pre = [{"cap": gold_cap, "card": canon[gold_cap]},
                       {"cap": d1_cap, "card": canon[d1_cap]},
                       {"cap": d2_cap, "card": canon[d2_cap]},
                       reject_item]
                emit(p1, r, split, "p1", pre, 0,
                     {"synthetic": False, "wrong_tool_id": None,
                      "original_gold_tool": gold_cap})
        drawn_report[split] = {
            "seed": f"{DRAW_SEED}:{split}", "fraction": DRAW_FRACTION,
            "rows": n, "n_synthetic": n_synth,
            "drawn_row_ids": [bundle[i]["id"] for i in drawn_idx],
        }

        # ── Part2: real Recall Top-3 + REJECT ────────────────────────────────
        p2: list[dict] = []
        for row in load_jsonl(V2FINAL / SPLIT_FILE[split]):
            rid = row["sample_id"]
            r = {"_src_id": rid, "_src_sha": src_sha[rid],
                 "lang": row["language"], "state": row["query"]}
            order3 = row["top3_order"]
            cards3 = row["top3_cards"]
            pre = [{"cap": c, "card": card} for c, card in zip(order3, cards3)]
            pre.append(reject_item)
            if row["gold_in_candidates"]:
                target = order3.index(row["gold_capability_id"])
            else:
                target = 3
                part2_natural += 1
            emit(p2, r, split, "p2", pre, target,
                 {"synthetic": False, "wrong_tool_id": None,
                  "gold_absent_from_candidates": not row["gold_in_candidates"],
                  "recall_top3_order": order3,
                  "recall_top3_scores": row["top3_scores"]})

        part_counts[split] = {"part1": len(p1), "part2": len(p2)}
        rows = p1 + p2                                    # MERGE first
        src_index[split] = [x["source_query_id"] for x in rows]
        merged[split] = rows

    # ── row-level shuffle AFTER merge, then write ────────────────────────────
    row_order_permutation = {}
    paths = {}
    final_rows_by_split = {}
    f_audit = (DEST / "v2_construction_audit.jsonl").open(
        "w", encoding="utf-8", newline="\n")
    for split in SPLITS:
        rows = merged[split]
        perm, sid = row_shuffle_order(src_index[split], split)
        row_order_permutation[split] = perm
        out_rows = [rows[p] for p in perm]
        final_rows_by_split[split] = out_rows
        p = DEST / f"v2_{split}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for r in out_rows:
                meta = r.pop("_audit")          # construction metadata -> audit file
                f_audit.write(json.dumps({"id": r["id"], "split": split,
                                          "part": r["part"], **meta},
                                         ensure_ascii=False) + "\n")
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[split] = sha256_file(p)
    f_audit.close()
    audit_path = sha256_file(DEST / "v2_construction_audit.jsonl")

    # ── calibration manifest on the FINAL train rows (mirror V1 structure) ────
    train_rows = final_rows_by_split["train"]
    n_train = len(train_rows)
    order = list(range(n_train))
    random.Random(CALIB_SEED).shuffle(order)
    n_calib = min(CALIB_MAX, n_train // 10)
    calib_idx = sorted(order[:n_calib])
    train_idx = sorted(order[n_calib:])
    calib_ids = [train_rows[i]["id"] for i in calib_idx]

    # ── manifest ─────────────────────────────────────────────────────────────
    gold_slot_dist = {s: {str(k): v for k, v in sorted(gold_idx_hist[s].items())}
                      for s in SPLITS}
    lang_hist = {s: dict(Counter(r["lang"] for r in final_rows_by_split[s]))
                 for s in SPLITS}
    src_files = {}
    for split in SPLITS:
        src_files[f"layachoice_v1_{split}_b_noprov.jsonl.gz"] = sha256_file(
            BUNDLE / f"layachoice_v1_{split}_b_noprov.jsonl.gz")
    for name in SPLIT_FILE.values():
        src_files[f"v2_final/{name}"] = sha256_file(V2FINAL / name)

    manifest = {
        "bench": "LayaChoice-v2",
        "formal_card_view": "B_noprov",
        "instructions": "Which capability should handle the user's request?",
        "option_slots": 4,
        "reject": {"label": REJECT_LABEL, "id": REJECT_ID, "card": REJECT_CARD,
                   "meaning": "no listed capability correctly handles the request; "
                              "present on every row as the 4th option"},
        "option_order_rule": "per-row deterministic 4-way shuffle; seed prefix "
                             + SHUFFLE_SEED + " (recorded in each row's shuffle field)",
        "gold_slot_distribution": gold_slot_dist,
        "source_files": src_files,
        "rows": {s: len(final_rows_by_split[s]) for s in SPLITS},
        "part_rows": part_counts,
        "language": lang_hist,
        "calibration": {
            "seed": CALIB_SEED, "source_train_rows": n_train,
            "calibration_rows": n_calib, "effective_train_rows": len(train_idx),
            "rule": "min(400, n_train // 10) over the FINAL (post-shuffle) train order",
            "indices": calib_idx, "ids": calib_ids,
        },
        "row_order": {"seed_prefix": ROW_SEED, "permutation": row_order_permutation,
                      "note": "output slot p holds the merged pre-shuffle row permutation[p]; "
                              "anti-adjacency repair applied so a source query's p1/p2 are never adjacent"},
        "base_checkpoint": {
            "repo_id": "convaiinnovations/laya",
            "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
            "subfolder": "multilingual",
        },
        "smoke": False,
    }
    (DEST / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ── consistency audit ────────────────────────────────────────────────────
    audit = {"per_split": {}, "invariants": {"ok": True, "failures": []},
             "part_versus_row_order": {}}
    CAPS = set(caps)

    def fail(msg):
        audit["invariants"]["ok"] = False
        audit["invariants"]["failures"].append(msg)

    for split in SPLITS:
        rows = final_rows_by_split[split]
        sid = [r["source_query_id"] for r in rows]
        # 1) row-level: every row well-formed and consistent
        for r in rows:
            if len(r["order"]) != 4 or len(r["options"]) != 4:
                fail(f"{r['id']}: not 4 options")
            if r["order"][r["gold_index"]] != (REJECT_ID if r["target_kind"] == "reject"
                                               else r["gold_capability_id"]):
                fail(f"{r['id']}: gold_index/target mismatch")
            if r["order"][r["reject_index"]] != REJECT_ID:
                fail(f"{r['id']}: reject_index mismatch")
            if len({c for c in r["order"] if c != REJECT_ID}) != 3:
                fail(f"{r['id']}: candidates not 3 distinct real caps")
            if any(c not in CAPS for c in r["order"] if c != REJECT_ID):
                fail(f"{r['id']}: unknown capability in options")
            perm = r["shuffle"]["permutation"]
            if sorted(perm) != [0, 1, 2, 3]:
                fail(f"{r['id']}: bad shuffle permutation")
        # 2) unique ids
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            fail(f"{split}: duplicate row ids")
        # 3) every source query appears exactly twice (p1 + p2)
        c = Counter(sid)
        if set(c.values()) != {2}:
            fail(f"{split}: source query not exactly p1+p2: { {k: v for k, v in c.items() if v != 2} }")
        # 4) p1/p2 not adjacent
        adj = sum(1 for i in range(len(sid) - 1) if sid[i] == sid[i + 1])
        if adj:
            fail(f"{split}: {adj} adjacent p1/p2 pairs")
        # 5) part counts match
        pc = Counter(r["part"] for r in rows)
        if dict(pc) != {"p1": part_counts[split]["part1"], "p2": part_counts[split]["part2"]}:
            fail(f"{split}: part counts mismatch")
        # part distribution across position deciles (no block formation)
        seg = 10
        seg_hist = defaultdict(Counter)
        for i, r in enumerate(rows):
            seg_hist[min(seg - 1, i * seg // len(rows))][r["part"]] += 1
        runs = {"p1": 0, "p2": 0}
        run = 0
        for i, r in enumerate(rows):
            run = run + 1 if i and r["part"] == rows[i - 1]["part"] else 1
            runs[r["part"]] = max(runs[r["part"]], run)
        audit["part_versus_row_order"][split] = {
            "counts": {"p1": pc["p1"], "p2": pc["p2"]},
            "adjacent_p1p2_pairs": adj,
            "max_same_part_run": runs,
            "position_decile_part_distribution":
                {str(d): dict(seg_hist[d]) for d in range(seg)},
        }
        audit["per_split"][split] = {
            "rows": len(rows),
            "part1": part_counts[split]["part1"], "part2": part_counts[split]["part2"],
            "target_kind": {k: reject_hist[split][k] for k in ("capability", "reject")},
            "gold_index_distribution": {str(k): v for k, v in sorted(gold_idx_hist[split].items())},
            "reject_index_distribution": {str(k): v for k, v in sorted(reject_idx_hist[split].items())},
            "role_position_distribution": {role: {str(k): v for k, v in sorted(cc.items())}
                                           for role, cc in role_pos[split].items()},
            "language": lang_hist[split],
            "sha256": paths[split],
        }

    report = {
        "scope": "Part1(Train/Val/Test) + Part2(Train/Val/Test), merged, row-shuffled",
        "reject_label": REJECT_LABEL, "reject_id": REJECT_ID, "reject_card": REJECT_CARD,
        "constants": {
            "draw_seed": DRAW_SEED, "draw_fraction": DRAW_FRACTION,
            "shuffle_seed": SHUFFLE_SEED, "wrong_seed": WRONG_SEED,
            "row_seed": ROW_SEED, "calib_seed": CALIB_SEED, "calib_max": CALIB_MAX,
        },
        "wrong_tool_rule": "deterministic per-row choice from the 17 real caps, "
                           "excluding {gold, d1, d2}; seed prefix " + WRONG_SEED,
        "shuffle_rule": "deterministic per-row 4-way shuffle; seed prefix " + SHUFFLE_SEED,
        "row_order_rule": "deterministic per-split row-order shuffle over the MERGED "
                          "rows, then deterministic anti-adjacency repair; seed prefix "
                          + ROW_SEED + " (Query<->options<->gold/REJECT pairing preserved)",
        "row_order_permutation": row_order_permutation,
        "training_schema_keys": sorted(final_rows_by_split["train"][0].keys()),
        "construction_audit_file": "v2_construction_audit.jsonl",
        "counts": {"train": len(final_rows_by_split["train"]),
                   "val": len(final_rows_by_split["val"]),
                   "test": len(final_rows_by_split["test"]),
                   "synthetic_reject": part1_synth, "natural_reject": part2_natural},
        "part_counts": part_counts,
        "draw": drawn_report,
        "wrong_tool_distribution": dict(wrong_hist.most_common()),
        "calibration": {"seed": CALIB_SEED, "n_calib": n_calib,
                        "effective_train_rows": len(train_idx)},
        "audit": audit,
        "sha256": paths,
    }
    (DEST / "v2_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sums = [f"{paths[s]}  v2_{s}.jsonl" for s in SPLITS]
    for extra in ("manifest.json", "v2_build_report.json", "v2_construction_audit.jsonl"):
        sums.append(f"{sha256_file(DEST / extra)}  {extra}")
    (DEST / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    print(f"[v2] caps={len(caps)} conflicts=0")
    _sk = sorted(final_rows_by_split["train"][0].keys())
    print(f"[v2] schema_keys({len(_sk)}) = {_sk}")
    print(f"[v2] construction_audit sha256={audit_path}")
    print(f"[v2] rows train={len(final_rows_by_split['train'])} "
          f"val={len(final_rows_by_split['val'])} test={len(final_rows_by_split['test'])}")
    print(f"[v2] part_counts={part_counts}")
    print(f"[v2] synthetic_reject={part1_synth} natural_reject={part2_natural}")
    print(f"[v2] calibration n_calib={n_calib} seed={CALIB_SEED} "
          f"effective={len(train_idx)}")
    print(f"[v2] invariants_ok={audit['invariants']['ok']} "
          f"failures={audit['invariants']['failures'][:5]}")
    for s in SPLITS:
        a = audit["part_versus_row_order"][s]
        print(f"[v2] {s}: adj_p1p2={a['adjacent_p1p2_pairs']} "
              f"max_run={a['max_same_part_run']} sha256={paths[s]}")


if __name__ == "__main__":
    main()
