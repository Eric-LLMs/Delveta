#!/usr/bin/env python3
"""V2 final generation — one uniform rule for ALL splits, in this exact order:

  current query
   -> SQL similar-query Recall   (already captured: min_score=0.0, NO threshold)
   -> remove self-hit            (matched source_query == query, verbatim)
   -> capability_id dedup/aggregate (same capability keeps the MAX score)
   -> sort by max score DESC (tie: capability_id)
   -> Top-3

Not any other order: self-hit removal happens BEFORE dedup and BEFORE Top-3.
No threshold. No manual gold. No candidate backfill. No Laya in the loop.
Train1/Val1/Test1, old T2@0.60 and old v2 are never touched.

Audit snapshots kept:
  v2_recall_full.jsonl     the raw recall as captured (self-hits still present)
  v2_recall_selfexcl.jsonl the same rows with self-hits removed (per-hit audit)

Inputs (this dir): v2_raw_recall.jsonl + V1 bundles (card map only).
Outputs (v2_final/): v2_{train2,val2,test2}.jsonl, v2_build_report.json,
                     v2_card_map.json, SHA256SUMS, the two recall snapshots.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DEST = OUT / "v2_final"
BUNDLE = REPO / "scripts" / "laya_finetune" / "data"
RAW = OUT / "v2_raw_recall.jsonl"

SRC_OF = {"train": "train2", "val": "val2", "test": "test2"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_card_map():
    cards = defaultdict(set)
    for split in ("train", "val", "test"):
        with gzip.open(BUNDLE / f"layachoice_v1_{split}_b_noprov.jsonl.gz", "rt",
                       encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                for cap, card in zip(r["order"], r["options"]):
                    cards[cap].add(card)
    canon = {c: next(iter(s)) for c, s in cards.items() if len(s) == 1}
    conflicts = {c: sorted(s) for c, s in cards.items() if len(s) > 1}
    return canon, conflicts


def is_self(hit: dict, query: str) -> bool:
    return hit["source_query"].strip() == query.strip()


def dedup_and_rank(hits: list[dict]) -> list[dict]:
    """capability_id dedup (keep MAX score) -> sort score DESC (tie capability_id)."""
    best: dict[str, dict] = {}
    for h in hits:
        cap = h["capability_id"]
        cur = best.get(cap)
        if cur is None or h["score"] > cur["max_similarity_score"]:
            best[cap] = {"capability_id": cap, "max_similarity_score": h["score"],
                         "n_support": 1, "top_query_id": h["source_query_id"],
                         "top_query": h["source_query"]}
        else:
            cur["n_support"] += 1
    return sorted(best.values(), key=lambda c: (-c["max_similarity_score"],
                                                c["capability_id"]))


def main():
    canon, conflicts = build_card_map()
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "v2_card_map.json").write_text(json.dumps(
        {"canonical_caps": sorted(canon), "n_canonical": len(canon),
         "conflicts": {k: v for k, v in conflicts.items()}},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    f_full = (DEST / "v2_recall_full.jsonl").open("w", encoding="utf-8", newline="\n")
    f_sx = (DEST / "v2_recall_selfexcl.jsonl").open("w", encoding="utf-8", newline="\n")

    by_src = defaultdict(list)
    stats = {}
    self_removed = Counter()
    for line in RAW.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        src = rec["source"]
        gold = rec["gold_capability_id"]
        raw = rec["raw"]
        query = rec["query"]

        f_full.write(json.dumps({"source": src, "sample_id": rec["sample_id"],
                                 "query": query, "language": rec["language"],
                                 "gold_capability_id": gold, "raw": raw},
                                ensure_ascii=False) + "\n")

        # STEP 2: remove self-hit (before dedup, before Top-3)
        kept = [h for h in raw if not is_self(h, query)]
        n_self = len(raw) - len(kept)
        self_removed[src] += n_self

        f_sx.write(json.dumps({"source": src, "sample_id": rec["sample_id"],
                               "query": query, "language": rec["language"],
                               "gold_capability_id": gold,
                               "self_hits_removed": n_self,
                               "raw": kept}, ensure_ascii=False) + "\n")

        # STEP 3-5: dedup cap max -> sort DESC -> Top-3
        agg = dedup_and_rank(kept)
        top3 = agg[:3]
        order = [c["capability_id"] for c in top3]
        scores = [c["max_similarity_score"] for c in top3]
        cards = [canon.get(c, "") for c in order]
        gi = order.index(gold) if gold in order else -1
        gold_rank = gi + 1 if gi >= 0 else None
        gold_kept = [h["score"] for h in kept if h["capability_id"] == gold]
        gold_score = max(gold_kept) if gold_kept else None

        row = {
            "source": src, "sample_id": rec["sample_id"], "query": query,
            "language": rec["language"], "gold_capability_id": gold,
            "K_aggregated": len(agg), "recall_raw_count": len(raw),
            "self_hits_removed": n_self,
            "aggregated_candidates": agg,
            "top3_order": order, "top3_scores": scores, "top3_cards": cards,
            "gold_in_candidates": gi >= 0, "gold_rank": gold_rank,
            "gold_recall_score": gold_score,
            "order": order, "options": cards, "gold": gi, "d1": 0, "d2": 1,
        }
        by_src[src].append(row)
        st = stats.setdefault(src, {"n": 0, "K": Counter(), "gold_rank": Counter(),
                                    "top1": 0, "top3": 0, "missing": 0, "empty": 0,
                                    "self": 0, "rows_with_self": 0})
        st["n"] += 1
        st["K"][len(agg)] += 1
        if n_self:
            st["self"] += n_self
            st["rows_with_self"] += 1
        if not order:
            st["empty"] += 1
        if gold_rank == 1:
            st["top1"] += 1
        if gi >= 0:
            st["top3"] += 1
            st["gold_rank"][gold_rank] += 1
        else:
            st["missing"] += 1

    f_full.close()
    f_sx.close()

    out_paths = {}
    for src, rows in by_src.items():
        name = SRC_OF[src]
        p = DEST / f"v2_{name}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_paths[name] = sha256_file(p)

    report = {
        "threshold": None, "self_hit_excluded": True,
        "rule_order": ["recall(min_score=0.0,no threshold)", "remove self-hit",
                       "capability_id dedup (keep max score)", "sort score DESC",
                       "Top-3"],
        "self_hit_def": "matched corpus row whose source_query equals the query verbatim",
        "card_map": {"n_canonical": len(canon), "conflicts": list(conflicts)},
        "self_hits_removed": dict(self_removed),
        "sha256": out_paths,
        "per_source": {},
    }
    for src, st in stats.items():
        n = st["n"]
        report["per_source"][SRC_OF[src]] = {
            "n": n,
            "K_distribution": {str(k): v for k, v in sorted(st["K"].items())},
            "self_hits_removed": st["self"], "rows_with_self_hit": st["rows_with_self"],
            "gold_in_top1": st["top1"], "gold_in_top1_pct": round(100*st["top1"]/n, 2),
            "gold_in_top3": st["top3"], "gold_in_top3_pct": round(100*st["top3"]/n, 2),
            "gold_missing": st["missing"], "gold_missing_pct": round(100*st["missing"]/n, 2),
            "empty_candidate_sets": st["empty"],
            "gold_rank_distribution": {str(k): st["gold_rank"][k]
                                       for k in sorted(st["gold_rank"])},
        }
    (DEST / "v2_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sums = []
    for name in ("train2", "val2", "test2"):
        sums.append(f"{out_paths[name]}  v2_{name}.jsonl")
    for extra in ("v2_build_report.json", "v2_card_map.json",
                  "v2_recall_full.jsonl", "v2_recall_selfexcl.jsonl"):
        sums.append(f"{sha256_file(DEST / extra)}  {extra}")
    (DEST / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    print(f"[v2-final] card_map caps={len(canon)} conflicts={list(conflicts)}")
    print(f"[v2-final] self_hits_removed={dict(self_removed)}")
    for src in ("train2", "val2", "test2"):
        r = report["per_source"][src]
        print(f"[v2-final] {src}: n={r['n']} K={r['K_distribution']} "
              f"self={r['self_hits_removed']} "
              f"gold Top1={r['gold_in_top1_pct']}% ({r['gold_in_top1']}) "
              f"Top3={r['gold_in_top3_pct']}% ({r['gold_in_top3']}) "
              f"missing={r['gold_missing']} empty={r['empty_candidate_sets']} "
              f"rank={r['gold_rank_distribution']}")
        print(f"           sha256={out_paths[src]}")


if __name__ == "__main__":
    main()
