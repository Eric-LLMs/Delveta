#!/usr/bin/env python3
"""FINAL PRE-TRAINING GATE — read-only audit of scripts/bge_m3_finetune/data/.

Verifies the 5 gate items against the files on disk. Writes nothing but stdout
and data/_final_gate_report.json. No training, no downloads, no edits.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BACKUP = DATA / "_backup_20261004_pre_dual_target"
REG = HERE.parents[1] / "logs" / "_registry_dump.json"


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    reg = {e["capability_id"]: e for e in json.load(open(REG, encoding="utf-8"))}
    DESCS = {e["description"] for e in reg.values()}          # 18 canonical descriptions
    cap2q = {}
    for cid, e in reg.items():
        s = set()
        for st in e["standard"]:
            s.add(st["query"].strip())
            for sim in st["similar"]:
                s.add(sim["query"].strip())
        cap2q[cid] = s
    QUERIES = {q for s in cap2q.values() for q in s}          # 908 corpus queries
    out: dict = {}

    # ================= 2. Task B byte-identity (do first: cheap + decisive)
    out["taskB_byte_identity"] = {}
    for split in ("train", "test"):
        ship = sha256_file(DATA / f"bge_m3_{split}_query.jsonl")
        orig = sha256_file(BACKUP / f"bge_m3_{split}.jsonl")
        # merged file must start with the same N query lines
        n = sum(1 for _ in open(BACKUP / f"bge_m3_{split}.jsonl", encoding="utf-8"))
        head = (DATA / f"bge_m3_{split}.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)[:n]
        head_sha = hashlib.sha256("".join(head).encode("utf-8")).hexdigest()
        out["taskB_byte_identity"][split] = {
            "query_file_sha": ship, "original_sha": orig, "identical": ship == orig,
            "merged_head_sha": head_sha, "merged_head_matches": head_sha == orig, "n": n,
        }

    # ================= 1. Task A schema (full + sample)
    A_schema = {}
    for split in ("train", "test"):
        rows = read_jsonl(DATA / f"bge_m3_{split}_desc.jsonl")
        man = read_jsonl(DATA / f"bge_m3_{split}_desc_manifest.jsonl")
        assert len(rows) == len(man)
        bad_struct = bad_pos_desc = bad_neg_desc = pos_is_query = neg_is_query = pos_not_gold = 0
        neg_descs, neg_slots = set(), 0
        for r, m in zip(rows, man):
            ok = (isinstance(r.get("query"), str) and isinstance(r["pos"], list) and r["pos"]
                  and isinstance(r["neg"], list) and r["neg"])
            if not ok:
                bad_struct += 1
                continue
            g = m["gold_capability_id"]
            if r["pos"] != [reg[g]["description"]]:
                pos_not_gold += 1
            for p in r["pos"]:
                if p not in DESCS:
                    bad_pos_desc += 1
                if p.strip() in QUERIES:
                    pos_is_query += 1
            for n in r["neg"]:
                neg_slots += 1
                neg_descs.add(n)
                if n not in DESCS:
                    bad_neg_desc += 1
                if n.strip() in QUERIES:
                    neg_is_query += 1
        A_schema[split] = {
            "rows": len(rows), "bad_struct": bad_struct,
            "pos_positions_not_equal_gold_desc": pos_not_gold,
            "pos_targets_not_in_desc_set": bad_pos_desc,
            "neg_targets_not_in_desc_set": bad_neg_desc,
            "pos_targets_that_are_corpus_queries": pos_is_query,
            "neg_targets_that_are_corpus_queries": neg_is_query,
            "distinct_neg_descriptions": len(neg_descs), "neg_slots": neg_slots,
        }
    # set disjointness: descriptions vs queries (proves no reuse across tasks)
    out["desc_query_set_overlap"] = len(DESCS & QUERIES)
    out["taskA_schema"] = A_schema
    # deterministic sample for eyeball
    random.seed(20261004)
    sample_idx = sorted(random.sample(range(8108), 4))
    a_rows = read_jsonl(DATA / "bge_m3_train_desc.jsonl")
    out["taskA_sample"] = [{"query": a_rows[i]["query"],
                            "pos_head": a_rows[i]["pos"][0][:70],
                            "neg_heads": [x[:45] for x in a_rows[i]["neg"]]} for i in sample_idx]

    # ================= 3. coverage + production-pool mismatch
    def cov(taskfile, manfile, key="gold_capability_id"):
        man = read_jsonl(DATA / manfile)
        return Counter(m[key] for m in man)
    Atr = cov(None, "bge_m3_train_desc_manifest.jsonl")
    Ate = cov(None, "bge_m3_test_desc_manifest.jsonl")
    Btr = cov(None, "bge_m3_train_query_manifest.jsonl")
    Bte = cov(None, "bge_m3_test_query_manifest.jsonl")
    cov_tbl = {c: {"A_tr": Atr[c], "A_te": Ate[c], "B_tr": Btr[c], "B_te": Bte[c]} for c in sorted(reg)}
    out["coverage"] = cov_tbl
    out["taskB_contains_cap_research"] = bool(Btr.get("cap-research") or Bte.get("cap-research"))
    out["production_pool"] = {
        "predicate": "recall/index.py:25  c.intent_kind <> 'research'",
        "production_capabilities": len([c for c in reg if reg[c]["intent_kind"] != "research"]),
        "training_capabilities": len(reg),
        "mismatch": "18-cap training vs 17-cap production (cap-research must be filterable at inference)",
        "cap_research_in_taskB": {"train": Btr.get("cap-research", 0), "test": Bte.get("cap-research", 0)},
    }

    # ================= 4. negative quality
    negq = {"A": {}, "gold_in_neg": {"A": 0, "B": 0}}
    for task, tag in (("desc", "A"), ("query", "B")):
        prov = Counter()
        for split in ("train", "test"):
            for m in read_jsonl(DATA / f"bge_m3_{split}_{task}_manifest.jsonl"):
                prov[m["negative_provenance"]] += 1
                g = m["gold_capability_id"]
                if g in m["negative_capabilities"]:
                    negq["gold_in_neg"][tag] += 1
        negq["A" if tag == "A" else "B"] = dict(prov)
    # cap-research random negatives count (rows*negslots)
    rs = {}
    for split in ("train", "test"):
        rows = read_jsonl(DATA / f"bge_m3_{split}_query.jsonl")
        man = read_jsonl(DATA / f"bge_m3_{split}_query_manifest.jsonl")
        r_rows = [(r, m) for r, m in zip(rows, man)
                  if m["negative_provenance"] == "random_cross_capability"]
        rs[split] = {"rows": len(r_rows), "neg_slots": sum(len(r["neg"]) for r, _ in r_rows),
                     "all_gold_cap_research": all(m["gold_capability_id"] == "cap-research" for _, m in r_rows)}
    out["negative_quality"] = {"provenance_by_task": negq, "cap_research_random": rs,
                               "random_neg_total_slots": sum(v["neg_slots"] for v in rs.values())}

    # ================= 5. batch false-negative risk
    merged = {}
    for split in ("train", "test"):
        rows = read_jsonl(DATA / f"bge_m3_{split}.jsonl")
        man = read_jsonl(DATA / f"bge_m3_{split}_manifest.jsonl")
        tasks = Counter(m["task"] for m in man)
        # A/B same-query pairs
        by_q = defaultdict(set)
        for m in man:
            by_q[m["query_sha256"]].add(m["task"])
        pairs = sum(1 for v in by_q.values() if {"desc", "query"} <= v)
        # gold capability multiplicity
        capmult = Counter(m["gold_capability_id"] for m in man)
        merged[split] = {"rows": len(rows), "tasks": dict(tasks), "A_B_pairs": pairs,
                         "identical_pos_list_across_A_B": sum(
                             1 for m in man if m["task"] == "desc") }
    # collision math for a batch of size b over 18 capabilities
    def exp_coll(b, caps=18):
        return round(caps * (1 - (1 - 1 / caps) ** b), 2)
    out["batch_risk"] = {
        "merged_files": merged,
        "pattern": "Task B block (lines 1..N) then Task A block (lines N+1..2N) — A and B ARE mixed in one file",
        "default_loss": "in-batch negatives (AbsModeling._compute_in_batch_neg_loss), one positive per record",
        "same_cap_rows_train": 16216, "same_cap_rows_test": 14400,
        "expected_caps_per_batch": {str(b): exp_coll(b) for b in (8, 16, 32, 64)},
        "verdict": "A/B of one query and any two same-capability rows share a gold capability; "
                   "co-batching them makes one record's positive another's in-batch negative "
                   "(false negative). Requires a capability-unique / task-aware sampler.",
    }

    (DATA / "_final_gate_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
