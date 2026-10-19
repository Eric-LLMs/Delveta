#!/usr/bin/env python3
"""Build the Delveta BGE-M3 dual-retrieval training data from the frozen
Query->Query set plus the live registry descriptions.

Two retrieval tasks share ONE query (never duplicated across a Task A/B pos list):

  Task B (query):  query -> the gold capability's standard/similar query anchor
                   (the frozen corpus908 rows, kept verbatim)
  Task A (desc):   query -> the gold capability's canonical registry description
                   (verbatim `description` from logs/_registry_dump.json)

Every unique query yields TWO physical records (one per task). The FlagEmbedding
collator samples exactly ONE positive per record, so the two positives must never
share a `pos` list (they are separate records sharing only the `query` text).

Manifest carries task / gold_capability_id / declared_lang / language_profile /
source / seed_family so the trainer and the evaluator never have to infer them.

Reads (read-only):  data/bge_m3_{train,test}.jsonl
                    data/bge_m3_{train,test}_manifest.jsonl
                    ../../../logs/_registry_dump.json
Writes (data/ only): bge_m3_{train,test}.jsonl            (merged A+B, canonical)
                     bge_m3_{train,test}_manifest.jsonl   (new schema)
                     bge_m3_{train,test}_{query,desc}.jsonl (+ manifests)
                     _build_dual_target_report.json        (machine-readable)
Originals are frozen under data/_backup_20261004_pre_dual_target/ before any write.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BACKUP = DATA / "_backup_20261004_pre_dual_target"
REG = HERE.parents[1] / "logs" / "_registry_dump.json"

# Script-presence rule (user ruling): a query is `mixed` when it carries BOTH a
# Han character and a Latin letter. It is never collapsed to `zh` merely because
# Han is present (that would hide the en-frame/zh-object generation artifact).
RE_HAN = re.compile(r"[\u4e00-\u9fff]")
RE_LATIN = re.compile(r"[A-Za-z]")


def read_jsonl(p: Path) -> list[dict]:
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def language_profile(text: str, fallback: str) -> str:
    h = bool(RE_HAN.search(text))
    l = bool(RE_LATIN.search(text))
    if h and l:
        return "mixed"
    if h:
        return "zh"
    if l:
        return "en"
    return fallback or "unknown"


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    assert BACKUP.is_dir(), f"backup missing: {BACKUP}"

    reg = {e["capability_id"]: e for e in json.load(open(REG, encoding="utf-8"))}
    desc = {cid: (e.get("description") or "") for cid, e in reg.items()}
    missing_desc = [c for c, d in desc.items() if not d.strip()]
    if missing_desc:
        raise RuntimeError(f"capabilities without a description: {missing_desc}")
    # A description duplicates another -> Task A false negative. Refuse.
    dup_desc = {d: c for d, c in Counter(desc.values()).items() if c > 1}
    if dup_desc:
        raise RuntimeError(f"duplicate descriptions across capabilities: {dup_desc}")

    report: dict = {
        "registry_capabilities": len(reg),
        "cap_research_intent_kind": reg["cap-research"]["intent_kind"],
        "cap_research_has_zh_description": "中文：" in desc["cap-research"],
        "description_chars": {c: len(d) for c, d in desc.items()},
        "backup_dir": BACKUP.name,
    }

    for split in ("train", "test"):
        rows = read_jsonl(DATA / f"bge_m3_{split}.jsonl")
        man = read_jsonl(DATA / f"bge_m3_{split}_manifest.jsonl")
        if len(rows) != len(man):
            raise RuntimeError(f"{split}: rows {len(rows)} != manifest {len(man)}")

        query_rows, query_man = [], []
        desc_rows, desc_man = [], []
        for r, m in zip(rows, man):
            gid = m["capability_id"]
            if gid not in reg:
                raise RuntimeError(f"{split}: unknown capability {gid!r}")
            prof = language_profile(r["query"], m.get("lang"))
            seed_family = m.get("seed_source")  # trustworthy only for test
            base_man = {
                "split": split,
                "gold_capability_id": gid,
                "declared_lang": m.get("lang"),
                "language_profile": prof,
                "source": m.get("source"),
                "seed_family": seed_family,
                "query_sha256": m.get("query_sha256") or sha256_text(r["query"]),
                "negative_capabilities": list(m.get("negative_capabilities") or []),
                "negative_provenance": m.get("negative_type") or "curated_confusable",
            }

            # ---- Task B: query -> real query anchor (frozen, verbatim) --------
            qm = dict(base_man, task="query")
            qm["positive_anchor"] = m.get("positive_anchor")
            query_rows.append({"query": r["query"], "pos": list(r["pos"]), "neg": list(r["neg"])})
            query_man.append(qm)

            # ---- Task A: query -> canonical description (verbatim) ------------
            neg_caps = list(m.get("negative_capabilities") or [])
            dm = dict(base_man, task="desc")
            dm["positive_source"] = "registry_description"
            dm["positive_language"] = "bilingual" if "中文：" in desc[gid] else "en_only"
            desc_rows.append({
                "query": r["query"],
                "pos": [desc[gid]],
                "neg": [desc[c] for c in neg_caps],
            })
            desc_man.append(dm)

        # canonical merged file: Task B block then Task A block.
        write_jsonl(DATA / f"bge_m3_{split}.jsonl", query_rows + desc_rows)
        write_jsonl(DATA / f"bge_m3_{split}_manifest.jsonl", query_man + desc_man)
        write_jsonl(DATA / f"bge_m3_{split}_query.jsonl", query_rows)
        write_jsonl(DATA / f"bge_m3_{split}_query_manifest.jsonl", query_man)
        write_jsonl(DATA / f"bge_m3_{split}_desc.jsonl", desc_rows)
        write_jsonl(DATA / f"bge_m3_{split}_desc_manifest.jsonl", desc_man)

        report[split] = {
            "input_rows": len(rows),
            "task_query_rows": len(query_rows),
            "task_desc_rows": len(desc_rows),
            "merged_rows": len(query_rows) + len(desc_rows),
            "gold_capabilities": len({m["capability_id"] for m in man}),
            "declared_lang": dict(Counter(m.get("lang") for m in man)),
            "language_profile_query": dict(Counter(m["language_profile"] for m in query_man)),
            "language_profile_desc": dict(Counter(m["language_profile"] for m in desc_man)),
            "negative_provenance": dict(Counter(m["negative_provenance"] for m in query_man)),
            "per_capability_query": dict(sorted(Counter(m["capability_id"] for m in man).items())),
            "per_capability_desc": dict(sorted(Counter(m["gold_capability_id"] for m in desc_man).items())),
        }

    (DATA / "_build_dual_target_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
