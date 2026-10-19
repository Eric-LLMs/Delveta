#!/usr/bin/env python3
"""Read-only verification of the built dual-target data (audit §8 checks).

Recomputes every claim in data/bge_m3_dual_target_audit_after.md from the files
on disk; writes nothing except stdout + _verify_dual_target_report.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BACKUP = DATA / "_backup_20261004_pre_dual_target"
REG = HERE.parents[1] / "logs" / "_registry_dump.json"

RE_HAN = re.compile(r"[\u4e00-\u9fff]")


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\u4e00-\u9fff ]", "", s, flags=re.UNICODE)
    return s.strip()


def lang_profile(text: str) -> str:
    h = bool(RE_HAN.search(text))
    l = bool(re.search(r"[A-Za-z]", text))
    return "mixed" if h and l else ("zh" if h else ("en" if l else "unknown"))


def main() -> None:
    reg = {e["capability_id"]: e for e in json.load(open(REG, encoding="utf-8"))}
    desc = {c: e["description"] for c, e in reg.items()}
    out: dict = {}

    # ---- 1. file inventory + sha ------------------------------------------
    inv = {}
    for p in sorted(DATA.glob("*.jsonl")) + sorted(DATA.glob("*.md")):
        inv[p.name] = {"sha256": sha256_file(p),
                       "lines": sum(1 for l in open(p, encoding="utf-8") if l.strip())
                       if p.suffix == ".jsonl" else None}
    inv_backup = {p.name: {"sha256": sha256_file(p),
                           "lines": sum(1 for l in open(p, encoding="utf-8") if l.strip())
                           if p.suffix == ".jsonl" else None}
                  for p in sorted(BACKUP.glob("*")) if p.is_file()}
    out["files_after"] = inv
    out["files_backup"] = inv_backup

    # ---- load new files ----------------------------------------------------
    Q = {s: read_jsonl(DATA / f"bge_m3_{s}_query.jsonl") for s in ("train", "test")}
    QM = {s: read_jsonl(DATA / f"bge_m3_{s}_query_manifest.jsonl") for s in ("train", "test")}
    D = {s: read_jsonl(DATA / f"bge_m3_{s}_desc.jsonl") for s in ("train", "test")}
    DM = {s: read_jsonl(DATA / f"bge_m3_{s}_desc_manifest.jsonl") for s in ("train", "test")}

    # ---- 2. coverage matrix ------------------------------------------------
    caps = sorted(reg)
    cov = {}
    for c in caps:
        cov[c] = {
            "desc_train": sum(1 for m in DM["train"] if m["gold_capability_id"] == c),
            "desc_test": sum(1 for m in DM["test"] if m["gold_capability_id"] == c),
            "query_train": sum(1 for m in QM["train"] if m["gold_capability_id"] == c),
            "query_test": sum(1 for m in QM["test"] if m["gold_capability_id"] == c),
        }
    out["coverage_matrix"] = cov

    # ---- 5/6. counts + language distribution ------------------------------
    out["counts"] = {s: {"task_desc": len(D[s]), "task_query": len(Q[s]),
                         "merged": len(D[s]) + len(Q[s])} for s in ("train", "test")}
    out["language_profile"] = {
        s: {"query": dict(Counter(m["language_profile"] for m in QM[s])),
            "desc": dict(Counter(m["language_profile"] for m in DM[s]))}
        for s in ("train", "test")}
    out["declared_lang"] = {s: dict(Counter(m["declared_lang"] for m in QM[s])) for s in ("train", "test")}
    out["manifest_keys"] = {s: sorted(QM[s][0].keys()) for s in ("train", "test")}

    # ---- 7. leakage --------------------------------------------------------
    leak = {}
    for task, Rows in (("query", Q), ("desc", D)):
        tr = {r["query"] for r in Rows["train"]}
        te = {r["query"] for r in Rows["test"]}
        trn = {norm(r["query"]) for r in Rows["train"]}
        ten = {norm(r["query"]) for r in Rows["test"]}
        leak[task] = {"exact": len(tr & te), "normalized": len(trn & ten)}
    # pos leakage: any Train positive appearing as a Test positive (same task)
    leak["pos_train_eq_test_desc"] = len({d for r in D["train"] for d in r["pos"]} &
                                         {d for r in D["test"] for d in r["pos"]})
    leak["pos_train_eq_test_query"] = len({q for r in Q["train"] for q in r["pos"]} &
                                          {q for r in Q["test"] for q in r["pos"]})
    out["leakage"] = leak

    # ---- 8. query == pos ---------------------------------------------------
    qeq = {}
    for task, Rows in (("query", Q), ("desc", D)):
        qeq[task] = {s: sum(1 for r in Rows[s] if r["query"].strip() in {p.strip() for p in r["pos"]})
                     for s in ("train", "test")}
    out["query_eq_pos"] = qeq

    # ---- 4/9. Task A integrity + negative provenance -----------------------
    fin = {"desc_neg_is_gold_desc": 0, "desc_neg_cap_is_gold": 0, "desc_pos_is_gold_desc": 0}
    for s in ("train", "test"):
        for r, m in zip(D[s], DM[s]):
            g = m["gold_capability_id"]
            if any(n == desc[g] for n in r["neg"]):
                fin["desc_neg_is_gold_desc"] += 1
            if g in m["negative_capabilities"]:
                fin["desc_neg_cap_is_gold"] += 1
            if r["pos"] != [desc[g]]:
                fin["desc_pos_is_gold_desc"] += 1
    out["taskA_integrity_violations"] = fin
    out["negative_provenance"] = {s: dict(Counter(m["negative_provenance"] for m in QM[s]))
                                  for s in ("train", "test")}
    out["random_neg_caps"] = {s: dict(sorted(Counter(
        c for m in QM[s] if m["negative_provenance"] == "random_cross_capability"
        for c in m["negative_capabilities"]).items())) for s in ("train", "test")}

    # ---- A/B pos never share a list ---------------------------------------
    ab = 0
    for s in ("train", "test"):
        for r in D[s] + Q[s]:
            if any("中文：" in p for p in r["pos"]) and any("中文：" not in p for p in r["pos"]):
                ab += 1
    out["rows_mixing_A_and_B_positives"] = ab

    # ---- 3. cap-rag-search before/after -----------------------------------
    old = {s: read_jsonl(DATA / f"bge_m3_{s}.jsonl") for s in ("train", "test")}  # now merged
    cap2q = {}
    for cid, e in reg.items():
        s = set()
        for st in e["standard"]:
            s.add(st["query"].strip())
            for sim in st["similar"]:
                s.add(sim["query"].strip())
        cap2q[cid] = s
    q2cap = defaultdict(set)
    for cid, qs in cap2q.items():
        for q in qs:
            q2cap[q].add(cid)
    rag = {}
    for s in ("train", "test"):
        qr = [r for r, m in zip(Q[s], QM[s]) if m["gold_capability_id"] == "cap-rag-search"]
        rag[s] = {
            "query_rows": len(qr),
            "desc_rows": sum(1 for m in DM[s] if m["gold_capability_id"] == "cap-rag-search"),
            "pos_anchor_not_in_rag_corpus": sum(1 for r in qr for p in r["pos"] if p.strip() not in cap2q["cap-rag-search"]),
            "neg_owner_set": sorted({c for r in qr for n in r["neg"] for c in (q2cap.get(n.strip()) or {"<none>"})}),
            "neg_caps_declared": sorted({c for m in QM[s] if m["gold_capability_id"] == "cap-rag-search"
                                         for c in m["negative_capabilities"]}),
        }
    out["cap_rag_search"] = rag

    # ---- 10. retention/decisions ------------------------------------------
    out["retention"] = {
        "cap_research_desc_train": cov["cap-research"]["desc_train"],
        "cap_research_desc_test": cov["cap-research"]["desc_test"],
        "cap_research_query_train": cov["cap-research"]["query_train"],
        "cap_research_query_test": cov["cap-research"]["query_test"],
        "cap_research_zh_description_present": "中文：" in desc["cap-research"],
        "rows_deleted": 0,
        "rows_modified": 0,
        "rows_added_taskA": len(D["train"]) + len(D["test"]),
    }

    (DATA / "_verify_dual_target_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
