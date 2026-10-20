import csv, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(r"e:/Work Space/Git Repositories/DeepDive")
RUN = REPO / "scripts/laya_finetune/data_v4_build/smoke50_r1"
D = RUN / "data"
V2 = REPO / "scripts/laya_finetune/V2/data"
TH = ("threshold_082", "threshold_085", "threshold_088")
SPLITS = ("train", "val", "test")
REJECT = "REJECT"


def load(path, split):
    return [json.loads(l) for l in (path / f"v2_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def jt(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


out = {}

# ---- schema: V2 vs V4 ------------------------------------------------------
v2_rows = [r for s in SPLITS for r in load(V2, s)]
v4_rows = [r for t in TH for s in SPLITS for r in load(D / t, s)]
def describe(rows):
    keys = sorted(rows[0])
    types = {k: sorted({jt(r.get(k)) for r in rows}) for k in keys}
    shuffle_types = {k: sorted({jt(r["shuffle"].get(k)) for r in rows if "shuffle" in r})
                     for k in sorted(rows[0].get("shuffle", {}))}
    counts = Counter(len(r["options"]) for r in rows)
    return {"keys": keys, "key_count": len(keys), "types": types,
            "shuffle_keys": sorted(rows[0]["shuffle"]),
            "shuffle_types": shuffle_types,
            "option_counts": dict(sorted(counts.items())),
            "rows": len(rows)}
out["schema"] = {"v2": describe(v2_rows), "v4": describe(v4_rows)}
out["schema"]["key_delta"] = {
    "v2_only": sorted(set(v2_rows[0]) - set(v4_rows[0])),
    "v4_only": sorted(set(v4_rows[0]) - set(v2_rows[0])),
}
out["schema"]["per_key_type_equal"] = {
    k: out["schema"]["v2"]["types"].get(k) == out["schema"]["v4"]["types"].get(k)
    for k in out["schema"]["v4"]["keys"]}
vm = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
v4m = json.loads((D / TH[0] / "manifest.json").read_text(encoding="utf-8"))
out["schema"]["manifest"] = {
    "v2_keys": sorted(vm), "v4_keys": sorted(v4m),
    "missing_from_v4": sorted(set(vm) - set(v4m)),
    "extra_in_v4": sorted(set(v4m) - set(vm)),
    "v2_option_slots": vm.get("option_slots"),
    "v4_option_slots": v4m.get("option_slots"),
    "loader_required": ["bench", "option_slots", "rows", "calibration"],
    "loader_required_present_v4": all(k in v4m for k in ("bench", "option_slots", "rows", "calibration")),
}

# ---- distributions ---------------------------------------------------------
sample = list(csv.DictReader(open(D / "sample_manifest.csv", encoding="utf-8-sig")))
by_key = {f'{r["source_file"]}:{r["source_row"]}:{r["source_row_id"]}': r for r in sample}
sel = [r for r in sample if r["smoke_selected"] == "True"]
sel_keys = [f'{r["source_file"]}:{r["source_row"]}:{r["source_row_id"]}' for r in sel]

stats = {}
for t in TH:
    rows = [r for s in SPLITS for r in load(D / t, s)]
    cap_gold = Counter(r["gold_index"] for r in rows if r["target_kind"] == "capability")
    rej = Counter(r["reject_index"] for r in rows if r["target_kind"] == "reject")
    stats[t] = {
        "rows": len(rows), "K": dict(sorted(Counter(len(r["order"]) - 1 for r in rows).items())),
        "kind": dict(Counter(r["target_kind"] for r in rows)),
        "capability_gold_index": dict(sorted(cap_gold.items())),
        "reject_gold_index": dict(sorted(rej.items())),
        "reject_position_all": dict(sorted(Counter(r["reject_index"] for r in rows).items())),
        "split": dict(Counter(r["split"] for r in rows)),
        "lang": dict(Counter(r["lang"] for r in rows)),
    }
out["stats"] = stats

# ---- threshold transitions -------------------------------------------------
by_q = {}
for t in TH:
    for s in SPLITS:
        for r in load(D / t, s):
            by_q.setdefault(r["source_query_id"], {})[t] = set(r["order"]) - {REJECT}
trans = {}
for lo, hi in zip(TH, TH[1:]):
    added = removed = changed = 0
    for q, sets in by_q.items():
        a, b = sets[lo], sets[hi]
        added += len(a - b)
        removed += len(b - a)
        changed += int(a != b)
    trans[f"{lo}->{hi}"] = {"queries_changed": changed,
                            "candidates_removed": removed,
                            "candidates_added": added}
out["threshold_transitions"] = trans

# ---- split / family leakage ------------------------------------------------
split_members = {s: set() for s in SPLITS}
family_splits = defaultdict(set)
norm_splits = defaultdict(set)
for r in sel:
    k = f'{r["source_file"]}:{r["source_row"]}:{r["source_row_id"]}'
    split_members[r["split"]].add(k)
    family_splits[r["query_family_id"]].add(r["split"])
    norm_splits[r["normalized_query_sha256"]].add(r["split"])
out["split_check"] = {
    "counts": {s: len(v) for s, v in split_members.items()},
    "overlap_train_val": len(split_members["train"] & split_members["val"]),
    "overlap_train_test": len(split_members["train"] & split_members["test"]),
    "overlap_val_test": len(split_members["val"] & split_members["test"]),
    "families_cross_split": sum(1 for v in family_splits.values() if len(v) > 1),
    "duplicate_queries_cross_split": sum(1 for v in norm_splits.values() if len(v) > 1),
    "families_total": len(family_splits),
}

# ---- row-level seed sensitivity -------------------------------------------
seed_sens = {}
for s in SPLITS:
    n = len(split_members[s])
    a = list(range(n)); random.Random(f"reject-rows-v1:{s}").shuffle(a)
    b = list(range(n)); random.Random(f"reject-rows-v1-alt:{s}").shuffle(b)
    seed_sens[s] = {"n": n, "positions_differ": sum(1 for x, y in zip(a, b) if x != y)}
out["row_seed_sensitivity"] = seed_sens

# ---- evidence coverage -----------------------------------------------------
missing = []
floor_below = 0
for line in (D / "shared_retrieval_evidence.jsonl").read_text(encoding="utf-8").splitlines():
    pass
ev_keys = set()
for line in (D / "shared_retrieval_evidence.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        ev_keys.add(json.loads(line)["source_query_id"])
missing = sorted(set(sel_keys) - ev_keys)
out["evidence"] = {"queries_with_evidence": len(ev_keys),
                   "queries_without_evidence": missing}
for k in missing:
    r = by_key[k]
    out["evidence"].setdefault("missing_detail", []).append(
        {"key": k, "query": r["query"], "split": r["split"]})

# ---- node-level example diff ----------------------------------------------
ex_v2 = next(r for r in load(V2, "train") if r["target_kind"] == "capability")
ex_v4 = next(r for r in load(D / TH[1], "train") if len(r["order"]) == 3 and r["target_kind"] == "capability")
out["example"] = {
    "v2": {"id": ex_v2["id"], "keys": sorted(ex_v2), "order_len": len(ex_v2["order"]),
           "types": {k: jt(ex_v2[k]) for k in sorted(ex_v2)},
           "shuffle_perm_len": len(ex_v2["shuffle"]["permutation"])},
    "v4": {"id": ex_v4["id"], "keys": sorted(ex_v4), "order_len": len(ex_v4["order"]),
           "types": {k: jt(ex_v4[k]) for k in sorted(ex_v4)},
           "shuffle_perm_len": len(ex_v4["shuffle"]["permutation"])},
}

print(json.dumps(out, ensure_ascii=False, indent=2))
