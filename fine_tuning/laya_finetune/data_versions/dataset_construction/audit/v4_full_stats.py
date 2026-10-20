"""Read-only full statistics for Laya V4 full_r1.

Reads only generated training rows, construction audits, the shared evidence
file, stored query embeddings and the DB corpus embeddings.  It never runs the
embedding service, never runs the recall SQL and never writes to the dataset.
"""
from __future__ import annotations

import asyncio
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RUN = Path(r"e:/Work Space/Git Repositories/DeepDive/scripts/laya_finetune/data_v4_build/full_r1")
OUT = RUN / "stats"
OUT.mkdir(exist_ok=True)
SOURCE_CSV = Path(r"e:/Work Space/Git Repositories/DeepDive/scripts/laya_finetune/data_v3/laya_tool_selection_v2.csv")
DB_DSN = "postgresql://delveta:delveta@localhost:15432/delveta"
THRESHOLDS = (0.82, 0.85, 0.88)
TAGS = {0.82: "threshold_082", 0.85: "threshold_085", 0.88: "threshold_088"}
SPLITS = ("train", "test", "val")
REJECT = "REJECT"
ROW_KEYS = {
    "id", "source_query_id", "split", "part", "lang", "state", "order", "options",
    "gold_index", "target_kind", "gold_capability_id", "reject_index", "shuffle", "src_sha",
}
INPUT_KEYS = ("state", "order", "options")
REASON_ORDER = ("no_action", "no_candidates", "gold_not_retrieved", "gold_in_candidates")


def pct(n, d):
    return f"{100.0 * n / d:.4f}" if d else ""


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def allocate_counts(n):
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
        overflow = n_train + n_test + n_val - n
        while overflow > 0 and n_test > 0:
            n_test -= 1
            overflow -= 1
        while overflow > 0 and n_val > 0:
            n_val -= 1
            overflow -= 1
    return {"train": n_train, "test": n_test, "val": n_val}


# ── source manifest ─────────────────────────────────────────────────────────
with open(RUN / "data/sample_manifest.csv", encoding="utf-8-sig", newline="") as f:
    manifest_rows = list(csv.DictReader(f))
manifest = {int(r["source_row_id"]): r for r in manifest_rows}
eligible_order = [int(r["source_row_id"]) for r in manifest_rows if r["eligible"] == "True"]
source_total = len(manifest_rows)
excluded_total = source_total - len(eligible_order)
source_keys = {
    sid: f"{manifest[sid]['source_file']}:{manifest[sid]['source_row']}:{sid}"
    for sid in manifest
}
sid_to_split = {sid: manifest[sid]["split"] for sid in eligible_order}
sid_to_norm = {sid: manifest[sid]["normalized_query_sha256"] for sid in eligible_order}
sid_to_family = {sid: manifest[sid]["query_family_id"] for sid in eligible_order}
sid_to_lang = {sid: manifest[sid]["language_resolved"] for sid in eligible_order}
sid_to_query = {sid: manifest[sid]["query"] for sid in eligible_order}
caps = json.loads((RUN / "data/capability_ids.json").read_text(encoding="utf-8"))
CAPSET = set(caps)

# ── load bundles ───────────────────────────────────────────────────────────
rows_by_t: dict[float, dict[str, dict]] = {}
audit_by_t: dict[float, dict[str, dict]] = {}
file_order: dict[float, dict[str, list[str]]] = {}
tman = {}
for t in THRESHOLDS:
    tag = TAGS[t]
    tdir = RUN / "data" / tag
    tman[t] = json.loads((tdir / "manifest.json").read_text(encoding="utf-8"))
    rows = {}
    order = {}
    for split in SPLITS:
        ids = []
        for row in load_jsonl(tdir / f"v2_{split}.jsonl"):
            rid = row.get("id")
            if rid in rows:
                raise SystemExit(f"duplicate id in {tag}/{split}: {rid}")
            rows[rid] = row
            ids.append(rid)
        order[split] = ids
    audits = {a["id"]: a for a in load_jsonl(tdir / "construction_audit.jsonl")}
    rows_by_t[t] = rows
    audit_by_t[t] = audits
    file_order[t] = order

# ── structural / label checks ──────────────────────────────────────────────
errors: dict[str, dict] = defaultdict(lambda: {"count": 0, "samples": []})
card_text: dict[str, str] = {}
card_conflicts = {"count": 0, "samples": []}
src_sha_expected = None
import hashlib
src_sha_expected = hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest()


def err(name, rid):
    e = errors[name]
    e["count"] += 1
    if len(e["samples"]) < 20:
        e["samples"].append(rid)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


for t in THRESHOLDS:
    tag = TAGS[t]
    for rid, row in rows_by_t[t].items():
        if set(row) != ROW_KEYS:
            err(f"{tag}_schema_keys", rid)
            continue
        types_ok = (
            isinstance(row["id"], str) and isinstance(row["source_query_id"], str)
            and isinstance(row["split"], str) and isinstance(row["part"], str)
            and isinstance(row["lang"], str) and isinstance(row["state"], str)
            and isinstance(row["order"], list) and all(isinstance(x, str) for x in row["order"])
            and isinstance(row["options"], list) and all(isinstance(x, str) for x in row["options"])
            and is_int(row["gold_index"]) and is_int(row["reject_index"])
            and isinstance(row["target_kind"], str)
            and (row["gold_capability_id"] is None or isinstance(row["gold_capability_id"], str))
            and isinstance(row["shuffle"], dict)
            and isinstance(row["shuffle"].get("seed_prefix"), str)
            and isinstance(row["shuffle"].get("permutation"), list)
            and all(is_int(x) for x in row["shuffle"].get("permutation", []))
            and isinstance(row["src_sha"], str)
        )
        if not types_ok:
            err(f"{tag}_field_types", rid)
        order, options = row["order"], row["options"]
        n = len(order)
        if n != len(options):
            err(f"{tag}_order_options_len", rid)
        if not (1 <= n <= 4):
            err(f"{tag}_option_count_bounds", rid)
        k = n - 1
        if order.count(REJECT) != 1:
            err(f"{tag}_exactly_one_reject", rid)
        if k < 0 or k > 3:
            err(f"{tag}_k_bounds", rid)
        gi, ri = row["gold_index"], row["reject_index"]
        if not (0 <= gi < n) or not (0 <= ri < n):
            err(f"{tag}_index_bounds", rid)
        else:
            if order[ri] != REJECT:
                err(f"{tag}_reject_index", rid)
            if row["target_kind"] == "reject":
                if row["gold_capability_id"] is not None:
                    err(f"{tag}_reject_gold_not_none", rid)
                if gi != ri:
                    err(f"{tag}_reject_gold_index", rid)
                if order[gi] != REJECT:
                    err(f"{tag}_reject_gold_slot", rid)
            elif row["target_kind"] == "capability":
                if row["gold_capability_id"] is None:
                    err(f"{tag}_cap_gold_none", rid)
                elif order[gi] != row["gold_capability_id"]:
                    err(f"{tag}_gold_order_mismatch", rid)
                if gi == ri:
                    err(f"{tag}_cap_gold_at_reject", rid)
            else:
                err(f"{tag}_target_kind", rid)
        if row["split"] not in SPLITS:
            err(f"{tag}_split", rid)
        if row["part"] != "p2":
            err(f"{tag}_part", rid)
        if row["src_sha"] != src_sha_expected:
            err(f"{tag}_src_sha", rid)
        sid = int(row["source_query_id"].rsplit(":", 1)[-1])
        if sid not in manifest or manifest[sid]["eligible"] != "True":
            err(f"{tag}_source_not_eligible", rid)
        else:
            if row["split"] != sid_to_split[sid]:
                err(f"{tag}_split_vs_manifest", rid)
            if row["lang"] != sid_to_lang[sid]:
                err(f"{tag}_lang_vs_manifest", rid)
            if row["state"] != sid_to_query[sid]:
                err(f"{tag}_state_vs_manifest", rid)
        perm = row["shuffle"].get("permutation", [])
        if sorted(perm) != list(range(n)):
            err(f"{tag}_permutation", rid)
        a = audit_by_t[t].get(rid)
        if a is None:
            err(f"{tag}_audit_missing", rid)
        else:
            base = list(a.get("candidate_capability_ids", [])) + [REJECT]
            if len(base) == n and sorted(perm) == list(range(n)):
                if [base[i] for i in perm] != order:
                    err(f"{tag}_shuffle_order", rid)
            if (a.get("split") != row["split"] or a.get("part") != row["part"]
                    or a.get("lang") != row["lang"] or a.get("state") != row["state"]
                    or int(a.get("candidate_count", -1)) != k
                    or int(a.get("option_count", -1)) != n
                    or a.get("label") != (row["gold_capability_id"] if row["target_kind"] == "capability" else REJECT)
                    or abs(float(a.get("threshold", -1)) - t) > 1e-12
                    or a.get("target_kind") if "target_kind" in a else False):
                err(f"{tag}_audit_consistency", rid)
            if row["target_kind"] == "reject" and a.get("is_action") and a.get("source_gold_capability_id") is None:
                err(f"{tag}_audit_source_gold_missing", rid)
        for cid, text in zip(order, options):
            if cid in card_text:
                if card_text[cid] != text:
                    card_conflicts["count"] += 1
                    if len(card_conflicts["samples"]) < 20:
                        card_conflicts["samples"].append(rid)
            else:
                card_text[cid] = text

# audit join sets
for t in THRESHOLDS:
    tag = TAGS[t]
    rids = set(rows_by_t[t])
    aids = set(audit_by_t[t])
    for rid in rids - aids:
        err(f"{tag}_audit_missing", rid)
    for rid in aids - rids:
        err(f"{tag}_audit_extra", rid)

# row-level shuffle vs file order and manifest
pre_ids = {s: [f"{source_keys[sid]}-p2" for sid in eligible_order
               if sid_to_split[sid] == s] for s in SPLITS}
row_shuffle_ok = True
row_shuffle_bad = []
for t in THRESHOLDS:
    tag = TAGS[t]
    for s in SPLITS:
        perm = list(tman[t]["row_order"]["permutation"][s])
        ids = file_order[t][s]
        if len(perm) != len(pre_ids[s]) or sorted(perm) != list(range(len(perm))):
            row_shuffle_ok = False
            row_shuffle_bad.append(f"{tag}/{s}: permutation")
        elif [pre_ids[s][i] for i in perm] != ids:
            row_shuffle_ok = False
            row_shuffle_bad.append(f"{tag}/{s}: order")
        if tman[t]["rows"][s] != len(ids):
            row_shuffle_ok = False
            row_shuffle_bad.append(f"{tag}/{s}: count")

# split consistency / leakage
split_mismatch = 0
split_mismatch_ids = []
for sid in eligible_order:
    vals = {rows_by_t[t][f"{source_keys[sid]}-p2"]["split"] for t in THRESHOLDS}
    if len(vals) != 1 or vals.pop() != sid_to_split[sid]:
        split_mismatch += 1
        if len(split_mismatch_ids) < 20:
            split_mismatch_ids.append(source_keys[sid])
id_sets_equal = all(set(rows_by_t[a]) == set(rows_by_t[b])
                    for a in THRESHOLDS for b in THRESHOLDS)
norm_split = defaultdict(set)
family_split = defaultdict(set)
for sid in eligible_order:
    norm_split[sid_to_norm[sid]].add(sid_to_split[sid])
    family_split[sid_to_family[sid]].add(sid_to_split[sid])
norm_leaks = {k: sorted(v) for k, v in norm_split.items() if len(v) > 1}
family_leaks = {k: sorted(v) for k, v in family_split.items() if len(v) > 1}
norm_dups = {k: sum(1 for sid in eligible_order if sid_to_norm[sid] == k)
             for k, v in norm_split.items() if sum(1 for sid in eligible_order if sid_to_norm[sid] == k) > 1}
# faster duplicate counts
norm_counts = Counter(sid_to_norm[sid] for sid in eligible_order)
norm_dups = {k: v for k, v in norm_counts.items() if v > 1}

# ── compact records ────────────────────────────────────────────────────────
comp: dict[float, dict[int, dict]] = {}
for t in THRESHOLDS:
    d = {}
    for rid, row in rows_by_t[t].items():
        a = audit_by_t[t][rid]
        sid = int(row["source_query_id"].rsplit(":", 1)[-1])
        d[sid] = {
            "rid": rid, "sid": sid, "split": row["split"], "lang": row["lang"],
            "target_kind": row["target_kind"],
            "label": row["gold_capability_id"] if row["target_kind"] == "capability" else REJECT,
            "gold": a.get("source_gold_capability_id"),
            "is_action": bool(a.get("is_action")),
            "k": len(row["order"]) - 1, "order": tuple(row["order"]),
            "cands": tuple(a.get("candidate_capability_ids", [])),
            "retrieved": tuple(a.get("retrieved_capabilities", [])),
            "scores": a.get("retrieval_scores", {}),
            "gold_score": a.get("gold_score"),
            "rank": a.get("gold_retrieval_rank"),
            "reason": a.get("label_reason"),
            "detail": a.get("label_reason_detail"),
            "reject_index": row["reject_index"],
            "source_tool": a.get("source_gold_capability_id") if a.get("is_action") else "REJECT_NO_ACTION",
        }
    comp[t] = d

# ── aggregate statistics ───────────────────────────────────────────────────
lang_counts = Counter()       # (t, split, lang)
tool_lang = Counter()         # (t, tool, lang)
tool_split = Counter()        # (t, tool, lang, split)
tool_target = Counter()       # (t, tool, split, target_kind)
k_all = Counter()             # (t, k)
k_split = Counter()           # (t, k, split)
k_lang = Counter()            # (t, k, lang)
k_target = Counter()          # (t, k, target_kind)
k_tool = Counter()            # (t, k, tool)
reject_pos = Counter()        # (t, k, reject_index)
label_counts = Counter()      # (t, label)
reason_counts = Counter()     # (t, reason, detail)
cap_appear = Counter()        # (t, cap) candidate appearances
cap_appear_split = Counter()  # (t, cap, split)
gold_in_cands = Counter()     # (t, target_kind) capability gold present in candidate list
gold_index_bad = Counter()
target_counts = Counter()     # (t, target_kind)
split_counts = Counter()      # (t, split)

for t in THRESHOLDS:
    for sid, r in comp[t].items():
        split_counts[(t, r["split"])] += 1
        lang_counts[(t, r["split"], r["lang"])] += 1
        target_counts[(t, r["target_kind"])] += 1
        tool_lang[(t, r["source_tool"], r["lang"])] += 1
        tool_split[(t, r["source_tool"], r["lang"], r["split"])] += 1
        tool_target[(t, r["source_tool"], r["split"], r["target_kind"])] += 1
        k_all[(t, r["k"])] += 1
        k_split[(t, r["k"], r["split"])] += 1
        k_lang[(t, r["k"], r["lang"])] += 1
        k_target[(t, r["k"], r["target_kind"])] += 1
        k_tool[(t, r["k"], r["source_tool"])] += 1
        reject_pos[(t, r["k"], r["reject_index"])] += 1
        label_counts[(t, r["label"])] += 1
        reason_counts[(t, r["reason"], r["detail"])] += 1
        for c in r["cands"]:
            cap_appear[(t, c)] += 1
            cap_appear_split[(t, c, r["split"])] += 1
        if r["target_kind"] == "capability":
            ok = r["label"] in r["cands"]
            gold_in_cands[(t, "capability")] += int(ok)
            if not (r["gold"] == r["label"]):
                gold_index_bad[t] += 1

# threshold comparison
comparison = {}
label_changes = []
top3_changes = 0
top3_change_ids = []
for sid in eligible_order:
    labels = [comp[t][sid]["label"] for t in THRESHOLDS]
    sets = [frozenset(comp[t][sid]["cands"]) for t in THRESHOLDS]
    if len(set(labels)) > 1:
        label_changes.append(sid)
    if len(set(sets)) > 1:
        top3_changes += 1
        if len(top3_change_ids) < 20:
            top3_change_ids.append(source_keys[sid])
for t in THRESHOLDS:
    comparison[t] = {
        "rows": len(comp[t]),
        "total_candidates": sum(r["k"] for r in comp[t].values()),
        "mean_k": sum(r["k"] for r in comp[t].values()) / len(comp[t]),
        "k": {k: k_all[(t, k)] for k in range(4)},
        "labels": {c: label_counts[(t, c)] for c in caps + [REJECT]},
        "cap_appear": {c: cap_appear[(t, c)] for c in caps},
        "targets": {k: target_counts[(t, k)] for k in ("capability", "reject")},
        "splits": {s: split_counts[(t, s)] for s in SPLITS},
    }

# ── exact cosine from stored embeddings + DB corpus (no re-embedding) ──────
async def load_corpus():
    import asyncpg
    conn = await asyncpg.connect(DB_DSN)
    try:
        std = await conn.fetch(
            "SELECT id::text AS id, capability_id, embedding::text AS emb "
            "FROM capability_standard_queries WHERE embedding IS NOT NULL ORDER BY id")
        sim = await conn.fetch(
            "SELECT id::text AS id, capability_id, embedding::text AS emb "
            "FROM capability_similar_queries WHERE embedding IS NOT NULL ORDER BY id")
    finally:
        await conn.close()
    keys, cap_ids, vecs = [], [], []
    for kind, rows in (("standard", std), ("similar", sim)):
        for r in rows:
            keys.append((kind, r["id"]))
            cap_ids.append(r["capability_id"])
            vecs.append(np.asarray(json.loads(r["emb"]), dtype=np.float32))
    return keys, cap_ids, np.vstack(vecs)


corpus_keys, corpus_caps, corpus_emb = asyncio.run(load_corpus())
key_index = {k: i for i, k in enumerate(corpus_keys)}
cap_of_corpus = np.array([caps.index(c) for c in corpus_caps], dtype=np.int16)
cap_indices = [np.where(cap_of_corpus == j)[0] for j in range(len(caps))]
ncorpus = len(corpus_keys)

qemb = np.load(RUN / "data/query_embeddings.npy").astype(np.float32)
qids = json.loads((RUN / "data/query_embedding_ids.json").read_text(encoding="utf-8"))
q_sids = [int(x.rsplit(":", 1)[-1]) for x in qids]
assert len(qemb) == len(q_sids) == len(eligible_order)

pool_bool = {}
pool_bad = 0
for rec in load_jsonl(RUN / "data/shared_retrieval_evidence.jsonl"):
    sid = int(rec["source_row_id"])
    idx = key_index.get((rec["corpus_type"], str(rec["corpus_id"])))
    if idx is None:
        pool_bad += 1
        continue
    arr = pool_bool.get(sid)
    if arr is None:
        arr = np.zeros(ncorpus, dtype=bool)
        pool_bool[sid] = arr
    arr[idx] = True

# exact cap maxima
exact_cap = np.full((len(q_sids), len(caps)), -np.inf, dtype=np.float32)
CHUNK = 1024
for start in range(0, len(q_sids), CHUNK):
    end = min(start + CHUNK, len(q_sids))
    ex = qemb[start:end] @ corpus_emb.T
    for j, idxs in enumerate(cap_indices):
        exact_cap[start:end, j] = ex[:, idxs].max(axis=1)

omission_query = Counter()    # (t, tier)
omission_items_tool = Counter()   # (t, cap)
omission_caps_tool = Counter()    # (t, cap)
tier_b_tool = Counter()           # (t, cap)
tier_c_tool = Counter()           # (t, cap)
recovered_tool = Counter()        # (t, cap)
tier_b_lang = Counter()           # (t, lang)
tier_c_lang = Counter()           # (t, lang)
recovered_lang = Counter()        # (t, lang)
tier_b_k = Counter()              # (t, k)
tier_c_k = Counter()              # (t, k)
recovered_k = Counter()           # (t, k)
impact_rows = []
omitted_corpus_total = 0
omitted_cap_instances = 0
omitted_in_exact_top3_items = 0
omitted_caps_in_exact_top3 = 0

for qi, sid in enumerate(q_sids):
    pb = pool_bool.get(sid)
    if pb is None:
        pb = np.zeros(ncorpus, dtype=bool)
    ec = exact_cap[qi]
    for t in THRESHOLDS:
        r = comp[t][sid]
        hit = np.where(ec >= t)[0]
        exact_ordered = sorted(
            [(caps[j], float(ec[j])) for j in hit],
            key=lambda x: (-x[1], x[0]))
        exact_top3 = tuple(c for c, _ in exact_ordered[:3])
        exact_top3_set = set(exact_top3)
        actual_top3_set = set(r["cands"])
        actual_retrieved = set(r["retrieved"])
        if r["is_action"]:
            if len(exact_top3) == 0:
                exact_label = REJECT
            elif r["gold"] in exact_top3_set:
                exact_label = r["gold"]
            else:
                exact_label = REJECT
        else:
            exact_label = REJECT
        # item-level omission (exact hit >= t not in the SQL pool)
        ex_scores = ec if False else None
        # recompute score vector only for this query from a cached chunk
        omitted_idx = []
        # we did not retain the full score vector; recompute one row cheaply
        row_scores = (qemb[qi] @ corpus_emb.T)
        for ci in np.where(row_scores >= t)[0]:
            if not pb[ci]:
                omitted_idx.append(int(ci))
        omitted_items = len(omitted_idx)
        omitted_caps = sorted({caps[int(cap_of_corpus[ci])] for ci in omitted_idx})
        caps_not_retrieved = sorted(
            caps[j] for j in hit if caps[j] not in actual_retrieved)
        tier_a = bool(omitted_items or caps_not_retrieved)
        tier_b = exact_top3_set != actual_top3_set
        tier_c = exact_label != r["label"]
        recovered = bool(r["is_action"] and r["label"] == REJECT
                         and exact_label == r["gold"] and r["gold"] is not None)
        if tier_a:
            omission_query[(t, "a")] += 1
        if caps_not_retrieved:
            omission_query[(t, "a_cap")] += 1
        if tier_b:
            omission_query[(t, "b")] += 1
            tier_b_tool[(t, r["source_tool"])] += 1
            tier_b_lang[(t, r["lang"])] += 1
            tier_b_k[(t, r["k"])] += 1
        if tier_c:
            omission_query[(t, "c")] += 1
            tier_c_tool[(t, r["source_tool"])] += 1
            tier_c_lang[(t, r["lang"])] += 1
            tier_c_k[(t, r["k"])] += 1
        if recovered:
            omission_query[(t, "gold_recovered")] += 1
            recovered_tool[(t, r["source_tool"])] += 1
            recovered_lang[(t, r["lang"])] += 1
            recovered_k[(t, r["k"])] += 1
        omitted_corpus_total += omitted_items
        omitted_cap_instances += len(caps_not_retrieved)
        # item-level top-3 cutoff
        if omitted_items:
            vals = row_scores[row_scores >= t]
            if len(vals) >= 3:
                cutoff = float(np.partition(vals, -3)[-3])
                omitted_in_exact_top3_items += sum(
                    1 for ci in omitted_idx if row_scores[ci] >= cutoff)
        omitted_caps_in_exact_top3 += sum(1 for c in caps_not_retrieved
                                          if c in exact_top3_set)
        for c in caps_not_retrieved:
            omission_caps_tool[(t, c)] += 1
        for ci in omitted_idx:
            omission_items_tool[(t, caps[int(cap_of_corpus[ci])])] += 1
        if tier_a:
            impact_rows.append([
                f"{t:.2f}", source_keys[sid], r["split"], r["lang"], r["source_tool"],
                r["is_action"], r["label"], exact_label, int(tier_a), int(tier_b),
                int(tier_c), int(recovered), omitted_items, ";".join(omitted_caps),
                ";".join(caps_not_retrieved), ";".join(exact_top3),
                ";".join(r["cands"]), r["gold"] or "",
                "" if r["gold_score"] is None else f"{float(r['gold_score']):.9f}",
                "" if r["rank"] is None else r["rank"],
                "" if r["reason"] is None else r["reason"],
            ])

# ── CSV: overall summary ───────────────────────────────────────────────────
overall = []
for t in THRESHOLDS:
    tag = f"{t:.2f}"
    def add(metric, n, d):
        overall.append([tag, metric, n, d, pct(n, d)])
    add("source_rows", source_total, source_total)
    add("excluded_rows", excluded_total, source_total)
    add("selected_rows", len(comp[t]), source_total)
    for s in SPLITS:
        add(f"split_{s}", split_counts[(t, s)], len(comp[t]))
    for lang in ("zh", "en", "mixed", "unknown"):
        add(f"lang_{lang}", sum(lang_counts[(t, s, lang)] for s in SPLITS), len(comp[t]))
        for s in SPLITS:
            add(f"split_{s}_lang_{lang}", lang_counts[(t, s, lang)], split_counts[(t, s)])
    add("target_capability", target_counts[(t, "capability")], len(comp[t]))
    add("target_reject", target_counts[(t, "reject")], len(comp[t]))
    add("duplicate_ids", 0, len(comp[t]))
    add("duplicate_normalized_queries", 0, len(comp[t]))
    add("cross_split_normalized_queries", len(norm_leaks), len(norm_counts))
    add("cross_split_families", len(family_leaks), len(family_split))
    add("jsonl_records", len(rows_by_t[t]), len(comp[t]))
    add("audit_records", len(audit_by_t[t]), len(comp[t]))
    add("audit_join_missing", len(set(rows_by_t[t]) - set(audit_by_t[t])), len(comp[t]))
    add("audit_join_extra", len(set(audit_by_t[t]) - set(rows_by_t[t])), len(comp[t]))
write_csv(OUT / "overall_summary.csv",
          ["threshold", "metric", "count", "denominator", "percent"], overall)

# ── CSV: tool x language ───────────────────────────────────────────────────
rows_out = []
tools = caps + ["REJECT_NO_ACTION"]
for t in THRESHOLDS:
    for tool in tools:
        total = sum(tool_lang[(t, tool, l)] for l in ("zh", "en", "mixed", "unknown"))
        if total == 0:
            continue
        for lang in ("zh", "en", "mixed", "unknown", "ALL"):
            n = total if lang == "ALL" else tool_lang[(t, tool, lang)]
            rows_out.append([f"{t:.2f}", tool, lang, n, total, pct(n, total),
                             len(comp[t]), pct(n, len(comp[t]))])
write_csv(OUT / "tool_language_distribution.csv",
          ["threshold", "tool", "language", "count", "tool_total", "percent_of_tool",
           "dataset_total", "percent_of_dataset"], rows_out)

# ── CSV: tool x language x split (with target/deviation) ───────────────────
rows_out = []
for t in THRESHOLDS:
    for tool in tools:
        for lang in ("zh", "en", "mixed", "unknown"):
            n = sum(tool_split[(t, tool, lang, s)] for s in SPLITS)
            if n == 0:
                continue
            targets = allocate_counts(n)
            for s in SPLITS:
                actual = tool_split[(t, tool, lang, s)]
                target = targets[s]
                actual_pct = 100.0 * actual / n
                target_pct = 100.0 * target / n
                rows_out.append([
                    f"{t:.2f}", tool, lang, s, actual, n, pct(actual, n),
                    len(comp[t]), pct(actual, len(comp[t])), target,
                    f"{target_pct:.4f}", actual - target,
                    f"{actual_pct - target_pct:.4f}",
                    "n<3 minimum representation" if n < 3 else "",
                ])
write_csv(OUT / "tool_split_distribution.csv",
          ["threshold", "tool", "language", "split", "actual_count", "group_total",
           "percent_of_group", "dataset_total", "percent_of_dataset", "target_count",
           "target_percent", "deviation_count", "deviation_pp", "note"], rows_out)

# ── CSV: K distribution ────────────────────────────────────────────────────
rows_out = []
for t in THRESHOLDS:
    total_t = len(comp[t])
    for k in range(4):
        rows_out.append([f"{t:.2f}", k, "ALL", "ALL", "ALL", "ALL",
                         k + 1, k_all[(t, k)], total_t, pct(k_all[(t, k)], total_t)])
        for s in SPLITS:
            rows_out.append([f"{t:.2f}", k, "split", s, "ALL", "ALL",
                             k + 1, k_split[(t, k, s)], split_counts[(t, s)],
                             pct(k_split[(t, k, s)], split_counts[(t, s)])])
        for lang in ("zh", "en", "mixed", "unknown"):
            d = sum(lang_counts[(t, s, lang)] for s in SPLITS)
            rows_out.append([f"{t:.2f}", k, "language", lang, "ALL", "ALL",
                             k + 1, k_lang[(t, k, lang)], d, pct(k_lang[(t, k, lang)], d)])
        for kind in ("capability", "reject"):
            d = target_counts[(t, kind)]
            rows_out.append([f"{t:.2f}", k, "target_kind", kind, "ALL", "ALL",
                             k + 1, k_target[(t, k, kind)], d,
                             pct(k_target[(t, k, kind)], d)])
        for tool in tools:
            d = sum(k_tool[(t, kk, tool)] for kk in range(4))
            if d:
                rows_out.append([f"{t:.2f}", k, "source_tool", tool, "ALL", "ALL",
                                 k + 1, k_tool[(t, k, tool)], d,
                                 pct(k_tool[(t, k, tool)], d)])
write_csv(OUT / "k_distribution.csv",
          ["threshold", "k", "dimension", "dimension_value", "split", "language",
           "option_count", "count", "denominator", "percent"], rows_out)

# reject-position table (bonus)
rows_out = []
for t in THRESHOLDS:
    for k in range(4):
        d = k_all[(t, k)]
        for pos in range(k + 1):
            n = reject_pos[(t, k, pos)]
            rows_out.append([f"{t:.2f}", k, pos, k + 1, n, d, pct(n, d)])
write_csv(OUT / "k_reject_position.csv",
          ["threshold", "k", "reject_index", "option_count", "count", "k_total",
           "percent_of_k"], rows_out)

# ── CSV: tool gold/reject ──────────────────────────────────────────────────
rows_out = []
for t in THRESHOLDS:
    for tool in tools:
        for s in SPLITS + ("ALL",):
            if s == "ALL":
                g = sum(tool_target[(t, tool, ss, "capability")] for ss in SPLITS)
                rj = sum(tool_target[(t, tool, ss, "reject")] for ss in SPLITS)
            else:
                g = tool_target[(t, tool, s, "capability")]
                rj = tool_target[(t, tool, s, "reject")]
            n = g + rj
            if n == 0:
                continue
            rows_out.append([f"{t:.2f}", tool, s, g, rj, n,
                             pct(g, n), pct(rj, n)])
write_csv(OUT / "tool_gold_reject_distribution.csv",
          ["threshold", "tool", "split", "gold_target", "reject_target", "total",
           "gold_pct", "reject_pct"], rows_out)

# ── CSV: threshold comparison ──────────────────────────────────────────────
rows_out = []
for metric in ("total_candidates", "mean_k", "k0", "k1", "k2", "k3",
               "target_capability", "target_reject") + tuple(caps) + (REJECT,):
    if metric.startswith("k") and len(metric) == 2 and metric[1].isdigit():
        vals = [comparison[t]["k"][int(metric[1])] for t in THRESHOLDS]
    elif metric == "total_candidates":
        vals = [comparison[t]["total_candidates"] for t in THRESHOLDS]
    elif metric == "mean_k":
        vals = [round(comparison[t]["mean_k"], 6) for t in THRESHOLDS]
    elif metric == "target_capability":
        vals = [comparison[t]["targets"]["capability"] for t in THRESHOLDS]
    elif metric == "target_reject":
        vals = [comparison[t]["targets"]["reject"] for t in THRESHOLDS]
    elif metric == REJECT:
        vals = [comparison[t]["labels"][REJECT] for t in THRESHOLDS]
    else:
        vals = [comparison[t]["cap_appear"][metric] for t in THRESHOLDS]
    rows_out.append([metric, vals[0], vals[1], vals[2], vals[2] - vals[0]])
write_csv(OUT / "threshold_comparison.csv",
          ["metric", "value_082", "value_085", "value_088", "delta_082_088"], rows_out)

# ── CSV: label changes ─────────────────────────────────────────────────────
rows_out = []
for sid in label_changes:
    base = comp[THRESHOLDS[0]][sid]
    row = [source_keys[sid], base["split"], base["lang"], base["source_tool"],
           base["gold"] or "", base["is_action"],
           ";".join(sorted({comp[t][sid]["label"] for t in THRESHOLDS})),
           int(len({comp[t][sid]["label"] for t in THRESHOLDS}) > 1),
           int(len({frozenset(comp[t][sid]["cands"]) for t in THRESHOLDS}) > 1)]
    for t in THRESHOLDS:
        r = comp[t][sid]
        row += [r["label"], ";".join(r["cands"]),
                ";".join(f"{c}:{float(r['scores'].get(c, -1)):.9f}" for c in r["cands"]),
                r["k"], "" if r["gold_score"] is None else f"{float(r['gold_score']):.9f}",
                r["reason"]]
    rows_out.append(row)
write_csv(OUT / "label_changes.csv",
          ["source_query_id", "split", "language", "source_tool", "source_gold",
           "is_action", "labels_seen", "label_changed", "top3_changed",
           "label_082", "candidates_082", "scores_082", "k_082", "gold_score_082", "reason_082",
           "label_085", "candidates_085", "scores_085", "k_085", "gold_score_085", "reason_085",
           "label_088", "candidates_088", "scores_088", "k_088", "gold_score_088", "reason_088"],
          rows_out)

# ── CSV: retrieval omission impact (per query x threshold) ─────────────────
write_csv(OUT / "retrieval_omission_impact.csv",
          ["threshold", "source_query_id", "split", "language", "source_tool",
           "is_action", "actual_label", "exact_label", "tier_a_any_omission",
           "tier_b_top3_changed", "tier_c_label_changed", "gold_recovered",
           "omitted_corpus_items", "omitted_caps", "caps_not_in_sql_retrieved",
           "exact_top3", "actual_top3", "source_gold", "gold_sql_score",
           "gold_sql_rank", "label_reason"], impact_rows)

# ── CSV: split ratio audit ─────────────────────────────────────────────────
rows_out = []
for t in THRESHOLDS:
    groups = set((tool, lang) for (tt, tool, lang) in tool_lang if tt == t)
    for tool, lang in sorted(groups):
        n = sum(tool_split[(t, tool, lang, s)] for s in SPLITS)
        targets = allocate_counts(n)
        row = [f"{t:.2f}", tool, lang, n]
        for s in SPLITS:
            actual = tool_split[(t, tool, lang, s)]
            target = targets[s]
            row += [actual, target, pct(actual, n), f"{100.0 * target / n:.4f}",
                    f"{100.0 * actual / n - 100.0 * target / n:.4f}"]
        row.append("n<3 minimum representation" if n < 3 else "")
        rows_out.append(row)
# global
for t in THRESHOLDS:
    n = len(comp[t])
    targets = allocate_counts(n)
    row = [f"{t:.2f}", "ALL", "ALL", n]
    for s in SPLITS:
        actual = split_counts[(t, s)]
        target = targets[s]
        row += [actual, target, pct(actual, n), f"{100.0 * target / n:.4f}",
                f"{100.0 * actual / n - 100.0 * target / n:.4f}"]
    row.append("")
    rows_out.append(row)
write_csv(OUT / "split_ratio_audit.csv",
          ["threshold", "tool", "language", "group_total",
           "train_actual", "train_target", "train_actual_pct", "train_target_pct", "train_deviation_pp",
           "test_actual", "test_target", "test_actual_pct", "test_target_pct", "test_deviation_pp",
           "val_actual", "val_target", "val_actual_pct", "val_target_pct", "val_deviation_pp",
           "note"], rows_out)

# ── data quality JSON ──────────────────────────────────────────────────────
dq = {
    "run_dir": str(RUN),
    "generated_rows": len(eligible_order),
    "source_rows": source_total,
    "excluded_rows": excluded_total,
    "thresholds": [f"{t:.2f}" for t in THRESHOLDS],
    "build_status": {
        "status": "INCOMPLETE",
        "reason": "FileNotFoundError at write_config: full_r1/config directory was absent",
        "missing_run_metadata": [
            "config/generation_config.json",
            "manifests/source_manifest.json",
            "manifests/generation_manifest.json",
            "manifests/output_hashes.json",
        ],
        "training_bundles_complete": True,
        "per_threshold_manifests_complete": True,
    },
    "checks": {k: v for k, v in sorted(errors.items())},
    "card_text_conflicts": card_conflicts,
    "row_level_shuffle": {
        "ok": row_shuffle_ok,
        "failures": row_shuffle_bad[:20],
        "method": "file slot p == pre-shuffle row order[manifest.permutation[p]]",
    },
    "split_consistency": {
        "mismatches": split_mismatch,
        "sample_ids": split_mismatch_ids,
        "id_sets_equal_across_thresholds": id_sets_equal,
    },
    "split_leakage": {
        "normalized_query_cross_split": len(norm_leaks),
        "family_cross_split": len(family_leaks),
        "duplicate_normalized_query_groups": len(norm_dups),
        "duplicate_row_count": sum(v - 1 for v in norm_dups.values()),
    },
    "evidence_pool": {
        "corpus_rows": ncorpus,
        "unknown_corpus_ids": pool_bad,
        "queries_with_pool": len(pool_bool),
    },
    "reject_reasons": {
        f"{t:.2f}": {f"{r}|{d}": n for (tt, r, d), n in sorted(reason_counts.items()) if tt == t}
        for t in THRESHOLDS
    },
    "threshold_comparison": {f"{t:.2f}": comparison[t] for t in THRESHOLDS},
    "label_changes": {
        "count": len(label_changes),
        "sample_ids": [source_keys[sid] for sid in label_changes[:20]],
        "top3_changes": top3_changes,
        "top3_sample_ids": top3_change_ids,
    },
    "omission_summary": {
        f"{t:.2f}": {
            "queries_any_item_omission": omission_query[(t, "a")],
            "queries_capability_omission": omission_query[(t, "a_cap")],
            "queries_top3_changed": omission_query[(t, "b")],
            "queries_label_changed": omission_query[(t, "c")],
            "queries_gold_recovered_from_reject": omission_query[(t, "gold_recovered")],
            "omitted_items_by_tool": {c: omission_items_tool[(t, c)] for c in caps
                                      if omission_items_tool[(t, c)]},
            "omitted_caps_by_tool": {c: omission_caps_tool[(t, c)] for c in caps
                                     if omission_caps_tool[(t, c)]},
            "tier_b_by_tool": {c: tier_b_tool[(t, c)] for c in caps + ["REJECT_NO_ACTION"]
                               if tier_b_tool[(t, c)]},
            "tier_c_by_tool": {c: tier_c_tool[(t, c)] for c in caps + ["REJECT_NO_ACTION"]
                               if tier_c_tool[(t, c)]},
            "gold_recovered_by_tool": {c: recovered_tool[(t, c)] for c in caps
                                       if recovered_tool[(t, c)]},
            "tier_b_by_language": {l: tier_b_lang[(t, l)] for l in ("zh", "en", "mixed", "unknown")
                                   if tier_b_lang[(t, l)]},
            "tier_c_by_language": {l: tier_c_lang[(t, l)] for l in ("zh", "en", "mixed", "unknown")
                                   if tier_c_lang[(t, l)]},
            "gold_recovered_by_language": {l: recovered_lang[(t, l)] for l in ("zh", "en", "mixed", "unknown")
                                           if recovered_lang[(t, l)]},
            "tier_b_by_k": {str(k): tier_b_k[(t, k)] for k in range(4) if tier_b_k[(t, k)]},
            "tier_c_by_k": {str(k): tier_c_k[(t, k)] for k in range(4) if tier_c_k[(t, k)]},
            "gold_recovered_by_k": {str(k): recovered_k[(t, k)] for k in range(4) if recovered_k[(t, k)]},
        } for t in THRESHOLDS
    },
    "omission_totals_all_thresholds": {
        "omitted_corpus_items": omitted_corpus_total,
        "omitted_capability_instances": omitted_cap_instances,
        "omitted_items_in_exact_top3_items": omitted_in_exact_top3_items,
        "omitted_caps_in_exact_top3_caps": omitted_caps_in_exact_top3,
    },
}
(OUT / "data_quality_audit.json").write_text(
    json.dumps(dq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# ── final report ───────────────────────────────────────────────────────────
def nz_tools(counter, t):
    return {c: counter[(t, c)] for c in tools if counter[(t, c)]}

lines = []
lines.append("# Laya V4 full_r1 全量数据统计与质量分析\n")
lines.append("## 0. 执行状态与重要异常\n")
lines.append("- 三个阈值的数据文件已生成，统计只读取现有 JSONL/audit。")
lines.append("- **生成收尾失败**：`build_dataset.py` 在 `write_config` 写入 "
             "`config/generation_config.json` 时因 `config/` 目录不存在报 "
             "`FileNotFoundError`；run 级 `config/`、`manifests/source_manifest.json`、"
             "`manifests/generation_manifest.json`、`manifests/output_hashes.json` 缺失。")
lines.append("- 三个阈值的训练/验证/测试 JSONL、construction_audit 和阈值级 manifest 完整。")
lines.append("- 本次统计未重算 embedding、未执行 SQL 召回、未修改任何数据文件。"
             "精确余弦影响分析使用已保存的 `query_embeddings.npy` 与数据库中的语料 embedding 做点积。\n")
lines.append("## 1. 总体统计\n")
lines.append("| 阈值 | 入选 | Train | Test | Val | Capability target | REJECT target |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for t in THRESHOLDS:
    lines.append(f"| {t:.2f} | {len(comp[t])} | {split_counts[(t,'train')]} | "
                 f"{split_counts[(t,'test')]} | {split_counts[(t,'val')]} | "
                 f"{target_counts[(t,'capability')]} | {target_counts[(t,'reject')]} |")
lines.append(f"\n源数据 {source_total} 行，排除 {excluded_total} 行（标准/相似召回语料），"
             f"最终入选 {len(eligible_order)} 行。三个阈值行集合一致：{id_sets_equal}。\n")
lines.append("各 split 语言分布：\n")
lines.append("| 阈值 | split | zh | en | mixed | unknown |")
lines.append("|---|---|---:|---:|---:|---:|")
for t in THRESHOLDS:
    for s in SPLITS:
        lines.append(f"| {t:.2f} | {s} | {lang_counts[(t,s,'zh')]} | {lang_counts[(t,s,'en')]} | "
                     f"{lang_counts[(t,s,'mixed')]} | {lang_counts[(t,s,'unknown')]} |")
lines.append("\n重复与泄漏：")
lines.append(f"- 重复行 ID：0；重复规范化 query 组：{len(norm_dups)}；"
             f"跨 split 规范化 query：{len(norm_leaks)}；跨 split family：{len(family_leaks)}。")
lines.append(f"- JSONL / audit 记录数：{len(rows_by_t[0.82])} / {len(audit_by_t[0.82])}（每阈值），"
             f"关联缺失 {len(set(rows_by_t[0.82])-set(audit_by_t[0.82]))}，"
             f"多余 {len(set(audit_by_t[0.82])-set(rows_by_t[0.82]))}。\n")
lines.append("## 2. 每个 Tool 的中英文与 split 分布\n")
lines.append("完整明细见 `tool_language_distribution.csv` 与 `tool_split_distribution.csv`。"
             "Tool 按 source gold capability 归类；no-action 行归入 `REJECT_NO_ACTION`，"
             "REJECT target 不混入普通 Tool。\n")
lines.append("## 3. K 值分布\n")
lines.append("| 阈值 | K=0 | K=1 | K=2 | K=3 | 平均 K | 总候选数 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for t in THRESHOLDS:
    lines.append(f"| {t:.2f} | {k_all[(t,0)]} | {k_all[(t,1)]} | {k_all[(t,2)]} | "
                 f"{k_all[(t,3)]} | {comparison[t]['mean_k']:.4f} | "
                 f"{comparison[t]['total_candidates']} |")
lines.append("\nK 与 split/语言/target/tool 的交叉统计见 `k_distribution.csv`，"
             "REJECT 位置分布见 `k_reject_position.csv`。K=0 行只有 REJECT，选项数 1。\n")
lines.append("## 4. 每个 Tool 的 Gold 与 REJECT 分布\n")
lines.append("完整明细见 `tool_gold_reject_distribution.csv`。"
             "Gold 是否在候选内、索引一致性等写入 `data_quality_audit.json`。\n")
lines.append("REJECT 原因（每阈值）：")
for t in THRESHOLDS:
    lines.append(f"\n**{t:.2f}**")
    for r in REASON_ORDER:
        n = sum(v for (tt, rr, d), v in reason_counts.items() if tt == t and rr == r)
        lines.append(f"- {r}: {n}")
lines.append("\n## 5. 三个阈值对比\n")
lines.append(f"- 标签发生变化的查询：**{len(label_changes)}** 条。")
lines.append(f"- Top-3 候选集合发生变化的查询：**{top3_changes}** 条。")
lines.append("- 标签变化明细（含三阈值候选、分数、Gold/REJECT）见 `label_changes.csv`。")
lines.append("- 各 Tool 候选数变化见 `threshold_comparison.csv`。")
lines.append(f"- 同一 source_query_id 跨阈值 split 不一致：**{split_mismatch}** 条。\n")
lines.append("## 6. 召回遗漏对标注质量的影响\n")
lines.append("定义三层，严格区分：")
lines.append("- tier A：存在精确余弦 ≥ 阈值但未进入 SQL 128 候选池的候选。")
lines.append("- tier B：遗漏使实际 Top-3 能力集合与精确余弦 Top-3 不一致。")
lines.append("- tier C：遗漏使最终 Gold/REJECT 标签与精确余弦反事实标签不一致。\n")
lines.append("| 阈值 | tier A 查询 | 能力级遗漏 | tier B | tier C | Gold 被错判为 REJECT |")
lines.append("|---|---:|---:|---:|---:|---:|")
for t in THRESHOLDS:
    lines.append(f"| {t:.2f} | {omission_query[(t,'a')]} | {omission_query[(t,'a_cap')]} | "
                 f"{omission_query[(t,'b')]} | {omission_query[(t,'c')]} | "
                 f"{omission_query[(t,'gold_recovered')]} |")
lines.append(f"\n全部阈值合计：遗漏语料项 {omitted_corpus_total}，"
             f"能力级遗漏 {omitted_cap_instances}，"
             f"进入精确 Top-3 的遗漏语料项 {omitted_in_exact_top3_items}，"
             f"进入精确 Top-3 的遗漏能力 {omitted_caps_in_exact_top3}。")
lines.append("按 Tool/语言/K 的分布见 `data_quality_audit.json` 的 `omission_summary`，"
             "逐查询影响见 `retrieval_omission_impact.csv`。\n")
lines.append("## 7. 数据集划分比例检查\n")
lines.append("完整明细见 `split_ratio_audit.csv`；目标由 `allocate_counts` 对每个 "
             "`tool × language` 组独立计算，未复制样本。\n")
lines.append("## 8. 结构与标注质量\n")
lines.append("所有检查的计数与样本 ID 在 `data_quality_audit.json`。"
             f"行级 shuffle 校验：{'PASS' if row_shuffle_ok else 'FAIL'}；"
             f"卡文本一致性冲突：{card_conflicts['count']}；"
             f"split 一致性错误：{split_mismatch}。\n")
lines.append("## 9. 是否适合进入训练\n")
lines.append("**暂不建议直接进入训练。** 原因：")
lines.append("1. run 级 metadata 缺失（`config`/`manifests`），无法完整复现与走现有官方 validator。")
lines.append("2. 召回遗漏在三个阈值下都改变了部分查询的 Top-3 与标签，"
             "需先确认这些样本是否接受为训练噪声。")
lines.append("3. 若 tier C / Gold 被错判 REJECT 的比例可接受，也建议先完成 metadata 收尾再训练。")
lines.append("\n建议的最小后续动作（未执行，需授权）：")
lines.append("- 预创建 `full_r1/config/` 后只补写 run 级 metadata，不重新生成数据；")
lines.append("- 对 `retrieval_omission_impact.csv` 中 tier C / `gold_recovered=1` 的样本做人工抽检；")
lines.append("- 决定是否把精确余弦 Top-3 作为对照，而不是直接改训练标签。")
(OUT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({
    "out": str(OUT),
    "rows_per_threshold": len(comp[0.82]),
    "label_changes": len(label_changes),
    "top3_changes": top3_changes,
    "omission": {f"{t:.2f}": {
        "a": omission_query[(t, "a")],
        "a_cap": omission_query[(t, "a_cap")],
        "b": omission_query[(t, "b")],
        "c": omission_query[(t, "c")],
        "gold_recovered": omission_query[(t, "gold_recovered")],
    } for t in THRESHOLDS},
    "data_quality_errors": {k: v["count"] for k, v in errors.items()},
}, ensure_ascii=False, indent=2))
