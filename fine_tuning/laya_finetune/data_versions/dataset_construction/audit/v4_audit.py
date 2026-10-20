import csv, json, random, sys, unicodedata, re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np

REPO = Path(r"e:/Work Space/Git Repositories/DeepDive")
sys.path.insert(0, str(REPO / "packages"))
from core.application.chat.intent_funnel.cap_router.card_renderer import (
    REJECT_CARD, REJECT_LABEL, render_card)

RUN = REPO / "scripts/laya_finetune/data_v4_build/smoke50_r1"
D = RUN / "data"
TH = {"threshold_082": 0.82, "threshold_085": 0.85, "threshold_088": 0.88}
KEYS = {"id", "source_query_id", "split", "part", "lang", "state", "order",
        "options", "gold_index", "target_kind", "gold_capability_id",
        "reject_index", "shuffle", "src_sha"}
CAND_KEYS = {"id", "source_query_id", "split", "part", "lang", "lang_raw",
             "state", "source_gold_capability_id", "is_action", "label",
             "label_reason", "label_reason_detail", "threshold",
             "candidate_count", "option_count", "candidate_capability_ids",
             "retrieved_capabilities", "retrieval_scores", "gold_score",
             "gold_retrieval_rank", "decision_state", "parameters", "source",
             "review_status", "query_family", "query_family_id", "csv_line",
             "source_row_id", "source_file", "source_row"}


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(t or "")).strip().casefold())


A = {}


def err(bucket, detail):
    A.setdefault(bucket, []).append(detail)


sample = list(csv.DictReader(open(D / "sample_manifest.csv", encoding="utf-8-sig")))
by_key = {f'{r["source_file"]}:{r["source_row"]}:{r["source_row_id"]}': r for r in sample}
sel = [r for r in sample if r["smoke_selected"] == "True"]
sel_keys = [f'{r["source_file"]}:{r["source_row"]}:{r["source_row_id"]}' for r in sel]
sm = json.loads((RUN / "manifests" / "source_manifest.json").read_text(encoding="utf-8"))
order_keys = sm["selected_source_record_keys"]
if set(order_keys) != set(sel_keys):
    err("selected_set", "source_manifest != smoke_selected")
split_of = {k: by_key[k]["split"] for k in sel_keys}
cap_ids = json.loads((D / "capability_ids.json").read_text(encoding="utf-8"))
cap_idx = {c: i for i, c in enumerate(cap_ids)}
cap_matrix = np.load(D / "capability_max_scores.npy")
snap = json.loads((REPO / "scripts/laya_finetune/V2/data/aligned/v2_registry_snapshot.json").read_text(encoding="utf-8"))
cards = {cid: render_card(SimpleNamespace(
    capability_id=e["capability_id"], tool_binding=e["tool_binding"],
    description=e["description"], negatives=e["negatives"],
    parameters=e["parameters"])) for cid, e in snap.items()}
ev = defaultdict(list)
for line in (D / "shared_retrieval_evidence.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        h = json.loads(line)
        ev[h["source_query_id"]].append(h)
missing_ev = sorted(set(order_keys) - set(ev))
EVIDENCE_FLOOR = 0.80
if missing_ev:
    for k in missing_ev:
        qi = order_keys.index(k)
        top = float(np.max(cap_matrix[qi]))
        if top >= EVIDENCE_FLOOR:
            err("evidence_gap_above_floor", (k, top))

summary = {}
for tag, t in TH.items():
    tdir = D / tag
    m = json.loads((tdir / "manifest.json").read_text(encoding="utf-8"))
    audit_rows = [json.loads(l) for l in (tdir / "construction_audit.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    audit = {r["id"]: r for r in audit_rows}
    v2 = {}
    file_order = {}
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in (tdir / f"v2_{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        file_order[split] = [r["id"] for r in rows]
        for r in rows:
            if r["id"] in v2:
                err("duplicate_row_id", r["id"])
            v2[r["id"]] = r
    if set(v2) != set(audit):
        err("audit_join", (tag, len(v2), len(audit)))
    stats = {"K": Counter(), "reject_pos": Counter(), "kind": Counter(), "split": Counter()}
    for rid, r in v2.items():
        a = audit.get(rid)
        if a is None:
            continue
        key = r["source_query_id"]
        src = by_key.get(key)
        if src is None:
            err("source_trace", (tag, rid, key))
            continue
        stats["K"][len(r["order"]) - 1] += 1
        stats["reject_pos"][r["reject_index"]] += 1
        stats["kind"][r["target_kind"]] += 1
        stats["split"][r["split"]] += 1
        if set(r) != KEYS:
            err("v2_schema", (tag, rid, sorted(set(r) ^ KEYS)))
        if set(a) != CAND_KEYS:
            err("audit_schema", (tag, rid, sorted(set(a) ^ CAND_KEYS)))
        if r["part"] != "p2":
            err("part", (tag, rid, r["part"]))
        if r["split"] != src["split"] or a["split"] != src["split"]:
            err("split_drift", (tag, rid))
        if r["state"] != src["query"]:
            err("state_drift", (tag, rid))
        if r["lang"] != src["language_resolved"]:
            err("lang_drift", (tag, rid, r["lang"], src["language_resolved"]))
        if r["src_sha"] != m["source_files"]["source_csv"]:
            err("src_sha", (tag, rid))
        if a["threshold"] != t:
            err("audit_threshold", (tag, rid, a["threshold"]))
        hits = ev[key]
        if any(h["query"] != src["query"] for h in hits):
            err("evidence_query", (tag, rid))
        best = {c: -float("inf") for c in cap_ids}
        for h in hits:
            best[h["capability_id"]] = max(best[h["capability_id"]], float(h["score"]))
        qi = order_keys.index(key)
        for c in cap_ids:
            got = float(cap_matrix[qi, cap_idx[c]])
            if best[c] != -float("inf"):
                if abs(best[c] - got) > 1e-12:
                    err("cap_matrix", (tag, rid, c, best[c], got))
            elif got >= EVIDENCE_FLOOR:
                err("cap_matrix_unexplained", (tag, rid, c, got))
        # Below the evidence floor the score cannot reach any threshold; use the
        # builder's matrix value only there.
        scores = {c: (best[c] if best[c] != -float("inf") else float(cap_matrix[qi, cap_idx[c]]))
                  for c in cap_ids}
        retrieved = sorted([c for c in cap_ids if scores[c] >= t], key=lambda c: (-scores[c], c))
        top3 = retrieved[:3]
        if top3 != a["candidate_capability_ids"]:
            err("candidate_recompute", (tag, rid, top3, a["candidate_capability_ids"]))
        if len(top3) != a["candidate_count"] or a["candidate_count"] + 1 != len(r["order"]) or a["option_count"] != len(r["order"]):
            err("k_count", (tag, rid, len(top3), a["candidate_count"], len(r["order"])))
        if set(a["retrieved_capabilities"]) != set(retrieved):
            err("retrieved_set", (tag, rid))
        for c, v in a["retrieval_scores"].items():
            if c not in scores or abs(float(v) - scores[c]) > 1e-12:
                err("retrieval_score", (tag, rid, c, v, scores.get(c)))
        if set(a["retrieval_scores"]) != set(retrieved):
            err("retrieval_keys", (tag, rid))
        if not (0 <= len(top3) <= 3):
            err("k_range", (tag, rid, len(top3)))
        is_act = src["is_action"] == "True"
        gold = src["gold_capability_id"] or None
        if is_act:
            reason = "no_candidates" if not top3 else ("gold_in_candidates" if gold in top3 else "gold_not_retrieved")
        else:
            reason = "no_action"
        exp_label = gold if reason == "gold_in_candidates" else REJECT_LABEL
        if a["label"] != exp_label or a["label_reason"] != reason:
            err("label_recompute", (tag, rid, exp_label, reason, a["label"], a["label_reason"]))
        if a["is_action"] != is_act:
            err("is_action", (tag, rid))
        if is_act and a["source_gold_capability_id"] != gold:
            err("source_gold", (tag, rid, gold, a["source_gold_capability_id"]))
        if not is_act and a["source_gold_capability_id"] is not None:
            err("source_gold_noaction", (tag, rid))
        gs = scores.get(gold) if gold else None
        if not is_act and a["gold_score"] is not None:
            err("gold_score_noaction", (tag, rid))
        if is_act and gold and abs(float(a["gold_score"]) - gs) > 1e-12:
            err("gold_score", (tag, rid, gs, a["gold_score"]))
        rank = retrieved.index(gold) + 1 if (is_act and gold and gs >= t) else None
        if a["gold_retrieval_rank"] != rank:
            err("gold_rank", (tag, rid, rank, a["gold_retrieval_rank"]))
        if a["decision_state"] != ("direct_reject" if not top3 else "choose"):
            err("decision_state", (tag, rid))
        order = list(r["order"])
        opts = list(r["options"])
        if len(order) != len(opts):
            err("order_options_len", (tag, rid))
        if order.count(REJECT_LABEL) != 1:
            err("reject_count", (tag, rid, order.count(REJECT_LABEL)))
        if sorted(c for c in order if c != REJECT_LABEL) != sorted(top3):
            err("order_set", (tag, rid))
        if len(set(order)) != len(order):
            err("duplicate_option", (tag, rid, order))
        base = top3 + [REJECT_LABEL]
        perm = list(range(len(base)))
        random.Random(f"reject-shuffle-v1:{rid}").shuffle(perm)
        if r["shuffle"] != {"seed_prefix": "reject-shuffle-v1", "permutation": perm}:
            err("perm_recompute", (tag, rid))
        if len(perm) != len(order) or sorted(perm) != list(range(len(order))):
            err("perm_shape", (tag, rid, perm))
        exp_order = [base[i] for i in perm]
        exp_opts = [(cards[c] if c != REJECT_LABEL else REJECT_CARD) for c in exp_order]
        if order != exp_order:
            err("row_order_map", (tag, rid, order, exp_order))
        if opts != exp_opts:
            err("row_options_map", (tag, rid))
        for cid, card in zip(order, opts):
            if card != (REJECT_CARD if cid == REJECT_LABEL else cards[cid]):
                err("card_text", (tag, rid, cid))
        gi, ri = r["gold_index"], r["reject_index"]
        if not (0 <= gi < len(order)) or not (0 <= ri < len(order)):
            err("index_range", (tag, rid, gi, ri, len(order)))
        if order[ri] != REJECT_LABEL or ri != order.index(REJECT_LABEL):
            err("reject_index", (tag, rid))
        if r["target_kind"] == "reject":
            if r["gold_capability_id"] is not None:
                err("reject_gold", (tag, rid))
            if gi != ri or order[gi] != REJECT_LABEL:
                err("reject_target", (tag, rid))
            if exp_label != REJECT_LABEL:
                err("reject_label", (tag, rid, exp_label))
        elif r["target_kind"] == "capability":
            if r["gold_capability_id"] is None or order[gi] != r["gold_capability_id"]:
                err("cap_target", (tag, rid))
            if opts[gi] != cards[r["gold_capability_id"]]:
                err("cap_target_card", (tag, rid))
            if exp_label != r["gold_capability_id"]:
                err("cap_label", (tag, rid, exp_label))
        else:
            err("target_kind", (tag, rid, r["target_kind"]))
        if key in split_of and split_of[key] != r["split"]:
            err("cross_threshold_split", (tag, rid))
    for split in ("train", "val", "test"):
        pre = [k + "-p2" for k in order_keys if split_of[k] == split]
        perm = m["row_order"]["permutation"][split]
        if sorted(perm) != list(range(len(pre))):
            err("row_perm_shape", (tag, split))
        recompute = list(range(len(pre)))
        random.Random(f"reject-rows-v1:{split}").shuffle(recompute)
        if perm != recompute:
            err("row_perm_recompute", (tag, split))
        expected = [pre[i] for i in perm]
        if file_order[split] != expected:
            err("row_file_order", (tag, split, file_order[split][:5], expected[:5]))
        changed = sum(1 for i, (x, y) in enumerate(zip(pre, expected)) if x != y)
        stats.setdefault("row_shuffle_changed", {})[split] = (changed, len(pre))
    summary[tag] = {"rows": len(v2), "K": dict(sorted(stats["K"].items())),
                    "reject_pos": dict(sorted(stats["reject_pos"].items())),
                    "kind": dict(stats["kind"]), "split": dict(stats["split"]),
                    "row_shuffle_changed": stats.get("row_shuffle_changed")}

by_t = {tag: {} for tag in TH}
for tag in TH:
    for split in ("train", "val", "test"):
        for line in (D / tag / f"v2_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by_t[tag][r["source_query_id"]] = r
sets = [set(by_t[t]) for t in TH]
if not (sets[0] == sets[1] == sets[2]):
    err("threshold_sample_set", "mismatch")
for key in by_t["threshold_082"]:
    r82, r85, r88 = [by_t[t].get(key) for t in TH]
    if not (r82 and r85 and r88):
        continue
    c = {t: set(by_t[t][key]["order"]) - {REJECT_LABEL} for t in TH}
    if not (c["threshold_082"] >= c["threshold_085"] >= c["threshold_088"]):
        err("threshold_nesting", key)
    if not (r82["split"] == r85["split"] == r88["split"]):
        err("threshold_split", key)
    if not (r82["state"] == r85["state"] == r88["state"]):
        err("threshold_state", key)

for tag in TH:
    ks = []
    for split in ("train", "val", "test"):
        for line in (D / tag / f"v2_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                ks.append(json.loads(line)["source_query_id"])
    if len(ks) != len(set(ks)):
        err("dup_source_key", tag)
    if set(ks) != set(sel_keys):
        err("sample_set", (tag, len(ks)))

self_hits = 0
for key, hits in ev.items():
    q = norm(by_key[key]["query"])
    self_hits += sum(1 for h in hits if norm(h["corpus_query"]) == q)

print(json.dumps({
    "summary": summary,
    "self_hits": self_hits,
    "selected_standard_query_flags": {
        f"{a}/{b}": n for (a, b), n in Counter(
            (by_key[k]["is_standard_query"], by_key[k]["is_standard_query_similar"])
            for k in sel_keys).items()},
    "anomaly_counts": {k: len(v) for k, v in A.items()},
    "anomaly_samples": {k: v[:5] for k, v in A.items()},
}, ensure_ascii=False, indent=2, default=str))
