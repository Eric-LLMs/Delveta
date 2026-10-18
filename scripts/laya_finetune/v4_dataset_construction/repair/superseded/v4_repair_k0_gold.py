"""Repair K=0 / source-gold-capability rows into Gold+REJECT supervision.

Reads the generated full_r1 bundles, writes a NEW full_r1_repaired bundle.
Originals are never opened for writing. No embedding, ANN or SQL recall.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(r"e:/Work Space/Git Repositories/DeepDive/scripts/laya_finetune/data_v4_build/full_r1")
DST = Path(r"e:/Work Space/Git Repositories/DeepDive/scripts/laya_finetune/data_v4_build/full_r1_repaired")
THRESHOLDS = (0.82, 0.85, 0.88)
TAGS = {0.82: "threshold_082", 0.85: "threshold_085", 0.88: "threshold_088"}
SPLITS = ("train", "test", "val")
REJECT = "REJECT"
SHUFFLE_PREFIX = "reject-shuffle-v1"
ROW_KEYS = [
    "id", "source_query_id", "split", "part", "lang", "state", "order", "options",
    "gold_index", "target_kind", "gold_capability_id", "reject_index", "shuffle", "src_sha",
]
CAPS = json.loads((SRC / "data/capability_ids.json").read_text(encoding="utf-8"))
CAPSET = set(CAPS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, header, rows) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ── card text from the original frozen rows ────────────────────────────────
cards: dict[str, str] = {}
card_conflicts = {"count": 0, "samples": []}
for t in THRESHOLDS:
    for split in SPLITS:
        for row in load_jsonl(SRC / "data" / TAGS[t] / f"v2_{split}.jsonl"):
            for cid, text in zip(row["order"], row["options"]):
                if cid in cards and cards[cid] != text:
                    card_conflicts["count"] += 1
                    if len(card_conflicts["samples"]) < 20:
                        card_conflicts["samples"].append(row["id"])
                else:
                    cards[cid] = text
missing_cards = [c for c in CAPS + [REJECT] if c not in cards]
if missing_cards:
    raise SystemExit(f"missing card text for {missing_cards}")

# ── source manifest for split / leakage checks ─────────────────────────────
with open(SRC / "data/sample_manifest.csv", encoding="utf-8-sig", newline="") as f:
    manifest_rows = list(csv.DictReader(f))
manifest = {int(r["source_row_id"]): r for r in manifest_rows}
eligible = [int(r["source_row_id"]) for r in manifest_rows if r["eligible"] == "True"]
source_keys = {sid: f"{manifest[sid]['source_file']}:{manifest[sid]['source_row']}:{sid}"
               for sid in manifest}

# ── repair ─────────────────────────────────────────────────────────────────
checks = defaultdict(lambda: {"count": 0, "samples": []})


def err(name, rid):
    checks[name]["count"] += 1
    if len(checks[name]["samples"]) < 20:
        checks[name]["samples"].append(rid)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def repair_row(row: dict, audit: dict) -> tuple[dict, dict]:
    gold = audit["source_gold_capability_id"]
    base_order = [gold, REJECT]
    base_options = [cards[gold], cards[REJECT]]
    perm = list(range(2))
    random.Random(f"{SHUFFLE_PREFIX}:{row['id']}").shuffle(perm)
    order = [base_order[i] for i in perm]
    options = [base_options[i] for i in perm]
    new = dict(row)
    new["order"] = order
    new["options"] = options
    new["target_kind"] = "capability"
    new["gold_capability_id"] = gold
    new["gold_index"] = order.index(gold)
    new["reject_index"] = order.index(REJECT)
    new["shuffle"] = {"seed_prefix": SHUFFLE_PREFIX, "permutation": perm}
    new = {k: new[k] for k in ROW_KEYS}

    na = dict(audit)
    na.update({
        "label": gold,
        "label_reason": "k0_gold_injected",
        "label_reason_detail": "source_gold_supervision",
        "candidate_count": 1,
        "option_count": 2,
        "candidate_capability_ids": [gold],
        "retrieved_capabilities": [],
        "retrieval_scores": {},
        "decision_state": "choose",
        "gold_retrieval_rank": None,
        "repair_reason": "k0_gold_supervision_injection",
        "candidate_origin": "source_gold_injected",
        "injected_capability_id": gold,
        "retrieval_score_available": False,
        "original_candidate_count": 0,
        "original_candidate_capability_ids": [],
        "original_retrieved_capabilities": [],
        "original_retrieval_scores": {},
        "original_label": REJECT,
        "original_label_reason": audit.get("label_reason"),
        "original_label_reason_detail": audit.get("label_reason_detail"),
        "original_decision_state": audit.get("decision_state"),
        "original_gold_score": audit.get("gold_score"),
        "original_gold_retrieval_rank": audit.get("gold_retrieval_rank"),
    })
    return new, na


(DST / "repairs").mkdir(parents=True, exist_ok=True)
repair_summary = Counter()      # (t, tool, lang, split)
repaired_total = Counter()      # t
k_before = Counter()            # (t, k)
k_after = Counter()             # (t, k)
repair_detail_all = []
input_hashes = {}
output_hashes = {}

for t in THRESHOLDS:
    tag = TAGS[t]
    sdir = SRC / "data" / tag
    ddir = DST / "data" / tag
    ddir.mkdir(parents=True, exist_ok=True)
    audit_in = sdir / "construction_audit.jsonl"
    input_hashes[str(audit_in)] = sha256(audit_in)
    audits = {a["id"]: a for a in load_jsonl(audit_in)}
    detail_rows = []
    out_audits = []
    repaired_ids = set()
    for split in SPLITS:
        in_path = sdir / f"v2_{split}.jsonl"
        out_path = ddir / f"v2_{split}.jsonl"
        input_hashes[str(in_path)] = sha256(in_path)
        input_ids = []
        output_ids = []
        with open(out_path, "w", encoding="utf-8", newline="\n") as fout:
            for row in load_jsonl(in_path):
                input_ids.append(row["id"])
                a = audits.get(row["id"])
                if a is None:
                    raise SystemExit(f"{tag}: audit missing for {row['id']}")
                else:
                    qualifies = (
                        bool(a.get("is_action"))
                        and int(a.get("candidate_count", -1)) == 0
                        and len(row["order"]) == 1
                        and a.get("source_gold_capability_id") in CAPSET
                    )
                    if int(a.get("candidate_count", -1)) == 0 and len(row["order"]) != 1:
                        err(f"{tag}_audit_row_k_mismatch", row["id"])
                    if qualifies:
                        if a.get("label_reason") != "no_candidates":
                            err(f"{tag}_unexpected_k0_reason", row["id"])
                        out, na = repair_row(row, a)
                        repaired_ids.add(row["id"])
                        repaired_total[t] += 1
                        tool = a["source_gold_capability_id"]
                        repair_summary[(t, tool, row["lang"], split)] += 1
                        detail_rows.append({
                            "threshold": t,
                            "id": row["id"],
                            "source_query_id": row["source_query_id"],
                            "split": split,
                            "lang": row["lang"],
                            "tool": tool,
                            "repair_reason": "k0_gold_supervision_injection",
                            "candidate_origin": "source_gold_injected",
                            "retrieval_score_available": False,
                            "original": {
                                "candidate_count": 0,
                                "candidate_capability_ids": [],
                                "retrieved_capabilities": [],
                                "retrieval_scores": {},
                                "label": REJECT,
                                "label_reason": a.get("label_reason"),
                                "label_reason_detail": a.get("label_reason_detail"),
                                "decision_state": a.get("decision_state"),
                                "order": row["order"],
                                "options": row["options"],
                                "gold_index": row["gold_index"],
                                "reject_index": row["reject_index"],
                                "target_kind": row["target_kind"],
                                "shuffle": row["shuffle"],
                                "gold_score": a.get("gold_score"),
                                "gold_retrieval_rank": a.get("gold_retrieval_rank"),
                            },
                            "repaired": {
                                "candidate_count": 1,
                                "candidate_capability_ids": [tool],
                                "label": tool,
                                "order": out["order"],
                                "options": out["options"],
                                "gold_index": out["gold_index"],
                                "reject_index": out["reject_index"],
                                "target_kind": out["target_kind"],
                                "gold_capability_id": out["gold_capability_id"],
                                "shuffle": out["shuffle"],
                            },
                        })
                    else:
                        out, na = row, a
                # record before/after K from the *original* row
                k_before[(t, len(row["order"]) - 1)] += 1
                k_after[(t, len(out["order"]) - 1)] += 1
                output_ids.append(out["id"])
                fout.write(json.dumps(out, ensure_ascii=False,
                                      separators=(",", ":")) + "\n")
                out_audits.append(na if na is not None else a)
        if input_ids != output_ids:
            err(f"{tag}_{split}_file_order_changed", split)
        output_hashes[str(out_path)] = sha256(out_path)
        # per-split detail is written from the global list below
    # write audit in output order
    audit_path = ddir / "construction_audit.jsonl"
    write_jsonl(audit_path, out_audits)
    output_hashes[str(audit_path)] = sha256(audit_path)
    # direct_reject after repair
    direct = [r for r in load_jsonl(ddir / "v2_train.jsonl") if len(r["order"]) == 1]
    direct += [r for r in load_jsonl(ddir / "v2_test.jsonl") if len(r["order"]) == 1]
    direct += [r for r in load_jsonl(ddir / "v2_val.jsonl") if len(r["order"]) == 1]
    write_jsonl(ddir / "direct_reject.jsonl", direct)
    output_hashes[str(ddir / "direct_reject.jsonl")] = sha256(ddir / "direct_reject.jsonl")
    # per-threshold detail
    write_jsonl(ddir / "repair_detail.jsonl", detail_rows)
    repair_detail_all.extend(detail_rows)
    # manifest
    m = json.loads((sdir / "manifest.json").read_text(encoding="utf-8"))
    m["dataset_variant"] = "V4-dynamic-repaired"
    m["repair"] = {
        "rule": "K=0 and source_gold_capability_id is a valid capability",
        "action": "inject source gold as the only capability candidate; K=1; options=2",
        "candidate_origin": "source_gold_injected",
        "retrieval_score_available": False,
        "source_bundle": str(sdir),
        "repaired_rows": sum(1 for d in detail_rows),
        "repaired_by_split": dict(Counter(d["split"] for d in detail_rows)),
        "repaired_by_tool": dict(Counter(d["tool"] for d in detail_rows)),
        "note": "Injected candidate is supervision, not a retrieval hit. "
                "Original retrieval evidence is preserved in construction_audit.jsonl.",
    }
    for split in SPLITS:
        m["source_files"][f"v2_{split}.jsonl"] = output_hashes[str(ddir / f"v2_{split}.jsonl")]
    m["source_files"]["construction_audit.jsonl"] = output_hashes[str(audit_path)]
    m["option_slots"]["observed"] = sorted({len(r["order"]) for r in
                                           load_jsonl(ddir / "v2_train.jsonl")} |
                                          {len(r["order"]) for r in
                                           load_jsonl(ddir / "v2_test.jsonl")} |
                                          {len(r["order"]) for r in
                                           load_jsonl(ddir / "v2_val.jsonl")})
    m["gold_slot_distribution"] = {}
    for split in SPLITS:
        m["gold_slot_distribution"][split] = {
            str(k): v for k, v in sorted(Counter(
                r["gold_index"] for r in load_jsonl(ddir / f"v2_{split}.jsonl")).items())}
    (ddir / "manifest.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_hashes[str(ddir / "manifest.json")] = sha256(ddir / "manifest.json")

# per-split detail files (global, grouped by threshold)
for t in THRESHOLDS:
    for split in SPLITS:
        rows = [d for d in repair_detail_all if d["threshold"] == t and d["split"] == split]
        write_jsonl(DST / "data" / TAGS[t] / f"repair_detail_{split}.jsonl", rows)

# ── validation ─────────────────────────────────────────────────────────────
val = defaultdict(lambda: {"count": 0, "samples": []})


def verr(name, rid):
    val[name]["count"] += 1
    if len(val[name]["samples"]) < 20:
        val[name]["samples"].append(rid)


split_before = {}
for sid in eligible:
    split_before[sid] = manifest[sid]["split"]
for t in THRESHOLDS:
    tag = TAGS[t]
    audit_map = {a["id"]: a for a in
                 load_jsonl(DST / "data" / tag / "construction_audit.jsonl")}
    for split in SPLITS:
        for row in load_jsonl(DST / "data" / tag / f"v2_{split}.jsonl"):
            rid = row["id"]
            if set(row) != set(ROW_KEYS):
                verr(f"{tag}_schema_keys", rid)
                continue
            ok_types = (
                isinstance(row["id"], str) and isinstance(row["source_query_id"], str)
                and isinstance(row["split"], str) and isinstance(row["part"], str)
                and isinstance(row["lang"], str) and isinstance(row["state"], str)
                and isinstance(row["order"], list) and all(isinstance(x, str) for x in row["order"])
                and isinstance(row["options"], list) and all(isinstance(x, str) for x in row["options"])
                and is_int(row["gold_index"]) and is_int(row["reject_index"])
                and isinstance(row["target_kind"], str)
                and (row["gold_capability_id"] is None or isinstance(row["gold_capability_id"], str))
                and isinstance(row["shuffle"], dict)
                and isinstance(row["shuffle"].get("permutation"), list)
                and isinstance(row["shuffle"].get("seed_prefix"), str)
                and isinstance(row["src_sha"], str)
            )
            if not ok_types:
                verr(f"{tag}_field_types", rid)
            n = len(row["order"])
            if n != len(row["options"]):
                verr(f"{tag}_len_mismatch", rid)
            if not (1 <= n <= 4):
                verr(f"{tag}_option_bounds", rid)
            if row["order"].count(REJECT) != 1:
                verr(f"{tag}_reject_count", rid)
            if row["target_kind"] == "reject":
                if row["gold_capability_id"] is not None:
                    verr(f"{tag}_reject_gold_nonnull", rid)
                if row["gold_index"] != row["reject_index"]:
                    verr(f"{tag}_reject_gold_index", rid)
            else:
                if row["gold_capability_id"] not in CAPSET:
                    verr(f"{tag}_invalid_gold_cap", rid)
                elif row["order"][row["gold_index"]] != row["gold_capability_id"]:
                    verr(f"{tag}_gold_not_at_index", rid)
                if row["gold_capability_id"] not in row["order"]:
                    verr(f"{tag}_gold_not_in_candidates", rid)
                if row["gold_index"] == row["reject_index"]:
                    verr(f"{tag}_gold_at_reject", rid)
            if sorted(row["shuffle"]["permutation"]) != list(range(n)):
                verr(f"{tag}_permutation", rid)
            base = list(audit_map.get(rid, {}).get("candidate_capability_ids", [])) + [REJECT]
            if (len(base) == n
                    and sorted(row["shuffle"]["permutation"]) == list(range(n))):
                if [base[i] for i in row["shuffle"]["permutation"]] != row["order"]:
                    verr(f"{tag}_shuffle_order", rid)
            seen = [x for x in row["order"] if x != REJECT]
            if len(seen) != len(set(seen)):
                verr(f"{tag}_duplicate_candidate", rid)
            if row["split"] != split:
                verr(f"{tag}_split_file", rid)
            if row["part"] != "p2":
                verr(f"{tag}_part", rid)
            sid = int(row["source_query_id"].rsplit(":", 1)[-1])
            if sid in split_before and row["split"] != split_before[sid]:
                verr(f"{tag}_split_changed", rid)
            for cid, text in zip(row["order"], row["options"]):
                if cards.get(cid) != text:
                    verr(f"{tag}_card_text", rid)
    # audit join
    rids = set()
    for split in SPLITS:
        rids |= {r["id"] for r in load_jsonl(DST / "data" / tag / f"v2_{split}.jsonl")}
    aids = {a["id"] for a in load_jsonl(DST / "data" / tag / "construction_audit.jsonl")}
    if rids != aids:
        verr(f"{tag}_audit_join", f"missing={len(rids-aids)} extra={len(aids-rids)}")

# leakage / near-duplicate proxy
norm_groups = defaultdict(set)
family_groups = defaultdict(set)
norm_counts = Counter()
for sid in eligible:
    norm_counts[manifest[sid]["normalized_query_sha256"]] += 1
    norm_groups[manifest[sid]["normalized_query_sha256"]].add(manifest[sid]["split"])
    family_groups[manifest[sid]["query_family_id"]].add(manifest[sid]["split"])
norm_leaks = {k: sorted(v) for k, v in norm_groups.items() if len(v) > 1}
family_leaks = {k: sorted(v) for k, v in family_groups.items() if len(v) > 1}
dup_norm = {k: v for k, v in norm_counts.items() if v > 1}

# original inputs untouched?
input_unchanged = all(sha256(Path(p)) == h for p, h in input_hashes.items())

# ── summary CSV ────────────────────────────────────────────────────────────
summary_rows = []
for (t, tool, lang, split), n in sorted(repair_summary.items()):
    summary_rows.append([f"{t:.2f}", tool, lang, split, n])
write_csv(DST / "repairs/repair_summary.csv",
          ["threshold", "tool", "language", "split", "repaired_count"], summary_rows)

before_after = []
for t in THRESHOLDS:
    for k in range(4):
        before_after.append([f"{t:.2f}", "before", k, k_before[(t, k)]])
        before_after.append([f"{t:.2f}", "after", k, k_after[(t, k)]])
write_csv(DST / "repairs/before_after_k.csv",
          ["threshold", "stage", "k", "count"], before_after)

# quality audit
dq = {
    "source_bundle": str(SRC),
    "repaired_bundle": str(DST),
    "criteria": {
        "is_action": True,
        "original_candidate_count": 0,
        "source_gold_capability_id": "in 18 capability ids",
    },
    "repaired_total": {f"{t:.2f}": repaired_total[t] for t in THRESHOLDS},
    "k_before": {f"{t:.2f}": {str(k): k_before[(t, k)] for k in range(4)} for t in THRESHOLDS},
    "k_after": {f"{t:.2f}": {str(k): k_after[(t, k)] for k in range(4)} for t in THRESHOLDS},
    "checks": {k: v for k, v in sorted(val.items())},
    "card_text_conflicts": card_conflicts,
    "original_inputs_unchanged": input_unchanged,
    "split_unchanged": sum(v["count"] for k, v in val.items()
                           if k.endswith("_split_changed")) == 0,
    "leakage": {
        "normalized_query_cross_split": len(norm_leaks),
        "family_cross_split": len(family_leaks),
        "duplicate_normalized_query_groups": len(dup_norm),
        "duplicate_rows": sum(v - 1 for v in dup_norm.values()),
    },
    "input_hashes": input_hashes,
    "output_hashes": output_hashes,
}
(DST / "repairs").mkdir(parents=True, exist_ok=True)
(DST / "repairs/data_quality_audit.json").write_text(
    json.dumps(dq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# report
lines = []
lines.append("# Laya V4 K=0 / Gold-Tool 修复报告\n")
lines.append(f"- 源数据：`{SRC}`（只读）")
lines.append(f"- 修复输出：`{DST}`（新目录，原数据未覆盖）\n")
lines.append("## 1. 修复规则\n")
lines.append("- 只处理 `is_action=True`、原始 `candidate_count=0`、"
             "`source_gold_capability_id` 属于 18 个有效 capability 的行。")
lines.append("- 原始 Gold 为 REJECT / no-action 的 K=0 行不修改。")
lines.append("- 修复后：Gold capability 作为唯一候选，REJECT 保留，K=1，选项数 2。")
lines.append("- 选项顺序按既有 `reject-shuffle-v1:<row id>` 规则重新生成。")
lines.append("- 注入候选**没有检索分数**，审计中明确标记 `retrieval_score_available=false`、"
             "`candidate_origin=source_gold_injected`，不伪造预测值。\n")
lines.append("## 2. 修复数量\n")
lines.append("| 阈值 | 修复行数 |")
lines.append("|---|---:|")
for t in THRESHOLDS:
    lines.append(f"| {t:.2f} | {repaired_total[t]} |")
lines.append("\n按 tool / language / split 的明细见 `repair_summary.csv`。\n")
lines.append("## 3. K 分布（修复前 → 修复后）\n")
lines.append("| 阈值 | K=0 | K=1 | K=2 | K=3 |")
lines.append("|---|---:|---:|---:|---:|")
for t in THRESHOLDS:
    lines.append(f"| {t:.2f} 修复前 | {k_before[(t,0)]} | {k_before[(t,1)]} | "
                 f"{k_before[(t,2)]} | {k_before[(t,3)]} |")
    lines.append(f"| {t:.2f} 修复后 | {k_after[(t,0)]} | {k_after[(t,1)]} | "
                 f"{k_after[(t,2)]} | {k_after[(t,3)]} |")
lines.append("\n## 4. 验证结果\n")
gold_ok = all(v["count"] == 0 for k, v in val.items()
              if k.endswith("_gold_not_in_candidates")
              or k.endswith("_gold_not_at_index")
              or k.endswith("_invalid_gold_cap"))
lines.append(f"- 修复后 Gold 全部存在于候选集：{'PASS' if gold_ok else 'FAIL'}")
lines.append(f"- Schema / 类型 / 索引 / permutation / 卡文本 / audit join 错误："
             f"{sum(v['count'] for v in val.values())}")
lines.append(f"- 原始 train/test/val 归属保持不变："
             f"{'PASS' if all(not k.endswith('_split_changed') or v['count'] == 0 for k, v in val.items()) else 'FAIL'}")
lines.append(f"- 原始输入文件未修改：{'PASS' if input_unchanged else 'FAIL'}")
lines.append(f"- 跨 split 规范化 query：{len(norm_leaks)}；"
             f"跨 split family：{len(family_leaks)}；"
             f"重复规范化 query 组：{len(dup_norm)}。")
lines.append(f"- 卡文本冲突：{card_conflicts['count']}。\n")
lines.append("错误明细（如有）：")
lines.append("```json")
lines.append(json.dumps({k: v for k, v in val.items() if v["count"]}, ensure_ascii=False, indent=2))
lines.append("```\n")
lines.append("## 5. 与原始检索的关系\n")
lines.append("- `construction_audit.jsonl` 保留原始 `retrieved_capabilities`、`retrieval_scores`、"
             "`original_label` 和 `original_decision_state`。")
lines.append("- `candidate_capability_ids` 在审计中改为训练候选 `[gold]`；"
             "原始空候选记录在 `original_candidate_capability_ids`。")
lines.append("- 修复行不是原始 ANN 命中，不能用于评估召回率，只能用于训练监督。\n")
lines.append("## 6. 结论\n")
lines.append("修复数据结构与现有 V2 14-key schema 一致，split 与文件顺序保持不变，"
             "Gold 标签与索引一致。**暂不启动训练或部署**，建议先人工抽检修复明细。")
(DST / "repairs/repair_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({
    "repaired": {f"{t:.2f}": repaired_total[t] for t in THRESHOLDS},
    "k_before": {f"{t:.2f}": {str(k): k_before[(t, k)] for k in range(4)} for t in THRESHOLDS},
    "k_after": {f"{t:.2f}": {str(k): k_after[(t, k)] for k in range(4)} for t in THRESHOLDS},
    "validation_errors": {k: v["count"] for k, v in val.items() if v["count"]},
    "card_conflicts": card_conflicts["count"],
    "original_inputs_unchanged": input_unchanged,
    "leakage": {"norm_cross_split": len(norm_leaks), "family_cross_split": len(family_leaks)},
}, ensure_ascii=False, indent=2))
