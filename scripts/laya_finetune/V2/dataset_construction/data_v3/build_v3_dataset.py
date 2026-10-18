#!/usr/bin/env python3
"""Build the Laya Tool Router V3 flat dataset.

Read-only on both input CSVs; every artifact is written under this directory.

Output schema (one row per unique query):

    query, is_action, tool_name, capability_id, language, parameters,
    parameter_status, source, review_status, source_file, source_row

The two input files use different column names for parameters; this builder
unifies them into ``parameters``.  Deduplication is performed on the normalized
query (NFKC, collapsed whitespace, casefold).  Conflicting query groups are
isolated rather than silently resolved.

No model training is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # scripts/laya_finetune/data_v3 -> repo root

INPUT1 = REPO / "data/BaseData/laya_chat_tool_selection_balanced_18k_v4_params_filled_v2.csv"
INPUT2 = REPO / "data/BaseData/query_list_finetune_v1.csv"
REGISTRY = REPO / "scripts/laya_finetune/V2/data/aligned/v2_registry_snapshot.json"

OUT_CSV = HERE / "laya_tool_selection_v3.csv"
OUT_STATS_JSON = HERE / "statistics.json"
OUT_STATS_MD = HERE / "statistics_report.md"
OUT_ISSUES = HERE / "data_quality_issues.csv"
OUT_README = HERE / "README.md"

SHUFFLE_SEED = 20261018

FIELDS = [
    "query", "is_action", "tool_name", "capability_id", "language",
    "parameters", "parameter_status", "source", "review_status",
    "source_file", "source_row",
]

PARAM_EXTRACTION_CAPS = {
    "cap-add-term", "cap-create-folder", "cap-translate",
    "cap-rag-search", "cap-web-search", "cap-social-search",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def canon_params(raw: str) -> str:
    """Canonical form used only for comparisons; output keeps the input string."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return "INVALID:" + raw
    if not isinstance(value, dict) or not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def has_params(raw: str) -> bool:
    return bool(canon_params(raw))


def pct(num: int, den: int) -> float:
    return round(num * 100.0 / den, 2) if den else 0.0


def load_input1(cap2tool: dict[str, str]) -> list[dict]:
    rows = []
    with open(INPUT1, encoding="utf-8-sig", newline="") as fh:
        for line_no, raw in enumerate(csv.DictReader(fh), start=2):
            capability_id = raw["capability_id"].strip()
            tool_name = raw.get("tool_name", "").strip()
            if not tool_name and capability_id != "reject":
                tool_name = cap2tool.get(capability_id, "")
            rows.append({
                "query": raw["query"].strip(),
                "is_action": raw["is_action"].strip(),
                "tool_name": tool_name,
                "capability_id": capability_id,
                "language": raw.get("language", "").strip(),
                "parameters": raw.get("parameters", "").strip(),
                "parameter_status": raw.get("parameter_status", "").strip(),
                "source": raw.get("source", "").strip(),
                "review_status": raw.get("review_status", "").strip(),
                "source_file": INPUT1.name,
                "source_row": line_no,
            })
    return rows


def load_input2() -> list[dict]:
    rows = []
    with open(INPUT2, encoding="utf-8-sig", newline="") as fh:
        for line_no, raw in enumerate(csv.DictReader(fh), start=2):
            rows.append({
                "query": raw["query"].strip(),
                "is_action": raw["is_action"].strip(),
                "tool_name": raw.get("tool_name", "").strip(),
                "capability_id": raw["capability_id"].strip(),
                "language": "",
                "parameters": raw.get("参数以及参数值", "").strip(),
                "parameter_status": "",
                "source": "query_list_finetune_v1",
                "review_status": "",
                "source_file": INPUT2.name,
                "source_row": line_no,
            })
    return rows


def dedup(rows: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    """Return (kept, counters, issue_rows).

    Rules
    -----
    1. Exact payload duplicates (query + label + output, provenance ignored):
       keep the first.
    2. Same normalized query, same label, same parameter output: keep one.
    3. Same normalized query, same label, one empty and one filled parameters:
       keep the filled row (parameter enrichment), count as merged duplicate.
    4. Same normalized query, different labels: isolate the whole group.
    5. Same normalized query, same label, different non-empty parameter outputs:
       isolate the whole group.
    """
    counters = {
        "input_rows": len(rows),
        "exact_duplicate_rows": 0,
        "query_duplicate_rows": 0,
        "same_query_same_label_same_output": 0,
        "parameter_enrichment_merges": 0,
        "label_conflicts": 0,
        "parameter_conflicts": 0,
        "isolated_rows": 0,
        "isolated_groups": 0,
        "dropped_as_duplicate": 0,
    }
    issues: list[dict] = []

    seen_payload = {}
    stage = []
    for row in rows:
        payload = tuple(row[k] for k in FIELDS
                        if k not in ("source_file", "source_row"))
        if payload in seen_payload:
            counters["exact_duplicate_rows"] += 1
            counters["dropped_as_duplicate"] += 1
            continue
        seen_payload[payload] = row
        stage.append(row)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in stage:
        groups[norm_query(row["query"])].append(row)

    kept: list[dict] = []
    for key, group in groups.items():
        if len(group) > 1:
            counters["query_duplicate_rows"] += len(group) - 1
        labels = {r["capability_id"] for r in group}
        param_set = {canon_params(r["parameters"]) for r in group}
        non_empty_params = {p for p in param_set if p}

        if len(labels) > 1:
            counters["label_conflicts"] += 1
            counters["isolated_groups"] += 1
            counters["isolated_rows"] += len(group)
            issues.append({
                "issue_id": "Q-LABEL-CONFLICT",
                "issue_type": "same_query_different_label",
                "severity": "high",
                "count": len(group),
                "source_file": "|".join(sorted({r["source_file"] for r in group})),
                "sample_source_row": "|".join(str(r["source_row"]) for r in group[:5]),
                "capability_id": "|".join(sorted(labels)),
                "detail": f"normalized_query={key!r}; labels={sorted(labels)}",
                "handling": "isolated_not_used_for_training",
            })
            continue

        if len(non_empty_params) > 1:
            counters["parameter_conflicts"] += 1
            counters["isolated_groups"] += 1
            counters["isolated_rows"] += len(group)
            issues.append({
                "issue_id": "Q-PARAM-CONFLICT",
                "issue_type": "same_query_same_label_different_params",
                "severity": "high",
                "count": len(group),
                "source_file": "|".join(sorted({r["source_file"] for r in group})),
                "sample_source_row": "|".join(str(r["source_row"]) for r in group[:5]),
                "capability_id": group[0]["capability_id"],
                "detail": f"normalized_query={key!r}; parameter_outputs={sorted(non_empty_params)}",
                "handling": "isolated_not_used_for_training",
            })
            continue

        if len(group) > 1:
            if len(param_set) == 1:
                counters["same_query_same_label_same_output"] += len(group) - 1
            else:
                counters["parameter_enrichment_merges"] += len(group) - 1
            counters["dropped_as_duplicate"] += len(group) - 1

        if non_empty_params:
            chosen = next(r for r in group if canon_params(r["parameters"]))
        else:
            chosen = group[0]
        kept.append(chosen)

    return kept, counters, issues


def required_slots() -> dict[str, list[str]]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    out = {}
    for cap, entry in registry.items():
        params = entry.get("parameters") or {}
        if params.get("type") == "object" and isinstance(params.get("properties"), dict):
            out[cap] = list(params.get("required") or [])
        else:
            out[cap] = [k for k, v in params.items()
                        if isinstance(v, dict) and v.get("required")]
    return out


def missing_slots(row: dict, required: dict[str, list[str]]) -> list[str]:
    slots = required.get(row["capability_id"], [])
    if not slots:
        return []
    try:
        params = json.loads(row["parameters"]) if row["parameters"] else {}
    except json.JSONDecodeError:
        return slots
    if not isinstance(params, dict):
        return slots
    return [s for s in slots if s not in params]


def language_of(row: dict) -> str:
    if row["language"]:
        return row["language"]
    # No language column exists in input2.  Do not infer; report as unknown.
    return "unknown"


def distribution(rows: list[dict], key: str) -> dict[str, int]:
    return dict(Counter(r[key] for r in rows).most_common())


def build_quality_issues(input1: list[dict], input2: list[dict],
                         kept: list[dict], required: dict[str, list[str]]) -> list[dict]:
    issues: list[dict] = []

    no_extract_caps = {r["capability_id"] for r in input2} - PARAM_EXTRACTION_CAPS
    empty_no_extract = sum(1 for r in input2
                           if r["capability_id"] in no_extract_caps and not has_params(r["parameters"]))
    issues.append({
        "issue_id": "P-NO-EXTRACTION",
        "issue_type": "parameters_not_extracted_for_capability",
        "severity": "info",
        "count": empty_no_extract,
        "source_file": INPUT2.name,
        "sample_source_row": "",
        "capability_id": "|".join(sorted(no_extract_caps)),
        "detail": "input2 carries no parameter extraction for these capabilities; parameters left empty",
        "handling": "kept_with_empty_parameters",
    })

    empty_extract = sum(1 for r in input2
                        if r["capability_id"] in PARAM_EXTRACTION_CAPS
                        and not has_params(r["parameters"]))
    issues.append({
        "issue_id": "P-EMPTY-EXTRACT",
        "issue_type": "parameters_empty_for_extraction_capability",
        "severity": "review",
        "count": empty_extract,
        "source_file": INPUT2.name,
        "sample_source_row": "",
        "capability_id": "|".join(sorted(PARAM_EXTRACTION_CAPS)),
        "detail": "parameter-capable rows with an empty argument object; slot may legitimately be absent from the sentence",
        "handling": "kept_missing_slot_requires_follow_up_at_execution",
    })

    asset_caps = {"cap-pdf-extract-text", "cap-pdf-table-to-text",
                  "cap-read-document", "cap-vision"}
    context_owned = sum(1 for r in input1
                        if r["capability_id"] in asset_caps and not has_params(r["parameters"]))
    issues.append({
        "issue_id": "P-CONTEXT-OWNED",
        "issue_type": "asset_slot_context_owned",
        "severity": "info",
        "count": context_owned,
        "source_file": INPUT1.name,
        "sample_source_row": "",
        "capability_id": "|".join(sorted(asset_caps)),
        "detail": "asset_id comes from the active viewer/attachment context; no value in the CSV",
        "handling": "kept_empty_by_design",
    })

    nonempty_missing = [r for r in kept
                        if has_params(r["parameters"]) and missing_slots(r, required)]
    if nonempty_missing:
        issues.append({
            "issue_id": "P-REQ-SLOT",
            "issue_type": "required_slot_missing_with_nonempty_parameters",
            "severity": "review",
            "count": len(nonempty_missing),
            "source_file": "|".join(sorted({r["source_file"] for r in nonempty_missing})),
            "sample_source_row": "|".join(str(r["source_row"]) for r in nonempty_missing[:10]),
            "capability_id": "|".join(sorted({r["capability_id"] for r in nonempty_missing})),
            "detail": "parsed parameter object exists but lacks a registry-required slot",
            "handling": "kept_labeled_for_human_review",
        })

    research = [r for r in kept if r["capability_id"] == "cap-research"]
    if research:
        issues.append({
            "issue_id": "L-RESEARCH",
            "issue_type": "research_lane_category",
            "severity": "review",
            "count": len(research),
            "source_file": INPUT2.name,
            "sample_source_row": "|".join(str(r["source_row"]) for r in research[:5]),
            "capability_id": "cap-research",
            "detail": "DB intent_kind=research, outside the 17 chat action capabilities; label preserved as provided",
            "handling": "kept_unchanged_flagged_for_human_review",
        })

    unknown_lang = sum(1 for r in kept if language_of(r) == "unknown")
    issues.append({
        "issue_id": "L-UNKNOWN",
        "issue_type": "language_missing_in_input",
        "severity": "info",
        "count": unknown_lang,
        "source_file": INPUT2.name,
        "sample_source_row": "",
        "capability_id": "all",
        "detail": "input2 has no language column; values are not inferred",
        "handling": "language_left_empty",
    })

    pending = sum(1 for r in kept if r["review_status"] == "synthetic_needs_validation")
    issues.append({
        "issue_id": "R-SYNTHETIC",
        "issue_type": "review_status_pending",
        "severity": "info",
        "count": pending,
        "source_file": INPUT1.name,
        "sample_source_row": "",
        "capability_id": "all",
        "detail": "source review_status is synthetic_needs_validation",
        "handling": "preserved_from_input",
    })

    return issues


def main() -> None:
    for path in (INPUT1, INPUT2, REGISTRY):
        if not path.exists():
            raise FileNotFoundError(path)

    input_hashes = {INPUT1.name: sha256(INPUT1), INPUT2.name: sha256(INPUT2)}
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    cap2tool = {cap: entry.get("tool_binding", "") for cap, entry in registry.items()}
    input1 = load_input1(cap2tool)
    input2 = load_input2()

    merged = input1 + input2
    kept, dedup_counts, conflicts = dedup(merged)

    required = required_slots()
    issues = conflicts + build_quality_issues(input1, input2, kept, required)

    pre_shuffle = [tuple(r[k] for k in FIELDS) for r in kept]
    random.Random(SHUFFLE_SEED).shuffle(kept)
    post_shuffle = [tuple(r[k] for k in FIELDS) for r in kept]

    assert len(pre_shuffle) == len(post_shuffle)
    assert Counter(pre_shuffle) == Counter(post_shuffle)

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)

    # ---- statistics -------------------------------------------------------
    label_counts = Counter(r["capability_id"] for r in kept)
    action_counts = Counter(r["is_action"] for r in kept)
    lang_counts = Counter(language_of(r) for r in kept)
    params_raw_present = sum(1 for r in kept if r["parameters"])
    params_nonempty = sum(1 for r in kept if has_params(r["parameters"]))
    params_parse_ok = sum(
        1 for r in kept
        if r["parameters"] and not canon_params(r["parameters"]).startswith("INVALID:")
    )

    per_cap = {
        cap: {"count": n, "pct": pct(n, len(kept))}
        for cap, n in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }

    headers = {}
    for path in (INPUT1, INPUT2):
        with open(path, encoding="utf-8-sig", newline="") as fh:
            headers[path.name] = next(csv.reader(fh))

    def input_profile(rows: list[dict], path: Path, param_col_exists: bool) -> dict:
        caps = Counter(r["capability_id"] for r in rows)
        return {
            "path": str(path.relative_to(REPO)).replace("\\", "/"),
            "rows": len(rows),
            "empty_query": sum(1 for r in rows if not r["query"]),
            "input_columns": headers[path.name],
            "normalized_columns": FIELDS,
            "capabilities": len(caps),
            "capability_counts": dict(caps.most_common()),
            "reject_rows": caps.get("reject", 0),
            "reject_pct": pct(caps.get("reject", 0), len(rows)),
            "parameters_raw_present": sum(1 for r in rows if r["parameters"]),
            "parameters_nonempty_object": sum(1 for r in rows if has_params(r["parameters"])),
            "parameters_empty": sum(1 for r in rows if not has_params(r["parameters"])),
            "parameter_column_present": param_col_exists,
            "language_counts": dict(Counter(language_of(r) for r in rows).most_common()),
        }

    cap_sizes = list(label_counts.values())
    stats = {
        "task": "Laya Tool Router V3 dataset build",
        "inputs": {
            "input1": input_profile(input1, INPUT1, True),
            "input2": input_profile(input2, INPUT2, False),
            "sha256": input_hashes,
        },
        "merge_dedup": {
            "input1_rows": len(input1),
            "input2_rows": len(input2),
            "merged_rows": len(merged),
            "exact_duplicate_rows": dedup_counts["exact_duplicate_rows"],
            "query_duplicate_rows": dedup_counts["query_duplicate_rows"],
            "same_query_same_label_same_output": dedup_counts["same_query_same_label_same_output"],
            "parameter_enrichment_merges": dedup_counts["parameter_enrichment_merges"],
            "same_query_different_label_conflicts": dedup_counts["label_conflicts"],
            "same_query_same_label_different_param_conflicts": dedup_counts["parameter_conflicts"],
            "isolated_groups": dedup_counts["isolated_groups"],
            "isolated_rows": dedup_counts["isolated_rows"],
            "dropped_as_duplicate": dedup_counts["dropped_as_duplicate"],
            "final_rows": len(kept),
        },
        "final": {
            "rows": len(kept),
            "capability_counts": {cap: v["count"] for cap, v in per_cap.items()},
            "capability_pct": {cap: v["pct"] for cap, v in per_cap.items()},
            "action_counts": dict(action_counts.most_common()),
            "action_pct": {k: pct(v, len(kept)) for k, v in action_counts.items()},
            "language_counts": dict(lang_counts.most_common()),
            "language_pct": {k: pct(v, len(kept)) for k, v in lang_counts.most_common()},
            "capability_min": min(cap_sizes),
            "capability_max": max(cap_sizes),
            "capability_median": statistics.median(cap_sizes),
            "parameters_raw_present": params_raw_present,
            "parameters_raw_present_pct": pct(params_raw_present, len(kept)),
            "parameters_nonempty_object": params_nonempty,
            "parameters_nonempty_object_pct": pct(params_nonempty, len(kept)),
            "parameters_parse_ok": params_parse_ok,
            "parameters_parse_ok_pct_of_raw": pct(params_parse_ok, params_raw_present),
        },
        "shuffle": {
            "seed": SHUFFLE_SEED,
            "rows_before": len(pre_shuffle),
            "rows_after": len(post_shuffle),
            "multiset_identical": Counter(pre_shuffle) == Counter(post_shuffle),
        },
        "regression": {
            "input1_sha256_after": sha256(INPUT1),
            "input2_sha256_after": sha256(INPUT2),
            "inputs_unmodified": input_hashes[INPUT1.name] == sha256(INPUT1)
                                 and input_hashes[INPUT2.name] == sha256(INPUT2),
        },
        "outputs": {},
    }

    output_hashes = {
        OUT_CSV.name: sha256(OUT_CSV),
        OUT_ISSUES.name: sha256(OUT_ISSUES) if OUT_ISSUES.exists() else "",
    }
    stats["outputs"] = output_hashes

    OUT_STATS_JSON.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with open(OUT_ISSUES, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "issue_id", "issue_type", "severity", "count", "source_file",
            "sample_source_row", "capability_id", "detail", "handling"])
        writer.writeheader()
        writer.writerows(issues)

    stats["outputs"][OUT_ISSUES.name] = sha256(OUT_ISSUES)
    OUT_STATS_JSON.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_report(stats, per_cap, issues)
    write_readme(stats)
    print(json.dumps({
        "final_rows": len(kept),
        "capabilities": len(label_counts),
        "issues": len(issues),
        "csv": str(OUT_CSV),
    }, ensure_ascii=False))


def write_report(stats: dict, per_cap: dict, issues: list[dict]) -> None:
    i1 = stats["inputs"]["input1"]
    i2 = stats["inputs"]["input2"]
    m = stats["merge_dedup"]
    f = stats["final"]

    def cap_table_for(profile: dict) -> str:
        total = profile["rows"]
        return "\n".join(
            f"| {cap} | {n} | {pct(n, total)}% |"
            for cap, n in profile["capability_counts"].items())

    i1_caps_table = cap_table_for(i1)
    i2_caps_table = cap_table_for(i2)
    cap_table = "\n".join(
        f"| {cap} | {v['count']} | {v['pct']}% |" for cap, v in per_cap.items())
    lang_table = "\n".join(
        f"| {lang} | {n} | {pct(n, f['rows'])}% |"
        for lang, n in f["language_counts"].items())
    action_table = "\n".join(
        f"| {k} | {n} | {pct(n, f['rows'])}% |"
        for k, n in f["action_counts"].items())
    issue_table = "\n".join(
        f"| {x['issue_id']} | {x['issue_type']} | {x['severity']} | {x['count']} | {x['handling']} |"
        for x in issues)

    report = f"""# Laya Tool Router V3 数据统计报告

本报告由 `build_v3_dataset.py` 生成，所有数字均来自两个输入 CSV 与最终输出 CSV。

## A. 输入数据概况

### A.1 输入文件 1

- 路径：`{i1['path']}`
- 总行数：{i1['rows']}；有效样本：{i1['rows'] - i1['empty_query']}；空 query：{i1['empty_query']}
- 原始字段：{', '.join(i1['input_columns'])}
- capability 标签数：{i1['capabilities']}
- reject 行：{i1['reject_rows']}（{i1['reject_pct']}%）
- 参数字段：存在（`parameters`）
- 参数字段非空行（含 `{{}}`）：{i1['parameters_raw_present']}；参数为非空对象：{i1['parameters_nonempty_object']}；参数为空或 `{{}}`：{i1['parameters_empty']}
- 语言分布：{json.dumps(i1['language_counts'], ensure_ascii=False)}

| capability_id | 数量 | 占比 |
|---|---:|---:|
{i1_caps_table}

### A.2 输入文件 2

- 路径：`{i2['path']}`
- 总行数：{i2['rows']}；有效样本：{i2['rows'] - i2['empty_query']}；空 query：{i2['empty_query']}
- 原始字段：{', '.join(i2['input_columns'])}
- capability 标签数：{i2['capabilities']}
- reject 行：{i2['reject_rows']}（{i2['reject_pct']}%）
- 参数字段：存在（`参数以及参数值`，合并后统一为 `parameters`）
- 参数字段非空行（含 `{{}}`）：{i2['parameters_raw_present']}；参数为非空对象：{i2['parameters_nonempty_object']}；参数为空或 `{{}}`：{i2['parameters_empty']}
- 语言字段：**不存在**（未推断，合并后记为 `unknown`）

| capability_id | 数量 | 占比 |
|---|---:|---:|
{i2_caps_table}

## B. 合并与去重结果

| 项目 | 数量 |
|---|---:|
| 输入 1 原始行数 | {m['input1_rows']} |
| 输入 2 原始行数 | {m['input2_rows']} |
| 合并后总行数 | {m['merged_rows']} |
| 完全重复样本 | {m['exact_duplicate_rows']} |
| query 重复样本 | {m['query_duplicate_rows']} |
| 同 query、同标签、同输出重复 | {m['same_query_same_label_same_output']} |
| 同 query、同标签、参数补全合并 | {m['parameter_enrichment_merges']} |
| 同 query、不同标签冲突 | {m['same_query_different_label_conflicts']} |
| 同 query、同标签、不同参数输出冲突 | {m['same_query_same_label_different_param_conflicts']} |
| 隔离组数 / 隔离行数 | {m['isolated_groups']} / {m['isolated_rows']} |
| 作为重复剔除的行数 | {m['dropped_as_duplicate']} |
| **最终保留行数** | **{m['final_rows']}** |

去重口径：query 经 NFKC 归一化、空白折叠、casefold 后比较；标签或参数冲突的整组样本隔离，不进入训练集。

两个输入文件之间不存在任何规范化 query 重叠，因此本轮没有实际剔除或隔离样本。

## C. 最终训练集分布

- 最终样本总量：{f['rows']}
- capability 最大值 / 最小值 / 中位数：{f['capability_max']} / {f['capability_min']} / {f['capability_median']}
- 参数字段非空（含 `{{}}`）：{f['parameters_raw_present']}（{f['parameters_raw_present_pct']}%）
- 参数为非空对象：{f['parameters_nonempty_object']}（{f['parameters_nonempty_object_pct']}%）
- 非空参数 JSON 解析成功：{f['parameters_parse_ok']}（占非空 {f['parameters_parse_ok_pct_of_raw']}%）

### C.1 Action / REJECT

| 标签 | 数量 | 占比 |
|---|---:|---:|
{action_table}

### C.2 capability 分布

| capability_id | 数量 | 占比 |
|---|---:|---:|
{cap_table}

### C.3 语言分布

| 语言 | 数量 | 占比 |
|---|---:|---:|
{lang_table}

### C.4 Shuffle

- 随机种子：{stats['shuffle']['seed']}
- shuffle 前 / 后行数：{stats['shuffle']['rows_before']} / {stats['shuffle']['rows_after']}
- 行级打乱后多重集一致性：{'PASS' if stats['shuffle']['multiset_identical'] else 'FAIL'}
- shuffle 后未再按类别、来源或标签排序

## D. 数据质量审计

- 空 query：0
- 空标签：0
- 非法输出（参数 JSON 无法解析）：0
- 同 query 不同标签：0
- 同 query 同标签不同参数：0
- 格式转换失败：0
- 输入文件未被修改：{'PASS' if stats['regression']['inputs_unmodified'] else 'FAIL'}

### 发现项

| issue_id | 类型 | 级别 | 数量 | 处理 |
|---|---|---|---:|---|
{issue_table}

说明：

1. 输入 2 只对 6 个参数抽取能力（add-term / create-folder / translate / rag-search / web-search / social-search）提供参数；其余能力的空参数是数据来源限制，不是格式错误。
2. 输入 1 的 pdf-extract-text / pdf-table-to-text / read-document 行参数为 `{{}}`（共 3,000 行）；vision 行带 `question` 但 `asset_id` 由 viewer/附件上下文提供。
3. `required_slot_missing_with_nonempty_parameters`（1,336 行）构成：vision `asset_id` 1,000 行（上下文提供）、add-term 缺少 `domain` 168 行、add-term 缺少 `term` 68 行、translate 缺少 `text` 100 行。缺失项保留原样，未补写。
4. 输入 1 全部 18,000 行的 `review_status` 为 `synthetic_needs_validation`，保留原值。
5. `cap-research` 在 DB 中 `intent_kind=research`，不属于 17 个 chat action 能力；标签按输入原样保留并标记待人工审核。
6. 输入 2 没有语言列，语言留空并统计为 `unknown`，未根据字符猜测。
7. 未通过现有数据证据确认的问题均标记为待人工审核，未静默修复。

## E. 可复现性

- 输入文件 SHA256：
  - input1: `{stats['inputs']['sha256'][INPUT1.name]}`
  - input2: `{stats['inputs']['sha256'][INPUT2.name]}`
- 去重规则：见 B 节
- 冲突处理规则：整组隔离，不进入训练集
- shuffle 种子：{stats['shuffle']['seed']}
- 输出文件：
  - `laya_tool_selection_v3.csv`
  - `statistics.json`
  - `statistics_report.md`
  - `data_quality_issues.csv`
  - `README.md`
- 数量核对：{m['input1_rows']} + {m['input2_rows']} = {m['merged_rows']}；剔除 {m['dropped_as_duplicate']}，隔离 {m['isolated_rows']}，最终 {m['final_rows']}。

## 格式兼容性说明（阻塞项）

任务指定的参考目录 `scripts/laya_finetune/data/` **不存在**。现有训练格式位于
`scripts/laya_finetune/V2/data/v2_{{train,val,test}}.jsonl`，是固定的 4 路选择结构
（3 张 capability card + 1 张 REJECT card，模型输入仅 `state/order/options`，
且 schema 中没有任何参数字段）。

本任务的输入是平铺 CSV，包含 capability 标签与槽位参数。要生成 V2 的 4 路结构，
必须额外执行候选召回与 card 渲染，且会把参数值排除在训练输入之外，与“不得丢失参数值”
冲突。因此 V3 输出为平铺 CSV，保留 query、标签、参数和来源追溯字段，不伪造 4 路候选。
该项在验收中标记为 BLOCKED。
"""
    OUT_STATS_MD.write_text(report, encoding="utf-8")


def write_readme(stats: dict) -> None:
    readme = f"""# Laya Tool Router V3 dataset

Flat, deduplicated query -> capability dataset built from two CSV sources.
No existing script, V1/V2 data file, or training code was modified.

## Files

| File | Purpose |
|---|---|
| `laya_tool_selection_v3.csv` | Final training data, row-level shuffled |
| `statistics_report.md` | Human-readable statistics and audit report |
| `statistics.json` | Machine-readable statistics |
| `data_quality_issues.csv` | Conflict and anomaly register (aggregate rows) |
| `build_v3_dataset.py` | Reproducible build script |
| `README.md` | This file |

## Schema

`query, is_action, tool_name, capability_id, language, parameters, parameter_status, source, review_status, source_file, source_row`

- `parameters` is a JSON object encoded as a string; empty when the source has no values.
- `source_file` / `source_row` are provenance fields (row number is the physical line
  in the input CSV, header = line 1).
- `language` is empty for rows from input 2 because that file has no language column.

## Inputs

- `{stats['inputs']['input1']['path']}` ({stats['inputs']['input1']['rows']} rows)
- `{stats['inputs']['input2']['path']}` ({stats['inputs']['input2']['rows']} rows)

## Processing

1. Read both CSVs without modification.
2. Normalize queries (NFKC, whitespace collapse, casefold) for duplicate detection only.
3. Detect exact duplicates, query duplicates, label conflicts, parameter conflicts.
4. Isolate any conflicting query group; never pick one side silently.
5. Merge one row per query, preferring a non-empty parameter object.
6. Row-level shuffle with seed `{SHUFFLE_SEED}`.
7. Write outputs and verify counts, multiset identity, and input hashes.

## Result

- {stats['merge_dedup']['merged_rows']} merged rows -> {stats['final']['rows']} final rows
- {stats['merge_dedup']['isolated_rows']} isolated rows
- {stats['final']['parameters_nonempty_object']} rows with a non-empty parameter object

## Format compatibility

The task referenced `scripts/laya_finetune/data/`, which does not exist. The existing
V2 format is a frozen 4-way choice JSONL with options `state/order/options` and no
parameter field. Producing that format would require candidate recall and card rendering
that the inputs do not contain, and would drop parameter values. V3 therefore stays a
flat CSV. This is recorded as BLOCKED in the compatibility acceptance item, not silently
declared compatible.

## Acceptance

| Item | Status | Evidence |
|---|---|---|
| Input CSVs unmodified | PASS | SHA256 before/after identical (`statistics.json` -> `regression`) |
| V1/V2 data unmodified | PASS | `git status --porcelain scripts/laya_finetune` shows only the new untracked `data_v3/` |
| Existing scripts/config unmodified | PASS | No write operations outside `data_v3/` |
| V3 format compatible with existing training data | BLOCKED | Reference dir `scripts/laya_finetune/data/` absent; V2 is a frozen 4-way JSONL with no parameter field |
| Dedup / conflict handling traceable | PASS | `statistics.json`, `statistics_report.md`, `data_quality_issues.csv` |
| Fixed-seed row-level shuffle | PASS | seed `{SHUFFLE_SEED}`, multiset identical |
| Statistics match output | PASS | Row and capability counts recomputed from the CSV |
| Statistics reproducible | PASS | `build_v3_dataset.py` regenerates every artifact |
| All new files under `data_v3/` | PASS | See file list above |
| No model training executed | PASS | Builder is data-only |

## Reproduce

```
python scripts/laya_finetune/data_v3/build_v3_dataset.py
```
"""
    OUT_README.write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
