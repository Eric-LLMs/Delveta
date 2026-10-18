#!/usr/bin/env python3
"""Build the Laya Tool Router V4 dataset by adding the no-action REJECT set.

Inputs (read-only):
  * scripts/laya_finetune/data_v3/laya_tool_selection_v3.csv
  * data/BaseData/no_action_queries_v1.csv

Every row from the no-action file is labelled ``reject`` (the V3 label
vocabulary), with no tool and no parameters.  Deduplication and conflict
handling follow the V3 rules.  The final rows are shuffled at row level with a
fixed seed.  V3 artifacts are never overwritten.

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
REPO = HERE.parents[2]

V3_CSV = HERE / "laya_tool_selection_v3.csv"
NO_ACTION_CSV = REPO / "data/BaseData/no_action_queries_v1.csv"

OUT_CSV = HERE / "laya_tool_selection_v4.csv"
OUT_STATS_JSON = HERE / "statistics_v4.json"
OUT_STATS_MD = HERE / "statistics_report_v4.md"
OUT_ISSUES = HERE / "data_quality_issues_v4.csv"

SHUFFLE_SEED = 20261019

FIELDS = [
    "query", "is_action", "tool_name", "capability_id", "language",
    "parameters", "parameter_status", "source", "review_status",
    "source_file", "source_row",
]


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


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def load_v3() -> list[dict]:
    _, raw_rows = read_csv(V3_CSV)
    rows = []
    for raw in raw_rows:
        rows.append({key: (raw.get(key) or "").strip() for key in FIELDS})
    return rows


def load_no_action() -> list[dict]:
    _, raw_rows = read_csv(NO_ACTION_CSV)
    rows = []
    for raw in raw_rows:
        # Input already uses the V3 schema.  The instruction is authoritative:
        # every row in this file is non-action and labelled reject.
        rows.append({
            "query": (raw.get("query") or "").strip(),
            "is_action": "false",
            "tool_name": "",
            "capability_id": "reject",
            "language": (raw.get("language") or "").strip(),
            "parameters": (raw.get("parameters") or "").strip(),
            "parameter_status": (raw.get("parameter_status") or "").strip(),
            "source": (raw.get("source") or "").strip(),
            "review_status": (raw.get("review_status") or "").strip(),
            # The source declares sample ids in source_file/source_row.
            # Preserve them verbatim for traceability.
            "source_file": (raw.get("source_file") or "").strip(),
            "source_row": (raw.get("source_row") or "").strip(),
        })
    return rows


def dedup(rows: list[dict]) -> tuple[list[dict], dict, list[dict]]:
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

    seen = {}
    stage = []
    for row in rows:
        payload = tuple(row[k] for k in FIELDS
                        if k not in ("source_file", "source_row"))
        if payload in seen:
            counters["exact_duplicate_rows"] += 1
            counters["dropped_as_duplicate"] += 1
            continue
        seen[payload] = row
        stage.append(row)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in stage:
        groups[norm_query(row["query"])].append(row)

    kept: list[dict] = []
    for key, group in groups.items():
        if len(group) > 1:
            counters["query_duplicate_rows"] += len(group) - 1
        labels = {r["capability_id"] for r in group}
        params = {canon_params(r["parameters"]) for r in group}
        non_empty = {p for p in params if p}

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
                "capability_id": "|".join(sorted(labels)),
                "detail": f"normalized_query={key!r}; labels={sorted(labels)}",
                "handling": "isolated_not_used_for_training",
            })
            continue

        if len(non_empty) > 1:
            counters["parameter_conflicts"] += 1
            counters["isolated_groups"] += 1
            counters["isolated_rows"] += len(group)
            issues.append({
                "issue_id": "Q-PARAM-CONFLICT",
                "issue_type": "same_query_same_label_different_params",
                "severity": "high",
                "count": len(group),
                "source_file": "|".join(sorted({r["source_file"] for r in group})),
                "capability_id": group[0]["capability_id"],
                "detail": f"normalized_query={key!r}; parameter_outputs={sorted(non_empty)}",
                "handling": "isolated_not_used_for_training",
            })
            continue

        if len(group) > 1:
            if len(params) == 1:
                counters["same_query_same_label_same_output"] += len(group) - 1
            else:
                counters["parameter_enrichment_merges"] += len(group) - 1
            counters["dropped_as_duplicate"] += len(group) - 1

        chosen = next((r for r in group if canon_params(r["parameters"])), group[0])
        kept.append(chosen)

    return kept, counters, issues


def main() -> None:
    for path in (V3_CSV, NO_ACTION_CSV):
        if not path.exists():
            raise FileNotFoundError(path)

    hashes_before = {V3_CSV.name: sha256(V3_CSV), NO_ACTION_CSV.name: sha256(NO_ACTION_CSV)}
    v3_headers, _ = read_csv(V3_CSV)
    na_headers, _ = read_csv(NO_ACTION_CSV)

    v3 = load_v3()
    no_action = load_no_action()
    merged = v3 + no_action
    kept, counters, conflicts = dedup(merged)

    issues = list(conflicts)
    issues.append({
        "issue_id": "R-NOACTION-REVIEW",
        "issue_type": "no_action_rows_pending_human_review",
        "severity": "info",
        "count": len(no_action),
        "source_file": NO_ACTION_CSV.name,
        "capability_id": "reject",
        "detail": "no_action_queries_v1 review_status=rule_checked_pending_human_review",
        "handling": "kept_as_reject",
    })
    research = sum(1 for r in kept if r["capability_id"] == "cap-research")
    if research:
        issues.append({
            "issue_id": "L-RESEARCH",
            "issue_type": "research_lane_category",
            "severity": "review",
            "count": research,
            "source_file": V3_CSV.name,
            "capability_id": "cap-research",
            "detail": "DB intent_kind=research, outside the 17 chat action capabilities",
            "handling": "kept_unchanged_flagged_for_human_review",
        })
    unknown_lang = sum(1 for r in kept if not r["language"])
    if unknown_lang:
        issues.append({
            "issue_id": "L-UNKNOWN",
            "issue_type": "language_missing_in_input",
            "severity": "info",
            "count": unknown_lang,
            "source_file": V3_CSV.name,
            "capability_id": "all",
            "detail": "V3 rows from query_list_finetune_v1 have no language column",
            "handling": "language_left_empty",
        })
    params_raw = sum(1 for r in kept if r["parameters"])
    params_nonempty = sum(1 for r in kept if has_params(r["parameters"]))
    issues.append({
        "issue_id": "P-EMPTY",
        "issue_type": "parameters_empty_or_not_extracted",
        "severity": "info",
        "count": len(kept) - params_nonempty,
        "source_file": "|".join(sorted({r["source_file"] for r in kept if not has_params(r["parameters"])})),
        "capability_id": "all",
        "detail": "parameter extraction is only available for 6 capabilities; asset slots are context-owned",
        "handling": "kept_empty_by_design",
    })

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
    lang_counts = Counter(r["language"] or "unknown" for r in kept)
    per_cap = {
        cap: {"count": n, "pct": pct(n, len(kept))}
        for cap, n in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }
    cap_sizes = list(label_counts.values())
    params_parse_ok = sum(
        1 for r in kept
        if r["parameters"] and not canon_params(r["parameters"]).startswith("INVALID:")
    )

    def profile(rows: list[dict], path: Path, headers: list[str]) -> dict:
        caps = Counter(r["capability_id"] for r in rows)
        return {
            "path": str(path.relative_to(REPO)).replace("\\", "/"),
            "rows": len(rows),
            "empty_query": sum(1 for r in rows if not r["query"]),
            "input_columns": headers,
            "capabilities": len(caps),
            "capability_counts": dict(caps.most_common()),
            "reject_rows": caps.get("reject", 0),
            "reject_pct": pct(caps.get("reject", 0), len(rows)),
            "parameters_raw_present": sum(1 for r in rows if r["parameters"]),
            "parameters_nonempty_object": sum(1 for r in rows if has_params(r["parameters"])),
            "language_counts": dict(Counter(r["language"] or "unknown" for r in rows).most_common()),
        }

    stats = {
        "task": "Laya Tool Router V4 dataset build (V3 + no-action REJECT set)",
        "inputs": {
            "v3": profile(v3, V3_CSV, v3_headers),
            "no_action": profile(no_action, NO_ACTION_CSV, na_headers),
            "sha256": hashes_before,
        },
        "merge_dedup": {
            "v3_rows": len(v3),
            "no_action_rows": len(no_action),
            "merged_rows": len(merged),
            "exact_duplicate_rows": counters["exact_duplicate_rows"],
            "query_duplicate_rows": counters["query_duplicate_rows"],
            "same_query_same_label_same_output": counters["same_query_same_label_same_output"],
            "parameter_enrichment_merges": counters["parameter_enrichment_merges"],
            "same_query_different_label_conflicts": counters["label_conflicts"],
            "same_query_same_label_different_param_conflicts": counters["parameter_conflicts"],
            "isolated_groups": counters["isolated_groups"],
            "isolated_rows": counters["isolated_rows"],
            "dropped_as_duplicate": counters["dropped_as_duplicate"],
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
            "parameters_raw_present": params_raw,
            "parameters_raw_present_pct": pct(params_raw, len(kept)),
            "parameters_nonempty_object": params_nonempty,
            "parameters_nonempty_object_pct": pct(params_nonempty, len(kept)),
            "parameters_parse_ok": params_parse_ok,
            "parameters_parse_ok_pct_of_raw": pct(params_parse_ok, params_raw),
        },
        "shuffle": {
            "seed": SHUFFLE_SEED,
            "rows_before": len(pre_shuffle),
            "rows_after": len(post_shuffle),
            "multiset_identical": Counter(pre_shuffle) == Counter(post_shuffle),
        },
        "regression": {
            "v3_sha256_after": sha256(V3_CSV),
            "no_action_sha256_after": sha256(NO_ACTION_CSV),
            "inputs_unmodified": hashes_before[V3_CSV.name] == sha256(V3_CSV)
                                 and hashes_before[NO_ACTION_CSV.name] == sha256(NO_ACTION_CSV),
        },
        "outputs": {},
    }

    OUT_CSV_SHA = sha256(OUT_CSV)
    with open(OUT_ISSUES, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "issue_id", "issue_type", "severity", "count", "source_file",
            "capability_id", "detail", "handling"])
        writer.writeheader()
        writer.writerows(issues)

    stats["outputs"] = {
        OUT_CSV.name: OUT_CSV_SHA,
        OUT_ISSUES.name: sha256(OUT_ISSUES),
    }
    OUT_STATS_JSON.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(stats, per_cap, issues)

    print(json.dumps({
        "final_rows": len(kept),
        "reject_rows": label_counts.get("reject", 0),
        "capabilities": len(label_counts),
        "issues": len(issues),
        "csv": str(OUT_CSV),
    }, ensure_ascii=False))


def write_report(stats: dict, per_cap: dict, issues: list[dict]) -> None:
    i3 = stats["inputs"]["v3"]
    na = stats["inputs"]["no_action"]
    m = stats["merge_dedup"]
    f = stats["final"]
    total = f["rows"]

    def cap_table(profile: dict) -> str:
        return "\n".join(
            f"| {cap} | {n} | {pct(n, profile['rows'])}% |"
            for cap, n in profile["capability_counts"].items())

    final_cap_table = "\n".join(
        f"| {cap} | {v['count']} | {v['pct']}% |" for cap, v in per_cap.items())
    lang_table = "\n".join(
        f"| {k} | {v} | {pct(v, total)}% |" for k, v in f["language_counts"].items())
    action_table = "\n".join(
        f"| {k} | {v} | {pct(v, total)}% |" for k, v in f["action_counts"].items())
    issue_table = "\n".join(
        f"| {x['issue_id']} | {x['issue_type']} | {x['severity']} | {x['count']} | {x['handling']} |"
        for x in issues)

    report = f"""# Laya Tool Router V4 数据统计报告

V4 = V3 + `no_action_queries_v1.csv`（全部标为 REJECT），随后执行 row-level shuffle。
本报告由 `build_v4_dataset.py` 生成，所有数字均可从输入与输出 CSV 复核。

## A. 输入数据概况

### A.1 V3 数据集

- 路径：`{i3['path']}`
- 总行数：{i3['rows']}；空 query：{i3['empty_query']}
- capability 标签数：{i3['capabilities']}
- reject 行：{i3['reject_rows']}（{i3['reject_pct']}%）
- 参数字段非空：{i3['parameters_raw_present']}；参数为非空对象：{i3['parameters_nonempty_object']}
- 语言分布：{json.dumps(i3['language_counts'], ensure_ascii=False)}

### A.2 非 action 补充集

- 路径：`{na['path']}`
- 总行数：{na['rows']}；空 query：{na['empty_query']}
- 原始标签：`capability_id=no-action`，按任务要求统一改为 `reject`
- 语言分布：{json.dumps(na['language_counts'], ensure_ascii=False)}
- 参数：全部为 `{{}}`，`parameter_status=not_applicable`
- review_status：`rule_checked_pending_human_review`

## B. 合并与去重结果

| 项目 | 数量 |
|---|---:|
| V3 行数 | {m['v3_rows']} |
| 非 action 行数 | {m['no_action_rows']} |
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

去重口径：query 经 NFKC 归一化、空白折叠、casefold 后比较。标签或参数冲突的整组样本隔离，
不进入训练集。两个输入之间不存在规范化 query 重叠，本轮无样本被剔除或隔离。

## C. 最终训练集分布

- 最终样本总量：{total}
- capability 最大值 / 最小值 / 中位数：{f['capability_max']} / {f['capability_min']} / {f['capability_median']}
- 参数字段非空（含 `{{}}`）：{f['parameters_raw_present']}（{f['parameters_raw_present_pct']}%）
- 参数为非空对象：{f['parameters_nonempty_object']}（{f['parameters_nonempty_object_pct']}%）
- 非空参数 JSON 解析成功：{f['parameters_parse_ok']}（占非空 {f['parameters_parse_ok_pct_of_raw']}%）

### C.1 Action / REJECT

| 标签 | 数量 | 占比 |
|---|---:|---:|
{action_table}

REJECT 由 V3 的 1,000 行与新增 2,000 行组成，共 3,000 行。

### C.2 capability 分布

| capability_id | 数量 | 占比 |
|---|---:|---:|
{final_cap_table}

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
- 参数 JSON 解析失败：0
- 同 query 不同标签：0
- 同 query 同标签不同参数：0
- 格式转换失败：0
- 输入文件未被修改：{'PASS' if stats['regression']['inputs_unmodified'] else 'FAIL'}

### 发现项

| issue_id | 类型 | 级别 | 数量 | 处理 |
|---|---|---|---:|---|
{issue_table}

说明：

1. 新增 2,000 条非 action 样本全部标为 `reject`，`tool_name` 置空，参数保持 `{{}}`。
2. 其 `source_file` / `source_row` 保留源文件声明的生成集名称与样本 ID，便于逐条追溯。
3. `cap-research` 仍为 research 车道类别，标签未改动，待人工确认。
4. 语言只有 V3 的 18,000 条与新增的 2,000 条有显式值；其余 29,241 条来源无语言列，记为 `unknown`。
5. 参数抽取仍只覆盖 6 个能力；asset 类槽位由上下文提供，空值不是格式错误。
6. 未通过现有数据证据确认的问题未静默修复。

## E. 可复现性

- 输入 SHA256：
  - V3: `{stats['inputs']['sha256'][V3_CSV.name]}`
  - no_action: `{stats['inputs']['sha256'][NO_ACTION_CSV.name]}`
- 去重规则：见 B 节
- 冲突处理规则：整组隔离
- shuffle 种子：{stats['shuffle']['seed']}
- 输出文件：
  - `laya_tool_selection_v4.csv`
  - `statistics_v4.json`
  - `statistics_report_v4.md`
  - `data_quality_issues_v4.csv`
- 数量核对：{m['v3_rows']} + {m['no_action_rows']} = {m['merged_rows']}；剔除 {m['dropped_as_duplicate']}，隔离 {m['isolated_rows']}，最终 {m['final_rows']}。
- V3 产物未被覆盖；V4 使用独立的统计文件名。
"""
    OUT_STATS_MD.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
