# Laya Tool Router V3 dataset

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

- `data/BaseData/laya_chat_tool_selection_balanced_18k_v4_params_filled_v2.csv` (18000 rows)
- `data/BaseData/query_list_finetune_v1.csv` (29241 rows)

## Processing

1. Read both CSVs without modification.
2. Normalize queries (NFKC, whitespace collapse, casefold) for duplicate detection only.
3. Detect exact duplicates, query duplicates, label conflicts, parameter conflicts.
4. Isolate any conflicting query group; never pick one side silently.
5. Merge one row per query, preferring a non-empty parameter object.
6. Row-level shuffle with seed `20261018`.
7. Write outputs and verify counts, multiset identity, and input hashes.

## Result

- 47241 merged rows -> 47241 final rows
- 0 isolated rows
- 22669 rows with a non-empty parameter object

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
| Fixed-seed row-level shuffle | PASS | seed `20261018`, multiset identical |
| Statistics match output | PASS | Row and capability counts recomputed from the CSV |
| Statistics reproducible | PASS | `build_v3_dataset.py` regenerates every artifact |
| All new files under `data_v3/` | PASS | See file list above |
| No model training executed | PASS | Builder is data-only |

## Reproduce

```
python scripts/laya_finetune/data_v3/build_v3_dataset.py
```
