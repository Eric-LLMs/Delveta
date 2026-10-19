# LayaChoice V4 (`full_r1`) build record

This record describes the construction of the official V4 dataset. Paths are
written as they were at build time; the post-build archive layout is noted at
the end.

## 1. Run

- Run id: `full_r1`
- Mode: `full`
- Entry point: `scripts/laya_finetune/data_v4_build/full_r1/scripts/build_dataset.py`
- Inferred command (the exact shell history was not recorded):

```
python scripts/laya_finetune/data_v4_build/full_r1/scripts/build_dataset.py --mode full --run-id full_r1
```

`generation.log` records the run:

```
run_id=full_r1 mode=full
source rows=49241 sha=e39adab214ed26992e26728a9e4c73659d5c924d283454d841abd789187c8445
eligible=48333 excluded=908
selected=48333
embedded 48333 queries in 1912.06s
recall: 48333 queries in 101.81s
ANN exact audit: missing>=0.82=34593, max cap gap=0.16702373
wrote three threshold versions
```

## 2. Inputs

| Input | Size (bytes) | SHA-256 |
|---|---:|---|
| `laya_tool_selection_v4.csv` | 11,996,876 | `e39adab214ed26992e26728a9e4c73659d5c924d283454d841abd789187c8445` |
| `laya_tool_selection_v3.csv` | 11,597,746 | `47149f0744d30501563717a0164fda9e518034e1c6c52b6c71580ba317110290` |
| `no_action_queries_v1.csv` | 417,255 | `bce032d0dbfdd9f5061fb0310373654d5831cdabe90b5f2f28175c19d7340e00` |
| `laya_chat_tool_selection_balanced_18k_v4_params_filled_v2.csv` | 4,751,438 | `d74b30969589402ddc42a395c671dedaeb4d6e9de8ab30f5f534b569b22b7e32` |
| `query_list_finetune_v1.csv` | 3,724,326 | `c0f65430714a34be1adab3666516bb7a998d1a184bd98047146b1830ab20f802` |

External runtime inputs (not captured in this repository):

- PostgreSQL/pgvector database. Default DSN:
  `postgresql://delveta:delveta@localhost:15432/delveta` (`DELVETA_DB_DSN`).
  The recall corpus is the live capability standard/similar query tables.
- Embedding service. Default URL: `http://localhost:18080/embed`
  (`DELVETA_EMBED_URL`); the model is read from `app_settings.embedding_profile`.
- Repository app code at commit `fdec472` (HEAD when this record was written),
  including `core.application.chat.intent_funnel.cap_router.card_renderer` and
  `core.application.chat.intent_funnel.registry.entry`.

## 3. Fixed construction parameters

| Parameter | Value |
|---|---|
| Thresholds | `0.82`, `0.85`, `0.88` |
| `K_MAX` | 3 capability cards |
| Option slots | `K + 1`, range 1..4 |
| ANN pool per lane | 64 |
| Evidence floor | 0.80 (scores below it are not written) |
| Split allocation seed (`SEED`) | 424242 |
| Row shuffle seed prefix (`ROW_SEED`) | `reject-rows-v1` |
| Per-row option shuffle prefix | `reject-shuffle-v1` |
| Calibration seed | 20260922 |
| Calibration max rows | 400 |
| `part` | `p2` for every row |
| Card view | `B_noprov` |

Split allocation: stratified by source gold label x resolved language; n=1 ->
train; n=2 -> train/test; n>=3 -> val=`max(1, round(0.05n))`,
test=`max(1, round(0.20n))`, train=remainder. Query families are allocated as
indivisible units and never cross splits. The same split assignment is reused
by all three thresholds.

Label precedence: source No-Action -> REJECT; K=0 -> REJECT; gold capability
inside the Top 3 -> that capability; otherwise REJECT.

## 4. Repair and official-version unification

The official bundle is the in-place K=0 -> K=1 repair, not the superseded
`full_r1_repaired` copy.

- Target rule: `K=0`, `is_action=true`, `candidate_count=0`, and the source gold
  is a valid capability.
- Action: inject the source gold as the only capability candidate, keep REJECT,
  set K=1 and `target_kind=capability`.
- Seed prefix: `k0-to-k1-inplace-v1:<threshold>:<split>` over the row id.
- Repaired rows: `0.82=152`, `0.85=603`, `0.88=2466`.
- `candidate_origin=source_gold_injected`; `retrieval_score_available=false`.
- `construction_audit.jsonl` intentionally preserves the pre-repair evidence.
- Scripts: `repair/v4_k0_to_k1_inplace.py`,
  `repair/verify_k0_to_k1_inplace.py`,
  `repair/unify_v4_official_version.py`.

Manifest variant: `V4-dynamic-official-r1`.

## 5. Outputs

Every threshold contains 48,333 rows: train 36,197, val 2,407, test 9,729;
400 calibration rows and 35,797 effective training rows.

| Threshold | train | val | test | construction lines | direct reject lines |
|---|---:|---:|---:|---:|---:|
| 0.82 | 82,353,142 | 5,518,900 | 22,235,374 | 48,333 | 1,208 |
| 0.85 | 67,703,160 | 4,483,754 | 18,239,534 | 48,333 | 2,541 |
| 0.88 | 60,442,968 | 4,017,178 | 16,297,860 | 48,333 | 5,130 |

Shared files: `capability_ids.json`, `ann_coverage_audit.json`,
`boundary_cases.jsonl`, `capability_max_scores.npy`,
`query_embedding_ids.json`, `query_embeddings.npy`,
`sample_manifest.csv`, `shared_retrieval_evidence.jsonl`.

Full file-level SHA-256 inventory:
`inventory/v4_data_files.json`.

## 6. Code and dependency versions

- Repository HEAD: `fdec472`
- `build_dataset.py`: 105,053 bytes,
  SHA-256 `9a882b556e4a06e7527e22847a450598319655c442dec8af0a5dbc46125fc4d5`
- Python: 3.13.11
- Installed at record time: `asyncpg 0.31.0`, `httpx 0.28.1`, `numpy 2.5.3`
- The build ran from the repository's app environment. The V2 fine-tuning
  `requirements.txt` is unrelated to this construction stage.

## 7. Known gaps and caveats

These gaps are reported, not fixed. Fixing them would require code or data
changes outside this backup task.

1. `full_r1/config/generation_config.json` is missing. Only
   `smoke50_r1/config/generation_config.json` exists; it is included under
   `reference/` as a schema example, not as the full-run config.
2. `full_r1/manifests/generation_manifest.json` and
   `full_r1/manifests/source_manifest.json` are missing. The same files exist
   only for `smoke50_r1`.
3. `full_r1/reports/threshold_comparison_082_085_088.json` is missing; the
   Markdown comparison report exists.
4. `build_dataset.py` points `SOURCE_CSV` at
   `scripts/laya_finetune/data_v3/laya_tool_selection_v2.csv`: the filename is
   synced with the renamed copy, but the directory prefix and `REPO_ROOT` are
   still pre-move. Its `THRESHOLD_TAGS` map still uses `threshold_082/085/088`.
   The source CSV lives in `data/laya_finetune/data_v4_build/data_v3/` and in
   `inputs/` under the renamed name, and the generated directories were renamed
   to `recall_threshold_*`. Running the script as archived will fail until the
   path and tag constants are repointed.
5. `repair/v4_k0_to_k1_inplace.py` and
   `repair/verify_k0_to_k1_inplace.py` still use the old `threshold_*` tags and
   reference the superseded `full_r1_repaired` directory. They are preserved
   as executed, not rewritten.
6. `repair/unify_v4_official_version.py` also uses the old tags and the
   pre-move `scripts/laya_finetune/data_v4_build` base path.
7. `audit/v4_schema_stats.py` targets `smoke50_r1`; the other audit scripts
   contain base paths from before the archive move. They are read-only audit
   tools and are not part of the generation entry point.
8. The exact shell command, the database snapshot and the embedding-model
   revision were not recorded. Recall therefore cannot be reproduced from
   this repository alone.
9. The `full_r1` sample manifest and construction audit record the pre-repair
   state where documented; repaired target rows are reconciled through the
   repair detail and backup files under the audit archive.

## 8. Post-build archive layout

The complete original build tree is archived at
`data/laya_finetune/data_v4_build/full_r1/`. The upstream V3/V4 source-CSV
build scripts, statistics and intermediate CSVs are archived at
`data/laya_finetune/data_v4_build/data_v3/`. The dataset copy installed for
the training layout is `scripts/laya_finetune/V2/data/`. The copies are
independent; no symlinks are used.
