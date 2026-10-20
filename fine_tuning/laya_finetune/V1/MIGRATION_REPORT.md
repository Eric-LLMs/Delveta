# LayaChoice-v1 four-field migration report

Scope: the **data contract only**. V1 keeps its 3-way choice task (gold + two frozen
distractors, `K = 3`), its label rule, its calibration split, its recipe and its evaluation
metrics. Nothing from V2's algorithm was adopted.

## Field mapping

| Before (10 columns) | After (4 fields) | How it is preserved |
|---|---|---|
| `id` | `message_id` | value copied verbatim |
| `~~split~~` | — | file name (`train.jsonl` / `val.jsonl` / `test.jsonl`) is injected by the loader |
| `lang` | `language` | value copied verbatim |
| `state` | `input.query` | value copied verbatim |
| `order` + `options` | `input.options` | one `"<capability_id>: <card>"` string per slot, same order; the adapter splits it back into `order` and the card text |
| `gold` | `label` | 0-based slot index, value copied verbatim |
| `d1` / `d2` | — | pure function of the gold capability; the loader derives them from the frozen distractor table (verified below) |
| `src_sha` | — | the frozen source hashes stay in `manifest.json` (`source_files`) |

The pre-migration gzip bundles are retained under `data/` for traceability; no pipeline
stage reads them.

## Preservation evidence

All rows, all splits:

| split | rows before | rows after | zh / en | label 0 / 1 / 2 | option count | duplicate `message_id` | duplicate content | label anomalies |
|---|---|---|---|---|---|---|---|---|
| train | 856 | 856 | 441 / 415 | 297 / 205 / 354 | 3 (856/856) | 0 | 0 | 0 |
| val | 150 | 150 | 77 / 73 | 52 / 35 / 63 | 3 (150/150) | 0 | 0 | 0 |
| test | 900 | 900 | 455 / 445 | 313 / 216 / 371 | 3 (900/900) | 0 | 0 | 0 |

Label anomalies means: `input.options[label]` is not the option whose capability id equals
the pre-migration `order[gold]`. Zero across 1,906 rows.

**Runtime row identity.** `layachoice_v1_dataset.load_bundle` was compared against the
pre-migration bundle row by row on
`(id, split, lang, state, order, options, gold, d1, d2)`:

| split | rows compared | identical |
|---|---|---|
| train | 856 | 856 |
| val | 150 | 150 |
| test | 900 | 900 |

**Distractor recovery.** `DISTRACTORS[gold_capability]` reproduces the stored `(d1, d2)`
capability pair on 1,906 / 1,906 rows (0 mismatches), so removing the two columns loses no
business information.

## Evaluation parity

The renderer, the one-hot target, the calibration membership and the evaluation groupings
read only `state`, `order`, `options`, `gold`, `split`, `lang` and `id` — all reproduced
exactly. The model input is therefore unchanged, and no V1 model was retrained.

The frozen V1 benchmark record (Final Test v3, 900 rows, best epoch 2) remains the
reference and is not recomputed here:

| Metric | Value |
|---|---|
| Top-1 | 0.935556 |
| EN (445) / ZH (455) | 0.914607 / 0.956044 |
| ECE | 0.029528 |
| Predicted gold / d1 / d2 | 842 / 42 / 16 |
| gold→d1 / gold→d2 | 0.046667 / 0.017778 |

These metrics are not directly re-runnable in this change because they require the trained
checkpoint and a GPU; they are unchanged by construction because the per-row decision inputs
are identical.

## Files

Changed:

- `layachoice_v1_dataset.py` (new) — four-field adapter, manifest-checked loader, frozen
  distractor table.
- `layachoice_finetune.py` — imports the dataset contract; the local ten-column reader is
  gone.
- `layachoice_prepare.py` — emits four-field JSONL and the new manifest keys.
- `layachoice_eval.py` — unchanged code; it consumes the same loader.
- `data/{train,val,test}.jsonl` — the four-field bundle.
- `data/manifest.json`, `data/SHA256SUMS` — schema keys and new hashes.
- `README.md` — data-format section and file map.
- `tests/test_laya_v1_dataset_contract.py` (new) and
  `tests/test_cap_router_service.py` — contract, label mapping, counts and lossless
  equivalence.

Unchanged: the V2 tree, the Hugging Face repositories, `run.sh`, the training recipe, the
evaluation code paths and the retained pre-migration bundles.
