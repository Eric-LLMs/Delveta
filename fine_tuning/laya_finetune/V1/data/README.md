---
language:
  - zh
  - en
task_categories:
  - text-classification
tags:
  - capability-selection
  - tool-routing
  - decision-model
  - frozen-benchmark
  - synthetic
pretty_name: Delveta LayaChoice v1 data
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/val.jsonl
      - split: test
        path: data/test.jsonl
---

> **本版本系统弃用，新版本见 [Delveta-LayaChoice-v2-data](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-data)。**

# Delveta LayaChoice v1 — frozen dataset

## Purpose

This dataset is the **frozen evaluation and training corpus** for
[`Delveta-LayaChoice-v1`](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1),
a capability-selection decision model used inside Delveta's chat Intent Funnel.
It exists so that the fine-tuning result is auditable and reproducible: the exact
rows used for training, validation and the final benchmark are published here with
their SHA256 digests.

## Task

Each row is a **capability-selection (multiple-choice) example**:

- one **user turn** (the "state") in Chinese or English;
- **three candidate capability cards** (`K = 3`), rendered from the `B_noprov` view;
- a single **gold option** index.

The model must choose the gold capability. Cards come from a fixed roster of **17
capabilities**:

```
cap-add-term          cap-artifact          cap-bash
cap-create-folder     cap-edit-file         cap-mindmap
cap-pdf-extract-text  cap-pdf-table-to-text cap-rag-search
cap-read-document     cap-read-file         cap-slides
cap-social-search     cap-summary           cap-translate
cap-vision            cap-web-search
```

## Splits

| Split | Rows | File | Role |
|---|---|---|---|
| train | 856 | `data/train.jsonl` | source training corpus |
| validation | 150 | `data/val.jsonl` | checkpoint selection **only** |
| test | 900 | `data/test.jsonl` | final benchmark **only** |

The training corpus of 856 rows is split further at model-training time:
**85** rows are held out for temperature calibration (fixed IDs, seed `20260922`),
leaving **771** effective training rows. Those 85 calibration rows are part of the
856 published here; they were never seen by the optimizer. The calibration indices
and IDs are recorded in `data/manifest.json`.

The pre-migration ten-column `.gz` bundles were retired from this repository after the
four-field bundle was published. The four-field JSONL files above are the only published
copy; the original bundles remain in the GitHub repository history for traceability.

`raw/` holds the two **non-regenerable** source artifacts the bundles were rendered
from:

- `raw/LayaChoice_v1_validation_raw_v3.jsonl`
- `raw/LayaChoice_v1_final_test_raw_v3.jsonl`

## Dataset Schema

Every row is a four-key record. Only `input` (`query` + `options`) and `label` are
training fields; `message_id` and `language` are metadata. The split is not stored in
the row; the containing file name is the authority.

```json
{"message_id": "<sample id>", "language": "zh|en",
 "input": {"query": "<user turn>",
           "options": ["<capability_id>: <card>", "<capability_id>: <card>", "<capability_id>: <card>"]},
 "label": 0}
```

| Key | Type | Present in every row | Used to build model input / target | Meaning |
|---|---|---|---|---|
| `message_id` | string | yes | no — metadata only | Construction sample id (`corpus-stan-*` / `corpus-simi-*` / `gen-val-*` / `gen-tes-*`), unique within each split; never enters the model input, target or calibration |
| `language` | string | yes | no — metadata only | `zh` or `en` |
| `input` | object | yes | yes — model input | Model input container; exactly `query` and `options` |
| `label` | integer | yes | yes — scalar class index | 0-based index of the gold option in `input.options`; the selected option is always a capability card |

`input` has exactly two keys, both used by the model:

| Key | Type | Present in every row | Used to build model input / target | Meaning |
|---|---|---|---|---|
| `input.query` | string | yes | yes — request segment | The user request shown to the model |
| `input.options` | list[string] | yes | yes — one `[MASK]` slot each | The exact option texts, in stored option order; exactly 3 entries |

Each option string is `"<capability_id>: <card>"`. V1 has no empty card and no
`REJECT`; all three options are real capability cards.

Three invariants hold for every row:

1. `input.options` has exactly 3 entries, with three distinct capability ids from the
   17-capability roster above.
2. `0 <= label < len(input.options)`.
3. The option at `label` is the gold capability; the other two are frozen hard negatives.

`message_id` is a construction-time id, not a fresh uuid4, and is unique within each
split. It is sample metadata: it never enters the model input, the target or calibration.

`laya.common.render_options` renders a choice option as
`"<capability_id>: <card>"`, so the stored option string **is** the model-visible
option. The frozen distractor roles (`d1` / `d2`, scores `0.62` / `0.55`) are a
pure function of the gold capability and are therefore not stored; they are
derived from the gold capability in the released
[`layachoice_v1_dataset.py`](https://github.com/Eric-LLMs/Delveta/blob/main/fine_tuning/laya_finetune/V1/layachoice_v1_dataset.py).

### Adapter view

The training adapter derives these values from a row:

| Derived value | Source |
|---|---|
| `order` | capability id of each option, in stored option order |
| `options` | card text of each option (the part after `"<id>: "`) |
| `gold_index` | `label` |
| `split` | the containing file name (`train` / `val` / `test`) |
| `d1`, `d2` | the frozen hard-negative pair looked up from `gold`; report-only, never model input or target |

### Fields used by training

| Row field | Use |
|---|---|
| `input.query` | Appended to the model input as the request segment |
| `input.options` | Each option is prefixed with one `[MASK]` marker and placed in the head |
| `label` | Scalar class index over the row's three option slots; expanded at training time to a one-hot vector (length 3) |
| `message_id` | Metadata only; not model input, not target |
| `language` | Metadata only; not model input, not target |

The instruction is not a row field. It comes from `manifest.instructions`
(`Which capability should handle the user's request?`), read at render time.

### Final model input

The renderer builds one flat token sequence per row:

```text
[CLS] choice question: <instructions> [SEP]
        [MASK] <option 0> [MASK] <option 1> [MASK] <option 2> [SEP]
        <query> [SEP]
```

- `input.options` are emitted in their stored order, one `[MASK]` marker per option.
- `input.query` is appended after the option block.
- `label` never appears in the sequence; it is expanded to a one-hot vector over the
  row's three option slots, not a global 0–16 capability vector.
- Each option is capped at 256 tokens, the head and options share 768 tokens, and the
  whole sequence is capped at 1024. Measured on this bundle, options are 136–230 tokens
  (median 171) and the longest complete sequence is 640, so nothing is truncated.

## Data contract migration

The rows were migrated from a ten-column record (`id`, `lang`, `state`, `order`,
`options`, `gold`, `d1`, `d2`, `src_sha`, no stored split) to the four-field
contract above. This changed the **encoding only**:

- `id` → `message_id`; `lang` → `language`; `state` → `input.query`;
- `order` + `options` → `input.options` (same slots, same order, same card text);
- `gold` → `label` (same 0-based slot);
- `d1` / `d2` are derived from the gold capability; `src_sha` stays in
  `data/manifest.json`.

All 1,906 runtime rows reconstruct identically on `(id, split, lang, state, order,
options, gold, d1, d2)`, and the frozen distractor table reproduces the stored
`(d1, d2)` pair on 1,906 / 1,906 rows. The model input, the one-hot target, the
calibration membership and the evaluation groupings are unchanged, so the published
benchmark is unaffected. Full evidence: `MIGRATION_REPORT.md` in the GitHub
repository.

## Frozen status

The dataset is **frozen**. Any change to the published rows invalidates the
published benchmark. Verify with:

```bash
sha256sum -c SHA256SUMS            # repository-wide
cd data && sha256sum -c SHA256SUMS # bundle-local, same digests
```

| File | SHA256 |
|---|---|
| `data/train.jsonl` | `41e702d72c4b63498c695bbcd867b069b176ea1228b7be15eff8d5df45a39511` |
| `data/val.jsonl` | `a45b4c63631939c5d8181f1cb2e6a22ea43d3d6ebda5d01d7de66681bcc5490b` |
| `data/test.jsonl` | `be2cdc95fbef658ecf7ce3b388322eb9bdde0d40610a347988ce988beace29b9` |
| `data/raw/LayaChoice_v1_validation_raw_v3.jsonl` | `974a3b99cf5883c527677373b53539f3ac859ddfb0ba59ea793058d03a6710ff` |
| `data/raw/LayaChoice_v1_final_test_raw_v3.jsonl` | `e0738991d193cae39531a384305dc5d3547ffdd69f9e80b062c91079f51b4613` |

Source-file digests (the artifacts the bundles were rendered from) are recorded in
`data/manifest.json` and in the model's experiment document.

## Data construction notes

- Cards are rendered from the **`B_noprov`** view: a capability's description
  **without** any `evidence:` line or provenance that would reveal the label. The
  same view is used for training, validation and the benchmark, so comparisons are
  like-for-like.
- **Exclusion / leakage prevention.** A row's card text is asserted **not** to
  contain the user turn verbatim; ingest hard-fails with an `ExclusionSetError` if
  the check is violated. Train / validation / test are disjoint by construction, and
  the 85 calibration rows are held out of the optimizer by explicit ID.
- The corpus is **synthetic / curated** — it was authored for this task, not sampled
  from live product traffic. Hard negatives (`d1`, `d2`) are chosen to be
  semantically near the gold capability, which is the intended difficulty of the
  benchmark.

## Known benchmark limitations

- The test split is a fixed, curated 900-row artifact over **17 known
  capabilities** with a known hard-negative construction. Accuracy on it is
  **benchmark accuracy** — **not** production-traffic accuracy and **not** open-world
  generalization accuracy.
- The candidate count is fixed at `K = 3`. Behaviour for `K = 2` or post-truncation
  `K >= 4` is not represented.
- Capabilities introduced after the freeze are out of scope.
- Because the corpus is synthetic, it carries no distributional guarantee with
  respect to live user requests.

## Licensing status

No license is asserted for this dataset. The `laya` **software package** is
Apache-2.0, but the license of the base **model weights**
(`convaiinnovations/laya`) could not be determined at release time. Downstream use
should confirm terms before redistribution.

## Reference

Full method, ablations, baselines and invalid runs:
[`docs/experiments/LayaChoice-v1-Fine-Tuning.md`](https://github.com/Eric-LLMs/Delveta/blob/main/docs/experiments/LayaChoice-v1-Fine-Tuning.md).

Related resources:

- Model: [`eric-ml-nlp/Delveta-LayaChoice-v1`](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1)
- Scripts: [`fine_tuning/laya_finetune/V1`](https://github.com/Eric-LLMs/Delveta/tree/main/fine_tuning/laya_finetune/V1)
- Checkpoints: [`eric-ml-nlp/Delveta-LayaChoice-v1-checkpoints`](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1-checkpoints)
