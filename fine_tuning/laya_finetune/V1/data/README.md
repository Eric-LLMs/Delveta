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
  - laya
pretty_name: Delveta LayaChoice v1 data
size_categories:
  - 1K<n<10K
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

# Delveta LayaChoice v1 — frozen dataset

This file is the source of the dataset card published at
[`eric-ml-nlp/Delveta-LayaChoice-v1-data`](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v1-data).
It documents the dataset only; it does not change any published row.

## 1. Dataset Purpose

This dataset is the **frozen evaluation and training corpus** for
[`Delveta-LayaChoice-v1`](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1),
a capability-selection decision model used inside Delveta's chat Intent Funnel. When the
intent matcher does not settle the capability, the funnel recalls candidate capabilities
and asks a Laya-style non-autoregressive decision model to pick one. This dataset supplies
the supervised examples and the final benchmark for that decision.

The dataset exists so the fine-tuning result is auditable and reproducible: the exact rows
used for training, validation and the final benchmark are published here with their SHA256
digests. It is not a general-purpose instruction corpus.

## 2. Intended Use and Out-of-Scope Use

**Intended use**

- Training and evaluating a multiple-choice **candidate-capability selector**: given one
  user turn and exactly three candidate capability cards, choose the correct card.
- Reproducing the published LayaChoice-v1 training procedure, checkpoint selection,
  temperature calibration and benchmark.
- Auditing the published benchmark at the row level, because every row is shipped.

**Out-of-scope use**

- **Parameter / argument extraction.** The dataset contains no argument values, no slot
  labels and no extraction targets. Capability selection and argument acquisition are
  separate stages in the product architecture.
- **Tool execution, tool-call synthesis or agent planning.** No tool is called; only a card
  index is predicted.
- **General chat, instruction following, summarization, translation or open-ended QA.**
  The rows are decision items, not dialogue training data.
- **Open-world routing.** The label space is the 17 frozen capabilities listed below.
  Capabilities introduced after the freeze are not representable.
- **Production-traffic accuracy claims.** Accuracy on this set is *benchmark accuracy*
  against a curated synthetic corpus, not production accuracy.
- **Rebalancing or resampling for a different K.** All rows have exactly three options.

## 3. Languages and Task Definition

- **Languages:** Chinese (`zh`) and English (`en`). V1 has no `mixed` language label.
- **Task:** 3-way multiple choice over capability cards.
- **Options per row:** exactly **3** — one gold capability plus two frozen hard-negative
  capabilities. There is no empty option, no padding option and **no `REJECT` option**.
- **Labels:** integers **0, 1 or 2**, indexing `input.options`.
- **Gold is always a real capability.** V1 has no "none of these" class. The two
  distractors are always other real capabilities; the task is to discriminate among
  plausible cards, not to detect absence.
- **Capability roster (17):**

```
cap-add-term          cap-artifact          cap-bash
cap-create-folder     cap-edit-file         cap-mindmap
cap-pdf-extract-text  cap-pdf-table-to-text cap-rag-search
cap-read-document     cap-read-file         cap-slides
cap-social-search     cap-summary           cap-translate
cap-vision            cap-web-search
```

`cap-research` is deliberately excluded from V1.

## 4. Dataset Splits and Calibration

| Split | Rows | `en` | `zh` | File | Role |
|---|---:|---:|---:|---|---|
| train | 856 | 415 | 441 | `data/train.jsonl` | source training corpus |
| validation | 150 | 73 | 77 | `data/val.jsonl` | checkpoint selection **only** |
| test | 900 | 445 | 455 | `data/test.jsonl` | final benchmark **only** |
| total | 1,906 | 933 | 973 | | |

The split is **not stored in the row**. The containing file name is the authority:
`train.jsonl`, `val.jsonl`, `test.jsonl`. In the Hugging Face configuration, `val.jsonl` is
exposed as the `validation` split.

**Calibration.** Temperature fitting uses a fixed holdout inside `train.jsonl`:

- rule: `min(400, n_train // 10)` → `min(400, 856 // 10)` = **85 rows**;
- seed: `20260922`;
- the 85 selected **row indices and message IDs** are recorded in `data/manifest.json`;
- 856 − 85 = **771 effective gradient rows**.

Those 85 rows stay in the published `train.jsonl` but are excluded from gradient updates.
Membership is positional and is re-verified on every run. Calibration is used **only** for
fitting the scalar temperature; it never selects a checkpoint and never gates a result.
Checkpoint selection uses validation top-1 only, with ties resolved to the earlier epoch.
The test split is never used to pick, gate or tune anything.

## 5. Dataset Structure and Four-Field Schema

```text
Delveta-LayaChoice-v1-data/
├── README.md                  # this card
├── SHA256SUMS                 # repository-root checksums (paths include data/)
├── data/
│   ├── SHA256SUMS             # bundle-local checksums (same five digests)
│   ├── manifest.json          # row counts, calibration indices/IDs, source digests
│   ├── train.jsonl            # 856 rows
│   ├── val.jsonl              # 150 rows
│   ├── test.jsonl             # 900 rows
│   └── raw/
│       ├── LayaChoice_v1_validation_raw_v3.jsonl
│       └── LayaChoice_v1_final_test_raw_v3.jsonl
```

Every row is a four-key record. Only `input` (`query` + `options`) and `label` are training
fields; `message_id` and `language` are metadata. The split is not stored in the row; the
containing file name is the authority.

```json
{
  "message_id": "gen-tes-8f43b0c2a252",
  "language": "zh",
  "input": {
    "query": "我刚在技术分享里听到一个挺准的说法，顺手帮我加到团队术语表里。",
    "options": [
      "cap-add-term: ### cap-add-term\ntool: add_term\ndoes: …",
      "cap-create-folder: ### cap-create-folder\ntool: …",
      "cap-edit-file: ### cap-edit-file\ntool: …"
    ]
  },
  "label": 0
}
```

Card bodies are abbreviated above with `…`; the stored strings are complete cards.
Validation is exact-key: a row must have exactly these four keys, `input` must have exactly
`query` and `options`, and the option and label invariants below must hold. The loader
rejects unknown or missing keys rather than ignoring them.

| Key | Type | Present in every row | Used to build model input / target | Meaning |
|---|---|---|---|---|
| `message_id` | string | yes | no — metadata only | Construction sample id; per-split format listed in §6. Never rendered into the model input, target or calibration. |
| `language` | string | yes | no — metadata only | Resolved language: `en` or `zh`. |
| `input` | object | yes | yes — model input | Model input container; exactly `query` and `options`. |
| `label` | integer | yes | yes — scalar class index | 0-based index of the gold option in `input.options`. The selected option is always one of the 17 capability cards. |

All four keys are present in every row, but only `input` and `label` participate in
training. `message_id` and `language` are metadata: they never enter the model input, the
target or calibration. `label` is a slot index, not a capability class.

`input` has exactly two keys, both used by the model:

| Key | Type | Present in every row | Used to build model input / target | Meaning |
|---|---|---|---|---|
| `input.query` | string | yes | yes — request segment | The user request shown to the model. |
| `input.options` | list[string] | yes | yes — one `[MASK]` slot each | The exact option texts, in stored option order; exactly 3 entries. |

Each option string is `"<capability_id>: <card>"`. V1 has no empty card and no `REJECT`
option; all three options are real capability cards.

Three invariants hold for every row:

1. `input.options` has exactly 3 entries, with three distinct capability ids drawn from the
   17-capability roster in §3.
2. `0 <= label < len(input.options)`.
3. The option at `label` is the gold capability card; the other two are hard negatives.

`message_id` is a construction-time sample id (not a fresh uuid4), unique within the bundle.
It is sample metadata: it never enters the model input, the target or calibration.

### 5.1 Adapter view

The training adapter derives these values from a row:

| Derived value | Source |
|---|---|
| `order` | capability id of each option, in stored option order |
| `options` | card text of each option (the part after `"<id>: "`) |
| `gold_index` | `label` |
| `split` | the containing file name (`train` / `val` / `test`) |
| `d1`, `d2` | the frozen hard-negative pair looked up from `gold`; report-only, never model input or target |

`d1` and `d2` are not stored in the row. They are recovered from the gold capability via the
frozen distractor table so the benchmark can report hard-negative and confusion rates; see
§9.

### 5.2 Fields used by training

| Row field | Use |
|---|---|
| `input.query` | Appended to the model input as the request segment |
| `input.options` | Each option is prefixed with one `[MASK]` marker and placed in the head |
| `label` | Scalar class index over the row's three option slots; expanded at training time to a one-hot vector (length 3) |
| `message_id` | Metadata only; not model input, not target |
| `language` | Metadata only; not model input, not target |

The instruction is not a row field. It comes from `manifest.instructions`
(`Which capability should handle the user's request?`), read at render time.

## 6. Field-by-Field Explanation

§5 defines the contract; this section records the per-split values and the audit role of
each field.

| Field | Per-split format / values | Audit use |
|---|---|---|
| `message_id` | train: `corpus-stan-<uuid>` or `corpus-simi-<uuid>`; validation: `gen-val-<12 hex>`; test: `gen-tes-<12 hex>` | Row identity and calibration membership; never model input, target or calibration feature |
| `language` | `zh` or `en` | Per-language reporting only (train 441 zh / 415 en; validation 77 / 73; test 455 / 445) |
| `input.query` | free text | The `state` segment, rendered after the option block |
| `input.options` | exactly 3 strings, `"<capability_id>: <card>"` | The exact model-visible options; the adapter splits id and card |
| `label` | integer `0`, `1` or `2` | Gold slot; one-hot length 3; not a global 0–16 capability vector |
| split | not a field | Injected from the file name: `train.jsonl` / `val.jsonl` / `test.jsonl` |

`laya.common.render_options` renders a choice option as `"%s: %s" % (capability_id, card)`,
so the stored option string **is** the model-visible option. The adapter splits it on the
first `": "` into an id and a card; the renderer joins them back into the identical string.

`manifest.json` is not training data. It records:

- `bench`, `formal_card_view` (`B_noprov`), `instructions`, `option_slots` (3);
- `gold_slot_distribution` and `candidate_order_diverges_from_rendered` per split;
- `rows` and `language` counts per split;
- `calibration` (seed, 85 indices, 85 IDs, effective 771);
- `base_checkpoint` (repo, revision, subfolder);
- `training_schema_keys`, `training_fields`, `not_used_in_training`;
- `source_files` (four SHA256 digests of the construction inputs).

## 7. Candidate Cards and `B_noprov` Representation

Cards are rendered from the **`B_noprov`** view: the production capability card with two
blocks removed:

1. the `query examples:` block (retrieval corpus examples) — removed **in full**;
2. the `evidence:` provenance line — removed.

What remains is the discriminative card body, in this fixed field order:

```text
### <capability_id>
tool: <tool_binding>
does: <description, including the Chinese description>
negative examples:
- <exclusions>
params:
- <parameter list>
```

Every option in the published files contains `###`, `tool:`, `does:`, `negative examples:`
and `params:`, and none contains `query examples:` or `evidence:`.

**Leak protection.** The builder asserts that the user turn does not reach a card through
the card text. It classifies leakage into three channels:

- `description` — counted and reported, not fatal;
- `example` — **fatal** if the user turn appears in an example block;
- `substring` — **fatal** if the user turn appears verbatim in the card body.

Because the `query examples:` block and the `evidence:` line are removed before the
assertion, the label is not spelled out by provenance. Re-running the classifier on the
published bundle gives **0 `example` and 0 `substring` hits** across all 5,718 options;
there are **2 non-fatal `description` hits**, where the curated description quotes a phrase
from the turn.

**Option order.** In 100% of rows the three options are in ascending
`capability_id` order (verified on all 1,906 rows: 856 / 150 / 900). V1 does **not**
shuffle options. The order therefore carries no retrieval rank or difficulty signal, but
it does mean the gold slot is determined by the id order, which is why the gold-slot
distribution is uneven:

| Split | label 0 | label 1 | label 2 |
|---|---:|---:|---:|
| train | 297 | 205 | 354 |
| validation | 52 | 35 | 63 |
| test | 313 | 216 | 371 |
| total | 662 | 456 | 788 |

## 8. Actual Model Input Construction

The model is `laya` (encoder + scoring head), not a generative LLM. Each row becomes one
flat token sequence:

```text
[CLS] choice question: Which capability should handle the user's request? [SEP]
        [MASK] <option 0> [MASK] <option 1> [MASK] <option 2> [SEP]
        <query> [SEP]
```

- The instruction is not a row field. It comes from `manifest.instructions`
  (`Which capability should handle the user's request?`).
- Each option is prefixed with one `[MASK]` marker; marker positions are the scoring
  positions.
- `input.query` is appended **after** the option block as the `state`.
- `label` never appears in the sequence. It is expanded at training time into a one-hot
  vector over the row's three actual marker slots.
- Tokenizer/encoding is the base checkpoint's tokenizer, revision
  `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, subfolder
  `multilingual`.

**Token budget.** Training, evaluation and export share three constants, so the input can
never be rendered differently between them:

| Constant | Value | Meaning |
|---|---:|---|
| `OPTION_MAX_TOKENS` | 256 | per-option hard cap |
| `HEAD_MAX_LEN` | 768 | budget shared by the instruction head and all options |
| `MAX_LEN` | 1024 | whole-sequence ceiling |

Measured on the frozen bundle: 5,718 options (1,906 × 3), **min 136 / median 171 / max 230**
tokens; all options exceed the library default of 48 and none exceeds 256; the instruction
head is 13 tokens; the longest query is 24 tokens; the longest complete sequence is 640
tokens, so `MAX_LEN` never binds.

**Inference parity.** The fine-tuned model is loaded through `laya.Agent`. `install()`
points the library's builder at the same `build_sequence` used here, and the exported
checkpoint records `option_max_tokens` / `head_max_len` / `max_len` in `rl_agent_config.json`.
Scoring is `softmax(logits[:k] / T)` followed by `argmax` — the Agent's own path. A stock
`laya` install (48-token options, `head_max_len=256`) renders a different input and may
score lower; the dataset assumes the published budget.

## 9. Label Construction and Training Target

- `label` is the **slot index** of the gold option, `0..2`.
- At training time `target = [0.0, 0.0, 0.0]; target[label] = 1.0`.
- The loss consumes this one-hot vector over the row's three option slots. It is **not** a
  global 0–16 capability-class vector, and option ids are never embedded as a separate
  classification head.

**`d1` / `d2` are not stored and never enter training.** Each gold capability has a frozen
pair of hard-negative capabilities (`d1`, `d2`) chosen to be semantically near the gold.
Construction scores were gold `1.0` (`matcher_hit`), `d1` `0.62` and `d2` `0.55` (both
`recall`). The pair is a pure function of the gold capability, so it is recovered at load
time from the frozen `DISTRACTORS` table rather than stored in the row. The two distractor
cards are ordinary options, but the **roles** are audit metadata only:

- not part of the model input (the model sees three indistinguishable cards);
- not part of the target;
- not part of calibration;
- used only for the report: hard-negative rate (`1 − top-1`), prediction role
  (gold / `d1` / `d2`), and the `gold → d1` / `gold → d2` confusion rates.

## 10. Data Provenance and Migration from the Legacy Schema

Construction source files (digests recorded in `data/manifest.json`):

| Source | SHA256 | Role |
|---|---|---|
| `LayaChoice_v1_train.jsonl` | `bf4139262b9a398aa2821e55d5abf5a0a28384feeab6d8a85b88334557561fa3` | train rows |
| `LayaChoice_v1_test_v3.jsonl` | `5a001c6a6b02a3c2fbe67b229d61e6c0b0ff023b7d542b9601697c2a48201f96` | test rows |
| `LayaChoice_v1_validation_raw_v3.jsonl` | `974a3b99cf5883c527677373b53539f3ac859ddfb0ba59ea793058d03a6710ff` | validation source (re-rendered) |
| `LayaChoice_v1_final_test_raw_v3.jsonl` | `e0738991d193cae39531a384305dc5d3547ffdd69f9e80b062c91079f51b4613` | final-test source |

The two `raw/` files are the **non-regenerable** artifacts the validation and final-test
bundles were rendered from. They are published because they cannot be recreated from the
live system.

**Legacy ten-column schema.** The rows were migrated from

```text
id, lang, state, order, options, gold, d1, d2, src_sha   (+ split implied by file)
```

to the four-field contract. The encoding changed; the content did not:

| Legacy | Current |
|---|---|
| `id` | `message_id` |
| `lang` | `language` |
| `state` | `input.query` |
| `order` + `options` | `input.options` (same slots, same order, same card text) |
| `gold` | `label` (same 0-based slot) |
| `d1` / `d2` | derived from the gold capability via the frozen table |
| `src_sha` | `manifest.json` → `source_files` |
| split (implicit) | file name (`train.jsonl` / `val.jsonl` / `test.jsonl`) |

All **1,906 / 1,906** runtime rows reconstruct identically on
`(id, split, lang, state, order, options, gold, d1, d2)`, and the frozen distractor table
reproduces the original `(d1, d2)` pair on **1,906 / 1,906** rows. The model input, the
one-hot target, calibration membership and evaluation groupings are unchanged, so the
published benchmark is unaffected.

**Relationship to the retired `.gz` bundles.** The pre-migration ten-column bundles
(`layachoice_v1_{train,val,test}_b_noprov.jsonl.gz`) were removed from this Hugging Face
repository after the four-field bundle was published. They remain, unchanged, in the
dataset's version-control history for traceability:

| Retired file | SHA256 |
|---|---|
| `layachoice_v1_train_b_noprov.jsonl.gz` | `e968fdeddc8f02eec19f37ed29fafc5f05a591ea259c97315639d366e3de4f13` |
| `layachoice_v1_val_b_noprov.jsonl.gz` | `3fa38aa099580b6f9a5146ac2a8e7beed0111d79d2f994adb6ca317547c90364` |
| `layachoice_v1_test_b_noprov.jsonl.gz` | `c665004a127496f4456a0b3ccd81bd0ce10654fef7808f27e45500d3a9d9b8bb` |

No pipeline stage reads these `.gz` files; they are a historical copy, not a second
authority. The four-field JSONL files are the only published training/evaluation copy, and
they are not derived at load time from the `.gz` files.

## 11. Frozen Data, Manifest and SHA256 Verification

The dataset is **frozen**. Any change to the published rows invalidates the published
benchmark and the model's reported score.

| File | SHA256 |
|---|---|
| `data/train.jsonl` | `41e702d72c4b63498c695bbcd867b069b176ea1228b7be15eff8d5df45a39511` |
| `data/val.jsonl` | `a45b4c63631939c5d8181f1cb2e6a22ea43d3d6ebda5d01d7de66681bcc5490b` |
| `data/test.jsonl` | `be2cdc95fbef658ecf7ce3b388322eb9bdde0d40610a347988ce988beace29b9` |
| `data/raw/LayaChoice_v1_validation_raw_v3.jsonl` | `974a3b99cf5883c527677373b53539f3ac859ddfb0ba59ea793058d03a6710ff` |
| `data/raw/LayaChoice_v1_final_test_raw_v3.jsonl` | `e0738991d193cae39531a384305dc5d3547ffdd69f9e80b062c91079f51b4613` |

Both checksum files cover the same five artifacts; only the path prefix differs.

```bash
# from the repository root
sha256sum -c SHA256SUMS

# from data/ — the same digests, bundle-local paths
cd data && sha256sum -c SHA256SUMS
```

Expected output is five `OK` lines per check. `data/manifest.json` is the machine-readable
record of row counts, language counts, gold-slot distribution, calibration indices and IDs,
base checkpoint revision and construction source digests.

**Published benchmark on this frozen test split** (fine-tuned V1 checkpoint, temperature
fitted on the 85 calibration rows):

| Metric | Value |
|---|---:|
| Top-1 accuracy (900 rows) | 0.935556 |
| Top-1, English (445 rows) | 0.914607 |
| Top-1, Chinese (455 rows) | 0.956044 |
| ECE on `max(p)` | 0.029528 |
| Predictions on gold / `d1` / `d2` | 842 / 42 / 16 |
| `gold → d1` / `gold → d2` | 0.046667 / 0.017778 |

These numbers are benchmark accuracy on a curated 900-row artifact; see §13.

## 12. Relationship to the V1 Model

| | |
|---|---|
| Base checkpoint | `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, subfolder `multilingual` |
| Fine-tuned model | [`eric-ml-nlp/Delveta-LayaChoice-v1`](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1) |
| Training checkpoints | [`eric-ml-nlp/Delveta-LayaChoice-v1-checkpoints`](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1-checkpoints) |

The published model was trained on the 771 effective rows (856 − 85 calibration), selected
by validation top-1, and benchmarked once on the 900 test rows. The model repository is
the deployment artifact; this dataset repository is the auditable data record.

The frozen distractor roles (`d1` / `d2`) are recovered from the gold capability at load
time. They are not stored, are never part of the model input or the training target, and
are used only for the benchmark's hard-negative and confusion reporting.

## 13. Limitations

- **Fixed `K = 3`.** Behaviour for `K = 2` or post-truncation `K >= 4` is not represented.
- **Closed label space.** 17 capabilities frozen at construction time. Capabilities added
  later — including `cap-research` — cannot be selected or evaluated.
- **No abstain class.** V1 cannot express "no candidate is correct".
- **Deterministic option order.** Options are sorted by capability id, so position is
  correlated with id and the gold-slot distribution is uneven. Do not treat slot frequency
  as a prior calibrated on shuffled data.
- **Synthetic / curated corpus.** Rows were authored for this task, not sampled from live
  product traffic. The corpus therefore carries no distributional guarantee with respect
  to production requests.
- **Hard negatives by construction.** Distractors are semantically near the gold
  capability, which is the intended difficulty; accuracy is not comparable to a random
  distractor benchmark.
- **Test size.** 900 rows; per-language and per-capability slices are small, so subgroup
  differences should be read with sampling error in mind.
- **Benchmark accuracy ≠ production accuracy.** The test split is a fixed curated artifact
  over known capabilities with a known hard-negative construction.

## 14. Citation and Related Repositories

If you use this dataset, cite the model and link the experiment record:

```bibtex
@misc{delveta_layachoice_v1_2026,
  title        = {Delveta LayaChoice v1: a frozen capability-selection dataset and fine-tuned Laya decision model},
  author       = {Delveta},
  year         = {2026},
  howpublished = {Hugging Face},
  url          = {https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v1-data}
}
```

No license is asserted for this dataset. The `laya` **software package** is Apache-2.0, but
the license of the base **model weights** (`convaiinnovations/laya`) could not be
determined at release time. Downstream use should confirm terms before redistribution.

Related resources:

- Fine-tuned model: [`eric-ml-nlp/Delveta-LayaChoice-v1`](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1)
- Training checkpoints: [`eric-ml-nlp/Delveta-LayaChoice-v1-checkpoints`](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1-checkpoints)
- Experiment document: [`docs/experiments/LayaChoice-v1-Fine-Tuning.md`](https://github.com/Eric-LLMs/Delveta/blob/main/docs/experiments/LayaChoice-v1-Fine-Tuning.md)
