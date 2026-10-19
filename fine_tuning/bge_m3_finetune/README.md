# Delveta BGE-M3 Fine-Tuning

Source package for **`Delveta-BGE-M3-v1`**, a Delveta-specific dense retrieval
embedder built by fine-tuning [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3).

> **Status: training complete; post-finetuning retrieval evaluation open.**
> Training artifacts are frozen and the inference variants are published. No
> Recall@k, MRR, or capability-retrieval result has been measured for this model
> yet. See [Evaluation and Limitations](#10-evaluation-and-limitations).

## 1. Project Overview

This project fine-tunes the BGE-M3 dense encoder for Delveta's capability and
semantic retrieval paths:

- **Capability retrieval** — embed a user query and the capability corpus so the
  downstream Intent Funnel / router can build a candidate set.
- **Semantic retrieval** — embed knowledge-base chunks and queries for dense
  similarity search.

The output is a **dense embedding model, not a classifier**. It does not emit a
capability label, does not contain a classification head, and is not a
multi-task model. It is one contrastive embedder with one dense objective.

The fine-tune is deliberately narrow: it targets the Delveta capability corpus
and is not a general-purpose replacement for `BAAI/bge-m3`.

## 2. Model

| | |
|---|---|
| Published model | [`eric-ml-nlp/Delveta-BGE-M3-v1`](https://huggingface.co/eric-ml-nlp/Delveta-BGE-M3-v1) |
| Base model | `BAAI/bge-m3` |
| Base revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| Architecture | XLM-RoBERTa-large |
| Embedding | 1024-dimensional dense vector |
| Pooling / normalization | CLS pooling + L2 normalization (`normalize(last_hidden_state[:, 0])`) |
| Heads | Dense only — no sparse head, no ColBERT head, no classifier |
| Published variants | FP32 (master), FP16, INT8 (ONNX, CPU) |

The Hugging Face repository stores the three variants in parallel subfolders
(`FP32/`, `FP16/`, `INT8/`), with no model copy at the repository root. Load a
PyTorch variant with `subfolder="FP32"` or `subfolder="FP16"`; load the
quantized variant with ONNX Runtime.

## 3. Training Objective

Training uses **one unified contrastive objective**:

```text
query -> positive target text -> negative target texts
```

A positive target is either:

1. a capability's canonical registry description, or
2. a standard/similar corpus query for the gold capability.

Both are the **same** contrastive task for the embedder. There is no separate
Task A / Task B loss, no classification objective, and no task-conditioned
forward pass. The `task` manifest field is provenance only and is never read by
the training code.

Key properties, as implemented in this directory:

- **Dense-only training.** `unified_finetuning=False` is pinned in
  [`train_bge_m3_dual_target.py`](train_bge_m3_dual_target.py). The sparse and
  ColBERT heads are not constructed or trained.
- **Default in-batch negatives are enabled.** The training file has no
  `.no_in_batch_neg` suffix; `uses_in_batch_negatives()` mirrors FlagEmbedding's
  filename rule and refuses a file that would disable them.
- **Capability-unique batching.** At most one training row per
  `gold_capability_id` appears in a batch. This prevents two near-duplicate rows
  of the same capability from treating each other's positive as an in-batch
  negative. `gold_capability_id` is sampling metadata only and never enters the
  loss.
- **One positive per record.** The FlagEmbedding collator samples exactly one
  positive from each row's `pos` list; the two positive families are stored as
  separate physical records that share only the query text.
- **Official loss and checkpoint path.** The trainer is
  `CapabilityUniqueSamplerMixin` + FlagEmbedding 1.4.2's
  `EncoderOnlyEmbedderM3Trainer`; the loss (`_compute_in_batch_neg_loss`,
  temperature `0.02`) and checkpoint layout are not reimplemented here.

Dense contrastive loss, for a batch of `B` queries and `G = train_group_size`
passages per query group:

```text
s_ij = (q_i · p_j) / tau
L    = mean_i CE(s_i, y_i)
```

where `q_i` and `p_j` are CLS-pooled, L2-normalized embeddings, `tau = 0.02`,
and `y_i` is the positive passage for query `i`.

## 4. Project Structure

All paths are relative to this directory.

| Path | Purpose |
|---|---|
| [`run.sh`](run.sh) | One-command entry with the pinned formal recipe, dependency/GPU checks, Dataset V1 alignment check, and an output-directory guard. Supports `--smoke` for a CPU-only plumbing check. |
| [`train_bge_m3_dual_target.py`](train_bge_m3_dual_target.py) | Real training entry. Validates the data/manifest, builds the FlagEmbedding M3 model/trainer, injects the capability-unique sampler, and starts training. |
| [`capability_unique_sampler.py`](capability_unique_sampler.py) | `CapabilityUniqueBatchSampler`: emits explicit index batches with at most one row per capability; covers every index exactly once and never drops the tail batch. |
| [`capability_unique_trainer.py`](capability_unique_trainer.py) | `CapabilityUniqueSamplerMixin`: overrides `_get_train_sampler` so the sampler reaches the real `Trainer` dataloader. Keeps algorithm and Trainer wiring separate. |
| [`dual_target_alignment.py`](dual_target_alignment.py) | Strict row-alignment check between a data file and its manifest: row count, required keys, and per-row `sha256(query) == query_sha256`. Read-only; usable as a CLI. |
| [`build_dual_target.py`](build_dual_target.py) | Dataset builder that merged the query-anchor family and the capability-description family into Dataset V1. **Writes over the data files**; see [Dataset Preparation](#6-dataset-preparation). |
| [`verify_dual_target.py`](verify_dual_target.py) | Recomputes the coverage, language, leakage, and integrity claims of the dual-target audit. Writes `data/_verify_dual_target_report.json`. |
| [`final_gate_audit.py`](final_gate_audit.py) | Final pre-training gate check (Task B byte identity, Task A schema, coverage, negative quality, batch false-negative risk). Writes `data/_final_gate_report.json`. |
| [`audit_dual_target.py`](audit_dual_target.py) | Audits the data against the two retrieval chains (Agent: query -> capability description; Intent Funnel: query -> standard/similar corpus). Prints a report; writes no data. |
| [`audit_dataset.py`](audit_dataset.py) | Historical step-1 audit for the **pre-merge** Query -> Query dataset. Its "every positive must be definitional" check no longer matches the merged dual-target file; it is not a current gate. |
| [`test_capability_unique_sampler.py`](test_capability_unique_sampler.py) | Offline unit tests for the sampler through a real `torch.utils.data.DataLoader`, including multi-epoch, world-size 1/2, tail batches, and the frozen manifest. |
| [`test_dual_target_training_entry.py`](test_dual_target_training_entry.py) | Offline tests for the training entry: alignment, sampler injection through a real `Trainer`, FlagEmbedding 1.4.2 model/trainer wiring, dense-only forward, bf16 propagation, and in-batch-negative flags. |
| [`data/`](data/) | Frozen Dataset V1 (data, manifests, split files), machine-readable audit reports, and the data design/audit documents. |

## 5. Requirements

The following library versions are recorded for the formal run. No
`requirements.txt` or lockfile is committed for this project; the local
test/training environment is `.venv-bge-train`.

| Component | Recorded version |
|---|---|
| Python | 3.11 (local `.venv-bge-train`: CPython 3.11.15) |
| PyTorch | 2.14.0 (`2.14.0+cu130` on the GPU host) |
| FlagEmbedding | 1.4.2 |
| Transformers | 5.17.0 |
| tokenizers | 0.23.2 |
| datasets | 5.0.1 |
| accelerate | 1.15.0 |
| sentence-transformers | 6.1.0 |
| peft | 0.21.2 |

The Python version is verified for the local `.venv-bge-train`; the experiment
record documents the library versions used by the formal GPU run.

`transformers` 5.17.0 pins `tokenizers >=0.23.1,<0.24.0`, which is why this
stack is kept in a separate environment rather than the production `.venv`.

Hardware and runtime:

| | |
|---|---|
| GPU (formal run) | NVIDIA L4 24 GB |
| Driver / CUDA | 580.126.20 / CUDA 13.0 |
| bf16 | Supported and used by the formal recipe |
| Peak training memory | ~11.46 GB allocated / ~13.56 GB reserved |
| Output disk | Checkpoints need several GB; `run.sh` warns below 5 GB free |

The formal recipe requires a bf16-capable CUDA GPU. Passing `--no-bf16` runs
fp32 and **deviates from the recorded recipe**. CPU is sufficient only for
`run.sh --smoke` and the offline test suites.

## 6. Dataset Preparation

Dataset V1 (the "dual-target set") is frozen and tracked in [`data/`](data/).
It is also archived publicly as
[`Delveta-BGE-M3-v1-Data`](https://huggingface.co/datasets/eric-ml-nlp/Delveta-BGE-M3-v1-Data).
The frozen bundle is used as-is; training does not regenerate it.

### 6.1 Composition

| Split | Task B rows (query anchor) | Task A rows (capability description) | Merged rows |
|---|---:|---:|---:|
| Train | 8,108 | 8,108 | 16,216 |
| Test | 7,200 | 7,200 | 14,400 |

- Capabilities: 18.
- Training row schema: `{"query": str, "pos": [str], "neg": [str]}` — exactly
  one positive and two negatives in the frozen train rows.
- Each unique query yields two physical records (one per positive family)
  sharing only the query text; the two positive lists never mix.

Language profile (computed from script presence, not from a declared label):

| Split / task family | en | zh | mixed |
|---|---:|---:|---:|
| Train query / desc | 4,040 | 3,002 | 1,066 |
| Test query / desc | 1,600 | 2,071 | 3,529 |

Negative provenance:

| Split | `curated_confusable` | `random_cross_capability` |
|---|---:|---:|
| Train | 7,656 | 452 (all `cap-research`) |
| Test | 6,800 | 400 (all `cap-research`) |

Known data note: the `cap-research` rows use random cross-capability negatives
rather than curated confusables. The audit records them for review and does not
silently remove or replace them.

### 6.2 Manifest

`bge_m3_{split}_manifest.jsonl` is row-aligned with the data file: row `i`
describes data row `i`. The only field training consumes is
`gold_capability_id`; `task`, `declared_lang`, `language_profile`, `source`,
`seed_family`, `negative_capabilities`, and `negative_provenance` are recorded
provenance.

[`dual_target_alignment.py`](dual_target_alignment.py) re-checks the alignment
from bytes on disk at every training start. A reordered, truncated, or
regenerated data file fails loudly instead of silently corrupting the sampler's
index -> capability map.

### 6.3 Build and audit flow

| Step | Script | Output |
|---|---|---|
| Merge the query-anchor family with the capability-description family | [`build_dual_target.py`](build_dual_target.py) | Merged train/test files, task-specific files and manifests, `data/_build_dual_target_report.json` |
| Recompute the post-merge audit claims | [`verify_dual_target.py`](verify_dual_target.py) | `data/_verify_dual_target_report.json` |
| Final pre-training gate | [`final_gate_audit.py`](final_gate_audit.py) | `data/_final_gate_report.json` |
| Two-chain structural audit | [`audit_dual_target.py`](audit_dual_target.py) | stdout |
| Historical pre-merge schema/leakage audit | [`audit_dataset.py`](audit_dataset.py) | stdout; legacy semantics |

The builder and the registry-aware audits read a live registry snapshot from
`../../logs/_registry_dump.json`. That snapshot is a local working artifact and
is not part of the public dataset bundle. Rebuilding is therefore not required
to reproduce training: the frozen files, manifests, and reports are already
tracked and published.

> **Do not run `build_dual_target.py` against the frozen bundle** unless you
> intentionally want to rebuild Dataset V1. It writes over the data and manifest
> files. The frozen artifact must not be regenerated to "fix" a hash mismatch.

### 6.4 Freeze record

| | |
|---|---|
| Freeze commit | `24ab4c7` |
| Freeze tag | `bge-m3-dual-target-v1` |
| `bge_m3_train.jsonl` (16,216 rows) | `163b181f4980e0fc4e0d47e9140e554f147a38a160e7a3a310e511e750fe99b8` |
| `bge_m3_train_manifest.jsonl` (16,216 rows) | `57a0fb79d7c0238c73ab704b67928c876aaaf6f434a0975b1ae9ff095cc3a33d` |
| `bge_m3_test.jsonl` (14,400 rows) | `22b1bfc791944ebe5900d202523bf0436d1771c9fef99653b87b75e8bf0df39b` |

Audited leakage facts: exact and normalized query overlap between train and
test is `0`; positives shared between train and test are shared retrieval
targets by design (18 canonical descriptions and the corpus anchors), not test
query leakage. The full evidence is in
[`data/bge_m3_dual_target_audit_after.md`](data/bge_m3_dual_target_audit_after.md).
The earlier generation record is in
[`data/bge_m3_dataset_generation_audit.md`](data/bge_m3_dataset_generation_audit.md).

## 7. Training

### 7.1 One-command entry (recommended)

[`run.sh`](run.sh) calls the real training entry with the pinned formal recipe
and adds environment checks, Dataset V1 alignment validation, and an
output-directory guard. It does not modify the dataset, sampler, loss, or
hyper-parameters.

From this directory:

```bash
# Linux / WSL
PYTHON=../../.venv-bge-train/bin/python bash run.sh

# Windows (Git Bash)
PYTHON=../../.venv-bge-train/Scripts/python.exe bash run.sh

# CPU-only plumbing check: environment, dependencies, alignment, entry --help.
# This does not train and writes nothing.
bash run.sh --smoke

bash run.sh --help
```

Interpreter resolution order: `--python` > `$PYTHON` >
`<repo>/.venv-bge-train/bin/python` > `python3` > `python`.

The default output directory is `out/` under this project. `run.sh` refuses to
run when that directory exists and is non-empty; pass `--output-dir <new>` or
`--allow-existing-output` to override.

### 7.2 Pinned formal recipe

| Setting | Value |
|---|---|
| Base model | `BAAI/bge-m3` |
| `per_device_train_batch_size` | 8 |
| `train_group_size` | 8 |
| Learning rate | `1e-5` |
| Epochs | 1 |
| `query_max_len` / `passage_max_len` | 64 / 64 |
| Mixed precision | bf16 |
| Optimizer steps | 2027 |
| In-batch negatives | Enabled |
| Batch sampling | Capability-unique |
| Dense-only | `unified_finetuning=False` |

The direct invocation used for the formal run was:

```bash
python train_bge_m3_dual_target.py \
  --train_data data/bge_m3_train.jsonl \
  --train_manifest data/bge_m3_train_manifest.jsonl \
  --model_name_or_path BAAI/bge-m3 \
  --local_files_only \
  --output_dir /workspace/_bge_formal_out \
  --per_device_train_batch_size 8 \
  --train_group_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --query_max_len 64 \
  --passage_max_len 64 \
  --bf16
```

`--local_files_only` requires the model and tokenizer to be present in the local
Hugging Face cache. The entry prints a startup summary before training:

```text
[dual-target] aligned rows=16216 capabilities=18 in_batch_negatives=True dense_only=True per_device_batch=8
```

There is **no checkpoint-resume flag**. Every run starts from the base model;
`run.sh` exposes no resume option either.

### 7.3 Outputs

The output directory receives the standard `Trainer` checkpoint layout written
by `EncoderOnlyEmbedderM3Trainer`. The formal run wrote checkpoints at steps
500, 1000, 1500, 2000, and 2027 (`save_steps=500`). The final artifact is
`checkpoint-2027`; the published FP32 master is byte-identical to it.

## 8. Validation and Tests

Run the commands below from this directory.

### 8.1 Offline test suites

```bash
# Linux / WSL
../../.venv-bge-train/bin/python -m unittest test_capability_unique_sampler -v
../../.venv-bge-train/bin/python -m unittest test_dual_target_training_entry -v

# Windows
../../.venv-bge-train/Scripts/python.exe -m unittest test_capability_unique_sampler -v
../../.venv-bge-train/Scripts/python.exe -m unittest test_dual_target_training_entry -v
```

Recorded results:

| Suite | Result | Notes |
|---|---|---|
| `test_capability_unique_sampler` | 8 tests, OK | Uses a real `DataLoader`; includes the frozen-manifest smoke test. |
| `test_dual_target_training_entry` | 19 tests, OK, 1 skipped | The skipped case requires a bf16-capable GPU. `TestTrainingPath` is skipped if `accelerate` is unavailable; the FlagEmbedding contract tests require FlagEmbedding 1.4.2. |

`test_dual_target_training_entry.py` also runs a subprocess check proving that
the entry module imports without FlagEmbedding installed.

### 8.2 Data validation

```bash
# Strict data <-> manifest row alignment. Read-only; prints a JSON report.
../../.venv-bge-train/bin/python dual_target_alignment.py \
  data/bge_m3_train.jsonl data/bge_m3_train_manifest.jsonl

# Recompute the dual-target audit. Writes data/_verify_dual_target_report.json.
../../.venv-bge-train/bin/python verify_dual_target.py

# Final pre-training gate. Writes data/_final_gate_report.json.
../../.venv-bge-train/bin/python final_gate_audit.py

# Two-chain structural audit. Prints to stdout; writes no data.
../../.venv-bge-train/bin/python audit_dual_target.py
```

On Windows, replace `../../.venv-bge-train/bin/python` with
`../../.venv-bge-train/Scripts/python.exe`.

`verify_dual_target.py`, `final_gate_audit.py`, and `audit_dual_target.py`
require the local registry snapshot at `../../logs/_registry_dump.json`.
`audit_dataset.py` is the historical step-1 audit for the pre-merge dataset and
is not a current acceptance gate.

## 9. Model Artifacts

The formal run produced `checkpoint-2027` (1 epoch, 2027 optimizer steps). Three
inference variants are published in one Hugging Face repository:

| Variant | Path | Format | File | Size (bytes) | SHA256 |
|---|---|---|---|---|---|
| FP32 (master) | `FP32/` | safetensors, float32 | `model.safetensors` | 2,271,064,456 | `2e7ef22274798217832200d8666d94f6ba0bdf18107a6dc321ad5ccfa5c2aac6` |
| FP16 (derived) | `FP16/` | safetensors, float16 | `model.safetensors` | 1,135,554,312 | `a57cab2edb389ef43467103e46cd0825cf1244fbb236ff442c5134b2eaa08db7` |
| INT8 (derived, CPU) | `INT8/` | ONNX, dynamic int8 | `model.onnx` | 568,511,234 | `0e0ada7fec9367bca6d816b86953e4f9fe8531ebd4e8251f6de0721c0d2b1f0d` |

- **FP32** is the trained checkpoint; no conversion is applied.
- **FP16** is a dtype cast of the FP32 weights.
- **INT8** is an ONNX export (opset 17, dynamic batch/sequence) followed by
  dynamic quantization to `QInt8`. The graph wraps the encoder plus CLS pooling
  and L2 normalization, so pooling and normalization are part of the exported
  model rather than a separate post-processing step.

Embedding consistency was checked on 12 frozen test queries. These are
**fidelity checks between variants, not retrieval-effectiveness results**:

| Comparison | min cosine | mean cosine |
|---|---:|---:|
| FP16 vs FP32 | 0.999999 | 0.999999 |
| INT8 vs FP32 | 0.988429 | 0.990545 |
| ONNX-FP32 vs PyTorch FP32 | 1.0 | 1.0 |

All variants produced shape `(12, 1024)` with no NaN/Inf and L2 norm ≈ 1.0.
The ONNX export is lossless relative to PyTorch FP32, so the ~0.99 INT8 figure
is quantization error rather than export error.

Variant conversion tooling is not part of this directory. Generation and
verification records, including the exact tool versions, are in the
[experiment record](../../docs/experiments/bge-m3-finetuning.md).

## 10. Evaluation and Limitations

Training loss decreased during the formal run:

| Optimizer step | Logged training loss |
|---:|---:|
| 500 | 1.1208 |
| 1000 | 0.8270 |
| 1500 | 0.8004 |
| 2000 | 0.7955 |

Final epoch-average training loss: **0.8835**. Training completed in 451.4 s
with no NaN/Inf observed.

**A decreasing training loss is not evidence of improved retrieval.** The four
measurement families are kept separate:

| Family | Question | Status |
|---|---|---|
| A. Training metrics | Did optimization proceed sanely? | Recorded above |
| B. Embedding numerical consistency | Do FP32 / FP16 / INT8 agree numerically? | Recorded (12-query sample) |
| C. Retrieval effectiveness | Is retrieval actually better? | **NOT YET COMPLETED** |
| D. Runtime performance | What does inference cost? | Partial (CPU figures recorded; GPU inference latency not measured) |

Open items:

1. **No post-finetuning retrieval benchmark exists.** No Recall@1/@3/@5/@10,
   MRR, or capability-recall number is reported. No validated harness has been
   run for this model.
2. **INT8 retrieval effectiveness is blocked.** The evaluation harness does not
   yet include an ONNX Runtime adapter for the quantized variant.
3. **The consistency sample is small.** Twelve queries on a single machine is a
   variant-fidelity check, not a general embedding-quality claim.
4. **The model is domain-specific.** It is optimized for the Delveta capability
   corpus and is not a general-purpose drop-in for `BAAI/bge-m3`.
5. **Determinism is not bit-exact.** The sampler is seeded (`seed=42`), but
   cuDNN/cuBLAS kernel selection across GPUs or torch builds is not fixed.
6. **Runtime numbers span two runtimes.** CPU INT8 figures were measured under
   ONNX Runtime and FP32 under PyTorch; only file size and peak RSS are directly
   comparable across those columns.

## 11. Links and Provenance

| Item | Location |
|---|---|
| Model (FP32 / FP16 / INT8) | [`eric-ml-nlp/Delveta-BGE-M3-v1`](https://huggingface.co/eric-ml-nlp/Delveta-BGE-M3-v1) |
| Frozen dataset | [`Delveta-BGE-M3-v1-Data`](https://huggingface.co/datasets/eric-ml-nlp/Delveta-BGE-M3-v1-Data) · [`data/`](data/) |
| Training archive (logs / summary) | [`Delveta-BGE-M3-v1-checkpoints`](https://huggingface.co/eric-ml-nlp/Delveta-BGE-M3-v1-checkpoints) |
| Experiment record | [`docs/experiments/bge-m3-finetuning.md`](../../docs/experiments/bge-m3-finetuning.md) |
| Architecture note | [`docs/architecture.md` §29](../../docs/architecture.md#29-bge-m3-embedding-fine-tuning) |
| Dual-target data audit | [`data/bge_m3_dual_target_audit_after.md`](data/bge_m3_dual_target_audit_after.md) |
| Data generation record | [`data/bge_m3_dataset_generation_audit.md`](data/bge_m3_dataset_generation_audit.md) |
| Build report | [`data/_build_dual_target_report.json`](data/_build_dual_target_report.json) |
| Verification report | [`data/_verify_dual_target_report.json`](data/_verify_dual_target_report.json) |
| Final gate report | [`data/_final_gate_report.json`](data/_final_gate_report.json) |
| Base model | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) |

Provenance:

| Field | Value |
|---|---|
| Base revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| Recorded training-code commit | `f7b1f21` |
| Dataset freeze commit / tag | `24ab4c7` / `bge-m3-dual-target-v1` |
| Directory migration commit | `ded3b19` (project moved to `fine_tuning/bge_m3_finetune/`) |
| Training dataset SHA256 | `163b181f4980e0fc4e0d47e9140e554f147a38a160e7a3a310e511e750fe99b8` |
| Training manifest SHA256 | `57a0fb79d7c0238c73ab704b67928c876aaaf6f434a0975b1ae9ff095cc3a33d` |

The SHA256 values in this README and in the experiment record are the authority
for the published copies.
