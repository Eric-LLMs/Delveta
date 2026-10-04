# BGE-M3 Fine-Tuning — Process

Reproducible process for fine-tuning `BAAI/bge-m3` into Delveta's dense retrieval
embedding model (architecture: [§29](architecture.md#29-bge-m3-embedding-fine-tuning)).
The measurement side — what was and was not verified — is a **separate** document:
[BGE-M3 Evaluation](bge-m3-evaluation.md).

---

## 1. Goal and scope

Fine-tune `BAAI/bge-m3` into a **dense-only** embedder with a **single unified
contrastive objective**:

```
query -> positive target text -> negative target texts
```

In scope: the training entry, the capability-unique sampler, the manifest
row-alignment check, their tests, and the FP32 / FP16 / INT8 model variants.

Out of scope (deliberately): retraining to change the recipe, regenerating
Dataset V1, and any claim about retrieval effectiveness — the fine-tune does not
by itself establish retrieval improvement.

Design invariants (must hold in any reproduction):

- **dense-only** (`unified_finetuning=False`) — no sparse / ColBERT heads, no
  composite loss;
- **default in-batch negatives ON**;
- **capability-unique batching** via `gold_capability_id`;
- `task` is provenance metadata only — it never enters the loss and selects no
  branch;
- no "Task A / Task B" split and no capability-classification head.

## 2. Environment and software versions

Fine-tuning runs in its **own** virtual environment, isolated from the production
runtime and from the LayaChoice environment.

Verified local environment (`.venv-bge-train`, Windows).

| Package | Version |
|---|---|
| FlagEmbedding | 1.4.2 |
| transformers | 5.17.0 |
| torch | 2.14.0 (CPU build locally; CUDA build `2.14.0+cu130` on the GPU host) |
| tokenizers | 0.23.2 |
| datasets | 5.0.1 |
| accelerate | 1.15.0 |
| sentence-transformers | 6.1.0 |
| peft | 0.21.2 |

The formal run was executed on a GPU host: **NVIDIA A30 24 GB**, driver
`580.126.20`, CUDA 13.0 (compute capability 8.0, bf16 supported).

> `transformers` 5.17.0 pins a narrow `tokenizers` range
> (`>=0.23.1,<0.24.0`). Installing the training stack **inside** the production
> `.venv` fails on that pin, which is exactly why the training stack is kept in a
> dedicated environment.

## 3. Fetching Dataset V1 and checking integrity

Dataset V1 is frozen and archived at
[`eric-ml-nlp/Delveta-BGE-M3-v1-Data`](https://huggingface.co/datasets/eric-ml-nlp/Delveta-BGE-M3-v1-Data).
A checkout of the repository also carries the frozen files under
[`scripts/bge_m3_finetune/data/`](../scripts/bge_m3_finetune/data).

Freeze record:

| Field | Value |
|---|---|
| Freeze commit | `24ab4c7` (message: *bge: freeze dual-target finetuning dataset v1*) |
| Freeze tag | `bge-m3-dual-target-v1` |
| Capabilities | 18 |

Integrity check (must match before training):

| File | Rows | SHA256 |
|---|---|---|
| `bge_m3_train.jsonl` | 16216 | `163b181f4980e0fc4e0d47e9140e554f147a38a160e7a3a310e511e750fe99b8` |
| `bge_m3_train_manifest.jsonl` | 16216 | `57a0fb79d7c0238c73ab704b67928c876aaaf6f434a0975b1ae9ff095cc3a33d` |
| `bge_m3_test.jsonl` | 14400 | `22b1bfc791944ebe5900d202523bf0436d1771c9fef99653b87b75e8bf0df39b` |

```bash
cd scripts/bge_m3_finetune/data
sha256sum bge_m3_train.jsonl bge_m3_train_manifest.jsonl bge_m3_test.jsonl
```

## 4. Training data schema

`bge_m3_train.jsonl` — one JSON object per row, keys exactly `query` / `pos` / `neg`:

```json
{"query": "Add this term to my glossary.",
 "pos": ["Add \"photosynthesis\" to my glossary."],
 "neg": ["Create a new folder.", "Edit this file and update the title line."]}
```

| Field | Type | Cardinality (frozen train set) |
|---|---|---|
| `query` | non-empty string | 1 |
| `pos` | non-empty list of strings | exactly 1 per row |
| `neg` | non-empty list of strings | exactly 2 per row |

`pos` is a *positive target text* — either a capability's canonical description or
a standard/similar corpus query. Both are the **same** contrastive task; the
distinction is provenance, not a training branch.

## 5. Manifest and capability alignment

`bge_m3_train_manifest.jsonl` is a **row-aligned** sidecar: row *i* describes row
*i* of the data file.

```json
{"split": "train", "gold_capability_id": "cap-add-term", "declared_lang": "en",
 "language_profile": "en", "source": "existing_corpus908", "seed_family": null,
 "query_sha256": "4d48b8a5...", "negative_capabilities": ["cap-create-folder", "cap-edit-file"],
 "negative_provenance": "curated_confusable", "task": "query",
 "positive_anchor": "Add \"photosynthesis\" to my glossary."}
```

- `task` is `query` (8108 rows) or `desc` (8108 rows) — **provenance only**. The
  training path never reads it and never branches on it.
- The **only** manifest field training consumes is `gold_capability_id`, and only
  to build the capability-unique sampler's index → capability map.
- Training order is fail-fast. The entry runs, in order:
  1. hard gate — `--train_data` must be a single JSONL **file**, never a directory
     (a directory would be concatenated by FlagEmbedding and break row alignment);
  2. `dual_target_alignment.validate(data, manifest)` — strict row alignment:
     equal row counts, required keys present, and
     `sha256(data[i]["query"]) == manifest[i]["query_sha256"]` for every row;
  3. `capabilities_from_manifest(manifest)` — the row-aligned capability list;
  4. `per_device_train_batch_size <= number of distinct capabilities`.

`dual_target_alignment.validate` recomputes alignment from bytes on disk, so a
silently reordered / truncated / regenerated data file can never desynchronise the
sampler.

## 6. `CapabilityUniqueBatchSampler` principle

With default in-batch negatives, any batch that carries two records of one
capability turns one record's positive text into another record's in-batch
**negative** — a false negative that corrupts the contrastive signal. The
dual-target set gives both of a query's targets the **same** `gold_capability_id`,
so this collision is likely by default.

`CapabilityUniqueBatchSampler` guarantees, at the `DataLoader` level, that **every
batch carries at most one record per `gold_capability_id`** — which removes that
class of false negative without touching the loss.

Contract per epoch:

- every dataset index is emitted exactly once (`index_loss = 0`);
- no index is emitted twice (`index_dup = 0`);
- each yielded batch has **distinct** capabilities;
- a trailing batch may be smaller than `batch_size` — it is **emitted**, never
  silently dropped;
- `batch_size > number of distinct capabilities` is **rejected** (a
  capability-unique batch is then impossible); the sampler never clamps.

Mechanics:

- rows are built by a **min-heap "smallest row first"** assignment, so each
  capability is popped at most once per row and row sizes stay balanced; the row
  count is `max(ceil(N / batch_size), max_capability_count)` (the second term is
  the feasibility floor);
- batches are produced as explicit index **lists** (a `batch_sampler`), so
  `torch.utils.data.DataLoader` yields them verbatim and never re-chunks them by
  `batch_size`;
- the class subclasses the fork's `BatchRebalanceSampler` so the vendored
  `Trainer._get_dataloader` routes it through the `batch_sampler=` branch and
  neutralises the accelerator shard — keeping default in-batch negatives intact;
- under data parallelism each rank takes `rows[rank::dp_size]`.

The sampler is injected by `CapabilityUniqueSamplerMixin._get_train_sampler` (the
single hook `Trainer.get_train_dataloader` calls). The mixin stays left-most in the
MRO — `M3DualTargetTrainer(CapabilityUniqueSamplerMixin, EncoderOnlyEmbedderM3Trainer)`
— so its override wins while the official M3 loss and checkpoint layout are left
untouched.

## 7. Training command and parameters

The entry is [`train_bge_m3_dual_target.py`](../scripts/bge_m3_finetune/train_bge_m3_dual_target.py).
Its flags (verified via `--help`):

| Flag | Default | Meaning |
|---|---|---|
| `--train_data` | `data/bge_m3_train.jsonl` | single training JSONL file (never a directory) |
| `--train_manifest` | `data/bge_m3_train_manifest.jsonl` | row-aligned manifest |
| `--model_name_or_path` | `BAAI/bge-m3` | base model id or local path |
| `--local_files_only` | off | load tokenizer/model from the local HF cache only (no network) |
| `--output_dir` | required | checkpoint output directory |
| `--per_device_train_batch_size` | 8 | per-process micro-batch; must be `<=` distinct capabilities |
| `--train_group_size` | 8 | passages grouped per query (positive + negatives) |
| `--learning_rate` | 1e-5 | |
| `--num_train_epochs` | 1.0 | |
| `--query_max_len` | 512 | query token budget (the formal run used 64) |
| `--passage_max_len` | 512 | passage token budget (the formal run used 64) |
| `--bf16` | off (fp32) | opt-in bf16 mixed precision; requires a bf16-capable GPU |

The **formal run** used the following configuration (values from the run's
`run_summary.json`; the output paths are the GPU host's actual artifact
directory):

```bash
python train_bge_m3_dual_target.py \
  --train_data   data/bge_m3_train.jsonl \
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

Notes:

- `--bf16` is passed as a **flag**; the entry forwards it into the
  `EncoderOnlyEmbedderM3TrainingArguments` **constructor** (not by mutating the
  object afterwards — `mixed_precision` is derived once in `__post_init__`).
- `unified_finetuning=False` is **pinned in code**, not a flag: the entry sets it
  explicitly because the M3 model constructor defaults it to `True` while the
  official M3 `TrainingArguments` defaults it to `False`.
- The entry prints a one-line banner before training:
  `[dual-target] aligned rows=16216 capabilities=18 in_batch_negatives=True dense_only=True per_device_batch=8`.

## 8. CPU smoke test

Code-level verification runs entirely offline and requires **no** GPU. From
`scripts/bge_m3_finetune/`:

```bash
../../.venv-bge-train/Scripts/python.exe -m unittest test_capability_unique_sampler -v
../../.venv-bge-train/Scripts/python.exe -m unittest test_dual_target_training_entry -v
```

The entry itself also has an offline self-check: it imports cleanly without
FlagEmbedding, and `--help` works without the training runtime installed.

For an end-to-end CPU training smoke, run the entry with a small subset and
`per_device_train_batch_size <= number of distinct capabilities in that subset`
(e.g. 8 rows / 8 capabilities, `bs=2`, `train_group_size=2`, `query_max_len=64`,
`passage_max_len=64`) and a throwaway `--output_dir`.

## 9. GPU smoke test

A GPU smoke validates the accelerator-only behaviour that CPU cannot: bf16
autocast actually reaching the encoder, optimizer steps advancing, and finite
losses.

- Requires a bf16-capable GPU (the formal host: A30).
- Pass `--bf16` to the real entry (no monkeypatching needed — the flag is part of
  the entry).
- Use a tiny subset and assert: optimizer steps advance, all losses are finite,
  and — if bf16 is under test — autocast is observed inside the encoder.
- The local machine has **no GPU**, so a GPU smoke is not runnable locally; it is
  a GPU-host step only.

## 10. Formal training

Formal run configuration (provenance record:

[`run_summary.json`](https://huggingface.co/eric-ml-nlp/Delveta-BGE-M3-v1-checkpoints/blob/main/run_summary.json)):

| Field | Value |
|---|---|
| Training code commit | `f7b1f21` |
| Base model | `BAAI/bge-m3` |
| Base revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| Epochs | 1 |
| Optimizer steps | 2027 |
| `per_device_train_batch_size` | 8 |
| `train_group_size` | 8 |
| `query_max_len` / `passage_max_len` | 64 / 64 |
| Learning rate | 1e-5 |
| bf16 | True |
| Dense-only (`unified_finetuning=False`) | True |
| In-batch negatives | enabled |
| Training rows / capabilities | 16216 / 18 |
| Hardware | NVIDIA A30 24 GB |
| Wall-clock (train runtime) | 451.4 s |
| Final checkpoint | `checkpoint-2027` |

Checkpoints written: `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`,
`checkpoint-2000`, `checkpoint-2027`.

## 11. Checkpoint saving and resume

- The official `EncoderOnlyEmbedderM3Trainer` owns the save layout — the mixin
  does **not** override `_save` or `compute_loss`, so checkpoints are standard M3
  checkpoints.
- `save_steps` in the formal run was `500`, which is why `checkpoint-500 …`
  exist.
- On resume, **re-run the same startup gates**: alignment is re-validated from
  disk and the batch size is re-checked against the capability count. Resuming
  with a data file that has been reordered or truncated is caught before any
  optimizer step.
- The **final** artifact is `checkpoint-2027`; the FP32 master is byte-identical
  to it.

## 12. FP32 / FP16 / INT8 conversion

All three variants share one embedding semantic: `normalize(last_hidden_state[:, 0])`
(CLS pooling + L2 normalization).

**FP32** — the reference master. No conversion; the weights are the trained
`checkpoint-2027`, byte-identical to the source.

**FP16** — a dtype cast of the FP32 master into `torch.float16`
(`torch` 2.14.1, `safetensors` 0.8.0). File size halves; embedding semantics are
unchanged.

**INT8** — an ONNX build for CPU inference:

1. export a wrapper module — encoder + **CLS pooling** + **L2 normalization** —
   with `torch.onnx.export(..., opset_version=17, dynamo=False,
   dynamic_axes={batch, seq})`, so pooling and normalization are *inside* the
   graph and the raw encoder hidden states are not exposed as the embedding;
2. quantize with
   `onnxruntime.quantization.quantize_dynamic(weight_type=QuantType.QInt8)`.

Tooling for INT8: `torch` 2.14.1+cpu, `onnx` 1.23.1, `onnxruntime` 1.30.0.
Conversion is done in an isolated environment (`/workspace/_onnx_env`) so the
production `.venv-bge-train` is never disturbed.

> `torch.onnx.export` may emit **external-data sidecar weight files** next to the
> `.onnx` (this happened for the fp32 export). Always verify the exported
> `model.onnx` loads from a directory containing **only** `model.onnx`, then remove
> the sidecars.

## 13. Hugging Face upload and verification

The three variants live side by side in **one** repository —
[`eric-ml-nlp/Delveta-BGE-M3-v1`](https://huggingface.co/eric-ml-nlp/Delveta-BGE-M3-v1)
— each a self-contained subfolder (weights + `config.json` + tokenizer files +
`README.md`). The repository root keeps only `README.md` + `.gitattributes`; there
is **no root model copy**.

Published layout and verified sizes / SHA256:

| Variant | File | Size (bytes) | SHA256 |
|---|---|---|---|
| `FP32/` | `model.safetensors` | 2,271,064,456 | `2e7ef22274798217832200d8666d94f6ba0bdf18107a6dc321ad5ccfa5c2aac6` |
| `FP16/` | `model.safetensors` | 1,135,554,312 | `a57cab2edb389ef43467103e46cd0825cf1244fbb236ff442c5134b2eaa08db7` |
| `INT8/` | `model.onnx` | 568,511,234 | `0e0ada7fec9367bca6d816b86953e4f9fe8531ebd4e8251f6de0721c0d2b1f0d` |

Verification after upload (read-only):

```python
from huggingface_hub import HfApi
api = HfApi()
for it in api.get_paths_info("eric-ml-nlp/Delveta-BGE-M3-v1",
                             ["FP32/model.safetensors", "FP16/model.safetensors", "INT8/model.onnx"]):
    print(it.path, it.size, it.lfs.get("sha256") if it.lfs else None)
```

- HF LFS is content-addressed, so re-uploading an identical file is near-instant.
- Before any delete of root copies, verify the three subfolder SHA256s on the
  remote first, then delete the root model files — never the other way round.

## 14. Training log and provenance

The training archive ([`Delveta-BGE-M3-v1-checkpoints`](https://huggingface.co/eric-ml-nlp/Delveta-BGE-M3-v1-checkpoints),
private) holds `train.log`, `run_summary.json`, and a `README.md`. The `train.log`
begins with the alignment banner and then the per-step progress bar; the summary
records the loss/learning-rate/gradient-norm history.

Provenance fields recorded for this run:

| Field | Value |
|---|---|
| Training code commit | `f7b1f21` |
| Dataset SHA256 | `163b181f4980e0fc4e0d47e9140e554f147a38a160e7a3a310e511e750fe99b8` |
| Manifest SHA256 | `57a0fb79d7c0238c73ab704b67928c876aaaf6f434a0975b1ae9ff095cc3a33d` |
| Dataset freeze commit | `24ab4c7` |
| Dataset freeze tag | `bge-m3-dual-target-v1` |
| Base revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| Hardware | NVIDIA A30 24 GB |

## 15. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: tokenizers>=0.23.1,<0.24.0 is required …` | Wrong environment. `transformers` 5.17.0 needs `tokenizers` 0.23.x; use the dedicated training env, not the production `.venv`. |
| `AttributeError: EncoderOnlyEmbedderM3Model has no attribute from_pretrained` | Expected. FlagEmbedding 1.4.2 exposes no `from_pretrained` for the M3 finetune model — build it via `EncoderOnlyEmbedderM3Runner.get_model(...)` → dict → `EncoderOnlyEmbedderM3Model(base_model=..., tokenizer=...)`. |
| Sparse / ColBERT heads unexpectedly present | `unified_finetuning` default mismatch: the M3 model constructor defaults `True`, the M3 `TrainingArguments` default `False`. The entry pins `False` explicitly. |
| `--bf16` has no effect | `mixed_precision` is derived in `__post_init__`; the flag must reach the arguments **constructor**. Mutating `args.bf16` after construction does not change `mixed_precision`. |
| `--train_data must be a single JSONL file, got directory` | A directory is concatenated by FlagEmbedding and breaks row alignment. Pass one file. |
| `per_device_train_batch_size … exceeds the N distinct capabilities` | A capability-unique batch is impossible; the sampler refuses to clamp. Lower the batch size. |
| `… carries the .no_in_batch_neg suffix` | The recipe requires default in-batch negatives. Rename the file (drop the suffix). |
| `row count mismatch` / `row N is misaligned` | Data file and manifest are not row-aligned. Regenerate or restore the frozen pair; never train on a misaligned pair. |
| ONNX export leaves many sidecar files | `torch.onnx.export` external-data. Verify the `.onnx` loads standalone, then delete the sidecars. |

## 16. Related source, tests, model and dataset

- Training entry: [`train_bge_m3_dual_target.py`](../scripts/bge_m3_finetune/train_bge_m3_dual_target.py)
- Capability-unique sampler: [`capability_unique_sampler.py`](../scripts/bge_m3_finetune/capability_unique_sampler.py)
- Trainer mixin: [`capability_unique_trainer.py`](../scripts/bge_m3_finetune/capability_unique_trainer.py)
- Row-alignment check: [`dual_target_alignment.py`](../scripts/bge_m3_finetune/dual_target_alignment.py)
- Tests: [`test_capability_unique_sampler.py`](../scripts/bge_m3_finetune/test_capability_unique_sampler.py) · [`test_dual_target_training_entry.py`](../scripts/bge_m3_finetune/test_dual_target_training_entry.py)
- Frozen Dataset V1: [`scripts/bge_m3_finetune/data/`](../scripts/bge_m3_finetune/data) · [Delveta-BGE-M3-v1-Data](https://huggingface.co/datasets/eric-ml-nlp/Delveta-BGE-M3-v1-Data)
- Model (FP32 / FP16 / INT8): [Delveta-BGE-M3-v1](https://huggingface.co/eric-ml-nlp/Delveta-BGE-M3-v1)
- Training archive: [Delveta-BGE-M3-v1-checkpoints](https://huggingface.co/eric-ml-nlp/Delveta-BGE-M3-v1-checkpoints)
- Evaluation methodology: [BGE-M3 Evaluation](bge-m3-evaluation.md)
- Architecture: [§29 BGE-M3 Embedding Fine-tuning](architecture.md#29-bge-m3-embedding-fine-tuning)

[↑ Back to top](#bge-m3-fine-tuning--process)
