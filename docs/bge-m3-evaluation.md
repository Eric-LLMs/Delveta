# BGE-M3 Evaluation — Metrics and Status

How the fine-tuned BGE-M3 embedder is measured, and — importantly — what has
**not** been measured yet. The training process is a separate document:
[BGE-M3 Fine-Tuning](bge-m3-finetuning.md).

Four measurement families are kept strictly apart. **They are not
interchangeable**, and none of them substitutes for another:

| Family | Question it answers | Status here |
|---|---|---|
| A. Training metrics | did optimisation proceed sanely? | recorded |
| B. Embedding numerical consistency | do FP32 / FP16 / INT8 agree numerically? | recorded (small sample) |
| C. Retrieval effectiveness | is retrieval actually better? | **NOT YET COMPLETED** |
| D. Runtime performance | what does it cost to run? | recorded (this CPU) |

> Training loss decreased, but this alone does not establish retrieval improvement.

> Post-finetuning retrieval evaluation: NOT YET COMPLETED.

## A. Training metrics

Source: the run's `run_summary.json` / `train.log` (provenance record published in
[Delveta-BGE-M3-v1-checkpoints](https://huggingface.co/eric-ml-nlp/Delveta-BGE-M3-v1-checkpoints)).

Configuration: 1 epoch, 2027 optimizer steps, `logging_steps=500`,
`save_steps=500`, batch size 8, `train_group_size` 8, seq len 64, lr 1e-5, bf16.

Logged sampling points:

| Step | Train loss | Learning rate | Gradient norm |
|---:|---:|---:|---:|
| 500 | 1.1208 | 7.538e-06 | 24.40 |
| 1000 | 0.8270 | 5.072e-06 | 109.50 |
| 1500 | 0.8004 | 2.605e-06 | 13.99 |
| 2000 | 0.7955 | 1.381e-07 | 13.49 |
| Final (epoch avg) | 0.8835 | — | — |

- **Optimizer steps:** 2027 (global step 2027).
- **Training duration:** 451.4 s wall-clock (train runtime).
- **Peak GPU memory:** 11,459.1 MB allocated / 13,558.0 MB reserved (A30 24 GB).
- **NaN / Inf:** no non-finite loss observed; all logged losses are finite.
- **Checkpoint save status:** `checkpoint-500`, `checkpoint-1000`,
  `checkpoint-1500`, `checkpoint-2000`, `checkpoint-2027` written; `checkpoint-2027`
  is the final artifact.

> **Statistical caveat.** The value labelled *Final* (0.8835) is the **epoch-level
> average training loss** reported by the trainer; it is **not** the same statistic
> as any single logged sampling point above (those are per-step sampling losses
> around the logged step). The two must not be compared as if they were one series.

Training loss decreased, but this alone does not establish retrieval improvement.

## B. Embedding numerical consistency

This family checks that the three variants produce **numerically consistent**
embeddings. It is a *fidelity* check — it does **not** measure retrieval quality.

Procedure (per variant, same inputs):

1. tokenize the same fixed queries (CLS pooling + L2 normalization, i.e.
   `normalize(last_hidden_state[:, 0])`);
2. assert embedding **shape** `(n, 1024)` and **dimension** 1024;
3. assert **no NaN / Inf**;
4. assert **L2 norm** ≈ 1.0;
5. compute **cosine similarity** to the FP32 embedding: **min / mean / p50 / p95 / max**.

Sample: **12 frozen test queries**. Results:

| Comparison | min | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| FP16 vs FP32 | 0.999999 | 0.999999 | 1.0 | 1.0 | 1.0 |
| INT8 vs FP32 | 0.988429 | 0.990545 | 0.990726 | 0.992231 | 0.992688 |
| ONNX-FP32 vs PyTorch FP32 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

- The ONNX export itself is **lossless** (fp32 ONNX vs torch FP32 = 1.0 exactly),
  so the ~0.99 for INT8 is **pure quantization error**, not export error.
- Shape / dimension / NaN-Inf / L2-norm checks: PASS for all three variants.

> **Scope caveat.** This is a **12-query** sample on a **single** machine. It is a
> consistency check between variants, **not** a general statement about embedding
> quality, and **not** a retrieval-effectiveness result. The sample size is too
> small to extend these numbers into a general conclusion.

## C. Retrieval effectiveness

**Post-finetuning retrieval evaluation: NOT YET COMPLETED.**

No official Recall@k / MRR / capability-recall number is reported for the
fine-tuned model, because **no validated post-finetuning retrieval evaluation
harness is in place**. Reporting numbers here would require running the full
retrieval pipeline; none has been run for this model, and no number is fabricated.

### Intended protocol (design only — not yet executed)

When a harness exists, FP32 / FP16 / INT8 must be compared under a **frozen**
setup, holding all of the following fixed:

- the **test query set**;
- the **candidate corpus**;
- **query preprocessing**;
- the **text encoding** method;
- the **retrieval and ranking** logic;
- the **gold labels**;
- the **metric implementation**.

Metrics to report (with their definitions):

| Metric | Definition |
|---|---|
| Recall@k (k = 1, 3, 5, 10) | fraction of queries whose gold target appears in the top-*k* retrieved items |
| MRR | mean over queries of `1 / rank` of the first correct item |
| Capability retrieval recall | fraction of queries whose gold **capability** is retrieved into the candidate set |
| Capability selection recall | fraction of queries whose gold capability is *selected* by the downstream selector given the candidate set |

Results would be reported as **absolute values per variant** plus the **delta
relative to FP32** (the reference).

### INT8 retrieval effectiveness — BLOCKED

**INT8 retrieval effectiveness: BLOCKED.**

The current evaluation harness does not support the ONNX Runtime path, so the
quantized variant cannot be scored through the same protocol as FP32 / FP16. What
is missing is a small adapter that runs the ONNX-INT8 build through the same
encode → rank → score path the harness uses for the PyTorch variants.

> Embedding cosine similarity (Family B) is **not** a retrieval-effectiveness
> metric and is **not** a substitute for this family.

## D. Runtime performance

What it costs to run each variant. **Load time, single-query latency, batch
latency, throughput, and CPU peak RSS** are recorded below; **GPU peak
allocated / reserved** is from the formal training/embedding host.

Measured CPU performance — host: **AMD EPYC 7542**, **8 threads**, input length 64.

| Metric | INT8 (ONNX Runtime) | FP32 (PyTorch, reference) |
|---|---:|---:|
| Model file size | 568,511,234 B (~542 MiB) | 2,271,064,456 B (~2.11 GiB) |
| Load time | 1.285 s | 0.278 s |
| Single-query latency | 32.7 ms | 95.7 ms |
| Batch latency (batch of 12) | 288.7 ms | 275.8 ms |
| Peak RSS | 1607 MB | 1910 MB |

Reported settings: CPU model AMD EPYC 7542; threads = 8; batch size = 12;
input length = 64; runtime = onnxruntime (INT8) vs PyTorch (FP32).

- **Do not compare these two columns as if they were the same runtime.** INT8 runs
  under **ONNX Runtime** and FP32 under **PyTorch**; the runtimes differ, so the
  latencies are not an apples-to-apples quantization-vs-baseline comparison. The
  meaningful signals here are the **file size** and **peak RSS**, and — within the
  ONNX Runtime column, if measured — latency.
- GPU peak (formal training): 11,459.1 MB allocated / 13,558.0 MB reserved on an
  NVIDIA A30 24 GB. GPU inference latency for the variants has **not** been
  measured.
- Warm-up count, if any, must be recorded alongside a latency figure; do not mix
  cold and warm measurements in one table.

> Latency measured on **different devices** or at **different batch settings** must
> not be used to conclude which quantization is "better".

## Summary of statuses

| Item | Status |
|---|---|
| A. Training metrics | recorded (loss / lr / grad-norm / steps / duration / memory / checkpoint) |
| B. Embedding numerical consistency | recorded on a 12-query sample (FP16 ≈ 1.0; INT8 min 0.988 / mean 0.991) |
| C. Retrieval effectiveness | **NOT YET COMPLETED** |
| C. INT8 retrieval effectiveness | **BLOCKED** (harness lacks ONNX support) |
| D. Runtime performance | recorded for this CPU (INT8 vs FP32) |

[↑ Back to top](#bge-m3-evaluation--metrics-and-status)
