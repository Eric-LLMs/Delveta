# LayaChoice-v2 — Fine-Tuning Laya with an Explicit REJECT Option

> **Status: released.** The formal run is complete and the artifacts are closed.
> This document is the record of that experiment; the stable architectural facts distilled
> from it live in [architecture.md §26](../architecture.md#26-layachoice-capability-selection).
>
> **Production integration status.** v2 is wired into production: the `cap_router` card
> renderer emits the `REJECT` criterion (§6), the `deploy/laya` sidecar loads the v2
> checkpoint, and the service backend maps a `REJECT` answer to a normal fourth decision
> routed to the Agent. It runs as the `cap-router` Compose service.
>
> Everything below is derived from the run artifacts (`scripts/laya_finetune/V2/` and the
> frozen bundle `scripts/laya_finetune/V2/data/`), not from recollection.

## 1. Experiment Overview

| | |
|---|---|
| Task | Choose the correct capability for a user request from a 3-candidate `criteria` set, **or** reject the whole set — a 4-way flat classification |
| Base model | `convaiinnovations/laya`, subfolder `multilingual`, revision `55cf4c4eb…` |
| Output | `Delveta LayaChoice v2` — full-parameter fine-tune, FP32 export |
| Data | Train 2 112 / Calibration 211 / Validation v3 300 / Final Test v3 1 400 (all frozen) |
| Input config | `OPTION_MAX_TOKENS=256`, `head_max_len=768`, `max_len=1024` |
| Best epoch | **7** (selected by Validation v3 top-1 only) |
| Validation v3 (best) | **0.966667** = 290/300 |
| Final Test v3 | **0.877857** — EN 0.861272 / ZH 0.894068 |
| REJECT on Final Test | recall 0.719298 / precision 0.557823 / FPR 0.050544 |
| Export | `model.safetensors`, FP32 preserve, 1 287 653 720 B, sha256 `c6331bd5…` |

One sentence: **v1 (3-way) could only pick among the candidates it was shown; v2 adds an
explicit `REJECT` fourth option and is trained and measured as a 4-way classification**,
with the best checkpoint chosen on a separate validation split the test set never touched.

## 2. Objective

v1 answered "which of these 3 capabilities handles the request?". That framing has no way
to say *"none of them"* — a turn whose true capability was not retrieved is forced onto a
wrong candidate. The objective of v2 was to close that gap with a number:

> Can the same base decision model, fine-tuned with a dedicated `REJECT` option as a
> fourth class, learn to abstain when the correct capability is not among the candidates —
> without giving up capability-selection accuracy?

Not objectives: not a general chat model, not a production-traffic model, and not a
re-opening of the v1 dataset (v1 is frozen, §5).

## 3. From 3-way to 4-way

The v1 model scored exactly the candidates it was shown; it had no "none" output. v2 makes
abstention a first-class decision:

- every row carries **three capability cards plus one `REJECT` card** (`OPTION_SLOTS = 4`);
- `REJECT` is a **normal 4th option**, not a threshold and not a post-hoc rule — the model
  is trained to assign it probability mass directly;
- the option order is a **per-row deterministic 4-way shuffle** (seed prefix
  `reject-shuffle-v1`, recorded in each row's `shuffle` field), so the model cannot learn a
  positional prior;
- `gold_index ∈ {0,1,2,3}` marks the target slot and `target_kind ∈ {capability, reject}`
  records which kind of target the row has.

Because the model now always sees four options, **a caller must render the `REJECT` card**.
A 3-option question does not match the trained pipeline and its scores are not meaningful.

## 4. Base Model

| | |
|---|---|
| Repo | `convaiinnovations/laya` |
| Revision | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (asserted on load) |
| Subfolder | `multilingual` |
| Encoder | `jhu-clsp/mmBERT-base` (~322 M params) |
| Package | `laya==0.3.21` (pinned; inference only — the wheel ships no trainer) |
| Checkpoint temperature | `[1.0, 1.0, 1.0]` on the base |

The upstream repository went private after the v1 work started, which is why the recipe is
pinned to an exact revision and to `laya==0.3.21`.

## 5. Dataset Definition

**LayaChoice-v2** is the v1 capability-selection benchmark extended to a 4-way choice. It
is derived from the frozen v1 bundle (`B_noprov` view), not rebuilt from the Registry:

| | |
|---|---|
| Card view | `B_noprov` — the production card with the `query examples:` block and the `evidence:` provenance line removed |
| Candidates per row | **K = 3** capability cards **+ 1 `REJECT` card** (`OPTION_SLOTS = 4`) |
| Capabilities | 17 + `REJECT` = 18 possible target slots |
| Languages | `zh` / `en` |
| Row shape | `{id, lang, state, order[4], options[4], gold, gold_index, target_kind, part, …}` |

The `REJECT` card text is frozen for all rows:

> `### REJECT` / `tool: none` / `does: No listed capability correctly handles the user's
> request. Choose this option only when none of the other options is the right
> capability.` plus a Chinese line.

`REJECT` rows are of two kinds: *natural* (the source query genuinely had no matching
capability) and *synthetic* (a capability row whose gold card was swapped for `REJECT`).
Both are training/exam items; the frozen bundle records which is which, and that metadata
is **never rendered** — only `state` and the four option cards enter the sequence.

v1's label-leakage argument still holds unchanged: the stored `evidence:` line names the
answer, so it stays dropped. The `B_noprov` view is what makes the task a decision rather
than a lookup.

## 6. Splits, Calibration and the Adjustment

```
source_train_rows            2 112  ── exclude the frozen 211 calibration rows (seed 20260922)
calibration_rows               211  ──► ids recorded in manifest.json
effective_train_rows         1 901  ── after exclusion (world-size aligned)
Validation v3                  300  ── per-epoch top-1; the ONLY selection signal
Final Test v3                1 400  ── benchmark of best_epoch only, never used to pick or gate
```

- **Language** — train `zh` 1 084 / `en` 1 028; val `zh` 154 / `en` 146; test `en` 692 /
  `zh` 708.
- **Partitions** — every split is half `part1` / half `part2` (train 1 056/1 056, val
  150/150, test 700/700). `part1` is the v1-style candidate set (REJECT added); `part2` is
  the Recall-based candidate set (self-hit removed). The final schema is identical across
  parts.
- **Calibration rule** — `min(400, n_train // 10)` over the FINAL (post-shuffle) train
  order, seed `20260922`, 211 ids recorded by index; excluded from training by
  construction.
- **Test→Train adjustment** — 200 whole source queries (400 rows: one `p1` + one `p2` per
  source) were moved from Test into Train; the **only** field changed on any row is
  `split`. The pre-adjustment 1 800-row test is kept as backup material
  (`V2/frozen_backup/v2_test_1800_pre_adjustment.jsonl`). The FINAL splits are the ones
  above.

Train / validation / test are disjoint, and the calibration ids are excluded before any
world-size truncation, so no rank can pull a calibration row into training.

## 7. Frozen Dataset SHA256

Verified unchanged at the end of the run:

| Artifact | SHA256 |
|---|---|
| `v2_train.jsonl` | `c92c0b5f6b5fbb019f6cf1e89a107ad4d1c4fee952f4d1a024c6cb319873b720` |
| `v2_val.jsonl` | `f7d97670c96c13330f07dad2a3aecabbae42b3194e54c2cf0dafddc8af51de7b` |
| `v2_test.jsonl` | `eb14e0455156d6dd3f2b0ddf2fa82feaf5248b10f993a4ded57cea45261fb9ea` |
| `manifest.json` | `224acf2f3bdec01ab6284942c1f8b87acaf9daa7c924e89a5c8e97b485dbf57e` |
| `v2_build_report.json` | `e2a3dae1d7228ea31b7d318473edd6a795a4561b62b7ebc18f4aad8dd6fc0703` |
| `v2_construction_audit.jsonl` | `2464e187a667d515b97f0082e18d37b66b2b3c377b6f54f681a265c7e4b6091a` |

Source (v1) bundles the construction read from:

| Artifact | SHA256 |
|---|---|
| `layachoice_v1_train_b_noprov.jsonl.gz` | `e968fdeddc8f02eec19f37ed29fafc5f05a591ea259c97315639d366e3de4f13` |
| `layachoice_v1_val_b_noprov.jsonl.gz` | `3fa38aa099580b6f9a5146ac2a8e7beed0111d79d2f994adb6ca317547c90364` |
| `layachoice_v1_test_b_noprov.jsonl.gz` | `c665004a127496f4456a0b3ccd81bd0ce10654fef7808f27e45500d3a9d9b8bb` |

## 8. Candidate Contract

The production contract the model is trained and benchmarked under — **3 capabilities +
`REJECT`, always**:

| Candidate count | Behaviour |
|---|---|
| K = 0 | never enters LayaChoice (funnel short-circuits) |
| K = 1 | the business layer executes it directly, LayaChoice is not called |
| K = 2 | LayaChoice (capabilities) — **not measured by v2** |
| K = 3 | LayaChoice — the trained and measured case (`+ REJECT`) |
| K ≥ 4 | the business layer truncates to top-3 **first**, then LayaChoice |

**Production candidate normalization must happen before LayaChoice**, not inside it. v2 is
trained and measured at exactly three capability candidates **plus the `REJECT` card**; K=2
and K≥4 post-truncation are **not** exercised by any number in this document. A caller must
render `REJECT` as a fourth criteria.

## 9. Token Budget

The v2 input builder is a **byte-verbatim copy** of `laya.common.build_sequence` with
exactly one line changed (`max_length=48` → `256`); the stock even-share fallback and the
head-budget floor are preserved. The library default of 48 tokens per option would cut
every frozen card, so the cap is raised to 256.

The frozen configuration:

| Constant | Value | Meaning |
|---|---|---|
| `OPTION_MAX_TOKENS` | **256** | per-option hard cap (capability cards and `REJECT`) |
| `HEAD_MAX_LEN` | **768** | budget the question head **and all four options share** |
| `MAX_LEN` | **1024** | whole-sequence ceiling |

`head_max_len` is a **combined budget**, not a length. At four option slots the
construction audit confirmed the 256/768/1024 budget is sufficient — **no option card is
truncated and the `REJECT` card enters the sequence in full**.

**Deployment consequence:** a stock `laya` install renders at 48 tokens per option and
`head_max_len=256`. Reproducing v2's behaviour requires applying the same override before
the Agent is constructed (`R.install()` in `layachoice_v2_render.py`) and passing the same
256/768/1024 budget; under stock rendering expect a materially lower score.

## 10. Fine-Tuning Hyperparameters

| | |
|---|---|
| Epochs | **10** |
| Micro batch / grad accum | 8 / 4 (effective batch 32) |
| Group size (policy-sampling term) | 4 |
| `lr_encoder` / `lr_head` | 2.5e-5 / 1.0e-4 |
| Weight decay | 0.01 |
| σ schedule | 0.4 → 0.1 across epochs |
| Loss | `loss_rl + 1.0 * loss_ce`; `w_sph=0.75`, `w_rps=1.0` |
| Gradient clip | `clip_grad_norm_(1.0)` |
| Total optimizer updates | 600 |
| Precision | bf16 autocast (no `GradScaler`) |
| Hardware | 1× NVIDIA A30 |
| Runtime | torch `2.14.1+cu130`, transformers `5.17.0`, python `3.11`, `laya==0.3.21` |
| Seed base | `42` |
| Fine-tuning | **full-parameter** (no PEFT/LoRA) |

## 11. Epoch 1–10 Training Results

From the per-epoch validation records. Validation v3 (300 rows, `B_noprov`); top-1 is
invariant under the scalar calibration temperature. Validation `REJECT` counts are small
(16 gold `REJECT` rows), so per-epoch `REJECT` recall/FPR are noisy.

| Epoch | Train loss | Val top-1 | Val ECE | Val REJECT recall | Val REJECT FPR | Val EN | Val ZH |
|---|---|---|---|---|---|---|---|
| 1 | 0.641706 | 85.67% | 0.080275 | 81.25% | 8.45% | 91.78% | 79.87% |
| 2 | 0.384151 | 90.00% | 0.036385 | 93.75% | 8.10% | 93.15% | 87.01% |
| 3 | 0.256366 | 94.00% | 0.056254 | 93.75% | 2.46% | 96.58% | 91.56% |
| 4 | 0.132143 | 89.67% | 0.101781 | 100.00% | 8.10% | 89.73% | 89.61% |
| 5 | 0.079365 | 94.67% | 0.054404 | 100.00% | 2.82% | 97.26% | 92.21% |
| 6 | 0.039875 | 94.33% | 0.056691 | 100.00% | 2.82% | 97.26% | 91.56% |
| **7** | **0.013119** | **96.67%** | **0.030319** | **100.00%** | **1.76%** | **100.00%** | **93.51%** |
| 8 | 0.000358 | 96.33% | 0.035276 | 100.00% | 2.11% | 98.63% | 94.16% |
| 9 | 0.000000 | 96.33% | 0.035276 | 100.00% | 2.11% | 98.63% | 94.16% |
| 10 | 0.000000 | 96.33% | 0.035276 | 100.00% | 2.11% | 98.63% | 94.16% |

- Train loss collapses from 0.642 to 0; validation peaks at epoch 7.
- Epochs 9–10 equal epoch 8 because the train loss had reached 0 and the weights stopped
  moving.
- ZH validation sits above EN in the later epochs, the same direction as the final test.

## 12. Validation Selection

The selection rule is fixed and deterministic, and was frozen before the run:

```
best_epoch = argmax(validation_top1); ties resolved to the EARLIER epoch
train loss is recorded and NEVER used to select
```

**Best epoch = 7, Validation v3 top-1 = 0.966667 = 290/300.** No tie-break was needed
(epochs 8–10 tie at 0.963333, below epoch 7).

Top-1 is invariant under the scalar temperature, so the selection does not depend on the
calibration step that follows it. **The Final Test never participated in selection or in
the export gate.**

## 13. Temperature Calibration

Temperature scaling is applied to the **logits**, not to probabilities. A single scalar is
fitted for the `choice` slot only and written into the length-3 temperature vector:

| | |
|---|---|
| Calibration rows | 211 (seed 20260922, the same rows removed from training) |
| Fitted on | epoch-7 (best checkpoint) |
| Raw fitted value | `6.389379501342773` |
| Clamp range | `[0.5, 5.0]` (the Agent's own `clamp_temperature`) |
| Final written value | `5.0` — **clamped** |
| Exported vector | `[5.0, 1.0, 1.0]` |
| Fallback value | `1.2` (not used) |

The clamp range is deliberately the Agent's own `[0.5, 5.0]`, so the fitted value, the
exported config and the Agent's effective value cannot silently disagree. The raw fitted
value exceeded the cap and was clamped to 5.0.

Calibration is a confidence-sharpening step only; it does not change top-1 (§12). It was
fitted on in-distribution held-out rows only and does **not** guarantee reliable confidence
under distribution shift (§16, §19).

## 14. Final Test v3 Results

Best epoch 7, calibrated temperature `[5.0, 1.0, 1.0]`, 1 400 rows:

| Metric | Value |
|---|---|
| **Top-1** | **0.877857** |
| Error rate (= 1 − top-1) | 0.122143 |
| EN (692 rows) | 0.861272 |
| ZH (708 rows) | 0.894068 |
| Capability targets (1 286 rows) | 0.891913 |
| REJECT targets (114 rows) | 0.719298 |
| p1 partition (700 rows) | 0.910000 |
| p2 partition (700 rows) | 0.845714 |
| ECE | 0.097189 |
| Mean confidence | 0.975046 |
| Mean gold probability | 0.871354 |

REJECT detection (positive class = `REJECT`):

| | Value |
|---|---|
| Gold `REJECT` n | 114 |
| Predicted `REJECT` n | 147 |
| TP / FP / FN / TN | 82 / 65 / 32 / 1221 |
| REJECT recall | 0.719298 |
| REJECT precision | 0.557823 |
| REJECT false-positive rate | 0.050544 |

**Scope statement — read this before quoting 87.79 %.** 0.877857 is **benchmark accuracy on
the frozen Final Test v3 split**. It is **not** production-traffic accuracy and **not**
open-world generalization accuracy. The split is a fixed, curated, 1 400-row artifact over
17 known capabilities with a known hard-negative construction and K=3 + `REJECT`; it says
nothing about capabilities introduced after the freeze or about distribution shift.

## 15. Error / Confusion Summary

Top confusion pairs on the Final Test (gold → predicted, count):

| Gold → predicted | n |
|---|---|
| cap-edit-file → REJECT | 12 |
| cap-mindmap → cap-vision | 12 |
| cap-vision → REJECT | 10 |
| cap-pdf-extract-text → REJECT | 9 |
| cap-bash → REJECT | 8 |
| REJECT → cap-edit-file | 7 |
| REJECT → cap-read-file | 6 |
| cap-social-search → cap-web-search | 6 |
| cap-mindmap → REJECT | 6 |
| cap-bash → cap-read-file | 5 |

The residual error splits into two families. First, the **REJECT boundary is the weakest
part**: `REJECT` recall is 71.93 % and precision 55.78 % — 65 capability rows are wrongly
rejected and 32 `REJECT` rows are wrongly routed to a capability. Second, the
**search cluster** (`web-search` / `social-search` / `rag-search`) and `bash`→`edit-file`
remain, exactly as in v1 — those are the pairs whose `B_noprov` cards are most similar by
construction.

## 16. Export and Reload Validation

`--export-dtype preserve` writes the trained **FP32** weights with no conversion.

| Check | Result |
|---|---|
| Export dir | `V2/out/epoch-7/` → published as the model repository root |
| Export dtype | `preserve` (FP32) |
| Size | 1 287 653 720 B |
| `model.safetensors` sha256 | `c6331bd5b6c044a8e4841bee8e9e69962cc2deaf1fd39bffb178703c07168b6f` |
| Export == best checkpoint | **true** — the export is the epoch-7 weights byte-for-byte |
| `laya.Agent` reload | passed, `load_state_dict(..., strict=True)` |
| Reload smoke | **11 / 12 hits** on 12 held-out rows |
| Temperature after reload | `[5.0, 1.0, 1.0]` — fitted == config == effective |
| Exported `max_len` / `head_max_len` / `option_max_tokens` | 1024 / 768 / 256 |

The reload is run through the public entry point (`laya.Agent(dir, device="cpu")`), not by
re-reading the files, so it also proves the published directory is Agent-loadable.

## 17. Artifact Sizes and SHA256

Published per-epoch checkpoint weights (`eric-ml-nlp/Delveta-LayaChoice-v2-checkpoints`):

| Artifact | Size | SHA256 |
|---|---|---|
| `epoch-1/model.safetensors` | 1 287 653 720 B | `30f28e23cf8b3e0fa76fd7b1f74337b49d69b6eec3932e21ebc221314be8203f` |
| `epoch-2/model.safetensors` | 1 287 653 720 B | `ea0f866aed715141733b25b00a643f46392b3a61230ee55e399ce1688866c2d2` |
| `epoch-3/model.safetensors` | 1 287 653 720 B | `27dd2756ec2a7547555fcae2b2acf6d37cc10d456a132fc6754357f1487fcb5b` |
| `epoch-4/model.safetensors` | 1 287 653 720 B | `e04b5375cf46273744d7a6e47698a258278735d74c0061eed227921fc80bc56f` |
| `epoch-5/model.safetensors` | 1 287 653 720 B | `4572dd2d3986b58476685bdc9151458a308d15808904a54d9b09b39120d832dc` |
| `epoch-6/model.safetensors` | 1 287 653 720 B | `a2f1046c5f8068ac13afe86516bc5ece7961eea410adabf590c1ca829218a207` |
| `model.safetensors` (epoch 7, selected) | 1 287 653 720 B | `c6331bd5b6c044a8e4841bee8e9e69962cc2deaf1fd39bffb178703c07168b6f` |

Epochs 8–10 weight files are **not** published (history only; they tie epoch 8 at
0.963333, below epoch 7). Latency/throughput numbers are deliberately omitted: no
performance baseline was established for this model, and inventing one after the fact
would be measuring the wrong thing (see architecture rule 17).

### Published artifacts

| Artifact | Location |
|---|---|
| Selected model (epoch 7), FP32 weights + model card | [Delveta-LayaChoice-v2](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2) (public) |
| Non-selected checkpoints (epochs 1–6) | [Delveta-LayaChoice-v2-checkpoints](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2-checkpoints) (public) |
| Frozen dataset (splits + `manifest.json` + audit) | [Delveta-LayaChoice-v2-Data](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data) (public) |
| Training / evaluation code | [`scripts/laya_finetune/V2/`](https://github.com/Eric-LLMs/Delveta/tree/main/scripts/laya_finetune/V2) |

## 18. Reproducibility

| | |
|---|---|
| GPU | 1× NVIDIA A30 |
| torch | 2.14.1+cu130 |
| transformers | 5.17.0 |
| Python | 3.11 |
| laya | 0.3.21 |
| Base revision | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (subfolder `multilingual`) |
| Seed base | 42; calibration split seed 20260922 |

What must hold for a rerun to be comparable:

1. `laya==0.3.21` and the pinned base revision (asserted on load).
2. The frozen bundle unchanged — the six SHA256s in §7.
3. The same three budget constants, and the same 256-token builder override (§9), on
   training, validation, final test, export and reload.
4. The same selection rule (§12) and the same calibration ids (§13).

## 19. Known Limitations

1. **The 87.79 % is benchmark accuracy on a frozen split**, not production accuracy and not
   open-world generalization (§14).
2. **The `REJECT` class is the weakest part.** On the final test, `REJECT` recall is
   71.93 % / precision 55.78 %; 65 capability rows are wrongly rejected and 32 `REJECT`
   rows are wrongly routed. `REJECT` recall on the small validation split (16 gold rejects)
   is not a reliable estimate.
3. **K is not validated outside 3 (+ `REJECT`).** K=2 and K≥4 post-truncation are unmeasured;
   no number here covers either.
4. **17 capabilities, as of the freeze.** Capabilities added later are out of distribution by
   construction.
5. **The model cannot be deployed on the current renderer.** The production `cap_router` and
   sidecar are v1-shaped and emit no `REJECT` option; v2 requires that option to be rendered
   (§3, §8). Integration is pending.
6. **The dataset is synthetic / curated**, not sampled from live traffic (§5).
7. **EN/ZH only.** No claim is made for any other language.
8. **Base-model license is undetermined.** The upstream repository went private and its
   snapshot ships no license; no license is asserted over the derived weights.

## 20. Final Conclusion

The formal result, on the frozen Final Test v3 (1 400 rows, K=3 + `REJECT`):

| Arm | Top-1 |
|---|---|
| **256/768 fine-tuned, best epoch 7, calibrated** | **0.877857** |
| capability-only subset (1 286 rows) | 0.891913 |
| `REJECT` subset (114 rows) | 0.719298 |

- Validation v3 selected epoch 7 at **0.966667**; the Final Test never participated in
  selection or in the export gate.
- The exported artifact is byte-identical to the selected checkpoint (`c6331bd5…`), passes
  `laya.Agent` `strict=True` reload, and reproduces the fitted temperature exactly.
- v2's job is the one v1 could not do: **abstain when the right capability is not among the
  candidates.** It does so at a **`REJECT` recall of 71.93 %** — useful but the weakest part
  of the model, and the first target of any v3.
- The two splits are **not directly comparable**: v1's 93.56 % was a 3-way 900-row split with
  no `REJECT`; v2's 87.79 % is a harder 4-way split that adds the abstain class.

Scope, restated once more because it is the easiest thing to over-read: **0.877857 is
benchmark accuracy on a frozen, curated, K=3 + `REJECT`, 17-capability, EN/ZH split.** It is
not a production-traffic number and not an open-world claim.
