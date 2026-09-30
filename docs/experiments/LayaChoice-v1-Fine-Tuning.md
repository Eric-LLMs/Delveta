# LayaChoice-v1 — Fine-Tuning Laya as Delveta's Capability-Selection Model

> **Status: frozen.** The formal run finished 2026-09-30 and the artifacts are closed. This
> document is the complete record of that experiment and is not edited after the fact; the
> stable architectural facts distilled from it live in
> [architecture.md §26](../architecture.md#26-layachoice-capability-selection).
>
> Everything below is derived from the run artifacts (`scripts/laya_finetune/out/`) and the
> read-only audits (`/root/_laya_replay/`), not from recollection.

## 1. Experiment Overview

| | |
|---|---|
| Task | Choose the correct capability for a user request, out of a 3-candidate `criteria` set |
| Base model | `convaiinnovations/laya`, subfolder `multilingual`, revision `55cf4c4eb…` |
| Output | `Delveta LayaChoice v1` — full-parameter fine-tune, FP32 export |
| Data | Train 856 / Calibration 85 / Validation v3 150 / Final Test v3 900 (all frozen) |
| Input config | `OPTION_MAX_TOKENS=256`, `head_max_len=768`, `max_len=1024` |
| Best epoch | **2** (selected by Validation v3 top-1 only) |
| Validation v3 (best) | **0.986667** |
| Final Test v3 | **0.935556** — EN 0.914607 / ZH 0.956044 |
| Main zero-shot baseline (256/768) | 0.370000 → **Δ +0.565556** |
| Reference zero-shot (48/256) | 0.578889 → Δ +0.356667 |
| Export | `final/model.safetensors`, FP32 preserve, 1 287 653 720 B, sha256 `8f139df0…` |

One sentence: **the base checkpoint scored 37.00 % on the frozen Final Test; the fine-tune
scored 93.56 %**, with the best checkpoint chosen on a separate validation split that the test
set never touched.

## 2. Objective

Delveta's chat control plane must resolve every turn to one `ExecutionPlan`. When a turn is an
ACTION, something has to pick *which* capability handles it, from the live capability Registry.
The objective of this experiment was to answer one question with a number:

> Can Laya's multilingual decision model be fine-tuned to select a Delveta capability from its
> real, long `B_noprov` capability card — and how much does it beat the same checkpoint
> un-finetuned on the same frozen rows?

Three things were explicitly **not** objectives: not to produce a general chat model, not to
tune for production traffic, and not to re-open the frozen dataset.

## 3. Why LayaChoice

The capability choice is a **closed-set, single-hop decision**: a handful of candidates, one
correct answer, no multi-turn state. Running it through a general chat LLM means paying
autoregressive decode for a `argmax` over ~3 slots.

Laya is architecturally the right shape for that: a non-autoregressive encoder (`mmBERT-base`,
322 M) plus a small typed decision head that scores one option per question. One forward pass,
no decode, and a native `choice` question type. It is a `System 1` decision engine, not a chat
model — which is exactly what the funnel's ToolIntent step needs.

The cost of that fit is the coupling to Laya's own input renderer: the model expects the
question's options rendered the way `laya.common.build_sequence` renders them. Making it work
on Delveta's cards without regressing the decision quality is the entire subject of §9 and §15.

## 4. Base Model

| | |
|---|---|
| Repo | `convaiinnovations/laya` |
| Revision | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (asserted on load) |
| Subfolder | `multilingual` |
| Encoder | `jhu-clsp/mmBERT-base`, 322 M params, context 1024 |
| Package | `laya==0.3.21` (pinned; inference only — the wheel ships no trainer) |
| Head | `DecisionModel`: encoder + 2-layer transformer head + `type_emb(3,d)` + `scorer` (LayerNorm→Linear→GELU→Linear→1) + `act_head` |
| Checkpoint temperature | `[1.0, 1.0, 1.0]` (used as-is for every zero-shot arm) |
| `amp_dtype` | `bf16` |
| Param count | 321 913 430 (1 287 653 720 B ÷ 4, FP32) |

The upstream repository went private after this work started, which is why the recipe was
pinned to an exact revision and to `laya==0.3.21`; `run.sh` refuses to run on any other
version rather than silently training against a different API.

## 5. Dataset Definition

**LayaChoice-v1** is a capability-selection benchmark built from Delveta's live capability
Registry. Each row is one user request paired with three candidate capability cards; one is
gold, the other two are hard negatives.

| | |
|---|---|
| Card view | `B_noprov` — the production card **without** the `query examples:` block and **without** the `evidence:` provenance line |
| Candidates per row | **K = 3** |
| Capabilities | 17 |
| Languages | `zh` / `en` |
| Row shape | `{id, lang, state, order[3], options[3], gold, d1, d2, src_sha}` |

The `B_noprov` view is not cosmetic. The stored registry card (view B) carries an `evidence:`
line that **names the answer in 856/856 training rows** — training on it would leak the label
straight into the input. Dropping that line, and the example block, is what makes the task a
real decision rather than a lookup.

## 6. Train / Calibration / Validation / Test Split

```
source_train_rows            856   ── exclude the frozen 85 calibration rows (seed 20260922)
calibration_rows              85      ──► 85 rows, ids recorded in train_manifest.json
effective_train_rows         771   ── after exclusion, BEFORE any world-size truncation
after_world_size_alignment   771   ── world_size=1, so no truncation
Validation v3                150   ── per-epoch top-1; the ONLY selection signal
Final Test v3                900   ── benchmark of best_epoch only, never used to pick or gate
```

Exclusion happens on the full 856 **before** any world-size truncation, so a multi-rank run
can never pull a calibration row into training. The calibration ids are recorded by index and
by id in `train_manifest.json` and re-verified on every run — the official notebook fixed them
by seed only.

The 900 Final Test rows are **445 EN / 455 ZH**, ~50–58 rows per capability.

## 7. Frozen Dataset SHA256

Verified unchanged at the end of the run (`sha256sum -c data/SHA256SUMS` → all OK).

| Artifact | SHA256 |
|---|---|
| `LayaChoice_v1_train.jsonl` (source) | `bf4139262b9a398aa2821e55d5abf5a0a28384feeab6d8a85b88334557561fa3` |
| `LayaChoice_v1_test_v3.jsonl` (rendered source) | `5a001c6a6b02a3c2fbe67b229d61e6c0b0ff023b7d542b9601697c2a48201f96` |
| `LayaChoice_v1_validation_raw_v3.jsonl` (raw) | `974a3b99cf5883c527677373b53539f3ac859ddfb0ba59ea793058d03a6710ff` |
| `LayaChoice_v1_final_test_raw_v3.jsonl` (raw) | `e0738991d193cae39531a384305dc5d3547ffdd69f9e80b062c91079f51b4613` |
| `layachoice_v1_train_b_noprov.jsonl.gz` | `e968fdeddc8f02eec19f37ed29fafc5f05a591ea259c97315639d366e3de4f13` |
| `layachoice_v1_val_b_noprov.jsonl.gz` | `3fa38aa099580b6f9a5146ac2a8e7beed0111d79d2f994adb6ca317547c90364` |
| `layachoice_v1_test_b_noprov.jsonl.gz` | `c665004a127496f4456a0b3ccd81bd0ce10654fef7808f27e45500d3a9d9b8bb` |

## 8. Candidate Contract

The production contract the model is trained and benchmarked under — **K = 3, always**:

| Candidate count | Behaviour |
|---|---|
| K = 0 | never enters LayaChoice |
| K = 1 | the business layer executes it directly, LayaChoice is not called |
| K = 2 | LayaChoice |
| K = 3 | LayaChoice |
| K ≥ 4 | the business layer truncates to top-3 **first**, then LayaChoice |

**Production candidate normalization must happen before LayaChoice**, not inside it. The
benchmark's fixed K=3 reflects the normal case; the K≥4 truncation is a business-layer
responsibility and is **not** exercised by any number in this document. No result here should
be read as validated for K=2 or for K≥4 post-truncation.

## 9. Token Budget Investigation

`laya.common.build_sequence` hard-codes `max_length=48` on each option, and the checkpoint's
own `head_max_len` is 256. Under view `B_noprov` every frozen option is **137–231 tokens**
(median 172, including the mask token) — so at the stock settings, `0.0 %` of the 2568 training
options keeps a complete `does:` line or any part of the `negative examples:` block. The
discriminative material is always cut.

**`head_max_len` is not a length — it is a combined budget.** `opt_budget = head_max_len −
Σ(options)`; when `opt_budget` drops below 16 the even-share fallback shrinks the options and
`head_ids[: max(8, opt_budget)]` also cuts the instruction head. At the 256 cap `Σ(3 options) +
3 markers` peaks at 599, so the checkpoint's own 256 drives `opt_budget` to ≈ −347: the
fallback cuts every option to 80 tokens (below the shortest true option, 137) and the head to
16. `768` leaves ~169 tokens of slack, so options and the 13-token head both survive whole.

The frozen configuration:

| Constant | Value | Meaning |
|---|---|---|
| `OPTION_MAX_TOKENS` | **256** | per-option hard cap |
| `HEAD_MAX_LEN` | **768** | budget the question head **and all options share** |
| `MAX_LEN` | **1024** | whole-sequence ceiling |

Read-only budget audit at 256/768/1024 (`_laya_final_audit.json`; lengths include the mask token):

| Corpus | rows | option complete | fallback rows | head 13/13 | option min/p50/p95/max | max sequence | over `max_len` |
|---|---|---|---|---|---|---|---|
| Final Test v3 | 900 | 1.0 | 0 | 1.0 | 137 / 172 / 231 / 231 | **631** | no |
| Validation v3 | 150 | 1.0 | 0 | 1.0 | 137 / 172 / 231 / 231 | **634** | no |
| Train `B_noprov` | 856 | 1.0 | 0 | 1.0 | 137 / 172 / 231 / 231 | **628** | no |
| hist corpus 908 view A | 856 | 1.0 | 0 | 1.0 | 96 / 113 / 140 / 140 | 394 | no |

`options_over_cap = 0` on every corpus, and the head-length histogram is `{13: n}` exactly.
`MAX_LEN = 1024` never binds: the longest complete sequence measured is 634.

Config-source consistency was proven separately: `read_cfg(train=True)` = `read_cfg(train=False)`
= `[1024, 768, 256]`, so training, validation, final test, export and reload cannot render the
input differently.

## 10. Historical 53.74% Investigation

An earlier artifact, `_laya_q5_eval2.jsonl`, reported **0.5374** top-1 over the 856-row
historical corpus and was for a while treated as a baseline. It is **not** a valid baseline.

The file was produced by an append-only evaluation script that had its configuration changed
**mid-run**: the script computes over `rows[0:168)`, then the option cap was switched, and it
resumed at `rows[168:856)`. One file therefore contains predictions from two different input
configurations.

Re-running it under a single frozen config exposes the seam:

| Block | n | stored top-1 | 48-cap rerun | agreement 48-cap ↔ stored | 256/768 rerun | agreement 256/768 ↔ stored |
|---|---|---|---|---|---|---|
| all 856 | 856 | 0.5374 | 0.6741 | 0.6308 | 0.4977 | 0.8528 |
| rows 0–167 | 168 | 0.6190 | **0.6190** | **1.0000** | 0.4286 | 0.4583 |
| rows 168–855 | 688 | 0.5174 | 0.6875 | 0.5407 | **0.5145** | **0.9491** |

`rows[0:168)` reproduces exactly under a 48-token cap (agreement 1.0000) and `rows[168:856)`
reproduces almost exactly under a 256-token cap (agreement 0.9491 — the residual is the
difference between the old 256/256 head budget and the current 256/768). The 0.5374 is a
**mixture of two configurations**, and is discarded.

## 11. Historical Mixed 48/256 Artifact Finding

The same seam is why 0.5374 must not be quietly reused as "the old number". Whichever single
config you compare it against, the comparison is invalid:

- against 48/256, the last 688 rows are wrong (they were computed at 256);
- against 256/768, the first 168 rows are wrong (they were computed at 48).

The only defensible reference points are the ones recomputed end-to-end under one config:
**0.6741** (48/256) and **0.4977** (256/768) on the historical 856 view-A corpus. Both are
recorded here, and neither is the main baseline — see §14 and §16.

## 12. 256/256 Invalid Run

An earlier GPU run used `OPTION_MAX_TOKENS=256` with `head_max_len=256` and a **forked**
builder in which the stock even-share fallback had been deleted. It must never be cited.

| | |
|---|---|
| Config | cap 256, head budget 256, custom builder with the stock even-share fallback **removed** |
| Final Test top-1 | 0.3622 |
| Option completeness | 0.0 — the head floor cut the head to 8 tokens |
| Status | **VOID.** Not a result. Checkpoints must not be reused or merged |

Why it is void rather than merely worse: removing the fallback did not fix the budget overflow,
it hid it. With `opt_budget` deeply negative and no fallback, `head_ids[: max(8, opt_budget)]`
silently truncated the *instruction head* to 8 tokens while the options stayed over budget —
so the model was answering a question whose question text had been destroyed. The measured
0.3622 describes a broken input, not the model.

The artifacts are quarantined at `scripts/laya_finetune/out.256cap_INVALID/` — a **sibling** of
`out/`, deliberately outside the archive and outside the release, so it cannot be picked up by
a directory glob.

## 13. Correct stock builder semantics

The fix was to stop forking the builder. The builder in `layachoice_finetune.py` is
`laya.common.build_sequence` **copied byte-for-byte**, with exactly one line changed:
`max_length=48` → `max_length=OPTION_MAX_TOKENS`.

Everything stock is preserved: the `opt_budget = head_max_len − Σoptions` computation, the
`if opt_budget < 16` even-share fallback (`per = max(4, (head_max_len − 16) // max(1, len(opt_ids)))`),
the `head_ids[: max(8, opt_budget)]` floor, the `#538` note, and the `return_stats` dict
including `tokens_per_option`.

Verification is a text diff, not an identity check. Normalising the `C.` module prefixes and
restoring `max_length=OPTION_MAX_TOKENS` → `48`, the diff against `laya.common.build_sequence`
is **empty**. The raw changed lines are exactly five: four `C.`-prefix lines and the cap line.

```diff
-    opts = render_options(q)
+    opts = C.render_options(q)
-    head_ids = _encode_question_text(tok, ...)
+    head_ids = C._encode_question_text(tok, ...)
-        opt_tokens = _encode_question_text(
+        opt_tokens = C._encode_question_text(
-            max_length=48,
+            max_length=OPTION_MAX_TOKENS,
-        state_ids = encode_text(tok, serialize_state(state)...
+        state_ids = C.encode_text(tok, C.serialize_state(state)...
```

Keeping the fallback is the point. `install()` points `laya.agent`'s call site at this builder,
so the baseline, the fine-tune and the benchmark all render under the same code.

*(Caveat worth recording: `_laya_budget_256_768.json` contains `"build_sequence_is_stock": false`.
That field is computed as `F.build_sequence is C.build_sequence` — object **identity**. The
builder is deliberately a distinct copy, so the field is `False` by construction and says
nothing about textual fidelity. The text diff above is the real check.)*

## 14. 48/256 Reference Baseline

The stock rendering (`cap 48`, `head_max_len 256`) was kept as a **reference-only** zero-shot
arm: it answers "how does the untouched checkpoint score under its own native input
distribution?", not "what should we ship?".

| | |
|---|---|
| Final Test v3 (900) | **0.578889** (EN 0.586517 / ZH 0.571429) |
| Historical 856 view A | 0.6741 |
| Option complete | 0.0 |
| max sequence | 188 |
| pred role | gold 521 / d1 263 / d2 116 |
| gold→d1 / gold→d2 | 0.292222 / 0.128889 |

**This arm is not the main training configuration and its higher zero-shot score is not a
reason to change the config.** The reference is higher because the base checkpoint was trained
on Laya's own 48-token rendering and Delveta's long cards push it off that distribution — which
is precisely what fine-tuning is for (§15, §16).

## 15. 256/768 Main Training Configuration

| | |
|---|---|
| `OPTION_MAX_TOKENS` | 256 |
| `HEAD_MAX_LEN` | 768 |
| `MAX_LEN` | 1024 |
| Candidates K | 3 |
| Card view | `B_noprov` |
| Fine-tuning | **full-parameter** (no PEFT/LoRA) |
| Precision | BF16 autocast, `GradScaler` disabled (bf16), weights kept FP32 |
| Hardware | NVIDIA A30 24 GB, world_size 1 |

Rationale, in one line each:

- **256, not 48** — the dataset was audited at 256: `over_library_cap_48 = 2568`,
  `over_patched_cap_256 = 0`. At 256 nothing is truncated.
- **768, not 256** — `head_max_len` is a combined head+options budget; 256 cannot cover
  `Σ(3 options) + 3 markers ≈ 599` and triggers the fallback (§9).
- **1024 unchanged** — the longest complete sequence is 634, so it never binds.

**Deployment consequence:** a stock `laya` install renders at 48 tokens and `head_max_len`
256. A deployment must either apply the same override
(`layachoice_finetune.install()`) and pass the same budget before constructing the Agent, or
expect a materially lower score under stock rendering.

## 16. Zero-shot Baseline

All three arms are the **same base checkpoint, same weights, same rows, temperature
`[1,1,1]`, no calibration** — only `(option_cap, head_max_len)` differ. The main baseline is
the arm that matches the shipped config.

| Arm | cap | `head_max_len` | Final Test v3 (900) | EN | ZH | hist 856 view A | option complete | max seq |
|---|---|---|---|---|---|---|---|---|
| A — library default | 48 | 256 | 0.5789 | 0.5865 | 0.5714 | 0.6741 | 0.0 | 188 |
| **B — main baseline** | **256** | **768** | **0.3700** | **0.3978** | **0.3429** | **0.4977** | **1.0** | **631** |
| C — invalid | 256 | 256 | 0.4444 | — | — | 0.4813 | 0.0 | 281 |

Arm B on the historical 856 view-A corpus: 0.4977 (EN 0.5494 / ZH 0.4490).

Reproduced directly from the checkpoint for this experiment (`_laya_final_audit.json`), then
re-run as a formal stage (`out/baseline/baseline.report.json`, sha `4b8e368b…`; identical to
`out/baseline_256_768/baseline.report.json`).

**The counter-intuitive result, recorded so it is not re-discovered:** zero-shot accuracy
falls monotonically as the option budget grows — 48 → 0.579, 80 (fallback) → 0.444, 137–231
(complete) → 0.370. Two things follow. First, "truncating options loses discriminative
information" is true information-theoretically but **false for this un-finetuned checkpoint**,
which was trained to expect the 48-token rendering. Second, the head length is *not* the
driver: the head-8 invalid run (0.3622) is barely different from the head-13 main baseline
(0.3700). The drop tracks the options.

## 17. Fine-tuning Hyperparameters

| | |
|---|---|
| Epochs | 4 |
| Micro batch / grad accum | 8 / 4 (effective batch 32) |
| Group size (GRPO-style term) | 4 |
| `lr_encoder` / `lr_head` | 2.5e-5 / 1.0e-4 |
| Weight decay | 0.01 |
| Optimizer | `AdamW([{encoder: 2.5e-5}, {head: 1.0e-4}], weight_decay=0.01)` |
| Scheduler | `CosineAnnealingLR(T_max=96, eta_min=1e-6)` |
| Gradient clip | `clip_grad_norm_(1.0)` |
| σ schedule | 0.4 → 0.1 across epochs |
| Loss | `loss_rl + 1.0 * loss_ce`; `w_sph=0.75`, `w_rps=1.0` |
| AMP | bf16 autocast, `GradScaler(enabled=False)` |
| Seed | `seed_base=42`; per-epoch `42 + epoch` (ranks 0) |
| Total updates (planned `T_max`) | 96 |

`total_updates=96` is the scheduler horizon `(771 // 32) * 4`; the **actual** optimizer steps
were **25 per epoch = 100**, because the last partial gradient-accumulation group also forces a
step. The scheduler therefore ran slightly past its `T_max`, which is expected and benign for
`CosineAnnealingLR`.

## 18. Epoch 1–4 Training Results

From `train.log` and the four `epoch-*/train_meta.json` files (all four retained).

| Epoch | σ | train loss | LR encoder | LR head | micro-batches | updates | sec | Val v3 top-1 | EN | ZH | Val ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.4 | 0.320957 | 2.1203e-05 | 8.4338e-05 | 97 | 25 | 21.35 | 0.933333 | 0.945205 | 0.922078 | 0.065180 |
| **2** | 0.3 | 0.063713 | 1.2215e-05 | 4.7263e-05 | 97 | 25 | 20.91 | **0.986667** | 0.986301 | 0.987013 | 0.013017 |
| 3 | 0.2 | 0.056521 | 3.7239e-06 | 1.2236e-05 | 97 | 25 | 21.04 | 0.980000 | 0.958904 | 1.000000 | 0.016209 |
| 4 | 0.1 | 0.001806 | 1.1027e-06 | 1.4235e-06 | 97 | 25 | 20.90 | 0.980000 | 0.958904 | 1.000000 | 0.019153 |

- Total wall clock: **84.2 s** for all four epochs; peak training memory ≈ **6579 MiB / 24576 MiB**.
- Train loss collapses from 0.321 to 0.0018; validation saturates at epoch 2.
- ZH validation reaches 1.0000 at epochs 3 and 4 while EN plateaus at 0.9589 — the same
  direction as the final test (ZH above EN), visible already on the small split.

## 19. Validation Selection

The selection rule is fixed and deterministic, and was frozen before the run:

```
best_epoch = argmax(validation_top1); ties resolved to the EARLIER epoch
train loss is recorded and NEVER used to select
```

| Epoch | Val v3 top-1 |
|---|---|
| 1 | 0.933333 |
| **2** | **0.986667** ← argmax |
| 3 | 0.980000 |
| 4 | 0.980000 |

**Best epoch = 2.** No tie-break was needed (epochs 3 and 4 tie with each other, both below 2).

Top-1 is invariant under the scalar temperature (`softmax(logits/T)` is monotone in
`logits/T`), so the selection does not depend on the calibration step that follows it — the
calibration cannot retroactively change which checkpoint was selected.

**The Final Test never participated in selection or in the export gate.** `run.sh` exports the
best checkpoint whether or not it beats the baseline; the benchmark only records the verdict
(`beats_baseline: true`).

## 20. Temperature Calibration

Fitted on the frozen 85-row calibration set (seed 20260922), on the best checkpoint, for the
`choice` slot only.

| | |
|---|---|
| Calibration rows | 85 |
| Fitted on | `out/epoch-2` |
| Raw fitted value | 2.038968563079834 |
| Clamp range | `[0.5, 5.0]` (`laya.common.clamp_temperature` = the Agent's own range) |
| Final written value | 2.038968563079834 — **no clamping occurred** |
| Exported vector | `[2.038968563079834, 1.0, 1.0]` |
| Temperature slots | size-3 list, `QTYPES = {choice: 0, score: 1, noul: 2}`; only slot 0 is fitted |

The clamp range is deliberately the Agent's own `[0.5, 5.0]`, **not** the official notebook's
wider `[0.1, 10.0]`: the Agent applies `clamp_temperature` on load, so a value fitted outside
that range would be silently rewritten and the "fitted value == exported config == Agent's
effective value" agreement would break. `temperature.json` records both `raw_fitted_value` and
`final_written_value`; the reload check reads the effective value back from a real
`laya.Agent` and compares all three.

Calibration is a confidence-sharpening step only. It does not change top-1 (§19).

## 21. Final Test v3 Results

Best epoch 2, calibrated temperature `[2.038968563079834, 1.0, 1.0]`, 900 rows,
`out/benchmark/final_test.report.json` (sha `d25d6e50…`) + full per-row
`final_test.rows.jsonl` (sha `6644d278…`).

| Metric | Value |
|---|---|
| Top-1 | **0.935556** |
| Hard-negative rate (= 1 − top-1) | 0.064444 |
| EN (445 rows) | 0.914607 |
| ZH (455 rows) | 0.956044 |
| ECE | 0.029528 |
| Mean confidence | 0.959938 |
| Mean gold probability | 0.923789 |
| Predicted gold / d1 / d2 | 842 / 42 / 16 |
| gold→d1 | 0.046667 |
| gold→d2 | 0.017778 |

| Comparison | Value |
|---|---|
| 256/768 zero-shot (main baseline) | 0.370000 → **Δ +0.565556** |
| 48/256 reference zero-shot | 0.578889 → Δ +0.356667 |
| `beats_baseline` | true |

**Scope statement — read this before quoting 93.56 %.** 0.935556 is **benchmark accuracy on the
frozen Final Test v3 split**. It is **not** production-traffic accuracy and **not** open-world
generalization accuracy. The split is a fixed, curated, 900-row artifact over 17 known
capabilities with K=3 and a known hard-negative construction; it says nothing about
capabilities introduced after the freeze, about K≥4 post-truncation, or about distribution
shift in live traffic.

## 22. Error / Confusion Summary

Per-capability top-1 on Final Test v3 (17 capabilities, ~50–58 rows each):

| Capability | n | top-1 | pred d1 | pred d2 |
|---|---|---|---|---|
| cap-add-term | 53 | 1.000000 | 0.0 | 0.0 |
| cap-artifact | 54 | 0.981481 | 0.018519 | 0.0 |
| cap-bash | 53 | 0.735849 | 0.264151 | 0.0 |
| cap-create-folder | 55 | 0.981818 | 0.0 | 0.018182 |
| cap-edit-file | 54 | 0.981481 | 0.018519 | 0.0 |
| cap-mindmap | 50 | 0.980000 | 0.020000 | 0.0 |
| cap-pdf-extract-text | 49 | 1.000000 | 0.0 | 0.0 |
| cap-pdf-table-to-text | 50 | 1.000000 | 0.0 | 0.0 |
| cap-rag-search | 54 | 0.907407 | 0.037037 | 0.055556 |
| cap-read-document | 55 | 0.981818 | 0.0 | 0.018182 |
| cap-read-file | 55 | 1.000000 | 0.0 | 0.0 |
| cap-slides | 51 | 1.000000 | 0.0 | 0.0 |
| cap-social-search | 57 | 0.877193 | 0.052632 | 0.070175 |
| cap-summary | 50 | 0.900000 | 0.100000 | 0.0 |
| cap-translate | 51 | 1.000000 | 0.0 | 0.0 |
| cap-vision | 58 | 1.000000 | 0.0 | 0.0 |
| cap-web-search | 51 | 0.568627 | 0.294118 | 0.137255 |

Top confusion pairs (gold → predicted, count):

| Confusion | n |
|---|---|
| cap-web-search → cap-social-search | 15 |
| cap-bash → cap-edit-file | 14 |
| cap-web-search → cap-rag-search | 7 |
| cap-summary → cap-mindmap | 5 |
| cap-social-search → cap-rag-search | 4 |
| cap-rag-search → cap-social-search | 3 |
| cap-social-search → cap-web-search | 3 |
| cap-rag-search → cap-web-search | 2 |
| cap-artifact → cap-slides | 1 |
| cap-create-folder → cap-bash | 1 |

The residual error is **semantically clustered, not random**: 44 of the 58 misses are 1–2
capabilities near the gold, and a single tight cluster (`web-search` / `social-search` /
`rag-search`) accounts for 30 of them. `cap-bash → cap-edit-file` (14) is the one non-search
cluster. These are the pairs whose `B_noprov` cards are most similar by construction — which is
the intended difficulty of the benchmark, and the honest description of what
93.5556 % leaves on the table.

## 23. Export and Reload Validation

`--export-dtype preserve` (the default) writes the trained **FP32** weights with no conversion.

| Check | Result |
|---|---|
| Export dir | `out/final/` — `model.safetensors`, `rl_agent_config.json`, `encoder/`, `tokenizer/` |
| Export dtype | `preserve` (FP32) |
| Size | 1 287 653 720 B |
| `final/model.safetensors` sha256 | `8f139df0fed909f8b66c09e9064db424536bc90c274b6fd08ae4485edee20ba6` |
| `epoch-2/model.safetensors` sha256 | `8f139df0fed909f8b66c09e9064db424536bc90c274b6fd08ae4485edee20ba6` |
| Export == best checkpoint | **true** — the export is the selected checkpoint's weights byte-for-byte |
| `laya.Agent` reload | passed, `load_state_dict(..., strict=True)` |
| Reload smoke | **5 / 5 hits** on 5 held-out rows |
| Temperature after reload | `[2.038968563079834, 1.0, 1.0]` — fitted == config == effective |
| `temperature_by_options` | dropped from the exported config |
| Exported `max_len` / `head_max_len` / `option_max_tokens` | 1024 / 768 / 256 |

The reload is run through the public entry point (`laya.Agent(dir, device="cpu")`), not by
re-reading the files, so it also proves the directory is Agent-loadable and that the three-way
temperature agreement survives a real load.

## 24. Artifact Sizes and SHA256

Run artifacts (`scripts/laya_finetune/out/SHA256SUMS`, verified after the run):

| Artifact | Size | SHA256 |
|---|---|---|
| `epoch-1/model.safetensors` | 1 287 653 720 B | `1f30e6de5df82da8aec69d1a2752acc908307b69374776aa613c59df9a93763f` |
| `epoch-2/model.safetensors` | 1 287 653 720 B | `8f139df0fed909f8b66c09e9064db424536bc90c274b6fd08ae4485edee20ba6` |
| `epoch-3/model.safetensors` | 1 287 653 720 B | `6e441b4e26c6bd2bab6e77de2e716197d10463629318627770db6cf080f946e7` |
| `epoch-4/model.safetensors` | 1 287 653 720 B | `cc39c5a06d16fdb0b987bb45fde09e0b30f3b7031f2729bc8e3273467ad4a63c` |
| `final/model.safetensors` (= epoch-2) | 1 287 653 720 B | `8f139df0fed909f8b66c09e9064db424536bc90c274b6fd08ae4485edee20ba6` |
| `final/rl_agent_config.json` | 744 B | `5d7ded751ed46ec455b333847b63cd139e15dd5608bd15231901459ca9fecd92` |
| `final/tokenizer/tokenizer.json` | 34 363 188 B | `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f` |
| `benchmark/final_test.report.json` | 3 767 B | `d25d6e501a592ba5b636249b15dc9f604c832c9166d87286cd2423ad686b71da` |
| `benchmark/final_test.rows.jsonl` | 382 177 B | `6644d278c55149d31c9321d29d5c87176b57f330598de3c157327738d419a0d7` |
| `select/selection.json` | — | `809a805d86ffe707435fe493a985b30e6bf1e708e2c7b99aa2c28c7bcf72c10b` |
| `select/epoch-1.val.report.json` | — | `259cb593d5a91801f64cbe5131584b5cb87c2016e45e4863a64ddf9e975545c5` |
| `select/epoch-2.val.report.json` | — | `bb650bf54d32e2b7e95f73911b9803c788bc60cf4d24acc0cd547fabda96bb1f` |
| `select/epoch-3.val.report.json` | — | `2d5de1fce09cbf05447018f1e3b6f5cb2abb483785d84cafbd2467ad71c5fffc` |
| `select/epoch-4.val.report.json` | — | `806aaa6fa7ad27e963604249b630ef60d431d329d69f4c4d4f209b3fe784a425` |
| `calibration/temperature.json` | — | `332830d6db8d654ae6010386c7db5c0f5836df7c2dfa45fbcd770da381266a35` |
| `train_manifest.json` | 7 743 B | `e6d53f71f4a72acefe120f9e19ab0e26bf3cd05dc151c7ed2d9fd7aa84ddfb7d` |
| `baseline/baseline.report.json` (256/768) | 3 645 B | `4b8e368bfe0e3607cfe8c5a5ad99e5170f55f287b157b2ec4ee862a9037f78bf` |
| `baseline/baseline.rows.jsonl` | 373 647 B | `afb27bcbd6ad0b8ecb1b9e5a59986a42e15a3ffcc1baa639c4f7ab4fea88bc27` |
| `baseline_48_256/baseline.report.json` | 3 644 B | `e5cca8b4ab902cebd4fdc841f0ac7759c1a374a4bd2d80500bd2928842f2d0e7` |

Per-epoch training state (optimizer + scheduler + scaler + RNG), not part of the deployment
artifact: `epoch-N/train_state.pt` at 2 575 429 585 B each; `epoch-N/train_meta.json` at 291 B.

Latency/throughput numbers are deliberately omitted: no performance baseline was established
for this model before or after the experiment, and inventing one after the fact would be
measuring the wrong thing (see architecture rule 17).

### Known metadata defect in the deployment artifact

`final/rl_agent_config.json` carries a `training` block that is **inherited verbatim from the
base checkpoint** and describes *upstream Laya's* training, not this fine-tune:

```json
"training": {
  "updates": 15987, "epochs_completed": 4, "hours": 4.97,
  "world_size": 1, "fine_tuned_from_checkpoint": false
}
```

This run did **100** optimizer updates in **84.2 s**. The cause is mechanical: the export does
`exported = dict(cfg)` on the base config, then overrides `fine_tuned`, `temperature`,
`model_name` and `layachoice` — it never rewrites `training`. The block is inert at runtime
(`laya.Agent` does not read it) and affects no weight, no score and no SHA of
`model.safetensors`; it is recorded here because a reader of the artifact would otherwise be
misled. The authoritative training metadata is `train_manifest.json` and the four
`epoch-*/train_meta.json`.

### Published artifacts

The selected model and evaluation output are published separately from this source
repository.

| Artifact | Location |
|---|---|
| Selected model (epoch 2), FP32 weights + eval reports | [Delveta-LayaChoice-v1](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1) (public) |
| Non-selected checkpoints (epoch 1 / 3 / 4) | [Delveta-LayaChoice-v1-checkpoints](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v1-checkpoints) (public) |
| Frozen dataset (bundles + raw + `SHA256SUMS`) | [`scripts/laya_finetune/data/`](https://github.com/Eric-LLMs/Delveta/tree/main/scripts/laya_finetune/data) |
| Training / evaluation code | [`scripts/laya_finetune/`](https://github.com/Eric-LLMs/Delveta/tree/main/scripts/laya_finetune) |

The SHA256 values in the table above are the authority for every published copy; the
uploaded weights were verified against them after upload.

## 25. Reproducibility

Environment of the formal run (`release_manifest.json` → `environment`, `versions`):

| | |
|---|---|
| GPU | NVIDIA A30, 24 GB |
| CUDA | 13.0 (bf16 supported) |
| torch | 2.11.0+cu130 |
| transformers | 5.7.0 |
| Python | 3.10.20 |
| laya | 0.3.21 |
| Platform | Linux-6.8.0-100-generic-x86_64 |
| world_size | 1 |

Stages, in order:

```bash
# repo host, once (needs the live capability registry) — already done; bundle is frozen
.venv/Scripts/python.exe scripts/laya_finetune/layachoice_prepare.py

# GPU box
python layachoice_eval.py --baseline-zero-shot --out out/baseline
python layachoice_finetune.py --train --out out
python layachoice_eval.py --select --run out --out out/select
python layachoice_finetune.py --fit-temperature --ckpt out/epoch-2
python layachoice_eval.py --benchmark --ckpt out/epoch-2 --out out/benchmark \
                          --baseline out/baseline/baseline.report.json
python layachoice_finetune.py --export --ckpt out/epoch-2 --out out/final \
                              --export-dtype preserve
```

What must hold for a rerun to be comparable:

1. `laya==0.3.21` and base revision `55cf4c4eb…` (asserted on load).
2. The frozen bundle unchanged — all seven SHA256s in §7.
3. The same three budget constants, read by **all** paths via `read_cfg` (§9).
4. The same selection rule (§19) and the same calibration ids (§20).
5. `git` state: the run executed from the working tree at
   `132499cfec3f9e4b79bfb829afff56d3cfb81498`; `release_manifest.json.git.previous_head`
   records it, and `release_commit` is filled in by the documentation commit that follows.
   The three `scripts/laya_finetune/*.py` files were verified byte-identical between the GPU
   box and this repository (`sha256` match), so the archived code is the code that ran.

Determinism caveat: seeds are fixed (`seed_base=42`, per-epoch `42+epoch`, plus explicit
NumPy/Torch/CUDA seeding), and each `epoch-N/train_state.pt` stores Python/NumPy/Torch/CUDA
RNG state. Bit-exact reproduction across a different GPU or torch build is **not** claimed —
cuDNN/cuBLAS kernel selection is not fixed — so a rerun should reproduce the *decision* (best
epoch 2, same selection) and the metrics to within normal run-to-run noise, not the bytes.

## 26. Known Limitations

1. **The 93.56 % is benchmark accuracy on a frozen split**, not production accuracy and not
   open-world generalization (§21).
2. **K is not validated outside 3.** K=2 is in contract but unmeasured in this run; K≥4 is a
   pre-LayaChoice truncation the model never sees. No number here covers either.
3. **17 capabilities, as of the freeze.** Capabilities added later are out of distribution by
   construction. Adding one requires a new candidate-set evaluation, not a re-read of 93.56 %.
4. **The zero-shot ordering is counter-intuitive and must not be "fixed".** The base model
   scores better at 48 tokens than at 256 (§16). Changing the config back to 48 on the strength
   of the zero-shot number would discard the entire reason for the fine-tune.
5. **The residual error is a semantic cluster.** ≈30 of 58 misses sit in
   `web-search`/`social-search`/`rag-search`, plus `bash`→`edit-file` (14). Improving past
   93.56 % means separating those cards, not adding training data at large.
6. **No latency/throughput baseline exists** for this model, so no performance claim is made
   (§24). A separate performance task would be needed to establish one.
7. **The deployment artifact's `training` block is inherited from the base checkpoint** and is
   wrong for this run (§24). Treat `train_manifest.json` as the authoritative source.
8. **Base-model license is undetermined.** The upstream repository went private, the cached
   checkpoint ships no license field, and this repository is AGPL-3.0. No new license should be
   asserted over the derived weights until that is resolved (§27, §29).
9. **EN/ZH only.** The multilingual checkpoint covers more languages than the task does; no
   claim is made for any language other than the two in the frozen split.

## 27. INT8 Follow-up

Read-only compatibility audit. **No INT8 artifact was generated**, and no INT16/INT32 artifact
will be: the FP32 export is the master, and INT8 is only ever a *derived* deployment artifact.

### Structure and load path

| Component | Detail | INT8 relevance |
|---|---|---|
| Encoder | `jhu-clsp/mmBERT-base`, ModernBERT family, SDPA attention, ~3.2e8 params | the bulk of the size; the natural quantization target |
| Decision head | 2 × `TransformerEncoderLayer(d, nhead=d//64, ff=4d, norm_first)` + `_DynamicMultiheadAttention` | small; attention over markers |
| `type_emb` | `nn.Embedding(3, d)` | tiny; quantization pointless |
| `scorer` | `LayerNorm → Linear(d,d) → GELU → Linear(d,1)` | produces the logits — quantization-sensitive |
| `act_head` | `Linear(d+4,256) → GELU → Linear(256,n_act)` | unused by the `choice` path in production |
| `temperature` | registered buffer, size 3, fp32, clamped on load | must stay exact |
| Forward detail | `logits = self.scorer(m).squeeze(-1).float()` then `masked_fill(~marker_mask, -1e4)` | logits are explicitly upcast to fp32; a quantized scorer must preserve that |
| Load path | `laya.Agent` → `build_model(..., pretrained=False)` (meta device) → `load_file(model.safetensors)` → `_verify_compatibility()` → `load_state_dict(strict=True)` | **the hard gate** |

`_verify_compatibility()` requires the encoder/type_emb/scorer/act_head prefixes to exist and
**every named parameter to match by name and shape**, and `load_state_dict` is then called with
`strict=True`. Any drop-in quantization that changes module types (bitsandbytes, torch
`quantize_dynamic`) changes the state dict — extra `SCB`/`absmax`/`weight_format` entries and
int8 `weight` shapes — and is therefore **rejected by `laya.Agent` before it runs**.

### Route assessment

| Route | Verdict | Why |
|---|---|---|
| **bitsandbytes INT8** | **not viable** | `LLM.int8()` needs CUDA (the target is a local PC); replacing `nn.Linear` breaks `_verify_compatibility` + `strict=True` |
| **`torch.ao.quantization.quantize_dynamic`** | **not viable as-is** | CPU-capable and Python-only, but produces `nn.quantized.dynamic.Linear` — same strict-load rejection; would need a bespoke loader that bypasses `laya.Agent` |
| **ONNX Runtime INT8** | **recommended route** | `laya.ONNXAgent` already exists in the wheel and loads a graph via `onnxruntime.InferenceSession` — it **never calls `load_state_dict`**, so the strict-load gate does not apply. ORT dynamic quantization is CPU-native, matching the local-deployment target |
| INT8 encoder + FP32 head | **allowed, and expressible in the ONNX route** | ORT can quantize a node subset (`op_types_to_quantize` / node exclusion), leaving the `scorer` subgraph in FP32 — which also protects the `.float()` logits path |

The ONNX route has one real prerequisite: **the wheel ships no exporter.** `onnx_agent.py`
consumes a `laya.onnx` produced elsewhere; nothing in `laya==0.3.21` writes one. Exporting the
`DecisionModel.forward` is feasible (dynamic axes for rows/tokens/markers) but must handle the
forward's shape-dependent branch in `act_head` (`p.size(-1) >= 2`): tracing with fewer than 2
markers would specialise the wrong branch. Under the K≤3 contract every LayaChoice call has
≥2 options, so a trace taken with ≥2 markers matches production — but that is a property to
**verify**, not assume.

### What a follow-up must verify before any INT8 artifact ships

1. The exported graph reproduces FP32 logits within tolerance on all 900 Final Test rows.
2. K≤3 behaviour and the capability mapping are unchanged (top-1 identical, or the delta
   characterised per capability).
3. Temperature handling survives: the temperature is applied *outside* the graph
   (`z = logits[:k] / t_scale` in the Agent), so it must stay fp32 and outside quantization.
4. Measured accuracy / latency / memory against the FP32 export on the local PC.

**Local load test status: not run.** Per the scope ruling for this phase (audit and design
only), no INT8 artifact exists to load. The audit's conclusion is that the ONNX Runtime route
is the only one compatible with Laya's load path without a bespoke loader, and that it is
blocked on writing an exporter that the `laya` wheel does not provide.

## 28. Invalid Experiments / Lessons Learned

| Artifact | What it was | Why it is void |
|---|---|---|
| `out.256cap_INVALID/` | cap 256 + head budget 256 + custom builder with the stock even-share fallback **deleted**; Final Test 0.3622 | the fallback had been hiding a budget overflow; without it the head floor cut the *instruction head* to 8 tokens. Not a result; checkpoints must not be reused or merged |
| `_laya_q5_eval2.jsonl` 0.5374 | a 856-row evaluation file | append-only script had its config changed mid-run: rows 0–167 at cap 48, rows 168–855 at cap 256. A mixture of two configurations (§10, §11) |
| 48/256 as "the" baseline | the stock-rendering zero-shot score, 0.5789 | it is higher than the main baseline, which invited the wrong conclusion. Reference only (§14) |

Lessons, in the order they cost time:

1. **A budget overflow does not announce itself.** `head_max_len` reads like a length; it is a
   head+options budget. The failure mode is silent truncation of the *question*, and the score
   drops without any error.
2. **Never fork a library's input renderer.** The fork that dropped the fallback scored 0.3622
   and produced a month of confusion. The correct fix was one line — raise the cap — in a
   byte-verbatim copy.
3. **Verify a "same as upstream" claim with a text diff, not an identity check.** `f is g` is
   `False` for a deliberate copy and says nothing (see the `build_sequence_is_stock` caveat in
   §13).
4. **An append-only evaluation script must stamp its config.** A mid-file config change turned
   one run into two half-runs and produced a plausible-looking 0.5374 that was pure artifact.
5. **Record the zero-shot arms before training, not after.** The 48 → 80 → complete ordering
   (§16) is the single most surprising fact in this experiment; it is only trustworthy because
   it was measured under one code revision on the same rows.
6. **A higher reference score is not a reason to change the config.** The goal was to learn
   Delveta's real long cards. The 48/256 arm measures the old distribution, by construction.

## 29. Final Conclusion

The formal result, on the frozen Final Test v3 (900 rows):

| Arm | Top-1 |
|---|---|
| **256/768 fine-tuned, best epoch 2, calibrated** | **0.935556** |
| 256/768 zero-shot (main baseline) | 0.370000 |
| 48/256 zero-shot (reference only) | 0.578889 |
| 256/256 invalid run (void) | 0.362200 |

- **Δ vs the main baseline: +0.565556** (37.00 % → 93.56 %).
- Validation v3 selected epoch 2 at **0.986667**; the Final Test never participated in
  selection or in the export gate.
- The exported artifact is byte-identical to the selected checkpoint
  (`8f139df0…`), passes `laya.Agent` `strict=True` reload, and reproduces the fitted
  temperature exactly.
- The residual 6.44 % is a semantic cluster, not noise: `web-search`/`social-search`/
  `rag-search` account for 30 of 58 misses, `bash`→`edit-file` for 14.

**The long-card question is answered.** At 256/768 the un-finetuned checkpoint scores 37.00 %
— worse than its own 48-token rendering — because Delveta's `B_noprov` cards fall outside the
input distribution it was trained on. After full-parameter fine-tuning on 771 rows the same
checkpoint scores 93.56 % on the identical frozen test, with every option rendered complete
and every instruction head intact. The 256/768 configuration is therefore the shipped one, and
48/256 remains what it was demoted to: a reference baseline describing the old input
distribution.

Scope, restated once more because it is the easiest thing to over-read: **0.935556 is
benchmark accuracy on a frozen, curated, K=3, 17-capability, EN/ZH split.** It is not a
production-traffic number and not an open-world claim.
