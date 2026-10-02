# LayaChoice-v2 — fine-tuning Laya's multilingual decision model

`convaiinnovations/laya` is a non-autoregressive decision engine ("system one"): an encoder
plus a small scoring head that picks one option out of a question's `criteria` dict. This
directory fine-tunes the **multilingual** checkpoint on **LayaChoice-v2**, and measures the
result against the same checkpoint un-finetuned on the same frozen test set.

v2 extends the v1 capability-selection task from 3-way to **4-way**: every question shows
**three capability cards plus one `REJECT` card** (`OPTION_SLOTS = 4`), so the model can
abstain when the correct capability is not among the candidates. `REJECT` is a normal fourth
option — trained directly, not a threshold and not a post-hoc rule. The base checkpoint,
token budget, seeds and training hyper-parameters are the frozen v1 values; **v2 differs from
v1 only in the dataset/task**, never by silently re-tuning the recipe.

> **Production integration status.** The production `cap_router` card renderer and the
> `deploy/laya` sidecar still render the 3-option **v1** question and emit no `REJECT`
> criteria, so v2 is **not yet wired into production**. A caller must render the `REJECT`
> card as a fourth criteria; a 3-option question does not match the trained pipeline and its
> scores are not meaningful.

```
scripts/laya_finetune/V2/
  README.md                     this file
  layachoice_v2_spec.py         the frozen constants: token budget, base checkpoint, seeds, hyper-parameters
  layachoice_v2_render.py       the model-input boundary: the 256-token builder and install()
  layachoice_v2_dataset.py      the dataset contract: adapter (V2Example) + manifest-checked loader
  layachoice_v2_finetune.py     stage 2: --train / --fit-temperature / --export
  layachoice_v2_eval.py         stage 3: --baseline-zero-shot / --select / --benchmark
  data/                         the frozen bundle, self-contained and also the GitHub backup
    v2_{train,val,test}.jsonl   the three splits
    manifest.json               shas, row counts, calibration ids, part/language counts, base revision
    v2_build_report.json        the construction report
    v2_construction_audit.jsonl the per-row construction audit
  frozen_backup/                the pre-adjustment Test split and its SHA256 record
    v2_test_1800_pre_adjustment.jsonl
    FROZEN_SHA256.json
  out/                          training artifacts (not committed)
```

## Deploy

The GPU box needs no registry, no database and no repo checkout beyond this directory.

```bash
# 1. env
python3 -m venv .venv && . .venv/bin/activate
pip install "laya==0.3.21" torch transformers safetensors huggingface_hub numpy

# 2. smoke first: CPU, float32, no GradScaler, no DDP, 8 rows per split, 1 training step
python layachoice_v2_finetune.py --train --smoke
python layachoice_v2_eval.py --baseline-zero-shot --out out/baseline --smoke

# 3. the real run (10 epochs, bf16 on Ampere+, fp16 + GradScaler on T4)
python layachoice_v2_finetune.py --train --epochs 10 --out out
```

Unlike v1 there is no `run.sh`; the stages are invoked directly (see *Stages* below), and each
stage re-runs independently.

`laya` is pinned to `0.3.21`. The upstream repository went private after this recipe was
written, so the modules fail loudly on any other version rather than silently training on a
different API.

Disk: each epoch checkpoint is ~3.9 GB (1.3 GB fp32 weights + 2.6 GB AdamW state), so budget
~40 GB for ten epochs.

## Recipe

Fine-tuning uses `transformers` + `torch` directly, not PEFT or the Trainer, because the
`laya` wheel ships inference only — `laya.common.build_model` / `proper_reward` /
`collate_items` are reusable, but there is no trainer. The recipe is the frozen v1 recipe
(item construction, then the official notebook's `train_ddp.py` loop), reproduced here:

| | |
|---|---|
| checkpoint | `convaiinnovations/laya`, subfolder `multilingual`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (asserted on load) |
| encoder | `jhu-clsp/mmBERT-base`, 322 M params, ctx 1024 |
| items | `build_sequence` → `{ids, markers, qtype=QTYPES["choice"], target, label}`; a one-hot target at `gold_index`, `label = gold_index`, over four slots (3 capabilities + `REJECT`) |
| loss | `loss_rl + 1.0*loss_ce`, where `loss_rl` is a GRPO-style group-advantage term over `proper_reward` (log score + 0.75·spherical) at `GROUP_SIZE=4`, and `loss_ce` is a masked cross-entropy on the gold slot |
| noise | `eps ~ N(0, sigma²)` zero-mean-projected onto the simplex-perpendicular and masked; `sigma` 0.4 → 0.1 across epochs; the advantage is standardized within the group |
| optim | `AdamW([{encoder: 2.5e-5}, {head: 1.0e-4}], weight_decay=0.01)`, `CosineAnnealingLR(T_max=updates, eta_min=1e-6)`, `clip_grad_norm_(1.0)` |
| batch | `MICRO_BATCH=8`, `GRAD_ACCUM=4`, `EPOCHS=10` (the module default is 4; the run passed `--epochs 10`) |
| AMP | `amp_dtype` from the checkpoint config (`bf16`): bf16 → autocast bf16 with `GradScaler` **disabled**; fp16 → autocast fp16 with `GradScaler` **enabled**; CPU smoke → float32, no autocast, no scaler |

Items are built from the frozen `B_noprov` card view — the production card with the
`query examples:` block **and** the `evidence:` provenance line removed. The provenance line
names the answer, so training on the stored `questions.tool.criteria` (view B) would leak the
label. The `REJECT` card text is frozen for all rows:

> `### REJECT` / `tool: none` / `does: No listed capability correctly handles the user's
> request. Choose this option only when none of the other options is the right capability.`

### How v2 differs from v1

1. **A fourth, `REJECT` option.** Every row is a flat 4-way classification; the model assigns
   the abstain class probability mass directly instead of deriving it from a threshold.
2. **A per-row deterministic 4-way shuffle** (seed prefix `reject-shuffle-v1`, recorded in
   each row's `shuffle` field), so the model cannot learn a positional prior.
3. **The target is positional.** `gold_index ∈ {0,1,2,3}` marks the target slot and
   `target_kind ∈ {capability, reject}` records the row's kind; the model input is exactly
   `{state, order, options}`, and every other frozen key (`part`, `source_query_id`, `src_sha`,
   `shuffle`, `lang`, `gold_index`, `target_kind`, `gold_capability_id`, `reject_index`) is
   metadata that **never enters the sequence**.
4. **A larger, re-frozen bundle** (Train 2 112 / Val 300 / Test 1 400), including a
   Test→Train adjustment (see *Pipeline*).

Everything else — base, seeds, optimiser, loss, noise, token budget — is the frozen v1
recipe, deliberately unchanged so v2 is comparable to v1 in every respect except the task.

### The input-token budget

Three constants, all in `layachoice_v2_spec.py` (the single place the training, evaluation and
export paths read them from), so the three can never render the input differently:

| | |
|---|---|
| `OPTION_MAX_TOKENS` | **256** — the per-option hard cap (capability cards and `REJECT`) |
| `HEAD_MAX_LEN` | **768** — the budget the question head and **all four options share** |
| `MAX_LEN` | **1024** — the whole-sequence ceiling |

**Why 256 and not the library's 48.** `laya.common.build_sequence` hard-codes
`max_length=48` on each option. The frozen `B_noprov` cards run 136–230 tokens (median 171),
so at 48 tokens *none* of the options keeps a complete `does:` line or any part of the
`negative examples:` block — the discriminative material is always cut. The construction audit
confirmed nothing is truncated at 256, and that the `REJECT` card enters the sequence in full.

**Why `head_max_len` is 768 and not the checkpoint's 256.** Despite the name, `head_max_len`
is not a length — it is a **combined budget**: `opt_budget = head_max_len − Σoptions`, and when
that drops below 16 the even-share fallback shrinks the options *and*
`head_ids[:max(8, opt_budget)]` cuts the instruction head. At the 256 cap the four options plus
four markers peak well above the checkpoint's own 256, so 768 leaves slack for both the options
and the head to survive whole.

**The builder is a verbatim copy, not a fork.** `build_sequence` in `layachoice_v2_render.py`
is the frozen v1 builder byte-for-byte — a copy of `laya.common.build_sequence` with ONE line
changed, `max_length=48` → `OPTION_MAX_TOKENS`. The `opt_budget < 16` even-share fallback, the
`head_ids[: max(8, opt_budget)]` floor and the `return_stats` dict are all stock; keeping the
fallback is what stops a budget too tight for the options from silently cutting the instruction
head to 8 tokens. `install()` points `laya.agent`'s call site at this builder. The baseline, the
fine-tune and the benchmark all run under the same budget, otherwise the comparison would not be
like-for-like.

**Deployment consequence:** a stock `laya` install renders at 48 tokens per option and
`head_max_len=256`. Reproducing v2 requires applying the same override
(`layachoice_v2_render.install()`) and passing the same 256/768/1024 budget before the Agent is
constructed, or expect a materially lower score under stock rendering.

## Pipeline

```
train 2112 ── exclude the frozen 211 calibration rows (seed 20260922) ──► 1901 effective
           ──► world_size alignment (1901 // world_size * world_size) ──► per-rank shards
Validation v3 300 ──► per-epoch top-1 ──► best_epoch = argmax (ties: earlier epoch)
Final Test v3 1400 ──► benchmark of best_epoch only, never used to pick or to gate
```

Exclusion happens on the full 2 112 **before** any world-size truncation, so a multi-rank run
can never pull a calibration row into training. The calibration rule is
`min(CALIB_MAX, n // 10)` (here 211) over the final train order, seed `20260922`; the 211
indices and ids are recorded in `manifest.json` and re-verified on every run. Selection is
fully deterministic: `argmax` on validation top-1, ties to the earlier epoch, train loss
recorded but never used. Top-1 is invariant under the scalar temperature, so selection does not
depend on the calibration step.

**Test→Train adjustment.** A set of 200 whole source queries (400 rows: one `p1` + one `p2` per
source) was moved from Test into Train; the **only** field changed on any row is `split`. The
pre-adjustment 1 800-row test is preserved as backup material
(`frozen_backup/v2_test_1800_pre_adjustment.jsonl`, with its SHA256 record in
`frozen_backup/FROZEN_SHA256.json`). The splits above are the final ones. `manifest.json`
declares each split's provenance, and the loader refuses a row whose source sha the manifest
does not declare for that split.

## Stages

The frozen bundle in `data/` is the input; the loader proves each split still describes the
manifest (row count, split tag, and every row's declared source sha) before a run starts.

Stages 2 and 3 run on the GPU box:

```bash
python layachoice_v2_finetune.py --train [--smoke] [--epochs N] [--out DIR]
python layachoice_v2_eval.py     --baseline-zero-shot --out out/baseline [--smoke]
python layachoice_v2_eval.py     --select --run out --out out/select [--smoke]
python layachoice_v2_finetune.py --fit-temperature --ckpt out/epoch-7
python layachoice_v2_eval.py     --benchmark --ckpt out/epoch-7 --out out/benchmark \
                                 --baseline out/baseline/baseline.report.json
python layachoice_v2_finetune.py --export --ckpt out/epoch-7 --out out/final \
                                 [--export-dtype preserve|fp16|bf16]
```

Every arm writes the **complete per-row result** (`*.rows.jsonl`: id, split, language, part,
the option order, gold, prediction, hit, per-option probabilities, confidence, which slot the
prediction took, and the `gold_is_reject` / `pred_is_reject` flags) alongside its
`*.report.json`; the aggregate is a derived quantity, never the only artifact. Reports carry
top-1 accuracy, the `REJECT` block (recall / precision / false-positive rate, as a normal
fourth class), ECE on `max(p)`, mean confidence and mean gold probability, and breakdowns by
language, target kind, part and gold id, plus the top confusion pairs.

`--select` scores every `epoch-N` checkpoint on the frozen Validation split and writes
`selection.json` with the per-epoch table and the chosen `best_epoch` (deterministic: highest
val top-1, ties to the earlier epoch; train loss is recorded and never used to pick).

`--fit-temperature` fits a single scalar for the `choice` slot on the frozen 211-row
calibration split (LBFGS, `lr=0.1`, `max_iter=100`, fewer than 10 rows → `1.0`, failure →
`1.2`) and writes it through `laya.common.clamp_temperature`, i.e. the `[0.5, 5.0]` range the
Agent itself enforces on load. `temperature.json` records both `raw_fitted_value` and
`final_written_value` (the v2 run fitted `6.389…`, clamped to `5.0`).

`--export` writes an Agent-loadable directory (`model.safetensors`, `encoder/config.json`,
`tokenizer/`, `rl_agent_config.json` with `fine_tuned=true` and `model_name="layachoice-v2"`,
the length-3 `temperature`, and `temperature_by_options` dropped), then reloads it through
`laya.Agent` and verifies the fitted value, the exported config and the Agent's effective value
agree. fp32 ≈ 1.29 GB, fp16/bf16 ≈ 645 MB.

## Results

Best epoch 7 (selected on Validation v3 top-1 only), calibrated temperature `[5.0, 1.0, 1.0]`,
on the frozen Final Test v3 (1 400 rows):

| Arm | Top-1 |
|---|---|
| Validation v3 (best epoch 7) | 0.966667 = 290/300 |
| **Final Test v3, fine-tuned** | **0.877857** |
| EN (692 rows) / ZH (708 rows) | 0.861272 / 0.894068 |
| capability subset (1 286 rows) | 0.891913 |
| `REJECT` subset (114 rows) | 0.719298 |
| zero-shot baseline | compared by `--benchmark --baseline` |

`REJECT` detection on the Final Test (positive class = `REJECT`): recall **0.719298**,
precision **0.557823**, false-positive rate **0.050544** (TP/FP/FN/TN = 82/65/32/1221).

The Final Test never gates the export: `--export` writes the best checkpoint whether or not it
beats the zero-shot baseline, and the benchmark's `baseline_comparison.beats_baseline` records
the verdict.

**Scope.** 0.877857 is **benchmark accuracy on the frozen Final Test v3 split** — a fixed,
curated, 1 400-row artifact over 17 known capabilities with K=3 + `REJECT` and EN/ZH only. It
is **not** production-traffic accuracy and **not** open-world generalization. The `REJECT`
class is the weakest part (65 capability rows wrongly rejected, 32 `REJECT` rows wrongly
routed). The v1 and v2 splits are **not directly comparable**: v1's 93.56 % was a 3-way,
900-row split with no `REJECT`; v2's 87.79 % is a harder 4-way split that adds the abstain
class. The full record — splits, frozen SHA256s, the 10-epoch table, calibration, export
verification and limitations — is in
[`docs/experiments/LayaChoice-v2-Fine-Tuning.md`](../../../docs/experiments/LayaChoice-v2-Fine-Tuning.md).
