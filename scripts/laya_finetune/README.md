# LayaChoice-v1 — fine-tuning Laya's multilingual decision model

`convaiinnovations/laya` is a non-autoregressive decision engine ("system one"): an encoder
plus a small scoring head that picks one option out of a question's `criteria` dict. This
directory fine-tunes the **multilingual** checkpoint on LayaChoice-v1 so that the Delveta
chat Intent Funnel can select a capability with it, and measures the result against the same
checkpoint un-finetuned on the same frozen test set.

```
scripts/laya_finetune/
  README.md                this file
  run.sh                   one-click: env check -> baseline -> train -> select -> temperature
                           -> benchmark -> export [-> push-to-hub]
  layachoice_prepare.py    stage 1, run on the repo host: freeze the bundle (needs the DB)
  layachoice_finetune.py   stage 2: --train / --fit-temperature / --export
  layachoice_eval.py       stage 3: --baseline-zero-shot / --select / --benchmark
  data/                    the frozen bundle, self-contained and also the GitHub backup
    layachoice_v1_{train,val,test}_b_noprov.jsonl.gz
    raw/LayaChoice_v1_{validation,final_test}_raw_v3.jsonl
    manifest.json          shas, row counts, calibration ids, base checkpoint revision
    SHA256SUMS
  out/                     training artifacts (not committed)
```

## Deploy

The GPU box needs no registry, no database and no repo checkout beyond this directory.

```bash
# 1. env
python3 -m venv .venv && . .venv/bin/activate
pip install "laya==0.3.21" torch transformers safetensors huggingface_hub numpy

# 2. smoke first: CPU, float32, no GradScaler, no DDP, 8 rows per split, 1 training step
bash run.sh --smoke

# 3. the real run (4 epochs, bf16 on Ampere+, fp16 + GradScaler on T4)
nohup bash run.sh > run.log 2>&1 &

# 4. optional upload of the export (private by default)
bash run.sh --push-to-hub <repo_id>
```

`laya` is pinned to `0.3.21`. The upstream repository went private after this recipe was
written, so `run.sh` fails loudly on any other version rather than silently training on a
different API.

`run.sh` re-runs stages independently — `--skip-baseline` to keep an existing baseline, or
invoke the stages directly (see *Stages* below). Disk: each epoch checkpoint is ~3.9 GB
(1.3 GB fp32 weights + 2.6 GB AdamW state), so budget ~20 GB for four epochs.

## Recipe

Fine-tuning uses `transformers` + `torch` directly, not PEFT or the Trainer, because the
`laya` wheel ships inference only — `laya.common.build_model` / `proper_reward` /
`collate_items` are reusable, but there is no trainer. The recipe is the official Laya
fine-tuning notebook (item construction, then its `train_ddp.py` loop), reproduced here:

| | |
|---|---|
| checkpoint | `convaiinnovations/laya`, subfolder `multilingual`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (asserted on load) |
| encoder | `jhu-clsp/mmBERT-base`, 322 M params, ctx 1024 |
| items | `build_sequence` → `{ids, markers, qtype=QTYPES["choice"], target, label}`; a one-hot target at the gold slot, `label = argmax(target)`; rows whose option count collapses are rejected |
| loss | `loss_rl + 1.0*loss_ce`, where `loss_rl` is a GRPO-style group-advantage term over `proper_reward` (log score + 0.75·spherical) at `GROUP_SIZE=4`, and `loss_ce` is a masked cross-entropy on the gold slot |
| noise | `eps ~ N(0, sigma²)` zero-mean-projected onto the simplex-perpendicular and masked; `sigma` 0.4 → 0.1 across epochs; the advantage is standardized within the group |
| optim | `AdamW([{encoder: 2.5e-5}, {head: 1.0e-4}], weight_decay=0.01)`, `CosineAnnealingLR(T_max=updates, eta_min=1e-6)`, `clip_grad_norm_(1.0)` |
| batch | `MICRO_BATCH=8`, `GRAD_ACCUM=4`, `EPOCHS=4` |
| AMP | `amp_dtype` from the checkpoint config (`bf16`): bf16 → autocast bf16 with `GradScaler` **disabled**; fp16 → autocast fp16 with `GradScaler` **enabled**; CPU smoke → float32, no autocast, no scaler |

Items are built from `views.B_noprov.tool` — the production card with the `query examples:`
block **and** the `evidence:` provenance line removed. The provenance line names the answer in
856/856 training rows, so training on the stored `questions.tool.criteria` (view B) would
leak the label.

### Deviations from the official notebook

All six are deliberate and were frozen before implementation:

1. **Per-epoch checkpoints.** One full-state checkpoint per epoch (`epoch-1 … epoch-N`,
   never overwritten) instead of a single rolling one, so the selection is auditable.
2. **Recorded calibration ids.** The notebook fixes the calibration holdout by seed only;
   the 85 chosen row indices are recorded in `manifest.json` and re-verified on every run.
3. **`--export-dtype preserve` by default.** The notebook unconditionally calls `.half()`.
   `preserve` keeps the trained fp32 weights (`fp16`/`bf16` are available explicitly).
4. **Autocast dtype read from `amp_dtype`.** The notebook hardcodes fp16.
5. **A CPU `--smoke` path.** The notebook has none.
6. **A zero-shot baseline arm.** The notebook has none.

### The option-token cap

`laya.common.build_sequence` caps each option at **48 tokens**. Every frozen option is
136–230 tokens (median 171) under view B_noprov, and at 48 tokens *none* of the 5718 options
keeps a complete `does:` line or any part of the `negative examples:` block — i.e. the
discriminative material is always cut. The dataset was audited under a **256-token** cap
(`logs/_laya_ds/coverage_audit.json` → `tokens`: 0/2568 B_noprov options exceed 256; head
max 610, never reaching `max_len`), so `layachoice_finetune.py` carries its own
`build_sequence` at 256 tokens with no even-share trim, and `install()` points
`laya.agent`'s call site at it. The baseline, the fine-tune and the benchmark all run under
this same cap, otherwise the comparison would not be like-for-like.

**Deployment consequence:** a stock `laya` install truncates to 48 tokens. Either apply the
same override (`layachoice_finetune.install()`) before constructing the Agent, or expect a
lower score under stock rendering.

## Pipeline

```
train 856 ── exclude the frozen 85 calibration rows (seed 20260922) ──► 771 effective
         ──► world_size alignment (771 // world_size * world_size) ──► per-rank shards
Validation v3 150 ──► per-epoch top-1 ──► best_epoch = argmax (ties: earlier epoch)
Final Test v3 900 ──► benchmark of best_epoch only, never used to pick or to gate
```

Exclusion happens on the full 856 **before** any world-size truncation, so a multi-rank run
can never pull a calibration row into training. Selection is fully deterministic: `argmax`
on validation top-1, ties to the earlier epoch, train loss recorded but never used. Top-1 is
invariant under the scalar temperature (`softmax(logits/T)` is monotone in `logits/T`), so
selection does not depend on the calibration step.

## Stages

Stage 1 runs on the repo host, once — it needs the live capability registry:

```bash
.venv/Scripts/python.exe scripts/laya_finetune/layachoice_prepare.py          # full
.venv/Scripts/python.exe scripts/laya_finetune/layachoice_prepare.py --smoke  # 8 rows/split
```

It refuses to build from inputs that are not the frozen revision (four pinned SHA-256s),
re-renders the validation split in memory through the same shared row contract the other
splits went through, and asserts per row: three option slots, a gold slot among them,
distinct gold/d1/d2 slots, non-empty cards, and no query text reaching a card through the
`example`/`substring` channels. `data/raw/` keeps the two raw files that cannot be
regenerated.

Stages 2 and 3 run on the GPU box:

```bash
python layachoice_finetune.py --train [--smoke] [--epochs N] [--out DIR]
python layachoice_eval.py     --baseline-zero-shot --out out/baseline [--smoke]
python layachoice_eval.py     --select --run out --out out/select [--smoke]
python layachoice_finetune.py --fit-temperature --ckpt out/epoch-3
python layachoice_eval.py     --benchmark --ckpt out/epoch-3 --out out/benchmark \
                              --baseline out/baseline/baseline.report.json
python layachoice_finetune.py --export --ckpt out/epoch-3 --out out/final \
                              [--export-dtype preserve|fp16|bf16]
```

Every arm writes the **complete per-row result** (`*.rows.jsonl`: id, language, the option
order, gold, prediction, hit, per-option probabilities, confidence, which slot the
prediction took) alongside its `*.report.json`; the aggregate is a derived quantity, never
the only artifact. Reports carry top-1 accuracy, the hard-negative rate (`1 - top-1`), the
`gold → d1` and `gold → d2` rates, zh/en and per-capability breakdowns, the top confusion
pairs, and ECE on `max(p)`.

`--export` writes an Agent-loadable directory (`model.safetensors`, `encoder/config.json`,
`tokenizer/`, `rl_agent_config.json` with `fine_tuned=true`, the fitted 3-element
`temperature` and `temperature_by_options` dropped), then reloads it through
`laya.Agent` and verifies that the fitted value, the exported config and the Agent's
effective value agree. fp32 ≈ 1.29 GB, fp16/bf16 ≈ 645 MB.

`--fit-temperature` fits a single scalar for the `choice` slot on the frozen 85-row
calibration split (LBFGS, `lr=0.1`, `max_iter=100`, fewer than 10 rows → `1.0`, failure →
`1.2`) and writes it through `laya.common.clamp_temperature`, i.e. the `[0.5, 5.0]` range
the Agent itself enforces on load — the notebook's wider `[0.1, 10.0]` is not used, or the
Agent would silently rewrite the fitted value and the three-way agreement would break.
`temperature.json` records both `raw_fitted_value` and `final_written_value`.

## Results

| arm | checkpoint | temperature | rows |
|---|---|---|---|
| zero-shot baseline | base `multilingual` | checkpoint's own `[1,1,1]` | 900 |
| fine-tuned | `best_epoch` by validation top-1 | fitted, then clamped | 900 |

The Final Test never gates the export: `run.sh` exports the best checkpoint whether or not
it beats the zero-shot baseline, and the benchmark's `baseline_comparison.beats_baseline`
records the verdict.
