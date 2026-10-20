# LayaChoice-v2 — fine-tuning Laya's multilingual decision model

`convaiinnovations/laya` is a non-autoregressive decision engine ("system one"): an encoder
plus a small scoring head that picks one option out of a question's `criteria` dict. This
directory fine-tunes the **multilingual** checkpoint on **LayaChoice-v2**, and measures the
result against the same checkpoint un-finetuned on the same frozen test set.

The task is dynamic capability selection: every question shows the **K real recalled
capability cards (0..3) plus one `REJECT` card**, so a row has **K+1 options (1..4)**. The
model can abstain when the correct capability is not among the candidates. `REJECT` is a
normal option — trained directly, not a threshold and not a post-hoc rule. `OPTION_SLOTS = 4`
remains the frozen flat-v2 constant; the adapter and loader validate the actual K+1 count
(`MIN_OPTION_SLOTS = 1`, `MAX_OPTION_SLOTS = 4`). The base checkpoint, token budget, seeds
and training hyper-parameters are the frozen v1 values; only the dataset/task differs.

> **Production integration.** `cap_router.card_renderer.render_choice_question` renders the
> same instructions, the K capability cards and the frozen `REJECT` card, and
> `deploy/laya/layachoice_sequence` builds the same token sequence as the training path.
> `tests/test_laya_v4_prompt_parity.py` pins that training/serving parity.

```
fine_tuning/laya_finetune/V2/
  README.md                     this file
  run.sh                        single entry point: a bare `bash run.sh` runs the whole
                                pipeline; each phase is also runnable alone
                                (setup / verify / baseline / train / select /
                                temperature / benchmark / export / analysis / all)
  requirements.txt              the pinned fine-tune environment (isolated from the app)
  layachoice_v2_spec.py         the frozen constants: token budget, base checkpoint, seeds, hyper-parameters
  layachoice_v2_render.py       the model-input boundary: the 256-token builder and install()
  layachoice_v2_dataset.py      the dataset contract: adapter (V2Example) + manifest-checked loader
  layachoice_v2_finetune.py     stage 2: --train / --fit-temperature / --export
  layachoice_v2_eval.py         stage 3: --baseline-zero-shot / --select / --benchmark
  layachoice_v2_verify.py       bundle verifier: hashes, schema, labels, splits, calibration,
                                threshold consistency (stdlib only — no GPU, no tokenizer,
                                no network; flat V2 and dynamic V4 layouts; the dynamic
                                layout reads `SHA256SUMS.all_thresholds` or `SHA256SUMS`)
  analysis/                     per-epoch Val curves -> PNGs + val_metrics.csv
                                (plot_val_curves.py reads val_metrics.json only; Test is
                                never read; the frozen val_metrics.json is the historical
                                flat-4-way run record, and `run.sh analysis` writes the
                                current run's metrics under `$OUT/analysis/`)
  data/                         the installed flat bundle (recall_threshold_082); the
                                JSONL splits are distributed via Hugging Face
    train.jsonl / val.jsonl / test.jsonl
                                the three splits (the only data training reads); the
                                four-key training schema (`message_id`, `language`,
                                `input`, `label`) matches the adapter contract and
                                the Hugging Face release — see ../data_versions/README.md
    manifest.json               row counts, calibration indices, K+1 option bounds,
                                base revision, the two fields training never reads
    SHA256SUMS                  flat-bundle hash index (run.sh verify)
  ../data_versions/             documentation and construction archive (not read by run.sh)
    README.md                   version roles, scale, schema, HF download and checksums
    recall_threshold_*/         LOCAL CONSTRUCTION ARCHIVE ONLY: v2_*.jsonl bundles and
                                audits, git-ignored and not the HF release
    message_id_map.jsonl        LOCAL ONLY: new `message_id` -> construction row id, for
                                audit joins; never uploaded and never read by training
    shared/                     construction evidence; only capability_ids.json is tracked
    dataset_construction/       the V4 build archive (BUILD_RECORD.md, scripts, audits);
                                inputs/*.csv is git-ignored
    build/, smoke/, audit/      historical construction, smoke and prompt-audit evidence
    SHA256SUMS.all_thresholds   checksums for the local construction copies above
  out/                          run artifacts (not committed)
```

## Data format

Every training row has exactly four fields:

```json
{"message_id": "0f4c0f9e-6a2b-4f8e-9f5a-7c1d2e3b4a5c",
 "language": "en",
 "input": {"query": "read this document and answer from it",
           "options": ["cap-read-document: ### cap-read-document\ntool: ...",
                       "cap-summary: ### cap-summary\ntool: ...",
                       "REJECT: ### REJECT\ntool: none\ndoes: ..."]},
 "label": 0}
```

* **`input.options`** is the exact text the model sees, one string per option, in
  option order: K recalled capability cards plus exactly one `REJECT` (K+1 options, 1..4).
  Each string is the production rendering `"<capability_id>: <card>"`; no padding.
* **`label`** is the 0-based index of the correct option in `input.options`. It may point
  at the `REJECT` card: `REJECT` is a normal class, not a threshold.
* **`message_id`** is a canonical uuid4, fresh in every threshold bundle. It identifies a
  sample for bookkeeping only: it never enters the model input, the label, calibration
  (which is positional) or cross-version alignment.
* **`language`** is reporting metadata. It is not used by training.
* **`split` is not a field.** The file name (`train.jsonl` / `val.jsonl` / `test.jsonl`,
  or the archived `v2_{split}.jsonl`) is the authority; `load_split` injects it.

The exact sequence the model receives is built from `input.query` and `input.options`
(the query is appended **after** the options):

```
[CLS] choice question: Which capability should handle the user's request?
[SEP] [MASK] <option 0> [MASK] <option 1> ... [MASK] <option K>
[SEP] <input.query> [SEP]
```

with the one-hot training target at `label`. Instructions and the token budget come from
`manifest.json` / `layachoice_v2_spec.py`, not from the row.

## Environment

| | |
|---|---|
| Python | **3.11** (the verified run used 3.11) |
| OS / shell | Linux or macOS. **Windows:** run the `*.py` stages directly (Git-Bash or WSL can run `run.sh`; PowerShell cannot). |
| GPU | **1× NVIDIA L4** for the verified run; any CUDA GPU with ≥40 GB free disk works (bf16 on Ampere+, fp16 + `GradScaler` on T4). A CPU-only box can run `--smoke` and `verify`. |
| CUDA / torch | verified on `torch 2.14.1+cu130`; the `--smoke` and `verify` paths run on `torch 2.14.0` CPU |
| transformers | `5.17.0` (exact-pinned) |
| laya | `0.3.21` (exact-pinned, asserted at every stage start) |
| network | the base checkpoint is fetched once from the pinned revision and then cached; nothing else is downloaded |

The environment is **isolated from the app**: install [`requirements.txt`](requirements.txt)
into a dedicated venv, not the Delveta app environment (which uses the repo-root
`requirements.txt`).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # laya==0.3.21, transformers==5.17.0, torch, ...
```

`laya` is pinned to `0.3.21`. The upstream repository went private after this recipe was
written, so `run.sh` and the modules fail loudly on any other version rather than silently
training on a different API.

Disk: each epoch checkpoint is ~3.9 GB (1.3 GB fp32 weights + 2.6 GB AdamW state), so budget
~40 GB for ten epochs.

## Deploy

> **Historical note.** The 2026-10-02 GPU training ran **without** a `run.sh` — there was
> none on the host; every stage was invoked directly (see `RUN_RECORD.md` §9–10). The
> `run.sh` below is a reproducible entry point assembled from the recovered and verified CLI
> workflow; it is **not** a transcript of the original run.

`run.sh` is the single entry point. **A bare `bash run.sh` runs the whole pipeline end to
end** — the fresh-clone "one command" path. It installs the pinned environment on first use
(creating `.venv` if the chosen interpreter lacks `laya`), then runs
`verify → baseline → train → select → temperature → benchmark → export → analysis`, stopping
on the first failing phase.

```bash
bash run.sh                                 # the whole pipeline, defaults (10 epochs)
bash run.sh --epochs 10                     # same, explicit epoch count
bash run.sh --python /path/to/python        # use a pre-built interpreter (skips setup)
```

Every phase is also runnable **alone**, so a run is resumable from any step:

```bash
bash run.sh setup                           # create .venv + pip install -r requirements.txt
bash run.sh verify                          # hashes + schema + labels + splits (no GPU)
bash run.sh baseline                        # zero-shot baseline on the frozen Test
bash run.sh train --epochs 10               # fine-tune (GPU)
bash run.sh select                          # per-epoch validation -> best_epoch
bash run.sh temperature                     # fit the temperature on the calibration split
bash run.sh benchmark                       # best checkpoint on the frozen Test
bash run.sh export                          # write the Agent-loadable artifact
bash run.sh analysis                        # per-epoch Val curves for this run
```

Add `--smoke` to any stage for a CPU run (8 rows per split, 1 training step), and
`--python <exe>` / `--data DIR` / `--out DIR` to point at a specific interpreter or
directories. `verify` auto-detects either the installed flat `data/` bundle or a
multi-threshold dynamic layout, whose checksum file is `SHA256SUMS.all_thresholds`
or `SHA256SUMS` at the layout root (`../data_versions/SHA256SUMS.all_thresholds`
covers the three threshold directories and `shared/`). A single threshold directory
can also be verified alone: `bash run.sh verify --data
../data_versions/recall_threshold_085` reads the parent checksum file and filters it
to that directory. Repository control files such as `.gitattributes` are not bundle
files and are ignored by the checksum-coverage check. Every stage defaults to the installed `data/` bundle
(recall_threshold_082). The other thresholds are archived, e.g.
`bash run.sh train --data ../data_versions/recall_threshold_085`. `run.sh` never deletes an output directory, never downloads anything unpinned and
never uploads; it stops on the first failing stage and returns non-zero. Each run appends a
record (git sha, config, benchmark and row counts) to `$OUT/run_ledger.jsonl`; the
`--data` path and threshold are not recorded, so keep them in the experiment note.

`all` is a complete experiment: it refuses to start if `$OUT` already holds a run
(`baseline/`, `select/`, `epoch-1/`, ...), because those stages write their results
unconditionally. Pass a fresh `--out DIR` for every experiment; individual phases stay
resumable for an existing run directory.

The GPU box needs no registry, no database and no repo checkout beyond this directory.

## Recipe

Fine-tuning uses `transformers` + `torch` directly, not PEFT or the Trainer, because the
`laya` wheel ships inference only — `laya.common.build_model` / `proper_reward` /
`collate_items` are reusable, but there is no trainer. The recipe is the frozen v1 recipe
(item construction, then the official notebook's `train_ddp.py` loop), reproduced here:

| | |
|---|---|
| checkpoint | `convaiinnovations/laya`, subfolder `multilingual`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (asserted on load) |
| encoder | `jhu-clsp/mmBERT-base`, 322 M params, ctx 1024 |
| items | `build_sequence` → `{ids, markers, qtype=QTYPES["choice"], target, label}`; a one-hot target at the row's `label`, over its K+1 option slots (K capabilities + `REJECT`) |
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

1. **An explicit `REJECT` option.** The model assigns the abstain class probability mass
   directly instead of deriving it from a threshold.
2. **A per-row deterministic K+1 shuffle at construction time** (seed prefix
   `reject-shuffle-v1`, keyed by the construction row id). Only the final option order is
   stored, in `input.options`, so the model cannot learn a positional prior.
3. **The target is positional.** `label` marks the correct slot within the row's actual
   option count; the model input is exactly `input.query` + `input.options`. Derived
   views (`order`, `target_kind`, `gold_capability_id`, `reject_index`) are computed by
   the adapter and are never stored or rendered. `message_id` and `language` are metadata
   that **never enter the sequence**.
4. **A dynamic bundle.** V4 carries 48,333 rows per threshold (train 36,197 / val 2,407 /
   test 9,729) with K ∈ 0..3, including an in-place K=0→K=1 repair; see
   `../data_versions/dataset_construction/BUILD_RECORD.md`.

Everything else — base, seeds, optimiser, loss, noise, token budget — is the frozen v1
recipe, deliberately unchanged so v2 is comparable to v1 in every respect except the task.

### The input-token budget

Three constants, all in `layachoice_v2_spec.py` (the single place the training, evaluation and
export paths read them from), so the three can never render the input differently:

| | |
|---|---|
| `OPTION_MAX_TOKENS` | **256** — the per-option hard cap (capability cards and `REJECT`) |
| `HEAD_MAX_LEN` | **768** — the budget the question head and **all options share** |
| `MAX_LEN` | **1024** — the whole-sequence ceiling |

The per-option cap was raised from the library default of 48 tokens because the frozen
capability cards require more space. These limits apply at different levels and should not
be treated interchangeably.

**Card-length audit:** all 18 capability cards fit within the 256-token per-option cap.
`cap-vision` and `cap-read-document` are the longest at 224 tokens.

See [LayaChoice V2 Fine-Tuning — Token Budget](../../../docs/experiments/LayaChoice-v2-Fine-Tuning.md#capability-card-token-audit-and-compression)
for the detailed rationale, card statistics and full card prompts.

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

**Token Budget Audit.** The current per-option budget is **256 tokens**. All 18 capability
cards fit within it; `cap-vision` and `cap-read-document` are the longest at 224 tokens. The
complete 18-card table, full card prompts and measured token counts are in
[LayaChoice-v2 Fine-Tuning — Token Budget](../../../docs/experiments/LayaChoice-v2-Fine-Tuning.md#capability-card-token-audit-and-compression).

## Pipeline

```
train 36,197 ── exclude the frozen 400 calibration rows (seed 20260922) ──► 35,797 effective
             ──► world_size alignment (n // world_size * world_size) ──► per-rank shards
Validation 2,407 ──► per-epoch top-1 ──► best_epoch = argmax (ties: earlier epoch)
Final Test 9,729 ──► benchmark of best_epoch only, never used to pick or to gate
```

Exclusion happens on the full train split **before** any world-size truncation, so a multi-rank
run can never pull a calibration row into training. The calibration rule is
`min(CALIB_MAX, n // 10)` (here 400) over the final train order, seed `20260922`; the indices
and ids are recorded in `manifest.json` and re-verified on every run. Selection is fully
deterministic: `argmax` on validation top-1, ties to the earlier epoch, train loss recorded but
never used. Top-1 is invariant under the scalar temperature, so selection does not depend on the
calibration step.

**Historical flat-bundle adjustment.** In the earlier fixed-4-option V2 bundle, 200 whole
source queries (400 rows) were moved from Test into Train, changing only the `split` field.
That adjustment belongs to the historical run recorded under Results; the installed V4 bundle
(36,197 / 2,407 / 9,729) has its own split assignment, and `manifest.json` records the source
files each split was built from.

## Stages

The frozen bundle in `data/` is the input; the loader proves each split still describes the
manifest (row count, four-field schema, K+1 option bounds, unique `message_id`s) before a
run starts.

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

Every arm writes the **complete per-row result** (`*.rows.jsonl`: `message_id`, split,
language, the option order, gold, prediction, hit, per-option probabilities, confidence,
which slot the prediction took, and the `gold_is_reject` / `pred_is_reject` flags) alongside
its `*.report.json`; the aggregate is a derived quantity, never the only artifact. Reports
carry top-1 accuracy, the `REJECT` block (recall / precision / false-positive rate, as a
normal class), ECE on `max(p)`, mean confidence and mean gold probability, and breakdowns by
language, target kind and gold id, plus the top confusion pairs.

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

`layachoice_v2_082_r1`, V4 bundle (`recall_threshold_082`, four-field input, K+1); best
epoch 9, selected on Validation only.

| Arm | Top-1 |
|---|---|
| Validation (2 407 rows) | 0.996676 |
| **Final Test (9 729 rows)** | **0.998355** (9 713/9 729) |
| en / zh / mixed | 0.998197 / 0.998306 / 0.999137 |
| capability / `REJECT` | 0.998537 / 0.996441 |
| zero-shot baseline | 0.416795 |
| delta vs zero-shot | **+0.581560** |

`REJECT` recall **0.996441**, precision **0.995261**, false-positive rate **0.000450**
(TP/FP/FN/TN = 840/4/3/8882). ECE **0.002576**. Temperature `[3.070228, 1.0, 1.0]`.

The Final Test never gates the export: `--export` writes the best checkpoint whether or not it
beats the zero-shot baseline, and the benchmark's `baseline_comparison.beats_baseline` records
the verdict.

## Artifacts

| Artifact | Repository | Commit |
|---|---|---|
| Model | [eric-ml-nlp/Delveta-LayaChoice-v2](https://huggingface.co/eric-ml-nlp/Delveta-LayaChoice-v2) | `2574addf` |
| Data | [eric-ml-nlp/Delveta-LayaChoice-v2-Data](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data) | `e913457b` |
