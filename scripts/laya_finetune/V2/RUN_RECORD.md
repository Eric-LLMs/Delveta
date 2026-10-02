# LayaChoice-v2 — Run Record

A traceable record of the **formal LayaChoice-v2 run** that produced
`eric-ml-nlp/Delveta-LayaChoice-v2` (published 2026-10-02, artifact commit
`bca44475f41f86dc68f4cf8319342f2791f217b4`).

> **Provenance of this document.** The run took place in a **working tree on a
> rented GPU host** (`/root/delveta/scripts/laya_finetune/V2`) that no longer
> exists and is not reachable. There was **no `run.sh`** on that host and **no
> shell/console log of the run was kept**. Every command shown below is
> **reconstructed** from the stage modules' CLI and from the run's own
> `release_manifest.json`; it is **not** a transcript of the original shell.
> Nothing here should be quoted as a verbatim command log. Where a fact comes
> from a recorded artifact it is cited; where it is inferred it is marked
> *reconstructed*.

The authoritative machine-readable record is the release manifest produced by the
run (kept out of this repo, archived in `.tmp/v2_release/release_manifest.json`):
`kind = formal-release`, `generated_utc = 2026-10-02T18:07:37Z`.

---

## 1. Environment (as recorded by the run)

| | |
|---|---|
| platform | `Linux-6.8.0-100-generic-x86_64-with-glibc2.35`, x86_64 |
| GPU | 1× **NVIDIA A30** |
| CUDA | 13.0, `cuda_available = true`, `bf16_supported = true` (AMP dtype bf16) |
| python | **3.11** |
| torch | **2.14.1+cu130** |
| transformers | **5.17.0** |
| laya | **0.3.21** |
| base checkpoint | `convaiinnovations/laya`, subfolder `multilingual`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (asserted on load) |
| encoder | `jhu-clsp/mmBERT-base` (~322 M params) |

Token budget (frozen constants, `layachoice_v2_spec.py`): `OPTION_MAX_TOKENS=256`,
`HEAD_MAX_LEN=768`, `MAX_LEN=1024`. The builder is a verbatim copy of
`laya.common.build_sequence` with `max_length` 48→256.

## 2. Data version (frozen; verified by SHA256)

Card view `B_noprov`; 3 capability cards + 1 `REJECT` card (`OPTION_SLOTS=4`);
17 capabilities + `REJECT`; languages en/zh.

| split | rows | sha256 |
|---|---|---|
| train | 2112 (calibration 211 → effective 1901) | `c92c0b5f6b5fbb019f6cf1e89a107ad4d1c4fee952f4d1a024c6cb319873b720` |
| val | 300 | `f7d97670c96c13330f07dad2a3aecabbae42b3194e54c2cf0dafddc8af51de7b` |
| test | 1400 | `eb14e0455156d6dd3f2b0ddf2fa82feaf5248b10f993a4ded57cea45261fb9ea` |
| manifest | — | `224acf2f3bdec01ab6284942c1f8b87acaf9daa7c924e89a5c8e97b485dbf57e` |
| build report | — | `e2a3dae1d7228ea31b7d318473edd6a795a4561b62b7ebc18f4aad8dd6fc0703` |
| construction audit | — | `2464e187a667d515b97f0082e18d37b66b2b3c377b6f54f681a265c7e4b6091a` |

**Build chain** (frozen `manifest.json.source_files` maps the final data to the
V1 bundles plus `v2_final/v2_{train2,val2,test2}.jsonl`):

```
V1 B_noprov bundles (856/150/900)
  → v2_capture.py            no-threshold SQL recall (index corpus1-d030fea9e32a)
                             → v2_raw_recall.jsonl (1906 rows)
  → v2_generate_final.py     drop self-hit → cap dedup (max) → sort → Top-3 → 4-way rows
                             → v2_final/v2_{train2,val2,test2}.jsonl
  → v2_adjust_test_to_train.py  move 400 Test rows (200 whole source queries) into Train
                             → v2_{train,val,test}.jsonl + frozen_backup of the 1800-row Test
```
The Test→Train adjustment (2026-10-02) changed **only the `split` field**; the
pre-adjustment 1800-row Test is preserved under `frozen_backup/`.

> `build/` scripts are restored **verbatim** from the run. `v2_capture.py`
> depends on the **production recall index** (`SessionLocal` + the V1 bundles),
> so it re-runs only in an environment with that database/DB access — it is not
> offline-reproducible. It also references the pre-archive V1 path
> (`scripts/laya_finetune/data/…`); V1 now lives under `scripts/laya_finetune/V1/`.

## 3. Training parameters (10 epochs)

| | |
|---|---|
| epochs | **10** (module default is 4; the run passed `--epochs 10`) |
| micro_batch | 8 |
| grad_accum | 4 |
| group_size | 4 (GRPO group-advantage) |
| lr (encoder / head) | 2.5e-5 / 1.0e-4 |
| weight_decay | 0.01 |
| loss | `loss_rl + 1.0 * loss_ce`; w_sph 0.75, w_rps 1.0 |
| noise σ | 0.4 → 0.1 across epochs |
| grad clip | 1.0 |
| seed (base / calibration) | 42 / 20260922 |
| total optimizer updates | 600 |
| AMP | bf16 autocast, `GradScaler` disabled (bf16) |

## 4. Checkpoint selection

Rule: `best_epoch = argmax(validation top-1)` on the frozen Val (300 rows),
ties to the earlier epoch; **train loss recorded but never used**; the Test split
is never read for selection.

| Epoch | Train Loss | Val Top-1 | Val ECE | REJECT Recall | REJECT FPR |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.641706 | 85.67% | 0.0803 | 81.25% | 8.45% |
| 2 | 0.384151 | 90.00% | 0.0364 | 93.75% | 8.10% |
| 3 | 0.256366 | 94.00% | 0.0563 | 93.75% | 2.46% |
| 4 | 0.132143 | 89.67% | 0.1018 | 100.00% | 8.10% |
| 5 | 0.079365 | 94.67% | 0.0544 | 100.00% | 2.82% |
| 6 | 0.039875 | 94.33% | 0.0567 | 100.00% | 2.82% |
| **7** | **0.013119** | **96.67%** | **0.0303** | **100.00%** | **1.76%** |
| 8 | 0.000358 | 96.33% | 0.0353 | 100.00% | 2.11% |
| 9 | 0.000000 | 96.33% | 0.0353 | 100.00% | 2.11% |
| 10 | 0.000000 | 96.33% | 0.0353 | 100.00% | 2.11% |

**best epoch = 7** (val top-1 0.966667 = 290/300; val by-language en 1.0 / zh 0.935065).

## 5. Calibration (temperature)

Fit on the frozen 211-row calibration split (seed 20260922), `choice` slot, LBFGS
(`lr=0.1`, `max_iter=100`), clamped through `laya.common.clamp_temperature` to
`[0.5, 5.0]`.

| | |
|---|---|
| checkpoint | `out/epoch-7` |
| raw fitted value | 6.389379501342773 |
| final written value | **5.0** (clamped) |
| base temperature | [1.0, 1.0, 1.0] |
| effective temperature | **[5.0, 1.0, 1.0]** |

## 6. Final Test (frozen Test v3, 1400 rows) — never used to select or gate

| slice | n | top-1 |
|---|---|---|
| **overall** | 1400 | **0.877857** |
| en / zh | 692 / 708 | 0.861272 / 0.894068 |
| capability / reject | 1286 / 114 | 0.891913 / 0.719298 |
| part p1 / p2 | 700 / 700 | 0.910000 / 0.845714 |

`REJECT` (positive class = REJECT): recall 0.719298, precision 0.557823, FPR
0.050544 (TP/FP/FN/TN = 82/65/32/1221; gold REJECT = 114, predicted REJECT = 147).
ECE 0.097189, mean confidence 0.975046, mean gold probability 0.871354.

Zero-shot baseline on the same frozen Test (checkpoint temperature, no
calibration): **0.321429** → **delta +0.556428**.

Scope: this is benchmark accuracy on a fixed 1400-row curated split (17 caps +
REJECT, en/zh only); it is **not** production-traffic accuracy.

## 7. Output layout (training host)

```
out/epoch-1..10/                       per-epoch full-state checkpoints
out/train_manifest.json                training record (not published)
out/baseline/baseline.report.json      zero-shot baseline (+ .rows.jsonl)
out/select/selection.json              chosen best epoch
out/select/epoch-N.val.report.json     per-epoch Val report (+ .rows.jsonl)
out/calibration/temperature.json       fitted temperature
out/benchmark/final_test.report.json   best ckpt on frozen Test (+ .rows.jsonl)
out/analysis/val_metrics.json|csv      analysis input/raw values
out/analysis/*_vs_epoch.png, val_curves.png
out/release_v2/                        Agent-loadable artifact (export)
  model.safetensors, encoder/config.json, tokenizer/,
  rl_agent_config.json, reload_smoke.json
```
The on-host layout is corroborated by the assembly helper
`_v2_assemble_release.py` (archived in `.tmp`), which reads exactly these paths.

## 8. Known SHA256

**Published artifact** (HF `eric-ml-nlp/Delveta-LayaChoice-v2`, commit `bca44475`):

| file | sha256 |
|---|---|
| `model.safetensors` (export = best checkpoint, FP32, 1 287 653 720 B) | `c6331bd5b6c044a8e4841bee8e9e69962cc2deaf1fd39bffb178703c07168b6f` |
| `rl_agent_config.json` | `97f1883b0a705b4151247cf3c5a039a3ab249ba4995ed5c439c16cedb9921f43` |
| `encoder/config.json` | `83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4` |
| `tokenizer/tokenizer.json` | `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f` |
| `tokenizer/tokenizer_config.json` | `6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1` |
| `code/layachoice_v2_spec.py` | `138a3eb231519528f43907f67240f9a207b21f7d2dac8ee60e1de0d28a7e791e` |
| `code/layachoice_v2_dataset.py` | `35c639870fdc893699d1a55b643862ce77c61f15cf79e346ec4b2bb099863e10` |
| `code/layachoice_v2_render.py` | `ee3ac78d55159bf69c3f7e0a6cc6bc3de0b2e68f576e6013e3b9c124a0f74a47` |

**Source files as restored in this repo** (recorded so future drift is detectable):

| file (in `V2/`) | sha256 | matches the run? |
|---|---|---|
| `layachoice_v2_spec.py` | `138a3eb231519528f43907f67240f9a207b21f7d2dac8ee60e1de0d28a7e791e` | **proven** (= published `code/`) |
| `layachoice_v2_dataset.py` | `35c639870fdc893699d1a55b643862ce77c61f15cf79e346ec4b2bb099863e10` | **proven** |
| `layachoice_v2_render.py` | `ee3ac78d55159bf69c3f7e0a6cc6bc3de0b2e68f576e6013e3b9c124a0f74a47` | **proven** |
| `layachoice_v2_finetune.py` | `35fb087bd69df3ee8ca105bfb5c519482ee11965a0967751c3437a70b1133995` | **unverifiable** (never published; see below) |
| `layachoice_v2_eval.py` | `f86c366a887d60284db541f03370b674d89c10cb345aa0326951e61469ae2f18` | **unverifiable** |
| `analysis/plot_val_curves.py` | `1b8682e7e82813daa5c8e3d3313072a908eaf73a455d6a14cbd88214b5f7e951` | **unverifiable** |
| `analysis/val_metrics.json` | `97129e7c99d8f14161b9aee7b57c85b1644a599528487bf4f9c02e6878c859b3` | unverifiable |
| `build/v2_capture.py` | `cf1861d0e1db6276cb6bc246536b979f7af8c18f0ce2d1afb164d0c7a6bd475b` | unverifiable |
| `build/v2_generate_final.py` | `f74367a6968517515d0b347f8f7ebd7fbbaf2d1cbb245895b8def0fa76fcad62` | unverifiable |
| `build/v2_adjust_test_to_train.py` | `9701267056727bff5d40822d2fde2abdab1093c1032b89e0a9db85bbb26c6c56` | unverifiable |
| `smoke/build_smoke_fixture.py` | `1ba41b7ed68bf22f3398e196e1427faedf0803f5755e23541418a154bf6cd76e` | unverifiable |
| `smoke/_step7_driver.py` | `2036afa0b51053c161379906f614079fd2c04878ddb212dd9425c9cc77d62713` | unverifiable |

**Why some rows are "unverifiable".** Only `spec`/`dataset`/`render` were shipped
inside the published model repo (`code/`), so only those three have a golden hash
that proves the restored file equals the file that ran. The fine-tune/eval
modules, the analysis script and the build/smoke scripts were **excluded from the
publish** (release manifest `huggingface.model_repo.excluded`); the host is gone,
so there is no reference hash. They are restored **verbatim from the surviving
copies** in the run workspace; their identity with the executed originals is
strongly indicated (same run tree, internally consistent outputs) but
**not cryptographically proven**.

## 9. Commands: recorded vs reconstructed

**There is no recorded command log.** No `run.sh` existed on the host; no shell
transcript was kept; the smokes' logs (`smoke/train_smoke.log`,
`smoke/eval_benchmark.log`, `_capture.log`) capture module *output* but not the
invocation lines. The following invocations are **reconstructed** from each
module's argparse and from the run's output layout (§7). Treat them as a faithful
*reconstruction*, not the original shell.

```bash
# ── data construction (evidence: _capture.log; outputs = frozen bundle) ──
python v2_capture.py                 # → v2_raw_recall.jsonl (1906 rows)
python v2_generate_final.py          # → v2_final/v2_{train2,val2,test2}.jsonl
python v2_adjust_test_to_train.py    # → v2_{train,val,test}.jsonl + 1800-row backup

# ── training / eval / analysis (host: /root/delveta/scripts/laya_finetune/V2) ──
python layachoice_v2_finetune.py --train --epochs 10 --data data --out out
python layachoice_v2_eval.py --baseline-zero-shot --data data --out out/baseline
python layachoice_v2_eval.py --select  --data data --run out --out out/select
python layachoice_v2_finetune.py --fit-temperature --data data --ckpt out/epoch-7   # → out/calibration/
python layachoice_v2_eval.py --benchmark --data data --ckpt out/epoch-7 --out out/benchmark \
                             --baseline out/baseline/baseline.report.json
python layachoice_v2_finetune.py --export --data data --ckpt out/epoch-7 --out out/release_v2
python plot_val_curves.py            # run with out/analysis/val_metrics.json present

# ── CPU smoke (evidence: smoke/*.log) ──
python layachoice_v2_finetune.py --train --smoke --data <smoke/data> --out <smoke/run>
python layachoice_v2_eval.py --benchmark --smoke --data <smoke/data> --ckpt <smoke/run/epoch-1> --out <smoke/benchmark>
```

What *is* recorded (and therefore not reconstructed): the smoke and capture logs
quoted above; the release manifest (hyper-parameters, selection table,
calibration, final-test metrics, output layout, published SHAs). What is
**reconstructed**: the exact argument vectors and the analysis/smoke working
directories.

## 10. Notes on files that are NOT part of the original run

`run.sh`, `requirements.txt`, `layachoice_v2_verify.py` and `data/SHA256SUMS` in
this directory were added **after** the run (they are not files that ran on the
GPU host). They are kept only as convenience tooling; they are **not** part of
the recovered original project. Historical exploration / diagnostic / early
build-iteration scripts are deliberately **not** in this repo — they remain in
`F:\WorkSpace\Delveta\.tmp\` as the historical archive.
