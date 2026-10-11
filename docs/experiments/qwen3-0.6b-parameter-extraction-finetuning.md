# Qwen3-0.6B Parameter-Extraction Fine-Tuning

The experiment record for **`Delveta-Qwen3-0.6B-PE-v1`** — a local
parameter-extraction model for the Delveta Intent Funnel's argument-acquisition
stage. The data-provenance and byte-render verification of the frozen bundle live
separately in
[Qwen Parameter Extraction — Provenance & Render Verification](Qwen-Parameter-Extraction-Fine-Tuning-Provenance.md).

## 1. Scope and position

The model does **argument extraction only**. Capability selection happens
upstream (§26 LayaChoice / the Matcher); this model's single job is

```
(capability_name, user_query, allowed_slots)  ->  slot_values
```

It never selects, re-selects or questions the capability, never authors a
capability name, and never converses. It is the current implementation of the
extractor seam described in the architecture (§28.4), reached by both acquisition
paths through the same `deps.argument_extractor` seam; the Binder remains the
final gate (§28.5). The deployment target is the local OpenAI-compatible
tool-intent service consumed by the funnel; the base HF weights are the training
input, never the Ollama GGUF (which is a deployment artifact, §8).

**Training protocol.** The native OpenAI **`messages + tools + assistant
tool_calls`** wire contract: one function whose `name` is the capability,
`tool_choice="required"`, and a JSON-schema face that exposes **only the
model-owned slots**. `function.arguments` is a flat JSON string `{slot: value}`
(no nested wrapper, no capability-selection field).

## 2. Base model

| Setting | Value |
|---|---|
| Base model | `iromu/Qwen3-0.6B-tools` |
| Base revision (pinned) | `10e61428cfbcc907c84d52a6b9d7505903aedd12` |
| Fine-tuning | LoRA (PEFT), no full-parameter update |
| Loaded dtype | `bfloat16` on a bf16 GPU (`float32` forced on CPU) |

## 3. Dataset

Built deterministically from the PE v2 candidates with the labels **re-derived**
against the frozen contract (`pe_contract.py`), because the retired C-class DET
lane made every argument MODEL-owned and invalidated PE v2's
`asked_slots` / `detected_arguments`. No model call is made in the build.

| Split | Rows | SHA-256 |
|---|---:|---|
| `train.jsonl` | 8,679 | `dbd06f3a3e8e30a0d6395ec0fb5d80f3bf0ba85ea4bb7a9dfd0cd78ddab51222` |
| `validation.jsonl` | 1,041 | `addc0baa8fa61624ad054a54dfcac43e15ec27f03b985a61543db18d3185f949` |
| `test.jsonl` | 1,080 | `507bc036b4d9a15a797097e8074ba7471abcf2188f3ab0ff85458b49281327e2` |

Six MODEL-extraction capabilities (`cap-web-search`, `cap-rag-search`,
`cap-social-search`, `cap-translate`, `cap-add-term`, `cap-create-folder`). Rows
are clustered per (capability, language) by normalized target / near-duplicate
target / near-duplicate input, and **whole clusters** are assigned to one split,
so no target (and no paraphrase of one) can span train/validation/test; the build
aborts on any leakage and is verified against Formal-500 for zero overlap. The
frozen bundle is the `data/` tree
([`scripts/qwen_parameter_extraction_finetune/data/`](../../scripts/qwen_parameter_extraction_finetune/data/))
and is published as
[`Delveta-Qwen3-0.6B-PE-v1-Data`](https://huggingface.co/datasets/eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1-Data).

## 4. Training configuration

| Setting | Value |
|---|---|
| LoRA | r = 8, alpha = 16, dropout = 0.0, bias = none, targets = `[q_proj, k_proj, v_proj, o_proj]` |
| Trainable parameters | 2,293,760 |
| Epochs | 3 |
| Batch | per-device 8 × grad-accum 2 = **effective 16** |
| Optimizer steps | **1,629** (= ⌈8,679 / 16⌉ × 3) |
| Learning rate | 2e-4, cosine decay, 3% warmup |
| Weight decay / grad-clip | 0.0 / 1.0 |
| Max sequence length | 1,024 (data max 914 — the supervised target is never truncated) |
| Precision | bf16 |
| Seed | 20261011 |
| Save / eval | every 200 steps, `save_total_limit` 20 |

Only the **assistant tool-call turn** is supervised; the system + user prompt is
masked. The config is
[`config.yaml`](../../scripts/qwen_parameter_extraction_finetune/config.yaml).

**Run-level record** (from the training run's final `trainer_state`):

| Field | Value |
|---|---|
| Optimizer steps | 1,629 |
| Wall-clock | ≈ 3,603 s (≈ 60 min) |
| Final train loss | ≈ 0.00442 |
| Final eval loss | ≈ 0.0009873 |

The published checkpoint archive keeps per-checkpoint `trainer_state.json`; the
run-level aggregates above are reported from the run (the archive does not carry
the final aggregate state). Checkpoint-level eval loss re-read from the archive:
step 1000 = 8.3375e-4, step 1600 = 9.6503e-4.

## 5. Checkpoint selection

Selection is a **task metric on the validation split**, never train / eval loss.
Rule: `exact_match`, mode `max`, ties broken on
`[required_missing_rate, field_accuracy, step]`.

| Step | Validation exact-match |
|---:|---:|
| 200 | 0.9750 |
| 400 | 0.9875 |
| 600 | 0.9914 |
| 800 | 0.9846 |
| **1000** | **0.9933** |
| 1200 | 0.9914 |
| 1400 | 0.9933 |
| 1600 | 0.9923 |
| 1629 | 0.9933 |

The selected checkpoint is **`checkpoint-1000`** (0.9933 — the earliest step at
the peak; later equal scores lose the `step` tie-break). This matches
`reports/selection.json` and `reports/validation_metrics.json` (published in the
model/checkpoints repos).

## 6. Validation metrics (selected checkpoint-1000)

Overall, on the 1,041-row validation split:

| Metric | Value |
|---|---:|
| Exact match | **0.9933** |
| Field accuracy | 0.9953 |
| Required-slot accuracy | 0.9942 |
| Required-slot missing rate | 0.0 |
| Illegal / unauthorized slot rate | 0.0 |
| JSON parse rate | 1.0 |

Per capability:

| Capability | Samples | Exact match |
|---|---:|---:|
| `cap-add-term` | 173 | 1.0 |
| `cap-create-folder` | 175 | 1.0 |
| `cap-rag-search` | 171 | 1.0 |
| `cap-social-search` | 175 | 0.96 |
| `cap-translate` | 180 | 1.0 |
| `cap-web-search` | 167 | 1.0 |

The only non-perfect capability is `cap-social-search` (0.96); the illegal-slot
and required-missing rates are 0.0 across the split, so every error is a
value/slot-content mismatch, never an unauthorized key or a fabricated slot.

## 7. Model variants

Published as four parallel variants in one Hugging Face repository
([`Delveta-Qwen3-0.6B-PE-v1`](https://huggingface.co/eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1))
— no model copy at the repository root:

| Variant | Path | Format | Size | Content SHA-256 |
|---|---|---|---|---|
| FP16 (recommended, GPU) | `FP16/` | safetensors, float16 | 1.19 GB | `bbbb00daaad0b1ff1e7c75b4fa76bafa502110a4f5f676bec62ac81ee803f795` |
| FP32 | `FP32/` | safetensors, float32 | 2.38 GB | `f088362c…da67` |
| INT8 (CPU) | `INT8/` | ONNX, dynamic int8 | 601.5 MB | `e001e661…1dc3` |
| Q5_K_M (Ollama) | `Q5_K_M/` | GGUF, Q5_K_M | 444.4 MB | `fd4d5095…f896d` |

Provenance chain: `checkpoint-1000` → PEFT `merge_and_unload` (float32) → cast to
FP16 / FP32; the INT8 ONNX is a dynamic-PTQ export of the merged model; the GGUF
is `convert_hf_to_gguf` (f16) → `llama-quantize Q5_K_M` from the merged model.
Every published variant's file content was verified byte-for-byte against the
staged local artifact (only HF-appended `.gitattributes` LFS rules differ).

## 8. Evaluation boundaries

- **Validation only.** Every number above is measured on the 1,041-row
  validation split.
- **Test split NOT evaluated** and **Formal-500 NOT evaluated.** Neither split
  was used for training or checkpoint selection; neither was scored.
- **Validation is not production accuracy.** These are benchmark numbers on a
  frozen curated split, not open-world or production-traffic accuracy, and they
  do not, on their own, authorize switching the production extractor.

## 9. Limitations

1. **INT8 numerical drift.** The dynamic-PTQ INT8 variant agrees with the FP32
   variant on ≈ 0.563 of per-token argmax positions, with a max logit difference
   ≈ 22.2. Documented in the variant card; not a "fake INT8".
2. **`source-grounding` false positives NOT fixed.** A known source-grounding
   limitation remains; this fine-tune does not address it and it must not be
   reported as resolved.
3. **Per-checkpoint default README templates.** Each published
   `checkpoint-*/README.md` is the HF Trainer auto-generated stub; the
   authoritative document is the archive root README.
4. **No `special_tokens_map.json`** in the exported variants; the tokenizer loads
   from `tokenizer.json` + `tokenizer_config.json`, stated in each variant card.
5. **FP16 / FP32 verification** is load + minimal `generate()` only; tool-call
   level verification is covered by the INT8 and Q5_K_M variants.

## 10. Reproducibility

The whole chain lives in
[`scripts/qwen_parameter_extraction_finetune/`](../../scripts/qwen_parameter_extraction_finetune/);
every stage shares one contract (`pe_contract.py`) and one loader/renderer
(`sft_data.py`), so build, train, eval and export cannot drift.

```bash
# 0. environment + frozen-data preflight (NEVER starts training)
bash scripts/qwen_parameter_extraction_finetune/run.sh --check

# 1. verify the frozen bundle (stdlib; hash + counts + leakage)
python scripts/qwen_parameter_extraction_finetune/validate_dataset.py

# 2. real LoRA training (preflight, then train.py)
bash scripts/qwen_parameter_extraction_finetune/run.sh

# 2'. launchability check without training
bash scripts/qwen_parameter_extraction_finetune/run.sh --dry-run

# 3. score every checkpoint and select
python scripts/qwen_parameter_extraction_finetune/evaluate.py --all-checkpoints

# 4. export adapter / merged HF / GGUF plan
python scripts/qwen_parameter_extraction_finetune/export.py --which all
```

`run.sh` is a thin, safe wrapper: it locates the script directory from its own
path (not the CWD), verifies python / deps / config / data, **aborts on any
frozen-bundle SHA-256 mismatch**, checks disk and GPU, then hands off to the REAL
entry (`train.py`). It never auto-deletes, never overwrites a checkpoint and
never resumes on its own. Env overrides: `PE_PYTHON`, `PE_CONFIG`,
`PE_MIN_GPU_MEM_MB` (8000), `PE_MIN_DISK_MB` (4096), `PE_ALLOW_CPU=1`,
`PE_SKIP_HASH=1` (debug only). No Hugging Face token is read or logged.

**Prerequisites:** Python ≥ 3.10 with `torch`, `transformers`, `peft`, `PyYAML`
(`requirements-smoke.txt` covers the CPU smoke path); a bf16-capable GPU for the
real run. `train.py` reads `data/train.jsonl` **only** (any other filename, or a
path naming `formal500`, is refused) and re-checks the frozen train hash before
training.

## 11. Artifacts and provenance

| Artifact | Home |
|---|---|
| Training code / entry | [`scripts/qwen_parameter_extraction_finetune/`](https://github.com/Eric-LLMs/Delveta/tree/main/scripts/qwen_parameter_extraction_finetune) |
| Model (FP16 / FP32 / INT8 / Q5_K_M) | [Delveta-Qwen3-0.6B-PE-v1](https://huggingface.co/eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1) |
| Frozen dataset | [Delveta-Qwen3-0.6B-PE-v1-Data](https://huggingface.co/datasets/eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1-Data) |
| Checkpoints (full trainer state) | [Delveta-Qwen3-0.6B-PE-v1-checkpoints](https://huggingface.co/eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1-checkpoints) |
| Data provenance & render check | [Provenance & Render Verification](Qwen-Parameter-Extraction-Fine-Tuning-Provenance.md) |

The published model, dataset and checkpoints are pinned by their repository
revisions; the base revision `10e61428…aedd12` fixes the starting weights. The
checkpoint archive keeps the five checkpoints 400 / 800 / 1000 / 1200 / 1600 with
their full HF Trainer state (adapter + optimizer + scheduler + RNG), so a run is
resumable — it deliberately does not re-host the final merged weights.
