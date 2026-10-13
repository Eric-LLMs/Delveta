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
| `train.jsonl` | 10,303 | `2065cdbedf1817f66b33ea21711a415d6ab9dc98ea65a077d4f28be09c1600a4` |
| `validation.jsonl` | 1,290 | `0a88be602d740e39b6b79753549fa3d405f2d6dc5213ceda6ad92d19dc06488a` |
| `test.jsonl` | 1,290 | `d048f1b53dbae1ea6a2f1f128382724f316badc32e08f6600d1037185a8cf47f` |

Six MODEL-extraction capabilities (`cap-web-search`, `cap-rag-search`,
`cap-social-search`, `cap-translate`, `cap-add-term`, `cap-create-folder`). Rows
are clustered per (capability, language) by normalized target / near-duplicate
target / near-duplicate input, and **whole clusters** are assigned to one split,
so no target (and no paraphrase of one) can span train/validation/test; the build
aborts on any leakage and is verified against Formal-500 for zero overlap. The
face is **write-only**: its JSON schema always sets `required: []`, and a target
may honestly be partial or **`{}`**; execution-side requiredness is enforced
separately by the Binder (§28.5). This release **keeps** rows whose target leaves a
required slot unfilled (`include_clarify_rows: true`) rather than isolating them.
The frozen bundle is the `data/` tree
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
| Optimizer steps | **1,932** (= ⌈10,303 / 16⌉ × 3) |
| Learning rate | 2e-4, cosine decay, 3% warmup |
| Weight decay / grad-clip | 0.0 / 1.0 |
| Max sequence length | 1,024 (data max below the cap — the supervised target is never truncated) |
| Precision | bf16 |
| Seed | 20261011 |
| Save / eval | every 200 steps, `save_total_limit` 20 |

Only the **assistant tool-call turn** is supervised; the system + user prompt is
masked. The config is
[`config.yaml`](../../scripts/qwen_parameter_extraction_finetune/config.yaml).

**Run-level record** (from the training run's final `trainer_state`):

| Field | Value |
|---|---|
| Optimizer steps | 1,932 |
| Wall-clock | ≈ 4,794 s (≈ 80 min) |
| Final train loss | ≈ 0.005787 |
| Final eval loss | ≈ 0.009731 |
| Hardware | 1 × NVIDIA L4 (24 GB) |

The published checkpoint archive keeps per-checkpoint `trainer_state.json`; the
run-level aggregates above are reported from the run (the archive does not carry
the final aggregate state).

## 5. Checkpoint selection

Selection is a **task metric on the validation split**, never train / eval loss.
Rule: `exact_match`, mode `max`, ties broken on
`[required_missing_rate, field_accuracy, step]`.

| Step | Validation exact-match |
|---:|---:|
| 200 | 0.9039 |
| 400 | 0.9202 |
| 600 | 0.9264 |
| 800 | 0.9271 |
| 1000 | 0.9403 |
| 1200 | 0.9372 |
| **1400** | **0.9504** |
| 1600 | 0.9488 |
| 1800 | 0.9465 |
| 1932 | 0.9496 |

The selected checkpoint is **`checkpoint-1400`** (0.9504 — the maximum). The final
step (1,932, 0.9496) is **not** the best checkpoint. This matches
`reports/selection.json` and `reports/validation_metrics.json` (published in the
model/checkpoints repos).

## 6. Validation metrics (selected checkpoint-1400)

Overall, on the 1,290-row validation split:

| Metric | Value |
|---|---:|
| Exact match | **0.9504** |
| Field accuracy | 0.9600 |
| Required-slot accuracy | 0.9940 |
| Required-slot missing rate | 0.0871 |
| Illegal / unauthorized slot rate | 0.0 |
| JSON parse rate | 1.0 |

Per capability:

| Capability | Samples | Exact match |
|---|---:|---:|
| `cap-add-term` | 203 | 1.0 |
| `cap-create-folder` | 221 | 1.0 |
| `cap-rag-search` | 203 | 0.9951 |
| `cap-social-search` | 253 | 0.9842 |
| `cap-translate` | 208 | 0.7163 |
| `cap-web-search` | 202 | 1.0 |

The weakest capability is `cap-translate` (0.7163); its validation rows most often
carry an *unfilled* model slot declared as `required_missing` (rate ≈ 0.49), so its
errors are dominated by legitimate partial/empty extractions. The illegal-slot
rate is 0.0 across the split, so no error is an unauthorized key or a fabricated
slot; the parse rate is 1.0 (every reply is a well-formed single tool call).

## 7. Model variants

Published as four parallel variants in one Hugging Face repository
([`Delveta-Qwen3-0.6B-PE-v1`](https://huggingface.co/eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1))
— no model copy at the repository root:

| Variant | Path | Format | Size | Content SHA-256 |
|---|---|---|---|---|
| FP16 (recommended, GPU) | `FP16/` | safetensors, float16 | 1.19 GB | `84892d296b4bf166597bd633971405296ef8eb0b29f28ff8d70528f239dff31f` |
| FP32 | `FP32/` | safetensors, float32 | 2.38 GB | `1d1577dc1aaa22d793fdcdd533b8bc3d3d2d272c8030a55b52cee802fb6477cf` |
| INT8 (CPU) | `INT8/` | ONNX, dynamic int8 | 753.8 MB | `94961bf49e4093ee353e3c91b12ff96deb1be72a3435104a60f5e1037e0d9ba2` |
| Q5_K_M (GGUF) | `Q5_K_M/` | GGUF, Q5_K_M | 444.4 MB | `17772c20f3bf356f251c29816b24000dfdbf4b412755024dc087ae63591b98f6` |

Provenance chain: `checkpoint-1400` → PEFT `merge_and_unload` (float32) → cast to
FP16 / FP32; the INT8 ONNX is a dynamic-PTQ export of the merged model (the graph
is converted to fp32 first, then `quantize_dynamic(QInt8)`); the GGUF is
`convert_hf_to_gguf` (f16) → `llama-quantize Q5_K_M` from the merged model.
Every published variant's file content was verified byte-for-byte against the
staged local artifact (only HF-appended `.gitattributes` LFS rules differ).

**Verification, stated honestly.** FP16 and FP32 both reproduce the selected
checkpoint: re-scored over the full 1,290-row validation split they match the
training-path baseline (exact-match 0.9504, parse 1.0, illegal-slot 0). INT8 is a
*dynamic* PTQ variant — it runs end-to-end but is not numerically identical.
Q5_K_M was verified for **native tool-calling under llama.cpp `llama-server
--jinja`** (the runtime Ollama embeds); the repository's `compat_probe.py`
**against Ollama has not been run/passed**.

## 8. Evaluation boundaries

- **Validation only.** Every number above is measured on the 1,290-row
  validation split.
- **Test split NOT evaluated** and **Formal-500 NOT evaluated.** Neither split
  was used for training or checkpoint selection; neither was scored.
- **Validation is not production accuracy.** These are benchmark numbers on a
  frozen curated split, not open-world or production-traffic accuracy, and they
  do not, on their own, authorize switching the production extractor.

## 9. Limitations

1. **INT8 numerical drift.** The dynamic-PTQ INT8 variant runs end-to-end but is
   **not** numerically identical to FP16 / FP32 — its per-token logits differ.
   Documented in the variant card; not a "fake INT8".
2. **Ollama not verified.** The GGUF (`Q5_K_M`) native tool-call was checked under
   llama.cpp `llama-server --jinja` only; the real-service `compat_probe.py`
   **against Ollama has not been run/passed** for this release. Q5_K_M must not be
   reported as Ollama-verified.
3. **`source-grounding` false positives NOT fixed.** A known source-grounding
   limitation remains; this fine-tune does not address it and it must not be
   reported as resolved.
4. **Per-checkpoint default README templates.** Each published
   `checkpoint-*/README.md` is the HF Trainer auto-generated stub; the
   authoritative document is the archive root README.
5. **No `special_tokens_map.json`** in the exported variants; the tokenizer loads
   from `tokenizer.json` + `tokenizer_config.json`, stated in each variant card.
6. **FP16 / FP32 verification** is a full-validation re-score under the frozen
   protocol (not a byte-identity check against the adapter); tool-call level
   verification for INT8 / Q5_K_M is the ONNX / llama.cpp native-tool-call test.

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
checkpoint archive publishes all ten training steps — 200 / 400 / 600 / 800 / 1000 /
1200 / 1400 / 1600 / 1800 / 1932 — plus a weights-only `final/` (its adapter is
identical to `checkpoint-1932`). Each `checkpoint-*/` keeps the full HF Trainer state
(adapter + optimizer + scheduler + RNG), so a run is resumable; only
`checkpoint-1400` is the selected checkpoint, the other steps and `final/` are
archived as training history. The archive deliberately does not re-host the final
merged weights.
