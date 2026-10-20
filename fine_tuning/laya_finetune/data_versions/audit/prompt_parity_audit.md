# Laya V2 Training-vs-Inference Prompt Parity Audit

Status: **PASS** (no unexpected train/serve divergence found; 4 findings are content-budget or test-coverage issues, all symmetric across both paths).

Method: read-only. Every claim below was checked by importing the real modules and running the real builders on frozen rows. No production code, config, data, checkpoint, or log was modified. Scratch runners and raw results are in `.tmp/` (`laya_prompt_parity_audit.py/.json`, `laya_prompt_parity_boundary.py/.json`).

Scope: `fine_tuning/laya_finetune/V2` (dynamic K+1, `LayaChoice-v2`) against the deployed `cap_router` -> `deploy/laya` sidecar path. Primary dataset `recall_threshold_082`; `085` and `088` were also checked for card/token parity.

---

## 1. Real call chains

### Training (input construction)

| Step | Location |
|---|---|
| Row -> validated example | `layachoice_v2_dataset.py:99` `adapt_row` (K, REJECT slot, shuffle, provenance) |
| Batch loop | `layachoice_v2_finetune.py:252` `mode_train`; `:329` `R.build_item` |
| Question shape | `layachoice_v2_render.py:110` `question_of` -> `{"t":"choice","ins":instructions,"crit":dict(zip(order,options))}` |
| Sequence builder | `layachoice_v2_render.py:41` `build_sequence`, called at `:128` |
| Target | `layachoice_v2_render.py:132-135`: one-hot at `gold_index` over `len(markers)`; `label=gold_index` |
| Collation | `layachoice_v2_finetune.py:331` -> `laya.common.collate_items` |
| Forward | `layachoice_v2_finetune.py:340` `model(ids, att, mpos, mmask, qtype)` |
| Constants | `layachoice_v2_spec.py:24-26` (256 / 768 / 1024) |

### Inference (real serving chain)

| Step | Location |
|---|---|
| Live Registry view | `orchestrator.py:160-167` `registry_active_view` -> `entries_by_id` |
| Business K rule | `candidate_aggregation.py:22` `K_DEFAULT=3`; `:41` `normalize_top_k` (K=1..3 kept whole, K>=4 top-3) |
| Selector call | `orchestrator.py:362` `selector.select(message, candidates, entries_by_id=...)` |
| Card rendering | `cap_router/card_renderer.py:51` `render_card`; `:67` `render_choice_question`; `:84` appends `REJECT` **last** |
| Wire payload | `cap_router/backends/service.py:55` `select`; `:64` question; `:65` `{"state": query, "questions": {"capability": question}}`; `:70` POST `/v1/systemone` |
| Sidecar | `deploy/laya/server.py:59` `install()`; `:60` `laya.Agent`; `:84-103` endpoint -> `system_one(max_len=1024, head_max_len=768)` |
| Serving builder | `deploy/laya/layachoice_sequence.py:35` `build_sequence` (patched into `laya.agent` by `:97` `install`) |
| Laya internals | `.venv-laya/.../laya/agent.py:1283` `system_one` -> `:1318` `predict_batch` -> `:1014` `_to_internal` -> `:777` `build_sequence` -> `:1040` `collate_items` -> `:866` `_forward` -> `:882` `logits[r, :k]` |
| Answer mapping | `_decode_answers` returns `keys[int(p.argmax())]` where `keys = q["crit"].keys()`; `service.py:90-102` maps the **label** back to capability/REJECT |

---

## 2. Itemized consistency table

| Item | Result | Evidence |
|---|---|---|
| Base checkpoint | **Consistent** | Both sides use `convaiinnovations/laya@55cf4c4e…`, subfolder `multilingual` (`spec.py`, `deploy/laya/server.py`) |
| Tokenizer source | **Consistent** | Same pinned `tokenizer/tokenizer.json`; sha256 `609d8f4c…b5b6f`; both call `laya.agent._fix_tokenizer_config`; mask=4, cls=2, sep=1, pad=0, vocab 256000 |
| Instructions | **Consistent** | Manifest == `card_renderer.INSTRUCTIONS` for 082/085/088; training uses the same string |
| Capability card text | **Consistent** | Every distinct card (18 caps + REJECT) in 082/085/088 equals `render_card()` of the frozen Registry snapshot: **0 diffs** |
| Card view (`B_noprov`) | **Consistent** | No query examples, provenance, or `matched_example`; same params/negatives rendering |
| REJECT representation | **Consistent** | Same card, same label, exactly one slot per row; production appends it last |
| Question object | **Consistent** | `question_of(ex)` == `Agent._to_internal(production question)` on every fixture |
| Option count / K+1 | **Consistent** | Training: 1..4 options (K=0..3 + REJECT). Production: `effective_k` caps + 1 REJECT, `effective_k <= 3`, so 1..4 |
| Option order | **Intentional difference** | Training: per-row deterministic shuffle. Production: candidate order + REJECT last. The model is trained on arbitrary permutations; both sides map slot -> **label**, never a fixed index |
| Markers | **Consistent** | One marker per option, at the same positions when the presented order is the same; `markers == len(options)` is asserted in both paths |
| `OPTION_MAX_TOKENS` | **Consistent** | 256 on both sides; same truncation call |
| `HEAD_MAX_LEN` | **Consistent** | 768 on both sides; same shared head+options budget and same `opt_budget < 16` even-share fallback |
| `MAX_LEN` | **Consistent** | 1024 on both sides; right-truncation of string states |
| State serialization | **Consistent** | Training rows are all `str` (36,197/36,197); production sends the query `str`; both call `serialize_state` + mask sanitization |
| Padding / attention mask | **Consistent** | Same `collate_items`: right pad with pad id 0, attention 0, marker mask per row |
| qtype | **Consistent** | `QTYPES["choice"]` both sides |
| Forward / logit slice | **Consistent** | Same `model(ids, att, mpos, mmask, qtype)`; both use `logits[:, :k]` with `k = len(markers)` |
| Temperature | **Intentional difference** | Training validation uses T=1; production uses the fitted temperature written by `mode_export`. Top-1 is invariant; probabilities differ by design. `temperature_by_options` is removed at export, so the Agent's bucket lookup falls back to the same scalar `temperature[choice]` |
| Precision | **Intentional difference** | Training may autocast (bf16); serving uses the exported dtype. Same math, not bit-identical across precisions |
| Candidate skipped when no live entry | **Production-only safety** | `card_renderer.py:80-82`; training cards already correspond to live entries. Not a mismatch when the Registry is consistent |
| dtype/shape of target | **Consistent** | One-hot `float`, length = actual option count; `collate_items` pads only across mixed-width batches |

---

## 3. Fixture results (real builders, real tokenizer)

For every fixture:
- training item = `R.build_item(...)`;
- serving item = `SEQ.build_sequence(..., Agent._to_internal(same question))`.

The serving builder reproduces the training sequence **token-for-token** when given the row's presented order. When the row's REJECT is already last, the **real production question** (candidate order + REJECT last) is also token-for-token identical.

| Dataset | K | options | seq len | markers | state tok / room | row-order ids | prod ids (REJECT-last) | target / gold / reject slot |
|---|---|---|---|---|---|---|---|---|
| 082 | 0 | 1 | 94 | 1 | 13 / 943 | exact | exact | ok |
| 082 | 1 | 2 | 250 | 2 | 23 / 797 | exact | order-only diff | ok |
| 082 | 2 | 3 | 480 | 3 | 26 / 570 | exact | exact | ok |
| 082 | 3 | 4 | 691 | 4 | 29 / 362 | exact | order-only diff | ok |
| 082 | 3 (boundary) | 4 | 661 | 4 | 16 / 379 | exact | exact | ok |
| 085 | 0..3 | 1..4 | 94..765 | = options | ≤39 / ≥270 | exact | exact when REJECT last | ok |
| 088 | 0..3 | 1..4 | 101..783 | = options | ≤29 / ≥270 | exact | exact when REJECT last | ok |

“order-only diff” = the two sequences contain the same cards and differ only in card position (training shuffle vs candidate order). This is the designed difference; the assertion in `tests/test_laya_v4_prompt_parity.py` encodes the same rule.

### Forward / logit-slot check (local smoke checkpoint)

Using `.tmp/laya_v2_smoke_recall_082/epoch-1` on CPU:

| K | same order? | max abs logit diff | row-order argmax label | production argmax label |
|---|---|---|---|---|
| 0 | yes | 0.0 | REJECT | REJECT |
| 1 | no | 0.060 | cap-summary | REJECT |
| 2 | yes | 0.0 | cap-translate | cap-translate |
| 3 | no | 6.232 | cap-read-document | cap-read-document |

- Same-order sequences produce **bit-identical** logits.
- Reordered sequences produce different logits because the encoder sees the same cards at different positions. This is expected: training shuffles options per row, so every order is in-distribution. Each path's argmax maps through its own `order`/`criteria` list; no index crosses the boundary.
- The K=1/K=3 prediction differences above are properties of a 1-step smoke checkpoint; they are **not** evidence of a prompt mismatch.

---

## 4. Token-level findings

- **Token budgets are identical**: 256 per option, 768 shared head+options, 1024 total.
- **`HEAD_MAX_LEN` semantics match**: it is a shared budget, not a per-option budget. The `opt_budget < 16` even-share fallback and `head_ids[:max(8,opt_budget)]` floor are identical in both copies.
- **AST-level identity**: the two `build_sequence` implementations are identical after normalizing the one source-level difference (`S.OPTION_MAX_TOKENS` vs `OPTION_MAX_TOKENS`).
- **No `MAX_LEN` truncation occurs** in 082: max state = 46 tokens; max assembled sequence ≈ 788 tokens; room ≥ 255. Zero rows lose state tokens.
- **Option-cap boundary**: one card, `cap-vision`, is **283 tokens raw**, so the 256 cap drops 27 tokens: the tail of the `asset_id` description and the entire optional `question` parameter. See finding F1.

---

## 5. Findings and minimal suggestions

### F1 — `cap-vision` card exceeds the 256-token option cap (medium; symmetric)

The raw card is 283 tokens. Both training and serving truncate it identically, so parity holds, but the model never sees the optional `question` slot. Affected rows (train):

| Dataset | K=1 | K=2 | K=3 |
|---|---|---|---|
| 082 | 942 | 890 | 345 |
| 085 | 1,352 | 243 | 26 |
| 088 | 1,404 | 28 | 0 |

Minimal fix options (choose one, then regenerate data and retrain):
1. Shorten/re-render the `cap-vision` card so it fits in 256 tokens.
2. Raise `OPTION_MAX_TOKENS` **together with** `HEAD_MAX_LEN`/`MAX_LEN` and the exported model config (256 is not safe for four long options: 4×283 > 768).
3. Accept it explicitly and document that `cap-vision.question` is not model-visible.

### F2 — Even-share fallback triggers on 3 rows (low; symmetric)

In 082, 3 K=3 rows have option sums > 752, so every option is cut to 188 tokens (max 96 tokens removed). 085/088 trigger 0 rows. Both paths apply the same rule, so this is not skew. If these rows matter, the same budget redesign as F1 covers it.

### F3 — Base config says `head_max_len=256`, both sides force 768 (low; config coupling)

`.../multilingual/rl_agent_config.json` still carries `head_max_len: 256`. Training `read_cfg` overwrites it to 768 (`layachoice_v2_finetune.py:76-77`), export writes 768, and the server passes 768 explicitly. Nothing diverges in this chain, but any tool that loads the base checkpoint without those overrides would use 256 and silently produce different inputs. Recommend documenting this or pinning 768 in the base config when the next checkpoint is exported.

### F4 — Test coverage is weaker than it looks (medium)

- `tests/test_laya_aligned_cards.py` is **entirely skipped**: its `ALIGNED` path (`V2/data/aligned`) and `V2/data/manifest.json` do not exist in the current layout.
- `tests/test_laya_v4_prompt_parity.py` is solid but covers only `recall_threshold_085`, and only runs where `laya` is installed (`.venv-laya`).
- `tests/test_cap_router_service.py::test_render_card_is_byte_identical_to_the_frozen_bundle` is tautological: it reverse-parses a rendered option into an entry and re-renders it. It cannot detect Registry drift.
- No test re-derives the current `full_r1` cards from the live Registry at CI time.

Minimal suggestion: point the aligned-card tests at the current `data/recall_threshold_*` bundles, extend parity to 082/085/088, and keep one live-DB drift test that actually reads `list_capabilities`.

---

## 6. Unverified / gaps

- **Live Registry drift at inference time was not verified**: no DB was reachable during the audit. The training cards match the frozen snapshot exactly, and `full_r1/build_dataset.py:2164-2168` derives them with the production `render_card` over the live DB at build time. However, production re-reads the live table every turn (`registry_active_view`). If a capability description or parameter schema changed after the bundle was built, cards would drift. Re-run `tests/test_laya_aligned_cards.py::test_snapshot_still_matches_the_live_registry` (or a live dump) with the DB up before training.
- The forward check used the 1-step smoke checkpoint, not the previous formal V2 checkpoint (not present locally). It validates the plumbing and slot mapping, not model quality.
- No GPU/bf16-vs-fp32 numerical comparison was run; precision is an intentional difference.

---

## 7. Verdict

**PASS** — the training input and the real inference input are constructed by equivalent code, with identical constants, tokenizer, card text, question shape, markers, and slot->label mapping. The only differences are the designed ones: per-row option shuffle in training vs candidate-order+REJECT-last in production, and fitted temperature at serving.

The findings above are input-budget content loss (F1/F2), a config-coupling caveat (F3), and test-coverage gaps (F4). None of them constitutes a train/serve prompt mismatch, and none was fixed in this audit.
