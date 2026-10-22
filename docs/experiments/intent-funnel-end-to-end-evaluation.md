<style>
/* Wide comparison tables: keep each cell on one line and scroll horizontally
   instead of wrapping the long column headers. */
table {
  display: block;
  width: max-content;
  max-width: 100%;
  overflow-x: auto;
  white-space: nowrap;
}
</style>

**[English](#chat-candidate-set-change--recall-threshold-experiment)** | **[中文](#聊天候选卡片改造--召回阈值实验报告)**

# Chat Candidate-Set Change + Recall Threshold Experiment

> **Document status:** Parts 1–2 preserve the original four-arm card-surface experiment results. The `after rule cleanup` columns and the `intent_rule_no_model` deletion comparison report new measurements from the rule-cleanup rerun; they do not modify or recompute the historical baseline. Decision Accuracy includes only requests that actually reached LayaChoice-v2, excluding requests handled directly by legacy intent rules.

## Part 1: Key Metrics and Conclusions

### Four Key Metrics

- **End-to-End Accuracy:** Of the 500 test requests, the share whose final decision or routing outcome matches the labeled target, with all 500 requests as the denominator. This measures overall accuracy across the intent pipeline's capability selection and routing; it does not verify whether the business task was actually executed successfully.
- **End-to-End Tool Routing Accuracy:** Of the 414 tool requests, the proportion correctly routed to their labeled target tools. The denominator is 414, not all 500 requests. This metric measures the overall correctness of tool routing in the intent pipeline and is affected by both candidate-set recall and tool selection. It does not measure whether the selected tool executes successfully or completes the requested task.
- **Decision Accuracy** — Of all requests evaluated by LayaChoice-v2, how many received the
  correct decision, either being passed to a tool or rejected. The denominator is the total
  number of turns that reached LayaChoice-v2.
- **False Pass Rate** — Of the 86 requests that should be handled by the Agent rather than
  directly executed as tools, how many were incorrectly passed through for tool execution.
  This measures how often non-allowlist requests were
  incorrectly allowed through.

| Metric | 0.88 · deleting `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.82 · deleting `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.88 · using `intent_rule_no_model`<br>· filters out non-allowlist intents | 0.88 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.82 · using `intent_rule_no_model`<br>· filters out non-allowlist intents | 0.82 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents |
|---|---|---|---|---|---|---|
| End-to-End Accuracy | **433/500 = 86.60%** | **462/500 = 92.40%** | 402/500 = 80.40% | 406/500 = 81.20% | 416/500 = 83.20% | 434/500 = 86.80% |
| End-to-End Tool Routing Accuracy (affected by both candidate-set recall and tool selection) | **347/414 = 83.82%** | **377/414 = 91.06%** | 318/414 = 76.81% | 320/414 = 77.29% | 348/414 = 84.06% | 349/414 = 84.30% |
| Decision Accuracy | **428/457 = 93.65%** | **459/493 = 93.10%** | 306/336 = 91.07% | 384/411 = 93.43% | 359/410 = 87.56% | 414/447 = 92.62% |
| False Pass Rate | **0/86 = 0%** | **1/86 = 1.16%** | 2/86 = 2.33% | 0/86 = 0% | 18/86 = 20.93% | 1/86 = 1.16% |

### Final Decision

The final design adopts three decisions:

1. **Use a recall threshold of 0.82.**
2. **Delete `intent_rule_no_model`.**
3. **Allow recall of non-allowlist intents, then apply the allowlist check after the model decision to block capabilities that may not run directly in Chat.**

Why this option:

- **Highest End-to-End Tool Routing Accuracy**: 0.82 with `intent_rule_no_model` deleted reaches **91.06%** (83.82% at 0.88; 84.30% at 0.82 with the path; 77.29% at 0.88 with the path).
- **Highest End-to-End Accuracy**: **92.40%** at 0.82, versus 86.60% at 0.88.
- **Better recall coverage**: K=0 has only 4 rows at 0.82 but 40 at 0.88; 38 tool requests at 0.88 get no card at all because the threshold is too high.
- **The model-driven decision pipeline outperforms the original rule-based routing approach overall:** In this 500-case evaluation, removing `intent_rule_no_model` improved both End-to-End Accuracy and End-to-End Tool Routing Accuracy compared with the corresponding configurations that retained the rule path. These results indicate that, within the scope of this evaluation, the model-driven decision pipeline can serve as an alternative to the original rule-based path, potentially reducing reliance on dedicated routing rules and their manual maintenance.

### Final Configuration: Processing Breakdown and Denominator Definitions for 500 Queries

The 500 queries split into three mutually exclusive groups that sum to 500:

| Processing path | Rows | Enters LayaChoice-v2 | In the Decision Accuracy denominator | End-to-end scoring |
|---|---|---|---|---|
| Entered LayaChoice-v2 and produced a decision | 493 | Yes | Yes | 459 correct / 34 incorrect |
| `NO_CANDIDATE`: no candidates recalled, handed directly to the Agent | 4 | No | No | 0 correct / 4 incorrect |
| `MATCHER_HIT`: R_HIT table match, capability already determined | 3 | No | No | 3 correct / 0 incorrect |
| **Total** | **500** | — | — | **462 correct / 38 incorrect** |

**Denominator definition.** The 493 rows that entered LayaChoice-v2 consist of `LAYA_SELECT` 383, `LAYA_REJECT` 31, and `FUNNEL_ALLOWLIST_REJECT` 79. The numerator is the 459 rows with a correct decision (374 + 6 + 79), hence **459/493 = 93.10%**. The remaining 7 rows are in neither the numerator nor the denominator: the 4 `NO_CANDIDATE` rows recalled no candidates and never reached LayaChoice-v2, and the 3 `MATCHER_HIT` rows had their capability determined by the Matcher and required no Laya selection. Thus **500 = 493 + 4 + 3**.

**Tool-routing numerator.** End-to-End Tool Routing Accuracy uses the 414 tool requests as its denominator, and its numerator counts a row when the final routing outcome is the labeled tool. It therefore includes the 3 `MATCHER_HIT` rows: at 0.82 the numerator is **377 = 374 correct Laya selections + 3**, and at 0.88 it is **347 = 344 + 3**. In the four baseline arms it also includes the 15 rows that `intent_rule_no_model` routed directly to the labeled tool (for example, 0.82 with recall enabled is 349 = 331 + 15 + 3). Those 15 rows never reached LayaChoice-v2 and are excluded from Decision Accuracy.

**How the two groups outside Laya are scored.**

- All 4 `NO_CANDIDATE` rows are tool requests (the target capability is `cap-web-search` and Gold is in the allowlist). The threshold recalled no cards, so they were handed directly to the Agent. End-to-end they score as failures: 0/4 correct.
- The 3 `MATCHER_HIT` rows are `vi-03` (`cap-vision`), `pe-01` (`cap-pdf-extract-text`), and `f5-cf-07` (`cap-create-folder`). The Matcher determined the capability and selected Gold directly. End-to-end they score as correct: 3/3.

**The two accuracies use different denominators.** Decision Accuracy counts only the 493 rows on which LayaChoice-v2 actually decided: **459/493 = 93.10%**. End-to-End Accuracy counts all 500 rows, including the 3 correct `MATCHER_HIT` rows and the 4 failed `NO_CANDIDATE` tool requests: **462/500 = 92.40%**. The difference is 7 rows in the denominator and 3 rows in the numerator, all attributable to the 7 queries that did not enter LayaChoice-v2.

**Corresponding breakdown at 0.88.** Under the same definitions, **500 = 457 + 40 + 3**: 457 rows entered LayaChoice-v2, giving Decision Accuracy **428/457 = 93.65%** (`LAYA_SELECT` 344 + `LAYA_REJECT` 5 + `FUNNEL_ALLOWLIST_REJECT` 79). Of the 40 `NO_CANDIDATE` rows, 38 are tool requests and score as failures, while 2 are non-allowlist requests (`cap-artifact`, `f5-ar-05` / `w3-ar-04`) correctly handed to the Agent and scored as correct. All 3 `MATCHER_HIT` rows are correct. End-to-End Accuracy is therefore **433/500 = 86.60%** (428 + 3 + 2). Neither threshold has any `LAYA_NO_DECISION` or `OTHER` rows.

> The first two columns are this run after deleting `intent_rule_no_model` with non-allowlist intents recalled (commit `833651f`). The last four columns are the historical baseline using `intent_rule_no_model`, split by whether non-allowlist intents are filtered out before recall, using the same frozen per-case audit as the rest of this report. Their Decision Accuracy numerators exclude the 2 requests that legacy intent rules handled directly (`intent_rule_no_model`) in every baseline arm, because they never reached LayaChoice-v2; the denominators (336 / 411 / 410 / 447) never contained these 2 REJECT judgments or their other handling results. See the `intent_rule_no_model` deletion comparison below for the before/after comparison.

> **Rounds that did not reach LayaChoice-v2:** Some requests for non-allowlist capabilities were rejected before reaching the decision model. The retrieval model's score threshold itself filtered out non-intent queries, leaving no candidates (`NO_CANDIDATE`) and resulting in immediate rejection.
>
> - **0.88 · using `intent_rule_no_model` ·** non-allowlist intents filtered out by the retrieval threshold: 76 REJECTs
> - **0.88 · using `intent_rule_no_model` ·** non-allowlist intents retrieved but rejected by the retrieval threshold: 2 REJECTs
> - **0.82 · using `intent_rule_no_model` ·** non-allowlist intents filtered out by the retrieval threshold: 37 REJECTs
> - **0.82 · using `intent_rule_no_model` ·** non-allowlist intents retrieved but rejected by the retrieval threshold: 0 REJECTs

### Conclusions

The old pipeline hid the cards that chat is not allowed to execute *before* the model chose.
A user asking to edit a file saw only the "read file" card and had no choice but to pick it.
The new pipeline shows every tool card, lets the model choose, and then checks whether the
chosen capability may run in chat; if it may not, the turn is rejected and handed to the
Agent.

1. **False Pass Rate dropped significantly.** In this test, the 0.88 configuration fell from 2.33% to 0%, and the 0.82 configuration from 20.93% to 1.16%. The results indicate that moving candidate-card filtering to an execution-permission check after model selection helps reduce false passes; remaining errors may still come from other stages, such as incorrect model selection.
2. **The card-surface change mainly cuts false passes; deleting `intent_rule_no_model` lifts
   the End-to-End Tool Routing Accuracy further.** The card-surface change alone moved this metric
   by +0.48pp /
   +0.24pp; after deleting the rule path, 0.88 rises from 77.29% to 83.82% (+6.53pp) and
   0.82 from 84.30% to 91.06% (+6.76pp). In the old pipeline the "no intent recalled" hidden
   requests were already counted as correctly handed to the Agent; the new pipeline sends
   them to the model and rejects them, with the same correct outcome, so the card-surface
   gain shows up mainly in false pass.
3. **Top-1 selection is stable.** After the card-surface change, strict accuracy is
   91.79% / 91.18%; after deleting the rule path it is 92.23% / 91.89%. There is no
   regression.
4. **The one remaining miss** is `w3-ef-16` at 0.82. Its pool already contained the correct
   `cap-edit-file`, but the model picked `cap-read-file`. This is a genuine selection error,
   not a forced choice.
5. **0.82 is the better overall trade-off.** It gives up one genuine false pass but gains
   7.24pp on End-to-End Tool Routing Accuracy and 5.80pp on End-to-End Accuracy; 0.88 keeps false pass at 0%,
   yet 38 tool requests at K=0 still receive no card at all.

## Part 2: Detailed Statistics

### What was wrong

A real example: the user says "change `strict` in tsconfig.json from false to true".

The right capability is `cap-edit-file`. The old pipeline dropped that card after recall
and before the intent candidate set recall, so the model saw only `cap-read-file`. It picked
"read file", and the system executed the request as a read.

The model was not the problem. The correct card had been filtered out.

All 18 false passes at 0.82 before the change have this shape: the pool contained no correct
answer, only one card in the allowlist.

### What changed

```
Old: recall → filter out non-allowlist intents → intent candidate set recall (remaining tools) → model chooses → execute
New: recall → intent candidate set recall (all tools, allowing recall of non-allowlist intents) → model chooses → if the predicted tool is non-allowlist → reject
```

- **Where the intent allowlist applies**: before the change it was checked right after recall,
  so non-allowlist intents were dropped before the intent candidate set recall.
  After the change non-allowlist intents are recalled, every candidate reaches the model, and the
  allowlist is checked on the decision.
- **Intent candidate set recall stage**: every enabled, non-research tool enters the intent
  candidate set. Allowlist membership is no longer consulted here.
- **After the model decides**: if the pick is a capability that chat may not execute
  (edit file, produce artifacts, run shell), the business layer returns a rejection and the
  turn falls open to the Agent. Nothing executes, and the system does not silently fall back
  to the second-best card.
- **The Agent plane is untouched**: all three tools stay registered and remain available to
  the page, the desktop client and the Agent. Only the chat surface stops hiding their cards.

In one sentence: it used to be "you cannot see it, so pick something else"; now it is
"you can see it, and I will stop you if you pick it".

### How it was measured

- Dataset: `formal500_dataset_v2`, 500 rows.
  - 414 rows: a tool should be used, and that tool is allowed in chat.
  - 86 rows: the true target is a capability chat may not execute (edit file 48 /
    shell 20 / artifact 18).
- Model: the production LayaChoice-v2 (dynamic K+1, 0–3 capability cards plus one `REJECT`
  card).
- Environment: the full production app driven through a real Electron UI. Tool execution is
  replaced by a virtual executor; nothing edits files, runs commands, or calls an online
  LLM.
- Four arms: before/after × 0.88/0.82. Each arm ran all 500 rows with 0 failures and
  0 timeouts.
- Argument extraction is off. This measures candidate reranking → top-1 selection only,
  not argument filling or real execution.

### Detailed results

#### Grouped by how many cards the model was given

> **Evaluation Methodology:** Each request first passes through the retrieval stage
> (embedding-based retrieval + score thresholding) to generate a candidate tool set. The
> decision model, LayaChoice-v2, then selects the target tool from the candidates or predicts
> REJECT. The table reports end-to-end correctness across the retrieval and decision stages,
> grouped by the number of candidate cards presented to the decision model (K). K=0 means no
> candidate cards were retrieved, so the model cannot select a tool and the request must be
> handed off to the Agent.

| Intent candidate list size | 0.88 · using `intent_rule_no_model`<br>· filters out non-allowlist intents | 0.88 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.82 · using `intent_rule_no_model`<br>· filters out non-allowlist intents | 0.82 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.88 · deleting `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.82 · deleting `intent_rule_no_model`<br>· allows recall of non-allowlist intents |
|---|---|---|---|---|---|---|
| K=0 (no intent recalled) | 76/115 = 66.09% | 2/40 = 5.00% | 37/41 = 90.24% | 0/4 = 0% | 2/40 = 5.00% | 0/4 = 0% |
| K=1 | 274/295 = 92.88% | 338/355 = 95.21% | 202/222 = 90.99% | 201/206 = 97.57% | 380/398 = 95.48% | 227/232 = 97.84% |
| K=2 | 28/37 = 75.68% | 42/52 = 80.77% | 95/111 = 85.59% | 107/116 = 92.24% | 44/55 = 80.00% | 117/127 = 92.13% |
| K=3 | 4/4 = 100% | 4/4 = 100% | 62/77 = 80.52% | 106/125 = 84.80% | 4/4 = 100% | 115/134 = 85.82% |
| Other paths | 20/49 = 40.82% | 20/49 = 40.82% | 20/49 = 40.82% | 20/49 = 40.82% | 3/3 = 100% | 3/3 = 100% |
| **Total** | **402/500 = 80.40%** | **406/500 = 81.20%** | **416/500 = 83.20%** | **434/500 = 86.80%** | **433/500 = 86.60%** | **462/500 = 92.40%** |
| **Turns that reached the model** | **336** | **411** | **410** | **447** | **457** | **493** |

> In the rule-cleanup arms, "Other paths" contains only the 3 R_HIT (matcher) rows. The 46 rows that legacy intent rules handled directly (`intent_rule_no_model`) no longer exist; they now enter the funnel and are counted in their actual K group.
>
> In the baseline arms, "Other paths" is 20/49 = 15 requests where the rule executed the tool directly + 31 requests where the rule handed off to the Agent + 3 `R_HIT`. All 46 of the first two are `intent_rule_no_model` (handled by legacy intent rules, never reaching the model), and the two outcomes are mutually exclusive. This table measures end-to-end correctness, so the 2 non-allowlist `intent_rule_no_model` rows (rule handed off to the Agent) count as correctly handed to the Agent; they are **not** part of the Laya turn count and are **not** in the reject numerator, so the same rows are not subtracted twice.

The key change: **turns with no intent recalled dropped from 115 to 40 at 0.88 and from 41 to 4 at
0.82.** Those turns never had a chance to reach the model before; now they do, and they are
correctly rejected.

> The K=0 percentage looks lower, but that does not mean regression. The old pipeline handed
> these "no intent recalled" turns straight to the Agent, which the new scoring counts as correct. The
> new pipeline sends them to the model, which recognises them and rejects them. The outcome
> is equally correct; only the path differs.

#### Reject decisions

| Item | 0.88 · using `intent_rule_no_model`<br>· filters out non-allowlist intents | 0.88 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.82 · using `intent_rule_no_model`<br>· filters out non-allowlist intents | 0.82 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.88 · deleting `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.82 · deleting `intent_rule_no_model`<br>· allows recall of non-allowlist intents |
|---|---|---|---|---|---|---|
| Requests that should be rejected | 86 | 86 | 86 | 86 | 86 | 86 |
| Correctly rejected | 6/86 = 6.98% | 82/86 = 95.35% | 29/86 = 33.72% | 83/86 = 96.51% | 84/86 = 97.67% | 85/86 = 98.84% |
| — model rejected directly | 6 | 4 | 29 | 5 | 5 | 6 |
| — model picked a hidden card; system blocked it | 0 | 78 | 0 | 78 | 79 | 79 |
| — `intent_rule_no_model` · rule handed off to the Agent (not a decision-model reject) | 2 | 2 | 2 | 2 | 0 | 0 |
| **False pass** | 2/86 = 2.33% | 0/86 = 0% | 18/86 = 20.93% | 1/86 = 1.16% | 0/86 = 0% | 1/86 = 1.16% |
| No intent recalled (listed separately, not a reject) | 76 | 2 | 37 | 0 | 2 | 0 |

> All four baseline columns contain the same 2 `intent_rule_no_model` rows (`f5-ar-03`, `w3-ar-01`), both handed off to the Agent by the legacy rule. They never reached LayaChoice-v2, so they are **not** counted as decision-model rejects and are not part of the LayaChoice-v2 turn count; the baseline numerators each exclude them. They remain counted as correctly handled in End-to-End Accuracy through the Agent handoff, which is a different metric — the same rows are not subtracted twice. The rule-cleanup arms contain no `intent_rule_no_model` rows: those 2 requests enter the funnel and are rejected by Laya, so correctly rejected rises by 2 to 84/86 (0.88) and 85/86 (0.82), with the total unchanged and only the path composition changing.

After the change, almost all rejections come from the new mechanism: **78 rows in each of the two card-surface arms with recall enabled** (79 in each rule-cleanup arm) are "the model picked a hidden card and the system blocked it after the decision". The model
is not rejecting on its own; it first identifies the right capability, and the business layer
then refuses to execute it.

#### Is the model still accurate

This looks only at rows that reached the model where the true target is a tool.

| Item | 0.88 · using `intent_rule_no_model`<br>· filters out non-allowlist intents | 0.88 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.82 · using `intent_rule_no_model`<br>· filters out non-allowlist intents | 0.82 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.88 · deleting `intent_rule_no_model`<br>· allows recall of non-allowlist intents | 0.82 · deleting `intent_rule_no_model`<br>· allows recall of non-allowlist intents |
|---|---|---|---|---|---|---|
| Tool requests that reached the model | 328 | 329 | 363 | 363 | 373 | 407 |
| **Strict selection accuracy** | 300/328 = 91.46% | 302/329 = 91.79% | 330/363 = 90.91% | 331/363 = 91.18% | 344/373 = 92.23% | 374/407 = 91.89% |
| Accuracy among rows with a selection | 300/303 = 99.01% | 302/305 = 99.02% | 330/338 = 97.63% | 331/338 = 97.93% | 344/348 = 98.85% | 374/382 = 97.91% |
| Wrong card selected | 3 | 3 | 8 | 7 | 4 | 8 |
| Wrongly rejected | 23 | 24 | 25 | 25 | 25 | 25 |

Conclusion: **the model was not damaged.** The wider card surface increased the number of
turns reaching the model (336→411, 410→447), yet strict selection accuracy rose slightly and
conditional accuracy stayed between 97.6% and 99.0%.

### Where the data lives

| Content | Path |
|---|---|
| 0.88 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents | `logs/_cardable088_20261021/` |
| 0.82 · using `intent_rule_no_model`<br>· allows recall of non-allowlist intents | `logs/_cardable082_20261021/` |
| 0.88 · using `intent_rule_no_model`<br>· filters out non-allowlist intents, re-audited with the same definitions | `logs/_cardable_4way_pre088/` |
| 0.82 · using `intent_rule_no_model`<br>· filters out non-allowlist intents, re-audited with the same definitions | `logs/_cardable_4way_pre082/` |
| Four-arm comparison | `logs/_cardable_compare_4way/comparison.md` |

Each arm has a per-case audit, a conservation check (`all_ok=true`) and a run manifest.

### Open items

- The production recall threshold is **0.82**, and 0.82 is the shipped default
  (`packages/core/config.py`, overridable through `CHAT_FUNNEL_MIN_SCORE`).
- The rule-path removal and the post-selection allowlist check are part of the committed production chain.
- Full-chain business success accuracy cannot be computed: argument extraction is off and
  the executor is virtual. End-to-End Accuracy and End-to-End Tool Routing Accuracy measure
  capability routing only.
- Only two threshold points (0.88 and 0.82) were sampled; there was no full sweep.

> **Update:** the rule-cleanup re-run (commit `833651f`) is reported in the `intent_rule_no_model` deletion comparison below (English) and in the Chinese section of the same name at the end of this report. The four baseline columns above use the report's frozen audit definitions; the only correction applied to them is the exclusion of the requests handled directly by legacy intent rules (`intent_rule_no_model`) from the Decision Accuracy numerator and the decision-model reject numerator (see the notes on those tables).

# Before/After Deleting `intent_rule_no_model`

> `intent_rule_no_model` is the path where legacy intent rules handled a request directly, before the funnel and LayaChoice-v2, so the model took no part in the decision. It has two mutually exclusive outcomes: the rule selected and executed a tool directly (15 rows), or the rule matched no tool and handed the request to the Agent (31 rows), 46 rows in total.
>
> This chapter compares the two code states: **using `intent_rule_no_model`** (before the cleanup) and **deleting `intent_rule_no_model`** (after the cleanup). Every table header, row label and note uses this pair; “before/after” alone is not used as the identifier.

## 1. Scope and data sources

The only variable in this comparison is whether `intent_rule_no_model` exists:

- **Using `intent_rule_no_model`**: legacy intent rules can bypass the funnel and either execute a tool directly or hand the request to the Agent. This is the “0.88 / 0.82 · allows recall of non-allowlist intents” pair earlier in this report (commit `3855260` + the card-surface change; the per-case audit already exists).
- **Deleting `intent_rule_no_model`**: after commit `833651f` removes this path, every request follows the single production chain Matcher → Recall → LayaChoice-v2 → argument acquisition. This is the full 500-row run at each threshold reported here.

Both states ran under the same conditions:

| Item | Value |
|---|---|
| Dataset | `formal500_dataset_v2.jsonl`, 500 rows, sha256 `c35d7d3281e27efa31ea3d19d1a26025aee4e03ee0cc617916de9f92e850895c` (unchanged) |
| Model | production LayaChoice-v2, `http://127.0.0.1:18092` (real selector, with a transparent counter wrapped around it) |
| Environment | full production app + real Electron desktop client, one request at a time; VirtualExecutor; Agent skipped; argument extraction off |
| Completion | 0.88: 500/500, 0 UI failures, 0 request failures, 0 timeouts; 0.82: 500/500, same |
| Isolation checks | 0 online LLM calls, 0 real Agent executions, 0 real-execution violations, 0 SSE errors |
| Account | `admin` |

## 2. Key metrics comparison

| Metric | 0.88 · using `intent_rule_no_model` | 0.88 · deleting `intent_rule_no_model` | Change | 0.82 · using `intent_rule_no_model` | 0.82 · deleting `intent_rule_no_model` | Change |
|---|---|---|---|---|---|---|
| **End-to-End Tool Routing Accuracy (tool rows)** | 320/414 = 77.29% | 347/414 = 83.82% | **+27 rows / +6.53pp** | 349/414 = 84.30% | 377/414 = 91.06% | **+28 rows / +6.76pp** |
| **End-to-End Accuracy** | 406/500 = 81.20% | 433/500 = 86.60% | +27 rows / +5.40pp | 434/500 = 86.80% | 462/500 = 92.40% | +28 rows / +5.60pp |
| **Decision Accuracy** | 384/411 = 93.43% | 428/457 = 93.65% | **+0.22pp** | 414/447 = 92.62% | 459/493 = 93.10% | **+0.48pp** |
| **False Pass Rate** | 0/86 = 0% | 0/86 = 0% | 0 rows / 0.00pp | 1/86 = 1.16% | 1/86 = 1.16% | 0 rows / 0.00pp |
| Correctly rejected (of the 86 rows that should be rejected) | 82/86 = 95.35% | 84/86 = 97.67% | **+2 rows / +2.32pp** | 83/86 = 96.51% | 85/86 = 98.84% | **+2 rows / +2.33pp** |
| **R_HIT (Matcher lookup hit)** | 3 | 3 | 0 | 3 | 3 | 0 |
| Reached recall/funnel (reached_funnel, includes the 3 R_HIT rows) | 454 | 500 | +46 | 454 | 500 | +46 |
| **Entered LayaChoice-v2** | 411 | 457 | **+46** | 447 | 493 | **+46** |
| **NO_CANDIDATE** | 40 | 40 | 0 | 4 | 4 | 0 |
| LAYA_REJECT (all) | 28 | 30 | +2 | 30 | 31 | +1 |
| FUNNEL_ALLOWLIST_REJECT (all) | 78 | 79 | +1 | 78 | 79 | +1 |
| `intent_rule_no_model` · rule executed the tool directly | 15 | 0 | −15 | 15 | 0 | −15 |
| `intent_rule_no_model` · rule handed the request to the Agent | 31 | 0 | −31 | 31 | 0 | −31 |
| **Strict tool-selection accuracy** | 302/329 = 91.79% | 344/373 = 92.23% | +42 rows / +0.44pp | 331/363 = 91.18% | 374/407 = 91.89% | +43 rows / +0.71pp |
| Selection accuracy (only rows with a selection) | 302/305 = 99.02% | 344/348 = 98.85% | −0.17pp | 331/338 = 97.93% | 374/382 = 97.91% | −0.02pp |
| Wrong card selected | 3 | 4 | +1 | 7 | 8 | +1 |
| Wrongly rejected (tool request rejected by Laya) | 24 | 25 | +1 | 25 | 25 | 0 |

> The Decision Accuracy denominator contains only turns that entered LayaChoice-v2. Under `intent_rule_no_model`, 2 non-allowlist requests were handed to the Agent by the rule and never entered LayaChoice-v2, so they are in neither the numerator nor the denominator; after the path is deleted, those 2 requests enter the funnel and are counted by their actual decision.
>
> End-to-End Accuracy still counts those 2 as correct because they were correctly handed to the Agent. That is a different metric, so the same rows are not subtracted twice.

## 3. Full path distribution (500 rows, mutually exclusive)

| Outcome | Meaning | 0.88 · using `intent_rule_no_model` | 0.88 · deleting `intent_rule_no_model` | Change | 0.82 · using `intent_rule_no_model` | 0.82 · deleting `intent_rule_no_model` | Change |
|---|---|---|---|---|---|---|---|
| MATCHER_HIT (R_HIT) | Lookup hit; capability already fixed; does not go through Laya | 3 | 3 | 0 | 3 | 3 | 0 |
| `intent_rule_no_model` · rule executed the tool directly | Legacy rule matched and executed a tool directly; never entered the funnel | 15 | 0 | −15 | 15 | 0 | −15 |
| `intent_rule_no_model` · rule handed the request to the Agent | Legacy rule matched no tool and handed the request to the Agent; never entered the funnel | 31 | 0 | −31 | 31 | 0 | −31 |
| LAYA_SELECT | Entered Laya and selected a capability | 305 | 348 | +43 | 339 | 383 | +44 |
| LAYA_REJECT | Entered Laya; the model returned REJECT | 28 | 30 | +2 | 30 | 31 | +1 |
| FUNNEL_ALLOWLIST_REJECT | Selected a non-allowlist capability; the business layer converted it to REJECT | 78 | 79 | +1 | 78 | 79 | +1 |
| NO_CANDIDATE | No recalled candidates; did not enter Laya; handed to the Agent | 40 | 40 | 0 | 4 | 4 | 0 |
| LAYA_NO_DECISION | Entered Laya but timed out / no decision | 0 | 0 | 0 | 0 | 0 | 0 |
| OTHER | Other pre-funnel failure | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | | **500** | **500** | — | **500** | **500** | — |

> The two `intent_rule_no_model` rows are mutually exclusive outcomes of the same path, split by whether the rule selected a tool directly; they total 46 rows and are not double-counted. Each arm's rows sum to 500.
>
> The per-case audit files label the allowlist rejection outcome `WHITELIST_REJECT`, the historical name for the same decision. The production reason code is `FUNNEL_ALLOWLIST_REJECT`; the audit wrapper maps the current reason to the historical label so every arm is scored with one outcome vocabulary.

## 4. K-group comparison

| Number of intent candidates | 0.88 · using `intent_rule_no_model` | 0.88 · deleting `intent_rule_no_model` | Change | 0.82 · using `intent_rule_no_model` | 0.82 · deleting `intent_rule_no_model` | Change |
|---|---|---|---|---|---|---|
| K=0 (no recalled intents) | 2/40 = 5.00% | 2/40 = 5.00% | 0 / 0.00pp | 0/4 = 0% | 0/4 = 0% | 0 / 0.00pp |
| K=1 | 338/355 = 95.21% | 380/398 = 95.48% | +42 rows / +0.27pp | 201/206 = 97.57% | 227/232 = 97.84% | +26 rows / +0.27pp |
| K=2 | 42/52 = 80.77% | 44/55 = 80.00% | +2 rows / −0.77pp | 107/116 = 92.24% | 117/127 = 92.13% | +10 rows / −0.11pp |
| K=3 | 4/4 = 100% | 4/4 = 100% | 0 / 0.00pp | 106/125 = 84.80% | 115/134 = 85.82% | +9 rows / +1.02pp |
| Other paths (under `intent_rule_no_model`, includes the 46 rows on that path) | 20/49 = 40.82% | 3/3 = 100% | −46 rows | 20/49 = 40.82% | 3/3 = 100% | −46 rows |
| **Total** | **406/500 = 81.20%** | **433/500 = 86.60%** | +27 rows / +5.40pp | **434/500 = 86.80%** | **462/500 = 92.40%** | +28 rows / +5.60pp |
| **Total turns entering LayaChoice-v2** | **411** | **457** | **+46** | **447** | **493** | **+46** |

> Under `intent_rule_no_model`, “other paths” is 20/49: 15 rows where the rule executed the tool directly + 31 rows where the rule handed off to the Agent + 3 R_HIT rows, mutually exclusive. After the path is deleted, “other paths” contains only the 3 R_HIT rows.
>
> K=0 is unchanged (40 rows at 0.88, 4 at 0.82), which shows the cleanup did not change recall coverage; it changed which requests can reach recall.

## 5. Where the original `intent_rule_no_model` requests went

Under `intent_rule_no_model`, 46 requests bypassed the funnel: 15 were selected and executed by the rule directly, and 31 were handed to the Agent after the rule found no matching tool. After the path is deleted, all of them enter the production funnel.

The two states are joined case by case on `case_id`:

| Item | 0.88 | 0.82 |
|---|---|---|
| Correct answers among these 46 under `intent_rule_no_model` | 17 (15 rows where the rule executed the correct tool + 2 hidden requests correctly handed to the Agent) | 17 (same) |
| Correct answers among these 46 after deleting `intent_rule_no_model` | 44 | 45 |
| Net gain in correct rows | **+27** | **+28** |
| Paths after deletion | LAYA_SELECT 43 (42 correct + 1 wrong card), LAYA_REJECT 2 (1 wrong tool rejection + 1 correct hidden rejection), FUNNEL_ALLOWLIST_REJECT 1 (correct hidden block) | LAYA_SELECT 44 (all correct), LAYA_REJECT 1 (wrong tool rejection), FUNNEL_ALLOWLIST_REJECT 1 (correct hidden block) |

The net gain in correctly handled tool requests (+27 / +28) matches the key metrics table exactly; `entered LayaChoice-v2 +46` is also exactly these 46 rows.

## 6. Composition of correct rejects

| Item | 0.88 · using `intent_rule_no_model` | 0.88 · deleting `intent_rule_no_model` | Change | 0.82 · using `intent_rule_no_model` | 0.82 · deleting `intent_rule_no_model` | Change |
|---|---|---|---|---|---|---|
| **Correctly rejected (of 86)** | **82/86 = 95.35%** | **84/86 = 97.67%** | **+2 rows / +2.32pp** | **83/86 = 96.51%** | **85/86 = 98.84%** | **+2 rows / +2.33pp** |
| Listed separately: handed to the Agent directly by `intent_rule_no_model` (not in the correct-reject numerator above) | 2 | 0 | −2 | 2 | 0 | −2 |
| Rejected after entering LayaChoice-v2 | 82 | 84 | +2 | 83 | 85 | +2 |
| 　of which: the model returned REJECT | 4 | 5 | +1 | 5 | 6 | +1 |
| 　of which: selected a hidden card and was stopped by the business layer | 78 | 79 | +1 | 78 | 79 | +1 |

> Under `intent_rule_no_model`, 2 non-allowlist requests were handed to the Agent by the rule; they **never entered LayaChoice-v2 and are not decision-model REJECTs**. This table's correct-reject numerator uses the decision-model definition, so it excludes those 2 rows (82/86 at 0.88, 83/86 at 0.82).
>
> After the path is deleted, those 2 requests enter the funnel and Laya rejects them, so correctly rejected rises by 2 to 84/86 and 85/86; the denominator stays at 86, and only the path composition changes.

## 7. Reconciliation and definitions

1. **Total conservation**: all four arms satisfy `500 = 414 tool rows + 86 non-allowlist rows`; each arm's path distribution sums to 500; every arm's `reconciliation.json` has `all_ok = true`.
2. **Entered-Laya conservation**: `LAYA_SELECT + LAYA_REJECT + FUNNEL_ALLOWLIST_REJECT`
   - 0.88: under `intent_rule_no_model`, 305 + 28 + 78 = 411; after deletion, 348 + 30 + 79 = 457.
   - 0.82: under `intent_rule_no_model`, 339 + 30 + 78 = 447; after deletion, 383 + 31 + 79 = 493.
3. **86 non-allowlist rows conservation**: `correctly rejected + false pass + no recalled candidate (+ 2 handled directly under intent_rule_no_model)`
   - 0.88: under `intent_rule_no_model`, 82 + 0 + 2 + 2 = 86; after deletion, 84 + 0 + 2 = 86.
   - 0.82: under `intent_rule_no_model`, 83 + 1 + 0 + 2 = 86; after deletion, 85 + 1 + 0 = 86.
4. **Tool rows entering Laya conservation**: `strictly correct + wrong card + wrongly rejected + no decision`
   - 0.88: under `intent_rule_no_model`, 302 + 3 + 24 + 0 = 329; after deletion, 344 + 4 + 25 + 0 = 373.
   - 0.82: under `intent_rule_no_model`, 331 + 7 + 25 + 0 = 363; after deletion, 374 + 8 + 25 + 0 = 407.
5. **Decision Accuracy definition**: denominator = turns that entered LayaChoice-v2; numerator = turns with a correct decision (a tool row selected Gold, or a non-allowlist row was correctly rejected). Requests handled directly by `intent_rule_no_model` never entered LayaChoice-v2, so they are in neither.
   - Using `intent_rule_no_model`: 0.88 = 384/411 = 93.43%, 0.82 = 414/447 = 92.62%.
   - Deleting `intent_rule_no_model`: 0.88 = 428/457 = 93.65%, 0.82 = 459/493 = 93.10%.
   - Corresponding change: **+0.22pp / +0.48pp**.
6. **R_HIT is unaffected**: both thresholds stay at 3 rows, which shows the cleanup only deleted `intent_rule_no_model` and did not change the Matcher lookup.

## 8. Conclusions and non-comparable metrics

### Conclusions

1. **The key metrics improve substantially and consistently at both thresholds**: End-to-End Tool Routing Accuracy +6.53pp at 0.88 and +6.76pp at 0.82; End-to-End Accuracy +5.40pp / +5.60pp. The 46 deleted `intent_rule_no_model` requests now take part in Recall and LayaChoice-v2, with 27 / 28 more correctly handled tool requests at the two thresholds. The experiment does not isolate how much of this improvement is attributable to deleting the rule path versus other factors.
2. **False passes do not get worse**: 0.88 stays at 0%, 0.82 stays at 1.16% (the same real wrong selection).
3. **Laya's selection ability remains stable**: strict selection accuracy 91.79%→92.23% (+0.44pp) and 91.18%→91.89% (+0.71pp); conditional selection accuracy remains 97.9%–98.9%. With 46 more turns entering Laya, wrong-card selections rise by only 1 row each.
4. **NO_CANDIDATE is completely unchanged** (40 / 4), which shows the cleanup did not change recall coverage; it only changed who can reach recall.
5. **0.82 remains the better operating point**: after deleting `intent_rule_no_model`, End-to-End Tool Routing Accuracy is 91.06% vs 83.82%, and false pass is 1.16% vs 0%; the 38 K=0 tool rows at 0.88 are still a hard recall loss, not a selection-model loss.
6. **Decision Accuracy rises slightly (+0.22pp / +0.48pp)**: the denominator grows from 411/447 to 457/493, and the 46 added requests include harder rows, yet Decision Accuracy is not dragged down; at the same time correctly rejected rises by 2 at each threshold (the `intent_rule_no_model` path is gone and Laya decides), and Laya's strict selection accuracy rises as well.

### Metrics that cannot be compared or computed directly

- **Full-chain end-to-end (correct capability ∧ exact arguments ∧ execution validation passed)**: cannot be computed. Argument extraction is off (MODEL-owned slots stop at `BIND_MISSING`) and the executor is VirtualExecutor, so no real business result is produced. This report compares the capability-selection layer only.
- **Real business success rate, real execution latency, token consumption**: not comparable. Tool execution is virtual, online LLM calls are 0, and the Agent is skipped; these metrics require a separate real-execution experiment.
- **The 0.88 K=0 hard loss (38 tool rows)**: a recall-threshold issue, not introduced by deleting `intent_rule_no_model`; both states have 38 rows.
- **Threshold sweep**: only 0.82 and 0.88 were sampled; the results do not extrapolate to other thresholds.

## 9. Data locations and notes

| Content | Path |
|---|---|
| 0.88 · using `intent_rule_no_model` (pre-cleanup baseline) | `logs/_cardable088_20261021/` |
| 0.82 · using `intent_rule_no_model` (pre-cleanup baseline) | `logs/_cardable082_20261021/` |
| 0.88 · deleting `intent_rule_no_model` (this run) | `logs/_formal500_v2_ui_threshold088_post_rule_cleanup/` |
| 0.82 · deleting `intent_rule_no_model` (this run) | `logs/_formal500_v2_ui_threshold082_post_rule_cleanup/` |
| Metrics reports | `metrics_report.md` in the two run directories above |
| Per-case audit (reconciliation basis) | `case_level_audit.jsonl`, `reconciliation.json` in the two run directories above |
| Run identity manifests | `_run_manifest.json`, `_server_state_final.json` in the two run directories above |
| Real UI operation logs | `ui_ops.jsonl`, `cases.jsonl` in the two run directories above |
| Four-arm summary | `logs/_post_rule_cleanup_compare/comparison.md` |
| Four-column baseline comparison | `logs/_cardable_compare_4way/comparison.md` |
| The two rule-handled columns (recomputed under the new definition) | `logs/_cardable_4way_pre088/`, `logs/_cardable_4way_pre082/` |

> Note: `logs/_post_rule_cleanup_compare/comparison.md` also contains the four-arm summary. Its “non-tool requests entered Laya and correctly rejected” column uses the “entered Laya only” definition (82/86 under `intent_rule_no_model` at 0.88, 83/86 at 0.82), which matches the corrected decision-model reject definition in this report; every number in this report is based on the `case_level_audit.jsonl` files above.

---

# 聊天候选卡片改造 + 召回阈值实验报告

> **文档状态：** 第一、二部分保留原始四组候选卡片实验结果。`after rule cleanup` 列和删除 `intent_rule_no_model` 的对比结果，来自规则清理后重新运行的实验，属于新的测量结果，不会修改或重新计算历史基线。决策准确率（Decision Accuracy）仅统计实际进入 LayaChoice-v2 的请求，不包括由旧版意图规则直接处理的请求。

## 第一部分：核心指标和结论

### 四个核心指标

- **端到端准确率（End-to-End Accuracy）**：500 条测试请求中，最终决策或路由结果与标注的目标结果一致的请求占比，分母为全部 500 条请求。该指标反映整个意图处理流程在能力选择与路由层面的总体准确性，不验证业务任务是否实际执行成功。
- **端到端工具路由准确率（End-to-End Tool Routing Accuracy）**：414 条工具请求中，正确路由到标注目标工具的请求所占比例，分母为 414，而不是全部 500 条请求。该指标衡量意图处理流程中工具路由的整体正确性，同时受到候选集召回和工具选择的影响，不代表工具执行成功率或业务任务完成率。
- **决策准确率（Decision Accuracy）**：所有进入 LayaChoice-v2 判断的请求里，有多少拿到了正确的决策——
  要么放行给工具，要么拒绝。分母是进入 LayaChoice-v2 的总轮数。
- **错误放行率（False Pass Rate）**：本该交给 Agent 处理、不该直接当工具执行的 86 条请求里,
  有多少被错误地放行去执行工具。它衡量的是：有多少 non-allowlist 请求被错误放行了。

| 指标 | 0.88 · 删除 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.82 · 删除 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.88 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents | 0.88 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.82 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents | 0.82 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents |
|---|---|---|---|---|---|---|
| 端到端准确率 | **433/500 = 86.60%** | **462/500 = 92.40%** | 402/500 = 80.40% | 406/500 = 81.20% | 416/500 = 83.20% | 434/500 = 86.80% |
| 端到端工具路由准确率（受候选集召回与工具选择共同影响） | **347/414 = 83.82%** | **377/414 = 91.06%** | 318/414 = 76.81% | 320/414 = 77.29% | 348/414 = 84.06% | 349/414 = 84.30% |
| 决策准确率 | **428/457 = 93.65%** | **459/493 = 93.10%** | 306/336 = 91.07% | 384/411 = 93.43% | 359/410 = 87.56% | 414/447 = 92.62% |
| 错误放行率 | **0/86 = 0%** | **1/86 = 1.16%** | 2/86 = 2.33% | 0/86 = 0% | 18/86 = 20.93% | 1/86 = 1.16% |

### 最终方案

最终采用以下三项决定：

1. **召回阈值使用 0.82。**
2. **删除 `intent_rule_no_model`。**
3. **允许召回 non-allowlist intents，在模型决策后再进行 allowlist 校验，阻止不允许在 Chat 中直接执行的能力。**

选择该方案的依据：

- **端到端工具路由准确率最高**：0.82 且删除 `intent_rule_no_model` 后为 **91.06%**（0.88 为 83.82%；0.82 使用该路径为 84.30%；0.88 使用该路径为 77.29%）。
- **端到端准确率最高**：0.82 为 **92.40%**，0.88 为 86.60%。
- **召回覆盖更好**：0.82 的 K=0 只有 4 条，0.88 有 40 条；0.88 有 38 条工具请求因阈值过高没有召回任何卡片。
- **模型决策链路整体表现优于原规则路径方案**：本次 500 条实验显示，删除 `intent_rule_no_model` 后，模型决策链路在端到端准确率和端到端工具路由准确率上均优于保留规则路径的对应配置。这表明，在本次测试范围内，模型决策链路具备替代原规则路径的能力，有望减少对特定规则路径及其人工维护的依赖。

### 最终方案：500 条 query 的处理去向与分母口径

500 条 query 按实际处理路径分成三组，互斥且合计 500：

| 处理去向 | 条数 | 进入 LayaChoice-v2 | 计入决策准确率分母 | 端到端计分 |
|---|---|---|---|---|
| 进入 LayaChoice-v2 并产出决策 | 493 | 是 | 是 | 459 正确 / 34 错误 |
| `NO_CANDIDATE`：无召回，直接交 Agent | 4 | 否 | 否 | 0 正确 / 4 错误 |
| `MATCHER_HIT`：R_HIT 查表命中，能力已确定 | 3 | 否 | 否 | 3 正确 / 0 错误 |
| **合计** | **500** | — | — | **462 正确 / 38 错误** |

**分母口径。** 493 条进入 LayaChoice-v2 的 query 由三类实际决策组成：`LAYA_SELECT` 383 条、`LAYA_REJECT` 31 条、`FUNNEL_ALLOWLIST_REJECT` 79 条，合计 493。决策准确率的分子是其中决策正确的 459 条（374 + 6 + 79），因此 **459/493 = 93.10%**。剩下的 7 条既不进分子也不进分母：4 条 `NO_CANDIDATE` 没有召回任何候选、未进入 LayaChoice-v2；3 条 `MATCHER_HIT` 已由 Matcher 查表确定能力，不需要 Laya 再选。所以 **500 = 493 + 4 + 3**。

**工具路由分子口径。** 端到端工具路由准确率的分母是 414 条工具请求；只要最终路由结果是标注工具就计入分子，因此包含 3 条 `MATCHER_HIT`：0.82 的分子为 **377 = 374 条 Laya 正确选择 + 3**，0.88 为 **347 = 344 + 3**。四个基线列还包含 `intent_rule_no_model` 直接路由到标注工具的 15 条（例如 0.82 允许召回的基线为 349 = 331 + 15 + 3）。这 15 条没有进入 LayaChoice-v2，不计入决策准确率。

**两类未进入 Laya 的 query 如何计分。**

- `NO_CANDIDATE` 4 条全部是工具请求（目标能力均为 `cap-web-search`，Gold 在 allowlist 内），因阈值过高没有召回任何卡片，直接交给 Agent。它们在端到端口径中按失败计分，0/4 正确。
- `MATCHER_HIT` 3 条分别是 `vi-03`（`cap-vision`）、`pe-01`（`cap-pdf-extract-text`）、`f5-cf-07`（`cap-create-folder`），Matcher 命中后能力已确定，直接选中 Gold，在端到端口径中按正确计分，3/3 正确。

**两个准确率的分母不同。** 决策准确率只统计 LayaChoice-v2 实际做出决策的 493 条，**459/493 = 93.10%**；端到端准确率统计全部 500 条，把 3 条 `MATCHER_HIT` 正确命中计入、4 条 `NO_CANDIDATE` 工具请求按失败计入，**462/500 = 92.40%**。两者相差 7 条分母和 3 条分子，差异全部来自这 7 条未进入 LayaChoice-v2 的 query，不是漏算。

**0.88 方案的对应口径。** 同一套口径下，0.88 为 **500 = 457 + 40 + 3**：457 条进入 LayaChoice-v2，决策准确率 **428/457 = 93.65%**（`LAYA_SELECT` 344 + `LAYA_REJECT` 5 + `FUNNEL_ALLOWLIST_REJECT` 79）；40 条 `NO_CANDIDATE` 中 38 条是工具请求、全部按失败计分，另 2 条是 non-allowlist 请求（`cap-artifact`，`f5-ar-05` / `w3-ar-04`），正确交给 Agent、按正确计分；3 条 `MATCHER_HIT` 全部正确。端到端因此为 **433/500 = 86.60%**（428 + 3 + 2）。两个阈值都没有 `LAYA_NO_DECISION` 或 `OTHER` 行。

> 前两列是删除 `intent_rule_no_model` 并允许召回 non-allowlist intents 后的实测结果（commit `833651f`）；后四列是使用 `intent_rule_no_model` 的历史基线，按 non-allowlist intents 是否在召回前被过滤分为两组，沿用全报告同一份逐条审计口径。四个基线列的决策准确率没有把传统意图规则直接处理的请求（`intent_rule_no_model`，未进入 LayaChoice-v2）算进去；分母 336 / 411 / 410 / 447 本身就不含这 2 条 REJECT 判定和其他处理结果。

> **没有经过 LayaChoice-v2 的轮次：** 部分 non-allowlist capability 请求在进入决策模型之前就已 REJECT——由召回模型阈值本身筛选掉非意图 query，召回没有候选（`NO_CANDIDATE`），直接 REJECT。
>
> - 0.88 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents：召回阈值 REJECT 76
> - 0.88 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents：召回阈值 REJECT 2
> - 0.82 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents：召回阈值 REJECT 37
> - 0.82 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents：召回阈值 REJECT 0

### 结论

以前系统在 Laya 看到候选卡片之前，就把"聊天里不该直接执行"的工具卡藏了起来。用户明明要改文件，
模型面前却只剩"读文件"一张卡，只能选错。这次改成：**所有工具卡都摆出来，模型选完，
系统再检查它选的能不能在聊天里直接执行**；不能执行就拦下来，交给 Agent 处理。

1. **错误放行率显著下降。** 在本次测试中，0.88 配置从 2.33% 降至 0%，0.82 配置从 20.93% 降至 1.16%。结果表明，将候选卡片过滤调整为模型选择后的执行权限检查，有助于减少错误放行；但剩余错误仍可能来自模型选错等其他环节。
2. **卡片面改造本身主要降低误执行；删除 `intent_rule_no_model` 进一步抬高端到端工具路由准确率。**
   只改卡片面时，端到端工具路由准确率变化为 +0.48pp / +0.24pp；删除规则路径后，0.88 从 77.29% 升到
   83.82%（+6.53pp），0.82 从 84.30% 升到 91.06%（+6.76pp）。旧流程里"没有召回意图"的
   隐藏请求本来就被算作正确交给 Agent；新流程把它们送进模型再拒绝，结果相同，因此卡片面
   改造的收益主要体现在错误放行。
3. **Laya 的 top-1 选择稳定。** 卡片面改造后严格准确率为 91.79% / 91.18%；删除规则路径后为
   92.23% / 91.89%，没有退化。
4. **唯一漏网**：0.82 的 `w3-ef-16`。池子里已经有正确答案 `cap-edit-file`，
   模型却选了 `cap-read-file`。这是真的选错，不是被迫。
5. **0.82 的综合权衡更好。** 与 0.88 相比，0.82 用 1 条真实错误放行换来端到端工具路由准确率 +7.24pp、
   端到端准确率 +5.80pp；0.88 虽然错误放行为 0，但 K=0 的 38 条工具请求仍然拿不到任何卡片。

## 第二部分：其他详细统计

### 原来的问题

举个真实例子：用户说"把 tsconfig.json 里的 strict 从 false 改成 true"。

真正该用的是 `cap-edit-file`(改文件)。旧流程在召回之后、意图候选集召回之前，就把这张卡过滤掉了,
模型面前只剩 `cap-read-file`(读文件)。它只能选"读文件"，系统也就把请求当成"读文件"执行了。

这不是模型笨，是卡片被人为筛掉了。

改前 0.82 的 18 条误执行，全部是这个形态：卡片池里根本没有正确答案，只有一张 allowlist 内的卡。

### 改了什么

```
旧: 召回 → 过滤掉 non-allowlist intents → 剩下的卡给模型选 → 执行
新: 召回 → 意图候选集召回(全部工具，允许召回 non-allowlist intents) → 模型选 → 如果预测 tool 属于 non-allowlist → 拒绝
```

- **allowlist 在哪一步过滤**：改前是在召回之后立刻过滤，non-allowlist intents 在意图候选集召回前就被丢掉,
  模型根本看不到。改后允许召回 non-allowlist intents，全部送进模型，allowlist 校验挪到模型选完之后。
- **意图候选集召回阶段**：只要工具是启用且非 research，就进入意图候选集召回，不再看它是否在 chat allowlist 内。
- **模型选完之后**：如果选中的是不允许在聊天里直接执行的工具(改文件、生成产物、跑命令)，
  系统直接判定为拒绝，整轮交给 Agent；不执行，也不会偷偷改选第二张卡。
- **Agent 侧不受影响**：这三个工具仍然正常注册，页面、PC、Agent 照常能用。
  只是聊天面不再提前藏卡。

直观地说：以前是"不给你看，逼你选别的"；现在是"给你看，选到了我再拦"。

### 怎么测的

- 数据集：`formal500_dataset_v2`，共 500 条。
  - 414 条：应该用工具，且工具有聊天权限。
  - 86 条：真实目标是"不能直接执行"的隐藏能力(改文件 48 / 跑命令 20 / 生成产物 18)。
- 模型：线上同款 LayaChoice-v2(动态 K+1，输入 0~3 张能力卡 + 1 张 REJECT 卡)。
- 环境：完整生产 app + 真实 Electron 界面操作，但工具执行换成虚拟执行器,
  不真的改文件、不真的跑命令，也不调用线上大模型。
- 四组对照：改造前 0.88、改造后 0.88、改造前 0.82、改造后 0.82。
  每组都是完整 500 条，0 失败、0 超时。
- 参数抽取关闭，只测 candidate reranking → top-1 selection 这一层，不测"填参数"和"真执行"。

### 详细结果

#### 按"给了模型几张卡"分组

> **统计说明：** 每条请求首先经过召回阶段（Embedding + 召回阈值），生成候选工具集；随后由决策模型 LayaChoice-v2 从候选集中选择目标 Tool 或作出 REJECT 决策。下表统计整个流程最终正确选中目标 Tool 或正确判定 REJECT 的结果，并按提供给决策模型的候选卡片数量（K）分组。K=0 表示没有召回候选卡片，模型无法进行工具选择，只能交由 Agent 处理。

| 意图候选列表数量 | 0.88 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents | 0.88 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.82 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents | 0.82 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.88 · 删除 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.82 · 删除 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents |
|---|---|---|---|---|---|---|
| K=0(没有召回意图) | 76/115 = 66.09% | 2/40 = 5.00% | 37/41 = 90.24% | 0/4 = 0% | 2/40 = 5.00% | 0/4 = 0% |
| K=1 | 274/295 = 92.88% | 338/355 = 95.21% | 202/222 = 90.99% | 201/206 = 97.57% | 380/398 = 95.48% | 227/232 = 97.84% |
| K=2 | 28/37 = 75.68% | 42/52 = 80.77% | 95/111 = 85.59% | 107/116 = 92.24% | 44/55 = 80.00% | 117/127 = 92.13% |
| K=3 | 4/4 = 100% | 4/4 = 100% | 62/77 = 80.52% | 106/125 = 84.80% | 4/4 = 100% | 115/134 = 85.82% |
| 其他路径 | 20/49 = 40.82% | 20/49 = 40.82% | 20/49 = 40.82% | 20/49 = 40.82% | 3/3 = 100% | 3/3 = 100% |
| **总计** | **402/500 = 80.40%** | **406/500 = 81.20%** | **416/500 = 83.20%** | **434/500 = 86.80%** | **433/500 = 86.60%** | **462/500 = 92.40%** |
| **进入模型的总轮数** | **336** | **411** | **410** | **447** | **457** | **493** |

> 规则清理后，「其他路径」只剩 3 条 R_HIT（Matcher 查表命中）。旧流程中由传统意图规则直接处理（`intent_rule_no_model`）的 46 条请求已不存在，它们现在进入漏斗并计入各自真实的 K 分组。
>
> 基线中的「其他路径」20/49 由三部分组成：15 条由传统意图规则直接执行工具、31 条由传统意图规则交给 Agent（两者都属于 `intent_rule_no_model`，互斥，合计 46 条），以及 3 条 `R_HIT`。本表统计端到端正确性，其中 2 条 non-allowlist 的 `intent_rule_no_model` 请求按「正确交给 Agent」计为正确；它们不在 Laya 轮数内，也不在拒绝判定分子内，因此没有被重复扣减。

关键变化：**"没有召回意图"的轮次，0.88 从 115 降到 40，0.82 从 41 降到 4。**
这些轮次以前根本没机会进模型，现在能进模型并被正确拒绝。

> K=0 的百分比看起来下降，不代表变差：旧流程把这些"没有召回意图"的轮次直接交给 Agent,
> 在新口径里算正确；新流程把它们送进模型，模型识别后拒绝，结果同样正确，只是路径不同。

#### 拒绝判定

| 项目 | 0.88 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents | 0.88 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.82 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents | 0.82 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.88 · 删除 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.82 · 删除 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents |
|---|---|---|---|---|---|---|
| 本该拒绝的请求 | 86 | 86 | 86 | 86 | 86 | 86 |
| 正确拒绝 | 6/86 = 6.98% | 82/86 = 95.35% | 29/86 = 33.72% | 83/86 = 96.51% | 84/86 = 97.67% | 85/86 = 98.84% |
| 其中:模型直接拒绝 | 6 | 4 | 29 | 5 | 5 | 6 |
| 其中:模型选了隐藏卡,被系统拦下 | 0 | 78 | 0 | 78 | 79 | 79 |
| 其中:传统意图规则直接交给 Agent（`intent_rule_no_model`，不计入决策模型拒绝） | 2 | 2 | 2 | 2 | 0 | 0 |
| **误执行** | 2/86 = 2.33% | 0/86 = 0% | 18/86 = 20.93% | 1/86 = 1.16% | 0/86 = 0% | 1/86 = 1.16% |
| 没有召回意图(单列,不算拒绝) | 76 | 2 | 37 | 0 | 2 | 0 |

> 四个基线列都包含同样的 2 条由传统意图规则直接交给 Agent 的请求（`intent_rule_no_model`；`f5-ar-03`、`w3-ar-01`）。它们没有经过 LayaChoice-v2，因此**不计入决策模型拒绝，也不计入 LayaChoice-v2 处理轮数**；四个基线列的分子均已剔除这 2 条。它们仍按 Agent 交接路径计入端到端准确率，那是另一项指标，不做重复扣减。规则清理后的两臂不再存在该路径。

改造后的拦截几乎全部来自新机制：两个允许召回的卡片面基线臂各有 **78 条**，规则清理后的两臂各有 **79 条**，都是"模型选了隐藏卡，系统在决策后拦下"。
不是模型自己拒绝，而是模型先认对，业务层再拒绝执行。

#### 模型选对率诊断

这里只看"进了模型、且真实目标是工具"的请求。

| 项目 | 0.88 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents | 0.88 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.82 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents | 0.82 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.88 · 删除 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | 0.82 · 删除 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents |
|---|---|---|---|---|---|---|
| 进入模型的工具请求 | 328 | 329 | 363 | 363 | 373 | 407 |
| **严格选对率** | 300/328 = 91.46% | 302/329 = 91.79% | 330/363 = 90.91% | 331/363 = 91.18% | 344/373 = 92.23% | 374/407 = 91.89% |
| 选对率(只看有选择的) | 300/303 = 99.01% | 302/305 = 99.02% | 330/338 = 97.63% | 331/338 = 97.93% | 344/348 = 98.85% | 374/382 = 97.91% |
| 选错卡 | 3 | 3 | 8 | 7 | 4 | 8 |
| 误拒绝 | 23 | 24 | 25 | 25 | 25 | 25 |

结论：**模型没有被改坏。** 出卡面变宽后，进入模型的轮次多了(336→411、410→447),
但严格选对率反而略升，条件选对率稳定在 97.6%~99.0%。

### 数据在哪

| 内容 | 路径 |
|---|---|
| 0.88 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | `logs/_cardable088_20261021/` |
| 0.82 · 使用 `intent_rule_no_model`<br>· 允许召回 non-allowlist intents | `logs/_cardable082_20261021/` |
| 0.88 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents,按新口径重算 | `logs/_cardable_4way_pre088/` |
| 0.82 · 使用 `intent_rule_no_model`<br>· 过滤掉 non-allowlist intents,按新口径重算 | `logs/_cardable_4way_pre082/` |
| 四组对比表 | `logs/_cardable_compare_4way/comparison.md` |

每组都有逐条审计、总数守恒检查(`all_ok=true`)、运行清单。

### 还没做的

- 生产召回阈值已定为 **0.82**，并且已是配置默认值（`packages/core/config.py`，可用 `CHAT_FUNNEL_MIN_SCORE` 覆盖）。
- 规则路径删除与模型选择后的 allowlist 校验均已进入当前生产链路。
- 全链业务成功率算不出来：参数抽取关闭，执行器是虚拟的；端到端准确率与端到端工具路由准确率只衡量能力路由。
- 只采了 0.88 和 0.82 两个点，没做完整阈值扫描。

---

# `intent_rule_no_model` 删除前后对比

> `intent_rule_no_model` 指传统意图规则在进入漏斗和 LayaChoice-v2 之前就直接处理请求的路径，模型没有参与决策。它有两种互斥结果：规则直接选定并执行工具（15 条），以及规则没有匹配到工具、直接把请求交给 Agent（31 条），合计 46 条。
>
> 本章比较这条路径的两种代码状态：**使用 `intent_rule_no_model`**（清理前）与**删除 `intent_rule_no_model`**（清理后）。所有表格的列名、行名和注释都沿用这一对表述，不再用「清理前 / 清理后」单独指代。

## 1. 对比范围与数据来源

本次唯一变量是 `intent_rule_no_model` 是否存在：

- **使用 `intent_rule_no_model`**：传统意图规则可以绕过漏斗，直接执行工具或把请求交给 Agent。对应本文前面的「0.88 / 0.82 · 使用 `intent_rule_no_model` · 允许召回 non-allowlist intents」两列（commit `3855260` + 出卡面改动；已有逐条审计）。
- **删除 `intent_rule_no_model`**：commit `833651f` 删除该路径后，所有请求统一走「Matcher → Recall → LayaChoice-v2 → 参数获取」的单一生产链路。对应本次 0.88 / 0.82 两组完整 500 条实测。

两种状态的运行条件保持一致：

| 项目 | 值 |
|---|---|
| 数据集 | `formal500_dataset_v2.jsonl`，500 行，sha256 `c35d7d3281e27efa31ea3d19d1a26025aee4e03ee0cc617916de9f92e850895c`（未改动） |
| 模型 | 生产 LayaChoice-v2，`http://127.0.0.1:18092`（真实 selector，仅外层加透明计数） |
| 环境 | 完整生产 app + 真实 Electron 桌面端逐条提交；VirtualExecutor；Agent 跳过；参数抽取关闭 |
| 完成度 | 0.88：500/500，UI 失败 0、请求失败 0、超时 0；0.82：500/500，同 |
| 隔离校验 | 在线 LLM 调用 **0**、真实 Agent 执行 **0**、真实执行违规 **0**、SSE 错误 **0** |
| 账号 | `admin` |

## 2. 核心指标对比

| 指标 | 0.88 · 使用 `intent_rule_no_model` | 0.88 · 删除 `intent_rule_no_model` | 变化量 | 0.82 · 使用 `intent_rule_no_model` | 0.82 · 删除 `intent_rule_no_model` | 变化量 |
|---|---|---|---|---|---|---|
| **端到端工具路由准确率（End-to-End Tool Routing Accuracy）** | 320/414 = 77.29% | 347/414 = 83.82% | **+27 条 / +6.53pp** | 349/414 = 84.30% | 377/414 = 91.06% | **+28 条 / +6.76pp** |
| **端到端准确率** | 406/500 = 81.20% | 433/500 = 86.60% | +27 条 / +5.40pp | 434/500 = 86.80% | 462/500 = 92.40% | +28 条 / +5.60pp |
| **决策准确率（Decision Accuracy）** | 384/411 = 93.43% | 428/457 = 93.65% | **+0.22pp** | 414/447 = 92.62% | 459/493 = 93.10% | **+0.48pp** |
| **错误放行率（False Pass Rate）** | 0/86 = 0% | 0/86 = 0% | 0 条 / 0.00pp | 1/86 = 1.16% | 1/86 = 1.16% | 0 条 / 0.00pp |
| 正确拒绝（全部应拒绝 86 条） | 82/86 = 95.35% | 84/86 = 97.67% | **+2 条 / +2.32pp** | 83/86 = 96.51% | 85/86 = 98.84% | **+2 条 / +2.33pp** |
| **R_HIT（Matcher 查表命中）** | 3 | 3 | 0 | 3 | 3 | 0 |
| 到达漏斗/召回阶段（reached_funnel，含 3 条 R_HIT） | 454 | 500 | +46 | 454 | 500 | +46 |
| **进入 LayaChoice-v2** | 411 | 457 | **+46** | 447 | 493 | **+46** |
| **NO_CANDIDATE** | 40 | 40 | 0 | 4 | 4 | 0 |
| LAYA_REJECT（全量） | 28 | 30 | +2 | 30 | 31 | +1 |
| FUNNEL_ALLOWLIST_REJECT（全量） | 78 | 79 | +1 | 78 | 79 | +1 |
| `intent_rule_no_model` · 规则直接执行工具 | 15 | 0 | −15 | 15 | 0 | −15 |
| `intent_rule_no_model` · 规则交 Agent | 31 | 0 | −31 | 31 | 0 | −31 |
| **工具请求严格选对率** | 302/329 = 91.79% | 344/373 = 92.23% | +42 条 / +0.44pp | 331/363 = 91.18% | 374/407 = 91.89% | +43 条 / +0.71pp |
| 选对率（只看有选择的） | 302/305 = 99.02% | 344/348 = 98.85% | −0.17pp | 331/338 = 97.93% | 374/382 = 97.91% | −0.02pp |
| 选错卡 | 3 | 4 | +1 | 7 | 8 | +1 |
| 误拒绝（工具请求被 Laya 拒绝） | 24 | 25 | +1 | 25 | 25 | 0 |

> 决策准确率的分母只包含进入 LayaChoice-v2 的轮数。使用 `intent_rule_no_model` 时，2 条 non-allowlist 请求由规则直接交给 Agent，没有进入 LayaChoice-v2，因此既不在分子也不在分母；删除该路径后，这 2 条进入漏斗，按实际决策计入。
>
> 端到端准确率仍把这 2 条按「正确交给 Agent」计为正确。这是另一项指标，与决策准确率不重复扣减。

## 3. 完整路径分布（500 条互斥）

| 路径结果（outcome） | 含义 | 0.88 · 使用 `intent_rule_no_model` | 0.88 · 删除 `intent_rule_no_model` | 变化量 | 0.82 · 使用 `intent_rule_no_model` | 0.82 · 删除 `intent_rule_no_model` | 变化量 |
|---|---|---|---|---|---|---|---|
| MATCHER_HIT（R_HIT） | 查表命中，能力已确定，不经 Laya | 3 | 3 | 0 | 3 | 3 | 0 |
| `intent_rule_no_model` · 规则直接执行工具 | 传统意图规则命中并直接执行工具，未进漏斗 | 15 | 0 | −15 | 15 | 0 | −15 |
| `intent_rule_no_model` · 规则交 Agent | 传统意图规则未命中工具，直接交 Agent，未进漏斗 | 31 | 0 | −31 | 31 | 0 | −31 |
| LAYA_SELECT | 进入 Laya 并选中能力 | 305 | 348 | +43 | 339 | 383 | +44 |
| LAYA_REJECT | 进入 Laya，模型输出 REJECT | 28 | 30 | +2 | 30 | 31 | +1 |
| FUNNEL_ALLOWLIST_REJECT | 选中 non-allowlist 能力，业务层转 REJECT | 78 | 79 | +1 | 78 | 79 | +1 |
| NO_CANDIDATE | 无召回候选，未进 Laya，交 Agent | 40 | 40 | 0 | 4 | 4 | 0 |
| LAYA_NO_DECISION | 进入 Laya 但超时/无决策 | 0 | 0 | 0 | 0 | 0 | 0 |
| OTHER | 其他前置故障 | 0 | 0 | 0 | 0 | 0 | 0 |
| **合计** | | **500** | **500** | — | **500** | **500** | — |

> 两行 `intent_rule_no_model` 是同一条路径下的两种互斥结果，按工具是否被规则直接选定区分；合计 46 条，不重复计数。各行之和均为 500。
>
> 逐条审计文件把 allowlist 拒绝这个结果标为 `WHITELIST_REJECT`，这是同一判定的历史名称。生产代码的 reason 是 `FUNNEL_ALLOWLIST_REJECT`；审计包装层把当前 reason 归一化为历史标签，使所有臂使用同一套 outcome 口径。

## 4. K 分组对比

| 意图候选列表数量 | 0.88 · 使用 `intent_rule_no_model` | 0.88 · 删除 `intent_rule_no_model` | 变化量 | 0.82 · 使用 `intent_rule_no_model` | 0.82 · 删除 `intent_rule_no_model` | 变化量 |
|---|---|---|---|---|---|---|
| K=0（没有召回意图） | 2/40 = 5.00% | 2/40 = 5.00% | 0 / 0.00pp | 0/4 = 0% | 0/4 = 0% | 0 / 0.00pp |
| K=1 | 338/355 = 95.21% | 380/398 = 95.48% | +42 条 / +0.27pp | 201/206 = 97.57% | 227/232 = 97.84% | +26 条 / +0.27pp |
| K=2 | 42/52 = 80.77% | 44/55 = 80.00% | +2 条 / −0.77pp | 107/116 = 92.24% | 117/127 = 92.13% | +10 条 / −0.11pp |
| K=3 | 4/4 = 100% | 4/4 = 100% | 0 / 0.00pp | 106/125 = 84.80% | 115/134 = 85.82% | +9 条 / +1.02pp |
| 其他路径（使用 `intent_rule_no_model` 时含 46 条该路径请求） | 20/49 = 40.82% | 3/3 = 100% | −46 条 | 20/49 = 40.82% | 3/3 = 100% | −46 条 |
| **总计** | **406/500 = 81.20%** | **433/500 = 86.60%** | +27 条 / +5.40pp | **434/500 = 86.80%** | **462/500 = 92.40%** | +28 条 / +5.60pp |
| **进入 LayaChoice-v2 总轮数** | **411** | **457** | **+46** | **447** | **493** | **+46** |

> 「其他路径」在使用 `intent_rule_no_model` 时为 20/49：15 条规则直接执行工具 + 31 条规则交 Agent + 3 条 R_HIT，三者互斥。删除该路径后，「其他路径」只剩 3 条 R_HIT。
>
> K=0 的数量不变（0.88 为 40 条、0.82 为 4 条），说明本次清理没有改变召回覆盖，只改变了哪些请求能到达召回。

## 5. 原 `intent_rule_no_model` 请求的去向

使用 `intent_rule_no_model` 时，46 条请求绕过了漏斗：15 条由规则直接选定并执行工具，31 条由规则判定没有匹配工具后直接交给 Agent。删除该路径后，它们全部进入正式漏斗。

逐条按 `case_id` 连接两种状态的审计文件：

| 项目 | 0.88 | 0.82 |
|---|---|---|
| 使用 `intent_rule_no_model` 时这 46 条的正确答案 | 17（15 条规则直接执行了正确工具 + 2 条隐藏请求被正确交给 Agent） | 17（同） |
| 删除 `intent_rule_no_model` 后这 46 条的正确答案 | 44 | 45 |
| 净增正确数 | **+27** | **+28** |
| 删除后的路径 | LAYA_SELECT 43（42 对 + 1 选错）、LAYA_REJECT 2（1 工具误拒 + 1 隐藏正确拒绝）、FUNNEL_ALLOWLIST_REJECT 1（隐藏正确拦截） | LAYA_SELECT 44（全对）、LAYA_REJECT 1（工具误拒）、FUNNEL_ALLOWLIST_REJECT 1（隐藏正确拦截） |

两臂工具请求正确处理净增（+27 / +28）与上表完全一致；`进入 LayaChoice-v2 +46` 也正好等于这 46 条。

## 6. 正确拒绝路径构成

| 项目 | 0.88 · 使用 `intent_rule_no_model` | 0.88 · 删除 `intent_rule_no_model` | 变化量 | 0.82 · 使用 `intent_rule_no_model` | 0.82 · 删除 `intent_rule_no_model` | 变化量 |
|---|---|---|---|---|---|---|
| **正确拒绝总数（/86）** | **82/86 = 95.35%** | **84/86 = 97.67%** | **+2 条 / +2.32pp** | **83/86 = 96.51%** | **85/86 = 98.84%** | **+2 条 / +2.33pp** |
| 另列：由 `intent_rule_no_model` 直接交给 Agent（不计入上面的正确拒绝分子） | 2 | 0 | −2 | 2 | 0 | −2 |
| 进入 LayaChoice-v2 后被拒绝 | 82 | 84 | +2 | 83 | 85 | +2 |
| 　其中：模型直接 REJECT | 4 | 5 | +1 | 5 | 6 | +1 |
| 　其中：选中隐藏卡被业务层拦下 | 78 | 79 | +1 | 78 | 79 | +1 |

> 使用 `intent_rule_no_model` 时，2 条 non-allowlist 请求由规则直接交给 Agent，**没有进入 LayaChoice-v2，不属于决策模型的 REJECT**；本表的正确拒绝分子按决策模型口径统计，因此不含这 2 条（0.88 为 82/86、0.82 为 83/86）。
>
> 删除该路径后，这 2 条进入漏斗并由 Laya 判定为 REJECT，正确拒绝随之各净增 2 条,为 84/86、85/86，总数不变，变化只在路径构成。

## 7. 对账与统计口径说明

1. **总量守恒**：四个版本均为 `500 = 414 条工具行 + 86 条 non-allowlist 行`；路径分布各行之和均为 500；各臂 `reconciliation.json` 的 `all_ok = true`。
2. **进入 Laya 守恒**：`LAYA_SELECT + LAYA_REJECT + FUNNEL_ALLOWLIST_REJECT`
   - 0.88：使用 `intent_rule_no_model` 时 305 + 28 + 78 = 411；删除后 348 + 30 + 79 = 457。
   - 0.82：使用 `intent_rule_no_model` 时 339 + 30 + 78 = 447；删除后 383 + 31 + 79 = 493。
3. **non-allowlist 86 条守恒**：`正确拒绝 + 错误放行 + 没有召回意图（+ 使用 `intent_rule_no_model` 时直接处理的 2 条）`
   - 0.88：使用 `intent_rule_no_model` 时 82 + 0 + 2 + 2 = 86；删除后 84 + 0 + 2 = 86。
   - 0.82：使用 `intent_rule_no_model` 时 83 + 1 + 0 + 2 = 86；删除后 85 + 1 + 0 = 86。
4. **工具行进入 Laya 守恒**：`严格选对 + 选错卡 + 误拒绝 + 无决策`
   - 0.88：使用 `intent_rule_no_model` 时 302 + 3 + 24 + 0 = 329；删除后 344 + 4 + 25 + 0 = 373。
   - 0.82：使用 `intent_rule_no_model` 时 331 + 7 + 25 + 0 = 363；删除后 374 + 8 + 25 + 0 = 407。
5. **决策准确率口径**：分母 = 进入 LayaChoice-v2 的轮数；分子 = 其中决策正确的轮数（工具行选中 Gold，或 non-allowlist 行被正确拒绝）。由 `intent_rule_no_model` 直接处理的请求没有进入 LayaChoice-v2，因此既不进分母也不进分子。
   - 使用 `intent_rule_no_model`：0.88 = 384/411 = 93.43%，0.82 = 414/447 = 92.62%。
   - 删除 `intent_rule_no_model`：0.88 = 428/457 = 93.65%，0.82 = 459/493 = 93.10%。
   - 对应变化：**+0.22pp / +0.48pp**。
6. **R_HIT 未受影响**：两阈值均保持 3 条，说明本次清理只删除 `intent_rule_no_model`，没有改动 Matcher 查表。

## 8. 结论与不可比较项

### 结论

1. **核心指标显著提升，且两个阈值一致**：端到端工具路由准确率 0.88 +6.53pp、0.82 +6.76pp；端到端准确率 +5.40pp / +5.60pp。46 条被删除的 `intent_rule_no_model` 请求现在参与 Recall 与 LayaChoice-v2，两阈值下分别多正确处理 27 / 28 条工具请求。本次实验未单独分离删除规则路径对提升幅度的具体贡献。
2. **错误放行没有变差**：0.88 仍为 0%，0.82 仍为 1.16%（同一条真实选错）。
3. **Laya 的选择能力保持稳定**：严格选对率 91.79%→92.23%（+0.44pp）与 91.18%→91.89%（+0.71pp）；条件选对率仍在 97.9%–98.9%。进入 Laya 的轮次增加 46 条后，选错卡只各增加 1 条。
4. **NO_CANDIDATE 完全不变**（40 / 4），说明清理没有改变召回覆盖；它只改变了「谁能到达召回」。
5. **0.82 仍是更优工作点**：删除 `intent_rule_no_model` 后端到端工具路由准确率 91.06% 对 83.82%，错误放行 1.16% 对 0%；0.88 的 38 条 K=0 工具请求仍是召回的硬损失，而非选择模型的损失。
6. **决策准确率小幅上升（+0.22pp / +0.48pp）**：分母从 411/447 扩大到 457/493，新增的 46 条请求包含较难样本，决策准确率仍未被拖低；同时正确拒绝各净增 2 条（`intent_rule_no_model` 路径消失，改由 Laya 判定），Laya 的严格选对率也同步上升。

### 不能直接比较或无法计算的指标

- **全链端到端（选对能力 ∧ 参数完全匹配 ∧ 执行校验通过）**：无法计算。参数抽取关闭（MODEL-owned 槽位停在 `BIND_MISSING`），执行器为 VirtualExecutor，不产生真实业务结果。本报告只比较能力选择层。
- **真实业务成功率、真实执行延迟、Token 消耗**：不可比。本次工具执行为虚拟执行，在线 LLM 调用为 0，Agent 被跳过；这些指标需要另开真实执行实验。
- **0.88 的 K=0 硬损失（38 条工具行）**：属于召回阈值问题，不是删除 `intent_rule_no_model` 引入；两种状态均为 38 条。
- **阈值扫描**：仍只采 0.82 与 0.88 两个点，不能外推到其他阈值。

## 9. 数据位置与补充说明

| 内容 | 路径 |
|---|---|
| 0.88 · 使用 `intent_rule_no_model`（清理前基线） | `logs/_cardable088_20261021/` |
| 0.82 · 使用 `intent_rule_no_model`（清理前基线） | `logs/_cardable082_20261021/` |
| 0.88 · 删除 `intent_rule_no_model`（本次实测） | `logs/_formal500_v2_ui_threshold088_post_rule_cleanup/` |
| 0.82 · 删除 `intent_rule_no_model`（本次实测） | `logs/_formal500_v2_ui_threshold082_post_rule_cleanup/` |
| 指标报告 | 上述两个实测目录下 `metrics_report.md` |
| 逐条审计（对账依据） | 上述两个实测目录下 `case_level_audit.jsonl`、`reconciliation.json` |
| 运行身份清单 | 上述两个实测目录下 `_run_manifest.json`、`_server_state_final.json` |
| 真实 UI 操作记录 | 上述两个实测目录下 `ui_ops.jsonl`、`cases.jsonl` |
| 四臂汇总 | `logs/_post_rule_cleanup_compare/comparison.md` |
| 四列基线对比 | `logs/_cardable_compare_4way/comparison.md` |
| 使用规则处理的两列（按新口径重算） | `logs/_cardable_4way_pre088/`、`logs/_cardable_4way_pre082/` |

> 说明：`logs/_post_rule_cleanup_compare/comparison.md` 也包含四臂汇总，其「进入 Laya 的非工具请求正确拒绝」列采用「仅进入 Laya」口径（0.88 使用 `intent_rule_no_model` 时 82/86、0.82 使用 `intent_rule_no_model` 时 83/86），与本报告修正后的决策模型拒绝口径一致；本报告所有数字均以上述 `case_level_audit.jsonl` 为准。
