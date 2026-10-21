# Chat Candidate-Set Change + Recall Threshold Experiment

## Part 1: Headline Metrics and Conclusions

### Three headline metrics

- **Correct Pass Rate:** Of the 500 total requests, 414 were tool requests. This metric measures how many of these 414 requests were correctly routed to their target tools, reflecting the local intent pipeline’s effectiveness in handling tool requests. Correctly routed requests can be handled locally without additional LLM calls, reducing response latency and production token consumption.
- **Decision Accuracy** — Of all requests evaluated by LayaChoice-v2, how many received the
  correct decision, either being passed to a tool or rejected. The denominator is the total
  number of turns that reached LayaChoice-v2.
- **False Pass Rate** — Of the 86 requests that should be handled by the Agent rather than
  directly executed as tools, how many were incorrectly passed through for tool execution.
  This measures how often requests outside the intent allowlist were
  incorrectly allowed through.

| Metric | 0.88 · filters out non-allowlisted intents | 0.88 · recalls non-allowlisted intents | 0.82 · filters out non-allowlisted intents | 0.82 · recalls non-allowlisted intents |
|---|---:|---:|---:|---:|
| Correct Pass Rate (affected by both candidate-set recall and tool selection) | 318/414 = 76.81% | 320/414 = 77.29% | 348/414 = 84.06% | 349/414 = 84.30% |
| End-to-End Accuracy | 402/500 = 80.40% | 406/500 = 81.20% | 416/500 = 83.20% | 434/500 = 86.80% |
| Decision Accuracy | 308/336 = 91.67% | 386/411 = 93.92% | 361/410 = 88.05% | 416/447 = 93.06% |
| False Pass Rate | 2/86 = 2.33% | 0/86 = 0% | 18/86 = 20.93% | 1/86 = 1.16% |

> **Rounds that did not reach LayaChoice-v2:** Some requests for non-allowlisted capabilities were rejected before reaching the decision model. The retrieval model's score threshold itself filtered out non-intent queries, leaving no candidates (`NO_CANDIDATE`) and resulting in immediate rejection.
>
> - **0.88 ·** Non-allowlisted intents filtered out by the retrieval threshold: 76 REJECTs
> - **0.88 ·** Non-allowlisted intents retrieved but rejected by the retrieval threshold: 2 REJECTs
> - **0.82 ·** Non-allowlisted intents filtered out by the retrieval threshold: 37 REJECTs
> - **0.82 ·** Non-allowlisted intents retrieved but rejected by the retrieval threshold: 0 REJECTs

### Conclusions

The old pipeline hid the cards that chat is not allowed to execute *before* the model chose.
A user asking to edit a file saw only the "read file" card and had no choice but to pick it.
The new pipeline shows every tool card, lets the model choose, and then checks whether the
chosen capability may run in chat; if it may not, the turn is rejected and handed to the
Agent.

1. **False passes are nearly eliminated.** 0.88 goes from 2.33% to 0%, and 0.82 from 20.93%
   to 1.16%. The root cause was not the model; the old pipeline removed the correct answer
   before selection.
2. **Correct pass barely moves (+0.48pp / +0.24pp).** In the old pipeline the "no intent
   recalled" hidden requests were already counted as correctly handed to the Agent. The new
   pipeline sends them to the model and rejects them, with the same correct outcome. The
   benefit shows up in false pass, not correct pass.
3. **Top-1 selection is stable.** Strict accuracy goes 91.46%→91.79% and 90.91%→91.18%.
   There is no regression.
4. **The one remaining miss** is `w3-ef-16` at 0.82. Its pool already contained the correct
   `cap-edit-file`, but the model picked `cap-read-file`. This is a genuine selection error,
   not a forced choice.
5. **Decision: 0.82.** Recalling intents that are not on the intent allowlist and letting the
   post-decision allowlist check reject them is the best option across the metrics: correct
   pass is 84.30% versus 77.29% at 0.88, the false-pass rate is 1.16% (one genuine selection
   miss), and strict selection accuracy stays at 91.18%. The 0.88 arm is more conservative
   but leaves 75 more turns without any recalled intent.

## Part 2: Detailed Statistics

### What was wrong

A real example: the user says "change `strict` in tsconfig.json from false to true".

The right capability is `cap-edit-file`. The old pipeline dropped that card after recall
and before the intent candidate set recall, so the model saw only `cap-read-file`. It picked
"read file", and the system executed the request as a read.

The model was not the problem. The correct card had been filtered out.

All 18 false passes at 0.82 before the change have this shape: the pool contained no correct
answer, only one allowlisted card.

### What changed

```
Old: recall → filter out intents not on the intent allowlist → intent candidate set recall (remaining tools) → model chooses → execute
New: recall → intent candidate set recall (all tools, intents not on the intent allowlist kept) → model chooses → if the predicted tool is not on the intent allowlist → reject
```

- **Where the intent allowlist applies**: before the change it was checked right after recall,
  so an intent not on the intent allowlist was dropped before the intent candidate set recall.
  After the change recall keeps those intents, every candidate reaches the model, and the
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

| Intent candidate list size | 0.88 · filters out non-allowlisted intents | 0.88 · recalls non-allowlisted intents | 0.82 · filters out non-allowlisted intents | 0.82 · recalls non-allowlisted intents |
|---|---:|---:|---:|---:|
| K=0 (no intent recalled) | 76/115 = 66.09% | 2/40 = 5.00% | 37/41 = 90.24% | 0/4 = 0% |
| K=1 | 274/295 = 92.88% | 338/355 = 95.21% | 202/222 = 90.99% | 201/206 = 97.57% |
| K=2 | 28/37 = 75.68% | 42/52 = 80.77% | 95/111 = 85.59% | 107/116 = 92.24% |
| K=3 | 4/4 = 100% | 4/4 = 100% | 62/77 = 80.52% | 106/125 = 84.80% |
| Other paths | 20/49 = 40.82% | 20/49 = 40.82% | 20/49 = 40.82% | 20/49 = 40.82% |
| **Total** | **402/500 = 80.40%** | **406/500 = 81.20%** | **416/500 = 83.20%** | **434/500 = 86.80%** |
| **Turns that reached the model** | **336** | **411** | **410** | **447** |

The key change: **turns with no intent recalled dropped from 115 to 40 at 0.88 and from 41 to 4 at
0.82.** Those turns never had a chance to reach the model before; now they do, and they are
correctly rejected.

> The K=0 percentage looks lower, but that does not mean regression. The old pipeline handed
> these "no intent recalled" turns straight to the Agent, which the new scoring counts as correct. The
> new pipeline sends them to the model, which recognises them and rejects them. The outcome
> is equally correct; only the path differs.

#### Reject decisions

| Item | 0.88 · filters out non-allowlisted intents | 0.88 · recalls non-allowlisted intents | 0.82 · filters out non-allowlisted intents | 0.82 · recalls non-allowlisted intents |
|---|---:|---:|---:|---:|
| Requests that should be rejected | 86 | 86 | 86 | 86 |
| Correctly rejected | 8/86 = 9.30% | 84/86 = 97.67% | 31/86 = 36.05% | 85/86 = 98.84% |
| — model rejected directly | 8 | 6 | 31 | 7 |
| — model picked a hidden card; system blocked it | 0 | 78 | 0 | 78 |
| **False pass** | 2/86 = 2.33% | 0/86 = 0% | 18/86 = 20.93% | 1/86 = 1.16% |
| No intent recalled (listed separately, not a reject) | 76 | 2 | 37 | 0 |

After the change, almost all rejections come from the new mechanism: **78 rows in each arm**
are "the model picked a hidden card and the system blocked it after the decision". The model
is not rejecting on its own; it first identifies the right capability, and the business layer
then refuses to execute it.

#### Is the model still accurate

This looks only at rows that reached the model where the true target is a tool.

| Item | 0.88 · filters out non-allowlisted intents | 0.88 · recalls non-allowlisted intents | 0.82 · filters out non-allowlisted intents | 0.82 · recalls non-allowlisted intents |
|---|---:|---:|---:|---:|
| Tool requests that reached the model | 328 | 329 | 363 | 363 |
| **Strict selection accuracy** | 300/328 = 91.46% | 302/329 = 91.79% | 330/363 = 90.91% | 331/363 = 91.18% |
| Accuracy among rows with a selection | 300/303 = 99.01% | 302/305 = 99.02% | 330/338 = 97.63% | 331/338 = 97.93% |
| Wrong card selected | 3 | 3 | 8 | 7 |
| Wrongly rejected | 23 | 24 | 25 | 25 |

Conclusion: **the model was not damaged.** The wider card surface increased the number of
turns reaching the model (336→411, 410→447), yet strict selection accuracy rose slightly and
conditional accuracy stayed between 97.6% and 99.0%.

### Where the data lives

| Content | Path |
|---|---|
| 0.88 · recalls non-allowlisted intents | `logs/_cardable088_20261021/` |
| 0.82 · recalls non-allowlisted intents | `logs/_cardable082_20261021/` |
| 0.88 · filters out non-allowlisted intents, re-audited with the same definitions | `logs/_cardable_4way_pre088/` |
| 0.82 · filters out non-allowlisted intents, re-audited with the same definitions | `logs/_cardable_4way_pre082/` |
| Four-arm comparison | `logs/_cardable_compare_4way/comparison.md` |

Each arm has a per-case audit, a conservation check (`all_ok=true`) and a run manifest.

### Open items

- The production recall threshold is decided as **0.82**; the config default is still 0.88
  and has not been switched yet.
- The three production-file changes are not committed.
- End-to-end accuracy cannot be computed: argument extraction is off and the executor is
  virtual.
- Only two threshold points (0.88 and 0.82) were sampled; there was no full sweep.

---

# 聊天候选卡片改造 + 召回阈值实验报告

## 第一部分：主要指标和结论

### 三个主指标

- **正确放行率（Correct Pass Rate）**：500 条请求中，414 条属于工具请求。衡量这 414 条工具请求中，有多少被正确路由到目标工具，反映本地意图 Pipeline 对工具请求的处理能力。正确路由的请求可由本地完成处理，无需额外调用 LLM，从而降低响应延迟和线上 Token 消耗。
- **决策准确率（Decision Accuracy）**：所有进入 LayaChoice-v2 判断的请求里，有多少拿到了正确的决策——
  要么放行给工具，要么拒绝。分母是进入 LayaChoice-v2 的总轮数。
- **错误放行率（False Pass Rate）**：本该交给 Agent 处理、不该直接当工具执行的 86 条请求里,
  有多少被错误地放行去执行工具。它衡量的是：有多少不在意图漏斗执行白名单里的请求被错误放行了。

| 指标 | 0.88 · 过滤掉不在白名单的 intent | 0.88 · 召回不在白名单的 intent | 0.82 · 过滤掉不在白名单的 intent | 0.82 · 召回不在白名单的 intent |
|---|---:|---:|---:|---:|
| 正确放行率（受候选集召回与工具选择共同影响） | 318/414 = 76.81% | 320/414 = 77.29% | 348/414 = 84.06% | 349/414 = 84.30% |
| 端到端准确率 | 402/500 = 80.40% | 406/500 = 81.20% | 416/500 = 83.20% | 434/500 = 86.80% |
| 决策准确率 | 308/336 = 91.67% | 386/411 = 93.92% | 361/410 = 88.05% | 416/447 = 93.06% |
| 错误放行率 | 2/86 = 2.33% | 0/86 = 0% | 18/86 = 20.93% | 1/86 = 1.16% |

> **没有经过 LayaChoice-v2 的轮次：** 部分 non-allowlisted capability 请求在进入决策模型之前就已 REJECT——由召回模型阈值本身筛选掉非意图 query，召回没有候选（`NO_CANDIDATE`），直接 REJECT。
>
> - 0.88 · 过滤掉不在白名单的 intent：召回阈值 REJECT 76
> - 0.88 · 召回不在白名单的 intent：召回阈值 REJECT 2
> - 0.82 · 过滤掉不在白名单的 intent：召回阈值 REJECT 37
> - 0.82 · 召回不在白名单的 intent：召回阈值 REJECT 0

### 结论

以前系统在 Laya 看到候选卡片之前，就把"聊天里不该直接执行"的工具卡藏了起来。用户明明要改文件,
模型面前却只剩"读文件"一张卡，只能选错。这次改成：**所有工具卡都摆出来，模型选完,
系统再检查它选的能不能在聊天里直接执行**；不能执行就拦下来，交给 Agent 处理。

1. **误执行几乎清零。** 0.88 从 2.33% 降到 0%，0.82 从 20.93% 降到 1.16%。
   根因不是模型，而是旧流程把正确答案提前筛掉了。
2. **正确放行几乎不动(+0.48pp / +0.24pp)。** 旧流程里"没有召回意图"的隐藏请求本来就被算作
   正确交给 Agent；新流程把它们送进模型再拒绝，结果相同。所以收益主要体现在错误放行。
3. **Laya 的 top-1 选择稳定。** 严格准确率 91.46%→91.79%、90.91%→91.18%，没有退化。
4. **唯一漏网**：0.82 的 `w3-ef-16`。池子里已经有正确答案 `cap-edit-file`,
   模型却选了 `cap-read-file`。这是真的选错，不是被迫。
5. **结论：选 0.82。** 让不在白名单里的意图也参与召回，再由决策后的白名单校验拦截,
   这组方案综合指标最好：正确放行为 84.30%(0.88 为 77.29%)，错误放行率 1.16%
   (只有一条真实选错)，严格选对率仍有 91.18%。0.88 更保守，但多出 75 轮
   「没有召回意图」。

## 第二部分：其他详细统计

### 原来的问题

举个真实例子：用户说"把 tsconfig.json 里的 strict 从 false 改成 true"。

真正该用的是 `cap-edit-file`(改文件)。旧流程在召回之后、意图候选集召回之前，就把这张卡过滤掉了,
模型面前只剩 `cap-read-file`(读文件)。它只能选"读文件"，系统也就把请求当成"读文件"执行了。

这不是模型笨，是卡片被人为筛掉了。

改前 0.82 的 18 条误执行，全部是这个形态：卡片池里根本没有正确答案，只有一张白名单卡。

### 改了什么

```
旧: 召回 → 过滤掉不在白名单的 intent → 剩下的卡给模型选 → 执行
新: 召回 → 意图候选集召回(全部工具，不在白名单的 intent 也保留) → 模型选 → 如果预测 tool 不在白名单 → 拒绝
```

- **白名单在哪一步过滤**：改前是在召回之后立刻过滤，不在白名单的 intent 意图候选集召回前就被丢掉,
  模型根本看不到。改后召回保留这些 intent，全部送进模型，白名单校验挪到模型选完之后。
- **意图候选集召回阶段**：只要工具是启用且非 research，就进入意图候选集召回，不再看它是否在聊天白名单里。
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

| 意图候选列表数量 | 0.88 · 过滤掉不在白名单的 intent | 0.88 · 召回不在白名单的 intent | 0.82 · 过滤掉不在白名单的 intent | 0.82 · 召回不在白名单的 intent |
|---|---:|---:|---:|---:|
| K=0(没有召回意图) | 76/115 = 66.09% | 2/40 = 5.00% | 37/41 = 90.24% | 0/4 = 0% |
| K=1 | 274/295 = 92.88% | 338/355 = 95.21% | 202/222 = 90.99% | 201/206 = 97.57% |
| K=2 | 28/37 = 75.68% | 42/52 = 80.77% | 95/111 = 85.59% | 107/116 = 92.24% |
| K=3 | 4/4 = 100% | 4/4 = 100% | 62/77 = 80.52% | 106/125 = 84.80% |
| 其他路径 | 20/49 = 40.82% | 20/49 = 40.82% | 20/49 = 40.82% | 20/49 = 40.82% |
| **总计** | **402/500 = 80.40%** | **406/500 = 81.20%** | **416/500 = 83.20%** | **434/500 = 86.80%** |
| **进入模型的总轮数** | **336** | **411** | **410** | **447** |

关键变化：**"没有召回意图"的轮次，0.88 从 115 降到 40，0.82 从 41 降到 4。**
这些轮次以前根本没机会进模型，现在能进模型并被正确拒绝。

> K=0 的百分比看起来下降，不代表变差：旧流程把这些"没有召回意图"的轮次直接交给 Agent,
> 在新口径里算正确；新流程把它们送进模型，模型识别后拒绝，结果同样正确，只是路径不同。

#### 拒绝判定

| 项目 | 0.88 · 过滤掉不在白名单的 intent | 0.88 · 召回不在白名单的 intent | 0.82 · 过滤掉不在白名单的 intent | 0.82 · 召回不在白名单的 intent |
|---|---:|---:|---:|---:|
| 本该拒绝的请求 | 86 | 86 | 86 | 86 |
| 正确拒绝 | 8/86 = 9.30% | 84/86 = 97.67% | 31/86 = 36.05% | 85/86 = 98.84% |
| 其中:模型直接拒绝 | 8 | 6 | 31 | 7 |
| 其中:模型选了隐藏卡,被系统拦下 | 0 | 78 | 0 | 78 |
| **误执行** | 2/86 = 2.33% | 0/86 = 0% | 18/86 = 20.93% | 1/86 = 1.16% |
| 没有召回意图(单列,不算拒绝) | 76 | 2 | 37 | 0 |

改造后的拦截几乎全部来自新机制：两臂各有 **78 条**是"模型选了隐藏卡，系统在决策后拦下"。
不是模型自己拒绝，而是模型先认对，业务层再拒绝执行。

#### 模型选对率诊断

这里只看"进了模型、且真实目标是工具"的请求。

| 项目 | 0.88 · 过滤掉不在白名单的 intent | 0.88 · 召回不在白名单的 intent | 0.82 · 过滤掉不在白名单的 intent | 0.82 · 召回不在白名单的 intent |
|---|---:|---:|---:|---:|
| 进入模型的工具请求 | 328 | 329 | 363 | 363 |
| **严格选对率** | 300/328 = 91.46% | 302/329 = 91.79% | 330/363 = 90.91% | 331/363 = 91.18% |
| 选对率(只看有选择的) | 300/303 = 99.01% | 302/305 = 99.02% | 330/338 = 97.63% | 331/338 = 97.93% |
| 选错卡 | 3 | 3 | 8 | 7 |
| 误拒绝 | 23 | 24 | 25 | 25 |

结论：**模型没有被改坏。** 出卡面变宽后，进入模型的轮次多了(336→411、410→447),
但严格选对率反而略升，条件选对率稳定在 97.6%~99.0%。

### 数据在哪

| 内容 | 路径 |
|---|---|
| 0.88 · 召回不在白名单的 intent | `logs/_cardable088_20261021/` |
| 0.82 · 召回不在白名单的 intent | `logs/_cardable082_20261021/` |
| 0.88 · 过滤掉不在白名单的 intent,按新口径重算 | `logs/_cardable_4way_pre088/` |
| 0.82 · 过滤掉不在白名单的 intent,按新口径重算 | `logs/_cardable_4way_pre082/` |
| 四组对比表 | `logs/_cardable_compare_4way/comparison.md` |

每组都有逐条审计、总数守恒检查(`all_ok=true`)、运行清单。

### 还没做的

- 生产召回阈值已定为 **0.82**；配置默认值仍是 0.88，尚未切换。
- 三个生产文件的改动还没提交。
- 全链准确率算不出来：参数抽取关闭，执行器是虚拟的。
- 只采了 0.88 和 0.82 两个点，没做完整阈值扫描。
