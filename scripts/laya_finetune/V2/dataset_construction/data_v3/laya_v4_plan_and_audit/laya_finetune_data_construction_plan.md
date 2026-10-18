# Laya Tool Router V4 微调数据构造 Plan

## 0. 任务目标与保护边界

以 `scripts/laya_finetune/data_v3/laya_tool_selection_v4.csv`（当前已提供副本 `laya_tool_selection_v4.csv`，共 **49,241** 条）为输入，利用系统当前 SQL 查询召回链路构造训练集、测试集和验证集。候选卡片数量动态变化，按 capability 检索证据生成最多 3 张 Action capability 卡片，并始终考虑 REJECT。

所有操作必须先审计、再构造、后复核。**不执行模型训练；不覆盖输入 CSV；不修改 V1/V2/V3 既有数据、训练脚本或配置。** 新输出统一放在 `scripts/laya_finetune/data_v4/`，若目录已存在，先检查并避免覆盖任何已有文件。

## 1. Phase 0 — 只读检查与 Gate

1. 只读核对输入路径、行数、SHA256、schema 与本文件的统计：49,241 行、11 列。
2. 阅读实际训练入口、Dataset/DataLoader、序列化代码、capability card 渲染代码和线上/本地 SQL recall 实现；不能只根据旧文档猜 schema。确认模型实际使用的输入字段、卡片结构、label 编码以及动态 options 是否被训练器支持。
3. 阅读正式 capability registry / tool registry / SQL 表 schema / 阈值配置，建立 `capability_id ↔ tool_name ↔ card_id` 的唯一映射。
4. **阻塞 Gate：确认 `cap-research` 的合法性。** V4 有 852 条 `cap-research` / `tool_name=research` 样本。如果它不是模型候选卡片中的有效 capability，禁止默默重映射成 reject 或丢弃；输出 `BLOCKED` 说明及这些行的明细，先保留输入不变。
5. 确认 SQL 数据源是否包含本次 V4 输入行，避免候选召回直接命中验证/测试样本造成泄漏。若无法查清 SQL schema 或 recall 实现，停止正式数据集构造，只交付诊断材料，不能模拟一个 SQL 召回结果冒充真实数据。

## 2. Phase 1 — 标签和源数据预审

- 校验 `query` 非空、`is_action` 布尔值有效、`capability_id` 在 registry 中存在、`tool_name` 与注册表一致。
- `is_action=false` / `capability_id=reject` 为 No-Action/REJECT；`tool_name` 应为空，参数保持 `{}` 或按权威 schema 的规则规范化到派生输出。
- Action 样本的 `capability_id` 必须是合法候选能力。
- 原始 `language` 为空时，不要直接当作独立语言或覆盖源字段。新增派生 `split_language`：结合 CJK/Latin 特征、代码/占位符、必要的人工复核来分成 `zh`、`en`、`mixed`、`other/undetermined`。只用于分层切分，原始 `language` 保留。
- 执行 NFKC + 空白折叠 + casefold 的 query 规范化；重复 query/语义近重复或来源模板族应尽量分到同一 split，降低 train/test leakage。
- 使用现有审计清单优先复核 `cap-research`、模板拼接句、重复中文表达和英文夹杂中文占位词的样本。不得把启发式候选一律判错，也不得静默自动改标签。

## 3. Phase 2 — 数据集切分

### 3.1 分层维度

以 `capability_id × split_language` 为基本分层单元。对有效 Action capability、REJECT 分别处理，确保 REJECT 的 zh/en 数据也进入分层统计。对 `mixed/other/undetermined` 独立分层并报告，不要把它们强行归入中文或英文。

### 3.2 比例

每个 capability 的中文、英文样本分别按以下比例切分：

- Train：75%
- Test：20%
- Validation：5%

按整数采用 largest-remainder/确定性取整，保证每组各 split 数量之和与该组源样本数完全相等。固定随机种子 `20261019`，报告每一组的目标数与实际数。

### 3.3 泄漏控制

- 先切分源 query，再构造训练选项；不能先对全量数据召回后再随机切分。
- 同一规范化 query 必须只属于一个 split。识别到的模板族或近重复组，尽量组级切分，避免近乎相同的 query 跨 train/test/validation。
- 如果 SQL recall 表本身就是由这批待切分 query 建立：候选索引只能使用 Train 的已标注 query/相似 query；训练样本召回时排除自身和自身规范化重复项；Test/Validation query 不得进入候选索引。
- 如果 SQL recall 表是独立的生产检索库：验证其不包含 test/validation 样本本身或其泄漏副本，并记录表名、版本/快照、查询时间、SQL 参数。

## 4. Phase 3 — 动态候选卡片构造

### 4.1 真实 SQL 召回

对每条样本 query 调用当前项目真实 SQL recall 接口，召回已标注 query 与其相似 query。不要新写一个“近似实现”替代生产召回逻辑。

1. 记录每条召回的原始 query ID、query 文本或受控摘要、`capability_id`、相似度分数、来源类型（标注 query / similar query）。
2. 应用项目现有召回阈值，只保留 `score >= threshold` 的记录。阈值必须取自当前项目配置/真实实现，并在报告中记录具体值、比较方向和配置位置。
3. 按 `capability_id` 去重聚合，得到每个 capability 的代表分数。优先沿用生产 SQL 的聚合方法；若线上没有定义，采用该 capability 的最高相似度 `MAX(score)`，不要因某 capability 仅仅拥有更多 similar queries 就按条数加分。
4. 按聚合分数降序排序，只保留 Top-3 个不同的 Action capability。并列时使用稳定、可复现的 tie-break（如 `capability_id` 字典序），不要依赖 SQL 不稳定返回顺序。
5. 选项卡由 Top-K capability cards 和一个 REJECT card 组成，其中 `K ∈ {0, 1, 2, 3}`。

### 4.2 K 与卡片规则

| K（Action capability 卡数量） | 选项卡 | 处理规则 |
|---:|---|---|
| 0 | 仅 REJECT | 直接标记/路由为 REJECT；是否跳过模型推理必须与线上行为一致，训练格式仍须可复现 |
| 1 | 1 张 capability + REJECT | 共 2 张卡 |
| 2 | 2 张 capability + REJECT | 共 3 张卡 |
| 3 | 3 张 capability + REJECT | 共 4 张卡 |

REJECT 必须始终出现在 options 中。不得为了凑够 3 张 Action 卡而随机补候选，也不得补入低于阈值的候选。

## 5. Phase 4 — 训练标签生成规则

对每条输入样本定义 `gold_capability`：

1. 若源样本是 No-Action（`is_action=false` 或 `capability_id=reject`），最终 label 为 `reject`，无条件覆盖召回结果。
2. 若样本是 Action，且 `gold_capability` 在最终 Top-K capability 卡片中，则 label 为该 `capability_id`。
3. 若 `gold_capability` 没有被阈值过滤后召回、没有进入 capability 聚合结果、或被 Top-3 截断，因此不在实际 options 中，则 label 为 `reject`。不可生成一个不在 options 里的 gold label。
4. 对每个被标为 REJECT 的 Action query 记录原因：`no_candidate_after_threshold`、`gold_capability_not_recalled`、`gold_capability_below_top3`、`no_action_gold` 或 `registry_blocked`。这些原因必须区分，不能统称为“无效样本”。
5. `cap-research` 等任何不在 registry 中的 label 不得直接转为 REJECT；应触发 Phase 0 Gate。

## 6. Phase 5 — 训练格式

- 以正在使用的训练入口和 DataLoader 为唯一格式权威。历史 `scripts/laya_finetune/data/` 如果不存在，不得假定文件格式。
- 如现有格式是 `state/order/options` 等 JSONL 结构，严格复用既有字段名、序列化、card schema、label/card ID 约定及 tokenizer 拼接规则；动态 options 数量必须获得训练器支持。
- 每条记录应能追溯源行并保留 `K`、语言分层、召回证据 ID、阈值和 label 生成原因。若模型输入 schema 不允许额外 metadata，将 audit metadata 放在平行 sidecar 文件，不要擅自在模型输入添加字段。
- 如果当前 trainer 假定固定 4 张卡而不支持动态选项，先做兼容性报告，不得悄悄填充假候选或伪造格式。

## 7. 输出文件

全部写入新目录 `scripts/laya_finetune/data_v4/`，不覆盖现有文件：

- `train.jsonl`
- `test.jsonl`
- `validation.jsonl`
- `build_report.md`：数据构造逻辑、阈值、SQL 来源/版本、切分规则、标签规则、K 逻辑、限制和结论。
- `statistics.json`：机器可读统计、数量核对、SHA256、随机种子、阈值与版本。
- `split_distribution.csv`：每个 capability × split_language × split 的数量/占比。
- `candidate_k_distribution.csv`：各 split 的 K=0/1/2/3 计数、label 分布与 REJECT 原因分布。
- `retrieval_audit.jsonl` 或 `retrieval_audit.csv`：每条样本的召回、阈值过滤、聚合排序、Top-K、最终 options 和 label 决策证据。
- `annotation_review_queue.csv`：未决 registry、低质量 query、语言不确定、语义复核样本。
- `README.md`：输入文件 hash、schema、复现命令、所有输出说明。
- 新建可复现构建脚本（如 `build_v4_dataset.py`），仅当能复用真实 SQL 接口并确认训练 schema 后创建；不得修改任何旧脚本。

## 8. 统计报告必须回答

1. 每种 capability 的 zh/en/mixed/undetermined 原始数量。
2. 每种 capability × zh/en 的 Train/Test/Validation 实际样本数和比例。
3. REJECT 的源 No-Action 数，以及 Action 因召回失败/top3 截断转为 REJECT 的数量。
4. 全体及各 split 中 K=0、1、2、3 的计数与占比。
5. 每个 split 的 options 卡片数量分布；所有 label 是否都存在于 options。
6. SQL raw recall 数、阈值后数、capability 聚合后数、Top-3 覆盖率、gold capability 召回率、gold-in-options 比例。
7. 各 capability 在训练候选库的覆盖情况，尤其是低频 `cap-research` 和样本数较少类别。
8. 是否发生 query、来源行、近重复模板或候选 query 的跨 split 泄漏。
9. 原始样本数、split 样本数、成功构造数、BLOCKED/隔离数必须逐项对账。

## 9. 最终 Gate / 验收标准

- 输入 V4 CSV 的 SHA256 前后相同；任何 V1/V2/V3 数据、旧脚本和配置均未改变。
- Train/Test/Validation 不含相同规范化 query；尽量无近重复/模板族泄漏。
- 每个 split 的每条 label 都存在于 options 中。
- `K=0` 只有 REJECT option 并按项目规定直接 REJECT。
- `K=1/2/3` 的 options 数量分别为 2/3/4，且不含低于阈值的候选。
- No-Action 始终 label=`reject`。
- 只有当 capability 在最终 options 中时，Action 标签才可保持为该 capability；否则 label 必须为 `reject`。
- 真实 SQL threshold 和 card schema 可追溯，无伪造候选召回。
- 所有输出数量、分布、hash、标签与 K 统计可以由独立脚本复核。
- 不执行模型训练。
- 任何必要信息无法验证，或者 `cap-research` registry gate 未解决，则相关数据集验收为 `BLOCKED`，不可声称 PASS。

## 10. 本次 V4 文件审计提示

静态审计独立报告：`laya_tool_selection_v4_annotation_audit.md`。复核队列：`laya_tool_selection_v4_annotation_review.csv`。

当前静态检查未见字段级一致性错误，但存在以下必须关注事项：

- `cap-research` 852 行的 registry 合法性尚未确认。
- 18,000 行 `laya_rebuilt_v3` 样本标记为 `synthetic_needs_validation`。
- 2,000 行新增 REJECT 样本标记为 `rule_checked_pending_human_review`。
- 29,241 行 `query_list_finetune_v1` 的语言字段及 review status 为空；需要派生语言分层并按来源做审核。
- 发现一些模板拼接句、重复中文表达、英文中嵌中文占位词，见复核清单；这些是质量风险候选，不等于确认错标。
