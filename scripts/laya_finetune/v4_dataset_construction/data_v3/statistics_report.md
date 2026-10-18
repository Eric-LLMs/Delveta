# Laya Tool Router V3 数据统计报告

本报告由 `build_v3_dataset.py` 生成，所有数字均来自两个输入 CSV 与最终输出 CSV。

## A. 输入数据概况

### A.1 输入文件 1

- 路径：`data/BaseData/laya_chat_tool_selection_balanced_18k_v4_params_filled_v2.csv`
- 总行数：18000；有效样本：18000；空 query：0
- 原始字段：query, is_action, capability_id, language, parameters, parameter_status, source, review_status
- capability 标签数：18
- reject 行：1000（5.56%）
- 参数字段：存在（`parameters`）
- 参数字段非空行（含 `{}`）：18000；参数为非空对象：11000；参数为空或 `{}`：7000
- 语言分布：{"en": 9000, "zh": 9000}

| capability_id | 数量 | 占比 |
|---|---:|---:|
| cap-add-term | 1000 | 5.56% |
| cap-artifact | 1000 | 5.56% |
| cap-bash | 1000 | 5.56% |
| cap-create-folder | 1000 | 5.56% |
| cap-edit-file | 1000 | 5.56% |
| cap-mindmap | 1000 | 5.56% |
| cap-pdf-extract-text | 1000 | 5.56% |
| cap-pdf-table-to-text | 1000 | 5.56% |
| cap-rag-search | 1000 | 5.56% |
| cap-read-document | 1000 | 5.56% |
| cap-read-file | 1000 | 5.56% |
| cap-slides | 1000 | 5.56% |
| cap-social-search | 1000 | 5.56% |
| cap-summary | 1000 | 5.56% |
| cap-translate | 1000 | 5.56% |
| cap-vision | 1000 | 5.56% |
| cap-web-search | 1000 | 5.56% |
| reject | 1000 | 5.56% |

### A.2 输入文件 2

- 路径：`data/BaseData/query_list_finetune_v1.csv`
- 总行数：29241；有效样本：29241；空 query：0
- 原始字段：query, is_action, tool_name, capability_id, 参数以及参数值
- capability 标签数：18
- reject 行：0（0.0%）
- 参数字段：存在（`参数以及参数值`，合并后统一为 `parameters`）
- 参数字段非空行（含 `{}`）：11669；参数为非空对象：11669；参数为空或 `{}`：17572
- 语言字段：**不存在**（未推断，合并后记为 `unknown`）

| capability_id | 数量 | 占比 |
|---|---:|---:|
| cap-social-search | 3445 | 11.78% |
| cap-create-folder | 3121 | 10.67% |
| cap-translate | 2981 | 10.19% |
| cap-rag-search | 2941 | 10.06% |
| cap-add-term | 2939 | 10.05% |
| cap-web-search | 2936 | 10.04% |
| cap-vision | 923 | 3.16% |
| cap-read-document | 916 | 3.13% |
| cap-read-file | 916 | 3.13% |
| cap-artifact | 914 | 3.13% |
| cap-edit-file | 914 | 3.13% |
| cap-bash | 912 | 3.12% |
| cap-slides | 909 | 3.11% |
| cap-mindmap | 906 | 3.1% |
| cap-pdf-table-to-text | 906 | 3.1% |
| cap-summary | 906 | 3.1% |
| cap-pdf-extract-text | 904 | 3.09% |
| cap-research | 852 | 2.91% |

## B. 合并与去重结果

| 项目 | 数量 |
|---|---:|
| 输入 1 原始行数 | 18000 |
| 输入 2 原始行数 | 29241 |
| 合并后总行数 | 47241 |
| 完全重复样本 | 0 |
| query 重复样本 | 0 |
| 同 query、同标签、同输出重复 | 0 |
| 同 query、同标签、参数补全合并 | 0 |
| 同 query、不同标签冲突 | 0 |
| 同 query、同标签、不同参数输出冲突 | 0 |
| 隔离组数 / 隔离行数 | 0 / 0 |
| 作为重复剔除的行数 | 0 |
| **最终保留行数** | **47241** |

去重口径：query 经 NFKC 归一化、空白折叠、casefold 后比较；标签或参数冲突的整组样本隔离，不进入训练集。

两个输入文件之间不存在任何规范化 query 重叠，因此本轮没有实际剔除或隔离样本。

## C. 最终训练集分布

- 最终样本总量：47241
- capability 最大值 / 最小值 / 中位数：4445 / 852 / 1914
- 参数字段非空（含 `{}`）：29669（62.8%）
- 参数为非空对象：22669（47.99%）
- 非空参数 JSON 解析成功：29669（占非空 100.0%）

### C.1 Action / REJECT

| 标签 | 数量 | 占比 |
|---|---:|---:|
| true | 46241 | 97.88% |
| false | 1000 | 2.12% |

### C.2 capability 分布

| capability_id | 数量 | 占比 |
|---|---:|---:|
| cap-social-search | 4445 | 9.41% |
| cap-create-folder | 4121 | 8.72% |
| cap-translate | 3981 | 8.43% |
| cap-rag-search | 3941 | 8.34% |
| cap-add-term | 3939 | 8.34% |
| cap-web-search | 3936 | 8.33% |
| cap-vision | 1923 | 4.07% |
| cap-read-document | 1916 | 4.06% |
| cap-read-file | 1916 | 4.06% |
| cap-artifact | 1914 | 4.05% |
| cap-edit-file | 1914 | 4.05% |
| cap-bash | 1912 | 4.05% |
| cap-slides | 1909 | 4.04% |
| cap-mindmap | 1906 | 4.03% |
| cap-pdf-table-to-text | 1906 | 4.03% |
| cap-summary | 1906 | 4.03% |
| cap-pdf-extract-text | 1904 | 4.03% |
| reject | 1000 | 2.12% |
| cap-research | 852 | 1.8% |

### C.3 语言分布

| 语言 | 数量 | 占比 |
|---|---:|---:|
| unknown | 29241 | 61.9% |
| zh | 9000 | 19.05% |
| en | 9000 | 19.05% |

### C.4 Shuffle

- 随机种子：20261018
- shuffle 前 / 后行数：47241 / 47241
- 行级打乱后多重集一致性：PASS
- shuffle 后未再按类别、来源或标签排序

## D. 数据质量审计

- 空 query：0
- 空标签：0
- 非法输出（参数 JSON 无法解析）：0
- 同 query 不同标签：0
- 同 query 同标签不同参数：0
- 格式转换失败：0
- 输入文件未被修改：PASS

### 发现项

| issue_id | 类型 | 级别 | 数量 | 处理 |
|---|---|---|---:|---|
| P-NO-EXTRACTION | parameters_not_extracted_for_capability | info | 10878 | kept_with_empty_parameters |
| P-EMPTY-EXTRACT | parameters_empty_for_extraction_capability | review | 6694 | kept_missing_slot_requires_follow_up_at_execution |
| P-CONTEXT-OWNED | asset_slot_context_owned | info | 3000 | kept_empty_by_design |
| P-REQ-SLOT | required_slot_missing_with_nonempty_parameters | review | 1336 | kept_labeled_for_human_review |
| L-RESEARCH | research_lane_category | review | 852 | kept_unchanged_flagged_for_human_review |
| L-UNKNOWN | language_missing_in_input | info | 29241 | language_left_empty |
| R-SYNTHETIC | review_status_pending | info | 18000 | preserved_from_input |

说明：

1. 输入 2 只对 6 个参数抽取能力（add-term / create-folder / translate / rag-search / web-search / social-search）提供参数；其余能力的空参数是数据来源限制，不是格式错误。
2. 输入 1 的 pdf-extract-text / pdf-table-to-text / read-document 行参数为 `{}`（共 3,000 行）；vision 行带 `question` 但 `asset_id` 由 viewer/附件上下文提供。
3. `required_slot_missing_with_nonempty_parameters`（1,336 行）构成：vision `asset_id` 1,000 行（上下文提供）、add-term 缺少 `domain` 168 行、add-term 缺少 `term` 68 行、translate 缺少 `text` 100 行。缺失项保留原样，未补写。
4. 输入 1 全部 18,000 行的 `review_status` 为 `synthetic_needs_validation`，保留原值。
5. `cap-research` 在 DB 中 `intent_kind=research`，不属于 17 个 chat action 能力；标签按输入原样保留并标记待人工审核。
6. 输入 2 没有语言列，语言留空并统计为 `unknown`，未根据字符猜测。
7. 未通过现有数据证据确认的问题均标记为待人工审核，未静默修复。

## E. 可复现性

- 输入文件 SHA256：
  - input1: `d74b30969589402ddc42a395c671dedaeb4d6e9de8ab30f5f534b569b22b7e32`
  - input2: `c0f65430714a34be1adab3666516bb7a998d1a184bd98047146b1830ab20f802`
- 去重规则：见 B 节
- 冲突处理规则：整组隔离，不进入训练集
- shuffle 种子：20261018
- 输出文件：
  - `laya_tool_selection_v3.csv`
  - `statistics.json`
  - `statistics_report.md`
  - `data_quality_issues.csv`
  - `README.md`
- 数量核对：18000 + 29241 = 47241；剔除 0，隔离 0，最终 47241。

## 格式兼容性说明（阻塞项）

任务指定的参考目录 `scripts/laya_finetune/data/` **不存在**。现有训练格式位于
`scripts/laya_finetune/V2/data/v2_{train,val,test}.jsonl`，是固定的 4 路选择结构
（3 张 capability card + 1 张 REJECT card，模型输入仅 `state/order/options`，
且 schema 中没有任何参数字段）。

本任务的输入是平铺 CSV，包含 capability 标签与槽位参数。要生成 V2 的 4 路结构，
必须额外执行候选召回与 card 渲染，且会把参数值排除在训练输入之外，与“不得丢失参数值”
冲突。因此 V3 输出为平铺 CSV，保留 query、标签、参数和来源追溯字段，不伪造 4 路候选。
该项在验收中标记为 BLOCKED。
