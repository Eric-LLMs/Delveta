# Laya Tool Router V4 数据统计报告

V4 = V3 + `no_action_queries_v1.csv`（全部标为 REJECT），随后执行 row-level shuffle。
本报告由 `build_v4_dataset.py` 生成，所有数字均可从输入与输出 CSV 复核。

## A. 输入数据概况

### A.1 V3 数据集

- 路径：`scripts/laya_finetune/data_v3/laya_tool_selection_v3.csv`
- 总行数：47241；空 query：0
- capability 标签数：19
- reject 行：1000（2.12%）
- 参数字段非空：29669；参数为非空对象：22669
- 语言分布：{"unknown": 29241, "zh": 9000, "en": 9000}

### A.2 非 action 补充集

- 路径：`data/BaseData/no_action_queries_v1.csv`
- 总行数：2000；空 query：0
- 原始标签：`capability_id=no-action`，按任务要求统一改为 `reject`
- 语言分布：{"en": 1000, "zh": 1000}
- 参数：全部为 `{}`，`parameter_status=not_applicable`
- review_status：`rule_checked_pending_human_review`

## B. 合并与去重结果

| 项目 | 数量 |
|---|---:|
| V3 行数 | 47241 |
| 非 action 行数 | 2000 |
| 合并后总行数 | 49241 |
| 完全重复样本 | 0 |
| query 重复样本 | 0 |
| 同 query、同标签、同输出重复 | 0 |
| 同 query、同标签、参数补全合并 | 0 |
| 同 query、不同标签冲突 | 0 |
| 同 query、同标签、不同参数输出冲突 | 0 |
| 隔离组数 / 隔离行数 | 0 / 0 |
| 作为重复剔除的行数 | 0 |
| **最终保留行数** | **49241** |

去重口径：query 经 NFKC 归一化、空白折叠、casefold 后比较。标签或参数冲突的整组样本隔离，
不进入训练集。两个输入之间不存在规范化 query 重叠，本轮无样本被剔除或隔离。

## C. 最终训练集分布

- 最终样本总量：49241
- capability 最大值 / 最小值 / 中位数：4445 / 852 / 1916
- 参数字段非空（含 `{}`）：31669（64.31%）
- 参数为非空对象：22669（46.04%）
- 非空参数 JSON 解析成功：31669（占非空 100.0%）

### C.1 Action / REJECT

| 标签 | 数量 | 占比 |
|---|---:|---:|
| true | 46241 | 93.91% |
| false | 3000 | 6.09% |

REJECT 由 V3 的 1,000 行与新增 2,000 行组成，共 3,000 行。

### C.2 capability 分布

| capability_id | 数量 | 占比 |
|---|---:|---:|
| cap-social-search | 4445 | 9.03% |
| cap-create-folder | 4121 | 8.37% |
| cap-translate | 3981 | 8.08% |
| cap-rag-search | 3941 | 8.0% |
| cap-add-term | 3939 | 8.0% |
| cap-web-search | 3936 | 7.99% |
| reject | 3000 | 6.09% |
| cap-vision | 1923 | 3.91% |
| cap-read-document | 1916 | 3.89% |
| cap-read-file | 1916 | 3.89% |
| cap-artifact | 1914 | 3.89% |
| cap-edit-file | 1914 | 3.89% |
| cap-bash | 1912 | 3.88% |
| cap-slides | 1909 | 3.88% |
| cap-mindmap | 1906 | 3.87% |
| cap-pdf-table-to-text | 1906 | 3.87% |
| cap-summary | 1906 | 3.87% |
| cap-pdf-extract-text | 1904 | 3.87% |
| cap-research | 852 | 1.73% |

### C.3 语言分布

| 语言 | 数量 | 占比 |
|---|---:|---:|
| unknown | 29241 | 59.38% |
| en | 10000 | 20.31% |
| zh | 10000 | 20.31% |

### C.4 Shuffle

- 随机种子：20261019
- shuffle 前 / 后行数：49241 / 49241
- 行级打乱后多重集一致性：PASS
- shuffle 后未再按类别、来源或标签排序

## D. 数据质量审计

- 空 query：0
- 空标签：0
- 参数 JSON 解析失败：0
- 同 query 不同标签：0
- 同 query 同标签不同参数：0
- 格式转换失败：0
- 输入文件未被修改：PASS

### 发现项

| issue_id | 类型 | 级别 | 数量 | 处理 |
|---|---|---|---:|---|
| R-NOACTION-REVIEW | no_action_rows_pending_human_review | info | 2000 | kept_as_reject |
| L-RESEARCH | research_lane_category | review | 852 | kept_unchanged_flagged_for_human_review |
| L-UNKNOWN | language_missing_in_input | info | 29241 | language_left_empty |
| P-EMPTY | parameters_empty_or_not_extracted | info | 26572 | kept_empty_by_design |

说明：

1. 新增 2,000 条非 action 样本全部标为 `reject`，`tool_name` 置空，参数保持 `{}`。
2. 其 `source_file` / `source_row` 保留源文件声明的生成集名称与样本 ID，便于逐条追溯。
3. `cap-research` 仍为 research 车道类别，标签未改动，待人工确认。
4. 语言只有 V3 的 18,000 条与新增的 2,000 条有显式值；其余 29,241 条来源无语言列，记为 `unknown`。
5. 参数抽取仍只覆盖 6 个能力；asset 类槽位由上下文提供，空值不是格式错误。
6. 未通过现有数据证据确认的问题未静默修复。

## E. 可复现性

- 输入 SHA256：
  - V3: `47149f0744d30501563717a0164fda9e518034e1c6c52b6c71580ba317110290`
  - no_action: `bce032d0dbfdd9f5061fb0310373654d5831cdabe90b5f2f28175c19d7340e00`
- 去重规则：见 B 节
- 冲突处理规则：整组隔离
- shuffle 种子：20261019
- 输出文件：
  - `laya_tool_selection_v4.csv`
  - `statistics_v4.json`
  - `statistics_report_v4.md`
  - `data_quality_issues_v4.csv`
- 数量核对：47241 + 2000 = 49241；剔除 0，隔离 0，最终 49241。
- V3 产物未被覆盖；V4 使用独立的统计文件名。
