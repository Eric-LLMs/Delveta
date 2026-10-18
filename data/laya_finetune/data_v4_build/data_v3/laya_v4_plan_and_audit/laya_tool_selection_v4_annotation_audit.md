# Laya Tool Selection V4 — 标注静态审计报告

## 1. 审计范围与结论

- 审计文件：`laya_tool_selection_v4.csv`（本次上传副本：`/mnt/data/laya_tool_selection_v4.csv`）
- 数据行数：**49,241**（不含表头）
- SHA256：`e39adab214ed26992e26728a9e4c73659d5c924d283454d841abd789187c8445`
- 本次检查范围：结构一致性、工具名映射、query 去重、参数 JSON 语法、来源追溯、审核状态和规则式可疑样本筛查。
- **静态结构检查未发现已确认的字段级标注冲突；但无法据此证明 49,241 条 query 的语义标签全部正确。** 完整语义验证需要与当前正式 capability registry、SQL 召回结果及执行期工具意图规则对照。

## 2. 结构与一致性检查

| 检查项 | 结果 | 说明 |
|---|---:|---|
| 总数据行 | 49,241 | 不含表头 |
| 空 query | 0 | PASS |
| 空 `capability_id` | 0 | PASS |
| 非 reject 行 `tool_name` 为空 | 0 | PASS |
| reject 行 `tool_name` 非空 | 0 | 期望为空字符串 |
| `is_action` 非 true/false | 0 | PASS |
| `is_action` 与 `capability_id=reject` 冲突 | 0 | PASS |
| 同一 capability 对应多个 tool_name | 0 | PASS（仅检查 CSV 内部一致性，不代表 registry 一定有效） |
| NFKC + trim/空白折叠 + casefold 后重复 query 组 | 0 | PASS |
| 同 `source_file + source_row` 重复 | 0 | PASS |
| 非空参数 JSON 解析失败 | 0 | PASS |
| 参数为合法 JSON 但顶层不是 object | 0 | PASS |

**参数字段注意：** 空参数字符串有 **17,572** 行。这些不属于“非空 JSON 解析失败”，但空字符串本身不是 JSON。数据构造时应把它视为缺失/未提供参数；若训练 schema 要求 JSON object，应在新生成的数据中统一序列化为 `{}`，不得回写输入 CSV。

### 能力与工具名映射（CSV 内部）

每个 capability 在本文件中只对应一个 tool_name：

- `cap-add-term` → `add_term`
- `cap-artifact` → `artifact`
- `cap-bash` → `bash`
- `cap-create-folder` → `create_folder`
- `cap-edit-file` → `edit_file`
- `cap-mindmap` → `mindmap_gen`
- `cap-pdf-extract-text` → `pdf_extract_text`
- `cap-pdf-table-to-text` → `pdf_table_to_text`
- `cap-rag-search` → `rag_search`
- `cap-read-document` → `read_document`
- `cap-read-file` → `read_file`
- `cap-research` → `research`
- `cap-slides` → `slides_gen`
- `cap-social-search` → `search_social`
- `cap-summary` → `summary_gen`
- `cap-translate` → `translate`
- `cap-vision` → `vision`
- `cap-web-search` → `web_search`
- `reject` → ``

其中 `reject` 对应空 `tool_name`，符合当前 V4 文件的约定。

## 3. 标签分布

| `capability_id` | 数量 | 占比 |
|---|---:|---:|
| `cap-social-search` | 4,445 | 9.03% |
| `cap-create-folder` | 4,121 | 8.37% |
| `cap-translate` | 3,981 | 8.08% |
| `cap-rag-search` | 3,941 | 8.00% |
| `cap-add-term` | 3,939 | 8.00% |
| `cap-web-search` | 3,936 | 7.99% |
| `reject` | 3,000 | 6.09% |
| `cap-vision` | 1,923 | 3.91% |
| `cap-read-file` | 1,916 | 3.89% |
| `cap-read-document` | 1,916 | 3.89% |
| `cap-edit-file` | 1,914 | 3.89% |
| `cap-artifact` | 1,914 | 3.89% |
| `cap-bash` | 1,912 | 3.88% |
| `cap-slides` | 1,909 | 3.88% |
| `cap-mindmap` | 1,906 | 3.87% |
| `cap-pdf-table-to-text` | 1,906 | 3.87% |
| `cap-summary` | 1,906 | 3.87% |
| `cap-pdf-extract-text` | 1,904 | 3.87% |
| `cap-research` | 852 | 1.73% |

总计：Action **46,241**；REJECT **3,000**。

## 4. 来源与审核状态风险

### 来源分布

- `query_list_finetune_v1`：29,241 行
- `laya_rebuilt_v3`：18,000 行
- `synthetic_original`：2,000 行

### `review_status` 分布

| review_status | 行数 |
|---|---:|
| `<blank>` | 29,241 |
| `synthetic_needs_validation` | 18,000 |
| `rule_checked_pending_human_review` | 2,000 |

- **18,000** 行来源为 `laya_rebuilt_v3`，状态为 `synthetic_needs_validation`（其中 17,000 个 Action、1,000 个 REJECT），不能当作已人工验收标签。
- **2,000** 行来源为 `synthetic_original`，状态为 `rule_checked_pending_human_review`，属于规则初检通过、仍待人工复核的 No-Action 数据。
- **29,241** 行来源为 `query_list_finetune_v1`，`review_status` 为空，不能据此认为已审核。

### 关键阻塞项：`cap-research`

- `cap-research` 共 **852** 行，全部来自 `query_list_finetune_v1.csv`，`tool_name=research`；当前 `language` 与 `review_status` 均为空。
- 这些 query 的文字意图大体是多步骤研究/系统调研，但是否能作为 Router 可选 capability，必须对照当前正式 capability registry、工具注册表、候选卡片定义和推理路由配置。
- 如果正式候选卡片集合不包含 `cap-research`，这 **852** 行就会成为标签空间不一致，不能直接训练；也不能悄悄把它们改成 REJECT。必须先确认是有效 capability、旧 capability，还是独立 Research lane。

## 5. 规则式可疑样本筛查（不是已确认错标）

以下规则只筛出需要进一步检查的候选项，不将其自动认定为错标。候选行可能重叠，合计去重后 **1,379** 行，明细见 `laya_tool_selection_v4_annotation_review.csv`。

| 候选问题类型 | 命中数 | 含义 |
|---|---:|---|
| `REGISTRY_CHECK_CAP_RESEARCH` | 852 |
| `QUERY_TEMPLATE_COMPOSITION_REVIEW` | 404 |
| `CHINESE_REPEATED_HELP_PHRASE_REVIEW` | 136 |
| `MIXED_LANGUAGE_PLACEHOLDER_REVIEW` | 34 |

典型情况：

- 英文模板拼接/句式残留，如 `I want this handled by Put this sentence into English for me.`。目标意图可能依然是翻译，但表述质量较差，可能使模型学到模板伪特征。
- 中文重复表达，如 `可以帮我帮我找社区里真实用户的经验分享吗`。意图标签可能正确，但 query 自然度需要复核。
- 英文模板中直接嵌入中文占位词，如 `Extract the full text of 上传的PDF`。这可能是有意混合语种，也可能是生成器残留，需要判断是否保留及如何在语言分层中处理。
- `cap-research` 852 行需要 registry 专项核验。

## 6. 语言字段检查

| `language` 原始值 | 行数 |
|---|---:|
| `<blank>` | 29,241 |
| `en` | 10,000 |
| `zh` | 10,000 |

原始 `language` 空值共 **29,241** 行。仅做粗略字符脚本启发式（不是正式语言标注）时，这些空值大致分为：

- `zh`：16,302
- `en`：12,493
- `en-mixed`：257
- `zh-mixed`：189

这说明空值很可能包含中文和英文 query，不能把 29,241 行整体当作一个语言类别。后续 split 应建立独立的派生字段（如 `split_language`），保留原 `language`，对混合语言与不确定样本单列报告，不要静默回写原始 V4。

## 7. 结论与建议

1. **结构级标注检查：PASS**。字段关系、tool_name 映射、参数 JSON 语法、query 去重和来源行唯一性未发现确认的矛盾。
2. **语义标签整体正确性：尚不能判为 PASS。** 本文件没有保存 SQL 候选召回排名/分数/最终卡片，因此无法验证某条 query 是否应选其 `capability_id`，也无法判断 REJECT 是否因召回不足而产生。
3. **构造 V4 训练数据前，必须先解决 registry gate：** `cap-research` 852 行是否是合法候选 capability。
4. 不能把 `synthetic_needs_validation` 或空 `review_status` 视为人工确认。建议使用 `laya_tool_selection_v4_annotation_review.csv` 进行优先级复核，并再按每个 capability × 派生语言分层抽样审查普通样本。
5. 为判断最终训练标签，还必须在构造阶段保存 SQL 召回证据：recall rows、相似度、阈值过滤结果、capability 聚合排序、Top-3 卡片和最终 label。
