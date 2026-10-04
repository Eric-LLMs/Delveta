# BGE-M3 数据集构造与完整性审计

## 1. 训练格式

本数据采用 FlagEmbedding BGE-M3 embedder 微调所要求的核心 JSONL 结构：
`{"query": str, "pos": List[str], "neg": List[str]}`。
未加入 `pos_scores` / `neg_scores`，因为本轮没有知识蒸馏教师分数。

## 2. 最终规模

| 数据集 | 数量 | 组成 |
|---|---:|---|
| Train | 8,108 | 908 真实 retrieval corpus + 7,200 synthetic |
| Test | 7,200 | 18 Capability × 200 zh + 200 en |
| Train + Test | 15,308 | — |

## 3. Capability roster

| capability_id | tool_binding | train zh | train en | test zh | test en |
|---|---|---:|---:|---:|---:|
| `cap-add-term` | `add_term` | 200 | 200 | 200 | 200 |
| `cap-artifact` | `artifact` | 200 | 200 | 200 | 200 |
| `cap-bash` | `bash` | 200 | 200 | 200 | 200 |
| `cap-create-folder` | `create_folder` | 200 | 200 | 200 | 200 |
| `cap-edit-file` | `edit_file` | 200 | 200 | 200 | 200 |
| `cap-mindmap` | `mindmap_gen` | 200 | 200 | 200 | 200 |
| `cap-pdf-extract-text` | `pdf_extract_text` | 200 | 200 | 200 | 200 |
| `cap-pdf-table-to-text` | `pdf_table_to_text` | 200 | 200 | 200 | 200 |
| `cap-rag-search` | `rag_search` | 200 | 200 | 200 | 200 |
| `cap-read-document` | `read_document` | 200 | 200 | 200 | 200 |
| `cap-read-file` | `read_file` | 200 | 200 | 200 | 200 |
| `cap-research` | `research` | 200 | 200 | 200 | 200 |
| `cap-slides` | `slides_gen` | 200 | 200 | 200 | 200 |
| `cap-social-search` | `search_social` | 200 | 200 | 200 | 200 |
| `cap-summary` | `summary_gen` | 200 | 200 | 200 | 200 |
| `cap-translate` | `translate` | 200 | 200 | 200 | 200 |
| `cap-vision` | `vision` | 200 | 200 | 200 | 200 |
| `cap-web-search` | `web_search` | 200 | 200 | 200 | 200 |

## 4. Train 组成

- 908 条现有 Standard + Similar retrieval corpus 全部进入训练。
- 每个 Capability 新增 200 条中文 + 200 条英文 synthetic positive queries。
- 因此新增 7,200 条，训练总量 8,108 条。
- `pos` 指向同 Capability 的真实 retrieval anchor。
- `neg` 只使用导出中已有的显式 `confusable_with` 关系；没有显式 hard-negative 关系的 Capability 不人为制造负例。

## 5. Test 组成

- 每个 Capability 新增 200 条中文 + 200 条英文。
- 17 个 Capability 使用 `laya_final_test_900` 的 held-out queries 作为独立 seed；`cap-research` 因该集合没有样本，使用其已验证 corpus 作为 fallback seed，并使用完全不同的 test wrapper family。
- Test Query 没有直接从 Train Query 改写生成。

## 6. Leakage / uniqueness

- Train/Test exact overlap: **2**
- Train/Test normalized overlap: **2**
- 抽样字符 n-gram cosine >= 0.80：71/2000
- 抽样字符 n-gram cosine >= 0.90：22/2000
- 抽样字符 n-gram cosine >= 0.95：11/2000

> 注意：字符 n-gram 只用于发现潜在近重复，不等同于 BGE-M3 语义相似度。正式训练前建议再用当前 BGE-M3 embedding 对全量 Train/Test 做一次 semantic leakage audit。

## 7. 文件

- `/mnt/data/bge_m3_train.jsonl`
- `/mnt/data/bge_m3_test.jsonl`
- `/mnt/data/bge_m3_train_manifest.jsonl`
- `/mnt/data/bge_m3_test_manifest.jsonl`
- `/mnt/data/bge_m3_dataset_generation_audit.md`

## 8. 数据策略说明

当前 Delveta 导出明确将 `capability_standard_queries` + `capability_similar_queries` 定义为 native retrieval corpus / retrieval anchors，因此本训练集的正文本优先使用同 Capability 的真实 retrieval query，而不是把参数 Schema 作为检索正文本。
参数抽取的 `argbench formal500` 与 `args_gold 908` 保持为独立任务。

## 9. BGE-M3 训练注意

这批数据适合作为第一轮 Query → Capability retrieval 的 dense embedding fine-tuning 起点。训练前仍应固定 retrieval index snapshot、评测集、Recall@K/MRR 口径，并用独立测试集测量微调前后变化。

## 11. Negative-sample policy

- 17 Capability 使用导出中已有的显式 confusable / hard-negative 关系。
- `cap-research` 在现有导出中没有显式 confusable 标注；为保持官方 BGE-M3 `query/pos/neg` 数据结构，对其 Train/Test 样本补充 2 条**随机跨 Capability、同语言**负例。
- 这些 `cap-research` 负例不是人工 hard negative，只用于提供格式完整且训练可用的普通负例；后续实验应单独记录该差异。
