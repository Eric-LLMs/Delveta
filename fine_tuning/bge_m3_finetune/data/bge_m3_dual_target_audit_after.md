# BGE-M3 双检索数据修改后审计(dual-target audit — after)

> 只读校验:未修改生产代码、训练代码、registry、RAG、Agent、Intent Funnel;未下载模型、未安装依赖、未训练、未 commit / push。所有改动仅落在 `scripts/bge_m3_finetune/data/`。原始文件已冻结备份于 `data/_backup_20261004_pre_dual_target/`。

## 0. 本轮做了什么 / 没做什么

- **保留全部 18 个 capability**(含 cap-research),不再执行上一版“删除 research、只覆盖 17 个”的方案。
- **新增 Task A(Query → canonical description)**:18 capability 全覆盖,positive 逐字取自 `logs/_registry_dump.json` 的 `description`,不润色、不补造。
- **保留 Task B(Query → standard/similar query)**:逐字节保持原样(见 §1 SHA 证据)。
- **修复 CAP_RAG**:经查 `CAP_RAG` = `cap-rag-search`;其现有数据经全量校验**无归属/负例/泄漏缺陷**,唯一证据支持的问题是**语言元数据**(en-declared 实为 mixed),按 §六以 manifest `language_profile` 修复,**不改 query 文本**。
- **未删除任何行**(`rows_deleted = 0`);未替换任何 hard negative;未重划 train/test。

---

## 1. 修改前后文件清单与 SHA-256

原文件备份:`data/_backup_20261004_pre_dual_target/`(含 `SHA256SUMS.txt`)。

| 文件 | 修改前 行数 / SHA-256 | 修改后 行数 / SHA-256 | 性质 |
|---|---|---|---|
| `bge_m3_train.jsonl` | 8108 / `1e9b4759…78d8c4` | 16216 / `163b181f…fe99b8` | 改为**合并版**(Task B 块 + Task A 块) |
| `bge_m3_test.jsonl` | 7200 / `032e0ac4…1ba5b5` | 14400 / `22b1bfc7…bf0df39b` | 改为**合并版** |
| `bge_m3_train_manifest.jsonl` | 8108 / `95b61c5f…07c6d2` | 16216 / `57a0fb79…c3a33d` | 新 schema(增 task/lang_profile/seed 等) |
| `bge_m3_test_manifest.jsonl` | 7200 / `c4a0429a…b524a58` | 14400 / `8fb48be8…128f6b65` | 新 schema |
| `bge_m3_train_query.jsonl` | —(新增) | 8108 / `1e9b4759…78d8c4` | Task B,与原 train **同 sha** |
| `bge_m3_test_query.jsonl` | —(新增) | 7200 / `032e0ac4…1ba5b5` | Task B,与原 test **同 sha** |
| `bge_m3_train_desc.jsonl` | —(新增) | 8108 / `0b5b794f…d7f5d0` | Task A |
| `bge_m3_test_desc.jsonl` | —(新增) | 7200 / `2ea3a013…622a9caa` | Task A |
| `bge_m3_{train,test}_query_manifest.jsonl` | —(新增) | 与 query 同行数 | Task B manifest |
| `bge_m3_{train,test}_desc_manifest.jsonl` | —(新增) | 与 desc 同行数 | Task A manifest |
| `bge_m3_dataset_generation_audit.md` | 未改动 | 未改动 | 生成器自述(非权威),保留 |

**关键证据**:`bge_m3_train_query.jsonl` 的 sha256 = 修改前 `bge_m3_train.jsonl` 的 sha256(`1e9b4759…`);`bge_m3_test_query.jsonl` 同理(`032e0ac4…`)。即 **Task B 逐字节原样保留**。

新 manifest 字段(逐行):`split, task, gold_capability_id, declared_lang, language_profile, source, seed_family, query_sha256, negative_capabilities, negative_provenance`,Task B 另有 `positive_anchor`,Task A 另有 `positive_source, positive_language`。

---

## 2. 18 × (Task A / Task B) 覆盖矩阵

| capability | Task A train | Task A test | Task B train | Task B test |
|---|---:|---:|---:|---:|
| cap-add-term | 450 | 400 | 450 | 400 |
| cap-artifact | 451 | 400 | 451 | 400 |
| cap-bash | 450 | 400 | 450 | 400 |
| cap-create-folder | 452 | 400 | 452 | 400 |
| cap-edit-file | 451 | 400 | 451 | 400 |
| cap-mindmap | 448 | 400 | 448 | 400 |
| cap-pdf-extract-text | 447 | 400 | 447 | 400 |
| cap-pdf-table-to-text | 448 | 400 | 448 | 400 |
| cap-rag-search | 451 | 400 | 451 | 400 |
| cap-read-document | 452 | 400 | 452 | 400 |
| cap-read-file | 452 | 400 | 452 | 400 |
| cap-research | 452 | 400 | 452 | 400 |
| cap-slides | 449 | 400 | 449 | 400 |
| cap-social-search | 454 | 400 | 454 | 400 |
| cap-summary | 448 | 400 | 448 | 400 |
| cap-translate | 449 | 400 | 449 | 400 |
| cap-vision | 455 | 400 | 455 | 400 |
| cap-web-search | 449 | 400 | 449 | 400 |

18/18 capability 在 Task A 与 Task B 均有覆盖;Task A 与 Task B 行数逐 capability 相等(同一 query 派生两条物理记录)。

---

## 3. CAP_RAG 问题证据与修复依据

1. **真实 capability ID**:仓库内**不存在**大写 `CAP_RAG` 标识符;唯一的 RAG 能力是 `cap-rag-search`(`tool_binding=rag_search`, `intent_kind=action`, registry `status=active/enabled=true`)。结论:`CAP_RAG ≡ cap-rag-search`。
2. **归属正确性(全量校验,0 违例)**:
   - train/test 中所有 `cap-rag-search` 行的 `pos` **100% 落在** `cap-rag-search` 的 standard/similar 语料内,且**不被任何其他 capability 拥有**(`pos_anchor_not_in_rag_corpus = 0`)。
   - 负例 owner 集恒为 `{cap-social-search, cap-web-search}`,与 curated `DISTRACTORS["cap-rag-search"]` 完全一致。
   - **不存在**“标记为 CAP_RAG 实际属于其他 capability”或反向误标的样本(全库 `dataset query 等于其他 cap 语料锚点 = 0`)。
3. **无 false negative / 无重复 / 无泄漏**:`neg==pos`、`query==pos`、exact/normalized train∩test 均为 0。
4. **唯一证据支持的问题 = 语言元数据**:`cap-rag-search` 测试集中 200 条 declared `en` 的行实为“英文框架 + 中文宾语”(如 `Find API超时记录 in my knowledge base.`),非纯英文。该现象共波及 **10 个 capability × 200 = 2000 行**(另 9 个:`pdf-extract-text, pdf-table-to-text, read-document, slides, social-search, summary, translate, vision, web-search`)。
5. **修复依据与做法**:按 §六,保留 query 文本不变,在 manifest 记录 `declared_lang`(原值)与 `language_profile`(重算值)。**未编造中文 description,未改写任何 query**。
6. **Task A 的 CAP_RAG description** 逐字取自 registry,与其余能力同源同规则。

> 说明:上一轮 recall 根因报告(`logs/_e2e_formal500/recall_coverage_rootcause_report.md`)中 `cap-rag-search` 12 条低于召回门 0.60,属**检索语料/门限**(registry+recall)问题,**不在本阶段可修改范围**(本阶段仅动 `data/`),故不改动、仅记录。

---

## 4. cap-research 保留情况

| 项 | 值 |
|---|---|
| Task A train / test | 452 / 400(**保留**) |
| Task B train / test | 452 / 400(**保留**,原样) |
| `intent_kind` | `research`(生产 Intent Funnel 召回谓词 `<> 'research'` 排除——**未强改候选空间**) |
| Task A description 来源 | registry 真实值 `"Multi-step Research OS project driver (plugin)."`(47 字符) |
| 中文 description | **缺失**(registry 无 `中文：` 段);`positive_language = en_only`;标记缺失,**未补造中文** |
| 负例 | 原 `negative_type=random_cross_capability`(train 452 / test 400),标为 `negative_provenance=random_cross_capability`,**保留未删**(见 §9) |

---

## 5. Task A / Task B 规模

| | Task A (desc) | Task B (query) | 合并 |
|---|---:|---:|---:|
| train | 8108 | 8108 | 16216 |
| test | 7200 | 7200 | 14400 |

Task A 为**新增**(唯一新增来源 = registry 18 条 canonical description 复用);Task B 行数与修改前完全相同。

---

## 6. en / zh / mixed 分布(`language_profile`,脚本存在规则)

规则:`含汉字且含拉丁字母 → mixed`;`仅汉字 → zh`;`仅拉丁 → en`。不因含汉字而并入 zh。

| split / task | en | zh | mixed |
|---|---:|---:|---:|
| train query | 4040 | 3002 | 1066 |
| train desc | 4040 | 3002 | 1066 |
| test query | 1600 | 2071 | 3529 |
| test desc | 1600 | 2071 | 3529 |

对照 `declared_lang`:train en=4040 / zh=4068;test en=3600 / zh=3600。
- train 的 1066 条差值是 **zh 句中嵌入拉丁技术词**(如 `把这份 PDF 的文字抽出来` / `docker-compose.yml`)。
- test 的 2000 条 en→mixed 差值 = 上述 §3.4 的“英文框架 + 中文宾语”生成族。
- **评测口径以 `language_profile` 为准**;`declared_lang` 原值保留备查。

---

## 7. 泄漏检查

| 检查 | Task A | Task B |
|---|---|---|
| exact train∩test(query 文本) | 0 | 0 |
| normalized train∩test(query 文本) | 0 | 0 |
| train∩test(positive 文本) | 18(=18 条 canonical description,设计如此) | 784(共享的 908 检索语料锚点) |

- **query 文本零泄漏**是硬结论:test query 不出现在 train(exact/normalized 均 0)。
- Task A 的 18 条 description 同时作为两边 positive,是“每 capability 一个检索目标”的**设计**,非 query 泄漏。
- Task B 的 784 条共享 positive 是**检索语料本身**(registry corpus 是两条链路共用的索引目标),非 test query 泄漏。
- **source-group 泄漏**:`seed_family` 仅 test 可信(`laya_final_test_900`=6800、`verified_corpus_cap_research_fallback`=400,二者互斥);train 无可信 seed 来源,**不伪造**,故 train↔test 的 source-group 对账**无法完成**(记为待定,见 §12)。因 query 零泄漏,此缺口不影响“test query 未参与训练”的结论。

---

## 8. query == pos 检查

| task | train | test |
|---|---:|---:|
| Task A | 0 | 0 |
| Task B | 0 | 0 |

无 identity 退化行。A/B 两个 positive **从不进入同一条 `pos` 列表**(`rows_mixing_A_and_B_positives = 0`)。

---

## 9. false-negative 与负例来源

| 检查 | 结果 |
|---|---|
| Task A 中 `neg` 等于 gold description | 0 |
| Task A 中 `negative_capabilities` 含 gold | 0 |
| Task A 中 `pos` ≠ gold description | 0 |
| Task B / A 中 `neg` 实为 gold 正例(false negative) | 0 |

负例来源(`negative_provenance`):

| split | curated_confusable | random_cross_capability |
|---|---:|---:|
| train | 7656 | 452(全部 cap-research) |
| test | 6800 | 400(全部 cap-research) |

- `cap-rag-search` 负例 = curated confusable(web/social),正确。
- **可疑负例(标记不删)**:cap-research 的 `random_cross_capability` 负例(共 852 条 train+test 行,1704 个 neg 槽)来源随机、非 curated,标为待复核;本阶段**未删除/未替换**(依 §五、§七“无依据不自动删除”)。

---

## 10. 修改 / 保留 / 排除 / 待复核

| 项 | 数量 |
|---|---:|
| 新增 Task A 行 | **15308**(train 8108 + test 7200;承载 = 18 条真实 description) |
| 保留 Task B 行(逐字节) | 15308 |
| 删除行 | **0** |
| 原地修改的 query/pos/neg 行 | **0** |
| 排除行 | **0**(cap-research 保留) |
| 待复核行(cap-research 随机负例) | 852 行 / 1704 neg 槽 |

---

## 11. 训练器 batch 约束与未解决风险

- **数据侧就绪**:manifest 携带 `task` 与 `gold_capability_id`,支持按能力/任务分组采样。
- **训练必须使用 capability-unique batch**(本阶段**不修改训练器**):
  - 同一 batch 内每个 `gold_capability_id` 最多出现一次;
  - 同一 query 的 Task A / Task B 两条记录不得同 batch。
- **理由**:FlagEmbedding 默认 in-batch-negative InfoNCE(`AbsModeling._compute_in_batch_neg_loss`),且 collator 每条记录只采样一个 positive。若同 batch 出现同能力的两条记录,A 的 positive 会被 B 当作负例推开 → **假负例伤害**;A/B 同 query 同 batch 同样冲突。
- **现有训练脚本无法保证该约束** → **需要一个 task-aware / capability-aware sampler**(或在确认后改用 `no_in_batch_neg`)。记为**未解决风险**,待训练阶段处理。
- **未解决风险:cap-research 随机负例**可能把合法正例当负例(与 §9 同一事项)。

---

## 12. 无法确定的 / 待人工确认

1. `CAP_RAG` 作为大写标识符在仓库中**无唯一对应**;本文按唯一 RAG 能力 `cap-rag-search` 处理。若用户所指另有其物,需指明。
2. **train 的 `seed_family` 缺失**:生成器未记录可复现 seed,`source` 仅到 `existing_corpus908` / `synthetic_from_existing_corpus_seeds` 粒度;未伪造 seed ID。
3. **`language_profile` 阈值未定义**:本版用“脚本存在”规则(→ train mixed=1066 含 zh+拉丁词句)。若需区分“zh 句嵌技术词”与“真混语”,需另定阈值,当前**未擅自细化**。
4. **cap-research 随机负例**:是否替换为 curated 或改为 no-negative,待确认。
5. **Task A 的 description 形态**:本版 positive = registry **整段 bilingual** description(逐字)。是否需要按 query 语言取对应单语段,属设计选择,**未擅自拆分**。
6. 上一轮 recall 报告的 `cap-rag-search` 12 条低分,根因在**检索语料/门限**(registry+recall),**不在本阶段可改范围**。

---

## 附:产物与校验

- 构建脚本:`scripts/bge_m3_finetune/build_dual_target.py`(输出 `data/_build_dual_target_report.json`)
- 校验脚本:`scripts/bge_m3_finetune/verify_dual_target.py`(输出 `data/_verify_dual_target_report.json`)
- 备份:`data/_backup_20261004_pre_dual_target/`(+`SHA256SUMS.txt`)
