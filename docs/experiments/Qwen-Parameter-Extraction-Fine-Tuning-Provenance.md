# Qwen 参数提取 — 数据来源 (Provenance) 与渲染核验报告

本报告由本地只读核查产生，**不代表任何远程执行结果**。所有哈希均为 SHA-256（除非显式标注 git blob = SHA-1）。

## 0. 结论速览

- 本地**完整契约核验实际执行且通过**：`validate_dataset.py` 退出码 0，输出 `live-contract check: on`，stderr 为空（无 `[validate] live prompt registry unavailable` 跳过行）。
- 为排除“校验是空操作”的可能，另做了**正/负对照探针**（§5.1）：正对照零失败；负对照（system 提示扰动一字符 / 工具面删一个 slot）立即失败。二者均符合预期，证明该比对是活的。
- **远程完整渲染校验未执行**，本报告不宣称该项已通过。远程只运行 `validate_dataset.py` 的数据哈希/计数/schema/泄漏检查，以及 `train.py --check`；因只上传 scripts 目录，远程 `prompt_for` 不可导入，**远程渲染检查会被跳过**。
- `manifest.source_commit = fdf96cf` **仅代表 HEAD**，不代表 dirty 源码快照；manifest 的 `source_files` 只记录了候选 JSONL 哈希，**未记录任何代码哈希**。

## 1. 仓库与 HEAD

| 项 | 值 |
|---|---|
| 仓库路径 | `e:/Work Space/Git Repositories/DeepDive` |
| HEAD (完整) | `bc6f496bb4e6f5a870f7f4b3bda8d7aca0b39cb1` |
| HEAD (短) | `bc6f496` |
| 分支 | `main` |
| manifest.source_commit | `fdf96cf`（数据构建时的 commit，**早于当前 HEAD**） |
| manifest.render_source | `production` |

## 2. 实际参与数据生成的源码文件路径 + SHA-256（工作副本）

数据生成路径：`prepare_dataset.py` 从 sys.path 导入生产模块，用 `prompt_for(cap)` 渲染 system、用 `_user_prompt(...)` 渲染 user 消息；`pe_contract.py` 提供 SCHEMA / model_slots / required 与标签推导。

| 文件 | 工作副本 SHA-256 | 与 HEAD 关系 |
|---|---|---|
| `packages/core/application/chat/intent_funnel/argument_acquisition/prompts.py` | `cc3bbc6105395cd855f340f412e1b6031736fe81ab19328d1fe78f348574c700` | 与 HEAD 一致（clean） |
| `packages/core/application/chat/intent_funnel/argument_acquisition/extractor.py` | `2139726507a8f5eff91d3729080e3bd2241895b2ee93334dc791d32a6b0c1e4a` | **已修改 (M)** |
| `packages/core/application/chat/intent_funnel/argument_acquisition/context_bundle.py` | `5c73532c559484664fd4c16751b68b19677eefb1aab9bf397222ffb5605f6022` | 与 HEAD 一致（clean） |
| `scripts/qwen_parameter_extraction_finetune/pe_contract.py` | `bd29aebba8c216523d172079e62ae69805c5810a77633628924cf9d8a6203e38` | **已修改 (M)** |
| `scripts/qwen_parameter_extraction_finetune/prepare_dataset.py` | `47bf8ca87f40a8763e7ce571f310ed33a27fe312bdb3b3c015445b62fc011d79` | **已修改 (M)** |

> 说明：`source_grounding.py` 已从工作副本移除（当前不存在于仓库）。`scripts/qwen_parameter_extraction_finetune/` 现已被 git 跟踪（`git ls-files` 计数 = 25）。

## 3. HEAD = bc6f496 与 dirty worktree 的区别

用 `git hash-object`（git blob SHA-1）对工作副本与 HEAD 逐文件比对：

| 文件 | worktree blob | HEAD blob | 判定 |
|---|---|---|---|
| `.../prompts.py` | `2e9e5a3acf6d017f7e91d252d1fb5812462dfd71` | `2e9e5a3acf6d017f7e91d252d1fb5812462dfd71` | IDENTICAL |
| `.../extractor.py` | `72462445c51a26afe43b729a4795ec04ffbb6bac` | `643848a05642e8347d2987a4f437fb93941e1b99` | **DIFFERS** |
| `.../context_bundle.py` | `2d722e67f54f6234bf06b851fa07f88c5b54554c` | `2d722e67f54f6234bf06b851fa07f88c5b54554c` | IDENTICAL |
| `scripts/.../pe_contract.py` | `dad9d02feb8df7a457d6e18294832cf05f9eb165` | `877accde559f41671a6a662dc3f0ca7a66444322` | **DIFFERS** |
| `scripts/.../prepare_dataset.py` | `656d93afa930deeb1eca08b4d4d14ddae3db82cb` | `eb1f982632bdb3304a3670a2ad02627dd99d2006` | **DIFFERS** |

**含义**：`prompt_for` 所在的 `prompts.py` 现已与 HEAD 一致（clean）；`extractor.py`（`_user_prompt`）、`pe_contract.py`、`prepare_dataset.py` 仍为 dirty 工作副本。`manifest.source_commit=fdf96cf` 锚定的是数据构建时的 commit，**早于当前 HEAD `bc6f496`**，也不锚定生成时的工作区状态。

## 4. train / validation / test 数据哈希 + 行数

| split | SHA-256 | 行数 | 与 manifest 一致 |
|---|---|---|---|
| train | `2065cdbedf1817f66b33ea21711a415d6ab9dc98ea65a077d4f28be09c1600a4` | 10303 | 是 |
| validation | `0a88be602d740e39b6b79753549fa3d405f2d6dc5213ceda6ad92d19dc06488a` | 1290 | 是 |
| test | `d048f1b53dbae1ea6a2f1f128382724f316badc32e08f6600d1037185a8cf47f` | 1290 | 是 |

manifest `counts`：`raw_candidates 12883 / isolated 0 / exact_duplicates_dropped 0 / valid_after_audit 12883 / kept_after_dedup 12883 / emitted 12883`，三联 split 合计 = 12883，与磁盘行数一致（`include_clarify_rows: true`，比早期版本多纳入曾按 `clarify_required_missing` 隔离的行）。

候选源文件（数据生成输入）哈希与 manifest `source_files` 完全一致：

| 源 | SHA-256 |
|---|---|
| `.../data/V5/delveta_pe_train_contract_aligned_v5_candidate.jsonl` | `4cf05489c005f13f3d99e46296d33bad65a14deacc1cbfaa33c090b124433647` |
| `.../data/V5/delveta_pe_test_contract_aligned_v5_candidate.jsonl` | `5f010c3ef4668b661e42765f9c9a84499c76c1a0ef288e1aa828c0112fcb52c6` |

## 5. 本地完整契约核验 —— 实际执行状态与结果

命令（工作目录 = `scripts/qwen_parameter_extraction_finetune/`，解释器 = 仓库 `.venv`，Python 3.13.11）：

```
<E:\...\DeepDive\.venv\Scripts\python.exe> validate_dataset.py
```

结果：

```
EXIT=0
----- STDOUT -----
VALIDATION PASSED
{"train": 10303, "validation": 1290, "test": 1290}
live-contract check: on
empty-{} negative targets (excluded from payload-leak check): {"train": 1169, "validation": 21, "test": 24}
[info] cross-capability payload collisions (NOT leakage): 4
  [info] payload shared across capabilities ['cap-social-search', 'cap-translate']
  [info] payload shared across capabilities ['cap-social-search', 'cap-translate']
  [info] payload shared across capabilities ['cap-social-search', 'cap-translate']
  [info] payload shared across capabilities ['cap-social-search', 'cap-translate']
----- STDERR -----
(空)
```

判定：
- `live-contract check: on` 直接表明 app prompt registry 可导入、`prompt_for` 可用（非 `static-fallback`）——system 提示的实时等式分支**未跳过**；stderr 为空（无 `[validate] live prompt registry unavailable` 跳过行）。
- 该次运行对全部 12883 行逐条执行 `check_sample`，包含：system 内容 `sys_c == prompt_for(cap)` 的**实时逐字节相等**；user 消息的**结构化**校验（roles / 静态 slot 块 / 恰好一个 `<user_sentence>` 且以其结尾，validate_dataset.py:113-128）；工具面 == 单条收窄规则 `search_scope.effective_model_slots` 产生的 model face；以及声明式缺槽闸门 `_check_meta`。
- 空 `{}` 负样本按 split 单列（train 1169 / val 21 / test 24），不计入 payload 泄漏判定；4 条跨能力 payload 碰撞为**信息性**报告，不判泄漏。

### 5.1 正/负对照探针（证明校验是“活的”，非空操作）

探针：`F:/WorkSpace/Delveta/.tmp/pe_probe_v2.py`（只读，不写仓库）。结果：

```json
{
  "prompt_for_is_none": false,
  "narrowing_rule_loaded": true,
  "sample_id": "aug-test-folder-session-007-t1",
  "capability_id": "cap-create-folder",
  "live_contract_check": "on",
  "positive_added_failures": 0,
  "negative_system_perturb_failures": 1,
  "negative_system_msg": "[test] aug-test-folder-session-007-t1: system content != live prompt_for(cap-create-folder)",
  "negative_face_narrow_failures": 1,
  "negative_face_msg": "[test] aug-test-folder-session-007-t1: function.parameters != the expected model face",
  "PASS": true
}
```

- **正对照**：未改动的冻结行经 `check_sample` 新增 **0** 条失败。
- **负对照 1（system 提示）**：把 system 内容末尾改一字符后，实时等式判定**立即失败**——证明 `sys_c == prompt_for(cap)` 确实在做比较，不会恒真。
- **负对照 2（工具面）**：从 function face 移除一个 slot 后，收窄面判定**立即失败**——证明 face 比对是活的。

### 5.2 由核验得到的源码一致性推论

- **system 提示**：全部 12883 行的 system 内容与**实时** `prompt_for(cap)` 逐字节相等，而 `prompt_for` 来自 `prompts.py`（当前与 HEAD 一致）。若该文件与生成数据所用版本有任何字节差异，此等式必然失败；等式全过 ⇒ 当前 `prompts.py` 的 system 提示与数据一致。
- **user 提示**：校验器**不再**在单轮假设下对 user 消息做字节重渲染（那正是多轮样本误报的来源），改为**结构化**校验。因此本报告**不宣称** user 消息与 `extractor._user_prompt` 逐字节一致；该等式由 `tests/test_prompt_contract_invariants.py` 单独保证。
- **工具面**：每行 face == 唯一收窄规则 `search_scope.effective_model_slots(cap, query, model_slots(cap))` 产生的 model face（静态 capability 为完整面；仅 `cap-social-search` 的 `r/<name>` 合法收窄）。

## 6. 边界与未执行项（重要）

1. **远程实时契约校验未执行**：本报告不宣称远程该项通过。远程仅执行 `validate_dataset.py` 的数据哈希/计数/schema/跨 split 与 Formal-500 泄漏检查，以及 `train.py --check`。
2. **远程实时契约检查会被跳过**：因上传范围为 scripts 目录、不含 `packages/core`，远程 system 提示等式因 `prompt_for` 不可导入而回落为 `static-fallback`（收窄面仍强制）。此跳过的**质量门由本地本次核验替代覆盖**（已在 §5 证明实际执行且通过）。
3. **manifest 未锚定代码**：`source_files` 仅含候选 JSONL 哈希，不含 prompts.py/extractor.py/pe_contract.py 等代码哈希；代码来源证据仅存在于本报告与 §5 的实时契约核验通过。
4. **提交状态**：`prompts.py` 已与 HEAD 一致（clean）；`extractor.py`、`pe_contract.py`、`prepare_dataset.py` 仍为已跟踪的未提交修改（M）；`source_grounding.py` 已从工作副本移除；`scripts/qwen_parameter_extraction_finetune/` 目录已被 git 跟踪。数据由 dirty 工作副本渲染，代码来源证据存在于本报告与 §5 的实时契约核验中。
5. 本报告未修改 manifest、未修改任何数据、未修改生产代码。

## 7. 附带核查（配置）

- `scripts/qwen_parameter_extraction_finetune/config.yaml`：`train.save_total_limit: 20`（第 107 行）、`output.manifests_dir: data`（manifest 与冻结 split 同目录，见 §4）。
- 该配置携带契约对齐改动：`source.*` 指向 V5 候选、`build.include_clarify_rows: true`（见 §4）。
