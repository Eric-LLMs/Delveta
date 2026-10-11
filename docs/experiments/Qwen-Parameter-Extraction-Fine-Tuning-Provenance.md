# Qwen 参数提取 — 数据来源 (Provenance) 与渲染核验报告

本报告由本地只读核查产生，**不代表任何远程执行结果**。所有哈希均为 SHA-256（除非显式标注 git blob = SHA-1）。

## 0. 结论速览

- 本地**完整逐字节渲染核验实际执行且通过**：`validate_dataset.py` 退出码 0，输出 `byte-render check: on`，stderr 为空（无 `[validate] live prompt registry unavailable` 跳过行）。
- 为排除“渲染校验是空操作”的可能，另做了**正/负对照探针**：正对照逐字节相等；负对照（查询扰动一字符）立即不等。二者均符合预期，证明该比对是活的。
- **远程完整渲染校验未执行**，本报告不宣称该项已通过。远程只运行 `validate_dataset.py` 的数据哈希/计数/schema/泄漏检查，以及 `train.py --check`；因只上传 scripts 目录，远程 `prompt_for` 不可导入，**远程渲染检查会被跳过**。
- `manifest.source_commit = fdf96cf` **仅代表 HEAD**，不代表 dirty 源码快照；manifest 的 `source_files` 只记录了候选 JSONL 哈希，**未记录任何代码哈希**。

## 1. 仓库与 HEAD

| 项 | 值 |
|---|---|
| 仓库路径 | `e:/Work Space/Git Repositories/DeepDive` |
| HEAD (完整) | `fdf96cf793177a648c64b1f5536b53a7d43f2d33` |
| HEAD (短) | `fdf96cf` |
| 分支 | `main` |
| manifest.source_commit | `fdf96cf`（= HEAD） |
| manifest.render_source | `production` |

## 2. 实际参与数据生成的源码文件路径 + SHA-256（工作副本）

数据生成路径：`prepare_dataset.py` 从 sys.path 导入生产模块，用 `prompt_for(cap)` 渲染 system、用 `_user_prompt(...)` 渲染 user 消息；`pe_contract.py` 提供 SCHEMA / model_slots / required 与标签推导。

| 文件 | 工作副本 SHA-256 | 与 HEAD 关系 |
|---|---|---|
| `packages/core/application/chat/intent_funnel/argument_acquisition/prompts.py` | `92b842325dd1cc2dbf3ccd9bf38ea483736f80de6dc2d7bd32e6c74c0ba494fa` | **已修改 (M)** |
| `packages/core/application/chat/intent_funnel/argument_acquisition/extractor.py` | `0b8024fe80dbde5135e18f621fc402651e57e6a58531d074c68853bb1814c5b9` | **已修改 (M)** |
| `packages/core/application/chat/intent_funnel/argument_acquisition/source_grounding.py` | `61fd8985c3d74ee97782171c2200fa293bffb4cd2e641cf2b67a1a969f5e04ce` | **未跟踪 (??)**，HEAD 中不存在 |
| `packages/core/application/chat/intent_funnel/argument_acquisition/context_bundle.py` | `5c73532c559484664fd4c16751b68b19677eefb1aab9bf397222ffb5605f6022` | 与 HEAD 一致（clean） |
| `scripts/qwen_parameter_extraction_finetune/pe_contract.py` | `48d197b8ac001aa5b6b55846cd0e7dd6a359b597ae096be4648a2a453236e5e8` | 未跟踪（整个 scripts 目录 untracked） |
| `scripts/qwen_parameter_extraction_finetune/prepare_dataset.py` | `5f692299f3451940990e291e8f0534705cd3120f55be321e9431572a0a4dd374` | 未跟踪 |

> 说明：`scripts/qwen_parameter_extraction_finetune/` 整棵目录为 untracked（`git ls-files` 计数 = 0），故其文件无 HEAD 版本可比对。

## 3. HEAD = fdf96cf 与 dirty worktree 的区别

用 `git hash-object`（git blob SHA-1）对工作副本与 HEAD 逐文件比对：

| 文件 | worktree blob | HEAD blob | 判定 |
|---|---|---|---|
| `.../prompts.py` | `5997dfd1a04d649591f40483dd2a4cedb3d1df74` | `92c697c1b3697dd1f13b60ff09169fe28d44cf76` | **DIFFERS** |
| `.../extractor.py` | `405cfacc9ed68d5779e2cf4ab4f9cf0d55b23547` | `25d5f8a525b995c974de432628e89f54d0948fdb` | **DIFFERS** |
| `.../context_bundle.py` | `2d722e67f54f6234bf06b851fa07f88c5b54554c` | `2d722e67f54f6234bf06b851fa07f88c5b54554c` | IDENTICAL |
| `.../source_grounding.py` | (untracked) | 不存在 | 无 HEAD 版本 |

**含义**：数据由 **dirty 工作副本**（prompts.py、extractor.py 已改动；source_grounding.py 未跟踪）渲染，而 `HEAD=fdf96cf` 对应的旧版本 prompts.py/extractor.py **不是**生成数据用的版本。因此 `manifest.source_commit=fdf96cf` 不能作为“源码快照”的证据——它只锚定 HEAD commit，不锚定生成时的工作区状态。

## 4. train / validation / test 数据哈希 + 行数

| split | SHA-256 | 行数 | 与 manifest 一致 |
|---|---|---|---|
| train | `dbd06f3a3e8e30a0d6395ec0fb5d80f3bf0ba85ea4bb7a9dfd0cd78ddab51222` | 8679 | 是 |
| validation | `addc0baa8fa61624ad054a54dfcac43e15ec27f03b985a61543db18d3185f949` | 1041 | 是 |
| test | `507bc036b4d9a15a797097e8074ba7471abcf2188f3ab0ff85458b49281327e2` | 1080 | 是 |

manifest `counts`：`raw_candidates 12000 / isolated 1200 / exact_duplicates_dropped 0 / valid_after_audit 10800 / kept_after_dedup 10800 / emitted 10800`，三联 split 合计 = 10800，与磁盘行数一致。

候选源文件（数据生成输入）哈希与 manifest `source_files` 完全一致：

| 源 | SHA-256 | 行数 |
|---|---|---|
| `data/delveta_parameter_extraction_v2/delveta_pe_train_candidate_6000.jsonl` | `4648503676de1245e8706475fea45a9fc9f091a3bb9558c21def3417c8ae6499` | 6000 |
| `data/delveta_parameter_extraction_v2/delveta_pe_test_candidate_6000.jsonl` | `b789213b91d4a755b15898719fdb1ddff2cb5f6020dfcdacde3ce114db5fa0e8` | 6000 |

## 5. 本地完整渲染核验 —— 实际执行状态与结果

命令（工作目录 = `scripts/qwen_parameter_extraction_finetune/`，解释器 = 仓库 `.venv`，Python 3.13.11）：

```
<E:\...\DeepDive\.venv\Scripts\python.exe> validate_dataset.py
```

结果：

```
EXIT=0
----- STDOUT -----
VALIDATION PASSED
{"train": 8679, "validation": 1041, "test": 1080}
byte-render check: on
----- STDERR -----
(空)
```

判定：
- `byte-render check: on` 直接表明 `_load_prompts()` 返回了真实 `prompt_for`（非 None）——渲染分支**未跳过**。
- stderr 为空，证明**未**触发 `[validate] live prompt registry unavailable ...; skipping byte-match render check`。
- 该次运行对全部 10800 行逐条执行了 `check_sample`，其中包含 system 内容 `sys_c == prompt_for(cap)` 与 user 内容逐字节重渲染比对（validate_dataset.py:125-135）。

### 5.1 正/负对照探针（证明比对是“活的”，非空操作）

探针：`F:/WorkSpace/Delveta/.tmp/pe_render_control_probe.py`（只读，不写仓库）。结果：

```json
{
  "prompt_for_is_none": false,
  "verdict": "LIVE",
  "sample_id": "test-add_term-en-0005",
  "capability_id": "cap-add-term",
  "query": "Add one term, offline benchmark, to information retrieval.",
  "sys_byte_equal": true,
  "usr_byte_equal": true,
  "perturbed_query": "Add one term, offline benchmark, to information retrievalX",
  "usr_byte_equal_after_perturb": false,
  "check_sample_added_failures": 0,
  "PASS": true
}
```

- **正对照**：存量 system/user 内容与实时重渲染逐字节相等（`sys_byte_equal=true`, `usr_byte_equal=true`）。
- **负对照**：把查询末尾改一字符后，实时重渲染**不再**相等（`usr_byte_equal_after_perturb=false`）——证明相等判定确实在做比较，不会因恒真而假通过。
- `check_sample` 对该样本新增 0 条失败。

### 5.2 由此得到的源码一致性推论

渲染核验逐字节通过了**全部** 10800 行。若当前 dirty 的 `prompts.py` / `_user_prompt`（`extractor.py`）与生成数据时的版本有任何字节差异，比对必然失败。既然全通过，**当前 dirty 工作副本与生成数据所用的生产渲染代码逐字节一致**——这是“本地实际参与数据生成的 dirty 工作副本”这一要求的直接证据。

## 6. 边界与未执行项（重要）

1. **远程完整渲染校验未执行**：本报告不宣称远程该项通过。远程仅执行 `validate_dataset.py` 的数据哈希/计数/schema/跨 split 与 Formal-500 泄漏检查，以及 `train.py --check`。
2. **远程渲染检查会被跳过**：因上传范围为 scripts 目录、不含 `packages/core`，远程 `_load_prompts()` 将因 `ImportError` 返回 None，渲染分支跳过并打印 stderr 跳过行。此跳过的**质量门由本地本次核验替代覆盖**（已在 §5 证明实际执行且通过）。
3. **manifest 未锚定代码**：`source_files` 仅含候选 JSONL 哈希，不含 prompts.py/extractor.py/pe_contract.py 等代码哈希；代码来源证据仅存在于本报告与 §5 的逐字节核验通过。
4. **未提交风险**：prompts.py、extractor.py 为已修改；source_grounding.py 与整个 scripts 目录为 untracked。若 dirty 工作区丢失，源码快照无法从 git 恢复。
5. 本报告未修改 manifest、未修改任何数据、未修改生产代码、未改配置（除本轮已批准的 `save_total_limit` 单项）。

## 7. 本轮附带核查（配置改动）

- 已批准的唯一配置改动：`scripts/qwen_parameter_extraction_finetune/config.yaml` 第 93 行 `save_total_limit: 3` → `20`。
- 只读解析确认：`pe_config.load_config()['train']['save_total_limit'] == 20`（其余项未变）。
- 相关测试：`tests/test_pe_finetune_pipeline.py` 26 passed；`tests/test_parameter_extraction_protocol.py` 36 passed。
