# V4 独立校验器 — 设计方案与测试计划（待批准）

状态：**未实施**。本文只提交方案，等待明确批准后再改代码。

## 1. 问题

`scripts/laya_finetune/V2/run.sh` 的 `verify` 阶段固定调用
`layachoice_v2_verify.py`，而该脚本：

- 固定读取 `V2/data/`，不接受 `--data`；
- 校验 `data/SHA256SUMS` 与 `frozen_backup/` 的 1800 行冻结备份；
- 硬断言 `manifest.option_slots == 4`。

V4 的 manifest 是 `option_slots.rule == "K+1"`、`min=1`、`max=4`，且
`full_r1` 没有 `SHA256SUMS`/`frozen_backup`。因此
`bash run.sh all --data .../full_r1/data/threshold_XXX` 会先校验 V2、再训练 V4，
形成版本混用。这是需要独立校验器的原因。

## 2. 目标与非目标

目标：

- 让 `verify` 阶段校验“本次 `--data` 指向的那份数据”，而不是固定校验 V2；
- V4 校验只依赖标准库，不加载模型、不联网、不占 GPU；
- V2 行为保持不变。

非目标：

- 不修改训练/评估/导出逻辑；
- 不修改数据构建器或数据内容；
- 不把校验器接入生产运行时。

## 3. 方案（推荐 A）

### A. 新增 `layachoice_v4_verify.py`，`run.sh` 按数据契约分派

1. 新增 `scripts/laya_finetune/V2/layachoice_v4_verify.py`：
   - 接口：`python layachoice_v4_verify.py --data DATA_DIR`；
   - 复用 `layachoice_v2_dataset.read_manifest/load_split/frozen_calibration`，
     保证“校验器与训练器对同一行的理解一致”；
   - 仅覆盖 V4 的差异（K+1）与冻结的 V2 契约。

2. `run.sh` 的 `do_verify`：
   - 读取 `$DATA/manifest.json` 的 `option_slots`；
   - `rule == "K+1"` → V4 校验器；
   - 其他（V2 的整数 `4`）→ 原有 `normalize_bundle_eol` + V2 校验器；
   - manifest 缺失 → 直接失败，不再静默回落到 V2。

### B. 改造 `layachoice_v2_verify.py` 支持 `--data`

不推荐。该脚本的 `SHA256SUMS`、`frozen_backup`、`gold_slot_distribution`
断言都是 V2 冻结资产，改造成两条契约分支会让 V2 门禁承担双重职责，
违反“一个模块一个职责边界”。

## 4. V4 校验项

1. manifest 身份：`bench == LayaChoice-v2`、`formal_card_view == B_noprov`、
   `option_slots.rule == "K+1"`、`min/max == 1/4`、14 个冻结行键。
2. `source_files` 中随包文件（`v2_*.jsonl`、`construction_audit.jsonl`）
   的 SHA-256 与磁盘一致；`source_csv` 为上游输入，只确认已记录。
3. 每个 split 的行数等于 manifest；所有行通过训练适配器
   `adapt_row`（键、索引、排列合法性）。
4. 每行：`len(options) == len(order)`、恰好 1 个 REJECT、
   `reject_index` 正确、`shuffle.permutation` 可由
   `<seed_prefix>:<row id>` 复现；REJECT 卡片文本等于 manifest 冻结卡片。
5. 标签语义：`target_kind == "reject"` → `gold_capability_id is None` 且
   `gold_index == reject_index`；`capability` → Gold 指向实际卡且与 REJECT 不同。
6. 动态 K：K ∈ 0..3、选项数 1..4、manifest `observed` 与实际集合一致。
7. 修复块：`shuffle.seed_prefix` 以 `k0-to-k1-inplace-v1` 开头的行数，
   按 split 等于 `manifest.repair.repaired_by_split`，总数等于
   `repaired_rows`；这些行必须是 K=1、2 个选项、`target_kind == "capability"`。
8. calibration：行数、effective train 行数、indices/ids 可复现，且与训练集不相交。
9. split 之间 `id` 与 `source_query_id` 均不相交。
10. `gold_slot_distribution` 与现有行一致。
11. `construction_audit.jsonl` 覆盖全部行 id。
12. 拒绝校验归档/废弃目录（路径含 `full_r1_repaired` 或 `/archive/`）。

关键口径：`full_r1` 的 `construction_audit.jsonl` 保留修复前证据，
因此校验器**不得**断言 `candidate_count + 1 == len(order)`；
该字段是 pre-repair 溯源，不是当前行状态。

## 5. 测试计划

### 5.1 正向

- `python layachoice_v4_verify.py --data full_r1/data/threshold_082|085|088`
  三份全部 PASS。
- 与独立验证脚本交叉核对：修复行数、K 分布、REJECT 位置分布、
  `gold_slot_distribution`、source hashes 一致。
- `bash run.sh verify --data <V4 dir>` 走 V4 分支并通过；
  `bash run.sh verify`（默认 V2）走 V2 分支并通过。

### 5.2 负向（临时副本，不碰正式数据）

每个用例只破坏一个约束，断言退出码非 0 且报错定位到该检查：

- 改一行 `gold_index`；
- 改一行 `reject_index`；
- 改一行 `shuffle.permutation`；
- 改一行选项数（例如 K=1 行删掉 REJECT）；
- 改一行 `target_kind`/`gold_capability_id`；
- 改一行 `seed_prefix` 为未知前缀；
- 改 `manifest.repair.repaired_by_split` 数字；
- 改 `gold_slot_distribution` 数字；
- 改一个 `v2_*.jsonl` 字节使 SHA 不匹配；
- 把 `--data` 指向归档目录，断言拒绝。

### 5.3 回归

- `pytest tests/test_laya_v4_dataset_contract.py` 保持绿色
  （该测试当前只读 `smoke50_r1`，本次不改它）。
- `pytest tests` 基线不被破坏。
- `bash run.sh verify`（V2 默认数据）输出与改动前一致。
- `git diff` 只包含 `layachoice_v4_verify.py` 与 `run.sh` 两处。

### 5.4 验收标准

- 三阈值 V4 校验 PASS；
- 负向用例全部按预期失败；
- V2 verify 行为不变；
- 不启动训练/评估/部署；
- 提交前提供 diff、命令输出和失败用例清单。

## 6. 风险与回滚

- 风险：把 pre-repair 的 `construction_audit` 当成当前标签证据 → 已在
  校验项中明确排除。
- 风险：`run.sh` 分派误伤 V2 → 用 manifest 契约而非路径判断，并做 V2 回归。
- 回滚：删除新增校验器，`git checkout -- scripts/laya_finetune/V2/run.sh`；
  数据侧不受影响。

## 7. 待批准事项

1. 是否采用方案 A。
2. 是否允许新增 `layachoice_v4_verify.py` 并修改 `run.sh` 的 `do_verify`。
3. 测试是否按 5.1–5.3 执行，负向用例是否用临时副本（不写正式目录）。
