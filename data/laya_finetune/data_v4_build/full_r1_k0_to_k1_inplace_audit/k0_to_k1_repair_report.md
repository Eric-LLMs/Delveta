# K=0 → K=1 原位修复报告

- 数据目录：`e:\Work Space\Git Repositories\DeepDive\scripts\laya_finetune\data_v4_build\full_r1`
- 只读对照：`e:\Work Space\Git Repositories\DeepDive\scripts\laya_finetune\data_v4_build\full_r1_repaired`
- 审计输出：`e:\Work Space\Git Repositories\DeepDive\scripts\laya_finetune\data_v4_build\full_r1_k0_to_k1_inplace_audit`
- 修复对象：原始 K=0、`is_action=True`、`candidate_count=0`、`source_gold_capability_id` 属于 18 个有效 capability 的行。
- 修复方式：补入唯一 Gold capability 卡 + 保留 REJECT，K=1，总选项 2。
- 注入候选不是 ANN 召回，`retrieval_score_available=false`、`candidate_source=source_gold_injected`，不伪造检索分数。
- 原始 `construction_audit.jsonl` 未修改；修复溯源见本目录明细。

## 1. 修改数量

| threshold | split | 修改行数 |
|---|---|---:|
| 0.82 | train | 107 |
| 0.82 | test | 38 |
| 0.82 | val | 7 |
| 0.82 | ALL | 152 |
| 0.85 | train | 421 |
| 0.85 | test | 149 |
| 0.85 | val | 33 |
| 0.85 | ALL | 603 |
| 0.88 | train | 1792 |
| 0.88 | test | 527 |
| 0.88 | val | 147 |
| 0.88 | ALL | 2466 |
| ALL | ALL | 3221 |

## 2. 修复前后 K 分布

| threshold | K | 修复前 | 修复后 |
|---|---:|---:|---:|
| 0.82 | 0 | 1208 | 1056 |
| 0.82 | 1 | 17848 | 18000 |
| 0.82 | 2 | 14955 | 14955 |
| 0.82 | 3 | 14322 | 14322 |
| 0.85 | 0 | 2541 | 1938 |
| 0.85 | 1 | 31857 | 32460 |
| 0.85 | 2 | 10665 | 10665 |
| 0.85 | 3 | 3270 | 3270 |
| 0.88 | 0 | 5130 | 2664 |
| 0.88 | 1 | 38954 | 41420 |
| 0.88 | 2 | 4036 | 4036 |
| 0.88 | 3 | 213 | 213 |

## 3. REJECT 位置分布

| threshold | split | 位置 0 | 位置 0 占比 | 位置 1 | 位置 1 占比 | 合计 |
|---|---|---:|---:|---:|---:|---:|
| 0.82 | train | 60 | 56.07% | 47 | 43.93% | 107 |
| 0.82 | test | 18 | 47.37% | 20 | 52.63% | 38 |
| 0.82 | val | 3 | 42.86% | 4 | 57.14% | 7 |
| 0.82 | ALL | 81 | 53.29% | 71 | 46.71% | 152 |
| 0.85 | train | 192 | 45.61% | 229 | 54.39% | 421 |
| 0.85 | test | 79 | 53.02% | 70 | 46.98% | 149 |
| 0.85 | val | 20 | 60.61% | 13 | 39.39% | 33 |
| 0.85 | ALL | 291 | 48.26% | 312 | 51.74% | 603 |
| 0.88 | train | 927 | 51.73% | 865 | 48.27% | 1792 |
| 0.88 | test | 272 | 51.61% | 255 | 48.39% | 527 |
| 0.88 | val | 77 | 52.38% | 70 | 47.62% | 147 |
| 0.88 | ALL | 1276 | 51.74% | 1190 | 48.26% | 2466 |
| ALL | ALL | 1648 | 51.16% | 1573 | 48.84% | 3221 |

随机种子规则：`k0-to-k1-inplace-v1:<threshold>:<split>:<row id>`，基础顺序 `[gold capability, REJECT]`，用 `random.Random(seed).shuffle([0,1])` 生成排列；每条的种子、排列和 REJECT 位置记录在明细 JSONL 中。

## 4. 验证结果

- 目标行：K=1、恰好 1 张 capability 卡 + 1 张 REJECT、总选项 2、Gold 索引指向正确 capability：PASS
- 非目标行逐字节一致、行顺序不变、文件行数不变：PASS（0 行变更）
- 文件归属、split、query、样本 ID、`src_sha` 未改变：PASS
- 与 `full_r1_repaired` 交叉核对：目标集合、Gold capability、卡片文本一致：PASS
- 待确认样本：0
- 未启动模型训练或部署。

## 5. 文件与备份

| 文件 | 修改前 SHA-256 | 修改后 SHA-256 | 目标行 | 非目标行变更 |
|---|---|---|---:|---:|
| `full_r1\data\threshold_082\v2_train.jsonl` | `f643093798bb9066…` | `c7f887b6dc19af4e…` | 107 | 0 |
| `full_r1\data\threshold_082\v2_test.jsonl` | `efc3dae7e9729d6a…` | `6d92dd264ab3ce90…` | 38 | 0 |
| `full_r1\data\threshold_082\v2_val.jsonl` | `9da7f5a68f9f8990…` | `0e95a68359675df0…` | 7 | 0 |
| `full_r1\data\threshold_085\v2_train.jsonl` | `d65225745d3e0af2…` | `0c32acb00e00a0ea…` | 421 | 0 |
| `full_r1\data\threshold_085\v2_test.jsonl` | `5180eb9cca3d7a57…` | `286c1586421ad4d0…` | 149 | 0 |
| `full_r1\data\threshold_085\v2_val.jsonl` | `db343273f63af9b9…` | `4278cb0dc0ae360e…` | 33 | 0 |
| `full_r1\data\threshold_088\v2_train.jsonl` | `ae3d5cf04e134a98…` | `7c08dd76df5a93d3…` | 1792 | 0 |
| `full_r1\data\threshold_088\v2_test.jsonl` | `4bdaec8af8889a12…` | `568685908b4977f6…` | 527 | 0 |
| `full_r1\data\threshold_088\v2_val.jsonl` | `66411a25add33a13…` | `3a5ca0701ba0322a…` | 147 | 0 |

完整哈希见 `k0_to_k1_diff_audit.json`；原始文件备份位于 `e:\Work Space\Git Repositories\DeepDive\scripts\laya_finetune\data_v4_build\full_r1_k0_to_k1_inplace_audit\backups`。

## 6. 待确认样本

无。所有目标样本均通过 Gold、引用行和唯一定位校验。

## 7. 结论

PASS。目标行已原位从 K=0 改为 K=1，其他数据行未改动；未启动训练或部署。
