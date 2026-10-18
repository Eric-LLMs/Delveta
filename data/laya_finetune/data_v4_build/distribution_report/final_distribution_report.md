# 正式训练数据分布统计报告

## 1. 统计范围与口径

- 唯一正式数据源：`scripts/laya_finetune/data_v4_build/full_r1/data/`，覆盖 `recall_threshold_082`、`recall_threshold_085`、`recall_threshold_088`。
- 未读取归档的 `full_r1_repaired`；未重跑 embedding、ANN、SQL 或数据构建。
- 行级标签、K、Gold 分布以修复后的 `v2_{train,val,test}.jsonl` 为准。
- K = 选项中 capability 候选卡数量（不含 REJECT），总选项数 = K + 1。
- Gold Tool：`target_kind=capability` 时取 `gold_capability_id`；`target_kind=reject` 时记为 `REJECT`，不并入任何 Tool。
- 语言：沿用行内 `lang`；`zh`=中文，`en`=英文，`mixed`=其他/混合。本次未发现未知语言值。
- Gold 注入样本：以修复审计明细 `k0_to_k1_repair_detail.jsonl` 的 id 为准；注入候选不计入 ANN 检索命中或召回率。
- `candidate_origin`：注入行 = `source_gold_injected`；其余行 = `ann_recall`（构建审计无该字段，按生成方式推断）。

## 2. 总体分布

分母：每行 `rows` 为该 threshold 全部 split 的总样本数；`unique qid` 为去重后的 `source_query_id` 数量。
`calib` 为 calibration 行数；`effective train` = train 行数 - calibration，不重复计入训练。

| threshold | rows | train | val | test | zh | en | mixed/other | Gold cap | Gold REJECT | K0 | K1 | K2 | K3 | injected | unique qid | calib | effective train |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.82 | 48333 | 36197 | 2407 | 9729 | 20667 | 22053 | 5613 | 44201 | 4132 | 1056 | 18000 | 14955 | 14322 | 152 | 48333 | 400 | 35797 |
| 0.85 | 48333 | 36197 | 2407 | 9729 | 20667 | 22053 | 5613 | 44009 | 4324 | 1938 | 32460 | 10665 | 3270 | 603 | 48333 | 400 | 35797 |
| 0.88 | 48333 | 36197 | 2407 | 9729 | 20667 | 22053 | 5613 | 44061 | 4272 | 2664 | 41420 | 4036 | 213 | 2466 | 48333 | 400 | 35797 |


## 3. K 分布

### 3.1 threshold × split

分母：每个 `threshold × split` 的样本总数。
| threshold | split | rows | K0 | K1 | K2 | K3 | K0% | K1% | K2% | K3% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.82 | train | 36197 | 787 | 13579 | 11174 | 10657 | 2.1742 | 37.5142 | 30.87 | 29.4417 |
| 0.82 | val | 2407 | 54 | 850 | 751 | 752 | 2.2435 | 35.3137 | 31.2007 | 31.2422 |
| 0.82 | test | 9729 | 215 | 3571 | 3030 | 2913 | 2.2099 | 36.7047 | 31.144 | 29.9414 |
| 0.85 | train | 36197 | 1477 | 24289 | 8005 | 2426 | 4.0804 | 67.1022 | 22.1151 | 6.7022 |
| 0.85 | val | 2407 | 86 | 1637 | 535 | 149 | 3.5729 | 68.01 | 22.2268 | 6.1903 |
| 0.85 | test | 9729 | 375 | 6534 | 2125 | 695 | 3.8545 | 67.16 | 21.8419 | 7.1436 |
| 0.88 | train | 36197 | 2013 | 31031 | 2999 | 154 | 5.5612 | 85.7281 | 8.2852 | 0.4254 |
| 0.88 | val | 2407 | 120 | 2091 | 186 | 10 | 4.9855 | 86.8716 | 7.7275 | 0.4155 |
| 0.88 | test | 9729 | 531 | 8298 | 851 | 49 | 5.4579 | 85.2914 | 8.747 | 0.5036 |

完整 `threshold × split × language × K` 见 `k_distribution_by_split_language.csv`；
`threshold × Gold Tool × language × K` 见 `tool_k_distribution.csv`；
`threshold × candidate_origin × K` 见下表。

### 3.2 threshold × candidate_origin × K

| threshold | candidate_origin | K | count | group_total | pct_within_group |
|---|---|---|---|---|---|
| 0.82 | ann_recall | 0 | 1056 | 48181 | 2.1917 |
| 0.82 | ann_recall | 1 | 17848 | 48181 | 37.0436 |
| 0.82 | ann_recall | 2 | 14955 | 48181 | 31.0392 |
| 0.82 | ann_recall | 3 | 14322 | 48181 | 29.7254 |
| 0.82 | source_gold_injected | 0 | 0 | 152 | 0.0 |
| 0.82 | source_gold_injected | 1 | 152 | 152 | 100.0 |
| 0.82 | source_gold_injected | 2 | 0 | 152 | 0.0 |
| 0.82 | source_gold_injected | 3 | 0 | 152 | 0.0 |
| 0.85 | ann_recall | 0 | 1938 | 47730 | 4.0603 |
| 0.85 | ann_recall | 1 | 31857 | 47730 | 66.7442 |
| 0.85 | ann_recall | 2 | 10665 | 47730 | 22.3444 |
| 0.85 | ann_recall | 3 | 3270 | 47730 | 6.851 |
| 0.85 | source_gold_injected | 0 | 0 | 603 | 0.0 |
| 0.85 | source_gold_injected | 1 | 603 | 603 | 100.0 |
| 0.85 | source_gold_injected | 2 | 0 | 603 | 0.0 |
| 0.85 | source_gold_injected | 3 | 0 | 603 | 0.0 |
| 0.88 | ann_recall | 0 | 2664 | 45867 | 5.8081 |
| 0.88 | ann_recall | 1 | 38954 | 45867 | 84.9282 |
| 0.88 | ann_recall | 2 | 4036 | 45867 | 8.7994 |
| 0.88 | ann_recall | 3 | 213 | 45867 | 0.4644 |
| 0.88 | source_gold_injected | 0 | 0 | 2466 | 0.0 |
| 0.88 | source_gold_injected | 1 | 2466 | 2466 | 100.0 |
| 0.88 | source_gold_injected | 2 | 0 | 2466 | 0.0 |
| 0.88 | source_gold_injected | 3 | 0 | 2466 | 0.0 |


## 4. 各 Gold Tool 的中英文分布

比例 A = 该语言在 Tool 内的占比（分母 `denom_tool`）；比例 B = 该 Tool 在该语言全部样本中的占比（分母为 `denom_zh_all_tools` / `denom_en_all_tools`）。
完整数值与分母见 `tool_language_distribution.csv`。

### threshold 0.82

语言总量：zh=20667，en=22053，mixed/other=5613。

| Gold tool | total | zh | en | mixed | zh% in tool | en% in tool | tool% of zh | tool% of en |
|---|---|---|---|---|---|---|---|---|
| cap-add-term | 3711 | 1810 | 1817 | 84 | 48.7739 | 48.9625 | 8.7579 | 8.2392 |
| cap-artifact | 1861 | 565 | 930 | 366 | 30.36 | 49.9731 | 2.7338 | 4.2171 |
| cap-bash | 1853 | 630 | 921 | 302 | 33.9989 | 49.7032 | 3.0483 | 4.1763 |
| cap-create-folder | 4007 | 1753 | 1988 | 266 | 43.7484 | 49.6132 | 8.4821 | 9.0146 |
| cap-edit-file | 1853 | 674 | 923 | 256 | 36.3734 | 49.8111 | 3.2612 | 4.1854 |
| cap-mindmap | 1835 | 900 | 906 | 29 | 49.0463 | 49.3733 | 4.3548 | 4.1083 |
| cap-pdf-extract-text | 1852 | 677 | 727 | 448 | 36.5551 | 39.2549 | 3.2758 | 3.2966 |
| cap-pdf-table-to-text | 1858 | 749 | 729 | 380 | 40.3122 | 39.2357 | 3.6241 | 3.3057 |
| cap-rag-search | 3777 | 1573 | 1688 | 516 | 41.6468 | 44.6916 | 7.6112 | 7.6543 |
| cap-read-document | 1824 | 838 | 730 | 256 | 45.943 | 40.0219 | 4.0548 | 3.3102 |
| cap-read-file | 1860 | 604 | 930 | 326 | 32.4731 | 50.0 | 2.9225 | 4.2171 |
| cap-research | 800 | 387 | 400 | 13 | 48.375 | 50.0 | 1.8726 | 1.8138 |
| cap-slides | 1860 | 728 | 730 | 402 | 39.1398 | 39.2473 | 3.5225 | 3.3102 |
| cap-social-search | 4181 | 1505 | 1837 | 839 | 35.9962 | 43.9369 | 7.2821 | 8.3299 |
| cap-summary | 1854 | 927 | 726 | 201 | 50.0 | 39.1586 | 4.4854 | 3.2921 |
| cap-translate | 3831 | 1909 | 1713 | 209 | 49.8303 | 44.7142 | 9.2369 | 7.7677 |
| cap-vision | 1849 | 898 | 732 | 219 | 48.5668 | 39.589 | 4.3451 | 3.3193 |
| cap-web-search | 3535 | 1598 | 1611 | 326 | 45.2051 | 45.5728 | 7.7321 | 7.3051 |
| REJECT | 4132 | 1942 | 2015 | 175 | 46.999 | 48.7657 | 9.3966 | 9.1371 |


### threshold 0.85

语言总量：zh=20667，en=22053，mixed/other=5613。

| Gold tool | total | zh | en | mixed | zh% in tool | en% in tool | tool% of zh | tool% of en |
|---|---|---|---|---|---|---|---|---|
| cap-add-term | 3706 | 1812 | 1810 | 84 | 48.8937 | 48.8397 | 8.7676 | 8.2075 |
| cap-artifact | 1857 | 565 | 929 | 363 | 30.4254 | 50.0269 | 2.7338 | 4.2126 |
| cap-bash | 1851 | 630 | 919 | 302 | 34.0357 | 49.6488 | 3.0483 | 4.1672 |
| cap-create-folder | 4010 | 1754 | 1988 | 268 | 43.7406 | 49.5761 | 8.487 | 9.0146 |
| cap-edit-file | 1854 | 674 | 924 | 256 | 36.3538 | 49.8382 | 3.2612 | 4.1899 |
| cap-mindmap | 1820 | 899 | 892 | 29 | 49.3956 | 49.011 | 4.3499 | 4.0448 |
| cap-pdf-extract-text | 1828 | 655 | 725 | 448 | 35.8315 | 39.6608 | 3.1693 | 3.2875 |
| cap-pdf-table-to-text | 1858 | 749 | 729 | 380 | 40.3122 | 39.2357 | 3.6241 | 3.3057 |
| cap-rag-search | 3736 | 1563 | 1661 | 512 | 41.8362 | 44.4593 | 7.5628 | 7.5319 |
| cap-read-document | 1799 | 813 | 731 | 255 | 45.1918 | 40.6337 | 3.9338 | 3.3147 |
| cap-read-file | 1860 | 604 | 930 | 326 | 32.4731 | 50.0 | 2.9225 | 4.2171 |
| cap-research | 800 | 387 | 400 | 13 | 48.375 | 50.0 | 1.8726 | 1.8138 |
| cap-slides | 1860 | 728 | 730 | 402 | 39.1398 | 39.2473 | 3.5225 | 3.3102 |
| cap-social-search | 4131 | 1494 | 1797 | 840 | 36.1656 | 43.5004 | 7.2289 | 8.1486 |
| cap-summary | 1852 | 926 | 725 | 201 | 50.0 | 39.1469 | 4.4806 | 3.2875 |
| cap-translate | 3753 | 1869 | 1675 | 209 | 49.8002 | 44.631 | 9.0434 | 7.5953 |
| cap-vision | 1850 | 901 | 732 | 217 | 48.7027 | 39.5676 | 4.3596 | 3.3193 |
| cap-web-search | 3584 | 1667 | 1605 | 312 | 46.5123 | 44.7824 | 8.066 | 7.2779 |
| REJECT | 4324 | 1977 | 2151 | 196 | 45.7216 | 49.7456 | 9.566 | 9.7538 |


### threshold 0.88

语言总量：zh=20667，en=22053，mixed/other=5613。

| Gold tool | total | zh | en | mixed | zh% in tool | en% in tool | tool% of zh | tool% of en |
|---|---|---|---|---|---|---|---|---|
| cap-add-term | 3736 | 1814 | 1838 | 84 | 48.5546 | 49.197 | 8.7773 | 8.3345 |
| cap-artifact | 1857 | 565 | 930 | 362 | 30.4254 | 50.0808 | 2.7338 | 4.2171 |
| cap-bash | 1855 | 630 | 924 | 301 | 33.9623 | 49.8113 | 3.0483 | 4.1899 |
| cap-create-folder | 4033 | 1767 | 1986 | 280 | 43.8135 | 49.2437 | 8.5499 | 9.0056 |
| cap-edit-file | 1848 | 673 | 919 | 256 | 36.4177 | 49.7294 | 3.2564 | 4.1672 |
| cap-mindmap | 1814 | 897 | 888 | 29 | 49.4487 | 48.9526 | 4.3403 | 4.0267 |
| cap-pdf-extract-text | 1802 | 632 | 722 | 448 | 35.0721 | 40.0666 | 3.058 | 3.2739 |
| cap-pdf-table-to-text | 1858 | 749 | 729 | 380 | 40.3122 | 39.2357 | 3.6241 | 3.3057 |
| cap-rag-search | 3690 | 1559 | 1625 | 506 | 42.2493 | 44.0379 | 7.5434 | 7.3686 |
| cap-read-document | 1741 | 784 | 727 | 230 | 45.0316 | 41.7576 | 3.7935 | 3.2966 |
| cap-read-file | 1850 | 597 | 927 | 326 | 32.2703 | 50.1081 | 2.8887 | 4.2035 |
| cap-research | 800 | 387 | 400 | 13 | 48.375 | 50.0 | 1.8726 | 1.8138 |
| cap-slides | 1859 | 727 | 730 | 402 | 39.107 | 39.2684 | 3.5177 | 3.3102 |
| cap-social-search | 4168 | 1498 | 1798 | 872 | 35.9405 | 43.1382 | 7.2483 | 8.1531 |
| cap-summary | 1839 | 914 | 724 | 201 | 49.7009 | 39.3692 | 4.4225 | 3.283 |
| cap-translate | 3780 | 1893 | 1678 | 209 | 50.0794 | 44.3915 | 9.1595 | 7.6089 |
| cap-vision | 1860 | 908 | 733 | 219 | 48.8172 | 39.4086 | 4.3935 | 3.3238 |
| cap-web-search | 3671 | 1702 | 1661 | 308 | 46.3634 | 45.2465 | 8.2354 | 7.5319 |
| REJECT | 4272 | 1971 | 2114 | 187 | 46.1376 | 49.485 | 9.5369 | 9.586 |


## 5. Gold = capability/tool 与 Gold = REJECT

完整 `threshold × split × language × gold_kind` 数值见 `gold_reject_distribution.csv`，每行明确给出 `group_total`。

| threshold | split | Gold capability/tool | pct | Gold REJECT | pct | split total |
|---|---|---|---|---|---|---|
| 0.82 | train | 33128 | 91.5214 | 3069 | 8.4786 | 36197 |
| 0.82 | val | 2187 | 90.86 | 220 | 9.14 | 2407 |
| 0.82 | test | 8886 | 91.3352 | 843 | 8.6648 | 9729 |
| 0.85 | train | 33000 | 91.1678 | 3197 | 8.8322 | 36197 |
| 0.85 | val | 2176 | 90.403 | 231 | 9.597 | 2407 |
| 0.85 | test | 8833 | 90.7904 | 896 | 9.2096 | 9729 |
| 0.88 | train | 33018 | 91.2175 | 3179 | 8.7825 | 36197 |
| 0.88 | val | 2191 | 91.0262 | 216 | 8.9738 | 2407 |
| 0.88 | test | 8852 | 90.9857 | 877 | 9.0143 | 9729 |


## 6. Gold 注入样本专项

注入样本 = 原始 K=0、Gold 为有效 capability 的目标行，原位补入唯一 Gold 卡并保留 REJECT。
注入行的 `candidate_origin=source_gold_injected`、`retrieval_score_available=false`。

| threshold | injected | train | val | test | zh | en | mixed |
|---|---|---|---|---|---|---|---|
| 0.82 | 152 | 107 | 7 | 38 | 124 | 25 | 3 |
| 0.85 | 603 | 421 | 33 | 149 | 411 | 155 | 37 |
| 0.88 | 2466 | 1792 | 147 | 527 | 1205 | 1063 | 198 |

按 Gold Tool 的注入分布（仅列出非零项）：

| threshold | Gold tool | injected | share of injected |
|---|---|---|---|
| 0.82 | cap-create-folder | 7 | 4.6053 |
| 0.82 | cap-rag-search | 1 | 0.6579 |
| 0.82 | cap-translate | 2 | 1.3158 |
| 0.82 | cap-vision | 2 | 1.3158 |
| 0.82 | cap-web-search | 140 | 92.1053 |
| 0.85 | cap-add-term | 23 | 3.8143 |
| 0.85 | cap-create-folder | 23 | 3.8143 |
| 0.85 | cap-edit-file | 1 | 0.1658 |
| 0.85 | cap-rag-search | 9 | 1.4925 |
| 0.85 | cap-read-document | 5 | 0.8292 |
| 0.85 | cap-social-search | 53 | 8.7894 |
| 0.85 | cap-translate | 70 | 11.6086 |
| 0.85 | cap-vision | 13 | 2.1559 |
| 0.85 | cap-web-search | 406 | 67.33 |
| 0.88 | cap-add-term | 178 | 7.2182 |
| 0.88 | cap-artifact | 3 | 0.1217 |
| 0.88 | cap-bash | 6 | 0.2433 |
| 0.88 | cap-create-folder | 59 | 2.3925 |
| 0.88 | cap-edit-file | 6 | 0.2433 |
| 0.88 | cap-mindmap | 33 | 1.3382 |
| 0.88 | cap-pdf-extract-text | 4 | 0.1622 |
| 0.88 | cap-pdf-table-to-text | 1 | 0.0406 |
| 0.88 | cap-rag-search | 172 | 6.9749 |
| 0.88 | cap-read-document | 40 | 1.6221 |
| 0.88 | cap-read-file | 1 | 0.0406 |
| 0.88 | cap-social-search | 394 | 15.9773 |
| 0.88 | cap-summary | 23 | 0.9327 |
| 0.88 | cap-translate | 611 | 24.777 |
| 0.88 | cap-vision | 40 | 1.6221 |
| 0.88 | cap-web-search | 895 | 36.2936 |

修复前后 K 变化：所有注入样本修复前 K=0、修复后 K=1。逐组数值见 `gold_injection_k_shift.csv`。

| threshold | K=0 总数 | K=0 且仅 REJECT 选项 | K=0 且 Gold=Tool | K=0 且 Gold=REJECT |
|---|---|---|---|---|
| 0.82 | 1056 | 1056 | 0 | 1056 |
| 0.85 | 1938 | 1938 | 0 | 1938 |
| 0.88 | 2664 | 2664 | 0 | 2664 |

Gold 候选与索引校验：所有行均检查 `gold_capability_id` 位于候选集中，且 `order[gold_index]` 等于 Gold；错误数见质量审计。

## 7. 数据质量检查

检查为只读，未自动修复。完整机器可读结果见 `data_quality_audit.json`。

| check | scope | errors |
|---|---|---|
| schema_14_keys | 144999 rows | 0 |
| field_types | 144999 rows | 0 |
| split_tag_matches_file | 144999 rows | 0 |
| option_length_equals_order | 144999 rows | 0 |
| exactly_one_REJECT | 144999 rows | 0 |
| option_count_equals_K_plus_1 | 144999 rows | 0 |
| no_duplicate_candidates | 144999 rows | 0 |
| capability_ids_valid | 144999 rows | 0 |
| gold_label_matches_target_kind | 144999 rows | 0 |
| gold_index_and_reject_index | 144999 rows | 0 |
| shuffle_permutation_and_seed | 144999 rows | 0 |
| shuffle_prefix_by_origin | 144999 rows | 0 |
| repair_detail_match | 3221 repaired rows | 0 |
| injected_row_shape | 3221 repaired rows | 0 |
| duplicate_row_ids | 144999 rows | 0 |
| cross_split_leakage | 9 pairs x 3 dimensions x 3 thresholds | 0 |
| calibration_double_counting | 3 thresholds x 400 calibration rows | 0 |

全部检查错误数为 0。

### 7.1 split 泄漏检查

统计 `normalized_query`、`source_query_id`、`query_family_id` 在不同 split 之间的交集。
非零交集按发现项报告，不自动处理。

| threshold | dimension:pair | intersection count | examples |
|---|---|---|---|
| 0.82 | normalized_query:train-val | 0 |  |
| 0.82 | source_query_id:train-val | 0 |  |
| 0.82 | query_family:train-val | 0 |  |
| 0.82 | normalized_query:train-test | 0 |  |
| 0.82 | source_query_id:train-test | 0 |  |
| 0.82 | query_family:train-test | 0 |  |
| 0.82 | normalized_query:val-test | 0 |  |
| 0.82 | source_query_id:val-test | 0 |  |
| 0.82 | query_family:val-test | 0 |  |
| 0.85 | normalized_query:train-val | 0 |  |
| 0.85 | source_query_id:train-val | 0 |  |
| 0.85 | query_family:train-val | 0 |  |
| 0.85 | normalized_query:train-test | 0 |  |
| 0.85 | source_query_id:train-test | 0 |  |
| 0.85 | query_family:train-test | 0 |  |
| 0.85 | normalized_query:val-test | 0 |  |
| 0.85 | source_query_id:val-test | 0 |  |
| 0.85 | query_family:val-test | 0 |  |
| 0.88 | normalized_query:train-val | 0 |  |
| 0.88 | source_query_id:train-val | 0 |  |
| 0.88 | query_family:train-val | 0 |  |
| 0.88 | normalized_query:train-test | 0 |  |
| 0.88 | source_query_id:train-test | 0 |  |
| 0.88 | query_family:train-test | 0 |  |
| 0.88 | normalized_query:val-test | 0 |  |
| 0.88 | source_query_id:val-test | 0 |  |
| 0.88 | query_family:val-test | 0 |  |

### 7.2 calibration

| threshold | calibration rows | unique indices | unique ids | effective train | manifest effective | errors |
|---|---|---|---|---|---|---|
| 0.82 | 400 | 400 | 400 | 35797 | 35797 | none |
| 0.85 | 400 | 400 | 400 | 35797 | 35797 | none |
| 0.88 | 400 | 400 | 400 | 35797 | 35797 | none |


## 8. 输出文件

- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\data_quality_audit.json`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\final_distribution_report.md`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\gold_injection_distribution.csv`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\gold_injection_k_shift.csv`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\gold_reject_distribution.csv`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\k_distribution_by_split_language.csv`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\k_distribution_by_threshold.csv`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\overall_distribution.csv`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\tool_k_distribution.csv`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\tool_language_distribution.csv`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\tool_split_distribution.csv`
- `e:\Work Space\Git Repositories\DeepDive\data\laya_finetune\data_v4_build\distribution_report\tool_summary.csv`


所有统计数据均来自上述正式 JSONL；`construction_audit.jsonl` 仅用于 query family 与修复前证据，不作为当前 K/标签依据。
