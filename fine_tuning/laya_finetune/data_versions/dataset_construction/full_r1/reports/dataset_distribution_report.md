# Dataset Distribution Report

Run: `full_r1`; mode: `full`; selected evaluation rows: 48333.

All percentages name their denominator. K counts only capability candidates; REJECT is never counted in K.

## Source eligibility

- Source rows: 49241
- Eligible rows: 48333
- Rows used in this run: 48333 (full).
- Excluded rows: 908

## Overall K distribution

| Threshold | K | Samples | Share (denominator = all rows at this threshold) |
|---|---|---|---|
| 0.82 | 0 | 1208 | 2.50% |
| 0.82 | 1 | 17848 | 36.93% |
| 0.82 | 2 | 14955 | 30.94% |
| 0.82 | 3 | 14322 | 29.63% |
| 0.85 | 0 | 2541 | 5.26% |
| 0.85 | 1 | 31857 | 65.91% |
| 0.85 | 2 | 10665 | 22.07% |
| 0.85 | 3 | 3270 | 6.77% |
| 0.88 | 0 | 5130 | 10.61% |
| 0.88 | 1 | 38954 | 80.60% |
| 0.88 | 2 | 4036 | 8.35% |
| 0.88 | 3 | 213 | 0.44% |

## K distribution by split

| Threshold | Split | K | Samples | Share (denominator = split rows) |
|---|---|---|---|---|
| 0.82 | train | 0 | 894 | 2.47% |
| 0.82 | train | 1 | 13472 | 37.22% |
| 0.82 | train | 2 | 11174 | 30.87% |
| 0.82 | train | 3 | 10657 | 29.44% |
| 0.82 | test | 0 | 253 | 2.60% |
| 0.82 | test | 1 | 3533 | 36.31% |
| 0.82 | test | 2 | 3030 | 31.14% |
| 0.82 | test | 3 | 2913 | 29.94% |
| 0.82 | val | 0 | 61 | 2.53% |
| 0.82 | val | 1 | 843 | 35.02% |
| 0.82 | val | 2 | 751 | 31.20% |
| 0.82 | val | 3 | 752 | 31.24% |
| 0.85 | train | 0 | 1898 | 5.24% |
| 0.85 | train | 1 | 23868 | 65.94% |
| 0.85 | train | 2 | 8005 | 22.12% |
| 0.85 | train | 3 | 2426 | 6.70% |
| 0.85 | test | 0 | 524 | 5.39% |
| 0.85 | test | 1 | 6385 | 65.63% |
| 0.85 | test | 2 | 2125 | 21.84% |
| 0.85 | test | 3 | 695 | 7.14% |
| 0.85 | val | 0 | 119 | 4.94% |
| 0.85 | val | 1 | 1604 | 66.64% |
| 0.85 | val | 2 | 535 | 22.23% |
| 0.85 | val | 3 | 149 | 6.19% |
| 0.88 | train | 0 | 3805 | 10.51% |
| 0.88 | train | 1 | 29239 | 80.78% |
| 0.88 | train | 2 | 2999 | 8.29% |
| 0.88 | train | 3 | 154 | 0.43% |
| 0.88 | test | 0 | 1058 | 10.87% |
| 0.88 | test | 1 | 7771 | 79.87% |
| 0.88 | test | 2 | 851 | 8.75% |
| 0.88 | test | 3 | 49 | 0.50% |
| 0.88 | val | 0 | 267 | 11.09% |
| 0.88 | val | 1 | 1944 | 80.76% |
| 0.88 | val | 2 | 186 | 7.73% |
| 0.88 | val | 3 | 10 | 0.42% |

## Labels and REJECT provenance

| Threshold | Label / reason | Count | Share (denominator = all rows) |
|---|---|---|---|
| 0.82 | gold_in_candidates | 44049 | 91.14% |
| 0.82 | gold_not_retrieved | 1132 | 2.34% |
| 0.82 | no_action | 3000 | 6.21% |
| 0.82 | no_candidates | 152 | 0.31% |
| 0.85 | gold_in_candidates | 43406 | 89.81% |
| 0.85 | gold_not_retrieved | 1324 | 2.74% |
| 0.85 | no_action | 3000 | 6.21% |
| 0.85 | no_candidates | 603 | 1.25% |
| 0.88 | gold_in_candidates | 41595 | 86.06% |
| 0.88 | gold_not_retrieved | 1272 | 2.63% |
| 0.88 | no_action | 3000 | 6.21% |
| 0.88 | no_candidates | 2466 | 5.10% |

`no_action` is source No-Action; `no_candidates` is an Action query with K=0; `gold_not_retrieved` is an Action query with K>0 whose gold capability is not in Top 3. The three are never merged.

## Gold retrieval metrics (Action rows only)

| Threshold | Action rows | Gold in candidates | Gold Top-1 | Gold Top-3 | gold_not_retrieved | no_candidates |
|---|---|---|---|---|---|---|
| 0.82 | 45333 | 44326 (97.78%) | 41991 (92.63%) | 44049 (97.17%) | 1132 (2.50%) | 152 (0.34%) |
| 0.85 | 45333 | 43453 (95.85%) | 41826 (92.26%) | 43406 (95.75%) | 1324 (2.92%) | 603 (1.33%) |
| 0.88 | 45333 | 41596 (91.76%) | 40804 (90.01%) | 41595 (91.75%) | 1272 (2.81%) | 2466 (5.44%) |

Denominator for every percentage in this table is the number of Action rows at that threshold.

## Per-capability distribution

For each threshold, denominators are the source Action rows of that capability (No-Action is reported as a separate `REJECT` bucket).

### Threshold 0.82

| Gold capability | Source rows | zh | en | mixed | Train | Test | Val | K=0 | K=1 | K=2 | K=3 | Gold Top-3 | gold_not_retrieved | REJECT label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cap-add-term | 3889 | 1864 | 1941 | 84 | 2918 | 777 | 194 | 0 | 1355 | 1023 | 1511 | 3711 | 178 | 178 |
| cap-artifact | 1863 | 565 | 931 | 367 | 1396 | 375 | 92 | 0 | 170 | 975 | 718 | 1861 | 2 | 2 |
| cap-bash | 1862 | 630 | 930 | 302 | 1396 | 372 | 94 | 0 | 846 | 888 | 128 | 1853 | 9 | 9 |
| cap-create-folder | 4069 | 1782 | 1988 | 299 | 3028 | 817 | 224 | 7 | 1768 | 1612 | 682 | 4000 | 62 | 69 |
| cap-edit-file | 1863 | 676 | 931 | 256 | 1397 | 372 | 94 | 0 | 316 | 580 | 967 | 1853 | 10 | 10 |
| cap-mindmap | 1858 | 900 | 929 | 29 | 1391 | 377 | 90 | 0 | 1194 | 422 | 242 | 1835 | 23 | 23 |
| cap-pdf-extract-text | 1857 | 681 | 728 | 448 | 1392 | 374 | 91 | 0 | 611 | 544 | 702 | 1852 | 5 | 5 |
| cap-pdf-table-to-text | 1858 | 749 | 729 | 380 | 1391 | 377 | 90 | 0 | 992 | 730 | 136 | 1858 | 0 | 0 |
| cap-rag-search | 3890 | 1616 | 1742 | 532 | 2915 | 782 | 193 | 1 | 862 | 1133 | 1894 | 3776 | 113 | 114 |
| cap-read-document | 1864 | 877 | 731 | 256 | 1395 | 377 | 92 | 0 | 297 | 649 | 918 | 1824 | 40 | 40 |
| cap-read-file | 1864 | 607 | 931 | 326 | 1396 | 377 | 91 | 0 | 0 | 237 | 1627 | 1860 | 4 | 4 |
| cap-research | 800 | 387 | 400 | 13 | 601 | 159 | 40 | 0 | 113 | 643 | 44 | 800 | 0 | 0 |
| cap-slides | 1860 | 728 | 730 | 402 | 1393 | 374 | 93 | 0 | 1165 | 474 | 221 | 1860 | 0 | 0 |
| cap-social-search | 4391 | 1520 | 1941 | 930 | 3290 | 887 | 214 | 0 | 2880 | 760 | 751 | 4181 | 210 | 210 |
| cap-summary | 1858 | 928 | 729 | 201 | 1392 | 375 | 91 | 0 | 780 | 557 | 521 | 1854 | 4 | 4 |
| cap-translate | 3932 | 1958 | 1765 | 209 | 2944 | 795 | 193 | 2 | 1291 | 1400 | 1239 | 3829 | 101 | 103 |
| cap-vision | 1868 | 915 | 734 | 219 | 1400 | 378 | 90 | 2 | 1212 | 450 | 204 | 1847 | 19 | 21 |
| cap-web-search | 3887 | 1784 | 1743 | 360 | 2912 | 784 | 191 | 140 | 1187 | 1303 | 1257 | 3395 | 352 | 492 |
| REJECT | 3000 | 1500 | 1500 | 0 | 2250 | 600 | 150 | 1056 | 809 | 575 | 560 | 0 | 0 | 3000 |

### Threshold 0.85

| Gold capability | Source rows | zh | en | mixed | Train | Test | Val | K=0 | K=1 | K=2 | K=3 | Gold Top-3 | gold_not_retrieved | REJECT label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cap-add-term | 3889 | 1864 | 1941 | 84 | 2918 | 777 | 194 | 23 | 2637 | 819 | 410 | 3683 | 183 | 206 |
| cap-artifact | 1863 | 565 | 931 | 367 | 1396 | 375 | 92 | 0 | 852 | 934 | 77 | 1857 | 6 | 6 |
| cap-bash | 1862 | 630 | 930 | 302 | 1396 | 372 | 94 | 0 | 1633 | 206 | 23 | 1851 | 11 | 11 |
| cap-create-folder | 4069 | 1782 | 1988 | 299 | 3028 | 817 | 224 | 23 | 3215 | 728 | 103 | 3987 | 59 | 82 |
| cap-edit-file | 1863 | 676 | 931 | 256 | 1397 | 372 | 94 | 1 | 1302 | 471 | 89 | 1853 | 9 | 10 |
| cap-mindmap | 1858 | 900 | 929 | 29 | 1391 | 377 | 90 | 0 | 1600 | 240 | 18 | 1820 | 38 | 38 |
| cap-pdf-extract-text | 1857 | 681 | 728 | 448 | 1392 | 374 | 91 | 0 | 1273 | 470 | 114 | 1828 | 29 | 29 |
| cap-pdf-table-to-text | 1858 | 749 | 729 | 380 | 1391 | 377 | 90 | 0 | 1623 | 228 | 7 | 1858 | 0 | 0 |
| cap-rag-search | 3890 | 1616 | 1742 | 532 | 2915 | 782 | 193 | 9 | 1631 | 1534 | 716 | 3727 | 154 | 163 |
| cap-read-document | 1864 | 877 | 731 | 256 | 1395 | 377 | 92 | 5 | 955 | 662 | 242 | 1794 | 65 | 70 |
| cap-read-file | 1864 | 607 | 931 | 326 | 1396 | 377 | 91 | 0 | 203 | 960 | 701 | 1860 | 4 | 4 |
| cap-research | 800 | 387 | 400 | 13 | 601 | 159 | 40 | 0 | 379 | 415 | 6 | 800 | 0 | 0 |
| cap-slides | 1860 | 728 | 730 | 402 | 1393 | 374 | 93 | 0 | 1666 | 173 | 21 | 1860 | 0 | 0 |
| cap-social-search | 4391 | 1520 | 1941 | 930 | 3290 | 887 | 214 | 53 | 3637 | 530 | 171 | 4078 | 260 | 313 |
| cap-summary | 1858 | 928 | 729 | 201 | 1392 | 375 | 91 | 0 | 1387 | 358 | 113 | 1852 | 6 | 6 |
| cap-translate | 3932 | 1958 | 1765 | 209 | 2944 | 795 | 193 | 70 | 2942 | 781 | 139 | 3683 | 179 | 249 |
| cap-vision | 1868 | 915 | 734 | 219 | 1400 | 378 | 90 | 13 | 1771 | 80 | 4 | 1837 | 18 | 31 |
| cap-web-search | 3887 | 1784 | 1743 | 360 | 2912 | 784 | 191 | 406 | 2318 | 872 | 291 | 3178 | 303 | 709 |
| REJECT | 3000 | 1500 | 1500 | 0 | 2250 | 600 | 150 | 1938 | 833 | 204 | 25 | 0 | 0 | 3000 |

### Threshold 0.88

| Gold capability | Source rows | zh | en | mixed | Train | Test | Val | K=0 | K=1 | K=2 | K=3 | Gold Top-3 | gold_not_retrieved | REJECT label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cap-add-term | 3889 | 1864 | 1941 | 84 | 2918 | 777 | 194 | 178 | 3416 | 276 | 19 | 3558 | 153 | 331 |
| cap-artifact | 1863 | 565 | 931 | 367 | 1396 | 375 | 92 | 3 | 1563 | 297 | 0 | 1854 | 6 | 9 |
| cap-bash | 1862 | 630 | 930 | 302 | 1396 | 372 | 94 | 6 | 1810 | 44 | 2 | 1849 | 7 | 13 |
| cap-create-folder | 4069 | 1782 | 1988 | 299 | 3028 | 817 | 224 | 59 | 3815 | 193 | 2 | 3974 | 36 | 95 |
| cap-edit-file | 1863 | 676 | 931 | 256 | 1397 | 372 | 94 | 6 | 1780 | 76 | 1 | 1842 | 15 | 21 |
| cap-mindmap | 1858 | 900 | 929 | 29 | 1391 | 377 | 90 | 33 | 1781 | 44 | 0 | 1781 | 44 | 77 |
| cap-pdf-extract-text | 1857 | 681 | 728 | 448 | 1392 | 374 | 91 | 4 | 1617 | 232 | 4 | 1798 | 55 | 59 |
| cap-pdf-table-to-text | 1858 | 749 | 729 | 380 | 1391 | 377 | 90 | 1 | 1848 | 9 | 0 | 1857 | 0 | 1 |
| cap-rag-search | 3890 | 1616 | 1742 | 532 | 2915 | 782 | 193 | 172 | 2495 | 1135 | 88 | 3518 | 200 | 372 |
| cap-read-document | 1864 | 877 | 731 | 256 | 1395 | 377 | 92 | 40 | 1461 | 330 | 33 | 1701 | 123 | 163 |
| cap-read-file | 1864 | 607 | 931 | 326 | 1396 | 377 | 91 | 1 | 1245 | 591 | 27 | 1849 | 14 | 15 |
| cap-research | 800 | 387 | 400 | 13 | 601 | 159 | 40 | 0 | 727 | 73 | 0 | 800 | 0 | 0 |
| cap-slides | 1860 | 728 | 730 | 402 | 1393 | 374 | 93 | 0 | 1822 | 38 | 0 | 1859 | 1 | 1 |
| cap-social-search | 4391 | 1520 | 1941 | 930 | 3290 | 887 | 214 | 394 | 3813 | 181 | 3 | 3774 | 223 | 617 |
| cap-summary | 1858 | 928 | 729 | 201 | 1392 | 375 | 91 | 23 | 1724 | 108 | 3 | 1816 | 19 | 42 |
| cap-translate | 3932 | 1958 | 1765 | 209 | 2944 | 795 | 193 | 611 | 3212 | 107 | 2 | 3169 | 152 | 763 |
| cap-vision | 1868 | 915 | 734 | 219 | 1400 | 378 | 90 | 40 | 1820 | 8 | 0 | 1820 | 8 | 48 |
| cap-web-search | 3887 | 1784 | 1743 | 360 | 2912 | 784 | 191 | 895 | 2677 | 286 | 29 | 2776 | 216 | 1111 |
| REJECT | 3000 | 1500 | 1500 | 0 | 2250 | 600 | 150 | 2664 | 328 | 8 | 0 | 0 | 0 | 3000 |

## Per-language distribution

### Threshold 0.82

| Language | Rows | K=0 | K=1 | K=2 | K=3 | Gold Top-3 (Action) | gold_not_retrieved |
|---|---|---|---|---|---|---|---|
| en | 22053 | 342 | 7849 | 7267 | 6595 | 20013 (97.37%) | 515 |
| mixed | 5613 | 3 | 2402 | 1450 | 1758 | 5435 (96.83%) | 175 |
| zh | 20667 | 863 | 7597 | 6238 | 5969 | 18601 (97.05%) | 442 |

### Threshold 0.85

| Language | Rows | K=0 | K=1 | K=2 | K=3 | Gold Top-3 (Action) | gold_not_retrieved |
|---|---|---|---|---|---|---|---|
| en | 22053 | 1024 | 14534 | 5035 | 1460 | 19747 (96.08%) | 651 |
| mixed | 5613 | 37 | 3935 | 1222 | 419 | 5380 (95.85%) | 196 |
| zh | 20667 | 1480 | 13388 | 4408 | 1391 | 18279 (95.37%) | 477 |

### Threshold 0.88

| Language | Rows | K=0 | K=1 | K=2 | K=3 | Gold Top-3 (Action) | gold_not_retrieved |
|---|---|---|---|---|---|---|---|
| en | 22053 | 2364 | 17986 | 1630 | 73 | 18876 (91.84%) | 614 |
| mixed | 5613 | 198 | 4748 | 644 | 23 | 5228 (93.14%) | 187 |
| zh | 20667 | 2568 | 16220 | 1762 | 117 | 17491 (91.26%) | 471 |

## Per-source distribution

### Threshold 0.82

| Source | Rows | Action rows | K=0 | K=1 | K=2 | K=3 | Gold Top-3 |
|---|---|---|---|---|---|---|---|
| laya_rebuilt_v3 | 18000 | 17000 | 31 | 6439 | 6330 | 5200 | 16939 |
| query_list_finetune_v1 | 28333 | 28333 | 152 | 10790 | 8397 | 8994 | 27110 |
| synthetic_original | 2000 | 0 | 1025 | 619 | 228 | 128 | 0 |

### Threshold 0.85

| Source | Rows | Action rows | K=0 | K=1 | K=2 | K=3 | Gold Top-3 |
|---|---|---|---|---|---|---|---|
| laya_rebuilt_v3 | 18000 | 17000 | 335 | 12380 | 4204 | 1081 | 16885 |
| query_list_finetune_v1 | 28333 | 28333 | 596 | 19133 | 6422 | 2182 | 26521 |
| synthetic_original | 2000 | 0 | 1610 | 344 | 39 | 7 | 0 |

### Threshold 0.88

| Source | Rows | Action rows | K=0 | K=1 | K=2 | K=3 | Gold Top-3 |
|---|---|---|---|---|---|---|---|
| laya_rebuilt_v3 | 18000 | 17000 | 912 | 15585 | 1429 | 74 | 16651 |
| query_list_finetune_v1 | 28333 | 28333 | 2324 | 23264 | 2606 | 139 | 24944 |
| synthetic_original | 2000 | 0 | 1894 | 105 | 1 | 0 | 0 |

## Split ratio check

Targets are per `gold label x language` group. Because the smoke run has mostly singleton groups, the global smoke ratio deviates from 75/20/5; per-group targets and actual counts are listed in `reports/split_integrity_report.md`.

| Threshold | Train | Test | Val | Actual ratio |
|---|---|---|---|---|
| 0.82 | 36197 | 9729 | 2407 | 74.89% / 20.13% / 4.98% |
| 0.85 | 36197 | 9729 | 2407 | 74.89% / 20.13% / 4.98% |
| 0.88 | 36197 | 9729 | 2407 | 74.89% / 20.13% / 4.98% |

