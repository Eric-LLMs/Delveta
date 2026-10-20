# Threshold Comparison Report

All three versions use the same eligible sample list and the same split assignment. Thresholds only change filtering/aggregation/Top-3 and therefore candidates and labels.

## Overall comparison

| Metric | 0.82 | 0.85 | 0.88 | Denominator |
|---|---|---|---|---|
| Rows | 48333 | 48333 | 48333 | same selected sample set |
| Mean K | 1.8771 | 1.3034 | 0.9862 | selected rows |
| Gold Top-1 | 41991 (92.63%) | 41826 (92.26%) | 40804 (90.01%) | Action rows |
| Gold Top-3 | 44049 (97.17%) | 43406 (95.75%) | 41595 (91.75%) | Action rows |
| gold_not_retrieved | 1132 | 1324 | 1272 | Action rows |
| no_candidates | 152 | 603 | 2466 | Action rows |
| no_action | 3000 | 3000 | 3000 | All rows |

## K distribution comparison

| Threshold | K=0 | K=1 | K=2 | K=3 |
|---|---|---|---|---|
| 0.82 | 1208 (2.50%) | 17848 (36.93%) | 14955 (30.94%) | 14322 (29.63%) |
| 0.85 | 2541 (5.26%) | 31857 (65.91%) | 10665 (22.07%) | 3270 (6.77%) |
| 0.88 | 5130 (10.61%) | 38954 (80.60%) | 4036 (8.35%) | 213 (0.44%) |

## Pairwise deltas

### 0.82 vs 0.85

| Metric | Value |
|---|---|
| Same sample rows | 48333 |
| Added capability candidates (sum over queries) | 36314 |
| Queries with at least one added capability candidate | 23771 |
| Added card slots (K_low - K_high, sum) | 27727 |
| Queries with at least one added card | 21968 |
| Gold Top-1 delta (percentage points) | 0.364 |
| Gold Top-3 delta (percentage points) | 1.4184 |
| gold_not_retrieved delta (percentage points) | -0.4235 |
| no_candidates delta (percentage points) | -0.9949 |
| Mean K delta | 0.5737 |

Added candidates by capability:

| Capability | Added count |
|---|---|
| cap-add-term | 1919 |
| cap-artifact | 4179 |
| cap-bash | 1218 |
| cap-create-folder | 2476 |
| cap-edit-file | 2936 |
| cap-mindmap | 1536 |
| cap-pdf-extract-text | 2335 |
| cap-pdf-table-to-text | 922 |
| cap-rag-search | 3374 |
| cap-read-document | 1557 |
| cap-read-file | 3071 |
| cap-research | 3228 |
| cap-slides | 1578 |
| cap-social-search | 912 |
| cap-summary | 2046 |
| cap-translate | 1069 |
| cap-vision | 843 |
| cap-web-search | 1115 |

Added candidates by language:

| Language | Added count |
|---|---|
| en | 17874 |
| mixed | 3614 |
| zh | 14826 |

### 0.82 vs 0.88

| Metric | Value |
|---|---|
| Same sample rows | 48333 |
| Added capability candidates (sum over queries) | 52429 |
| Queries with at least one added capability candidate | 29033 |
| Added card slots (K_low - K_high, sum) | 43059 |
| Queries with at least one added card | 28897 |
| Gold Top-1 delta (percentage points) | 2.6184 |
| Gold Top-3 delta (percentage points) | 5.4133 |
| gold_not_retrieved delta (percentage points) | -0.3088 |
| no_candidates delta (percentage points) | -5.1044 |
| Mean K delta | 0.8909 |

Added candidates by capability:

| Capability | Added count |
|---|---|
| cap-add-term | 2481 |
| cap-artifact | 5842 |
| cap-bash | 1822 |
| cap-create-folder | 3192 |
| cap-edit-file | 4374 |
| cap-mindmap | 1925 |
| cap-pdf-extract-text | 3166 |
| cap-pdf-table-to-text | 1054 |
| cap-rag-search | 5839 |
| cap-read-document | 2218 |
| cap-read-file | 4418 |
| cap-research | 4944 |
| cap-slides | 1992 |
| cap-social-search | 1394 |
| cap-summary | 2908 |
| cap-translate | 1755 |
| cap-vision | 1120 |
| cap-web-search | 1985 |

Added candidates by language:

| Language | Added count |
|---|---|
| en | 25852 |
| mixed | 5191 |
| zh | 21386 |

### 0.85 vs 0.88

| Metric | Value |
|---|---|
| Same sample rows | 48333 |
| Added capability candidates (sum over queries) | 16115 |
| Queries with at least one added capability candidate | 12543 |
| Added card slots (K_low - K_high, sum) | 15332 |
| Queries with at least one added card | 12465 |
| Gold Top-1 delta (percentage points) | 2.2544 |
| Gold Top-3 delta (percentage points) | 3.9949 |
| gold_not_retrieved delta (percentage points) | 0.1147 |
| no_candidates delta (percentage points) | -4.1096 |
| Mean K delta | 0.3172 |

Added candidates by capability:

| Capability | Added count |
|---|---|
| cap-add-term | 562 |
| cap-artifact | 1663 |
| cap-bash | 604 |
| cap-create-folder | 716 |
| cap-edit-file | 1438 |
| cap-mindmap | 389 |
| cap-pdf-extract-text | 831 |
| cap-pdf-table-to-text | 132 |
| cap-rag-search | 2465 |
| cap-read-document | 661 |
| cap-read-file | 1347 |
| cap-research | 1716 |
| cap-slides | 414 |
| cap-social-search | 482 |
| cap-summary | 862 |
| cap-translate | 686 |
| cap-vision | 277 |
| cap-web-search | 870 |

Added candidates by language:

| Language | Added count |
|---|---|
| en | 7978 |
| mixed | 1577 |
| zh | 6560 |

## Data bias and representativeness

- The 50-row smoke sample is coverage-driven, not random: it takes one Chinese and one English row for every capability plus mixed-language and split-exercise rows. It therefore does NOT mirror source frequencies and must not be used to estimate production prevalence.
- Added candidates at lower thresholds by language:

| Pair | en | mixed | zh | Total |
|---|---|---|---|---|
| 0.82 vs 0.85 | 17874 | 3614 | 14826 | 36314 |
| 0.82 vs 0.88 | 25852 | 5191 | 21386 | 52429 |
| 0.85 vs 0.88 | 7978 | 1577 | 6560 | 16115 |

- Source capability/language/source frequencies are reported in `source_audit.md`; the full run's selected-sample frequencies belong in `dataset_distribution_report.md`.

## Findings

- Lower thresholds mechanically add capability candidates and can increase K, but the smoke sample is not large enough to treat these deltas as a production threshold recommendation.
- Any gold-recall gain must be weighed against added REJECT ambiguity, card length and model cost; no production threshold is changed here.
