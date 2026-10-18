# Source Audit

- Source file: `scripts\laya_finetune\data_v3\laya_tool_selection_v4.csv`
- SHA-256: `e39adab214ed26992e26728a9e4c73659d5c924d283454d841abd789187c8445`
- Bytes: 11996876
- Rows: 49241
- Fields: query, is_action, tool_name, capability_id, language, parameters, parameter_status, source, review_status, source_file, source_row
- Action rows: 46241; No-Action/reject rows: 3000
- Parameter JSON parse errors: 0

## Capability distribution (source labels)

| Label | Rows | Share |
|---|---|---|
| REJECT | 3000 | 6.09% |
| cap-add-term | 3939 | 8.00% |
| cap-artifact | 1914 | 3.89% |
| cap-bash | 1912 | 3.88% |
| cap-create-folder | 4121 | 8.37% |
| cap-edit-file | 1914 | 3.89% |
| cap-mindmap | 1906 | 3.87% |
| cap-pdf-extract-text | 1904 | 3.87% |
| cap-pdf-table-to-text | 1906 | 3.87% |
| cap-rag-search | 3941 | 8.00% |
| cap-read-document | 1916 | 3.89% |
| cap-read-file | 1916 | 3.89% |
| cap-research | 852 | 1.73% |
| cap-slides | 1909 | 3.88% |
| cap-social-search | 4445 | 9.03% |
| cap-summary | 1906 | 3.87% |
| cap-translate | 3981 | 8.08% |
| cap-vision | 1923 | 3.91% |
| cap-web-search | 3936 | 7.99% |

## Language

| Raw language | Rows |
|---|---|
| (blank) | 29241 |
| en | 10000 |
| zh | 10000 |

| Resolved language | Rows | Share |
|---|---|---|
| en | 22493 | 45.68% |
| mixed | 5735 | 11.65% |
| zh | 21013 | 42.67% |

Resolution rule: explicit `zh`/`en` is kept; blank language is derived from script (Han only -> zh, Latin only -> en, Han+Latin -> mixed, neither -> unknown). Mixed/unknown are never folded into zh/en.

## Source and review status

| Source | Rows |
|---|---|
| laya_rebuilt_v3 | 18000 |
| query_list_finetune_v1 | 29241 |
| synthetic_original | 2000 |

| Review status | Rows |
|---|---|
| (blank) | 29241 |
| rule_checked_pending_human_review | 2000 |
| synthetic_needs_validation | 18000 |

## No-Action / REJECT provenance

No row in the source is marked as a human-confirmed real No-Action. The reject rows split into:

| Review status | Rows |
|---|---|
| rule_checked_pending_human_review | 2000 |
| synthetic_needs_validation | 1000 |

Synthetic/rule-checked rows are reported separately and are never described as real user No-Action.

## Standard / similar corpus exclusion

- Corpus: 36 standard + 872 similar rows (36 + 872 embedded).
- Corpus SHA-256: `d7d8041c3a9bc153636b70ef60adea82cee6c591ab98143ceaab22d412b75c94`
- Source rows matching a standard query: 36
- Source rows matching a similar query: 872
- Source rows matching both: 0
- Unique standard corpus rows matched in source: 36 / 36
- Unique similar corpus rows matched in source: 872 / 872
- Eligible evaluation rows after all exclusions: 48333

| Exclusion reason | Rows |
|---|---|
| standard_query | 36 |
| standard_query_similar | 872 |

Exclusion reasons are not additive when several apply to one row; the unique excluded total is 908.

No parent/derived-record ID exists in the source CSV or the live corpus tables, so only exact/normalised equality can be used to identify standard/similar records. Rows that are merely derived from a corpus query (but not textually identical) are not silently removed; they remain eligible and are flagged as a limitation.

## Database / embedding environment

- DB: `PostgreSQL 16.14 (Debian 16.14-1.pgdg12+1) on x86_64-pc-linux-gnu, compiled by gcc (Debian 12.2.0-14+deb12u1) 12.2.0, 64-bit`
- Embedding profile: `{"dim": 1024, "model": "eric-ml-nlp/Delveta-BGE-M3-v1#INT8", "base_url": "http://localhost:18080", "provider": "tei"}`
- Embedding service: `http://localhost:18080/embed`
- Model SHA-256: `0e0ada7fec9367bca6d816b86953e4f9fe8531ebd4e8251f6de0721c0d2b1f0d`
- ANN pool per lane: 64
- Thresholds: 0.82, 0.85, 0.88

## Known source-quality caveats

- The source `language` column is blank for 29,241 rows; those rows are handled by the explicit resolution rule above.
- 1,000 reject rows are `synthetic_needs_validation`; 2,000 are `rule_checked_pending_human_review`. Neither category is a human-confirmed real No-Action.
- Source labels include 18 capabilities, including `cap-research`, which production chat-plane filtering would exclude. This dataset experiment deliberately includes all 18 capabilities.
