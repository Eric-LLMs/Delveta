# Dataset Generation Logic

## Stage A - source load

- Loaded `scripts\laya_finetune\data_v3\laya_tool_selection_v4.csv` with `utf-8-sig`.
- SHA-256 `e39adab214ed26992e26728a9e4c73659d5c924d283454d841abd789187c8445`.
- Assigned stable `source_row_id` = CSV data-row index and `csv_line` = physical line number.
- Parsed `is_action`, `capability_id`, `language`, `parameters`, `source`, `review_status`, `source_file`, `source_row`.

## Stage B - eligible evaluation queries

1. Normalise query text with NFKC, strip, casefold and whitespace collapse.
2. Compare each normalised query against the live standard-query and similar-query corpora.
3. Exclude rows that are a standard query, a similar query, or both.
4. Exclude empty queries, invalid `is_action`, Action rows without a valid capability, No-Action rows carrying a capability, and normalised duplicate queries (keep the first deterministic row).
5. Standard/similar corpus rows remain in the database and are still used for recall; only their use as evaluation samples is prohibited.

No source row is silently dropped: every row is written to `data/sample_manifest.csv` with `eligible` and `exclude_reason`.

## Stage C - split assignment

Stratification key: `source gold label x resolved language`, where the gold label is the source capability id for Action rows and `REJECT` for No-Action rows.

Integer allocation:

- n=1 -> train=1; n=2 -> train=1, test=1;
- n>=3 -> val=max(1, round(0.05n)), test=max(1, round(0.20n)), train=remainder.

This guarantees non-empty test/val for small groups and converges to 75/20/5 for large groups. Actual counts and deviations are reported.

Query families are computed with a fixed heuristic: NFKC/casefold, quoted spans -> `<q>`, digit runs -> `<n>`; queries with >=3 whitespace tokens use the first 5 tokens, otherwise the first 12 characters. Families are assigned as indivisible units by a deterministic greedy allocator that minimises squared deviation from the group targets. A family never crosses train/test/val.

One split assignment is computed once and reused by all three threshold versions.

## Stage D - shared real recall

For every selected query the script calls the live embedding service `http://localhost:18080/embed` (model from `app_settings.embedding_profile`) and then executes the real two-lane SQL/ANN recall against PostgreSQL/pgvector:

- standard lane: `capability_standard_queries`, ANN pool 64;
- similar lane: `capability_similar_queries` joined to its standard query, ANN pool 64;
- no `enabled`, `status`, `intent_kind` or UI-hidden filter is applied (all 18 capabilities participate by explicit experiment rule);
- scores are `1 - cosine_distance`, stored unrounded.

Hits with score >= 0.8 are written to `data/shared_retrieval_evidence.jsonl`. Hits below that floor cannot affect any threshold >= 0.82; this floor is stated explicitly.

Per query and capability, the maximum score is kept. This is the same aggregation the production candidate aggregation uses.

## Stage E - three threshold versions

For each threshold t in 0.82, 0.85, 0.88:

1. Keep capability maxima with `score >= t`.
2. Sort deterministically by `(-score, capability_id)`.
3. Take at most Top 3; never pad.
4. K = number of selected capabilities (REJECT excluded).
5. Build K capability cards + exactly one REJECT card.
6. Label:
   - No-Action source -> REJECT / `no_action`;
   - K=0 -> REJECT / `no_candidates`;
   - gold capability in Top 3 -> that capability / `gold_in_candidates`;
   - otherwise -> REJECT / `gold_not_retrieved` (`gold_below_threshold` or `gold_outside_top3`).
7. Shuffle the K+1 options deterministically with seed prefix `reject-shuffle-v1`, recomputing `gold_index` and `reject_index`.

Label precedence: source No-Action is evaluated before K=0, so a No-Action row that also has K=0 is recorded as `no_action`, not `no_candidates`. The three REJECT provenances are never merged.

## Stage F - V2-format output

`v2_train.jsonl`, `v2_val.jsonl` and `v2_test.jsonl` are the training files. Each row has EXACTLY the frozen V2 14 keys: `id`, `source_query_id`, `split`, `part`, `lang`, `state`, `order`, `options`, `gold_index`, `target_kind`, `gold_capability_id`, `reject_index`, `shuffle`, `src_sha`. No threshold, candidate-count or label-provenance field enters a training row.

The only contract difference from frozen V2 is the option count: `order`/`options` carry K capability cards plus exactly one `REJECT` (K+1 options, 1..4). `part` is `p2` (real recall) for every row. `gold_index` points at the label; a reject row has `gold_capability_id = null`. The per-row shuffle uses the V2 seed prefix `reject-shuffle-v1` over the row id, and REJECT is shuffled with the capability cards exactly as in V2.

Threshold, candidate scores, label provenance and the original source gold are written to `construction_audit.jsonl`, joined by `id`. `{split}.csv` is a human-readable audit export, not a training input.

K=0 rows carry one REJECT option and are also written to `direct_reject.jsonl` in each threshold directory for a separate direct-reject path.

## Stage G - statistics and validation

Statistics are computed from the generated files after all three threshold versions are written. An independent validator re-reads the files, re-derives candidates from shared evidence, checks labels/options/indexes, split consistency, family isolation and schema integrity. Boundary tests are separate from real-sample counts.
