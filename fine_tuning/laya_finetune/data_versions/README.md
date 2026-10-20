# LayaChoice V4 data versions

> **Dataset download:** [**eric-ml-nlp/Delveta-LayaChoice-v2-Data**](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data)
>
> The training data is hosted on Hugging Face. This GitHub directory holds the version
> documentation, the construction chain, audits and checksum records. The per-version
> JSONL bundles are **not** stored in this directory — fetch them from the dataset repo
> linked above.

LayaChoice-v2 is a multiple-choice dataset for **capability selection**: given a user
request and a set of candidate capability cards, choose the card that should handle the
request, or choose the single `REJECT` card when no candidate is correct. It is built for
fine-tuning and evaluating a Laya-style non-autoregressive decision model.

## 1. Versions

All three versions contain the same 48,333 source queries, the same `id` and
`source_query_id` sets, the same split assignment and the same 400 calibration ids. They
differ only in the similarity threshold that decides which capabilities become candidates.

| Version | Hugging Face location | Role | Threshold |
|---|---|---|---:|
| `recall_threshold_082` | repository root (`train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json`) | **default, formal training bundle** | 0.82 |
| `recall_threshold_085` | `data_versions/recall_threshold_085/` | historical comparison only | 0.85 |
| `recall_threshold_088` | `data_versions/recall_threshold_088/` | historical comparison only | 0.88 |

`recall_threshold_082` is the bundle used by the formal training run
(`layachoice_v2_082_r1`) and by the deployed model. `recall_threshold_085` and
`recall_threshold_088` are frozen alternatives kept for comparison; they are not alternate
training runs.

A lower threshold admits more candidates; a higher threshold admits fewer and produces more
`REJECT` labels.

| Version | Threshold | Gold = capability | Gold = `REJECT` | K=0 / 1 / 2 / 3 | Repaired K=0→K=1 | Mean K |
|---|---:|---:|---:|---|---:|---:|
| `recall_threshold_082` (default) | 0.82 | 44,201 | 4,132 | 1,056 / 18,000 / 14,955 / 14,322 | 152 | 1.880206 |
| `recall_threshold_085` | 0.85 | 44,009 | 4,324 | 1,938 / 32,460 / 10,665 / 3,270 | 603 | 1.315871 |
| `recall_threshold_088` | 0.88 | 44,061 | 4,272 | 2,664 / 41,420 / 4,036 / 213 | 2,466 | 1.037200 |

`K` counts capability cards only. `K=0` rows contain the `REJECT` card alone (one option).

## 2. Scale and splits

| | |
|---|---|
| Rows per version | 48,333 (train 36,197 / validation 2,407 / test 9,729) |
| Capabilities | 18 |
| Languages | `en` 22,053 / `zh` 20,667 / `mixed` 5,613 |
| Option count | 1 to 4 (`K + 1`) |
| Training rows | 36,197, of which 400 are held out for calibration |
| Effective gradient rows | 35,797 |
| Card view | `B_noprov` (production card without query examples and provenance) |

| Split | Rows | Share | `en` | `zh` | `mixed` | Type |
|---|---:|---:|---:|---:|---:|---|
| train | 36,197 | 74.89% | 16,499 | 15,505 | 4,193 | gradient training, minus calibration |
| val | 2,407 | 4.98% | 1,116 | 1,030 | 261 | checkpoint selection |
| test | 9,729 | 20.13% | 4,438 | 4,132 | 1,159 | final benchmark only |
| total | 48,333 | 100% | 22,053 | 20,667 | 5,613 | |

Splits are stratified by source gold label × resolved language and allocated with
family-aware indivisible units. The target is about 75/20/5, with minimum representation for
small groups:

- group size 1: all rows in train;
- group size 2: one train, one test;
- group size 3 or more: at least one validation and one test row, converging to 75/20/5 for
  large groups.

Rows from the same query family never cross splits. One split assignment is computed once
and reused by all three thresholds.

**Calibration.** The first 400 rows selected from the final post-shuffle train order (seed
`20260922`, rule `min(400, n_train // 10)`) form the calibration split. Those rows remain in
`train.jsonl` but are excluded from gradient updates, leaving 35,797 effective training
rows. `manifest.json` records both the indices and the ids. Calibration is used for
temperature fitting, not for checkpoint selection.

The test split is never used for training or checkpoint selection. Checkpoint selection uses
validation top-1 only.

## 3. Row schema

Every row has exactly 14 keys. The model input is `{state, order, options}`; the remaining
keys are metadata.

| Key | Type | Meaning |
|---|---|---|
| `id` | string | Stable row id, `<source_query_id>-p2` |
| `source_query_id` | string | Stable source-record id, `<source_file>:<source_row>:<csv_row_index>` |
| `split` | string | `train`, `val` or `test`; equals the containing file |
| `part` | string | Always `p2` (real recall) |
| `lang` | string | Resolved language: `en`, `zh` or `mixed` |
| `state` | string | The user request |
| `order` | list[string] | Stored option order; each element is a capability id or `REJECT` |
| `options` | list[string] | Rendered card texts, aligned one-to-one with `order` |
| `gold_index` | integer | Index of the correct option in `order` / `options` |
| `target_kind` | string | `capability` or `reject` |
| `gold_capability_id` | string or null | Gold capability id; `null` for `REJECT` rows |
| `reject_index` | integer | Index of the `REJECT` option; always present |
| `shuffle` | object | `{seed_prefix, permutation}` used to place the options |
| `src_sha` | string | SHA-256 of the frozen source CSV |

Two invariants hold for every row:

1. `order` and `options` have the same length and are positionally aligned.
2. There is exactly one `REJECT` in `order`; its text in `options` is the frozen `REJECT`
   card; `reject_index` points at it.

For a capability row, `order[gold_index] == gold_capability_id`. For a `REJECT` row,
`gold_index == reject_index`, `order[gold_index] == "REJECT"` and `gold_capability_id` is
`null`. The stored `order` is the **shuffled** order; it does not encode retrieval rank.

The full field contract and validation rules are in
[DATASET_SCHEMA.md](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data/blob/main/DATASET_SCHEMA.md)
in the dataset repository.

## 4. Construction pipeline

The construction reuses the production card renderer and the production recall SQL shape,
and deliberately disables production capability-availability filters so that all 18
capabilities participate.

```mermaid
flowchart TD
    A["Source CSV<br/>49,241 rows"] --> B["Eligibility and deduplication<br/>48,333 rows"]
    B --> C["Family-aware split<br/>train 36,197 / val 2,407 / test 9,729"]
    C --> D["Query embedding<br/>BGE-M3 INT8, 1024-d, L2-normalised"]
    D --> E["Two-lane ANN recall per query<br/>standard lane: 64 / similar lane: 64"]
    E --> F["Per-capability maximum score<br/>score = 1 - cosine distance"]
    F --> G{"maximum score >= threshold?"}
    G -->|yes| H["Deterministic sort<br/>(-score, capability_id)"]
    G -->|no| I["Not a candidate"]
    H --> J["Top-3 at most, never padded<br/>K = number of selected capabilities"]
    J --> K["K capability cards + 1 REJECT card<br/>options = K + 1, 1..4"]
    K --> L["In-place K=0 to K=1 repair<br/>action rows with a valid gold"]
    L --> M["Deterministic option shuffle<br/>reject-shuffle-v1 over the row id"]
    M --> N["Frozen 14-key JSONL row"]
```

Construction order matters: the split is assigned **before** embeddings and recall, and the
same split is reused by all thresholds. Thresholds are applied **after** recall, to the
per-capability maximum scores, so the three versions do not re-run retrieval.

**Vector recall.** Queries are embedded with the frozen profile
`eric-ml-nlp/Delveta-BGE-M3-v1#INT8` through the TEI service: 1024 dimensions,
L2-normalized. Recall uses the same two-lane shape as the production retriever:

| Lane | Corpus table | Pool per query | Score |
|---|---|---|---:|
| standard | `capability_standard_queries` | 64 | `1 - cosine_distance` |
| similar | `capability_similar_queries` joined to its standard query | 64 | `1 - cosine_distance` |

The ANN index is approximate (HNSW), not an exhaustive exact-cosine scan. Hits are stored at
or above an evidence floor of 0.80; every published threshold is at least 0.82, so the floor
cannot change the candidate sets in this release. Business filters (`enabled`, `status`,
`intent_kind`, UI visibility) are intentionally **not** applied: all 18 capabilities,
including `cap-research`, participate.

**Thresholding and aggregation.** All hits from both lanes are reduced to one maximum score
per capability. A capability becomes a candidate when `max_score >= threshold` (inclusive).
Candidates are sorted by `(-score, capability_id)` and the first three are kept:
`K = len(top3)`, `options = K capability cards + 1 REJECT card`. There is no padding.
`OPTION_SLOTS = 4` is only the adapter's upper bound.

**Labels and `REJECT`.** Every row contains exactly one frozen `REJECT` card. Label
precedence:

| Condition | Label | `target_kind` | Provenance (audit only) |
|---|---|---|---|
| Source row is No-Action | `REJECT` | `reject` | `no_action` |
| `K = 0` | `REJECT` | `reject` | `no_candidates` |
| Source gold capability is in Top-3 | that capability | `capability` | `gold_in_candidates` |
| Gold is valid but not in Top-3 | `REJECT` | `reject` | `gold_not_retrieved` |

**In-place K=0 → K=1 repair.** Source action rows whose gold is valid but which had no
candidate above the threshold would otherwise have a single `REJECT` option with no
capability to learn from. The official repair injects the source gold capability as the only
capability candidate: `K=1`, `options=2`, order `[gold capability, REJECT]`, shuffled with
seed prefix `k0-to-k1-inplace-v1:<threshold>:<split>`. The audit marks the injected
candidate as `candidate_origin = "source_gold_injected"`; no retrieval score exists for it.
Post-repair, every remaining `K=0` row is a source No-Action row.

**Language resolution.** An explicit `zh` or `en` source value is kept; otherwise the query
script is inspected. Han only → `zh`, Latin only → `en`, Han plus Latin → `mixed`; mixed and
unknown are never folded into `zh` or `en`.

## 5. Leakage protection

| Risk | Protection |
|---|---|
| The evaluation query is also a retrieval-corpus query | Exclude source rows whose normalized text equals a standard or similar corpus query (908 excluded: 36 standard, 872 similar) |
| Paraphrased members of one family cross splits | Assign whole query families to one split; families are indivisible units |
| The card view names the answer | Store the `B_noprov` view: `query examples:` and the `evidence:` provenance line are removed |
| The model learns a slot prior | Shuffle all `K + 1` options per row with a deterministic row-keyed seed |
| Retrieval rank leaks through option order | Store only the shuffled `order`; scores and ranks are not training fields |
| Test data influences training or model selection | Training and selection read validation only; test is benchmark-only |
| Calibration rows receive gradient updates | Exclude the frozen 400 calibration rows from the effective train set |

The published JSONL files were checked for cross-split overlap by `id`, `source_query_id`,
exact `state`, normalized `state` and the construction query-family heuristic: no overlap
was found. There are 48,333 unique normalized queries and 36,621 unique family signatures
across the release.

Limitation: the source has no parent/derived-record id, so corpus overlap can only be
detected by exact or normalized text equality. The dataset does not prove that no semantic
paraphrase of a corpus query exists.

## 6. Getting the data

The dataset repository is
[**eric-ml-nlp/Delveta-LayaChoice-v2-Data**](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data).
Pin a revision for reproducible runs.

### Default bundle (082) — flat layout

The repository root is the flat `recall_threshold_082` bundle
(`manifest.json` plus `train.jsonl`, `val.jsonl`, `test.jsonl`), the file layout
`run.sh` expects at `V2/data/`. The bundle uses the 14-key construction schema; see §8
for the adapter-schema compatibility note before pointing training at it.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="eric-ml-nlp/Delveta-LayaChoice-v2-Data",
    repo_type="dataset",
    allow_patterns=["train.jsonl", "val.jsonl", "test.jsonl", "manifest.json",
                    "SHA256SUMS", "README.md", "DATASET_SCHEMA.md", ".gitattributes"],
    local_dir="fine_tuning/laya_finetune/V2/data",
)
PY
```

This downloads `manifest.json`, `train.jsonl`, `val.jsonl`, `test.jsonl`, `SHA256SUMS`
and the dataset card files, without the historical `data_versions/` trees. Once the row
schema is accepted by the adapter (§8), training reads only the manifest and the three
splits.

### Historical versions (085 / 088)

Download one version into its own flat directory and pass it with `--data`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

for version in ("recall_threshold_085", "recall_threshold_088"):
    snapshot_download(
        repo_id="eric-ml-nlp/Delveta-LayaChoice-v2-Data",
        repo_type="dataset",
        allow_patterns=[f"data_versions/{version}/*"],
        local_dir=f".local-data/{version}",
    )
PY
mv .local-data/recall_threshold_085/data_versions/recall_threshold_085/* .local-data/recall_threshold_085/
rm -rf .local-data/recall_threshold_085/data_versions
```

Repeat for 088, or keep the nested layout and pass the inner directory directly. Verify a
downloaded bundle with the bundled checksums:

```bash
cd .local-data/recall_threshold_085 && sha256sum -c SHA256SUMS
```

The Hugging Face bundles use `train.jsonl` / `val.jsonl` / `test.jsonl`. The archived
working-copy layout under `data_versions/recall_threshold_0XX/` on a construction machine
uses `v2_train.jsonl` / `v2_val.jsonl` / `v2_test.jsonl`; see §8 for the difference.

## 7. Manifests and checksums

Each Hugging Face bundle carries its own `manifest.json` (split sizes, calibration indices
and ids, option contract, base revision) and `SHA256SUMS`. The dataset repo is the source of
truth for downloaded bytes.

| Version | Manifest | Checksum file |
|---|---|---|
| 082 (root) | [manifest.json](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data/blob/main/manifest.json) | [SHA256SUMS](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data/blob/main/SHA256SUMS) |
| 085 | [manifest.json](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data/blob/main/data_versions/recall_threshold_085/manifest.json) | [SHA256SUMS](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data/blob/main/data_versions/recall_threshold_085/SHA256SUMS) |
| 088 | [manifest.json](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data/blob/main/data_versions/recall_threshold_088/manifest.json) | [SHA256SUMS](https://huggingface.co/datasets/eric-ml-nlp/Delveta-LayaChoice-v2-Data/blob/main/data_versions/recall_threshold_088/SHA256SUMS) |

Frozen bundle checksums (from the Hugging Face repository):

| File | 082 (default) | 085 | 088 |
|---|---|---|---|
| `train.jsonl` | `1a9472337f64cd2a89ba06e29cb2cfd97bcb3ce4c786c684350c4030893cd3b1` | `302d022b83e2c3537d5a7decc22daed9186abbef4fd80558c1c8c6ca54ef9e47` | `f9eb897015312a874e3f8fcb582222954d8bd52f61733575bcd7df1ee64e57d4` |
| `val.jsonl` | `943000f2e809568accac537bf82444580a97bf1327c594b6973405b81fbd9d90` | `2e3799ea8542747964eae4453beec192c8dc6bf0f63acf582366f99998a2f976` | `e9b2146ad08e57325466a5a0659a184307c56a12445f3fb03050be42b3705944` |
| `test.jsonl` | `3abd7b3c6fdf45435a99c78bc87a415f0d0e91b41274abbdf85a5a99c23c3c15` | `80dd558465a6b25bbe043908e48562994526eac6b2fc8891a2d6b24b2ab33111` | `74c66a6907f4bb2259fce4b567ef5fbc3a5c782d8a0c53b60f854bfb62e18709` |
| `manifest.json` | `5bb80da52f484cb10a24a23fb1f6333fb796dc6943f958e08bba473bbc101dd8` | `e0e38c80fc476e63fdc3a8769ce0e55eb7cf1bf8d28f4a70a8b96a4ab1836921` | `6ccfe2eddc72e03a6aa1644d13e107881cccbad7e6f008ccf028e0e2c3fee295` |

[SHA256SUMS.all_thresholds](SHA256SUMS.all_thresholds) in this directory is a different
record: it covers the construction workspace copies (`recall_threshold_0XX/v2_*.jsonl`,
construction audits, `direct_reject.jsonl` and `shared/` artifacts). Those files are not the
Hugging Face release and are not distributed with this repository.

## 8. Training-code data contract and schema compatibility

The training code is not changed by this document. Its current contract:

- [layachoice_v2_dataset.py](../V2/layachoice_v2_dataset.py) defines
  `DATA = <V2>/data` as the default bundle root. It reads `manifest.json` and
  `{split}.jsonl` (falling back to `v2_{split}.jsonl`).
- [run.sh](../V2/run.sh) defaults to `DATA="$HERE/data"` and accepts `--data DIR` on every
  stage (also `--python EXE` and `--out DIR`).
- [layachoice_v2_verify.py](../V2/layachoice_v2_verify.py) detects two layouts:
  - **flat**: `manifest.json` at the bundle root;
  - **dynamic**: `recall_threshold_082/`, `recall_threshold_085/` and
    `recall_threshold_088/` side by side, read from the parent
    `SHA256SUMS.all_thresholds`. This is the construction-workspace archive layout, with
    `v2_*.jsonl` file names.

**Schema difference.** The adapter validates exactly four row keys
(`message_id`, `language`, `input`, `label`) and requires the manifest's
`training_schema_keys` to name the same four keys. The Hugging Face release uses the
14-key construction schema (`id`, `source_query_id`, `split`, `part`, `lang`, `state`,
`order`, `options`, `gold_index`, `target_kind`, `gold_capability_id`, `reject_index`,
`shuffle`, `src_sha`). The file names and split counts match, but the row contract does not:

- `adapt_row` rejects a Hugging Face row as schema drift;
- `read_manifest` and the verifier reject its 14-key `training_schema_keys`.

Therefore the Hugging Face 082 bundle is **not directly readable** by the current training
code. The local construction archive under `data_versions/recall_threshold_0XX/` carries the
4-key rows the current adapter expects, but it is a different byte artifact (different
SHA256), not the Hugging Face release. Using the release for training requires an adapter
revision that accepts the 14-key schema; that is a code change subject to separate approval.

Practical consequences:

- The default entry point expects a flat bundle at `V2/data/` with the row schema accepted
  by the adapter. Downloading the Hugging Face root bundle there gives the same file names
  but not a readable row contract; do not replace an installed bundle without addressing
  the schema difference above.
- Checksums of a downloaded bundle can be checked directly with `sha256sum -c SHA256SUMS`.
  The repository's verifier is written for the 4-key contract described above.
- A historical version can be pointed at with `--data`; the same schema difference applies.
- The dynamic archive layout is **not** byte-compatible with the Hugging Face bundles
  (different file names and different bytes). Do not mix a Hugging Face download into a
  `recall_threshold_0XX/` directory that is covered by `SHA256SUMS.all_thresholds`.
- `message_id_map.jsonl` and the `shared/` retrieval intermediates are local audit material.
  Training never reads them.

## 9. Limitations

1. **No human-confirmed No-Action.** The 3,000 source no-action rows are either
   `rule_checked_pending_human_review` (2,000) or `synthetic_needs_validation` (1,000). They
   are not human-confirmed real user refusals.
2. **Synthetic and rebuilt source material.** The source combines rebuilt, query-list and
   synthetic records; it is not a sample of production traffic.
3. **Approximate retrieval.** Recall uses an HNSW ANN index with a 64-row pool per lane, not
   exact cosine search.
4. **Only exact/normalized leakage is excluded.** No parent/derived-record id exists in the
   source, so paraphrased or derived corpus queries may remain.
5. **Not every production capability filter is modeled.** All 18 capabilities participate,
   including `cap-research`.
6. **Language labels are partly derived.** 29,241 source rows have a blank `language` field
   and are resolved from script; `mixed` is kept as its own class.
7. **Threshold is a design choice.** The default 0.82 is the installed training bundle, not
   a statement that 0.82 is universally optimal. 0.85 and 0.88 are frozen alternatives with
   materially different candidate distributions.
8. **The test split is a fixed benchmark, not a fresh holdout.** It is frozen for this
   benchmark and should not be treated as an untouched held-out set for repeated model
   development.

## 10. Related files in this repository

| Path | Purpose |
|---|---|
| [SHA256SUMS.all_thresholds](SHA256SUMS.all_thresholds) | Checksums of the construction-workspace copies (not the Hugging Face release) |
| [dataset_construction/README.md](dataset_construction/README.md) | Construction bundle: layout, reproduction status and gaps |
| [dataset_construction/BUILD_RECORD.md](dataset_construction/BUILD_RECORD.md) | Build record, source lineage and known reproduction gaps |
| [dataset_construction/full_r1/](dataset_construction/full_r1/) | Construction entry, generation log and build reports |
| [dataset_construction/repair/](dataset_construction/repair/) | Official K=0 → K=1 repair and verification scripts |
| [dataset_construction/audit/](dataset_construction/audit/) | Read-only statistics, schema, parity and coverage audits |
| [dataset_construction/inventory/](dataset_construction/inventory/) | File-level SHA-256 inventories of the construction chain |
| [build/](build/) | Capture, generation, repair and card-refresh scripts |
| [smoke/](smoke/) | Smoke-fixture builder and driver |
| [audit/prompt_parity_audit.md](audit/prompt_parity_audit.md) | Prompt/rendering parity audit |
| [shared/capability_ids.json](shared/capability_ids.json) | The 18 capability ids |

Training recipe and run record: [V2/README.md](../V2/README.md).

## 11. License and citation

**License.** The Delveta repository code is licensed under the GNU Affero General Public
License v3.0. No separate license has been declared for this dataset content. This document
does not grant a dataset license; until the maintainer declares one, do not assume that the
data may be redistributed or used commercially. The source data's own terms also apply.

**Citation.** There is no associated paper, DOI or formal citation for this dataset. If a
reference is required, cite the dataset name and the exact version or revision used:

```text
Delveta. Delveta-LayaChoice-v2-Data. Version recall_threshold_082 (or the specific
threshold directory), <revision>.
```

Do not cite a paper that does not exist. The frozen checksums in each bundle's
`SHA256SUMS` identify the exact data bytes used in an experiment.
