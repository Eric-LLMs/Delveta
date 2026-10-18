# LayaChoice V4 dataset construction chain

This directory preserves the source code, inputs and build records used to
construct the official V4 dataset. It is a code-and-provenance bundle, not a
training-data directory.

The generated dataset itself lives in two independent copies:

- `scripts/laya_finetune/V2/data/` — the copy used by the existing V2/V4
  training directory layout.
- `data/laya_finetune/data_v4_build/` — the complete build archive, including
  the original `full_r1` tree and the large retrieval intermediates.

## Layout

| Path | Purpose |
|---|---|
| `full_r1/scripts/build_dataset.py` | Main construction entry: source load, eligibility, split, embedding, recall, K+1 option construction, labels, shuffle, manifest and audits |
| `full_r1/logs/generation.log` | Run log of the `full_r1` generation |
| `full_r1/reports/` | Reports emitted by the construction entry and its validation stages |
| `data/laya_finetune/data_v4_build/data_v3/` | Upstream V3/V4 source-CSV build scripts, statistics and intermediate CSVs (kept in the data archive, not duplicated here) |
| `inputs/laya_tool_selection_v2.csv` | Final source CSV consumed by `build_dataset.py` (same bytes as the build-time `laya_tool_selection_v4.csv`); the raw upstream inputs are recorded in `BUILD_RECORD.md` and kept under `data/BaseData/` |
| `repair/v4_k0_to_k1_inplace.py` | Official in-place K=0 -> K=1 repair |
| `repair/verify_k0_to_k1_inplace.py` | Repair verification |
| `repair/unify_v4_official_version.py` | Official manifest/hash/archive unification |
| `repair/superseded/v4_repair_k0_gold.py` | Superseded repair approach that wrote a separate `full_r1_repaired` bundle |
| `audit/` | Read-only statistics, schema, parity, position-coverage and comparison scripts used during validation |
| `inventory/` | File-level SHA-256 inventories for the generated data and the pre-replacement V2 data |
| `reference/smoke50_r1_generation_config.json` | Configuration file from the smoke run; the `full_r1` configuration file is missing (see `BUILD_RECORD.md`) |

## Reproduction status

The chain is preserved byte-for-byte and is traceable, but it is not a
turnkey reproduction: the full-run generation config/manifests are missing,
the main entry still contains pre-move paths, and the recall stage depends on
an external PostgreSQL/pgvector database and embedding service. `BUILD_RECORD.md`
lists every gap explicitly.
