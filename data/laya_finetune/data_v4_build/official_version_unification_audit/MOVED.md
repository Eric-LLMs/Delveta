# Intermediate V4 artifacts moved

Only the fine-tuning bundle stays in the code tree:

- Active training data: `scripts/laya_finetune/data_v4_build/full_r1`

Moved here (gitignored `data/` root):

- `smoke50_r1/`
- `archive/full_r1_repaired_superseded/`
- `full_r1_k0_to_k1_inplace_audit/`
- `official_version_unification_audit/`
- `data_v3/` (source CSV, build scripts, and statistics previously at
  `scripts/laya_finetune/data_v3`)

Manifest `repair.audit_dir` values were updated to the new audit location; pre-move manifest copies are under `manifest_backup_pre_move/`.

Known stale references after moving `data_v3` (code changes not applied):

- `scripts/laya_finetune/data_v4_build/full_r1/scripts/build_dataset.py`
  still points `SOURCE_CSV` at `scripts/laya_finetune/data_v3/laya_tool_selection_v4.csv`.
- `full_r1/reports/*.md` record the old source path as historical provenance.
- `tests/test_laya_v4_dataset_contract.py` and
  `tests/test_laya_v4_prompt_parity.py` still reference
  `scripts/laya_finetune/data_v4_build/smoke50_r1` and will be skipped.
