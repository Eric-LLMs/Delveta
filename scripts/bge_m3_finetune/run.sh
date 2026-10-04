#!/usr/bin/env bash
# Delveta BGE-M3 dual-target fine-tune -- one-command entry.
#
#   bash run.sh                 FORMAL training: the recorded bf16 / 1-epoch recipe.
#   bash run.sh --smoke         CPU plumbing check (env + Dataset V1 integrity + flags).
#                               It does NOT train and is NOT a substitute for the formal run.
#   bash run.sh --help          usage
#
# This wrapper only calls the real entry
# scripts/bge_m3_finetune/train_bge_m3_dual_target.py with the pinned formal recipe.
# It never edits the frozen Dataset V1, the capability-unique sampler, the loss, or any
# hyper-parameter. See docs/experiments/bge-m3-finetuning.md.
#
# The interpreter must provide FlagEmbedding 1.4.2 + transformers + torch + accelerate +
# datasets (the repo venv .venv-bge-train). Resolution order:
#   --python  >  $PYTHON  >  <repo>/.venv-bge-train/bin/python  >  python3  >  python
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
ENTRY="$HERE/train_bge_m3_dual_target.py"

# --- pinned formal recipe (mirrors run_summary.json; does not change here) ---
DATA="$HERE/data/bge_m3_train.jsonl"
MANIFEST="$HERE/data/bge_m3_train_manifest.jsonl"
MODEL="BAAI/bge-m3"
BATCH=8          # per_device_train_batch_size
GROUP=8          # train_group_size
LR="1e-5"        # learning_rate
EPOCHS=1         # num_train_epochs
SEQ=64           # query_max_len = passage_max_len (formal run)

OUT="$HERE/out"
PY=""
SMOKE=0
BF16=1           # formal recipe is bf16; --no-bf16 deviates
LOCAL_ONLY=0
ALLOW_EXISTING=0

usage() {
  cat <<'EOF'
Delveta BGE-M3 dual-target fine-tune -- one-command entry.

Usage:
  bash run.sh [options]

Modes:
  (default)        FORMAL training on the frozen Dataset V1, using the recorded
                   recipe: bf16, 1 epoch, per_device_batch=8, train_group=8,
                   lr=1e-5, query/passage max len=64. Requires a bf16-capable
                   CUDA GPU. This is the run that produced the published model.
  --smoke          CPU-only plumbing check: resolves paths, checks the Python
                   environment and dependencies, verifies Dataset V1 <-> manifest
                   row alignment (reusing dual_target_alignment.py), prints the
                   exact formal command, and runs `train_bge_m3_dual_target.py
                   --help`. It does NOT start training and writes nothing.

Options:
  --smoke                  run the CPU plumbing check instead of training
  --output-dir DIR         checkpoint output dir (default: <script>/out)
  --python PATH            python interpreter to use (see resolution order below)
  --model ID_OR_PATH       base model id or local path (default: BAAI/bge-m3)
  --local-files-only       load the model/tokenizer from the local HF cache only
  --no-bf16                run fp32 (NOT the recorded formal recipe)
  --allow-existing-output  proceed even if --output-dir already exists and is
                           non-empty (otherwise run.sh refuses to disturb it)
  -h, --help               show this help

Interpreter resolution:
  --python  >  $PYTHON  >  <repo>/.venv-bge-train/bin/python  >  python3  >  python

Examples:
  bash run.sh                                   # formal training
  PYTHON=../.venv-bge-train/bin/python bash run.sh
  bash run.sh --smoke                           # CPU check, no training
  bash run.sh --output-dir /workspace/_bge_out  # explicit output location
EOF
}

# --- argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)                 SMOKE=1; shift ;;
    --no-bf16)               BF16=0; shift ;;
    --local-files-only)      LOCAL_ONLY=1; shift ;;
    --allow-existing-output) ALLOW_EXISTING=1; shift ;;
    --python)                PY="$2"; shift 2 ;;
    --output-dir)            OUT="$2"; shift 2 ;;
    --model)                 MODEL="$2"; shift 2 ;;
    -h|--help)               usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; echo "try: bash $0 --help" >&2; exit 2 ;;
  esac
done

# --- resolve the python interpreter -----------------------------------------
if [[ -z "$PY" ]]; then
  if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
  elif [[ -x "$REPO_ROOT/.venv-bge-train/bin/python" ]]; then
    PY="$REPO_ROOT/.venv-bge-train/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
  else
    PY="python"
  fi
fi
if ! command -v "$PY" >/dev/null 2>&1 && [[ ! -x "$PY" ]]; then
  echo "ERROR: python interpreter not found: '$PY'" >&2
  echo "       pass --python <path> or activate the .venv-bge-train environment." >&2
  exit 3
fi

stage() { printf '\n=== %s ===\n' "$1"; }

# --- output-dir overwrite guard (formal mode only; fail fast) ----------------
if [[ $SMOKE -eq 0 && -d "$OUT" && -n "$(ls -A "$OUT" 2>/dev/null || true)" ]]; then
  if [[ $ALLOW_EXISTING -eq 0 ]]; then
    echo >&2
    echo "ERROR: output dir already exists and is not empty: $OUT" >&2
    echo "       refusing to run so existing checkpoints are not disturbed." >&2
    echo "       use --output-dir <new>, or --allow-existing-output to override." >&2
    exit 4
  fi
  echo "WARNING: $OUT is not empty and --allow-existing-output was given; continuing" >&2
fi

# --- environment + dependency + GPU check -----------------------------------
stage "environment"
"$PY" -c 'import sys; print("python", sys.executable, sys.version.split()[0])'
"$PY" - "$SMOKE" "$BF16" <<'PY'
import importlib, sys

smoke = sys.argv[1] == "1"
want_bf16 = sys.argv[2] == "1"

need = ("FlagEmbedding", "torch", "transformers", "accelerate", "datasets", "huggingface_hub")
vers, missing = {}, []
for name in need:
    try:
        vers[name] = getattr(importlib.import_module(name), "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{name} ({exc})")
if missing:
    sys.exit(
        "ERROR: missing Python dependencies: " + "; ".join(missing) + "\n"
        "       install FlagEmbedding 1.4.2 and the training stack into .venv-bge-train,\n"
        "       then re-run with that interpreter (--python / $PYTHON)."
    )
print("deps:", "  ".join(f"{k}={v}" for k, v in vers.items()))

import torch

cuda = torch.cuda.is_available()
bf16 = bool(cuda and torch.cuda.is_bf16_supported())
if cuda:
    print(f"cuda: {torch.version.cuda}  {torch.cuda.get_device_name(0)}  bf16={bf16}")
else:
    print("cuda: unavailable (CPU only)")

if not smoke:
    if not cuda:
        sys.exit(
            "ERROR: CUDA unavailable; the formal recipe targets a GPU.\n"
            "       use --smoke for a CPU plumbing check."
        )
    if want_bf16 and not bf16:
        sys.exit(
            "ERROR: the formal recipe needs a bf16-capable GPU; this device is not.\n"
            "       run on a bf16 GPU, or pass --no-bf16 (deviates from the formal recipe)."
        )
PY

# --- Dataset V1 integrity (reuses the aligned-row check) ---------------------
stage "dataset V1 integrity"
"$PY" - "$DATA" "$MANIFEST" "$HERE" <<'PY'
import pathlib, sys

data, manifest, here = (pathlib.Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(here))

for path, label in ((data, "train data"), (manifest, "train manifest")):
    if not path.is_file():
        sys.exit(f"ERROR: missing {label}: {path}\n"
                 "       Dataset V1 is frozen and must not be regenerated; restore it from the repo.")

from dual_target_alignment import AlignmentError, validate
from capability_unique_sampler import capabilities_from_manifest

try:
    report = validate(data, manifest)
except AlignmentError as exc:
    sys.exit(f"ERROR: dataset/manifest alignment failed: {exc}")

caps = capabilities_from_manifest(manifest)
print(f"aligned rows={report['rows']}  capabilities={len(set(caps))}  "
      f"sha256_mismatches={report['sha256_mismatches']}")
PY

# --- assemble the formal training command ------------------------------------
TRAIN_CMD=("$PY" "$ENTRY"
  --train_data "$DATA"
  --train_manifest "$MANIFEST"
  --model_name_or_path "$MODEL"
  --output_dir "$OUT"
  --per_device_train_batch_size "$BATCH"
  --train_group_size "$GROUP"
  --learning_rate "$LR"
  --num_train_epochs "$EPOCHS"
  --query_max_len "$SEQ"
  --passage_max_len "$SEQ")
[[ $LOCAL_ONLY -eq 1 ]] && TRAIN_CMD+=(--local_files_only)
[[ $BF16 -eq 1 ]] && TRAIN_CMD+=(--bf16)

# --- smoke: report and stop (no training) ------------------------------------
if [[ $SMOKE -eq 1 ]]; then
  stage "SMOKE CHECK ONLY -- no training will start"
  echo "resolved python : $PY"
  echo "entry           : $ENTRY"
  echo "output dir      : $OUT   (nothing is written in --smoke)"
  echo "recipe          : bf16=$BF16 local_files_only=$LOCAL_ONLY batch=$BATCH " \
       "group=$GROUP lr=$LR epochs=$EPOCHS seq=$SEQ"
  echo "would run       :"
  printf '    %q' "${TRAIN_CMD[@]}"; echo
  stage "entry flag surface (train_bge_m3_dual_target.py --help)"
  "$PY" "$ENTRY" --help >/dev/null && echo "entry imports and parses --help: OK"
  stage "smoke done -- no training performed"
  exit 0
fi

# --- formal training ---------------------------------------------------------
stage "formal training"
echo "python : $PY"
echo "entry  : $ENTRY"
echo "model  : $MODEL"
echo "output : $OUT"
echo "config : batch=$BATCH group=$GROUP lr=$LR epochs=$EPOCHS seq=$SEQ " \
     "local_files_only=$LOCAL_ONLY bf16=$BF16"
echo "running:"
printf '    %q' "${TRAIN_CMD[@]}"; echo

if command -v df >/dev/null 2>&1; then
  avail="$(df -Pk "$OUT" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}' || true)"
  if [[ -n "$avail" && "$avail" -lt 5 ]]; then
    echo "WARNING: only ${avail} GB free under $OUT; checkpoints need several GB" >&2
  fi
fi

exec "${TRAIN_CMD[@]}"
