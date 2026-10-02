#!/usr/bin/env bash
# LayaChoice-v2 single entry point. Every phase is explicit: there is no implicit
# "run everything" default, so a bare invocation can never start GPU training by
# accident. Phases are resumable — re-run any one of them alone.
#
#   bash run.sh verify                          # data/hash/schema checks (stdlib, no GPU)
#   bash run.sh baseline                        # zero-shot baseline on the frozen Test
#   bash run.sh train  [--epochs N]             # fine-tune (GPU unless --smoke)
#   bash run.sh select                          # per-epoch validation; pick best_epoch
#   bash run.sh temperature                     # fit the temperature on the calibration split
#   bash run.sh benchmark                       # best checkpoint on the frozen Test
#   bash run.sh export                          # write the Agent-loadable artifact
#   bash run.sh all     [--epochs N]            # baseline -> train -> select -> temperature
#                                               #           -> benchmark -> export
#
# Common flags: --smoke (CPU, 8 rows, 1 step)   --python <exe>   --data DIR   --out DIR
#               --export-dtype preserve|fp16|bf16
#
# Requires a POSIX bash (Linux, macOS, WSL, or Git-Bash on Windows). A phase that
# fails returns non-zero and stops the script. This script never deletes an output
# directory, never downloads anything unpinned, and never uploads.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
DATA="$HERE/data"
OUT="$HERE/out"
SMOKE=0
EPOCHS=10
EXPORT_DTYPE=preserve
LAYA_PIN="0.3.21"
PHASE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)        SMOKE=1; shift ;;
    --epochs)       EPOCHS="$2"; shift 2 ;;
    --export-dtype) EXPORT_DTYPE="$2"; shift 2 ;;
    --python)       PY="$2"; shift 2 ;;
    --data)         DATA="$2"; shift 2 ;;
    --out)          OUT="$2"; shift 2 ;;
    -h|--help|"")   sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SMOKE_ARGS=()
[[ $SMOKE -eq 1 ]] && SMOKE_ARGS=(--smoke)

phase() { printf '\n=== %s ===\n' "$1"; }

# Record what was run (git state, config, data version) so a later reader can tie an
# output directory back to a code revision. Written under $OUT (git-ignored).
ledger() {
  mkdir -p "$OUT"
  "$PY" - "$HERE" "$DATA" "$OUT" "$1" "$SMOKE" "$EPOCHS" <<'PY' >> "$OUT/run_ledger.jsonl"
import json, subprocess, sys, time, pathlib
here, data, out, p, smoke, epochs = sys.argv[1:7]
def git(*a):
    try: return subprocess.check_output(["git", "-C", here, *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
manifest = json.loads((pathlib.Path(data) / "manifest.json").read_text(encoding="utf-8"))
print(json.dumps({
    "phase": p, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
    "python": sys.version.split()[0], "epochs": int(epochs), "smoke": smoke == "1",
    "out": out,
    "data": {"bench": manifest.get("bench"), "rows": manifest.get("rows"),
             "base_revision": manifest.get("base_checkpoint", {}).get("revision")},
}, ensure_ascii=False))
PY
}

env_check() {
  phase "env check"
  "$PY" - "$LAYA_PIN" "$DATA" "$SMOKE" <<'PY'
import json, sys, pathlib
pin, data, smoke = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3] == "1"
import torch, transformers, laya
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"transformers {transformers.__version__}  laya {laya.__version__}")
if laya.__version__ != pin:
    raise SystemExit(f"laya {laya.__version__} != pinned {pin}; install requirements.txt")
if smoke:
    print("smoke: CPU, float32, no GradScaler, no DDP")
elif not torch.cuda.is_available():
    raise SystemExit("no CUDA device; use --smoke, or run on a GPU box")
else:
    print(f"cuda {torch.version.cuda}  {torch.cuda.get_device_name(0)}  "
          f"bf16={torch.cuda.is_bf16_supported()}")
manifest = data / "manifest.json"
if not manifest.exists():
    raise SystemExit(f"missing bundle: {manifest}")
m = json.loads(manifest.read_text(encoding="utf-8"))
print(f"bundle rows={m['rows']} view={m['formal_card_view']} "
      f"calibration={m['calibration']['calibration_rows']} "
      f"effective_train={m['calibration']['effective_train_rows']}")
PY
  if [[ $SMOKE -eq 0 ]]; then
    avail="$(df -Pk "$OUT" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}' || echo 0)"
    if [[ "${avail:-0}" -lt 20 ]]; then
      echo "WARNING: only ${avail} GB free under $OUT; ten epoch checkpoints need ~40 GB" >&2
    fi
  fi
}

do_verify() {
  phase "verify the frozen bundle (stdlib only, no GPU)"
  "$PY" "$HERE/layachoice_v2_verify.py"
}

do_baseline() {
  phase "zero-shot baseline (frozen Test, checkpoint temperature, no calibration)"
  "$PY" "$HERE/layachoice_v2_eval.py" --baseline-zero-shot "${SMOKE_ARGS[@]}" \
      --data "$DATA" --out "$OUT/baseline"
}

do_train() {
  phase "train ($EPOCHS epochs, one full-state checkpoint per epoch)"
  "$PY" "$HERE/layachoice_v2_finetune.py" --train "${SMOKE_ARGS[@]}" \
      --epochs "$EPOCHS" --data "$DATA" --out "$OUT"
}

do_select() {
  phase "select best epoch on Validation (argmax top-1, ties to the earlier epoch)"
  "$PY" "$HERE/layachoice_v2_eval.py" --select "${SMOKE_ARGS[@]}" \
      --data "$DATA" --run "$OUT" --out "$OUT/select"
}

best_ckpt() {
  "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_ckpt'])" \
      "$OUT/select/selection.json"
}

do_temperature() {
  local ckpt; ckpt="$(best_ckpt)"
  phase "fit temperature on the frozen calibration split ($ckpt)"
  "$PY" "$HERE/layachoice_v2_finetune.py" --fit-temperature "${SMOKE_ARGS[@]}" \
      --data "$DATA" --ckpt "$ckpt"
}

do_benchmark() {
  local ckpt; ckpt="$(best_ckpt)"
  phase "benchmark best checkpoint on the frozen Test"
  "$PY" "$HERE/layachoice_v2_eval.py" --benchmark "${SMOKE_ARGS[@]}" \
      --data "$DATA" --ckpt "$ckpt" --out "$OUT/benchmark" \
      --baseline "$OUT/baseline/baseline.report.json"
}

do_export() {
  local ckpt; ckpt="$(best_ckpt)"
  phase "export the deployment artifact (--export-dtype $EXPORT_DTYPE)"
  "$PY" "$HERE/layachoice_v2_finetune.py" --export "${SMOKE_ARGS[@]}" \
      --data "$DATA" --ckpt "$ckpt" --out "$OUT/release_v2" --export-dtype "$EXPORT_DTYPE"
}

case "$PHASE" in
  verify)      ledger verify; do_verify ;;
  baseline)    ledger baseline; env_check; do_baseline ;;
  train)       ledger train; env_check; do_train ;;
  select)      ledger select; env_check; do_select ;;
  temperature) ledger temperature; env_check; do_temperature ;;
  benchmark)   ledger benchmark; env_check; do_benchmark ;;
  export)      ledger export; env_check; do_export ;;
  all)         ledger all; env_check; do_baseline; do_train; do_select
               do_temperature; do_benchmark; do_export ;;
  -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
  *) echo "usage: bash run.sh {verify|baseline|train|select|temperature|benchmark|export|all} [flags]" >&2
     exit 2 ;;
esac

phase "done: $PHASE"
echo "outputs under $OUT"
