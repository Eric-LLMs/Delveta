#!/usr/bin/env bash
# LayaChoice-v1 end-to-end run: env check -> zero-shot baseline -> train -> select ->
# temperature -> benchmark -> export. Every stage is resumable by re-running it alone;
# see README.md for the individual commands.
#
#   bash run.sh --smoke                      # CPU, 8 rows per split, 1 training step
#   bash run.sh                              # the real 4-epoch GPU run
#   bash run.sh --push-to-hub <repo_id>      # ...and upload the export (private by default)
#
# The Final Test never gates anything: the benchmark always exits 0 and the export runs
# whether or not the fine-tuned model beats the zero-shot baseline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
DATA="$HERE/data"
OUT="$HERE/out"
SMOKE=0
EPOCHS=4
EXPORT_DTYPE=preserve
PUSH=""
SKIP_BASELINE=0
LAYA_PIN="0.3.21"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)          SMOKE=1; shift ;;
    --epochs)         EPOCHS="$2"; shift 2 ;;
    --export-dtype)   EXPORT_DTYPE="$2"; shift 2 ;;
    --push-to-hub)    PUSH="$2"; shift 2 ;;
    --skip-baseline)  SKIP_BASELINE=1; shift ;;
    --python)         PY="$2"; shift 2 ;;
    --data)           DATA="$2"; shift 2 ;;
    --out)            OUT="$2"; shift 2 ;;
    -h|--help)        sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SMOKE_ARGS=()
[[ $SMOKE -eq 1 ]] && SMOKE_ARGS=(--smoke)

stage() { printf '\n=== %s ===\n' "$1"; }

# ── env check ───────────────────────────────────────────────────────────────
stage "env check"
"$PY" - "$LAYA_PIN" "$DATA" "$SMOKE" <<'PY'
import json, sys, pathlib
pin, data, smoke = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3] == "1"
import torch, transformers, laya
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"transformers {transformers.__version__}  laya {laya.__version__}")
if laya.__version__ != pin:
    raise SystemExit(f"laya {laya.__version__} != pinned {pin}; the fine-tune recipe was "
                     f"verified against {pin}. pip install laya=={pin}")
if smoke:
    print("smoke: CPU, float32, no GradScaler, no DDP")
else:
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; use --smoke, or pass a GPU")
    name = torch.cuda.get_device_name(0)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"cuda {torch.version.cuda}  {name}  bf16={bf16}  "
          f"amp_dtype={'bf16' if bf16 else 'fp16 (fallback, GradScaler enabled)'}")
manifest = data / "manifest.json"
if not manifest.exists():
    raise SystemExit(f"missing bundle: {manifest}; run layachoice_prepare.py first")
m = json.loads(manifest.read_text(encoding="utf-8"))
print(f"bundle rows={m['rows']} view={m['formal_card_view']} "
      f"calibration={m['calibration']['calibration_rows']} rows "
      f"(effective train {m['calibration']['effective_train_rows']})")
PY
if [[ $SMOKE -eq 0 ]]; then
  avail="$(df -Pk "$OUT" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}' || echo 0)"
  if [[ "${avail:-0}" -lt 20 ]]; then
    echo "WARNING: only ${avail} GB free under $OUT; four epoch checkpoints need ~16 GB" >&2
  fi
fi

# ── zero-shot baseline (before any training) ────────────────────────────────
if [[ $SKIP_BASELINE -eq 0 && ! -f "$OUT/baseline/baseline.report.json" ]]; then
  stage "zero-shot baseline (frozen Final Test v3, checkpoint temperature, no calibration)"
  "$PY" "$HERE/layachoice_eval.py" --baseline-zero-shot "${SMOKE_ARGS[@]}" \
      --data "$DATA" --out "$OUT/baseline"
elif [[ $SKIP_BASELINE -eq 1 ]]; then
  stage "zero-shot baseline SKIPPED (--skip-baseline)"
else
  stage "zero-shot baseline already present; not re-run"
fi

# ── train ───────────────────────────────────────────────────────────────────
stage "train ($EPOCHS epochs, one full-state checkpoint per epoch)"
"$PY" "$HERE/layachoice_finetune.py" --train "${SMOKE_ARGS[@]}" \
    --epochs "$EPOCHS" --data "$DATA" --out "$OUT"

# ── select ──────────────────────────────────────────────────────────────────
stage "select the best epoch on Validation v3 (argmax top-1, ties to the earlier epoch)"
"$PY" "$HERE/layachoice_eval.py" --select "${SMOKE_ARGS[@]}" \
    --data "$DATA" --run "$OUT" --out "$OUT/select"

BEST_CKPT="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_ckpt'])" \
    "$OUT/select/selection.json")"
BEST_EPOCH="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['best_epoch'])" \
    "$OUT/select/selection.json")"
echo "best_epoch=$BEST_EPOCH  ckpt=$BEST_CKPT"

# ── temperature ─────────────────────────────────────────────────────────────
stage "fit the temperature on the frozen 85-row calibration split"
"$PY" "$HERE/layachoice_finetune.py" --fit-temperature "${SMOKE_ARGS[@]}" \
    --data "$DATA" --ckpt "$BEST_CKPT"

# ── benchmark ───────────────────────────────────────────────────────────────
stage "benchmark best_epoch=$BEST_EPOCH on the frozen Final Test v3"
"$PY" "$HERE/layachoice_eval.py" --benchmark "${SMOKE_ARGS[@]}" \
    --data "$DATA" --ckpt "$BEST_CKPT" --out "$OUT/benchmark" \
    --baseline "$OUT/baseline/baseline.report.json"

# ── export (never gated on the benchmark verdict) ───────────────────────────
stage "export the deployment artifact (--export-dtype $EXPORT_DTYPE)"
"$PY" "$HERE/layachoice_finetune.py" --export "${SMOKE_ARGS[@]}" \
    --data "$DATA" --ckpt "$BEST_CKPT" --out "$OUT/final" --export-dtype "$EXPORT_DTYPE"

# ── optional upload ─────────────────────────────────────────────────────────
if [[ -n "$PUSH" ]]; then
  stage "upload $OUT/final -> $PUSH (private)"
  if command -v hf >/dev/null 2>&1; then CLI=hf; else CLI=huggingface-cli; fi
  "$CLI" upload "$PUSH" "$OUT/final" --repo-type model --private
  echo "uploaded to https://huggingface.co/$PUSH"
fi

stage "done"
echo "baseline   $OUT/baseline/baseline.report.json"
echo "selection  $OUT/select/selection.json"
echo "benchmark  $OUT/benchmark/final_test.report.json"
echo "export     $OUT/final  (+ reload_smoke.json)"
