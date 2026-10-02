#!/usr/bin/env bash
# LayaChoice-v2 — single end-to-end entry point.
#
#   bash run.sh                 # === run the WHOLE pipeline (default) ===
#   bash run.sh all             # identical to the bare invocation
#   bash run.sh <phase>         # run ONE phase only (every phase is resumable)
#
# Phases, in pipeline order:
#   setup        create .venv and `pip install -r requirements.txt` (skipped if the
#                chosen interpreter already has laya==the pinned version)
#   verify       data hashes + schema + labels + splits (stdlib, no GPU, no network)
#   baseline     zero-shot baseline on the frozen Test
#   train        fine-tune, one full-state checkpoint per epoch (GPU)  [--epochs N]
#   select       per-epoch Validation -> best_epoch (argmax top-1, ties to earlier)
#   temperature  fit the temperature on the frozen calibration split
#   benchmark    best checkpoint on the frozen Test (+ comparison to the baseline)
#   export       write the Agent-loadable deployment artifact
#   analysis     build out/analysis/val_metrics.json from the run, plot the Val curves
#
# Flags: --smoke (CPU, 8 rows, 1 step)   --epochs N
#        --python EXE   --data DIR   --out DIR   --export-dtype preserve|fp16|bf16
#
# `set -euo pipefail`: the first failing phase stops the run with a non-zero exit.
# This script never deletes an output directory, never downloads anything unpinned
# and never uploads. Each invocation appends one line to $OUT/run_ledger.jsonl.
set -euo pipefail

# Real-time logs: child Python runs unbuffered, so per-epoch results and every phase
# summary appear in the log the instant they are produced (not only when a buffer fills).
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
DATA="$HERE/data"
OUT="$HERE/out"
VENV="$HERE/.venv"
SMOKE=0
EPOCHS=10
EXPORT_DTYPE=preserve
LAYA_PIN="0.3.21"
PHASE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)        SMOKE=1; shift ;;
    --epochs)       EPOCHS="$2"; shift 2 ;;
    --export-dtype) EXPORT_DTYPE="$2"; shift 2 ;;
    --python)       PY="$2"; shift 2 ;;
    --data)         DATA="$2"; shift 2 ;;
    --out)          OUT="$2"; shift 2 ;;
    -h|--help)      sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SMOKE_ARGS=()
[[ $SMOKE -eq 1 ]] && SMOKE_ARGS=(--smoke)

phase() { printf '\n=== %s ===\n' "$1"; }

venv_py() {
  if   [[ -x "$VENV/bin/python" ]];        then echo "$VENV/bin/python"
  elif [[ -x "$VENV/Scripts/python.exe" ]]; then echo "$VENV/Scripts/python.exe"
  fi
}

# ── environment ──────────────────────────────────────────────────────────────
has_laya() { "$1" -c "import laya;assert laya.__version__=='$LAYA_PIN'" >/dev/null 2>&1; }

ensure_env() {
  if has_laya "$PY"; then return 0; fi
  if [[ -n "$(venv_py)" ]] && has_laya "$(venv_py)"; then PY="$(venv_py)"; return 0; fi
  phase "env setup: create $VENV and pip install -r requirements.txt"
  "$PY" -m venv "$VENV"
  PY="$(venv_py)"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$HERE/requirements.txt"
  has_laya "$PY" || { echo "laya==$LAYA_PIN still not importable after install" >&2; exit 1; }
}

check_env() {
  phase "env check"
  "$PY" - "$LAYA_PIN" "$DATA" "$SMOKE" <<'PY'
import json, sys, pathlib
pin, data, smoke = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3] == "1"
import torch, transformers, laya
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"transformers {transformers.__version__}  laya {laya.__version__}")
if laya.__version__ != pin:
    raise SystemExit(f"laya {laya.__version__} != pinned {pin}; run `bash run.sh setup`")
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
}

# Append one ledger line per invocation (git state + data version), under $OUT.
ledger() {
  mkdir -p "$OUT"
  "$PY" - "$HERE" "$DATA" "$OUT" "$1" "$SMOKE" "$EPOCHS" <<'PY' >> "$OUT/run_ledger.jsonl"
import json, subprocess, sys, time, pathlib
here, data, out, p, smoke, epochs = sys.argv[1:7]
def git(*a):
    try: return subprocess.check_output(["git", "-C", here, *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
try:
    manifest = json.loads((pathlib.Path(data) / "manifest.json").read_text(encoding="utf-8"))
except Exception:
    manifest = {}
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

# ── phases ───────────────────────────────────────────────────────────────────
# A few bundle JSON files were hashed on Windows (CRLF) but committed as LF, so a
# fresh Linux checkout fails the SHA gate. Restore the exact byte form each file's
# own SHA256SUMS records — no parsed content changes — before verifying. If a file
# already matches, nothing is touched; originals are copied under $OUT/eol_backup.
normalize_bundle_eol() {
  [[ -f "$DATA/SHA256SUMS" ]] || return 0
  "$PY" - "$DATA" "$OUT" <<'PY'
import hashlib, pathlib, shutil, sys
data, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

def sha(b):
    return hashlib.sha256(b).hexdigest()

recorded = {}
for line in (data / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        h, name = line.split(None, 1)
        recorded[name.strip()] = h

changed = []
for name, want in recorded.items():
    p = data / name
    if not p.is_file():
        continue
    raw = p.read_bytes()
    if sha(raw) == want:
        continue
    lf = raw.replace(b"\r\n", b"\n")
    hit = None
    for variant, tag in ((lf.replace(b"\n", b"\r\n"), "CRLF"), (lf, "LF")):
        if sha(variant) == want:
            hit = (variant, tag)
            break
    if hit is None:
        print(f"[eol] {name}: no single EOL change reproduces the recorded hash")
        continue
    variant, tag = hit
    bak = out / "eol_backup"
    bak.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, bak / name)
    p.write_bytes(variant)
    changed.append(f"{name}->{tag}")

if changed:
    print("[eol] normalised to the recorded bytes: " + ", ".join(changed))
PY
}

do_verify() {
  phase "verify the frozen bundle (stdlib only, no GPU)"
  normalize_bundle_eol
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
  "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['best_ckpt'])" \
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

# Build val_metrics.json from THIS run (select reports + train_meta), then plot.
# The frozen analysis/val_metrics.json is left untouched; the fresh one lands in $OUT.
do_analysis() {
  phase "analysis: per-epoch Val curves from this run"
  mkdir -p "$OUT/analysis"
  cp "$HERE/analysis/plot_val_curves.py" "$OUT/analysis/plot_val_curves.py"
  "$PY" - "$OUT/select" "$OUT/analysis" "$OUT" <<'PY'
import json, sys, pathlib
sel, dest, run = (pathlib.Path(a) for a in sys.argv[1:4])
rows = []
for rep in sorted(sel.glob("epoch-*.val.report.json"),
                  key=lambda p: int(p.name.split("-")[1].split(".")[0])):
    e = int(rep.name.split("-")[1].split(".")[0])
    r = json.loads(rep.read_text(encoding="utf-8"))
    rj = r.get("reject", {})
    tm = run / f"epoch-{e}" / "train_meta.json"
    tl = json.loads(tm.read_text(encoding="utf-8")).get("mean_loss") if tm.exists() else None
    rows.append({"epoch": e, "train_loss": tl, "val_top1": r["top1"], "val_ece": r["ece"],
                 "val_reject_recall": rj.get("recall"),
                 "val_reject_fpr": rj.get("false_positive_rate"),
                 "val_reject_precision": rj.get("precision"),
                 "gold_reject_n": rj.get("gold_reject_n")})
if not rows:
    raise SystemExit(f"no epoch-*.val.report.json under {sel}; run `bash run.sh select` first")
(dest / "val_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
print(f"[analysis] {len(rows)} epochs -> {dest / 'val_metrics.json'}")
PY
  "$PY" "$OUT/analysis/plot_val_curves.py"
}

# ── dispatch ─────────────────────────────────────────────────────────────────
case "$PHASE" in
  setup)       ledger setup;       ensure_env; check_env ;;
  verify)      ledger verify;      do_verify ;;
  baseline)    ledger baseline;    ensure_env; check_env; do_baseline ;;
  train)       ledger train;       ensure_env; check_env; do_train ;;
  select)      ledger select;      ensure_env; check_env; do_select ;;
  temperature) ledger temperature; ensure_env; check_env; do_temperature ;;
  benchmark)   ledger benchmark;   ensure_env; check_env; do_benchmark ;;
  export)      ledger export;      ensure_env; check_env; do_export ;;
  analysis)    ledger analysis;    ensure_env; do_analysis ;;
  all)         ledger all;         ensure_env; check_env
               do_verify; do_baseline; do_train; do_select
               do_temperature; do_benchmark; do_export; do_analysis ;;
  -h|--help)   sed -n '2,25p' "$0"; exit 0 ;;
  *) echo "usage: bash run.sh {setup|verify|baseline|train|select|temperature|benchmark|export|analysis|all} [flags]" >&2
     exit 2 ;;
esac

phase "done: $PHASE"
echo "outputs under $OUT"
