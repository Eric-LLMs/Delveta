#!/usr/bin/env bash
# Reproducible entry for the Qwen3-0.6B parameter-extraction LoRA fine-tune.
#
# This is a THIN, SAFE wrapper around the REAL training entry (`train.py`). It
# never re-implements training, never downloads/uploads anything, never deletes
# or overwrites a checkpoint, and never resumes on its own. It runs an
# environment + frozen-data preflight, then hands off to `train.py` unchanged.
#
# Usage:
#   bash run.sh --check        preflight ONLY (python / deps / config / data /
#                              SHA-256 / disk / GPU). NEVER starts training.
#   bash run.sh --dry-run      preflight, then build the full wiring and STOP
#                              before trainer.train() (proves the run launches).
#   bash run.sh                preflight, then the REAL LoRA training run.
#   bash run.sh <train.py ...> any other argument is passed straight through to
#                              train.py (e.g. --resume-from-checkpoint auto).
#   bash run.sh --help         this help.
#
# Environment overrides (all optional; defaults shown are safe):
#   PE_PYTHON          Python interpreter (default: this dir's .venv, else
#                      python3 / python on PATH).
#   PE_CONFIG          config.yaml to use (default: the one beside this script).
#   PE_MIN_GPU_MEM_MB  Minimum FREE GPU memory, MiB (default: 8000).
#   PE_MIN_DISK_MB     Minimum FREE disk for the output dir, MiB (default: 4096).
#   PE_ALLOW_CPU=1     Permit preflight / training WITHOUT a GPU (default: 0).
#   PE_SKIP_HASH=1     Skip the frozen-bundle SHA-256 gate (DEBUG ONLY — warns).
#
# No Hugging Face token is read, written or logged here. The base checkpoint is
# public; if auth is ever required it is handled by the HF client from the
# ambient environment (e.g. `hf auth login`), never by this script.
set -euo pipefail

# ── locate everything from the SCRIPT PATH, never the caller's CWD ────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PE_CONFIG="${PE_CONFIG:-${SCRIPT_DIR}/config.yaml}"
PE_MIN_GPU_MEM_MB="${PE_MIN_GPU_MEM_MB:-8000}"
PE_MIN_DISK_MB="${PE_MIN_DISK_MB:-4096}"
PE_ALLOW_CPU="${PE_ALLOW_CPU:-0}"
PE_SKIP_HASH="${PE_SKIP_HASH:-0}"

# ── tiny logging helpers ─────────────────────────────────────────────────────────
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*" >&2; }
die()  { printf '  [FAIL] %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s ==\n' "$*"; }

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

# ── resolve the interpreter ──────────────────────────────────────────────────────
resolve_python() {
    if [[ -n "${PE_PYTHON:-}" ]]; then
        command -v "${PE_PYTHON}" >/dev/null 2>&1 || [[ -x "${PE_PYTHON}" ]] \
            || die "PE_PYTHON='${PE_PYTHON}' is not executable"
        printf '%s\n' "${PE_PYTHON}"; return
    fi
    local cand
    for cand in "${SCRIPT_DIR}/.venv/bin/python" "${SCRIPT_DIR}/.venv/Scripts/python.exe"; do
        [[ -x "${cand}" ]] && { printf '%s\n' "${cand}"; return; }
    done
    if command -v python3 >/dev/null 2>&1; then printf 'python3\n'; return; fi
    if command -v python  >/dev/null 2>&1; then printf 'python\n';  return; fi
    die "no Python interpreter found; set PE_PYTHON=/path/to/python"
}

# ── preflight ────────────────────────────────────────────────────────────────────
preflight() {
    local python="$1"

    step "python"
    "${python}" -c 'import sys; assert sys.version_info>=(3,10), sys.version; print(sys.version.split()[0])' \
        >/dev/null 2>&1 || die "python >= 3.10 required (got: $("${python}" --version 2>&1))"
    ok "interpreter: ${python} ($("${python}" --version 2>&1))"

    step "dependencies"
    "${python}" -c 'import yaml' 2>/dev/null \
        || die "missing PyYAML (pip install -r requirements-smoke.txt)"
    ok "PyYAML"
    if [[ "${MODE}" == "check" ]]; then
        ok "torch/transformers/peft not required for --check (skipped)"
    else
        "${python}" -c 'import torch, transformers, peft' 2>/dev/null \
            || die "training deps missing (pip install torch transformers peft)"
        ok "torch / transformers / peft"
    fi

    step "config"
    [[ -f "${PE_CONFIG}" ]] || die "config not found: ${PE_CONFIG}"
    ok "config: ${PE_CONFIG}"

    step "frozen data + bundle SHA-256 gate"
    [[ -f "${SCRIPT_DIR}/data/train.jsonl" ]]      || die "missing data/train.jsonl"
    [[ -f "${SCRIPT_DIR}/data/validation.jsonl" ]] || die "missing data/validation.jsonl"
    [[ -f "${SCRIPT_DIR}/manifests/manifest.json" ]] || die "missing manifests/manifest.json"
    if [[ "${PE_SKIP_HASH}" == "1" ]]; then
        warn "PE_SKIP_HASH=1 — frozen-bundle SHA-256 gate SKIPPED (debug only)"
    else
        # Authoritative stdlib verifier: re-checks manifest hashes + SHA256SUMS,
        # counts, schema, cross-split and Formal-500 leakage; exits non-zero on drift.
        "${python}" "${SCRIPT_DIR}/validate_dataset.py" --config "${PE_CONFIG}" \
            || die "frozen bundle failed validation (abort)"
        ok "validate_dataset.py passed (hashes + counts + no leakage)"
        # trainer's own frozen-train-hash gate (stdlib, no torch)
        "${python}" "${SCRIPT_DIR}/train.py" --check --config "${PE_CONFIG}" >/dev/null \
            || die "train.py --check failed (abort)"
        ok "train.py --check passed (contract + frozen train hash)"
    fi

    step "disk"
    local out_dir
    out_dir="$("${python}" - "${PE_CONFIG}" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
d = Path(cfg["train"]["output_dir"])
print(d if d.is_absolute() else Path(sys.argv[1]).resolve().parent / d)
PY
)"
    mkdir -p "${out_dir}"
    if command -v df >/dev/null 2>&1; then
        local free_mb
        free_mb="$(df -Pm "${out_dir}" | awk 'NR==2 {print $4}')"
        [[ -n "${free_mb}" && "${free_mb}" -ge "${PE_MIN_DISK_MB}" ]] \
            || die "free disk ${free_mb:-?} MiB < PE_MIN_DISK_MB=${PE_MIN_DISK_MB} at ${out_dir}"
        ok "${free_mb} MiB free at ${out_dir} (>= ${PE_MIN_DISK_MB})"
    else
        warn "df not available — disk check skipped"
    fi

    step "GPU"
    if [[ "${MODE}" == "check" && "${PE_ALLOW_CPU}" == "1" ]]; then
        warn "PE_ALLOW_CPU=1 — GPU check skipped"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        local free_gpu
        free_gpu="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
        if [[ -n "${free_gpu}" && "${free_gpu}" -ge "${PE_MIN_GPU_MEM_MB}" ]]; then
            ok "${free_gpu} MiB free GPU memory (>= ${PE_MIN_GPU_MEM_MB})"
        elif [[ "${PE_ALLOW_CPU}" == "1" ]]; then
            warn "free GPU memory ${free_gpu:-0} MiB < ${PE_MIN_GPU_MEM_MB} but PE_ALLOW_CPU=1"
        else
            die "free GPU memory ${free_gpu:-0} MiB < PE_MIN_GPU_MEM_MB=${PE_MIN_GPU_MEM_MB} (set PE_ALLOW_CPU=1 to override)"
        fi
    else
        [[ "${PE_ALLOW_CPU}" == "1" ]] || die "no GPU (nvidia-smi not found); set PE_ALLOW_CPU=1 to run on CPU"
        warn "nvidia-smi not found but PE_ALLOW_CPU=1 — proceeding on CPU"
    fi
}

# ── drive ────────────────────────────────────────────────────────────────────────
PY="$(resolve_python)"

if [[ "${1:-}" == "--check" ]]; then
    MODE="check"
    printf 'Parameter-extraction fine-tune — ENVIRONMENT CHECK (no training)\n'
    preflight "${PY}"
    printf '\nENV CHECK OK — no training was started.\n'
    exit 0
fi

MODE="run"
printf 'Parameter-extraction fine-tune — preflight, then train.py %s\n' "${*:-<real run>}"
preflight "${PY}"
printf '\nPREFLIGHT OK — handing off to train.py %s\n\n' "${*:-<real run>}"
exec "${PY}" "${SCRIPT_DIR}/train.py" "$@"
