#!/usr/bin/env python3
"""LayaChoice-v2 — the frozen experiment spec (constants only, no logic).

Every value here is COPIED VERBATIM from the archived LayaChoice-v1 recipe
(``../V1/layachoice_finetune.py``). V2 is a NEW, independent task, but the token
budget, the base checkpoint, the seeds and the training hyper-parameters are the
same frozen values the V1 run was accepted under - V2 must differ from V1 ONLY in
the dataset/task, never by silently re-tuning the recipe.

Token budget (unchanged; the V2 audit proved it is sufficient at K=4):
  OPTION_MAX_TOKENS  per-option hard cap (stock laya.common uses 48, which would
                     cut every frozen card).
  HEAD_MAX_LEN       the budget the question head and every option SHARE, not a
                     length - below 16 tokens of slack the stock even-share
                     fallback shrinks the options and the head is cut.
  MAX_LEN            whole-sequence ceiling.

Base checkpoint (official original base, NOT a V1 checkpoint / NOT a fork):
  convaiinnovations/laya @ 55cf4c4e.../multilingual
"""
from __future__ import annotations

# ── input-token budget (train, eval and export all read these; they must not diverge) ─
OPTION_MAX_TOKENS = 256
HEAD_MAX_LEN = 768
MAX_LEN = 1024

# ── training recipe ─────────────────────────────────────────────────────────
EPOCHS = 4
MICRO_BATCH = 8
GRAD_ACCUM = 4
GROUP_SIZE = 4
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
WEIGHT_DECAY = 0.01
SIGMA_START = 0.4
SIGMA_END = 0.1
CE_WEIGHT = 1.0
W_SPH = 0.75
W_RPS = 1.0
CLIP = 1.0

# ── determinism / calibration ───────────────────────────────────────────────
SEED_BASE = 42
CALIB_SEED = 20260922
CALIB_MAX = 400
SMOKE_ROWS = 8
SMOKE_CALIB_ROWS = 5
FALLBACK_TEMP = 1.2

# ── base checkpoint (official original base) ────────────────────────────────
BASE_REPO = "convaiinnovations/laya"
BASE_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
BASE_SUBFOLDER = "multilingual"

# ── question contract ───────────────────────────────────────────────────────
DEFAULT_INSTRUCTIONS = "Which capability should handle the user's request?"
QID = "capability"
DECISION_CHUNK = 16

# ── V2 task shape (the only values that differ from V1) ─────────────────────
V2_BENCH = "LayaChoice-v2"
OPTION_SLOTS = 4                       # 3 capability cards + 1 REJECT card
REJECT_LABEL = "REJECT"                # the 4th option's label and capability id
