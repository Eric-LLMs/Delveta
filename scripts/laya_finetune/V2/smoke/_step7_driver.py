#!/usr/bin/env python3
"""STEP 7 verification driver — assertions the CLI smoke cannot express.

Covers, on the SAME fixture rows the real train/eval ran on:
  * loader + rendering + target shape/position per coverage case;
  * REJECT target/index correctness through the full chain;
  * base = the official PINNED revision; NOT a V1 checkpoint;
  * the optimizer step actually moved the weights;
  * save -> load -> eval is bit-identical to eval before save;
  * the driver's eval agrees with the CLI benchmark report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
SMK = Path(__file__).resolve().parent
V2 = REPO / "scripts" / "laya_finetune" / "V2"
sys.path.insert(0, str(V2))

import layachoice_v2_spec as S        # noqa: E402
import layachoice_v2_dataset as D     # noqa: E402
import layachoice_v2_render as R      # noqa: E402
import layachoice_v2_finetune as F    # noqa: E402
import layachoice_v2_eval as E        # noqa: E402

FIX = SMK / "data"
RUN = SMK / "run"
ok = True
def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and bool(cond)

# ── base checkpoint identity ─────────────────────────────────────────────────
base = F.base_model_dir()
print("[base] model_dir =", base)
check(S.BASE_REVISION in str(base), f"base dir is the PINNED revision {S.BASE_REVISION[:12]}…")
check(S.BASE_SUBFOLDER in str(base), f"base subfolder = {S.BASE_SUBFOLDER}")
check("V1" not in str(base), "base dir is NOT under the V1 archive")
check("V1" not in str(Path(R.__file__).resolve().parent),
      "V2 code dir is self-contained (no V1 path)")
ckpt = RUN / "epoch-1"
check("V1" not in str(ckpt), "checkpoint is the V2 smoke ckpt, not a V1 artifact")

# ── loader ───────────────────────────────────────────────────────────────────
manifest = D.read_manifest(FIX)
rows = D.load_split("train", FIX)
cover = json.loads((SMK / "smoke_rows.json").read_text(encoding="utf-8"))
print(f"\n[loader] {len(rows)} rows; coverage:")
for c in cover:
    kind = c["target_kind"] if c["target_kind"] == "capability" else (
        "synthetic_REJECT" if c["synthetic"] else "natural_REJECT")
    print(f"    {kind:16s} gold_index={c['gold_index']} reject_index={c['reject_index']}")
check(len(rows) == 8, "loader returned 8 adapted rows")
check({c["gold_index"] for c in cover} == {0, 1, 2, 3}, "target index set covers 0/1/2/3")
check(any(c["synthetic"] for c in cover), "synthetic REJECT present")
check(any(c["gold_absent_from_candidates"] for c in cover), "natural REJECT present")

# ── rendering + target shape/position (per row) ───────────────────────────────
tok = F.load_tokenizer(base)
instructions = manifest["instructions"]
print("\n[rendering/target]")
for ex in rows:
    it = R.build_item(tok, ex, instructions)
    tgt = it["target"]
    pos_ok = tgt[ex.gold_index] == 1.0 and sum(tgt) == 1.0 and len(tgt) == S.OPTION_SLOTS
    reject_ok = (ex.order[ex.gold_index] == S.REJECT_LABEL) if ex.is_reject \
        else (ex.order[ex.gold_index] == ex.gold_capability_id and ex.gold_index != ex.reject_index)
    check(pos_ok and len(it["markers"]) == S.OPTION_SLOTS and reject_ok,
          f"{ex.id[-18:]} kind={ex.target_kind:10s} gold_index={ex.gold_index} "
          f"target={tgt} markers=4 seq={len(it['ids'])}")

# ── optimizer step moved the weights (backward/step happened) ────────────────
print("\n[backward/optimizer]")
from safetensors.torch import load_file
base_state = load_file(str(base / "model.safetensors"))
ckpt_state = load_file(str(ckpt / "model.safetensors"))
moved = sum(1 for k in base_state if not torch.equal(base_state[k], ckpt_state[k]))
check(moved > 0, f"{moved}/{len(base_state)} tensors changed from base -> epoch-1 (optimizer stepped)")
meta = json.loads((ckpt / "train_meta.json").read_text(encoding="utf-8"))
check(meta["optimizer_updates"] == 1, f"train_meta optimizer_updates={meta['optimizer_updates']}")
check(meta["mean_loss"] == meta["mean_loss"], f"recorded mean_loss={meta['mean_loss']}")

# ── save -> load -> eval identical ───────────────────────────────────────────
print("\n[save/load/eval consistency]")
device = torch.device("cpu")
cfg = F.read_cfg(base, train=False)
model_a, _ = F.model_from_checkpoint(ckpt, base, device)          # "before save"
dec_a = F.decide(model_a, tok, rows, cfg, instructions, [1.0, 1.0, 1.0], device)
resave = SMK / "resave"
F.save_state_dict(model_a.state_dict(), resave / "model.safetensors")
model_b, _ = F.model_from_checkpoint(resave, base, device)        # "after load"
dec_b = F.decide(model_b, tok, rows, cfg, instructions, [1.0, 1.0, 1.0], device)
same = all(x["pred"] == y["pred"] and x["probs"] == y["probs"] and x["gold_index"] == y["gold_index"]
           for x, y in zip(dec_a, dec_b))
check(same, "eval(before save) == eval(after load) — bit-identical predictions & probabilities")

# ── REJECT correctness through eval ──────────────────────────────────────────
for d in dec_a:
    if d["target_kind"] == "reject":
        check(d["order"][d["gold_index"]] == S.REJECT_LABEL and d["gold_is_reject"]
              and d["gold_index"] == next(e.reject_index for e in rows if e.id == d["id"]),
              f"REJECT row {d['id'][-18:]} gold_index={d['gold_index']} carried correctly through eval")

# ── driver eval agrees with the CLI benchmark report ─────────────────────────
cli = json.loads((SMK / "benchmark" / "final_test.report.json").read_text(encoding="utf-8"))
rep = E.summarise(dec_a, arm="driver", split="test", temperature=[1.0, 1.0, 1.0],
                  card_view=manifest["formal_card_view"])
check(rep["top1"] == cli["top1"], f"driver top1 {rep['top1']} == CLI top1 {cli['top1']}")
check(rep["reject"] == cli["reject"], "driver REJECT block == CLI REJECT block")

print("\nSTEP 7 driver:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
