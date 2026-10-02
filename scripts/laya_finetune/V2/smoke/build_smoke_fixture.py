#!/usr/bin/env python3
"""STEP 7 — build a smoke fixture dataset from VERBATIM frozen V2 rows.

The smoke must push SPECIFIC cases through the real train/eval entry points, so
mode_train must see them as its 8 smoke rows. We therefore materialise a fixture
under .tmp/.../smoke/data/ whose train/test split IS the 8 chosen frozen rows
(copied byte-for-byte; only the metadata `split` tag is retagged so the loader's
per-split check holds). The manifest counts are reduced to match.

Coverage: capability target at gold_index 0/1/2/3, synthetic REJECT x2, natural
REJECT x2.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "scripts" / "laya_finetune" / "V2" / "data"
OUT = Path(__file__).resolve().parent / "data"

rows = {json.loads(l)["id"]: json.loads(l)
        for l in (SRC / "v2_train.jsonl").open(encoding="utf-8") if l.strip()}
audit = {json.loads(l)["id"]: json.loads(l)
         for l in (SRC / "v2_construction_audit.jsonl").open(encoding="utf-8")
         if l.strip() and json.loads(l)["split"] == "train"}

# ── pick the coverage rows ──────────────────────────────────────────────────
picked: list[dict] = []
by_index: dict[int, str] = {}
for cid, r in rows.items():
    if r["target_kind"] == "capability" and r["gold_index"] not in by_index:
        by_index[r["gold_index"]] = cid
    if len(by_index) == 4:
        break
for gi in sorted(by_index):
    picked.append(rows[by_index[gi]])

synth = [cid for cid, a in audit.items() if a.get("synthetic") and rows[cid]["target_kind"] == "reject"]
nat = [cid for cid, a in audit.items()
       if a.get("gold_absent_from_candidates") and rows[cid]["target_kind"] == "reject"]
# prefer distinct reject_index values for the two of each kind
def two_distinct(cids):
    out, seen = [], set()
    for cid in cids:
        if rows[cid]["reject_index"] not in seen:
            out.append(cid); seen.add(rows[cid]["reject_index"])
        if len(out) == 2:
            break
    return out
picked += [rows[c] for c in two_distinct(synth)]
picked += [rows[c] for c in two_distinct(nat)]

assert len(picked) == 8, f"expected 8 rows, got {len(picked)}"

report = Path(__file__).resolve().parent / "smoke_rows.json"
cover = []
for r in picked:
    a = audit[r["id"]]
    cover.append({"id": r["id"], "gold_index": r["gold_index"], "reject_index": r["reject_index"],
                  "target_kind": r["target_kind"], "gold_capability_id": r["gold_capability_id"],
                  "synthetic": a.get("synthetic"),
                  "gold_absent_from_candidates": a.get("gold_absent_from_candidates")})

OUT.mkdir(parents=True, exist_ok=True)
for split in ("train", "val", "test"):
    with (OUT / f"v2_{split}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in picked:
            rr = dict(r); rr["split"] = split          # retag for the fixture only
            fh.write(json.dumps(rr, ensure_ascii=False) + "\n")

manifest = {
    "bench": "LayaChoice-v2",
    "smoke_fixture": True,
    "note": "STEP 7 smoke fixture: 8 verbatim frozen train rows; manifest counts reduced.",
    "formal_card_view": "B_noprov",
    "instructions": "Which capability should handle the user's request?",
    "option_slots": 4,
    "reject": {"label": "REJECT", "id": "REJECT"},
    "gold_slot_distribution": {"train": {str(i): sum(1 for r in picked if r["gold_index"] == i)
                                         for i in range(4)}},
    "source_files": {},
    "rows": {"train": 8, "val": 8, "test": 8},
    "language": {"train": {"zh": sum(r["lang"] == "zh" for r in picked),
                           "en": sum(r["lang"] == "en" for r in picked)}},
    "calibration": {"seed": 20260922, "source_train_rows": 8, "calibration_rows": 0,
                    "effective_train_rows": 8, "indices": [], "ids": []},
    "base_checkpoint": {"repo_id": "convaiinnovations/laya",
                        "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
                        "subfolder": "multilingual"},
    "smoke": True,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
report.write_text(json.dumps(cover, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("fixture:", OUT)
for c in cover:
    kind = "capability" if c["target_kind"] == "capability" else (
        "synthetic_REJECT" if c["synthetic"] else "natural_REJECT")
    print(f"  {kind:16s} gold_index={c['gold_index']} reject_index={c['reject_index']} "
          f"gold_id={c['gold_capability_id'] or 'REJECT':16s} {c['id'][-24:]}")
