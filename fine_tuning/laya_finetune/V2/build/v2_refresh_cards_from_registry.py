#!/usr/bin/env python3
"""Refresh the Laya option cards from the LIVE Registry, through the SAME
renderer production uses.

Why: the frozen V2 option cards were produced by an earlier, scratch build path
(`logs/_laya_choice_row.card_text` fed by `logs/_cap_only_eval.load_entries`) that
(a) did not normalize the Registry JSON-Schema parameter shape and (b) was frozen
before registry migrations 0016-0023 changed several descriptions. Production
renders from the live Registry via `cap_router.card_renderer.render_card`, so the
two faces drifted.

This script makes the training card a DERIVED value of the production renderer:
for every option it re-renders `render_card(entry)` from the production loader
(`registry.store.list_capabilities` -> normalized `CapabilityEntry`), then writes
NEW aligned splits (never overwriting the frozen `v2_*.jsonl`). Everything else
in the row (state / order / gold_index / shuffle metadata) is carried through
unchanged.

Every aligned row keeps the frozen 14-key schema (it is the frozen row with only
``options`` swapped), so the existing loader adapter accepts it unchanged. The
provenance marker lives OUTSIDE the row, in the report/snapshot, never as an
extra row key.

Outputs (new files, ``data/aligned/`` — the frozen ``data/`` top level is left
byte-for-byte untouched, so the bundle's ``SHA256SUMS`` exact-file check keeps
passing):
  data/aligned/v2_{train,val,test}.aligned.jsonl   rows with refreshed cards
  data/aligned/v2_card_refresh_report.json         frozen-vs-live drift + the
                                                   generator/commit/sha provenance
  data/aligned/v2_registry_snapshot.json           the normalized entries the
                                                   cards were rendered from (so
                                                   the training card is
                                                   reproducible and byte-checkable
                                                   without a live DB)
  data/aligned/SHA256SUMS                          sha256 of every aligned file

Read-only w.r.t. production code; never touches the frozen dataset.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_V2, _ROOT, os.path.join(_ROOT, "packages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.application.chat.intent_funnel.cap_router.card_renderer import (  # noqa: E402
    REJECT_CARD,
    REJECT_LABEL,
    render_card,
)
from core.application.chat.intent_funnel.registry import store  # noqa: E402

import layachoice_v2_dataset as D  # noqa: E402

DATA = os.path.join(_HERE, "..", "data")
ALIGNED = os.path.join(DATA, "aligned")
SPLITS = ("train", "val", "test")

GENERATOR = "v2_refresh_cards_from_registry.py"
GENERATOR_VERSION = 2


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return ""


def _head(card: str) -> str:
    first = card.split("\n", 1)[0].strip()
    return first[4:] if first.startswith("### ") else ""


async def _entries() -> dict:
    caps = await store.list_capabilities()
    return {e.capability_id: e for e in caps if e.enabled}


def main() -> None:
    os.makedirs(ALIGNED, exist_ok=True)
    entries = asyncio.run(_entries())
    drift: dict[str, dict] = {}
    written: list[str] = []
    for split in SPLITS:
        src = os.path.join(DATA, f"v2_{split}.jsonl")
        dst = os.path.join(ALIGNED, f"v2_{split}.aligned.jsonl")
        out = []
        with open(src, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                new_opts = []
                for opt in row["options"]:
                    cid = _head(opt)
                    if cid == REJECT_LABEL:
                        new = REJECT_CARD
                    else:
                        entry = entries.get(cid)
                        new = render_card(entry) if entry is not None else opt
                    if cid and new != opt and cid not in drift:
                        drift[cid] = {"frozen": opt, "live": new}
                    new_opts.append(new)
                # The aligned row IS the frozen row with `options` swapped: no new
                # key is added, so `adapt_row` (and thus the V2 loader) accepts it.
                row = dict(row)
                row["options"] = new_opts
                if set(row) != D.ROW_KEYS:
                    raise RuntimeError(
                        f"{row.get('id')}: aligned row schema drift "
                        f"(missing={sorted(D.ROW_KEYS - set(row))}, "
                        f"extra={sorted(set(row) - D.ROW_KEYS)})")
                D.adapt_row(row)   # full schema + label guard, same adapter the loader uses
                out.append(row)
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        written.append(dst)
        print(f"[refresh] {split}: {len(out)} rows -> {os.path.relpath(dst, _ROOT)}")

    # The exact entries the cards were rendered from. Committing this alongside
    # the aligned splits makes "training card == render_card(entry)" checkable
    # without a live DB (the acceptance test renders from THIS snapshot).
    snapshot = {
        cid: {
            "capability_id": e.capability_id,
            "tool_binding": e.tool_binding,
            "description": e.description,
            "negatives": list(e.negatives),
            "parameters": e.parameters,
        }
        for cid, e in sorted(entries.items())
    }
    sp = os.path.join(ALIGNED, "v2_registry_snapshot.json")
    with open(sp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2)
    print(f"[refresh] snapshot ({len(snapshot)} entries) -> {os.path.relpath(sp, _ROOT)}")

    report = {
        "renderer": "cap_router.card_renderer.render_card",
        "entry_source": "registry.store.list_capabilities (normalized CapabilityEntry)",
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "source_commit": _source_commit(),
        "registry_snapshot": "v2_registry_snapshot.json",
        "registry_snapshot_sha256": _sha256_file(sp),
        "frozen_bundle": {"dir": "data", "note": "unchanged; cards re-rendered from "
                          "the live Registry, frozen rows preserved byte-for-byte "
                          "except `options`"},
        "drifted_capabilities": sorted(drift),
        "n_drifted": len(drift),
        "drift": drift,
        "aligned_files": {os.path.basename(p): _sha256_file(p) for p in written},
    }
    rp = os.path.join(ALIGNED, "v2_card_refresh_report.json")
    with open(rp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    written.append(rp)
    written.append(sp)
    print(f"[refresh] drifted capabilities ({len(drift)}): {sorted(drift)}")
    print(f"[refresh] report -> {os.path.relpath(rp, _ROOT)}")

    sums = [f"{_sha256_file(p)}  {os.path.basename(p)}" for p in sorted(written)]
    with open(os.path.join(ALIGNED, "SHA256SUMS"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("\n".join(sums) + "\n")
    print(f"[refresh] wrote SHA256SUMS ({len(written)} files)")


if __name__ == "__main__":
    main()
