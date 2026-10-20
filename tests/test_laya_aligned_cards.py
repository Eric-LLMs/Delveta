"""Laya trained-card ↔ production-renderer alignment + single-source pin.

The LayaChoice-v2 option cards the model trains on are a DERIVED value of the
production renderer: ``cap_router.card_renderer.render_card`` against the live
Registry. ``fine_tuning/laya_finetune/V2/build/v2_refresh_cards_from_registry.py``
writes them to ``data/aligned/`` (the frozen ``data/`` bundle is left untouched).
These tests pin that contract without a database and detect drift when one is
reachable:

  1. every aligned row keeps the frozen 14-key schema and passes the SAME
     ``adapt_row`` the loader uses (so the loader can load it);
  2. every aligned card is byte-identical to ``render_card`` of the snapshot
     entry it was rendered from;
  3. the REJECT card — the single-source pin — equals the production constant
     AND the frozen manifest's recorded card;
  4. when the live Registry is reachable, the snapshot still equals it (i.e. the
     aligned cards are NOT stale); an honest skip otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import exc as sa_exc

_REPO = Path(__file__).resolve().parents[1]
_V2 = _REPO / "fine_tuning" / "laya_finetune" / "V2"
if str(_V2) not in sys.path:
    sys.path.insert(0, str(_V2))

from core.application.chat.intent_funnel.cap_router.card_renderer import (  # noqa: E402
    REJECT_CARD,
    REJECT_LABEL,
    render_card,
)
from core.application.chat.intent_funnel.registry import list_capabilities  # noqa: E402
from core.infrastructure.db import SessionLocal  # noqa: E402

import layachoice_v2_dataset as D  # noqa: E402

ALIGNED = _V2 / "data" / "aligned"
FROZEN_MANIFEST = _V2 / "data" / "manifest.json"
SPLITS = ("train", "val", "test")

pytestmark = pytest.mark.skipif(
    not (ALIGNED / "v2_registry_snapshot.json").is_file(),
    reason="aligned bundle not built (run v2_refresh_cards_from_registry.py)",
)


def _head(card: str) -> str:
    first = card.split("\n", 1)[0].strip()
    return first[4:] if first.startswith("### ") else ""


def _rows(split: str) -> list[dict]:
    path = ALIGNED / f"v2_{split}.aligned.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _snapshot() -> dict:
    return json.loads((ALIGNED / "v2_registry_snapshot.json").read_text(encoding="utf-8"))


def _stub(entry: dict) -> SimpleNamespace:
    """A render_card-compatible view of one snapshot entry (same fields the
    production renderer reads)."""
    return SimpleNamespace(
        capability_id=entry["capability_id"],
        tool_binding=entry["tool_binding"],
        description=entry["description"],
        negatives=entry["negatives"],
        parameters=entry["parameters"],
    )


@pytest.mark.parametrize("split", SPLITS)
def test_aligned_rows_keep_the_frozen_schema(split: str) -> None:
    """The aligned row IS the frozen row with ``options`` swapped: the exact
    14-key schema, accepted by the very adapter the loader calls."""
    rows = _rows(split)
    assert rows, f"{split}: empty aligned split"
    for row in rows:
        assert set(row) == D.ROW_KEYS, f"{row['id']}: schema drift"
        D.adapt_row(row)   # raises on any schema / label drift


@pytest.mark.parametrize("split", SPLITS)
def test_aligned_cards_equal_render_card(split: str) -> None:
    """Every aligned option is byte-identical to the production renderer's output
    for the snapshot entry it came from (REJECT == the frozen REJECT card)."""
    snapshot = _snapshot()
    for row in _rows(split):
        for cid, card in zip(row["order"], row["options"]):
            if cid == REJECT_LABEL:
                expected = REJECT_CARD
            else:
                assert cid in snapshot, f"{row['id']}: {cid} missing from snapshot"
                expected = render_card(_stub(snapshot[cid]))
            assert card == expected, f"{row['id']}: card for {cid} != render_card"


def test_reject_card_is_the_single_source() -> None:
    """The REJECT option is defined once (the production constant); the frozen
    manifest must record that exact string, and every aligned REJECT slot must
    carry it."""
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["reject"]["card"] == REJECT_CARD
    for split in SPLITS:
        for row in _rows(split):
            assert row["options"][row["reject_index"]] == REJECT_CARD


async def test_snapshot_still_matches_the_live_registry() -> None:
    """Drift detector: the aligned cards are derived from the snapshot, so the
    snapshot must still equal the live Registry. A mismatch means the Registry
    changed since the aligned split was rendered — regenerate it."""
    try:
        entries = await list_capabilities(session_factory=SessionLocal)
    except (sa_exc.SQLAlchemyError, OSError) as exc:  # DB down — honest skip
        pytest.skip(f"live Registry unreachable: {exc!r}")
    live = {
        e.capability_id: {
            "capability_id": e.capability_id,
            "tool_binding": e.tool_binding,
            "description": e.description,
            "negatives": list(e.negatives),
            "parameters": e.parameters,
        }
        for e in entries if e.enabled
    }
    snapshot = _snapshot()
    drifted = sorted(c for c in snapshot if live.get(c) != snapshot[c])
    assert not drifted, (
        f"Registry drifted from the aligned snapshot for {drifted}; "
        "re-run fine_tuning/laya_finetune/data_versions/build/v2_refresh_cards_from_registry.py")
