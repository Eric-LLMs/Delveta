"""V4 dynamic-option dataset — the four-field supervised contract, with K+1 options.

Every row is exactly

    {"message_id": "<uuid4>", "language": "<lang>",
     "input": {"query": "<request text>",
               "options": ["<capability_id>: <card>", ...]},
     "label": <0-based index of the correct option>}

The checks pinned here are the V2 contract minus the metadata that is now
derived or file-level:

  * exactly the four frozen keys (and exactly ``query``/``options`` inside
    ``input``); no ``split``, no construction metadata;
  * ``message_id`` is a canonical uuid4, unique across the bundle, and never
    participates in the model input, the label, or calibration;
  * one REJECT card per row, rendered exactly like production;
  * ``label`` addresses the ACTUAL option count, not 4;
  * ``input.options`` is the exact production rendering (``"<id>: <card>"``)
    and round-trips through the adapter byte for byte;
  * ``split`` comes from the file name and is injected by the loader.

``message_id_map.jsonl`` is a local-only traceability artifact (never uploaded);
the construction-audit joins below use it and skip when it is absent. The row
contract itself never needs it.
"""
from __future__ import annotations

import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
_V2 = _REPO / "fine_tuning" / "laya_finetune" / "V2"
_DATA = _V2 / "data"
_VERSIONS = _REPO / "fine_tuning" / "laya_finetune" / "data_versions"
for _p in (_V2, _REPO / "packages"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layachoice_v2_dataset as D  # noqa: E402
import layachoice_v2_spec as S  # noqa: E402
from core.application.chat.intent_funnel.cap_router.card_renderer import (  # noqa: E402
    INSTRUCTIONS,
    REJECT_CARD,
    REJECT_LABEL,
    render_card,
)

THRESHOLDS = {"recall_threshold_082": 0.82, "recall_threshold_085": 0.85,
              "recall_threshold_088": 0.88}
SPLITS = ("train", "val", "test")
SHUFFLE_SEED_PREFIX = "reject-shuffle-v1"
SNAPSHOT = _REPO / "tests" / "fixtures" / "laya_v4_registry_snapshot.json"
MAP = _VERSIONS / "message_id_map.jsonl"
REPAIR_DETAIL = (_REPO / "data" / "laya_finetune" / "data_v4_build"
                 / "full_r1_k0_to_k1_inplace_audit"
                 / "k0_to_k1_repair_detail.jsonl")


def _bundle(tag: str) -> Path:
    """The installed flat ``data/`` holds 082; the other thresholds are archived
    beside ``V2/`` under ``data_versions/``."""
    flat = _DATA / "manifest.json"
    if tag == "recall_threshold_082" and flat.is_file():
        return _DATA
    return _VERSIONS / tag


pytestmark = pytest.mark.skipif(
    not (_bundle("recall_threshold_082") / "manifest.json").is_file(),
    reason="dynamic-option bundle not present",
)


def _rows(tag: str, split: str) -> list[dict]:
    """The formal flat bundle is ``{split}.jsonl``; archives keep ``v2_{split}``."""
    path = D.split_path(split, _bundle(tag))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _row_map(tag: str) -> dict[str, dict]:
    """Local-only new-id -> old-id map used by the construction-audit joins."""
    if not MAP.is_file():
        pytest.skip(f"message id map not present: {MAP}")
    threshold = THRESHOLDS[tag]
    return {r["message_id"]: r for r in
            (json.loads(line) for line in
             MAP.read_text(encoding="utf-8").splitlines() if line.strip())
            if r["threshold"] == threshold}


def _audit(tag: str) -> dict[str, dict]:
    path = _VERSIONS / tag / "construction_audit.jsonl"
    return {r["id"]: r for r in
            (json.loads(line) for line in
             path.read_text(encoding="utf-8").splitlines() if line.strip())}


def _repairs(tag: str) -> dict[str, dict]:
    """The in-place K=0 -> K=1 rows, keyed by old row id.

    ``construction_audit.jsonl`` deliberately keeps the pre-repair evidence, so
    a repaired row's final ``options`` is one option longer than the audit says.
    The repair detail is the reconciliation record.
    """
    if not REPAIR_DETAIL.is_file():
        pytest.skip(f"repair detail not present: {REPAIR_DETAIL}")
    threshold = THRESHOLDS[tag]
    return {r["id"]: r for r in
            (json.loads(line) for line in
             REPAIR_DETAIL.read_text(encoding="utf-8").splitlines() if line.strip())
            if r["threshold"] == threshold}


def _snapshot_stub(entry: dict) -> SimpleNamespace:
    return SimpleNamespace(
        capability_id=entry["capability_id"],
        tool_binding=entry["tool_binding"],
        description=entry["description"],
        negatives=entry["negatives"],
        parameters=entry["parameters"],
    )


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
@pytest.mark.parametrize("split", SPLITS)
def test_rows_are_exactly_the_four_field_schema(tag: str, split: str) -> None:
    rows = _rows(tag, split)
    assert rows, f"{tag}/{split}: empty"
    for row in rows:
        assert set(row) == D.ROW_KEYS, f"{row['message_id']}: schema drift"
        assert set(row["input"]) == D.INPUT_KEYS
        assert "split" not in row
        assert isinstance(row["label"], int) and not isinstance(row["label"], bool)
        parsed = uuid.UUID(row["message_id"])
        assert parsed.version == 4
        assert str(parsed) == row["message_id"]
        ex = D.adapt_row(row, split)  # the loader's own schema/label checks
        assert ex.split == split
        assert ex.rendered_options == tuple(row["input"]["options"])


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_dynamic_option_count_and_reject_semantics(tag: str) -> None:
    counts = set()
    for split in SPLITS:
        for row in _rows(tag, split):
            ex = D.adapt_row(row, split)
            n = len(ex.options)
            counts.add(n)
            assert 1 <= n <= 4
            assert len(row["input"]["options"]) == n
            assert ex.order.count(REJECT_LABEL) == 1
            assert len(set(ex.order)) == n
            assert 0 <= row["label"] < n
            if ex.order[row["label"]] == REJECT_LABEL:
                assert ex.target_kind == "reject"
                assert ex.gold_capability_id is None
                assert ex.gold_index == ex.reject_index
            else:
                assert ex.target_kind == "capability"
                assert ex.gold_capability_id == ex.order[row["label"]]
                assert ex.gold_index != ex.reject_index
    assert len(counts) > 1, f"{tag}: expected real dynamic option counts, saw {counts}"


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_real_loader_reads_the_bundle(tag: str) -> None:
    """``load_split`` + ``frozen_calibration`` accept the bundle unchanged."""
    data_dir = _bundle(tag)
    manifest = D.read_manifest(data_dir)
    assert manifest["bench"] == S.V2_BENCH
    assert manifest["instructions"] == INSTRUCTIONS
    assert manifest["option_slots"]["rule"] == "K+1"
    assert manifest["option_slots"]["min"] == 1
    assert manifest["option_slots"]["max"] == 4
    assert set(manifest["training_schema_keys"]) == D.ROW_KEYS
    assert manifest["training_fields"] == ["input.query", "input.options", "label"]
    assert sorted(manifest["not_used_in_training"]) == ["language", "message_id"]
    assert manifest["reject"]["card"] == REJECT_CARD

    train = D.load_split("train", data_dir)
    D.load_split("val", data_dir)
    D.load_split("test", data_dir)
    calib, fit = D.frozen_calibration(train, manifest)
    assert len(calib) == manifest["calibration"]["calibration_rows"]
    assert [e.message_id for e in calib] == [
        train[i].message_id for i in manifest["calibration"]["indices"]]
    assert {e.message_id for e in calib}.isdisjoint({e.message_id for e in fit})


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_message_id_is_unique_across_splits(tag: str) -> None:
    ids = [row["message_id"] for split in SPLITS for row in _rows(tag, split)]
    assert len(ids) == len(set(ids)), f"{tag}: duplicate message_id"


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_audit_carries_the_threshold_and_original_gold(tag: str) -> None:
    row_map = _row_map(tag)
    audit = _audit(tag)
    manifest = json.loads((_bundle(tag) / "manifest.json").read_text(encoding="utf-8"))
    repair_counts = manifest["repair"]["repaired_by_split"]
    seen = 0
    repaired_by_split: Counter[str] = Counter()
    for split in SPLITS:
        for row in _rows(tag, split):
            rec = row_map[row["message_id"]]
            assert rec["split"] == split
            ex = D.adapt_row(row, split)
            a = audit[rec["old_id"]]
            seen += 1
            assert a["threshold"] == THRESHOLDS[tag]
            # A source no-action row has no gold capability; an action row must
            # carry the original gold even when the model-facing label is
            # REJECT (gold not retrieved).
            if a["is_action"]:
                assert a["source_gold_capability_id"] is not None
                if ex.target_kind == "capability":
                    assert a["source_gold_capability_id"] == ex.gold_capability_id
            else:
                assert a["source_gold_capability_id"] is None
                assert ex.target_kind == "reject"

            # The audit is the PRE-repair view. An action row with zero recalled
            # candidates and a valid source gold was repaired in place by
            # injecting that gold, so the row has 2 options while the audit
            # still records option_count=1.
            repaired = bool(a["is_action"] and a["candidate_count"] == 0
                            and a["source_gold_capability_id"] is not None)
            if repaired:
                assert a["option_count"] == 1
                assert len(ex.options) == 2
                assert ex.target_kind == "capability"
                assert ex.gold_capability_id == a["source_gold_capability_id"]
                repaired_by_split[split] += 1
            else:
                assert a["candidate_count"] + 1 == len(ex.options)
                assert a["option_count"] == len(ex.options)
    assert seen == sum(len(_rows(tag, s)) for s in SPLITS)
    assert dict(repaired_by_split) == repair_counts


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_cards_are_the_live_production_renderer(tag: str) -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for split in SPLITS:
        for row in _rows(tag, split):
            ex = D.adapt_row(row, split)
            for cid, card in zip(ex.order, ex.options):
                expected = (REJECT_CARD if cid == REJECT_LABEL
                            else render_card(_snapshot_stub(snapshot[cid])))
                assert card == expected, f"{row['message_id']}: {cid} card drift"


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_option_order_is_preserved_and_reject_is_shuffled(tag: str) -> None:
    positions = set()
    for split in SPLITS:
        for row in _rows(tag, split):
            ex = D.adapt_row(row, split)
            # The stored option list is the model input; the adapter round-trips it.
            assert ex.rendered_options == tuple(row["input"]["options"])
            assert len(set(ex.order)) == len(ex.order)
            positions.add(ex.reject_index)
    # REJECT is shuffled, not pinned to a position.
    assert len(positions) > 1, f"{tag}: REJECT position fixed at {positions}"


def _base_row(**overrides) -> dict:
    row = {
        "message_id": str(uuid.uuid4()), "language": "en",
        "input": {"query": "q", "options": [f"{REJECT_LABEL}: {REJECT_CARD}"]},
        "label": 0,
    }
    row.update(overrides)
    return row


def test_adapter_accepts_k0_to_k3_and_rejects_drift() -> None:
    for k in range(4):
        options = [f"cap-{i}: card-{i}" for i in range(k)] + [f"{REJECT_LABEL}: {REJECT_CARD}"]
        ex = D.adapt_row(_base_row(input={"query": "q", "options": options},
                                   label=k), "train")
        assert len(ex.options) == k + 1
        assert ex.target_index == k
        assert ex.gold_id == REJECT_LABEL
        assert ex.is_reject
        assert ex.gold_capability_id is None

    five = [f"cap-{i}: card-{i}" for i in range(4)] + [f"{REJECT_LABEL}: {REJECT_CARD}"]
    with pytest.raises(ValueError, match="expected 1..4 options"):
        D.adapt_row(_base_row(input={"query": "q", "options": five}, label=4), "train")
    with pytest.raises(ValueError, match="out of range"):
        D.adapt_row(_base_row(label=1), "train")
    with pytest.raises(ValueError, match="REJECT must appear exactly once"):
        D.adapt_row(_base_row(input={"query": "q", "options": ["cap-a: card-a"]}),
                    "train")
    with pytest.raises(ValueError, match="duplicate capability ids"):
        D.adapt_row(_base_row(input={"query": "q",
                                     "options": ["cap-a: card-a",
                                                 "cap-a: card-b",
                                                 f"{REJECT_LABEL}: {REJECT_CARD}"]}),
                    "train")
    with pytest.raises(ValueError, match="not canonical"):
        D.adapt_row(_base_row(input={"query": "q",
                                     "options": [f"cap-a:card-a",
                                                 f"{REJECT_LABEL}: {REJECT_CARD}"]}),
                    "train")
    with pytest.raises(ValueError, match="canonical uuid4"):
        D.adapt_row(_base_row(message_id="not-a-uuid"), "train")
    with pytest.raises(ValueError, match="schema drift"):
        D.adapt_row(_base_row(extra=1), "train")
    with pytest.raises(ValueError, match="input must carry exactly"):
        D.adapt_row(_base_row(input={"query": "q"}), "train")
    with pytest.raises(ValueError, match="unknown split"):
        D.adapt_row(_base_row(), "heldout")


def test_capability_row_derives_its_gold() -> None:
    ex = D.adapt_row(_base_row(
        input={"query": "q", "options": ["cap-a: card-a",
                                         f"{REJECT_LABEL}: {REJECT_CARD}"]},
        label=0), "val")
    assert ex.target_kind == "capability"
    assert ex.gold_capability_id == "cap-a"
    assert ex.gold_id == "cap-a"
    assert ex.reject_index == 1
    assert not ex.is_reject
