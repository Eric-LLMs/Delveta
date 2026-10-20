"""V4 dynamic-option dataset — the frozen V2 contract, with K+1 options.

V4 is allowed ONE difference from LayaChoice-v2: the option count is the real
K capability cards + one REJECT, instead of a fixed 3 + 1. Everything else is
the V2 row contract:

  * exactly the 14 frozen keys;
  * ``part == "p2"`` (real recall) and no threshold field;
  * reject rows carry ``gold_capability_id = None``;
  * one REJECT, shuffled with the capability cards using the V2 seed prefix;
  * ``gold_index`` and ``reject_index`` are validated against the ACTUAL
    option count, not 4.

These tests read the installed V2 bundle, run the SAME adapter/loader the
training path uses, and pin every option card to the production renderer.
"""
from __future__ import annotations

import json
import random
import sys
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
PART = "p2"
SHUFFLE_SEED_PREFIX = "reject-shuffle-v1"
SNAPSHOT = _REPO / "tests" / "fixtures" / "laya_v4_registry_snapshot.json"
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
    reason="V2 dynamic-option bundle not present",
)


def _rows(tag: str, split: str) -> list[dict]:
    """The formal flat bundle is ``{split}.jsonl``; archives keep ``v2_{split}``."""
    path = D.split_path(split, _bundle(tag))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _audit(tag: str) -> dict[str, dict]:
    path = _VERSIONS / tag / "construction_audit.jsonl"
    return {r["id"]: r for r in
            (json.loads(line) for line in
             path.read_text(encoding="utf-8").splitlines() if line.strip())}


def _repairs(tag: str) -> dict[str, dict]:
    """The in-place K=0 -> K=1 rows, keyed by row id.

    ``construction_audit.jsonl`` deliberately keeps the pre-repair evidence, so
    a repaired row's final ``order`` is one option longer than the audit says.
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
def test_rows_are_exactly_the_frozen_v2_schema(tag: str, split: str) -> None:
    rows = _rows(tag, split)
    assert rows, f"{tag}/{split}: empty"
    for row in rows:
        assert set(row) == D.ROW_KEYS, f"{row['id']}: schema drift"
        assert row["part"] == PART
        assert "threshold" not in row
        D.adapt_row(row)  # the loader's own schema/index/permutation checks


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_dynamic_option_count_and_reject_semantics(tag: str) -> None:
    counts = set()
    for split in SPLITS:
        for row in _rows(tag, split):
            n = len(row["order"])
            counts.add(n)
            assert 1 <= n <= 4
            assert len(row["options"]) == n
            assert row["order"].count(REJECT_LABEL) == 1
            assert row["reject_index"] == row["order"].index(REJECT_LABEL)
            assert sorted(row["shuffle"]["permutation"]) == list(range(n))
            if row["target_kind"] == "reject":
                assert row["gold_capability_id"] is None
                assert row["gold_index"] == row["reject_index"]
                assert row["order"][row["gold_index"]] == REJECT_LABEL
            else:
                assert row["gold_capability_id"] == row["order"][row["gold_index"]]
                assert row["gold_index"] != row["reject_index"]
    assert len(counts) > 1, f"{tag}: expected real dynamic option counts, saw {counts}"


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_real_loader_reads_the_v2_bundle(tag: str) -> None:
    """``load_split`` + ``frozen_calibration`` accept the bundle unchanged."""
    data_dir = _bundle(tag)
    manifest = D.read_manifest(data_dir)
    assert manifest["bench"] == S.V2_BENCH
    assert manifest["instructions"] == INSTRUCTIONS
    assert manifest["option_slots"]["rule"] == "K+1"
    assert manifest["option_slots"]["min"] == 1
    assert manifest["option_slots"]["max"] == 4
    assert set(manifest["training_schema_keys"]) == D.ROW_KEYS
    assert manifest["reject"]["card"] == REJECT_CARD

    train = D.load_split("train", data_dir)
    D.load_split("val", data_dir)
    D.load_split("test", data_dir)
    calib, fit = D.frozen_calibration(train, manifest)
    assert {e.id for e in calib} == set(manifest["calibration"]["ids"])
    assert {e.id for e in calib}.isdisjoint({e.id for e in fit})


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_audit_carries_the_threshold_and_original_gold(tag: str) -> None:
    audit = _audit(tag)
    manifest = json.loads((_bundle(tag) / "manifest.json").read_text(encoding="utf-8"))
    repair_counts = manifest["repair"]["repaired_by_split"]
    seen = 0
    repaired_by_split: Counter[str] = Counter()
    for split in SPLITS:
        for row in _rows(tag, split):
            a = audit[row["id"]]
            seen += 1
            assert a["threshold"] == THRESHOLDS[tag]
            # A source no-action row has no gold capability; an action row must
            # carry the original gold even when the model-facing label is
            # REJECT (gold not retrieved).
            if a["is_action"]:
                assert a["source_gold_capability_id"] is not None
                if row["target_kind"] == "capability":
                    assert a["source_gold_capability_id"] == row[
                        "gold_capability_id"]
            else:
                assert a["source_gold_capability_id"] is None
                assert row["target_kind"] == "reject"

            # The audit is the PRE-repair view. An action row with zero recalled
            # candidates and a valid source gold was repaired in place by
            # injecting that gold, so the row has 2 options while the audit
            # still records option_count=1.
            repaired = bool(a["is_action"] and a["candidate_count"] == 0
                            and a["source_gold_capability_id"] is not None)
            if repaired:
                assert a["option_count"] == 1
                assert len(row["order"]) == 2
                assert row["target_kind"] == "capability"
                assert row["gold_capability_id"] == a["source_gold_capability_id"]
                repaired_by_split[split] += 1
            else:
                assert a["candidate_count"] + 1 == len(row["order"])
                assert a["option_count"] == len(row["order"])
    assert seen == sum(len(_rows(tag, s)) for s in SPLITS)
    assert dict(repaired_by_split) == repair_counts


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_cards_are_the_live_production_renderer(tag: str) -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for split in SPLITS:
        for row in _rows(tag, split):
            for cid, card in zip(row["order"], row["options"]):
                expected = (REJECT_CARD if cid == REJECT_LABEL
                            else render_card(_snapshot_stub(snapshot[cid])))
                assert card == expected, f"{row['id']}: {cid} card drift"


@pytest.mark.parametrize("tag", sorted(THRESHOLDS))
def test_shuffle_is_the_v2_rule(tag: str) -> None:
    assert SHUFFLE_SEED_PREFIX == "reject-shuffle-v1"
    positions = set()
    for split in SPLITS:
        for row in _rows(tag, split):
            prefix = row["shuffle"]["seed_prefix"]
            if prefix.startswith("k0-to-k1-inplace-v1:"):
                # In-place repair shuffles the two-option base order
                # [gold capability, REJECT] with a per-threshold/per-split seed.
                assert prefix == (f"k0-to-k1-inplace-v1:"
                                  f"{THRESHOLDS[tag]}:{split}")
                assert len(row["order"]) == 2
            else:
                assert prefix == SHUFFLE_SEED_PREFIX
            perm = list(range(len(row["order"])))
            random.Random(f"{prefix}:{row['id']}").shuffle(perm)
            assert row["shuffle"] == {"seed_prefix": prefix,
                                      "permutation": perm}
            positions.add(row["reject_index"])
    # REJECT is shuffled, not pinned to a position.
    assert len(positions) > 1, f"{tag}: REJECT position fixed at {positions}"


def _base_row(**overrides) -> dict:
    row = {
        "id": "x-p2", "source_query_id": "x", "split": "train", "part": "p2",
        "lang": "en", "state": "q", "order": ["REJECT"],
        "options": [REJECT_CARD], "gold_index": 0, "target_kind": "reject",
        "gold_capability_id": None, "reject_index": 0,
        "shuffle": {"seed_prefix": SHUFFLE_SEED_PREFIX, "permutation": [0]},
        "src_sha": "sha",
    }
    row.update(overrides)
    return row


def test_adapter_accepts_k0_to_k3_and_rejects_drift() -> None:
    for k in range(4):
        order = [f"cap-{i}" for i in range(k)] + [REJECT_LABEL]
        options = [f"card-{i}" for i in range(k)] + [REJECT_CARD]
        ex = D.adapt_row(_base_row(
            order=order, options=options, gold_index=k, reject_index=k,
            shuffle={"seed_prefix": SHUFFLE_SEED_PREFIX,
                     "permutation": list(range(k + 1))}))
        assert len(ex.options) == k + 1
        assert ex.target_index == k
        assert ex.gold_id == REJECT_LABEL

    five = [f"cap-{i}" for i in range(4)] + [REJECT_LABEL]
    with pytest.raises(ValueError, match="expected 1..4 options"):
        D.adapt_row(_base_row(order=five, options=five, gold_index=4,
                              reject_index=4,
                              shuffle={"seed_prefix": SHUFFLE_SEED_PREFIX,
                                       "permutation": list(range(5))}))
    with pytest.raises(ValueError, match="not a permutation"):
        D.adapt_row(_base_row(
            order=["cap-a", REJECT_LABEL],
            options=["card", REJECT_CARD], gold_index=0, reject_index=1,
            target_kind="capability", gold_capability_id="cap-a",
            shuffle={"seed_prefix": SHUFFLE_SEED_PREFIX,
                     "permutation": [0, 0]}))
    with pytest.raises(ValueError, match="reject row carries"):
        D.adapt_row(_base_row(gold_capability_id="cap-a"))
