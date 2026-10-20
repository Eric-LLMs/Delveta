"""Regression tests for the four-field projection in the archived V4 builder.

``validate_in_build`` runs AFTER the threshold files are written, so any stale
reference to a pre-migration construction key aborts an otherwise complete
rebuild (the original bug raised ``KeyError: 'order'`` on the projected row).
These tests pin, without a database:

  1. the four-field projection of a construction record;
  2. the deterministic candidate-order rule (``reject-shuffle-v1``);
  3. that ``validate_in_build`` accepts valid K=0..3 records (no stale key);
  4. that projection drift is REPORTED in the checks, never raised.
"""
from __future__ import annotations

import copy
import importlib.util
import random
import uuid
from pathlib import Path

import pytest

pytest.importorskip("asyncpg")
pytest.importorskip("httpx")
pytest.importorskip("numpy")

_REPO = Path(__file__).resolve().parents[1]
_BUILD = (_REPO / "fine_tuning" / "laya_finetune" / "data_versions" /
          "dataset_construction" / "full_r1" / "scripts" / "build_dataset.py")

try:
    _spec = importlib.util.spec_from_file_location(
        "laya_build_dataset_under_test", _BUILD)
    B = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(B)
except ImportError as exc:  # archived builder's app deps unavailable
    pytest.skip(f"builder dependencies unavailable: {exc}", allow_module_level=True)

CAP_IDS = [f"cap-{i:02d}" for i in range(18)]
CARDS = {cid: f"### {cid}\nUse {cid}." for cid in CAP_IDS}
THRESHOLD = 0.82


def _source_row(key: str = "q-1", *, is_action: bool = False,
                gold: str | None = None, lang: str = "en",
                query: str = "hello world") -> dict:
    return {
        "_capability_id": gold or "",
        "_is_action": is_action,
        "_review_status": "verified",
        "_source_record_key": key,
        "_split": "train",
        "_lang": lang,
        "_lang_raw": lang.upper(),
        "_query": query,
        "_parameters": {},
        "_source": "v4",
        "_family_sig": "sig",
        "_family_key": "fam",
        "_csv_line": 1,
        "_row_id": key,
        "_source_file": "src.csv",
        "_source_row": 1,
    }


def _scores(above: dict[str, float]) -> list[float]:
    return [above.get(cid, 0.0) for cid in CAP_IDS]


def _record(key: str = "q-1", *, is_action: bool = False,
            gold: str | None = None, above: dict[str, float] | None = None,
            lang: str = "en", query: str = "hello world") -> dict:
    return B.make_record(
        _source_row(key, is_action=is_action, gold=gold, lang=lang,
                    query=query),
        THRESHOLD, CARDS, CAP_IDS, _scores(above or {}), "srcsha")


def _expected_order(rec: dict) -> tuple[list[str], list[str]]:
    """Independent re-derivation of the frozen candidate-order rule."""
    retrieved = sorted(rec["retrieved_capabilities"],
                       key=lambda c: (-rec["retrieval_scores"][c], c))
    base = retrieved[:B.K_MAX] + [B.REJECT_LABEL]
    base_cards = [CARDS[c] for c in base[:-1]] + [B.REJECT_CARD]
    perm = list(range(len(base)))
    random.Random(f"{B.SHUFFLE_SEED_PREFIX}:{rec['id']}").shuffle(perm)
    return [base[i] for i in perm], [base_cards[i] for i in perm]


def test_to_v2_row_keeps_only_the_four_frozen_fields() -> None:
    """The training row is exactly {message_id, language, input, label}."""
    rec = _record(is_action=True, gold="cap-01",
                  above={"cap-00": 0.95, "cap-01": 0.90, "cap-02": 0.85},
                  lang="zh", query="帮我总结")
    row = B.to_v2_row(rec)
    assert tuple(row) == B.V2_ROW_KEYS
    assert set(row) == {"message_id", "language", "input", "label"}
    assert uuid.UUID(row["message_id"]).version == 4
    assert row["language"] == "zh"
    assert row["input"]["query"] == "帮我总结"
    assert row["label"] == rec["gold_index"]
    parsed = [B.parse_option(o) for o in row["input"]["options"]]
    assert [cid for cid, _ in parsed] == rec["order"]
    assert [card for _, card in parsed] == rec["options"]


@pytest.mark.parametrize("k,gold", [(3, "cap-01"), (1, "cap-05"),
                                    (0, None)])
def test_candidate_order_is_deterministic(k: int, gold: str | None) -> None:
    """Same source row + scores -> same order, and the order follows the
    independently re-derived ``reject-shuffle-v1:<id>`` permutation."""
    above = {cid: 0.95 - 0.05 * i for i, cid in enumerate(CAP_IDS[:k])}
    rec = _record(is_action=k > 0, gold=gold, above=above)
    again = _record(is_action=k > 0, gold=gold, above=above)
    assert rec["order"] == again["order"]
    assert rec["options"] == again["options"]
    assert len(rec["order"]) == k + 1
    assert rec["order"].count(B.REJECT_LABEL) == 1
    expected_order, expected_options = _expected_order(rec)
    assert rec["order"] == expected_order
    assert rec["options"] == expected_options
    assert rec["order"][rec["gold_index"]] == rec["label"]


def test_validate_in_build_accepts_four_field_records() -> None:
    """Every check passes for K=0..3, including a reject-targeted row whose
    gold capability WAS retrieved. No stale construction key is touched."""
    records = {THRESHOLD: [
        # K=3, gold in the top 3 -> capability target.
        _record("q-k3", is_action=True, gold="cap-01",
                above={"cap-00": 0.95, "cap-01": 0.90, "cap-02": 0.85}),
        # K=1 -> capability target.
        _record("q-k1", is_action=True, gold="cap-05",
                above={"cap-05": 0.90}),
        # K=0 -> REJECT, direct reject.
        _record("q-k0"),
        # Gold retrieved (< threshold) while others are above -> REJECT label
        # but a non-null gold_capability_id.
        _record("q-not-retrieved", is_action=True, gold="cap-11",
                above={"cap-00": 0.95, "cap-01": 0.90, "cap-02": 0.85}),
    ]}
    checks = B.validate_in_build(records, CAP_IDS, CARDS)
    failed = {name: c["detail"] for name, c in checks.items() if not c["ok"]}
    assert not failed, failed
    assert checks["v2_training_row_projection"]["ok"] is True


@pytest.mark.parametrize("mutation", ["gold_index", "reject_index",
                                      "option_count", "candidate_count",
                                      "order_target"])
def test_validate_in_build_reports_projection_drift(mutation: str) -> None:
    """An internally inconsistent record is flagged in the checks — never a
    raise (the failure mode that aborted a completed rebuild)."""
    good = _record("q-drift", is_action=True, gold="cap-01",
                   above={"cap-00": 0.95, "cap-01": 0.90, "cap-02": 0.85})
    bad = copy.deepcopy(good)
    if mutation == "gold_index":
        bad["gold_index"] = (bad["gold_index"] + 1) % len(bad["order"])
    elif mutation == "reject_index":
        bad["reject_index"] = (bad["reject_index"] + 1) % len(bad["order"])
    elif mutation == "option_count":
        bad["option_count"] += 1
    elif mutation == "candidate_count":
        bad["candidate_count"] += 1
    else:
        idx = bad["gold_index"]
        bad["order"][idx] = ("cap-11" if bad["order"][idx] != "cap-11"
                             else "cap-12")
    checks = B.validate_in_build({THRESHOLD: [good, bad]}, CAP_IDS, CARDS)
    assert checks["v2_training_row_projection"]["ok"] is False
