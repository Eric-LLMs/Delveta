"""LayaChoice-v1 four-field dataset contract.

The v1 bundle was migrated from ten stored columns to the four-field contract
already used by v2 (``message_id``, ``language``, ``input.query``,
``input.options``, ``label``). These tests pin the contract and prove the
migration is lossless: the loader must reproduce the pre-migration runtime rows
byte for byte, so the renderer, targets and evaluation see exactly the same
inputs as before.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

V1 = Path(__file__).resolve().parents[1] / "fine_tuning" / "laya_finetune" / "V1"
DATA = V1 / "data"
SPLITS = ("train", "val", "test")
EXPECTED_ROWS = {"train": 856, "val": 150, "test": 900}
LEGACY = DATA / "layachoice_v1_train_b_noprov.jsonl.gz"

_spec = importlib.util.spec_from_file_location("layachoice_v1_dataset",
                                               V1 / "layachoice_v1_dataset.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def _raw(split: str) -> list[dict]:
    path = DATA / f"{split}.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _legacy(split: str) -> list[dict]:
    with gzip.open(DATA / f"layachoice_v1_{split}_b_noprov.jsonl.gz", "rt",
                   encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


@pytest.mark.parametrize("split", SPLITS)
def test_rows_are_exactly_four_fields(split):
    assert D.ROW_KEYS == {"message_id", "language", "input", "label"}
    for row in _raw(split):
        assert set(row) == D.ROW_KEYS
        assert set(row["input"]) == D.INPUT_KEYS
        assert isinstance(row["message_id"], str) and row["message_id"]
        assert row["language"] in ("zh", "en")
        assert isinstance(row["input"]["query"], str) and row["input"]["query"].strip()
        assert isinstance(row["label"], int) and not isinstance(row["label"], bool)


@pytest.mark.parametrize("split", SPLITS)
def test_option_count_is_frozen_three(split):
    assert all(len(row["input"]["options"]) == 3 for row in _raw(split))
    assert D.OPTION_SLOTS == 3


@pytest.mark.parametrize("split", SPLITS)
def test_label_indexes_the_gold_option(split):
    for row in _raw(split):
        cids = [o.split(": ", 1)[0] for o in row["input"]["options"]]
        assert len(set(cids)) == 3
        assert 0 <= row["label"] < len(row["input"]["options"])
        # The label names one of the stored options; the option string is canonical.
        assert row["input"]["options"][row["label"]].startswith(cids[row["label"]] + ": ")


@pytest.mark.parametrize("split", SPLITS)
def test_split_counts_and_label_distribution_are_preserved(split):
    raw = _raw(split)
    manifest = D.read_manifest(DATA)
    assert len(raw) == EXPECTED_ROWS[split] == manifest["rows"][split]
    assert {str(i): sum(1 for r in raw if r["label"] == i) for i in range(3)} \
        == manifest["gold_slot_distribution"][split]
    assert {l: sum(1 for r in raw if r["language"] == l) for l in ("zh", "en")} \
        == manifest["language"][split]
    assert len({r["message_id"] for r in raw}) == len(raw)


def test_calibration_membership_is_preserved():
    manifest = D.read_manifest(DATA)
    train = _raw("train")
    assert [train[i]["message_id"] for i in manifest["calibration"]["indices"]] \
        == manifest["calibration"]["ids"]
    assert len(manifest["calibration"]["indices"]) == 85


@pytest.mark.skipif(not LEGACY.exists(), reason="pre-migration bundle not present")
@pytest.mark.parametrize("split", SPLITS)
def test_loader_reproduces_the_pre_migration_runtime_rows(split):
    """`order`, `options`, `gold`, `d1`/`d2`, `lang`, `state` and `id` are unchanged.

    Everything downstream reads only these keys, so identical rows mean the same
    renderer input, the same one-hot target and the same evaluation grouping.
    """
    old = _legacy(split)
    new = D.load_bundle(split, DATA)
    assert len(old) == len(new)
    keys = ("id", "split", "lang", "state", "order", "options", "gold", "d1", "d2")
    for a, b in zip(old, new):
        assert {k: b[k] for k in keys} == {k: a[k] for k in keys}


@pytest.mark.skipif(not LEGACY.exists(), reason="pre-migration bundle not present")
@pytest.mark.parametrize("split", SPLITS)
def test_distractors_are_recoverable_from_the_label(split):
    for raw, adapted in zip(_raw(split), D.load_bundle(split, DATA)):
        gold = adapted["order"][adapted["gold"]]
        assert D.DISTRACTORS[gold] == (adapted["order"][adapted["d1"]],
                                       adapted["order"][adapted["d2"]])
        # The stored label is exactly the old gold slot; the old gold capability
        # is the label's option.
        assert adapted["gold"] == raw["label"]
        assert raw["input"]["options"][raw["label"]] == \
            f"{gold}: {adapted['options'][adapted['gold']]}"


def test_fifth_field_is_rejected():
    row = dict(_raw("train")[0])
    row["split"] = "train"
    with pytest.raises(RuntimeError, match="schema drift"):
        D.adapt_row(row, "train")


def test_label_out_of_range_is_rejected():
    row = json.loads(json.dumps(_raw("train")[0]))
    row["label"] = 3
    with pytest.raises(RuntimeError, match="label"):
        D.adapt_row(row, "train")
