"""Regression guard for the mandatory cross-split exclusion set in the
LayaChoice-v1 ingest gate (logs/_laya_ds_ingest.py).

The final-test split is checked for leakage against the FROZEN validation v3 raw
file. If that file were silently skipped, the leakage gate would degrade to an
empty exclusion set and let near-duplicates through. These tests pin the hard
assertion: missing / malformed -> fail, normal -> 150 rows.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INGEST = ROOT / "logs" / "_laya_ds_ingest.py"
VAL_V3_RAW = ROOT / "data" / "LayaChoice_v1_validation_raw_v3.jsonl"

# The ingest script and the frozen validation-v3 raw file both live in local-only
# scratch trees (``logs/`` and ``data/``); skip the whole module where absent.
pytestmark = pytest.mark.skipif(
    not (INGEST.exists() and VAL_V3_RAW.exists()),
    reason="LayaChoice ingest script / frozen validation_v3 not present (scratch artifacts)",
)


def _load_ingest():
    spec = importlib.util.spec_from_file_location("_laya_ds_ingest_under_test", INGEST)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ingest():
    return _load_ingest()


def test_final_test_exclusion_is_the_frozen_validation_v3(ingest):
    assert Path(ingest.VAL_V3_RAW) == VAL_V3_RAW
    assert VAL_V3_RAW.exists()


def test_missing_validation_v3_fails(ingest, tmp_path):
    missing = tmp_path / "LayaChoice_v1_validation_raw_v3.jsonl"
    with pytest.raises(ingest.ExclusionSetError) as exc:
        ingest.read_queries(str(missing), required=True, label="validation_v3")
    msg = str(exc.value)
    assert "validation_v3" in msg
    assert str(missing) in msg
    assert "missing" in msg


def test_malformed_validation_v3_fails(ingest, tmp_path):
    bad = tmp_path / "LayaChoice_v1_validation_raw_v3.jsonl"
    bad.write_text('{"query": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ingest.ExclusionSetError) as exc:
        ingest.read_queries(str(bad), required=True, label="validation_v3")
    msg = str(exc.value)
    assert "validation_v3" in msg
    assert str(bad) in msg
    assert "malformed" in msg


def test_valid_validation_v3_yields_150_rows(ingest):
    rows = ingest.read_queries(str(VAL_V3_RAW), required=True, label="validation_v3")
    assert len(rows) == 150
