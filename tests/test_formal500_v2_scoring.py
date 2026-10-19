"""Tests for the Formal-500 v2 ownership map, generation rules and scorer.

Everything imported here is TRACKED (``fine_tuning/qwen_parameter_extraction_finetune``);
no untracked ``logs/`` module is imported, so CI collection stays clean.

The v1 Formal-500 evaluator (``logs/_e2e_virtual_metrics.py`` + the frozen
``logs/_argbench/ownership.py``) is deliberately NOT touched by these tests — the
v2 scorer is an independent, versioned口径.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "fine_tuning" / "qwen_parameter_extraction_finetune"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import formal500_ownership as OWN  # noqa: E402
import formal500_rules as R  # noqa: E402
import formal500_v2_score as S  # noqa: E402
import pe_contract as C  # noqa: E402


# ── ownership map ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cap", sorted(C.SCHEMA))
def test_contract_caps_pe_slots_equal_pe_contract(cap):
    """The six contract capabilities take their PE slots straight from pe_contract."""
    assert OWN.pe_model_slots(cap) == tuple(C.model_slots(cap))


def test_function_defaults_are_not_pe_slots():
    assert OWN.classify("cap-web-search", "top_k") == OWN.TOOL_DEFAULT
    assert OWN.classify("cap-rag-search", "top_k") == OWN.TOOL_DEFAULT
    assert OWN.classify("cap-social-search", "limit") == OWN.TOOL_DEFAULT
    assert OWN.classify("cap-social-search", "subreddit") == OWN.TOOL_DEFAULT
    assert "top_k" not in OWN.pe_model_slots("cap-web-search")
    assert "subreddit" not in OWN.pe_model_slots("cap-social-search")


def test_handler_owned_caps_have_no_pe_slots():
    """The handler-only capabilities never call the model (no slot_plan)."""
    for cap in ("cap-vision", "cap-read-document", "cap-read-file",
                "cap-pdf-extract-text", "cap-pdf-table-to-text"):
        assert OWN.pe_model_slots(cap) == ()
    assert OWN.classify("cap-vision", "question") == OWN.HANDLER
    assert OWN.classify("cap-vision", "asset_id") == OWN.SYSTEM_BINDER
    assert OWN.classify("cap-read-document", "pages") == OWN.CONTEXT
    assert OWN.classify("cap-read-file", "path") == OWN.HANDLER


def test_generic_chain_caps():
    """Capabilities on the generic chain keep the declaration.py default MODEL."""
    assert set(OWN.pe_model_slots("cap-edit-file")) == {"path", "new_text", "old_text"}
    assert OWN.pe_model_slots("cap-bash") == ("command",)
    assert OWN.classify("cap-bash", "timeout") == OWN.TOOL_DEFAULT
    assert OWN.pe_model_slots("cap-artifact") == ("action",)
    assert OWN.classify("cap-artifact", "project_id") == OWN.UNAVAILABLE


def test_undeclared_caps():
    for cap in ("cap-slides", "cap-summary", "cap-mindmap"):
        assert OWN.pe_model_slots(cap) == ()
        assert OWN.slot_buckets(cap) == {}
    assert OWN.classify("cap-slides", "count") == OWN.UNDECLARED


# ── generation rules ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,bare", [
    ("工程词汇库", "工程"),
    ("machine learning vocabulary", "machine learning"),
    ("信息论词库", "信息论"),
    ("架构生词本", "架构"),
    ("流处理 词汇本", "流处理"),
    ("深度学习术语库", "深度学习"),
    ("LLM terms", "LLM"),
    ("database glossary", "database"),
])
def test_strip_container(raw, bare):
    assert R.strip_container(raw) == bare


def test_strip_container_noop_when_already_bare():
    assert R.strip_container("信息论") is None
    assert R.strip_container("工程") is None
    assert R.strip_container("词库") is None  # never empty a value out


def test_query_verbatim():
    assert R.query_is_verbatim("搜索 NVIDIA H200 最新价格，给我 5 条", "NVIDIA H200 最新价格")
    assert not R.query_is_verbatim(
        "look up discussions on X about the new MacBook Pro thermals",
        "discussions about the new MacBook Pro thermals")
    assert R.query_is_verbatim('把 "hello world" 翻译成中文', "hello world")


# ── scorer ───────────────────────────────────────────────────────────────────
def _row(cid, cap, gold, pe_slots, expected, forbidden=(), ownership=None):
    return {
        "id": cid, "capability_id": cap,
        "gold_arguments": gold, "pe_model_slots": pe_slots,
        "expected_tool_arguments": expected, "forbidden_arguments": list(forbidden),
        "arg_ownership": ownership or {},
    }


def _pred(cid, cap, args, binder="COMPLETE", called=True, fallback="-", deepest="certified"):
    return {"id": cid, "pred_capability_id": cap, "pred_arguments": args,
            "binder_state": binder, "virtual_executor_called": called,
            "fallback_reason": fallback, "deepest_stage": deepest}


def test_scorer_numerators_and_denominators():
    dataset = [
        _row("a", "cap-web-search", {"query": "x"}, ["query"], {"query": "x"},
             forbidden=["top_k"]),
        _row("b", "cap-read-document", {}, [], {"pages": "2"},
             forbidden=["asset_id"], ownership={"pages": OWN.CONTEXT}),
    ]
    preds = [
        _pred("a", "cap-web-search", {"query": "x"}),           # full hit
        # correct context slot, wrong capability, and a forbidden asset_id leak
        _pred("b", "cap-web-search", {"pages": "2", "asset_id": "bad"}),
    ]
    rep = S.score(dataset, preds)
    m = rep["metrics"]

    assert rep["scorer_version"] == S.SCORER_VERSION
    assert rep["dataset_version"] == S.DATASET_VERSION
    assert rep["missing_predictions"] == 0

    # M1 tool selection: 1 of 2
    assert m["M1_tool_selection"]["numerator"] == 1
    assert m["M1_tool_selection"]["denominator"] == 2

    # M2 PE strict: only case "a" has a PE slot; it matches -> 1/1
    assert m["M2_pe_model_strict"]["numerator"] == 1
    assert m["M2_pe_model_strict"]["denominator"] == 1

    # M3 PE lenient: 1 gold PE slot instance, matched -> 1/1
    assert m["M3_pe_model_lenient"]["numerator"] == 1
    assert m["M3_pe_model_lenient"]["denominator"] == 1

    # M5 virtual exec reached: 2 of 2
    assert m["M5_virtual_exec_reached"]["numerator"] == 2
    assert m["M5_virtual_exec_reached"]["denominator"] == 2

    # M7 complete contract: both cases match on the restricted view — case "b"
    # carries an EXTRA forbidden asset_id, which M7 does not score.
    assert m["M7_tool_contract_match"]["numerator"] == 2
    assert m["M7_tool_contract_match"]["denominator"] == 2

    # M6 flags the forbidden emission on case "b"
    assert m["M6_stage_failures"]["forbidden_emissions"] == 1

    # M8 reports the non-PE side separately (the CONTEXT slot on "b" matched)
    buckets = m["M8_non_pe_slot_sources"]["buckets"]
    assert buckets[OWN.CONTEXT]["matched"] == 1
    assert buckets[OWN.CONTEXT]["expected"] == 1


def test_scorer_counts_missing_prediction_as_miss():
    dataset = [_row("a", "cap-web-search", {"query": "x"}, ["query"], {"query": "x"})]
    rep = S.score(dataset, [])
    assert rep["missing_predictions"] == 1
    assert rep["metrics"]["M1_tool_selection"]["numerator"] == 0
    assert rep["metrics"]["M2_pe_model_strict"]["numerator"] == 0
    assert rep["metrics"]["M2_pe_model_strict"]["denominator"] == 1


def test_scorer_pe_strict_penalizes_extra_pe_slot():
    dataset = [_row("a", "cap-rag-search", {"query": "x"}, ["query", "domain"],
                    {"query": "x"})]
    preds = [_pred("a", "cap-rag-search", {"query": "x", "domain": "extra"})]
    rep = S.score(dataset, preds)
    # strict: key set differs (domain present in pred, absent in gold) -> miss
    assert rep["metrics"]["M2_pe_model_strict"]["numerator"] == 0
    # lenient: the single gold PE slot (query) still matches -> 1/1
    assert rep["metrics"]["M3_pe_model_lenient"]["numerator"] == 1
    assert rep["metrics"]["M3_pe_model_lenient"]["denominator"] == 1
    assert rep["metrics"]["M3_pe_model_lenient"]["extra_pe_slots_emitted"] == 1


def test_scorer_canonical_equality():
    assert S._eq(5, 5.0)
    assert not S._eq("5", 5)          # a string count is NOT the integer count
    assert S._eq(" x ", "x")           # whitespace is not significant


def test_from_execution_trace_adapter():
    trace = {"case_id": "ws-01", "stages": {
        "decided_capability": "cap-web-search", "model_values": {"query": "x"},
        "binder_state": "COMPLETE", "virtual_executor_called": True,
        "fallback_reason": "-", "deepest_stage": "certified"}}
    norm = S.from_execution_trace(trace)
    assert norm["id"] == "ws-01"
    assert norm["pred_capability_id"] == "cap-web-search"
    assert norm["pred_arguments"] == {"query": "x"}
    assert norm["virtual_executor_called"] is True
