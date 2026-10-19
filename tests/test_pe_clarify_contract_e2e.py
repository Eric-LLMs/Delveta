"""End-to-end: MODEL-vs-EXECUTION slot separation across the PE contract.

These cases pin the behaviour the separation exists to guarantee: the MODEL face
(`pe_contract.model_schema` / online `extractor._function_schema`, both
``required: []``) lets the model OMIT a required slot it cannot honestly extract,
while the EXECUTION face (registry ``parameters``, ``required=True``) stays the
final gate — the Binder lands MISSING and the Agent clarifies. The model never
FABRICATES a value, and the non-model counts (top_k / limit / subreddit) never
enter the model target.

Covered (the five required cases):
  1. cap-add-term, neither ``term`` nor ``domain`` stated -> empty model args;
     Binder MISSING; BOTH slots named as unfilled.
  2. cap-add-term, ``term`` stated / ``domain`` missing -> model omits ``domain``;
     Binder MISSING; only ``domain`` named.
  3. cap-add-term, ``term`` + ``domain`` stated -> Binder COMPLETE (executes).
  4. cap-translate, no target language -> ``target_language`` NOT fabricated
     (it is a model slot but NOT required), execution proceeds.
  5. a search request with no query -> ``query`` NOT fabricated; top_k / limit
     are NOT model slots and never reach the model target.

Read-only: no model endpoint, no DB, no production writes.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "fine_tuning" / "qwen_parameter_extraction_finetune"
for _p in (str(_SCRIPTS), str(_REPO), str(_REPO / "packages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pe_contract as C  # noqa: E402
import prepare_dataset as P  # noqa: E402
from core.application.chat.intent_funnel.argument_acquisition.extractor import (  # noqa: E402
    _function_schema,
)
from core.application.chat.intent_funnel.binder.validator import (  # noqa: E402
    validate_against_schema,
)
from core.application.chat.intent_funnel.contract import (  # noqa: E402
    BIND_COMPLETE,
    BIND_MISSING,
)


def _row(cap, query, *, missing=None, outcome=None, language="en", rid="t"):
    return {"id": rid, "capability_id": cap, "language": language, "query": query,
            "expected_missing_required_slots": list(missing or []),
            "expected_outcome": outcome}


def _rendered_arguments(sample: dict) -> dict:
    """The model target as it is rendered into the assistant tool call."""
    calls = sample["messages"][-1]["tool_calls"]
    assert len(calls) == 1
    return json.loads(calls[0]["function"]["arguments"])


def _exec_schema(cap: str) -> dict:
    """The EXECUTION face: the registry ``parameters`` map (strict required)."""
    return C.SCHEMA[cap]["params"]


# ── case 1: neither term nor domain stated ───────────────────────────────────────


def test_add_term_neither_slot__empty_model_args_binder_missing_both():
    row = _row("cap-add-term", "把刚才那个词加到词库里",
               missing=["term", "domain"], outcome="clarify")
    assert P.is_clarify(row)
    fatal, _ = P.audit_row(row, {}, include_clarify=True)
    assert fatal == []

    sample = P.build_sample(row, {})
    assert sample["tools"][0]["function"]["parameters"]["required"] == []
    assert _rendered_arguments(sample) == {}          # honest empty extraction
    assert sample["meta"]["missing_required"] == ["term", "domain"]

    bound = validate_against_schema(_exec_schema("cap-add-term"), _rendered_arguments(sample))
    assert bound.state == BIND_MISSING
    unfilled = set(C.required_model_slots("cap-add-term")) - set(_rendered_arguments(sample))
    assert unfilled == {"term", "domain"}             # Binder clarifies BOTH


# ── case 2: term present, domain missing ─────────────────────────────────────────


def test_add_term_term_stated_domain_missing__binder_asks_only_domain():
    row = _row("cap-add-term", "把 diffusion 这个词加到我的词库里",
               missing=["domain"], outcome="clarify")
    assert P.is_clarify(row)
    target = {"term": "diffusion"}
    fatal, _ = P.audit_row(row, target, include_clarify=True)
    assert fatal == []

    sample = P.build_sample(row, target)
    assert _rendered_arguments(sample) == {"term": "diffusion"}
    assert "domain" not in _rendered_arguments(sample)   # never fabricated

    bound = validate_against_schema(_exec_schema("cap-add-term"), _rendered_arguments(sample))
    assert bound.state == BIND_MISSING
    unfilled = set(C.required_model_slots("cap-add-term")) - set(_rendered_arguments(sample))
    assert unfilled == {"domain"}                        # Binder asks ONLY the domain


# ── case 3: term + domain complete ───────────────────────────────────────────────


def test_add_term_complete__binder_complete_executes():
    target = {"term": "diffusion", "domain": "工程词汇库"}
    row = _row("cap-add-term", "把 diffusion 加到工程词汇库")
    assert not P.is_clarify(row)
    fatal, _ = P.audit_row(row, target, include_clarify=True)
    assert fatal == []

    sample = P.build_sample(row, target)
    args = _rendered_arguments(sample)
    bound = validate_against_schema(_exec_schema("cap-add-term"), args)
    assert bound.state == BIND_COMPLETE
    assert bound.args == {"term": "diffusion", "domain": "工程词汇库"}


# ── case 4: translate without a target language ──────────────────────────────────


def test_translate_without_target_language__not_fabricated_but_completes():
    row = _row("cap-translate", "把这句话翻译一下")
    target = {"text": "把这句话翻译一下"}
    fatal, _ = P.audit_row(row, target, include_clarify=True)
    assert fatal == []

    sample = P.build_sample(row, target)
    args = _rendered_arguments(sample)
    assert args == {"text": "把这句话翻译一下"}
    assert "target_language" not in args                 # not fabricated

    # target_language IS a model slot, but it is NOT required -> execution proceeds.
    assert "target_language" in C.model_slots("cap-translate")
    assert "target_language" not in C.required_model_slots("cap-translate")
    bound = validate_against_schema(_exec_schema("cap-translate"), args)
    assert bound.state == BIND_COMPLETE

    # and an empty text still fails closed (the required slot is owned by execution)
    empty = validate_against_schema(_exec_schema("cap-translate"), {})
    assert empty.state == BIND_MISSING


# ── case 5: a search request carries no query ────────────────────────────────────


def test_search_without_query__not_fabricated_and_counts_are_not_model_slots():
    for cap in ("cap-web-search", "cap-rag-search", "cap-social-search"):
        assert "top_k" not in C.model_slots(cap)
        assert "limit" not in C.model_slots(cap)
        assert "subreddit" not in C.model_slots(cap)
        schema = C.model_schema(cap)
        assert schema["required"] == []
        for count_slot in ("top_k", "limit", "subreddit"):
            assert count_slot not in schema["properties"]

    row = _row("cap-web-search", "帮我搜一下", missing=["query"], outcome="clarify")
    assert P.is_clarify(row)
    fatal, _ = P.audit_row(row, {}, include_clarify=True)
    assert fatal == []

    sample = P.build_sample(row, {})
    assert _rendered_arguments(sample) == {}             # query not fabricated
    props = set(sample["tools"][0]["function"]["parameters"]["properties"])
    assert props <= set(C.model_slots("cap-web-search"))
    assert "top_k" not in props and "limit" not in props

    bound = validate_against_schema(_exec_schema("cap-web-search"), {})
    assert bound.state == BIND_MISSING


# ── the online function face is the SAME contract (one source of truth) ──────────


@pytest.mark.parametrize("cap", ["cap-web-search", "cap-add-term", "cap-translate",
                                 "cap-rag-search", "cap-social-search"])
def test_online_function_face_matches_the_shared_contract(cap):
    entry = types.SimpleNamespace(
        capability_id=cap, tool_binding=C.tool_name(cap),
        description=C.function_description(cap), parameters=C.SCHEMA[cap]["params"])
    face = _function_schema(entry, C.model_slots(cap))
    assert face == C.model_schema(cap)                   # identical face
    assert face["required"] == []
    assert C.required_model_slots(cap)                   # execution stays strict


# ── Binder regression: a PARTIAL prediction is still gated by execution ──────────


def test_partial_prediction_missing_one_required_slot_is_binder_missing():
    """The model may legally emit a partial target (one required slot filled, one
    omitted). Execution must NOT proceed: the Binder lands BIND_MISSING for the
    omitted required slot. This is the behaviour the model/execution separation
    exists to preserve — the model never fabricates the missing value, and the
    Binder — not the model — is the final gate."""
    partial = {"term": "diffusion"}                      # domain omitted, not fabricated
    bound = validate_against_schema(_exec_schema("cap-add-term"), partial)
    assert bound.state == BIND_MISSING
    assert set(C.required_model_slots("cap-add-term")) - set(partial) == {"domain"}


# ── the shipped bundle renders the no-slot negatives as empty tool calls ─────────


_V5_TRAIN = (_SCRIPTS / "data" / "train.jsonl")


@pytest.mark.skipif(not _V5_TRAIN.is_file(), reason="shipped PE bundle not present")
def test_shipped_bundle_renders_no_slot_negatives_as_empty_tool_calls():
    rows = [json.loads(l) for l in _V5_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    neg = [r for r in rows if str(r["id"]).startswith("v4-neg-")]
    assert neg, "expected the v4 no-slot negatives inside the shipped bundle"
    for r in neg:
        call = r["messages"][-1]["tool_calls"][0]["function"]
        assert call["name"] == r["capability_id"]
        assert json.loads(call["arguments"]) == {}       # honest empty tool call
        assert r["tools"][0]["function"]["parameters"]["required"] == []
