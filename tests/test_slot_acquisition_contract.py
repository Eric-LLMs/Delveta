"""Unified slot acquisition — the PURE contract units (no cascade, no model).

Scheme A ("unified Handler orchestration") splits a handler's per-slot
disposition out of ``acquire()``:

* ``search_args.states_result_count`` — the shared DET rule for the optional
  count slot (model-authorized ONLY when the sentence states a count);
* ``SlotPlan`` / each search handler's ``slot_plan()`` — the EXPLICIT four-state
  disposition (ACQUIRED / PENDING_MODEL / DEFAULTED / MISSING) the orchestrator
  reads INSTEAD of inferring "needs a model" from ``slot not in draft``;
* ``slot_refine.accept_model_values`` — the fold of authorized, valid model
  values into the deterministic draft (never an unauthorized overwrite);
* ``prompts.prompt_for`` — the per-capability system prompt (default
  byte-compatible for a capability that registers none);
* the extractor's ``_payload`` — the ``prompt`` override is the system message
  and an absent one is byte-identical to the frozen default.

The MODEL itself is not exercised here; the real invocation lives in
``test_orchestrator_slot_extraction`` (and its real-Qwen integration test).
"""
from __future__ import annotations

import types

import pytest

from core.application.chat.intent_funnel.argument_acquisition import context_bundle
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    SYSTEM_EXTRACT,
    _payload,
)
from core.application.chat.intent_funnel.argument_acquisition.prompts import (
    SEARCH_QUERY_PROMPT,
    prompt_for,
)
from core.application.chat.intent_funnel.argument_acquisition.slot_refine import (
    accept_model_values,
)
from core.application.chat.intent_funnel.cap_handler import (
    AddTermHandler,
    RagSearchHandler,
    SlotPlan,
    SlotPlanningHandler,
    SocialSearchHandler,
    TranslateHandler,
    WebSearchHandler,
)
from core.application.chat.intent_funnel.cap_handler.search_args import (
    states_result_count,
)
from core.application.chat.intent_funnel.registry.entry import CapabilityEntry


# ── search_args.states_result_count ──────────────────────────────────────────────


@pytest.mark.parametrize("message", [
    "帮我搜一下注意力机制，前 5 条",
    "search for python asyncio best practices, top 3",
    "给我 10 results",
    "取 3 个",
    "show me 7 items",
    "列出 4 篇",
])
def test_states_result_count_true(message):
    assert states_result_count(message) is True


@pytest.mark.parametrize("message", [
    "帮我搜一下注意力机制",              # no count at all
    "python 3 的新特性",                # a version, not a result count
    "2026 年的最新 AI 新闻",            # a year
    "RTX 4090 的性能",                 # a model number, no count noun
    "search for transformer papers",    # no count
])
def test_states_result_count_false(message):
    assert states_result_count(message) is False


# ── SlotPlan + the three search handlers' slot_plan ──────────────────────────────


def test_slot_plan_defaults_are_empty_and_frozen():
    plan = SlotPlan()
    assert plan.model_slots == () and plan.default_slots == ()
    with pytest.raises(Exception):
        plan.model_slots = ("x",)  # frozen dataclass


async def _draft(handler, message):
    return await handler.acquire(query=message, facts=types.SimpleNamespace())


async def test_web_slot_plan_without_count_defaults_top_k():
    h = WebSearchHandler()
    draft = await _draft(h, "帮我搜一下注意力机制")
    plan = h.slot_plan(query="帮我搜一下注意力机制", facts=None, draft=draft)
    assert plan.model_slots == ("query",)
    assert plan.default_slots == ("top_k",)


async def test_web_slot_plan_with_count_authorizes_top_k():
    h = WebSearchHandler()
    msg = "search for transformers, top 3"
    draft = await _draft(h, msg)
    plan = h.slot_plan(query=msg, facts=None, draft=draft)
    assert plan.model_slots == ("query", "top_k")
    assert plan.default_slots == ()


async def test_rag_slot_plan_without_count_authorizes_query_and_domain():
    h = RagSearchHandler()
    draft = await _draft(h, "帮我搜一下注意力机制")
    plan = h.slot_plan(query="帮我搜一下注意力机制", facts=None, draft=draft)
    assert plan.model_slots == ("query", "domain")
    assert plan.default_slots == ("top_k",)


async def test_rag_slot_plan_with_count_authorizes_top_k():
    h = RagSearchHandler()
    msg = "查一下注意力机制，前 5 条"
    draft = await _draft(h, msg)
    plan = h.slot_plan(query=msg, facts=None, draft=draft)
    assert plan.model_slots == ("query", "top_k", "domain")
    assert plan.default_slots == ()


async def test_social_slot_plan_authorizes_platform_when_none_named():
    h = SocialSearchHandler()
    msg = "搜索社区里对注意力机制的讨论"
    draft = await _draft(h, msg)
    assert draft["platform"] == "auto"
    plan = h.slot_plan(query=msg, facts=None, draft=draft)
    assert plan.model_slots == ("query", "platform")   # the genuinely-needed fallback
    assert plan.default_slots == ("limit",)


async def test_social_slot_plan_skips_platform_and_model_cannot_narrow_a_set():
    h = SocialSearchHandler()
    msg = "reddit 和知乎上都搜一下"
    draft = await _draft(h, msg)
    assert draft["platform"] == "auto"                  # several named -> honest auto
    plan = h.slot_plan(query=msg, facts=None, draft=draft)
    assert "platform" not in plan.model_slots           # never narrowed by a model


async def test_social_slot_plan_skips_platform_when_det_named_one():
    h = SocialSearchHandler()
    msg = "在 reddit 上搜一下 transformer"
    draft = await _draft(h, msg)
    assert draft["platform"] == "reddit"
    plan = h.slot_plan(query=msg, facts=None, draft=draft)
    assert "platform" not in plan.model_slots           # DET value is authoritative


async def test_social_slot_plan_with_count_authorizes_limit():
    h = SocialSearchHandler()
    msg = "搜索社区讨论，取 3 条"
    draft = await _draft(h, msg)
    plan = h.slot_plan(query=msg, facts=None, draft=draft)
    assert "limit" in plan.model_slots
    assert plan.default_slots == ()


def test_search_handlers_opt_in_other_handlers_do_not():
    for h in (WebSearchHandler(), RagSearchHandler(), SocialSearchHandler()):
        assert isinstance(h, SlotPlanningHandler)
    # non-search handlers carry no slot_plan -> the orchestrator calls no model.
    for h in (AddTermHandler(), TranslateHandler()):
        assert not isinstance(h, SlotPlanningHandler)


# ── slot_refine.accept_model_values ──────────────────────────────────────────────


_PARAMS = {
    "query": {"type": "string", "required": True, "max_len": 200},
    "top_k": {"type": "integer", "required": False},
    "platform": {"type": "string", "required": False, "enum": ["reddit", "x", "zhihu", "auto"]},
}


def test_accept_folds_authorized_valid_values():
    out = accept_model_values({"query": "raw sentence"}, {"query": "clean topic", "top_k": 3},
                              model_slots=("query", "top_k"), parameters=_PARAMS)
    assert out == {"query": "clean topic", "top_k": 3}   # integer coerced


def test_accept_never_writes_an_unauthorized_slot():
    out = accept_model_values({"query": "raw", "platform": "reddit"},
                              {"query": "clean", "platform": "zhihu"},
                              model_slots=("query",), parameters=_PARAMS)
    assert out == {"query": "clean", "platform": "reddit"}  # DET value kept


def test_accept_drops_unknown_slot():
    out = accept_model_values({"query": "raw"}, {"query": "clean", "bogus": "x"},
                              model_slots=("query", "bogus"), parameters=_PARAMS)
    assert out == {"query": "clean"}                     # never invent a slot


@pytest.mark.parametrize("raw", [None, "", "   ", "\n\t"])
def test_accept_rejects_empty_value_keeps_draft(raw):
    out = accept_model_values({"query": "raw sentence"}, {"query": raw},
                              model_slots=("query",), parameters=_PARAMS)
    assert out == {"query": "raw sentence"}


def test_accept_rejects_type_invalid_integer():
    out = accept_model_values({"query": "raw"}, {"top_k": "three"},
                              model_slots=("query", "top_k"), parameters=_PARAMS)
    assert "top_k" not in out


def test_accept_rejects_over_length_string():
    out = accept_model_values({"query": "raw"}, {"query": "x" * 201},
                              model_slots=("query",), parameters=_PARAMS)
    assert out == {"query": "raw"}


def test_accept_rejects_out_of_enum():
    out = accept_model_values({"platform": "auto"}, {"platform": "facebook"},
                              model_slots=("platform",), parameters=_PARAMS)
    assert out == {"platform": "auto"}


def test_accept_does_not_mutate_the_input_draft():
    draft = {"query": "raw"}
    accept_model_values(draft, {"query": "clean"}, model_slots=("query",),
                        parameters=_PARAMS)
    assert draft == {"query": "raw"}


# ── prompts.prompt_for ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("cid", ["cap-web-search", "cap-rag-search", "cap-social-search"])
def test_prompt_for_search_caps_is_the_search_prompt(cid):
    assert prompt_for(cid) == SEARCH_QUERY_PROMPT


@pytest.mark.parametrize("cid", ["cap-add-term", "cap-translate", "", None])
def test_prompt_for_unregistered_cap_is_the_frozen_default(cid):
    assert prompt_for(cid) == SYSTEM_EXTRACT


# ── extractor._payload: the prompt override ──────────────────────────────────────


def _entry() -> CapabilityEntry:
    return CapabilityEntry(
        capability_id="cap-web-search", tool_binding="web_search",
        parameters={"query": {"type": "string", "required": True, "description": "q"}},
    )


def test_payload_uses_provided_prompt_as_the_system_message():
    bundle = context_bundle.build("帮我搜一下注意力机制")
    payload = _payload(_entry(), ("query",), bundle, "CUSTOM SYSTEM")
    assert payload["messages"][0] == {"role": "system", "content": "CUSTOM SYSTEM"}


def test_payload_without_prompt_is_byte_compatible_with_the_default():
    bundle = context_bundle.build("帮我搜一下注意力机制")
    payload = _payload(_entry(), ("query",), bundle)
    assert payload["messages"][0] == {"role": "system", "content": SYSTEM_EXTRACT}
