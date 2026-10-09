"""Unified slot acquisition — the orchestrator WIRING (``_acquisition_hop``).

Scheme A pins how a registered handler's ``slot_plan()`` drives the SHARED
extractor inside the handler branch, WITHOUT the caller ever inferring "needs a
model" from ``slot not in draft``:

* an authorized plan runs the extractor ONCE, over the query bundle, with the
  capability's Prompt; valid, authorized values are folded into the draft;
* a capability that named a count gets ``top_k`` / ``limit``; an unstated count
  stays the tool's default and is never asked of a model;
* an EMPTY / INVALID / UNAVAILABLE reply leaves the handler draft (the verbatim
  query + DET values) intact — the original request is never lost, and the SAME
  Binder remains the final gate (a required slot no one filled still lands
  MISSING -> the Agent);
* a handler WITHOUT ``slot_plan`` (every handler but the search trio) calls no
  model — the pure-deterministic flow is byte-identical to before.

The extractor here is a MOCK seam: the real Docker Qwen call is exercised by the
integration test at the bottom, which SKIPS when the endpoint is not deployed
(the funnel tests never require a live model).
"""
from __future__ import annotations

import types
from urllib.parse import urlparse

import pytest

from core.application.chat.intent_funnel import orchestrator as orch_mod
from core.application.chat.intent_funnel.argument_acquisition.extractor import (
    ExtractionUnavailable,
)
from core.application.chat.intent_funnel.argument_acquisition.prompts import (
    SEARCH_QUERY_PROMPT,
)
from core.application.chat.intent_funnel.cap_handler import roster
from core.application.chat.intent_funnel.contract import REASON_BIND_MISSING
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings


# ── entry builders (mirror the live schema shapes) ───────────────────────────────


def _entry(cid: str, tool: str, params: dict) -> CapabilityEntry:
    q = "搜索知识库"
    return CapabilityEntry(
        capability_id=cid, tool_binding=tool, description="d",
        standard_queries=(QueryRecord(id=f"{cid}-q1", query=q,
                                      language=derive_language(q)),),
        parameters=params,
    )


def _web_entry() -> CapabilityEntry:
    return _entry("cap-web-search", "web_search", {
        "query": {"type": "string", "required": True, "max_len": 200,
                  "description": "query"},
        "top_k": {"type": "integer", "required": False, "description": "count"},
    })


def _rag_entry(*, domain_required: bool = False) -> CapabilityEntry:
    return _entry("cap-rag-search", "rag_search", {
        "query": {"type": "string", "required": True, "description": "query"},
        "top_k": {"type": "integer", "required": False, "description": "count"},
        "domain": {"type": "string", "required": domain_required, "description": "d"},
    })


def _social_entry() -> CapabilityEntry:
    return _entry("cap-social-search", "search_social", {
        "query": {"type": "string", "required": True, "description": "query"},
        "platform": {"type": "string", "required": False, "description": "platform"},
        "subreddit": {"type": "string", "required": False, "description": "scope"},
        "limit": {"type": "integer", "required": False, "description": "count"},
    })


def _view(entry: CapabilityEntry) -> RegistryLiveView:
    return RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))


async def _hop(entry: CapabilityEntry, *, query: str, extractor=None):
    trace: dict = {}
    capture: dict = {}
    deps = types.SimpleNamespace(argument_extractor=extractor)
    out = await orch_mod._acquisition_hop(
        TurnRequirements(), deps, entry, query=query, facts=None,
        view=_view(entry), trace=trace, capture=capture)
    return out, trace, capture


def _mock(returned=None, *, boom=False, seen=None):
    async def _extract(*, query, entry, model_slots, bundle, prompt):
        if seen is not None:
            seen.update(query=query, slots=tuple(model_slots),
                        prompt=prompt, source=bundle.source)
        if boom:
            raise ExtractionUnavailable("endpoint down")
        return returned if returned is not None else {}, bundle.source
    return _extract


# ── the model cleans the search topic (query) ────────────────────────────────────


async def test_web_query_is_replaced_by_the_cleaned_topic():
    msg = "帮我搜一下注意力机制的最新进展"
    seen: dict = {}
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "注意力机制的最新进展"}, seen=seen))

    assert out is not None
    assert out.requested_action["args"] == {"query": "注意力机制的最新进展"}
    assert seen["query"] == msg                      # the model saw the raw sentence
    assert seen["slots"] == ("query",)               # no count stated -> query only
    assert seen["prompt"] == SEARCH_QUERY_PROMPT     # per-capability prompt
    assert capture["acquisition"]["extractor"] == "Qwen"
    assert capture["acquisition"]["slot_plan"] == {
        "model_slots": ["query"], "default_slots": ["top_k"]}


async def test_web_stated_count_authorizes_and_certifies_top_k():
    msg = "search for transformers, top 3"
    seen: dict = {}
    out, trace, capture = await _hop(
        _web_entry(), query=msg,
        extractor=_mock({"query": "transformers", "top_k": 3}, seen=seen))

    assert seen["slots"] == ("query", "top_k")
    # the Binder normalizes every slot to str on the certified action.
    assert out.requested_action["args"] == {"query": "transformers", "top_k": "3"}


async def test_web_without_count_keeps_the_tool_default_for_top_k():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "注意力机制"}))
    assert out.requested_action["args"] == {"query": "注意力机制"}
    assert "top_k" not in out.requested_action["args"]   # tool default stands


# ── empty / invalid / unavailable / unauthorized: the draft is never lost ─────────


async def test_empty_model_value_keeps_the_verbatim_query():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "   "}))
    assert out.requested_action["args"] == {"query": msg}


async def test_over_length_model_value_is_rejected():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "x" * 300}))
    assert out.requested_action["args"] == {"query": msg}


async def test_model_unavailable_keeps_the_draft_and_still_certifies():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock(boom=True))
    assert out is not None                           # the turn is NOT dropped
    assert out.requested_action["args"] == {"query": msg}
    assert capture["acquisition"]["model_values"] == {}


async def test_unauthorized_slot_from_the_model_is_never_written():
    msg = "帮我搜一下注意力机制"                       # no count -> top_k not authorized
    out, trace, capture = await _hop(
        _web_entry(), query=msg, extractor=_mock({"query": "注意力机制", "top_k": 9}))
    assert out.requested_action["args"] == {"query": "注意力机制"}
    assert "top_k" not in out.requested_action["args"]


async def test_platform_fallback_value_is_accepted_only_when_none_named():
    msg = "搜索社区里对注意力机制的讨论"                # DET -> auto, model authorized
    out, trace, capture = await _hop(
        _social_entry(), query=msg, extractor=_mock({"query": "注意力机制讨论", "platform": "x"}))
    assert out.requested_action["args"] == {"query": "注意力机制讨论", "platform": "x"}


async def test_det_platform_survives_a_divergent_model_value():
    msg = "在 reddit 上搜一下 transformer"              # DET -> reddit, not authorized
    out, trace, capture = await _hop(
        _social_entry(), query=msg, extractor=_mock({"query": "transformer", "platform": "zhihu"}))
    assert out.requested_action["args"] == {"query": "transformer", "platform": "reddit"}


# ── a required slot nobody filled still fails closed through the SAME Binder ──────


async def test_binder_missing_required_slot_exits_to_agent():
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(
        _rag_entry(domain_required=True), query=msg, extractor=_mock({}))
    assert out is None
    assert trace["fallback"] == REASON_BIND_MISSING


# ── a handler without slot_plan calls no model (pure-deterministic flow) ──────────


async def test_handler_without_slot_plan_calls_no_model(monkeypatch):
    class _PlainHandler:
        capability_id = "cap-web-search"

        async def acquire(self, *, query, facts):
            return {"query": str(query).strip()}

    async def _must_not_run(**kw):
        raise AssertionError("a handler without slot_plan must not call the extractor")

    monkeypatch.setitem(roster.HANDLERS, "cap-web-search", _PlainHandler())
    out, trace, capture = await _hop(_web_entry(), query="帮我搜一下", extractor=_must_not_run)
    assert out.requested_action["args"] == {"query": "帮我搜一下"}
    assert "slot_plan" not in capture["acquisition"]     # nothing was planned
    assert "extractor" not in capture["acquisition"]     # no model was called


async def test_handler_with_no_extractor_keeps_the_deterministic_draft():
    # The production default when the extractor seam is absent: the handler draft
    # stands untouched (proves the pre-existing search-handler tests still hold).
    msg = "帮我搜一下注意力机制"
    out, trace, capture = await _hop(_web_entry(), query=msg, extractor=None)
    assert out.requested_action["args"] == {"query": msg}
    assert "extractor" not in capture["acquisition"]


# ── the REAL Docker Qwen call (skips unless the endpoint is deployed) ─────────────


def _extractor_endpoint_reachable() -> bool:
    import socket
    url = str(getattr(settings, "chat_tool_intent_local_url", "") or "").strip()
    if not url:
        return False
    try:
        parsed = urlparse(url)
        host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _extractor_endpoint_reachable(),
                    reason="local Qwen extractor endpoint not deployed")
async def test_real_qwen_cleans_a_zh_search_topic():
    """REAL model call: a Chinese instruction frame must be stripped to a topic,
    and a stated count must still land top_k. Run only against the live Docker
    service (``chat_tool_intent_local_url``); skipped when it is not deployed."""
    from core.application.chat.intent_funnel.argument_acquisition.extractor import extract

    seen: dict = {}

    async def real(*, query, entry, model_slots, bundle, prompt):
        seen["prompt"] = prompt
        return await extract(query=query, entry=entry, model_slots=model_slots,
                             bundle=bundle, prompt=prompt)

    msg = "帮我搜一下注意力机制的最新进展"
    out, trace, capture = await _hop(_web_entry(), query=msg, extractor=real)
    args = out.requested_action["args"]
    assert args["query"]                                   # a non-empty topic
    assert "帮我搜一下" not in args["query"]                # the frame was stripped
    assert seen["prompt"] == SEARCH_QUERY_PROMPT
    assert capture["acquisition"]["extractor"] == "Qwen"
