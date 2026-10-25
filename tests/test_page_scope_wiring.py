"""Page-scope wiring — the online-model lane through the acquisition hop.

Pins the integration boundary rather than the resolver's own rules:

* only the handlers that declare ``accepts_page_scope`` receive the resolver;
* the resolver is called ONLY when the turn already carries a document fact —
  a document-less turn keeps its existing missing/clarify exit untouched;
* a valid model verdict is the page producer: an explicit page / range or a
  whole document overrides the handler's viewer page;
* only a SEMANTIC ``ambiguous`` verdict keeps the viewer's current page of the
  same document;
* an unavailable / invalid verdict, a resolver fault or a missing seam ends the
  turn as TERMINAL (``ACQUISITION_TERMINAL``) — never an Agent hand-off, never a
  guess, never a silent widening to the whole document and never a disguised
  success on the handler's page;
* the draft still passes the SAME Binder -> certified ACTION handoff, and the
  document identity is always the turn fact's ``asset_id`` — never model output.
"""
from __future__ import annotations

import asyncio
import types

from core.application.chat.intent_funnel import orchestrator as orch_mod
from core.application.chat.intent_funnel import policy as policy_mod
from core.application.chat.intent_funnel.argument_acquisition import (
    page_scope_resolver as psr,
)
from core.application.chat.intent_funnel.contract import (
    REASON_ACQUISITION_MISSING,
    REASON_ACQUISITION_TERMINAL,
    REASON_ACQUISITION_TIMEOUT,
    REASON_CAP_ROUTER_TIMEOUT,
    TurnFacts,
)
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements
from core.config import settings

_READ_DOC_PARAMS = {
    "asset_id": {"type": "string", "required": True,
                 "description": "from turn facts"},
    "pages": {"type": "string", "required": False,
              "description": "page spec"},
}
_EXTRACT_PARAMS = dict(_READ_DOC_PARAMS)


def _entry(cid: str, tool: str, params: dict) -> CapabilityEntry:
    q = "做点事"
    return CapabilityEntry(
        capability_id=cid, tool_binding=tool, description="d",
        standard_queries=(QueryRecord(id=f"{cid}-q", query=q,
                                      language=derive_language(q)),),
        parameters=params,
    )


def _view(entry: CapabilityEntry) -> RegistryLiveView:
    return RegistryLiveView(fingerprint=content_fingerprint([entry]),
                            entries=(entry,))


def _facts(**over):
    base = {
        "has_viewer": True, "viewer_asset_id": "asset-123",
        "viewer_current_page": 4, "viewer_kind": "pdf",
        "viewer_name": "survey.pdf", "viewer_page_from": 4,
        "viewer_page_to": 4,
    }
    base.update(over)
    return TurnFacts(**base)


def _scope(**over) -> psr.PageScope:
    base = {
        "target": psr.TARGET_DOCUMENT, "page_scope": psr.SCOPE_ALL,
        "pages": None, "status": psr.STATUS_OK, "reason": "",
    }
    base.update(over)
    return psr.PageScope(**base)


def _resolver(scope, *, seen=None):
    async def _resolve(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        return scope
    return _resolve


async def _hop(entry, *, query, facts, resolver=None, history=(), trace=None):
    trace = {} if trace is None else trace
    capture: dict = {}
    deps = types.SimpleNamespace(argument_extractor=None,
                                 page_scope_resolver=resolver)
    out = await orch_mod._acquisition_hop(
        TurnRequirements(), deps, entry, query=query, facts=facts,
        view=_view(entry), trace=trace, capture=capture, history=history)
    return out, trace, capture


async def test_hop_certifies_the_resolver_pages_and_records_the_verdict():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)
    seen: dict = {}

    out, trace, capture = await _hop(
        entry, query="extract pages 3 to 5", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_RANGE, pages="3-5"),
                           seen=seen))

    assert out.requested_action["tool"] == "read_document"
    assert out.requested_action["args"] == {"asset_id": "asset-123",
                                            "pages": "3-5"}
    assert trace["acquisition"] == "CAP_HANDLER"
    assert capture["page_scope"] == {
        "capability_id": "cap-read-document",
        "status": psr.STATUS_OK,
        "target": psr.TARGET_DOCUMENT,
        "scope": psr.SCOPE_RANGE,
        "pages": "3-5",
        "reason": "",
        "usage": {},
    }
    assert seen["capability"] == "cap-read-document"
    assert seen["asset_id"] == "asset-123"


async def test_hop_passes_the_sanctioned_history_through_to_the_resolver():
    entry = _entry("cap-pdf-extract-text", "pdf_extract_text", _EXTRACT_PARAMS)
    seen: dict = {}

    await _hop(entry, query="extract that range",
               facts=_facts(),
               resolver=_resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="2"),
                                  seen=seen),
               history=[{"role": "user", "content": "pages 1 to 3"},
                        {"role": "assistant", "content": "ignored"}])

    assert seen["history"] == [{"role": "user", "content": "pages 1 to 3"},
                               {"role": "assistant", "content": "ignored"}]


async def test_hop_all_scope_certifies_without_a_pages_slot():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)

    out, _, _ = await _hop(entry, query="summarise it", facts=_facts(),
                           resolver=_resolver(_scope()))

    assert out.requested_action["args"] == {"asset_id": "asset-123"}


async def test_hop_unavailable_resolver_is_terminal_and_never_falls_back():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)

    out, trace, capture = await _hop(
        entry, query="summarise pages 3-5", facts=_facts(),
        resolver=_resolver(_scope(status=psr.STATUS_UNAVAILABLE,
                                  reason="timeout")))

    # A transport fault is a technical failure: it must neither become a
    # whole-document read the user never asked for nor be disguised as a
    # successful read of the viewer's page.
    assert out is not None and out.terminal_error is not None
    assert trace["fallback"] == REASON_ACQUISITION_TERMINAL
    assert capture["page_scope"]["status"] == psr.STATUS_UNAVAILABLE
    assert capture["page_scope_evidence"]["source"] == "none"
    assert capture["page_scope_evidence"]["merged"]["pages"] is None
    assert capture["page_scope_evidence"]["handler_pages"] == "4"
    assert capture["page_scope_evidence"]["fallback_applied"] is False


async def test_hop_ambiguous_resolver_keeps_the_viewer_page():
    entry = _entry("cap-pdf-extract-text", "pdf_extract_text", _EXTRACT_PARAMS)

    out, trace, capture = await _hop(
        entry, query="extract some pages", facts=_facts(),
        resolver=_resolver(_scope(status=psr.STATUS_AMBIGUOUS,
                                  page_scope=psr.SCOPE_UNKNOWN)))

    # A semantic "not extracted" reply keeps the handler value: the viewer is
    # showing the same document on page 4.
    assert out is not None and out.terminal_error is None
    assert out.requested_action["args"] == {"asset_id": "asset-123",
                                            "pages": "4"}
    assert trace["acquisition"] == "CAP_HANDLER"
    assert capture["page_scope"]["status"] == psr.STATUS_AMBIGUOUS
    assert capture["page_scope_evidence"]["source"] == "viewer"
    assert capture["page_scope_evidence"]["fallback_applied"] is True


async def test_hop_without_the_seam_is_unsettled_not_the_viewer_page():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)

    out, trace, capture = await _hop(
        entry, query="summarise it", facts=_facts(viewer_page_from=3,
                                                  viewer_page_to=5))

    # No semantic judge: the missing seam is a technical failure, so the turn
    # ends. Neither the viewer's open range nor its current page is substituted.
    assert out is not None and out.terminal_error is not None
    assert trace["fallback"] == REASON_ACQUISITION_TERMINAL
    assert "page_scope" not in capture
    assert capture["page_scope_evidence"]["source"] == "none"
    assert capture["page_scope_evidence"]["llm"]["status"] == "not_wired"
    assert capture["page_scope_evidence"]["handler_pages"] == "4"
    assert capture["page_scope_evidence"]["fallback_applied"] is False


async def test_hop_never_calls_the_resolver_without_a_document_fact():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)

    async def _boom(**kwargs):
        raise AssertionError("resolver must not run without a document")

    out, trace, capture = await _hop(entry, query="summarise it",
                                     facts=_facts(viewer_asset_id=""),
                                     resolver=_boom)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
    assert "page_scope" not in capture


async def test_hop_does_not_hand_the_seam_to_a_handler_that_does_not_accept_it():
    entry = _entry("cap-read-file", "read_file", {
        "path": {"type": "string", "required": True, "description": "path"},
    })

    async def _boom(**kwargs):
        raise AssertionError("this capability has no page axis")

    out, _, capture = await _hop(entry, query="read ./data/train.csv",
                                 facts=None, resolver=_boom)

    assert out.requested_action["args"] == {"path": "./data/train.csv"}
    assert "page_scope" not in capture


# ── the other two page-addressed capabilities ─────────────────────────────────


async def test_hop_hands_the_seam_to_the_vision_page_axis():
    entry = _entry("cap-vision", "vision", {
        "asset_id": {"type": "string", "required": True},
        "question": {"type": "string", "required": False},
        "page": {"type": "integer", "required": False},
    })

    out, _, capture = await _hop(
        entry, query="what is on page 3", facts=_facts(),
        resolver=_resolver(_scope(page_scope=psr.SCOPE_SINGLE, pages="3")))

    assert out.requested_action["args"] == {
        "asset_id": "asset-123", "question": "what is on page 3", "page": "3"}
    assert capture["page_scope"]["pages"] == "3"


async def test_hop_summary_never_consults_the_page_scope_seam():
    entry = _entry("cap-summary", "summary_gen", {
        "asset_id": {"type": "string", "required": False},
        "pages": {"type": "string", "required": False},
        "context": {"type": "string", "required": False},
        "output_dir": {"type": "string", "required": False},
        "auto": {"type": "string", "required": False},
    })
    history = [{"role": "user", "content": "什么是 RRF？"},
               {"role": "assistant", "content": "RRF 是倒数排名融合。"}]

    async def _boom(**kwargs):
        raise AssertionError("cap-summary has no page-scope seam")

    out, _, capture = await _hop(
        entry, query="总结一下我们刚才聊的", facts=_facts(),
        resolver=_boom, history=history)

    # The summary tool's FIRST call decides the target, the readiness and the save
    # intent. No page-scope model runs on this lane, so a broken resolver cannot
    # turn the turn into a document read or an Agent hand-off.
    assert out.requested_action["tool"] == "summary_gen"
    assert out.requested_action["args"] == {"auto": "1", "asset_id": "asset-123"}
    assert "page_scope" not in capture


# ── the acquisition lane is its OWN stage for the wall-clock budget ─────────────


async def test_the_acquisition_stage_is_current_before_the_resolver_runs():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)
    trace: dict = {"stage": "cap_router"}
    seen: list = []

    async def _resolver_seeing_the_stage(**kwargs):
        seen.append(trace.get("stage"))
        return _scope()

    await _hop(entry, query="summarise it", facts=_facts(),
               resolver=_resolver_seeing_the_stage, trace=trace)

    # Selection has already returned before the resolver is consulted: the
    # budget must be attributable to THIS hop from its first await onward.
    assert seen == ["acquisition"]


async def test_an_acquisition_timeout_is_not_reported_as_a_cap_router_timeout(
        monkeypatch):
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)

    async def _slow_resolver(**kwargs):
        await asyncio.sleep(1.0)
        return _scope()

    deps = types.SimpleNamespace(argument_extractor=None,
                                 page_scope_resolver=_slow_resolver)
    trace: dict = {}
    monkeypatch.setattr(settings, "chat_funnel_timeout_seconds", 0.05)

    async def _fake_run_nodes(ctx, deps_, requirements, trace_, **kwargs):
        # The selection hop just finished and left its own stage behind.
        trace_["stage"] = "cap_router"
        return await orch_mod._acquisition_hop(
            requirements, deps_, entry, query="summarise it", facts=_facts(),
            view=_view(entry), trace=trace_, capture=kwargs.get("capture"))

    monkeypatch.setattr(orch_mod, "run_nodes", _fake_run_nodes)

    out = await orch_mod.run_cascade(None, deps, TurnRequirements(), trace)

    # The hop advanced the stage before awaiting: a timeout in the page-scope
    # call belongs to acquisition, not to the selection hop that returned.
    assert out is None
    assert trace["stage"] == "acquisition"
    assert trace["fallback"] == REASON_ACQUISITION_TIMEOUT


def test_stage_reason_names_the_acquisition_lane():
    assert policy_mod.stage_reason(
        "acquisition", timed_out=True) == REASON_ACQUISITION_TIMEOUT
    # The selection hop keeps its own code — the new one never shadows it.
    assert policy_mod.stage_reason(
        "cap_router", timed_out=True) == REASON_CAP_ROUTER_TIMEOUT
