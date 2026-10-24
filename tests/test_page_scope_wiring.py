"""Page-scope wiring — the online-model lane through the acquisition hop.

Pins the integration boundary rather than the resolver's own rules:

* only the handlers that declare ``accepts_page_scope`` receive the resolver;
* the resolver is called ONLY when the turn already carries a document fact —
  a document-less turn keeps its existing missing/clarify exit untouched;
* the model's verdict is the ONLY page producer: an unavailable / ambiguous /
  invalid verdict ends the turn as TERMINAL (``ACQUISITION_TERMINAL``) — never
  an Agent hand-off, never a guess, never a silent widening to the whole
  document;
* without the seam (tests, undecorated deps) the scope is equally unsettled —
  the viewer's open page/range is never substituted for a stated scope;
* the draft still passes the SAME Binder -> certified ACTION handoff, and the
  document identity is always the turn fact's ``asset_id`` — never model output.
"""
from __future__ import annotations

import types

from core.application.chat.intent_funnel import orchestrator as orch_mod
from core.application.chat.intent_funnel.argument_acquisition import (
    page_scope_resolver as psr,
)
from core.application.chat.intent_funnel.contract import (
    REASON_ACQUISITION_MISSING,
    REASON_ACQUISITION_TERMINAL,
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


async def _hop(entry, *, query, facts, resolver=None, history=()):
    trace: dict = {}
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


async def test_hop_unavailable_resolver_is_terminal_and_never_widens():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)

    out, trace, capture = await _hop(
        entry, query="summarise pages 3-5", facts=_facts(),
        resolver=_resolver(_scope(status=psr.STATUS_UNAVAILABLE,
                                  reason="timeout")))

    # No rule exists to fall back on: a transport fault must not become a
    # whole-document read the user never asked for.
    assert out is not None and out.terminal_error is not None
    assert trace["fallback"] == REASON_ACQUISITION_TERMINAL
    assert capture["page_scope"]["status"] == psr.STATUS_UNAVAILABLE
    assert capture["page_scope_evidence"]["source"] == "none"
    assert capture["page_scope_evidence"]["merged"]["pages"] is None


async def test_hop_ambiguous_resolver_is_terminal():
    entry = _entry("cap-pdf-extract-text", "pdf_extract_text", _EXTRACT_PARAMS)

    out, trace, capture = await _hop(
        entry, query="extract some pages", facts=_facts(),
        resolver=_resolver(_scope(status=psr.STATUS_AMBIGUOUS)))

    # The model did not settle the scope: TERMINAL, never a whole-document widening
    # and never an Agent hand-off that would guess the pages.
    assert out is not None and out.terminal_error is not None
    assert trace["fallback"] == REASON_ACQUISITION_TERMINAL
    assert capture["page_scope"]["status"] == psr.STATUS_AMBIGUOUS
    assert capture["terminal"]["capability_id"] == "cap-pdf-extract-text"


async def test_hop_without_the_seam_is_unsettled_not_the_viewer_window():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)

    out, trace, capture = await _hop(
        entry, query="summarise it", facts=_facts(viewer_page_from=3,
                                                  viewer_page_to=5))

    # No semantic judge, so no page decision: the turn ends. The viewer's open
    # range is NEVER substituted merely because a viewer page exists.
    assert out is not None and out.terminal_error is not None
    assert trace["fallback"] == REASON_ACQUISITION_TERMINAL
    assert "page_scope" not in capture
    assert capture["page_scope_evidence"]["source"] == "none"
    assert capture["page_scope_evidence"]["llm"]["status"] == "not_wired"


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


async def test_hop_summary_conversation_survives_an_unknown_page_scope():
    entry = _entry("cap-summary", "summary_gen", {
        "asset_id": {"type": "string", "required": False},
        "pages": {"type": "string", "required": False},
        "context": {"type": "string", "required": False},
        "output_dir": {"type": "string", "required": False},
    })
    history = [{"role": "user", "content": "什么是 RRF？"},
               {"role": "assistant", "content": "RRF 是倒数排名融合。"}]

    out, _, capture = await _hop(
        entry, query="总结一下我们刚才聊的", facts=_facts(),
        resolver=_resolver(_scope(target=psr.TARGET_CONVERSATION,
                                  page_scope=psr.SCOPE_UNKNOWN)),
        history=history)

    # A conversation has no page axis: ``page_scope=unknown`` must not turn the
    # turn into a document read or an Agent hand-off. The Binder still settles
    # the asset_id context slot from the facts, and ``summary_gen``'s context
    # branch ignores it — exactly as before this change.
    assert out.requested_action["tool"] == "summary_gen"
    assert "context" in out.requested_action["args"]
    assert "RRF 是倒数排名融合" in out.requested_action["args"]["context"]
    assert "pages" not in out.requested_action["args"]
