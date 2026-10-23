"""Page-scope wiring — the resolver seam through the acquisition hop.

Pins the integration boundary rather than the resolver's own rules:

* only the handlers that declare ``accepts_page_scope`` receive the resolver;
* the resolver is called ONLY when the turn already carries a document fact —
  a document-less turn keeps its existing missing/clarify exit untouched;
* an unsettled verdict exits through the EXISTING acquisition-failure mechanism
  (``ACQUISITION_MISSING``) and records its own status in the telemetry, so
  "LLM unavailable", "ambiguous", "invalid" and "permission/missing" are never
  reported as the same event;
* without the seam (tests, undecorated deps) both handlers keep the legacy
  viewer-window draft byte-for-byte;
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


async def test_hop_unavailable_resolver_keeps_its_own_status_and_exits_missing():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)

    out, trace, capture = await _hop(
        entry, query="summarise pages 3-5", facts=_facts(),
        resolver=_resolver(_scope(status=psr.STATUS_UNAVAILABLE,
                                  reason="timeout")))

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
    assert capture["page_scope"]["status"] == psr.STATUS_UNAVAILABLE
    assert capture["page_scope"]["pages"] is None


async def test_hop_ambiguous_resolver_does_not_widen_to_the_whole_document():
    entry = _entry("cap-pdf-extract-text", "pdf_extract_text", _EXTRACT_PARAMS)

    out, trace, capture = await _hop(
        entry, query="extract some pages", facts=_facts(),
        resolver=_resolver(_scope(status=psr.STATUS_AMBIGUOUS)))

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
    assert capture["page_scope"]["status"] == psr.STATUS_AMBIGUOUS


async def test_hop_without_the_seam_keeps_the_legacy_viewer_window():
    entry = _entry("cap-read-document", "read_document", _READ_DOC_PARAMS)

    out, trace, capture = await _hop(
        entry, query="summarise 3-5", facts=_facts(viewer_page_from=3,
                                                   viewer_page_to=5))

    assert out.requested_action["args"] == {"asset_id": "asset-123",
                                            "pages": "3-5"}
    assert trace["acquisition"] == "CAP_HANDLER"
    assert "page_scope" not in capture


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
