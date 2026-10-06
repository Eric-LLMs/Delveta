"""cap_handler — RagSearchHandler + its wiring into ``_acquisition_hop``.

Two layers are pinned here:

* the HANDLER contract itself (unit): ``cap-rag-search`` copies the turn's
  sentence VERBATIM as ``query``, NEVER emits ``top_k`` (tool default) or
  ``domain`` (no fact source), and returns ``None`` for a blank query;
* the WIRING: a capability whose handler is registered short-circuits the
  generic acquisition chain (provider / path_router / Qwen) and its draft goes
  through the SAME ``_certify`` -> Binder -> ``tool_intent`` handoff, while a
  capability with no handler is untouched.

The funnel test drives the REAL ``funnel.route`` -> orchestrator cascade with a
faked Matcher HIT (mirroring ``test_phase4_acquisition_step2``); no provider /
extractor is injected, so any fall-through would exit to the Agent — proving the
handler, not the generic chain, produced the certified action.
"""
from __future__ import annotations

import logging
import re
import types

import pytest

from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    RagSearchHandler,
    handler_for,
)
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    REASON_ACQUISITION_MISSING,
    MatchResult,
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

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"

MESSAGE = "帮我搜一下注意力机制"


# ── Layer 1: the RagSearchHandler contract (unit) ────────────────────────────────


async def test_rag_query_is_the_verbatim_sentence():
    handler = RagSearchHandler()
    draft = await handler.acquire(query=f"  {MESSAGE}  ", facts=None)
    assert draft == {"query": MESSAGE}          # only surrounding spaces stripped


async def test_rag_never_emits_top_k_or_domain():
    handler = RagSearchHandler()
    draft = await handler.acquire(query=MESSAGE, facts=types.SimpleNamespace())
    assert set(draft) == {"query"}              # top_k/domain deliberately absent


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_rag_blank_query_returns_none(blank):
    handler = RagSearchHandler()
    assert await handler.acquire(query=blank, facts=None) is None


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_rag_search():
    handler = handler_for("cap-rag-search")
    assert isinstance(handler, RagSearchHandler)
    assert handler.capability_id == "cap-rag-search"
    assert HANDLERS["cap-rag-search"] is handler


def test_roster_leaves_other_capabilities_on_the_generic_chain():
    # capabilities with no handler stay on the generic acquisition chain.
    for cid in ("cap-open-pdf", ""):
        assert handler_for(cid) is None


# ── Layer 3: the funnel wiring (real cascade, faked Matcher HIT) ─────────────────


def _ctx(message: str):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _rag_entry() -> CapabilityEntry:
    q = "搜索知识库"
    return CapabilityEntry(
        capability_id="cap-rag-search", tool_binding="rag_search",
        description="search the knowledge base",
        standard_queries=(QueryRecord(id="cap-rag-search-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "query": {"type": "string", "required": True, "description": "query"},
            "top_k": {"type": "integer", "required": False, "description": "count"},
            "domain": {"type": "string", "required": False, "description": "domain id"},
        },
    )


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="搜索知识库")

    async def fake_select_and_extract(query, candidates, *, entries_by_id,
                                      llm=None, facts=None):
        rec.legacy.append(query)
        raise AssertionError("legacy select_and_extract must never run on the new lane")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(tool_intent_mod, "select_and_extract", fake_select_and_extract)
    monkeypatch.setattr(settings, "chat_cap_router_backend", "stub")
    return rec, view


class _Rec:
    def __init__(self) -> None:
        self.legacy: list = []


def _deps(provider):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider)


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_rag_turn_certifies_verbatim_query_via_handler(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _rag_entry()
    rec, view = _wire(monkeypatch, entry)
    # no provider injected: any generic-chain fall-through → Agent (undeclared).
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE), _deps(None), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "rag_search"
    assert act["capability_id"] == "cap-rag-search"
    assert act["args"] == {"query": MESSAGE}           # verbatim; no top_k/domain
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_handler_bypasses_a_populated_generic_chain(monkeypatch, caplog):
    """Even with generic provider inputs that WOULD have produced a different
    draft, the registered handler owns the result — proof the branch precedes
    the generic chain."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _rag_entry()
    rec, view = _wire(monkeypatch, entry)
    poisoned = {"cap-rag-search": AcquisitionInputs(
        model_values={"query": "HALLUCINATED", "top_k": "999", "domain": "d1"})}
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE), _deps(lambda cid: poisoned.get(cid)), req)

    assert out is not req
    assert out.requested_action["args"] == {"query": MESSAGE}
    assert not rec.legacy


async def test_whitespace_turn_is_guardrail_vetoed(monkeypatch, caplog):
    """A blank/whitespace message never reaches the cascade at all — it is
    caught upstream as non-pure text, so the handler is never consulted."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _rag_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("   "), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert lines == []                       # the cascade never ran
    assert not rec.legacy


async def test_handler_returning_none_is_acquisition_missing(monkeypatch):
    """The orchestrator branch: a handler with no legal draft (``None``) exits
    the turn to the Agent with ``ACQUISITION_MISSING`` — fail-closed."""
    from core.application.chat.intent_funnel import orchestrator as orch
    from core.application.chat.intent_funnel.cap_handler import roster

    class _NoneHandler:
        capability_id = "cap-rag-search"

        async def acquire(self, *, query, facts):
            return None

    monkeypatch.setitem(roster.HANDLERS, "cap-rag-search", _NoneHandler())
    entry = _rag_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    trace: dict = {}
    out = await orch._acquisition_hop(
        TurnRequirements(), types.SimpleNamespace(), entry,
        query=MESSAGE, facts=None, view=view, trace=trace, capture=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
