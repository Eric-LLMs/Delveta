"""cap_handler — SocialSearchHandler + its wiring into ``_acquisition_hop``.

Two layers are pinned here:

* the HANDLER contract itself (unit): ``cap-social-search``'s ``acquire()``
  copies the turn's sentence VERBATIM as ``query`` (the deterministic FALLBACK
  the model may clean) and runs NO natural-language extraction rule — the ``limit``
  count rule and the NL platform detector had no requirement basis and are removed.
  One DETERMINISTIC structured token IS honoured: an explicit ``r/<name>`` scope is
  extracted into the tool-only ``subreddit`` arg, with ``platform`` pinned to reddit
  so the scope can never be silently widened downstream. Every natural-language slot
  (``query``, ``platform``) is MODEL-owned; the tool enum and name resolution stay
  tool-side. A blank query returns ``None``.
* the WIRING: a capability whose handler is registered owns its draft and its
  OPTIONAL ``slot_plan()`` authorizes the shared extractor (query cleanup + the
  named platform; an explicit ``r/<sub>`` scope drops ``platform`` from the model
  slots — see ``test_orchestrator_slot_extraction``); the draft goes through the
  SAME ``_certify`` -> Binder -> ``tool_intent`` handoff. The generic chain
  (provider / path_router) is NOT consulted for a handled capability.

The funnel test drives the REAL ``funnel.route`` -> orchestrator cascade with a
faked Matcher HIT (mirroring ``test_phase4_acquisition_step2``); no provider /
extractor is injected, so any fall-through would exit to the Agent — proving the
handler, not the generic chain, produced the certified action.
"""
from __future__ import annotations

import logging
import types

import pytest

from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import cap_router as cap_router_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    SocialSearchHandler,
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

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"

MESSAGE = "搜索社区里对注意力机制的讨论"


# ── Layer 1: the SocialSearchHandler contract (unit) ─────────────────────────────


async def test_social_query_is_the_verbatim_sentence():
    handler = SocialSearchHandler()
    draft = await handler.acquire(query=f"  {MESSAGE}  ", facts=None)
    assert draft == {"query": MESSAGE}          # only surrounding spaces stripped


@pytest.mark.parametrize("message", [
    MESSAGE,                                    # no platform at all
    "在 reddit 上搜一下 transformer 的讨论",       # a named platform is the MODEL's read
    "搜索社区讨论，取 3 条",                       # a stated count is the tool's default
])
async def test_social_acquire_emits_only_the_verbatim_query(message):
    # NO natural-language extraction rule runs: the platform is MODEL-owned and the
    # count is a tool default, so ``acquire()`` carries neither and the sentence is
    # the fallback. (An explicit ``r/<sub>`` token is the one deterministic
    # exception — see the scope tests below.)
    handler = SocialSearchHandler()
    draft = await handler.acquire(query=message, facts=None)
    assert draft == {"query": message}


@pytest.mark.parametrize("message,expected_sub", [
    ("r/MachineLearning 里怎么讨论的", "MachineLearning"),
    ("r/learnpython 上有没有关于 asyncio 的帖子", "learnpython"),
])
async def test_social_acquire_extracts_an_explicit_subreddit_scope(message, expected_sub):
    # An explicit ``r/<name>`` token is a deterministic STRUCTURED scope (not NL
    # guessing): it fills the tool-only ``subreddit`` arg and ``platform`` is pinned
    # to reddit so the search scope can never be silently widened downstream.
    handler = SocialSearchHandler()
    draft = await handler.acquire(query=message, facts=None)
    assert draft == {"query": message, "subreddit": expected_sub, "platform": "reddit"}


async def test_social_acquire_does_not_mistake_a_word_slash_for_a_scope():
    # ``car/road`` has no ``r/`` token boundary — it must NOT be read as a scope.
    handler = SocialSearchHandler()
    msg = "社区里 car/road 的讨论"
    draft = await handler.acquire(query=msg, facts=None)
    assert draft == {"query": msg}


async def test_social_slot_plan_authorizes_the_model_lane():
    handler = SocialSearchHandler()
    msg = "在 reddit 上搜一下 transformer"
    draft = await handler.acquire(query=msg, facts=None)
    plan = handler.slot_plan(query=msg, facts=None, draft=draft)
    # platform is MODEL-owned (the sentence names it); subreddit is NOT a model slot
    # (only a deterministic ``r/<sub>`` token fills it); limit is the tool default.
    assert set(plan.model_slots) == {"query", "platform"}
    assert plan.default_slots == ()


async def test_social_slot_plan_drops_platform_when_a_scope_is_pinned():
    # An explicit ``r/<sub>`` pins platform to reddit deterministically, so the model
    # must NOT be authorized to re-own platform (it could otherwise overwrite the pin
    # and widen the search); only ``query`` is model-owned that turn.
    handler = SocialSearchHandler()
    msg = "r/MachineLearning 上关于 transformer 的讨论"
    draft = await handler.acquire(query=msg, facts=None)
    plan = handler.slot_plan(query=msg, facts=None, draft=draft)
    assert plan.model_slots == ("query",)
    assert plan.default_slots == ()


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_social_blank_query_returns_none(blank):
    handler = SocialSearchHandler()
    assert await handler.acquire(query=blank, facts=None) is None


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_social_search():
    handler = handler_for("cap-social-search")
    assert isinstance(handler, SocialSearchHandler)
    assert handler.capability_id == "cap-social-search"
    assert HANDLERS["cap-social-search"] is handler


def test_roster_leaves_unwired_capabilities_on_the_generic_chain():
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


def _social_entry() -> CapabilityEntry:
    q = "搜索社区讨论"
    return CapabilityEntry(
        capability_id="cap-social-search", tool_binding="search_social",
        description="search social platforms",
        standard_queries=(QueryRecord(id="cap-social-search-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "query": {"type": "string", "required": True, "description": "query"},
            "platform": {"type": "string", "required": False, "description": "platform"},
            "subreddit": {"type": "string", "required": False, "description": "scope"},
            "limit": {"type": "integer", "required": False, "description": "count"},
        },
    )


class _Rec:
    def __init__(self) -> None:
        self.legacy: list = []


class _NoSelect:
    """Selector guard: a HIT-lane test fails loudly if selection is re-entered."""

    def __init__(self, rec: "_Rec") -> None:
        self.rec = rec

    async def select(self, query, candidates, *, entries_by_id, facts=None):
        self.rec.legacy.append(query)
        raise AssertionError("the selector must never run on a HIT lane")


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="搜索社区讨论")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    # A HIT is already decided: the ONE selector must never be consulted.
    monkeypatch.setattr(cap_router_mod, "default_selector", lambda: _NoSelect(rec))
    return rec, view


def _deps(provider):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider)


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_social_turn_certifies_verbatim_query_via_handler(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _social_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE), _deps(None), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "search_social"
    assert act["capability_id"] == "cap-social-search"
    assert act["args"] == {"query": MESSAGE}           # verbatim; no platform/limit
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_social_named_platform_is_left_to_the_model(monkeypatch, caplog):
    """A sentence naming a platform certifies with the VERBATIM query only — the
    handler no longer detokenizes it; the model (via slot_plan) owns ``platform``."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _social_entry()
    rec, view = _wire(monkeypatch, entry)
    msg = "在 reddit 上搜一下 transformer"
    req = TurnRequirements()
    out = await _route(_ctx(msg), _deps(None), req)

    assert out.requested_action["args"] == {"query": msg}


async def test_handler_bypasses_a_populated_generic_chain(monkeypatch, caplog):
    """Even with generic provider inputs that WOULD have produced a different
    draft, the registered handler owns the result — proof the branch precedes
    the generic chain."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _social_entry()
    rec, view = _wire(monkeypatch, entry)
    poisoned = {"cap-social-search": AcquisitionInputs(
        model_values={"query": "HALLUCINATED", "platform": "GLOBAL", "limit": "3"})}
    req = TurnRequirements()
    out = await _route(_ctx(MESSAGE), _deps(lambda cid: poisoned.get(cid)), req)

    assert out is not req
    assert out.requested_action["args"] == {"query": MESSAGE}
    assert not rec.legacy


async def test_whitespace_turn_is_guardrail_vetoed(monkeypatch, caplog):
    """A blank/whitespace message never reaches the cascade at all — it is
    caught upstream as non-pure text, so the handler is never consulted."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _social_entry()
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
        capability_id = "cap-social-search"

        async def acquire(self, *, query, facts):
            return None

    monkeypatch.setitem(roster.HANDLERS, "cap-social-search", _NoneHandler())
    entry = _social_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    trace: dict = {}
    out = await orch._acquisition_hop(
        TurnRequirements(), types.SimpleNamespace(), entry,
        query=MESSAGE, facts=None, view=view, trace=trace, capture=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
