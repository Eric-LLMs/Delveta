"""cap_handler — AddTermHandler and the add_term WRITE hardening.

Layers pinned here:

* the HANDLER contract (unit): ``cap-add-term``'s two public slots are ``term``
  and ``domain``. Per the acquisition contract BOTH are MODEL-owned natural-
  language slots, so the handler runs NO extraction rule — a quoted span, a
  ``把/将`` clause, an English insert verb and a vocabulary-name regex all had no
  requirement basis and are removed. ``acquire()`` returns the EMPTY ``{}`` draft
  for ANY non-blank sentence and ``None`` only for a blank one; ``slot_plan()``
  authorizes the model for both slots. The tool keeps the domain NAME→entity
  resolution and the tool enum; ``definition`` is NEVER emitted.
* the TOOL hardening: ``add_term`` explicitly declares ``{WRITE}`` — without it
  the auto-classifier defaults this INSERT tool to READ.
* the FUNNEL wiring: ``handler_for("cap-add-term")`` owns the draft and short-
  circuits the generic acquisition chain; the empty draft is completed by the
  shared extractor (``deps.argument_extractor``) and only the authorized slots
  are folded, then the SAME ``_certify`` -> Binder -> ``tool_intent`` gate runs.
  With the extractor seam ABSENT the required slots stay unfilled and the Binder
  exits ``BIND_MISSING`` — the Agent still owns the turn, but for the right
  reason (no legal value, never a pre-model short-circuit).
* the model-assisted round-trip lives in ``test_orchestrator_slot_extraction``.

The 44 ``cap-add-term`` cases of the frozen Formal-500 argument benchmark
(``logs/_argbench/dataset_formal500.jsonl``) are reused here as a REPRESENTATIVE
corpus: every one must now yield the EMPTY handler draft — the old DET-only
baseline (which byte-matched the gold spans) is exactly what the contract
retires. The gold itself is NOT changed.
"""
from __future__ import annotations

import logging
import re
import types
from pathlib import Path

import pytest
from agent.tools.definition import classify_permissions
from agent.tools.tool_permissions import ToolPermission
from api.tools import add_term_tool
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    AddTermHandler,
    handler_for,
)
from core.application.chat.intent_funnel.contract import (
    MATCH_HIT,
    REASON_ACQUISITION_MISSING,
    REASON_BIND_MISSING,
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

GOLD_PATH = "logs/_argbench/dataset_formal500.jsonl"

# The frozen arg benchmark lives in the untracked ``logs/`` scratch tree; the
# representative corpus below is skipped when it is absent.
_GOLD_PRESENT = Path(GOLD_PATH).exists()


def _gold_messages() -> list[str]:
    if not _GOLD_PRESENT:
        return []
    import json

    rows = [
        json.loads(line)
        for line in Path(GOLD_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r["query"] for r in rows if r.get("capability_id") == "cap-add-term"]


GOLD_MESSAGES = _gold_messages()


# ── Layer 0: the handler runs NO rule — the empty draft for every gold case ───────


@pytest.mark.skipif(not _GOLD_PRESENT,
                    reason="frozen arg benchmark logs/_argbench/dataset_formal500.jsonl not present")
@pytest.mark.parametrize("message", GOLD_MESSAGES)
async def test_handler_yields_the_empty_draft_for_every_gold_case(message):
    # No extraction rule: a quoted term + a named domain in the sentence does NOT
    # populate the draft — the model owns both slots.
    handler = AddTermHandler()
    assert await handler.acquire(query=message, facts=None) == {}


@pytest.mark.skipif(not _GOLD_PRESENT, reason="benchmark not present")
def test_the_representative_corpus_is_the_44_gold_cases():
    assert len(GOLD_MESSAGES) == 44


# ── Layer 1: the handler contract (empty draft on any input; None on blank) ───────


@pytest.mark.parametrize("message", [
    '把"keystone"加入我的工程词汇库',                 # quoted span — no rule
    "帮我把 arch 这个词收进架构生词本",                # 把 … 这个词 — no rule
    "add gradient to my machine learning vocabulary",  # EN insert verb — no rule
    "帮我加个词到词库里",                             # no term at all
    "add this word to the glossary",                  # deictic
])
async def test_acquire_always_returns_the_empty_draft(message):
    handler = AddTermHandler()
    assert await handler.acquire(query=message, facts=None) == {}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_fails_closed(blank):
    handler = AddTermHandler()
    assert await handler.acquire(query=blank, facts=None) is None


async def test_slot_plan_authorizes_both_slots():
    handler = AddTermHandler()
    msg = "帮我加个词到词库里"
    draft = await handler.acquire(query=msg, facts=None)
    plan = handler.slot_plan(query=msg, facts=None, draft=draft)
    assert set(plan.model_slots) == {"term", "domain"}
    assert plan.default_slots == ()


async def test_asset_context_never_supplies_a_term_or_domain():
    # facts are ignored on purpose: no viewer/attachment id can leak in as a term
    # or a domain. The empty draft is the model's gap, never the asset's.
    handler = AddTermHandler()
    facts = types.SimpleNamespace(
        has_viewer=True, viewer_asset_id="abc", viewer_current_page=1,
        viewer_page_from=None, viewer_page_to=None, has_viewer_selection=False,
        has_attachment=True, attachment_asset_id="abc",
        path_asset_id="abc", has_turn_context=True)
    assert await handler.acquire(query="帮我加个词到词库里", facts=facts) == {}


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_add_term():
    handler = handler_for("cap-add-term")
    assert isinstance(handler, AddTermHandler)
    assert handler.capability_id == "cap-add-term"
    assert HANDLERS["cap-add-term"] is handler


def test_cap_add_term_is_no_longer_on_the_generic_chain():
    assert handler_for("cap-add-term") is not None
    for cid in ("cap-open-pdf", ""):
        assert handler_for(cid) is None


# ── Layer 3: the funnel wiring (real cascade, faked Matcher HIT) ─────────────────


def _ctx(message: str):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _add_term_entry() -> CapabilityEntry:
    q = "把生词加入词库"
    return CapabilityEntry(
        capability_id="cap-add-term", tool_binding="add_term",
        description="Add a word to a vocabulary domain named by the user.",
        standard_queries=(QueryRecord(id="cap-add-term-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "term": {"type": "string", "required": True, "max_len": 120,
                     "description": "the word to add, quotes removed"},
            "domain": {"type": "string", "required": True, "max_len": 60,
                       "description": "the vocabulary domain to add it into"},
        },
    )


class _Rec:
    def __init__(self) -> None:
        self.legacy: list = []


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    rec = _Rec()

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="把生词加入词库")

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


def _deps(provider, extractor=None):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider, argument_extractor=extractor)


def _mock_extractor(returned, boom=False):
    async def _extract(*, query, entry, model_slots, bundle, prompt):
        if boom:
            from core.application.chat.intent_funnel.argument_acquisition.extractor import (
                ExtractionUnavailable,
            )
            raise ExtractionUnavailable("down")
        return returned, bundle.source
    return _extract


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_complete_turn_certifies_the_model_payload(monkeypatch):
    """The model fills both authorized slots; the certified turn carries them."""
    entry = _add_term_entry()
    rec, view = _wire(monkeypatch, entry)
    extractor = _mock_extractor({"term": "keystone", "domain": "工程词汇库"})
    req = TurnRequirements()
    out = await _route(_ctx('把"keystone"加入我的工程词汇库'),
                       _deps(None, extractor), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "add_term"
    assert act["capability_id"] == "cap-add-term"
    assert act["args"] == {"term": "keystone", "domain": "工程词汇库"}
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_model_unavailable_leaves_required_slots_missing(monkeypatch, caplog):
    """An unavailable extractor fabricates nothing: both required slots stay
    unfilled and the Binder exits ``BIND_MISSING``."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _add_term_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx('把 "keystone" 加入我的工程词汇库'),
                       _deps(None, _mock_extractor({}, boom=True)), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_BIND_MISSING
    assert not rec.legacy


async def test_empty_draft_without_extractor_exits_bind_missing(monkeypatch, caplog):
    """No model value and no extractor seam: the SAME Binder gate exits
    ``BIND_MISSING`` (no legal value), never an ``ACQUISITION_MISSING`` bail."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _add_term_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("帮我加个词到词库里"), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_BIND_MISSING
    assert not rec.legacy                              # no legacy extraction


async def test_a_partially_filled_model_reply_still_exits_bind_missing(monkeypatch, caplog):
    """The model names only ``term``; the required ``domain`` stays MISSING and the
    turn escalates to the Agent to clarify — the tool never runs."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _add_term_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx('把 "正则化" 加到词库'),
                       _deps(None, _mock_extractor({"term": "正则化"})), req)

    assert out is req
    assert out.requested_action is None
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_BIND_MISSING


async def test_handler_bypasses_a_forged_generic_chain(monkeypatch):
    """A forged MODEL value in the generic chain never reaches the draft: the
    handler branch precedes the generic acquisition chain."""
    entry = _add_term_entry()
    rec, _view = _wire(monkeypatch, entry)
    poisoned = {"cap-add-term": AcquisitionInputs(
        model_values={"term": "HALLUCINATED", "domain": "HALLUCINATED"})}
    req = TurnRequirements()
    extractor = _mock_extractor({"term": "keystone", "domain": "工程词汇库"})
    out = await _route(_ctx('把"keystone"加入我的工程词汇库'),
                       _deps(lambda cid: poisoned.get(cid), extractor), req)

    assert out is not req
    assert out.requested_action["args"] == {"term": "keystone", "domain": "工程词汇库"}
    assert not rec.legacy


async def test_handler_returning_none_is_acquisition_missing(monkeypatch):
    """The orchestrator branch: a handler with no legal draft (``None``) exits
    the turn to the Agent with ``ACQUISITION_MISSING`` — fail-closed."""
    from core.application.chat.intent_funnel import orchestrator as orch
    from core.application.chat.intent_funnel.cap_handler import roster

    class _NoneHandler:
        capability_id = "cap-add-term"

        async def acquire(self, *, query, facts):
            return None

    monkeypatch.setitem(roster.HANDLERS, "cap-add-term", _NoneHandler())
    entry = _add_term_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    trace: dict = {}
    out = await orch._acquisition_hop(
        TurnRequirements(), types.SimpleNamespace(), entry,
        query='把 "x" 加入工程词汇库', facts=None, view=view, trace=trace, capture=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING


# ── Layer 4: add_term declares WRITE explicitly ──────────────────────────────────


class _CapRuntime:
    def __init__(self) -> None:
        self.tools: dict = {}

    def register(self, defn) -> None:
        self.tools[defn.name] = defn


def test_add_term_declares_write_explicitly():
    runtime = _CapRuntime()
    ctx = types.SimpleNamespace(resolve=lambda key: None)
    add_term_tool.register(runtime, ctx, None)
    defn = runtime.tools["add_term"]

    # The declaration is EXPLICIT (not None): without it the auto-classifier
    # (term/domain/definition carry no write hint) would default this INSERT to READ.
    assert defn.permission is not None
    assert ToolPermission.WRITE in defn.permission
    assert classify_permissions(defn) == frozenset({ToolPermission.WRITE})
