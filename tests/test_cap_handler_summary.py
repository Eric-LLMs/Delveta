"""cap_handler — SummaryHandler + its funnel wiring.

The business contract pinned here:

* ``cap-summary`` runs NO Qwen extraction and NO page-scope model. The handler
  certifies only two deterministic outcomes: a literal workspace ``path`` when
  the sentence itself fully specifies one, else the explicit ``auto`` marker.
* the tool's FIRST LLM call owns the source decision (previous message /
  document / current page / session), the readiness and the save intent; a
  second call reads only the chosen material (``toolkit/summary_auto.py``).
* the Binder still injects ``asset_id`` from turn facts, so the open document's
  identity rides the action without being a model output.
"""
from __future__ import annotations

import types

import pytest
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import cap_router as cap_router_mod
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    SummaryHandler,
    handler_for,
)
from core.application.chat.intent_funnel.cap_handler.acquisition_failure import (
    CATEGORY_NO_CONTENT,
)
from core.application.chat.intent_funnel.contract import MATCH_HIT, MatchResult
from core.application.chat.intent_funnel.registry import content_fingerprint
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    RegistryLiveView,
    derive_language,
)
from core.application.chat.understanding import TurnRequirements


def _facts(**overrides):
    from core.application.chat.intent_funnel.contract import TurnFacts

    return TurnFacts(**overrides)


# ── Layer 1: the handler contract ───────────────────────────────────────────────


@pytest.mark.parametrize("message", [
    "总结一下",
    "总结前面几条的聊天内容",
    "总结第3页",
    "总结第3页到第5页内容",
    "summarize the whole session",
])
async def test_a_request_without_a_literal_path_is_the_auto_marker(message):
    handler = SummaryHandler()
    assert await handler.acquire(query=message, facts=None) == {"auto": "1"}


async def test_a_literal_workspace_path_is_copied_verbatim():
    handler = SummaryHandler()
    assert await handler.acquire(query="总结 ./reports/q3.md", facts=None) == {
        "path": "./reports/q3.md"}


async def test_a_turn_asset_wins_over_a_literal_path():
    """The turn already resolved a document identity: the tool decides from it."""
    handler = SummaryHandler()
    draft = await handler.acquire(
        query="总结 ./reports/q3.md",
        facts=_facts(viewer_asset_id="a1", viewer_kind="pdf"))
    assert draft == {"auto": "1"}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_fails_closed(blank):
    handler = SummaryHandler()
    draft = await handler.acquire(query=blank, facts=None)
    assert draft.category == CATEGORY_NO_CONTENT
    assert draft.terminal is False


def test_no_slot_plan_and_no_page_scope():
    handler = SummaryHandler()
    assert not hasattr(handler, "slot_plan")
    assert not getattr(handler, "accepts_page_scope", False)


# ── Layer 2: the roster wiring ──────────────────────────────────────────────────


def test_roster_wires_cap_summary():
    handler = handler_for("cap-summary")
    assert isinstance(handler, SummaryHandler)
    assert handler.capability_id == "cap-summary"
    assert HANDLERS["cap-summary"] is handler


# ── Layer 3: the funnel wiring ──────────────────────────────────────────────────


def _ctx(message: str, *, viewer=None):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=viewer),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


class _Viewer:
    def __init__(self, asset_id: str, kind: str = "pdf", page: int | None = 2):
        self.asset_id = asset_id
        self.kind = kind
        self.page = page
        self.page_from = page
        self.page_to = page
        self.provenance = "cloud"
        self.name = "report.pdf"
        self.t_ms = None
        self.duration_ms = None
        self.selections = []


def _summary_entry() -> CapabilityEntry:
    q = "总结内容"
    return CapabilityEntry(
        capability_id="cap-summary", tool_binding="summary_gen",
        description="summarize one content source",
        standard_queries=(QueryRecord(id="cap-summary-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "paths": {"type": "array", "required": False, "description": "files"},
            "path": {"type": "string", "required": False, "description": "one file"},
            "asset_id": {"type": "string", "required": False, "description": "doc id"},
            "pages": {"type": "string", "required": False, "description": "pages"},
            "context": {"type": "string", "required": False, "description": "chat"},
            "output_dir": {"type": "string", "required": False, "description": "dir"},
            "auto": {"type": "string", "required": False, "description": "marker"},
        },
    )


class _ExplodingExtractor:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, **kwargs):
        self.calls += 1
        raise AssertionError("cap-summary must never call the shared extractor")


def _wire(monkeypatch, entry: CapabilityEntry):
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    calls: list = []

    async def fake_active(**kw):
        return view

    def fake_match(message, facts, v):
        return MatchResult(state=MATCH_HIT, capability_id=entry.capability_id,
                           matched_literal="总结内容")

    class _NoSelect:
        async def select(self, query, candidates, *, entries_by_id, facts=None):
            calls.append(query)
            raise AssertionError("the selector must never run on a HIT lane")

    monkeypatch.setattr(
        "core.application.chat.intent_funnel.registry.active_view", fake_active)
    monkeypatch.setattr(matcher_mod, "match", fake_match)
    monkeypatch.setattr(cap_router_mod, "default_selector", lambda: _NoSelect())
    return calls, view


def _deps(extractor=None):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=None, argument_extractor=extractor)


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_auto_is_certified_without_any_extractor_call(monkeypatch):
    entry = _summary_entry()
    calls, view = _wire(monkeypatch, entry)
    extractor = _ExplodingExtractor()
    req = TurnRequirements()
    out = await _route(_ctx("总结前面几条的聊天内容"), _deps(extractor), req)

    assert out is not req
    act = out.requested_action
    assert act["tool"] == "summary_gen"
    assert act["args"] == {"auto": "1"}
    assert act["funnel_stage"] == "tool_intent"
    assert act["funnel_registry_version"] == view.fingerprint
    assert extractor.calls == 0
    assert not calls


async def test_the_binder_injects_the_open_document(monkeypatch):
    entry = _summary_entry()
    _wire(monkeypatch, entry)
    out = await _route(_ctx("总结第3页", viewer=_Viewer("doc-1")),
                       _deps(_ExplodingExtractor()), TurnRequirements())
    assert out is not None
    assert out.requested_action["args"] == {"auto": "1", "asset_id": "doc-1"}


async def test_a_literal_path_is_certified_deterministically(monkeypatch):
    entry = _summary_entry()
    _wire(monkeypatch, entry)
    out = await _route(_ctx("总结 ./reports/q3.md"),
                       _deps(_ExplodingExtractor()), TurnRequirements())
    assert out is not None
    assert out.requested_action["args"] == {"path": "./reports/q3.md"}
