"""cap_handler — CreateFolderHandler and the create_folder WRITE hardening.

Layers pinned here:

* the HANDLER contract (unit): ``cap-create-folder``'s single required slot is
  ``name``. Per the acquisition contract ``name`` is a MODEL-owned natural-
  language slot and the handler runs NO extraction rule — the quoted-span /
  naming-lead / ``for the X`` / ``X 文件夹`` regex ladder had no requirement basis
  and is removed. ``acquire()`` returns the EMPTY ``{}`` draft for ANY non-blank
  sentence and ``None`` only for a blank one; ``slot_plan()`` authorizes the model
  for ``name`` ONLY. ``parent_path`` is never emitted and never asked of the model
  (the drive root is the tool default); no asset field is injected.
* the TOOL hardening: ``create_folder`` explicitly declares ``{WRITE}`` so the
  sandbox's ASK/DENY gate no longer depends on the ``parent_path`` heuristic.
* the FUNNEL wiring: ``handler_for("cap-create-folder")`` owns the draft and
  short-circuits the generic chain; the empty draft is completed by the shared
  extractor (``deps.argument_extractor``) for ``name`` only, then the SAME
  ``_certify`` -> Binder -> ``tool_intent`` gate runs. A name no source filled
  exits ``BIND_MISSING`` (the handler no longer bails with ``ACQUISITION_MISSING``).
* the model-assisted round-trip lives in ``test_orchestrator_slot_extraction``.

The 20 ``cap-create-folder`` cases of the frozen Formal-500 argument benchmark
(``logs/_argbench/dataset_formal500.jsonl``) are reused as a REPRESENTATIVE
corpus: every one must now yield the EMPTY handler draft. The gold is unchanged.
"""
from __future__ import annotations

import logging
import re
import types
from pathlib import Path

import pytest
from agent.tools.definition import classify_permissions
from agent.tools.tool_permissions import ToolPermission
from api.tools import create_folder_tool
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    CreateFolderHandler,
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
    return [r["query"] for r in rows if r.get("capability_id") == "cap-create-folder"]


GOLD_MESSAGES = _gold_messages()


# ── Layer 0: the handler runs NO rule — the empty draft for every gold case ───────


@pytest.mark.skipif(not _GOLD_PRESENT, reason="benchmark not present")
@pytest.mark.parametrize("message", GOLD_MESSAGES)
async def test_handler_yields_the_empty_draft_for_every_gold_case(message):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=message, facts=None) == {}


@pytest.mark.skipif(not _GOLD_PRESENT, reason="benchmark not present")
def test_the_representative_corpus_is_the_20_gold_cases():
    assert len(GOLD_MESSAGES) == 20


# ── Layer 1: the handler contract (empty draft on any input; None on blank) ───────


@pytest.mark.parametrize("message", [
    '帮我在网盘里新建一个"会议纪要2026"文件夹',        # quoted — no rule
    "创建一个叫 2026-Q1 报表 的文件夹",               # naming lead — no rule
    "make a new folder for the Q3 backlog",           # for-the clause — no rule
    "帮我建个 归档 文件夹",                            # X 文件夹 modifier — no rule
    "帮我建个文件夹",                                  # no name at all
])
async def test_acquire_always_returns_the_empty_draft(message):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=message, facts=None) == {}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_fails_closed(blank):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=blank, facts=None) is None


async def test_slot_plan_authorizes_name_only():
    handler = CreateFolderHandler()
    msg = "帮我建个文件夹"
    draft = await handler.acquire(query=msg, facts=None)
    plan = handler.slot_plan(query=msg, facts=None, draft=draft)
    assert plan.model_slots == ("name",)                 # name only — never a root/path
    assert plan.default_slots == ()


async def test_asset_context_never_rescues_a_nameless_sentence():
    handler = CreateFolderHandler()
    facts = types.SimpleNamespace(
        has_viewer=True, viewer_asset_id="abc", viewer_current_page=1,
        viewer_page_from=None, viewer_page_to=None, has_viewer_selection=False,
        has_attachment=True, attachment_asset_id="abc",
        path_asset_id="abc", has_turn_context=True)
    assert await handler.acquire(query="帮我建个文件夹", facts=facts) == {}


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_create_folder():
    handler = handler_for("cap-create-folder")
    assert isinstance(handler, CreateFolderHandler)
    assert handler.capability_id == "cap-create-folder"
    assert HANDLERS["cap-create-folder"] is handler


def test_cap_create_folder_is_no_longer_on_the_generic_chain():
    assert handler_for("cap-create-folder") is not None
    for cid in ("cap-open-pdf", ""):
        assert handler_for(cid) is None


# ── Layer 3: the funnel wiring (real cascade, faked Matcher HIT) ─────────────────


def _ctx(message: str):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _create_folder_entry() -> CapabilityEntry:
    q = "在工作区新建一个文件夹"
    return CapabilityEntry(
        capability_id="cap-create-folder", tool_binding="create_folder",
        description="Create a folder in the user's cloud drive by name.",
        standard_queries=(QueryRecord(id="cap-create-folder-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "name": {"type": "string", "required": True,
                     "description": "the folder name, quotes removed"},
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
                           matched_literal="在工作区新建一个文件夹")

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


def _mock_extractor(returned):
    async def _extract(*, query, entry, model_slots, bundle, prompt):
        return returned, bundle.source
    return _extract


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_create_folder_turn_certifies_the_model_name(monkeypatch):
    entry = _create_folder_entry()
    rec, view = _wire(monkeypatch, entry)
    extractor = _mock_extractor({"name": "会议纪要2026", "parent_path": "/etc"})
    req = TurnRequirements()
    out = await _route(_ctx('帮我在网盘里新建一个"会议纪要2026"文件夹'),
                       _deps(None, extractor), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "create_folder"
    assert act["capability_id"] == "cap-create-folder"
    assert act["args"] == {"name": "会议纪要2026"}      # unauthorized parent_path dropped
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_create_folder_turn_without_a_name_exits_bind_missing(monkeypatch, caplog):
    """The empty draft is completed by the extractor; a name no source filled
    leaves the required slot MISSING, so the SAME Binder gate exits BIND_MISSING."""
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _create_folder_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("帮我建个文件夹"), _deps(None, _mock_extractor({})), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_BIND_MISSING
    assert not rec.legacy                              # no legacy extraction


async def test_create_folder_without_extractor_exits_bind_missing(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _create_folder_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("给我建个叫 学习笔记 的文件夹"), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_BIND_MISSING


async def test_handler_bypasses_a_forged_generic_chain(monkeypatch):
    """A forged MODEL value in the generic chain never reaches the draft."""
    entry = _create_folder_entry()
    rec, _view = _wire(monkeypatch, entry)
    poisoned = {"cap-create-folder": AcquisitionInputs(
        model_values={"name": "HALLUCINATED"})}
    req = TurnRequirements()
    out = await _route(_ctx("给我建个叫 学习笔记 的文件夹"),
                       _deps(lambda cid: poisoned.get(cid),
                             _mock_extractor({"name": "学习笔记"})), req)

    assert out is not req
    assert out.requested_action["args"] == {"name": "学习笔记"}
    assert not rec.legacy


async def test_handler_returning_none_is_acquisition_missing(monkeypatch):
    """The orchestrator branch: a handler with no legal draft (``None``) exits
    the turn to the Agent with ``ACQUISITION_MISSING`` — fail-closed."""
    from core.application.chat.intent_funnel import orchestrator as orch
    from core.application.chat.intent_funnel.cap_handler import roster

    class _NoneHandler:
        capability_id = "cap-create-folder"

        async def acquire(self, *, query, facts):
            return None

    monkeypatch.setitem(roster.HANDLERS, "cap-create-folder", _NoneHandler())
    entry = _create_folder_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    trace: dict = {}
    out = await orch._acquisition_hop(
        TurnRequirements(), types.SimpleNamespace(), entry,
        query='create a folder named "x"', facts=None, view=view, trace=trace, capture=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING


# ── Layer 4: create_folder declares WRITE explicitly ─────────────────────────────


class _CapRuntime:
    def __init__(self) -> None:
        self.tools: dict = {}

    def register(self, defn) -> None:
        self.tools[defn.name] = defn


def test_create_folder_declares_write_explicitly():
    runtime = _CapRuntime()
    ctx = types.SimpleNamespace(resolve=lambda key: None)
    create_folder_tool.register(runtime, ctx, None)
    defn = runtime.tools["create_folder"]

    # The declaration is EXPLICIT (not None): the sandbox gate no longer relies on
    # `parent_path` happening to contain "path".
    assert defn.permission is not None
    assert ToolPermission.WRITE in defn.permission
    assert classify_permissions(defn) == frozenset({ToolPermission.WRITE})
