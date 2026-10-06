"""cap_handler — CreateFolderHandler and the create_folder WRITE hardening.

Three layers are pinned here:

* the HANDLER contract (unit): ``cap-create-folder``'s single required slot is
  ``name``, resolved DETERMINISTICALLY — no model call, no Qwen. ``name`` comes
  ONLY from a literal folder name in the sentence, by a priority ladder: a
  quoted span; an explicit naming lead (``叫``/``命名为``/``named``/``called``)
  whose multi-token value is NEVER cut at whitespace (``2026-Q1 报表`` stays
  whole); the English ``for the X`` clause; the Chinese ``X 文件夹`` modifier.
  The value is copied VERBATIM (casing/spaces survive). A generic sentence, a
  verb-only request, a blank query, or a slash-bearing candidate all fail closed
  (``None``). ``parent_path`` is never emitted, and no asset field is injected.
* the TOOL hardening: ``create_folder`` explicitly declares ``{WRITE}`` so the
  sandbox's ASK/DENY gate no longer depends on the inference that ``parent_path``
  contains "path".
* the FUNNEL wiring: ``handler_for("cap-create-folder")`` is the registered
  handler, and the handler short-circuits the generic acquisition chain — its
  draft flows through the SAME ``_certify`` -> Binder -> ``tool_intent`` handoff,
  a ``None`` draft exits to the Agent with ``ACQUISITION_MISSING``, and a forged
  generic MODEL value never reaches ``name``.

The gold spans are the 20 ``cap-create-folder`` cases of the frozen Formal-500
argument benchmark (``logs/_argbench/dataset_formal500.jsonl``), so this suite is
a direct reconciliation against that evidence.
"""
from __future__ import annotations

import logging
import re
import types

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


# ── Layer 0: the Formal-500 gold spans (20 cases) ────────────────────────────────
# (message, expected draft) — expected is ``{"name": <verbatim literal>}`` or
# ``None`` (gold ``{}`` = fail-closed, the Agent owns the turn). Auto-verified
# against the frozen dataset in ``test_gold_matches_the_frozen_dataset`` below.

GOLD_CASES: list[tuple[str, dict[str, str] | None]] = [
    ('帮我在网盘里新建一个"会议纪要2026"文件夹', {"name": "会议纪要2026"}),
    ("给我建个叫 学习笔记 的文件夹", {"name": "学习笔记"}),
    ('create a folder named "Release Notes"', {"name": "Release Notes"}),
    ("create a new folder called Project Phoenix", {"name": "Project Phoenix"}),
    ("帮我建个文件夹", None),
    ('新建文件夹"RAG-2.0"', {"name": "RAG-2.0"}),
    ('新建文件夹 "2026 年度计划"', {"name": "2026 年度计划"}),
    ("create a folder called Project Atlas", {"name": "Project Atlas"}),
    ("帮我建个 归档 文件夹", {"name": "归档"}),
    ('make a new folder named "release-v2.0"', {"name": "release-v2.0"}),
    ("创建一个叫 2026-Q1 报表 的文件夹", {"name": "2026-Q1 报表"}),
    ('create folder "Meeting Notes"', {"name": "Meeting Notes"}),
    ("新建一个文件夹", None),
    ("make a new folder", None),
    ("在 项目 下面建个叫 archive 的子文件夹", {"name": "archive"}),
    ("建一个叫 2026Q2 的文件夹", {"name": "2026Q2"}),
    ('create a folder named "Design Docs"', {"name": "Design Docs"}),
    ("make a new folder for the Q3 backlog", {"name": "Q3 backlog"}),
    ('create a folder called "invoices-2026"', {"name": "invoices-2026"}),
    ("new folder please", None),
]


@pytest.mark.parametrize("message,expected", GOLD_CASES)
async def test_formal500_gold_spans(message, expected):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=message, facts=None) == expected


def test_gold_matches_the_frozen_dataset():
    """The embedded gold mirrors ``dataset_formal500.jsonl`` byte for byte — a
    literal reconciliation against the benchmark rather than a paraphrase."""
    import json
    from pathlib import Path

    rows = [
        json.loads(line)
        for line in Path(GOLD_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual = {
        (r.get("message") or r.get("query")): (r.get("gold_arguments") or None)
        for r in rows
        if r.get("capability_id") == "cap-create-folder"
    }
    embedded = {message: expected for message, expected in GOLD_CASES}
    assert actual == embedded
    assert len(actual) == 20


# ── Layer 1a: quoted spans — ASCII and full-width pairs, quotes dropped ──────────


@pytest.mark.parametrize("message,expected", [
    ('create folder "alpha"', "alpha"),
    ("create folder 'alpha'", "alpha"),
    ("新建文件夹“项目”", "项目"),
    ("建立文件夹「delta」", "delta"),
    ("建立文件夹『delta』", "delta"),
    ('create a folder named "has space"', "has space"),
    ('新建文件夹"2026 年度计划"', "2026 年度计划"),        # inner space kept
    ('create a folder named "RAG-2.0"', "RAG-2.0"),        # punctuation kept
])
async def test_quoted_span_is_extracted_verbatim(message, expected):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=message, facts=None) == {"name": expected}


# ── Layer 1b: explicit naming leads — multi-token names are never truncated ───────


@pytest.mark.parametrize("message,expected", [
    ("给我建个叫 学习笔记 的文件夹", "学习笔记"),
    ("建一个叫 2026Q2 的文件夹", "2026Q2"),
    ("命名为 Project Mercury", "Project Mercury"),
    ("create a new folder called Project Phoenix", "Project Phoenix"),   # two tokens
    ("create a folder named Release Notes", "Release Notes"),            # no quotes
    ("创建一个叫 2026-Q1 报表 的文件夹", "2026-Q1 报表"),                  # NOT cut at space
    ("在 项目 下面建个叫 archive 的子文件夹", "archive"),
])
async def test_naming_lead_keeps_the_whole_name(message, expected):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=message, facts=None) == {"name": expected}


# ── Layer 1c: the English "for the X" clause ─────────────────────────────────────


@pytest.mark.parametrize("message,expected", [
    ("make a new folder for the Q3 backlog", "Q3 backlog"),
    ("create a folder for the 2026 roadmap", "2026 roadmap"),
])
async def test_for_the_clause(message, expected):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=message, facts=None) == {"name": expected}


# ── Layer 1d: the Chinese "X 文件夹" modifier ─────────────────────────────────────


@pytest.mark.parametrize("message,expected", [
    ("帮我建个 归档 文件夹", "归档"),
    ("新建一个 草稿 文件夹", "草稿"),
])
async def test_zh_folder_modifier(message, expected):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=message, facts=None) == {"name": expected}


# ── Layer 1e: slash defense — a path is not a folder name ⇒ fail closed ──────────


@pytest.mark.parametrize("message", [
    'create a folder named "a/b"',
    '新建文件夹"a\\b"',
    "建一个叫 项目/归档 的文件夹",
    'create a folder named "a／b"',        # full-width slash
])
async def test_slash_bearing_name_fails_closed(message):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=message, facts=None) is None


# ── Layer 1f: strict fail-closed ─────────────────────────────────────────────────


@pytest.mark.parametrize("message", [
    "帮我建个文件夹",          # verb + generic noun, no name
    "新建一个文件夹",
    "make a new folder",
    "new folder please",
    "create a folder",         # nothing but intent
])
async def test_generic_or_verb_only_sentence_fails_closed(message):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=message, facts=None) is None


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_fails_closed(blank):
    handler = CreateFolderHandler()
    assert await handler.acquire(query=blank, facts=None) is None


# ── Layer 1g: only ``name`` is emitted — never ``parent_path`` ───────────────────


async def test_only_the_name_slot_is_ever_emitted():
    handler = CreateFolderHandler()
    draft = await handler.acquire(query='create a folder named "x"', facts=types.SimpleNamespace())
    assert set(draft) == {"name"}          # no parent_path, no other slot


def _facts_viewer(asset_id):
    return types.SimpleNamespace(
        has_viewer=True, viewer_asset_id=str(asset_id), viewer_current_page=1,
        viewer_page_from=None, viewer_page_to=None, has_viewer_selection=False,
        has_attachment=True, attachment_asset_id=str(asset_id),
        path_asset_id=str(asset_id), has_turn_context=True,
    )


async def test_asset_context_never_rescues_a_nameless_sentence():
    # a generic sentence with an asset present must STILL fail closed: facts are
    # ignored on purpose, so no viewer/attachment id can leak in as a name.
    handler = CreateFolderHandler()
    assert await handler.acquire(query="帮我建个文件夹", facts=_facts_viewer("abc")) is None


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


def _deps(provider):
    return types.SimpleNamespace(
        session_factory=None, embedder=lambda: object(), llm=object(),
        acquisition_inputs=provider)


async def _route(ctx, deps, req):
    from core.application.chat.intent_funnel import funnel
    return await funnel.route(ctx, deps=deps, requirements=req)


async def test_create_folder_turn_certifies_handler_payload(monkeypatch):
    entry = _create_folder_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx('create a folder named "Meeting Notes"'), _deps(None), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "create_folder"
    assert act["capability_id"] == "cap-create-folder"
    assert act["args"] == {"name": "Meeting Notes"}    # sentence literal, verbatim
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_create_folder_turn_without_a_name_exits_to_agent(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    entry = _create_folder_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("帮我建个文件夹"), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_ACQUISITION_MISSING
    assert not rec.legacy                              # no Qwen, no legacy extraction


async def test_handler_bypasses_a_forged_generic_chain(monkeypatch):
    """A forged MODEL value never reaches the draft: the handler owns ``name``."""
    entry = _create_folder_entry()
    rec, _view = _wire(monkeypatch, entry)
    poisoned = {"cap-create-folder": AcquisitionInputs(
        model_values={"name": "HALLUCINATED"})}
    req = TurnRequirements()
    out = await _route(_ctx("给我建个叫 学习笔记 的文件夹"),
                       _deps(lambda cid: poisoned.get(cid)), req)

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
