"""cap_handler — ReadFileHandler + the runtime drive-prefix alignment.

Three layers are pinned here:

* the HANDLER contract (unit): ``cap-read-file``'s single required slot is
  ``path``, resolved DETERMINISTICALLY — no model call, no Qwen. ``path`` may
  come ONLY from a literal path token in the sentence: a drive path
  (``My Drive/...`` / ``我的云盘/...``, matched case-insensitively, copied
  VERBATIM so casing/spaces survive) or a workspace token (``./`` prefix kept,
  multi-level dirs kept, CJK filenames never transliterated). A pronoun referent,
  a vague sentence, no literal path, or a blank query fails closed (``None``).
  ``max_chars`` is never emitted, and no asset field is ever injected as ``path``.
* the RUNTIME alignment: ``fs_tools._DRIVE_PATH_RE`` matches the drive prefix
  case-insensitively while preserving the caller's literal (comparison is
  case-insensitive, the value is exact) — ``my drive/x`` / ``My Drive/x`` /
  ``MY DRIVE/x`` all take the Drive branch, no workspace fallback.
* the FUNNEL wiring: ``handler_for("cap-read-file")`` is the registered handler,
  and the handler short-circuits the generic acquisition chain — its draft flows
  through the SAME ``_certify`` -> Binder -> ``tool_intent`` handoff, a ``None``
  draft exits to the Agent with ``ACQUISITION_MISSING``, and a forged generic
  MODEL value never reaches ``path``.

The gold spans are the 28 ``cap-read-file`` cases of the frozen Formal-500
argument benchmark (``logs/_argbench/dataset_formal500.jsonl``), so this suite is
a direct reconciliation against that evidence.
"""
from __future__ import annotations

import hashlib
import logging
import re
import types
import uuid
from pathlib import Path

import pytest
from agent.tools import fs_tools
from agent.tools.fs_tools import _DRIVE_PATH_RE, read_file_tool
from core.application.chat.intent_funnel import matcher as matcher_mod
from core.application.chat.intent_funnel import tool_intent as tool_intent_mod
from core.application.chat.intent_funnel.argument_acquisition import AcquisitionInputs
from core.application.chat.intent_funnel.cap_handler import (
    HANDLERS,
    ReadFileHandler,
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
from core.application.drive_service import READY
from core.config import settings
from core.infrastructure.storage import object_key

from tests._drive_fakes import make_drive

FUNNEL_LOGGER = "core.application.chat.intent_funnel.funnel"

GOLD_PATH = "logs/_argbench/dataset_formal500.jsonl"

# The frozen arg benchmark lives in the untracked ``logs/`` scratch tree; the
# byte-for-byte reconciliation below is a local-only guard, skipped when absent.
_GOLD_PRESENT = Path(GOLD_PATH).exists()


# ── Layer 0: the Formal-500 gold spans (28 cases) ────────────────────────────────
# (message, expected draft) — expected is ``{"path": <verbatim literal>}`` or
# ``None`` (gold ``{}`` = fail-closed, the Agent owns the turn). Auto-verified
# against the frozen dataset in ``test_gold_matches_the_frozen_dataset`` below.

GOLD_CASES: list[tuple[str, dict[str, str] | None]] = [
    ("读取 My Drive/如何炒西红柿_v1.pdf", {"path": "My Drive/如何炒西红柿_v1.pdf"}),
    ("read the file My Drive/Agent Harness Engineering A Survey.pdf",
     {"path": "My Drive/Agent Harness Engineering A Survey.pdf"}),
    ("读取这个文档", None),
    ("读取 my drive/如何炒西红柿_v1.pdf", {"path": "my drive/如何炒西红柿_v1.pdf"}),
    ("帮我读一下文件", None),
    ("打开 My Drive/如何炒西红柿_v1.pdf", {"path": "My Drive/如何炒西红柿_v1.pdf"}),
    ("read My Drive/Agent Harness Engineering A Survey.pdf",
     {"path": "My Drive/Agent Harness Engineering A Survey.pdf"}),
    ("把 config.json 读出来", {"path": "config.json"}),
    ("open the file src/utils/helper.py", {"path": "src/utils/helper.py"}),
    ("读一下 学习笔记/第九周.md", {"path": "学习笔记/第九周.md"}),
    ("read my drive/annual report 2025.pdf", {"path": "my drive/annual report 2025.pdf"}),
    ("读一下文档", None),
    ("open notes.txt and show me the first 500 chars", {"path": "notes.txt"}),
    ("看一眼 报告.docx", {"path": "报告.docx"}),
    ("read this file", None),
    ("打开 学习笔记/第三周.md", {"path": "学习笔记/第三周.md"}),
    ("读取 configs/prod.yaml", {"path": "configs/prod.yaml"}),
    ("把 README.md 读出来", {"path": "README.md"}),
    ("看一下 报告/2026Q1.pdf", {"path": "报告/2026Q1.pdf"}),
    ("读一下文件", None),
    ("open docs/architecture.md", {"path": "docs/architecture.md"}),
    ("read src/main/java/App.java", {"path": "src/main/java/App.java"}),
    ("read the file scripts/deploy.sh", {"path": "scripts/deploy.sh"}),
    ("open my drive/annual report 2025.pdf", {"path": "my drive/annual report 2025.pdf"}),
    ("read ./data/train.csv", {"path": "./data/train.csv"}),
    ("open the notebook experiments/ablation.ipynb", {"path": "experiments/ablation.ipynb"}),
    ("read that file we talked about", None),
    ("read the log at var/log/app.log", {"path": "var/log/app.log"}),
]


@pytest.mark.parametrize("message,expected", GOLD_CASES)
async def test_formal500_gold_spans(message, expected):
    handler = ReadFileHandler()
    assert await handler.acquire(query=message, facts=None) == expected


@pytest.mark.skipif(
    not _GOLD_PRESENT,
    reason="frozen arg benchmark logs/_argbench/dataset_formal500.jsonl not present",
)
def test_gold_matches_the_frozen_dataset():
    """The embedded gold mirrors ``dataset_formal500.jsonl`` byte for byte — a
    literal reconciliation against the benchmark rather than a paraphrase."""
    import json

    rows = [
        json.loads(line)
        for line in Path(GOLD_PATH).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actual = {
        (r.get("message") or r.get("query")): (r.get("gold_arguments") or None)
        for r in rows
        if r.get("capability_id") == "cap-read-file"
    }
    embedded = {message: expected for message, expected in GOLD_CASES}
    assert actual == embedded
    assert len(actual) == 28


# ── Layer 1a: drive prefix — case-insensitive match, verbatim value ──────────────


@pytest.mark.parametrize("message,expected", [
    ("read my drive/foo.txt", "my drive/foo.txt"),         # lowercase preserved
    ("read My Drive/foo.txt", "My Drive/foo.txt"),
    ("read MY DRIVE/foo.txt", "MY DRIVE/foo.txt"),         # uppercase preserved
    ("读取 我的云盘/笔记.md", "我的云盘/笔记.md"),             # Chinese drive token
    ("open My Drive/Agent Harness Engineering A Survey.pdf",
     "My Drive/Agent Harness Engineering A Survey.pdf"),    # spaces in filename kept
])
async def test_drive_prefix_is_verbatim(message, expected):
    handler = ReadFileHandler()
    assert await handler.acquire(query=message, facts=None) == {"path": expected}


# ── Layer 1b: workspace paths — ./, single file, multi-level dirs ────────────────


@pytest.mark.parametrize("message,expected", [
    ("把 config.json 读出来", "config.json"),               # bare file + extension
    ("read ./data/train.csv", "./data/train.csv"),          # ./ prefix NOT stripped
    ("open the file src/utils/helper.py", "src/utils/helper.py"),   # multi-level
    ("read src/main/java/App.java", "src/main/java/App.java"),
    ("读取 configs/prod.yaml", "configs/prod.yaml"),
    ("读一下 学习笔记/第九周.md", "学习笔记/第九周.md"),        # CJK directory
])
async def test_workspace_path_verbatim(message, expected):
    handler = ReadFileHandler()
    assert await handler.acquire(query=message, facts=None) == {"path": expected}


# ── Layer 1c: no translation — the path literal crosses languages untouched ─────


@pytest.mark.parametrize("message,expected", [
    ("看一眼 报告.docx", "报告.docx"),                        # never report.docx
    ("把 README.md 读出来", "README.md"),                     # ASCII kept as-is
    ("read 学习笔记/第九周.md", "学习笔记/第九周.md"),          # CJK path stays CJK
])
async def test_path_is_never_translated(message, expected):
    handler = ReadFileHandler()
    assert await handler.acquire(query=message, facts=None) == {"path": expected}


# ── Layer 1d: strict fail-closed ─────────────────────────────────────────────────


@pytest.mark.parametrize("message", [
    "读取这个文档",              # pronoun referent
    "read this file",
    "帮我读一下文件",            # vague spoken sentence
    "读一下文档",
    "读一下文件",
    "read that file we talked about",
])
async def test_pronoun_or_vague_sentence_fails_closed(message):
    handler = ReadFileHandler()
    assert await handler.acquire(query=message, facts=None) is None


@pytest.mark.parametrize("blank", ["", "   ", "\n\t", None])
async def test_blank_query_fails_closed(blank):
    handler = ReadFileHandler()
    assert await handler.acquire(query=blank, facts=None) is None


# ── Layer 1e: only ``path`` is emitted — never ``max_chars`` ─────────────────────


async def test_only_the_path_slot_is_ever_emitted():
    handler = ReadFileHandler()
    draft = await handler.acquire(query="读取 My Drive/报告.pdf", facts=types.SimpleNamespace())
    assert set(draft) == {"path"}          # no max_chars, no other slot


async def test_char_count_in_the_sentence_never_becomes_max_chars():
    # "first 500 chars" is framing; the handler emits the path only.
    handler = ReadFileHandler()
    draft = await handler.acquire(
        query="open notes.txt and show me the first 500 chars", facts=None)
    assert draft == {"path": "notes.txt"}
    assert "max_chars" not in draft


# ── Layer 1f: asset context is NEVER injected as ``path`` ────────────────────────


def _facts_viewer(asset_id):
    return types.SimpleNamespace(
        has_viewer=True, viewer_asset_id=str(asset_id), viewer_current_page=1,
        viewer_page_from=None, viewer_page_to=None, has_viewer_selection=False,
        has_attachment=True, attachment_asset_id=str(asset_id),
        path_asset_id=str(asset_id), has_turn_context=True,
    )


async def test_viewer_asset_id_is_never_injected_as_path():
    handler = ReadFileHandler()
    asset = uuid.uuid4()
    # a real sentence path wins — the asset id is nowhere in the draft
    draft = await handler.acquire(query="读取 config.json", facts=_facts_viewer(asset))
    assert draft == {"path": "config.json"}
    assert str(asset) not in draft.values()


async def test_asset_context_does_not_rescue_a_pathless_sentence():
    # a pronoun sentence with an asset present must STILL fail closed: facts are
    # ignored on purpose, so no viewer/attachment id can leak in as a path.
    handler = ReadFileHandler()
    asset = uuid.uuid4()
    assert await handler.acquire(query="读取这个文档", facts=_facts_viewer(asset)) is None


# ── Layer 2: the roster wiring ───────────────────────────────────────────────────


def test_roster_wires_cap_read_file():
    handler = handler_for("cap-read-file")
    assert isinstance(handler, ReadFileHandler)
    assert handler.capability_id == "cap-read-file"
    assert HANDLERS["cap-read-file"] is handler


def test_cap_read_file_is_no_longer_on_the_generic_chain():
    assert handler_for("cap-read-file") is not None
    for cid in ("cap-open-pdf", ""):
        assert handler_for(cid) is None


# ── Layer 3: runtime drive dispatch is case-insensitive, literal preserved ──────


@pytest.mark.parametrize("raw,remainder", [
    ("My Drive/foo.txt", "foo.txt"),
    ("my drive/foo.txt", "foo.txt"),
    ("MY DRIVE/foo.txt", "foo.txt"),
    ("My Drive/Agent Harness Engineering A Survey.pdf",
     "Agent Harness Engineering A Survey.pdf"),
    ("我的云盘/笔记.md", "笔记.md"),
])
def test_drive_path_re_matches_case_insensitively_and_keeps_the_literal(raw, remainder):
    match = _DRIVE_PATH_RE.match(raw)
    assert match is not None
    assert match.group(1) == remainder          # value is exact, only compare is fuzzy


def test_drive_path_re_rejects_a_workspace_path():
    assert _DRIVE_PATH_RE.match("src/main.py") is None
    assert _DRIVE_PATH_RE.match("./data/train.csv") is None


async def _seed_note(svc, owner, name="scope.md"):
    data = b"# Scope\n\nline one\n"
    digest = hashlib.sha256(data).hexdigest()
    await svc.storage.put(object_key(digest), data)
    await svc.objects.upsert_and_increment(digest, len(data), object_key(digest),
                                           "text/markdown")
    return await svc.assets.create(owner, name, mime_type="text/markdown",
                                   size=len(data), object_sha256=digest,
                                   file_status=READY, rag_status="NOT_STARTED")


@pytest.mark.parametrize("raw", ["my drive/scope.md", "My Drive/scope.md", "MY DRIVE/scope.md"])
async def test_all_casing_variants_take_the_drive_branch(tmp_path, monkeypatch, raw):
    svc = make_drive(tmp_path)
    owner = uuid.uuid4()
    monkeypatch.setattr(fs_tools, "get_request_user_id", lambda: owner)
    await _seed_note(svc, owner)
    (tmp_path / "scope.md").write_text("WORKSPACE LOOKALIKE", encoding="utf-8")

    tool = read_file_tool(tmp_path, svc)
    out = await tool.execute({"path": raw}, None)
    # The Drive plane wins (case-insensitive prefix) and reads the REAL asset —
    # never the identically-named workspace file, never lowercased onto disk.
    assert out == "# Scope\n\nline one\n"


# ── Layer 4: the funnel wiring (real cascade, faked Matcher HIT) ─────────────────


def _ctx(message: str):
    return types.SimpleNamespace(
        body=types.SimpleNamespace(message=message, attach=None, viewer=None),
        owned_asset_id=None, path_asset_id=None, research_turn=False,
        effective_handoff=None, session_id="s1", history=[],
    )


def _read_file_entry() -> CapabilityEntry:
    q = "读取工作区里的文件"
    return CapabilityEntry(
        capability_id="cap-read-file", tool_binding="read_file",
        description="Read an existing file from the workspace or My Drive by its path.",
        standard_queries=(QueryRecord(id="cap-read-file-q1", query=q,
                                      language=derive_language(q)),),
        parameters={
            "path": {"type": "string", "required": True,
                     "description": "the file path, copied from the sentence"},
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
                           matched_literal="读取工作区里的文件")

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


async def test_read_file_turn_certifies_handler_payload(monkeypatch):
    entry = _read_file_entry()
    rec, view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("读取 config.json"), _deps(None), req)

    assert out is not req                              # a CERTIFIED turn
    act = out.requested_action
    assert act["tool"] == "read_file"
    assert act["capability_id"] == "cap-read-file"
    assert act["args"] == {"path": "config.json"}      # sentence literal, verbatim
    assert act["funnel_stage"] == "tool_intent"        # the EXISTING handoff shape
    assert act["funnel_registry_version"] == view.fingerprint
    assert not rec.legacy                              # no legacy hop


async def test_read_file_turn_without_a_path_exits_to_agent(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=FUNNEL_LOGGER)
    # NB: "帮我读一下文件" (not a deictic "这个文档" sentence) — the latter is vetoed
    # at the funnel door by ``referenced_input_absent`` before the handler ever runs.
    entry = _read_file_entry()
    rec, _view = _wire(monkeypatch, entry)
    req = TurnRequirements()
    out = await _route(_ctx("帮我读一下文件"), _deps(None), req)

    assert out is req
    lines = [r.getMessage() for r in caplog.records
             if r.name == FUNNEL_LOGGER and "funnel_trace" in r.getMessage()]
    assert len(lines) == 1, lines
    assert re.search(r"fallback_reason=(\S+)", lines[0]).group(1) == REASON_ACQUISITION_MISSING
    assert not rec.legacy                              # no Qwen, no legacy extraction


async def test_handler_bypasses_a_forged_generic_chain(monkeypatch):
    """A forged MODEL value never reaches the draft: the handler owns ``path``."""
    entry = _read_file_entry()
    rec, _view = _wire(monkeypatch, entry)
    poisoned = {"cap-read-file": AcquisitionInputs(
        model_values={"path": "HALLUCINATED/PATH.pdf"})}
    req = TurnRequirements()
    out = await _route(_ctx("读取 config.json"), _deps(lambda cid: poisoned.get(cid)), req)

    assert out is not req
    assert out.requested_action["args"] == {"path": "config.json"}
    assert not rec.legacy


async def test_handler_returning_none_is_acquisition_missing(monkeypatch):
    """The orchestrator branch: a handler with no legal draft (``None``) exits
    the turn to the Agent with ``ACQUISITION_MISSING`` — fail-closed."""
    from core.application.chat.intent_funnel import orchestrator as orch
    from core.application.chat.intent_funnel.cap_handler import roster

    class _NoneHandler:
        capability_id = "cap-read-file"

        async def acquire(self, *, query, facts):
            return None

    monkeypatch.setitem(roster.HANDLERS, "cap-read-file", _NoneHandler())
    entry = _read_file_entry()
    view = RegistryLiveView(fingerprint=content_fingerprint([entry]), entries=(entry,))
    trace: dict = {}
    out = await orch._acquisition_hop(
        TurnRequirements(), types.SimpleNamespace(), entry,
        query="读取 config.json", facts=None, view=view, trace=trace, capture=None)

    assert out is None
    assert trace["fallback"] == REASON_ACQUISITION_MISSING
