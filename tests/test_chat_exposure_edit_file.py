"""Chat-plane exposure hiding for edit_file / cap-edit-file (ruling).

Two independent gates, one per surface:

1. TOOL gate — the CHAT-process composition passes ``exclude`` into
   :func:`register_fs_tools`; the unregistered tool then disappears from EVERY
   model-visible surface at once: prompt catalog (render_index), tool_search,
   gateway.mount, the tools array. No kernel/sandbox/permission/destructive change.
2. FUNNEL gate — ``settings.chat_funnel_hidden_capabilities`` drops the capability
   from :func:`chat_plane_candidate` (one predicate, four consumers): no Matcher
   exact hit, no Candidate Card, no certification. The Registry ROW stays
   enabled+active — page/PC, worker/research, admin are unaffected.

Default (empty settings) must reproduce the historical behavior exactly.
"""
import pytest
from core.config import settings

from agent.engine.runtime import ToolRuntime
from agent.tools import fs_tools
from agent.tools.fs_tools import register_fs_tools
from agent.tools.tool_gateway import ToolCatalog, ToolGateway
from core.application.chat.intent_funnel.matcher.index import build_index
from core.application.chat.intent_funnel.registry.entry import (
    CapabilityEntry,
    QueryRecord,
    chat_plane_candidate,
)
from core.application.chat.intent_funnel.tool_intent.prompt import build_prompt


class _NoSandbox:  # bash_tool only stores it; no execution in this file
    pass


@pytest.fixture
def chat_runtime(tmp_path):
    """The chat-process posture: edit_file hidden, everything else registered."""
    rt = ToolRuntime()
    register_fs_tools(rt, tmp_path, sandbox=_NoSandbox(), exclude={"edit_file"})
    return rt


@pytest.fixture
def worker_runtime(tmp_path):
    """The worker/research posture: no exclude — original registration."""
    rt = ToolRuntime()
    register_fs_tools(rt, tmp_path, sandbox=_NoSandbox())
    return rt


# ── 1. tool surface: 看不到 / 搜不到 / 挂载不到 ────────────────────────────────

def test_hidden_tool_absent_from_runtime_roster(chat_runtime):
    assert chat_runtime.get("edit_file") is None
    assert chat_runtime.get("read_file") is not None
    assert chat_runtime.get("bash") is not None          # bash exposure unchanged


def test_hidden_tool_invisible_in_prompt_catalog(chat_runtime):
    index = ToolCatalog(chat_runtime).render_index()
    assert "edit_file" not in index
    assert "- read_file:" in index and "- bash:" in index  # 看不到


async def test_hidden_tool_unsearchable(chat_runtime):
    catalog = ToolCatalog(chat_runtime)
    hits = await catalog.search("edit file", limit=10)
    assert "edit_file" not in {h.name for h in hits}       # 搜不到


def test_hidden_tool_unmountable(chat_runtime):
    gateway = ToolGateway(chat_runtime)
    assert gateway.mount("edit_file") is False             # 挂载不到
    assert gateway.mount("bash") is True                   # 在册工具仍可挂载
    names = {s["function"]["name"] for s in gateway.visible_schemas({})}
    assert "edit_file" not in names                        # mounted set ∩ runtime 也不含之
    assert "bash" in names


def test_worker_posture_keeps_edit_file_on_every_surface(worker_runtime):
    assert worker_runtime.get("edit_file") is not None
    index = ToolCatalog(worker_runtime).render_index()
    assert "- edit_file:" in index
    gateway = ToolGateway(worker_runtime)
    assert gateway.mount("edit_file") is True
    # Worker/research behavior is byte-identical to the historical registration.


# ── 2. funnel view: cap-edit-file 不可被路由(Registry 行保持 enabled+active) ──

def _edit_entry(**over):
    base = dict(
        capability_id="cap-edit-file", tool_binding="edit_file",
        description="Edit an existing file",
        standard_queries=(QueryRecord(id="s1", query="把这个文件改一下", language="zh", enabled=True),),
        similar_queries=(QueryRecord(id="q1", query="修改文件内容", language="zh", enabled=True),),
        enabled=True, status="active",
    )
    base.update(over)
    return CapabilityEntry(**base)


def _other_entry():
    return CapabilityEntry(
        capability_id="cap-read-file", tool_binding="read_file",
        description="Read a file",
        standard_queries=(QueryRecord(id="s2", query="读一下这个文件", language="zh", enabled=True),),
        enabled=True, status="active",
    )


class _View:
    def __init__(self, entries, fingerprint):
        self.entries = tuple(entries)
        self.fingerprint = fingerprint


def test_default_empty_settings_keep_everything_routable(monkeypatch):
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", "", raising=False)
    assert chat_plane_candidate(_edit_entry())   # global row NOT disabled by this change
    assert chat_plane_candidate(_other_entry())


def test_hidden_capability_excluded_from_chat_plane_view(monkeypatch):
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", " cap-edit-file ", raising=False)
    e = _edit_entry()
    assert e.enabled and e.status == "active"      # Registry row untouched…
    assert not chat_plane_candidate(e)             # …but invisible to the funnel
    assert chat_plane_candidate(_other_entry())    # neighbors unaffected


def test_hidden_capability_absent_from_matcher_exact_index(monkeypatch):
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", "cap-edit-file", raising=False)
    view = _View([_edit_entry(), _other_entry()], fp_hidden := "fp-hidden-cap-edit")
    exact = build_index(view)
    from core.application.chat.intent_funnel.matcher.normalize import normalize
    assert "cap-edit-file" not in {cid for ids in exact.values() for cid in ids}
    assert set(exact[normalize("读一下这个文件")]) == {"cap-read-file"}


def test_hidden_capability_cannot_be_certified(monkeypatch):
    """entries_by_id is built through the same predicate: a hidden capability
    yields no Candidate Card, and the orchestrator's
    ``entries_by_id.get(jv.capability_id) is None`` guard refuses certification."""
    from core.application.chat.intent_funnel.contract import Candidate
    monkeypatch.setattr(settings, "chat_funnel_hidden_capabilities", "cap-edit-file", raising=False)
    view = _View([_edit_entry(), _other_entry()], "fp-cert")
    entries_by_id = {e.capability_id: e for e in view.entries if chat_plane_candidate(e)}
    assert "cap-edit-file" not in entries_by_id
    # Even when Recall hands back a hit for the hidden capability, the card
    # builder skips it (entry is None) and the orchestrator's get()->None guard
    # refuses certification.
    cand = Candidate(capability_id="cap-edit-file", score=0.95, matched_example="把这个文件改一下",
                     origin="recall", query_kind="standard", language="zh", query_id="s1")
    prompt = build_prompt("把这个文件改一下", [cand], entries_by_id)
    assert "cap-edit-file" not in prompt                   # no card produced
    assert entries_by_id.get("cap-edit-file") is None      # cannot certify


def test_settings_keys_default_posture(monkeypatch):
    """Shipped defaults (single-path ruling): the chat-plane funnel
    hides cap-edit-file out of the box; agent_hidden_tools stays empty because the
    agent factory is shared with the worker — the API process injects edit_file
    hiding via AGENT_HIDDEN_TOOLS at startup (start_server.sh / start_desktop.sh)."""
    assert settings.agent_hidden_tools == ""
    assert settings.chat_funnel_hidden_capabilities == "cap-edit-file"


# ── 3. registration gate parsing (chat composition posture) ──────────────────

def test_register_fs_tools_exclude_only_skips_named(tmp_path):
    rt = ToolRuntime()
    tools = register_fs_tools(rt, tmp_path, sandbox=_NoSandbox(), exclude={"edit_file"})
    assert {t.name for t in tools} == {"read_file", "bash"}
    assert rt.get("edit_file") is None


def test_register_fs_tools_empty_exclude_registers_all(tmp_path):
    rt = ToolRuntime()
    tools = register_fs_tools(rt, tmp_path, sandbox=_NoSandbox(), exclude=())
    assert {t.name for t in tools} == {"read_file", "edit_file", "bash"}
