"""The Cloud Drive working folder as TURN CONTEXT, and the writes that honour it.

Two layers pinned here:

* ``ChatRequest.drive`` → ``_resolve_drive_cwd`` → ``agent_context["drive_cwd"]``.
  The folder id is authoritative (a real row the user can see), the (workspace,
  path) pair is membership-checked, and a declared folder that cannot be honoured
  becomes an ERROR the write tools refuse — never a silent fallback.
* the write tools (``create_folder`` / ``create_file``) actually land rows in that
  folder: the default is the working folder, an explicitly named destination
  overrides it for that one call, and the pre-existing My-Drive-root behaviour
  survives when the turn carries no Cloud context.
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from agent import Context
from agent.engine.context import _TURN_CTX, AgentTurn
from agent.engine.decisions import ToolExecution
from agent.engine.runtime import ToolRuntime

from api.schemas import DriveContext
from api.tools import create_file_tool, create_folder_tool
from core.application.chat.context import _resolve_drive_cwd
from core.infrastructure.request_context import set_request_user
from tests._drive_fakes import make_drive

USER = uuid4()


# ── helpers ────────────────────────────────────────────────────────────────────


def _body(**drive) -> SimpleNamespace:
    return SimpleNamespace(drive=DriveContext(**drive) if drive else None)


async def _mk(svc, uid, path: str, *parts: str) -> dict:
    full = "/".join((path,) + parts)
    parent, _, name = full.rpartition("/")
    return await svc.create_folder(uid, None, parent or None, name)


def _turn_cwd(cwd: dict):
    return _TURN_CTX.set(AgentTurn(user_msg="", context={"drive_cwd": cwd}))


def _runtime(mod, drive, monkeypatch) -> ToolRuntime:
    runtime = ToolRuntime()
    ctx = Context()
    ctx.provide("session_factory", object)   # unused: the drive is swapped below
    monkeypatch.setattr(mod, "DriveService", lambda session_factory: drive)
    mod.register(runtime, ctx, None)
    return runtime


async def _execute(runtime, tool: str, args: dict):
    return await runtime.execute(ToolExecution("t1", tool, args))


@pytest.fixture(autouse=True)
def _user():
    set_request_user(USER)
    yield
    set_request_user(None)


# ── context resolution ─────────────────────────────────────────────────────────


async def test_no_drive_payload_means_no_working_folder(tmp_path):
    svc = make_drive(tmp_path)
    assert await _resolve_drive_cwd(_body(), svc, USER) == (None, "")


async def test_guest_turn_has_no_working_folder(tmp_path):
    svc = make_drive(tmp_path)
    cwd, error = await _resolve_drive_cwd(_body(path="Docs"), svc, None)
    assert (cwd, error) == (None, "")


async def test_folder_id_is_authoritative_over_the_path(tmp_path):
    svc = make_drive(tmp_path)
    folder = await _mk(svc, USER, "12100203")
    cwd, error = await _resolve_drive_cwd(
        _body(folder_id=UUID(folder["id"]), path="somewhere/else"), svc, USER)
    assert error == ""
    assert cwd["folder_id"] == folder["id"]
    assert cwd["path"] == "12100203"
    assert cwd["workspace_id"] is None


async def test_unknown_folder_id_refuses_instead_of_falling_back(tmp_path):
    svc = make_drive(tmp_path)
    await _mk(svc, USER, "12100203")
    cwd, error = await _resolve_drive_cwd(
        _body(folder_id=uuid4(), path="12100203"), svc, USER)
    assert cwd is None
    assert "no longer available" in error


async def test_personal_path_without_a_row_is_accepted(tmp_path):
    """A folder implied by files has no row of its own but is still a real location."""
    svc = make_drive(tmp_path)
    cwd, error = await _resolve_drive_cwd(_body(path="implied/by/files"), svc, USER)
    assert error == ""
    assert cwd["path"] == "implied/by/files"
    assert cwd["folder_id"] is None


async def test_workspace_working_folder_requires_membership(tmp_path):
    svc = make_drive(tmp_path)
    foreign = await svc.create_workspace(uuid4(), "Someone else")
    cwd, error = await _resolve_drive_cwd(
        _body(workspace_id=UUID(foreign["id"])), svc, USER)
    assert cwd is None
    assert "not accessible" in error


async def test_workspace_working_folder_is_resolved_for_a_member(tmp_path):
    svc = make_drive(tmp_path)
    ws = await svc.create_workspace(USER, "Team")
    ws_id = UUID(ws["id"])
    await svc.create_folder(USER, ws_id, None, "Docs")
    cwd, error = await _resolve_drive_cwd(
        _body(workspace_id=ws_id, path="Docs", label="Team / Docs"), svc, USER)
    assert error == ""
    assert cwd["workspace_id"] == str(ws_id)
    assert cwd["path"] == "Docs"
    assert cwd["label"] == "Team / Docs"


# ── the write tools ────────────────────────────────────────────────────────────


async def test_create_folder_defaults_to_the_working_folder(tmp_path, monkeypatch):
    drive = make_drive(tmp_path / "drive")
    await _mk(drive, USER, "12100203")
    runtime = _runtime(create_folder_tool, drive, monkeypatch)
    token = _turn_cwd({"workspace_id": None, "folder_id": None,
                       "path": "12100203", "label": "My Drive / 12100203"})
    try:
        res = await _execute(runtime, "create_folder", {"name": "test"})
    finally:
        _TURN_CTX.reset(token)
    assert not res.is_error, res.error
    assert res.value == "Created folder '12100203/test' in your drive."
    assert {f["path"] for f in await drive.list_folders(USER)} == {
        "12100203", "12100203/test"}


async def test_create_folder_honours_a_destination_the_user_named(tmp_path, monkeypatch):
    drive = make_drive(tmp_path / "drive")
    await _mk(drive, USER, "12100203")
    await _mk(drive, USER, "Documents")
    runtime = _runtime(create_folder_tool, drive, monkeypatch)
    token = _TURN_CTX.set(AgentTurn(
        user_msg="在 Documents 文件夹下面创建一个文件夹 test",
        context={"drive_cwd": {"workspace_id": None, "folder_id": None,
                               "path": "12100203", "label": "My Drive / 12100203"}}))
    try:
        res = await _execute(runtime, "create_folder", {"name": "test"})
    finally:
        _TURN_CTX.reset(token)
    assert not res.is_error, res.error
    assert res.value == "Created folder 'Documents/test' in your drive."


async def test_create_file_lands_in_the_working_folder(tmp_path, monkeypatch):
    drive = make_drive(tmp_path / "drive")
    await _mk(drive, USER, "12100203")
    runtime = _runtime(create_file_tool, drive, monkeypatch)
    token = _turn_cwd({"workspace_id": None, "folder_id": None,
                       "path": "12100203", "label": "My Drive / 12100203"})
    try:
        res = await _execute(
            runtime, "create_file", {"name": "abc.txt", "content": "hello"})
    finally:
        _TURN_CTX.reset(token)
    assert not res.is_error, res.error
    assert res.value == "Created file 'My Drive / 12100203 / abc.txt' in your drive."
    files = await drive.list_files(USER)
    assert [(f["name"], f["folder_path"]) for f in files] == [("abc.txt", "12100203")]


async def test_create_file_explicit_root_overrides_the_working_folder(tmp_path, monkeypatch):
    drive = make_drive(tmp_path / "drive")
    await _mk(drive, USER, "12100203")
    runtime = _runtime(create_file_tool, drive, monkeypatch)
    token = _TURN_CTX.set(AgentTurn(
        user_msg="把摘要保存到 My Drive 下面的 abc.txt",
        context={"drive_cwd": {"workspace_id": None, "folder_id": None,
                               "path": "12100203", "label": "My Drive / 12100203"}}))
    try:
        res = await _execute(
            runtime, "create_file", {"name": "abc.txt", "content": "hi"})
    finally:
        _TURN_CTX.reset(token)
    assert not res.is_error, res.error
    assert [(f["name"], f["folder_path"]) for f in await drive.list_files(USER)] == [
        ("abc.txt", None)]


async def test_create_file_without_cloud_context_keeps_the_drive_root(tmp_path, monkeypatch):
    drive = make_drive(tmp_path / "drive")
    runtime = _runtime(create_file_tool, drive, monkeypatch)
    token = _TURN_CTX.set(AgentTurn(user_msg="创建一个文件 abc.txt", context={}))
    try:
        res = await _execute(runtime, "create_file", {"name": "abc.txt"})
    finally:
        _TURN_CTX.reset(token)
    assert not res.is_error, res.error
    assert [(f["name"], f["folder_path"]) for f in await drive.list_files(USER)] == [
        ("abc.txt", None)]


async def test_a_refused_working_folder_writes_nothing(tmp_path, monkeypatch):
    drive = make_drive(tmp_path / "drive")
    await _mk(drive, USER, "12100203")
    runtime = _runtime(create_file_tool, drive, monkeypatch)
    token = _TURN_CTX.set(AgentTurn(
        user_msg="创建一个文件 abc.txt",
        context={"drive_error": "the selected Cloud Drive folder is gone"}))
    try:
        res = await _execute(runtime, "create_file", {"name": "abc.txt"})
    finally:
        _TURN_CTX.reset(token)
    assert res.is_error
    assert "gone" in str(res.error.message)
    assert await drive.list_files(USER) == []


# ── the summary save step ──────────────────────────────────────────────────────


async def test_summary_save_uses_the_file_name_the_user_wrote(tmp_path, monkeypatch):
    from api.tools.toolkit import summary_auto

    drive = make_drive(tmp_path / "drive")
    await _mk(drive, USER, "12100203")
    monkeypatch.setattr(summary_auto, "DriveService", lambda sf: drive)
    ctx = SimpleNamespace(resolve=lambda name: object)
    token = _TURN_CTX.set(AgentTurn(
        user_msg="总结一下这段内容并保存成 report.md",
        context={"drive_cwd": {"workspace_id": None, "folder_id": None,
                               "path": "12100203", "label": "My Drive / 12100203"}}))
    try:
        line = await summary_auto._save_to_drive(
            ctx, "# Summary\n", "report.md", stem="summary")
    finally:
        _TURN_CTX.reset(token)
    assert line == "- My Drive / 12100203 / report.md"
    files = await drive.list_files(USER)
    assert [(f["name"], f["folder_path"]) for f in files] == [("report.md", "12100203")]
    assert await drive.read_text(USER, UUID(files[0]["id"])) == "# Summary\n"


async def test_summary_save_defaults_the_name_and_honours_the_root(tmp_path, monkeypatch):
    from api.tools.toolkit import summary_auto

    drive = make_drive(tmp_path / "drive")
    await _mk(drive, USER, "12100203")
    monkeypatch.setattr(summary_auto, "DriveService", lambda sf: drive)
    ctx = SimpleNamespace(resolve=lambda name: object)
    token = _TURN_CTX.set(AgentTurn(
        user_msg="把总结放到 My Drive 下面",
        context={"drive_cwd": {"workspace_id": None, "folder_id": None,
                               "path": "12100203", "label": "My Drive / 12100203"}}))
    try:
        line = await summary_auto._save_to_drive(ctx, "body", "", stem="conversation")
    finally:
        _TURN_CTX.reset(token)
    assert line == "- My Drive / conversation.md"
    assert [(f["name"], f["folder_path"]) for f in await drive.list_files(USER)] == [
        ("conversation.md", None)]


async def test_summary_save_honours_a_my_drive_prefixed_save_path(tmp_path, monkeypatch):
    """The model may copy the whole path it saw ("My Drive/abc.txt")."""
    from api.tools.toolkit import summary_auto

    drive = make_drive(tmp_path / "drive")
    await _mk(drive, USER, "12100203")
    monkeypatch.setattr(summary_auto, "DriveService", lambda sf: drive)
    ctx = SimpleNamespace(resolve=lambda name: object)
    token = _TURN_CTX.set(AgentTurn(
        user_msg="把摘要保存成 My Drive/abc.txt",
        context={"drive_cwd": {"workspace_id": None, "folder_id": None,
                               "path": "12100203", "label": "My Drive / 12100203"}}))
    try:
        line = await summary_auto._save_to_drive(
            ctx, "body", "My Drive/abc.txt", stem="summary")
    finally:
        _TURN_CTX.reset(token)
    assert line == "- My Drive / abc.txt"
    assert [(f["name"], f["folder_path"]) for f in await drive.list_files(USER)] == [
        ("abc.txt", None)]


async def test_summary_save_is_a_no_op_without_a_cloud_turn():
    from api.tools.toolkit import summary_auto

    ctx = SimpleNamespace(resolve=lambda name: object)
    token = _TURN_CTX.set(AgentTurn(user_msg="总结一下", context={}))
    try:
        assert await summary_auto._save_to_drive(
            ctx, "body", "report.md", stem="summary") is None
    finally:
        _TURN_CTX.reset(token)
