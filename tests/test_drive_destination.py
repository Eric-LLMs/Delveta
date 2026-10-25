"""The Cloud Drive destination rule shared by every chat-facing drive write.

Pinned here (business rules, not implementation details):

* the working folder is the DEFAULT — a bare file name never picks another one;
* a destination the user names explodes the default for THAT operation only:
  "放到 My Drive 下面" (the personal root), "在 <name> 文件夹下面 …" (one named
  folder), "My Drive/a/b" (an explicit path);
* an explicit tool argument outranks the sentence;
* a name that matches zero or several visible folders is REFUSED — the chat never
  guesses, and never writes into a different directory;
* a turn whose declared working folder could not be verified refuses too (no
  silent fallback);
* a turn with no Cloud context at all returns ``None`` so the caller keeps its
  pre-existing local / My-Drive-root behaviour.
"""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from agent.engine.context import _TURN_CTX, AgentTurn

from api.tools.drive_destination import (
    DestinationError,
    DriveDestination,
    current_cwd,
    resolve_destination,
)
from tests._drive_fakes import make_drive


class _Turn:
    def __init__(self, query: str = "", *, cwd: dict | None = None, error: str = ""):
        self.query = query
        self.context: dict = {}
        if cwd is not None:
            self.context["drive_cwd"] = cwd
        if error:
            self.context["drive_error"] = error

    def __enter__(self) -> "_Turn":
        self._token = _TURN_CTX.set(
            AgentTurn(user_msg=self.query, context=dict(self.context)))
        return self

    def __exit__(self, *exc) -> None:
        _TURN_CTX.reset(self._token)


def _cwd(path: str = "12100203", *, ws: str | None = None,
         label: str = "") -> dict:
    return {"workspace_id": ws, "folder_id": None, "path": path,
            "label": label or (f"My Drive / {path}" if path else "My Drive")}


async def _mk(svc, uid, path: str) -> dict:
    """Create one personal-drive folder at ``path`` (ancestors included)."""
    parent, _, name = path.rpartition("/")
    return await svc.create_folder(uid, None, parent or None, name)


# ── no Cloud context ───────────────────────────────────────────────────────────


async def test_no_drive_context_keeps_the_callers_own_behaviour(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    with _Turn("创建一个文件夹 test"):
        assert await resolve_destination(svc, uid, query="创建一个文件夹 test") is None


async def test_current_cwd_is_none_outside_a_drive_turn(tmp_path):
    with _Turn("随便聊聊"):
        assert current_cwd() is None


# ── the working folder is the default ──────────────────────────────────────────


async def test_working_folder_is_the_default_destination(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    with _Turn("创建一个文件 abc.txt", cwd=_cwd()):
        dest = await resolve_destination(svc, uid, query="创建一个文件 abc.txt")
    assert dest == DriveDestination(None, "12100203", "My Drive / 12100203")


async def test_a_bare_file_name_never_moves_the_destination(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    with _Turn("把摘要保存成 report.md", cwd=_cwd()):
        dest = await resolve_destination(svc, uid, query="把摘要保存成 report.md")
    assert dest.folder_path == "12100203"


# ── a destination the USER named ───────────────────────────────────────────────


async def test_my_drive_phrase_selects_the_personal_root(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    with _Turn("把摘要放到 My Drive 下面", cwd=_cwd()):
        dest = await resolve_destination(svc, uid, query="把摘要放到 My Drive 下面")
    assert dest == DriveDestination(None, "", "My Drive")


async def test_current_folder_phrase_keeps_the_working_folder(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    with _Turn("在当前目录创建一个文件夹 test", cwd=_cwd()):
        dest = await resolve_destination(
            svc, uid, query="在当前目录创建一个文件夹 test")
    assert dest.folder_path == "12100203"


async def test_named_folder_in_the_sentence_wins_for_one_operation(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    await _mk(svc, uid, "Documents")
    with _Turn("在 Documents 文件夹下面创建 test", cwd=_cwd()):
        dest = await resolve_destination(
            svc, uid, query="在 Documents 文件夹下面创建 test")
    assert dest == DriveDestination(None, "Documents", "My Drive / Documents")


async def test_explicit_drive_path_pins_the_personal_scope(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "Documents/Reports")
    with _Turn("把摘要放到 My Drive/Documents/Reports", cwd=_cwd()):
        dest = await resolve_destination(
            svc, uid, query="把摘要放到 My Drive/Documents/Reports")
    assert dest == DriveDestination(None, "Documents/Reports",
                                    "My Drive / Documents / Reports")


async def test_unknown_named_folder_is_refused_not_guessed(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    with _Turn("在 Nope 文件夹下面创建 test", cwd=_cwd()):
        with pytest.raises(DestinationError, match="no folder named 'Nope'"):
            await resolve_destination(svc, uid, query="在 Nope 文件夹下面创建 test")


async def test_ambiguous_named_folder_is_refused(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "Reports")
    ws = await svc.create_workspace(uid, "Team")
    ws_id = UUID(ws["id"])
    await svc.create_folder(uid, ws_id, None, "Reports")
    with _Turn("在 Reports 文件夹下面创建 test", cwd=_cwd()):
        with pytest.raises(DestinationError, match="more than one folder"):
            await resolve_destination(
                svc, uid, query="在 Reports 文件夹下面创建 test")


# ── an explicit TOOL argument outranks the sentence ────────────────────────────


async def test_explicit_argument_outranks_the_sentence(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    await _mk(svc, uid, "Documents")
    query = "在 Documents 文件夹下面创建 test"
    with _Turn(query, cwd=_cwd()):
        dest = await resolve_destination(svc, uid, query=query, explicit="12100203")
    assert dest.folder_path == "12100203"


async def test_explicit_path_keeps_the_scope_of_the_working_folder(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    ws = await svc.create_workspace(uid, "Team")
    ws_id = UUID(ws["id"])
    await svc.create_folder(uid, ws_id, None, "Docs")
    cwd = _cwd("Docs", ws=str(ws_id), label="Team / Docs")
    with _Turn("在这里建一个子文件夹", cwd=cwd):
        dest = await resolve_destination(
            svc, uid, query="在这里建一个子文件夹", explicit="Docs/Sub")
    assert dest.workspace_id == str(ws_id)
    assert dest.folder_path == "Docs/Sub"


async def test_bare_root_alias_as_an_explicit_argument_is_the_root(tmp_path):
    """The toolkit's file-save passes the parent folder ("My Drive") explicitly."""
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    with _Turn("把摘要保存成 abc.txt", cwd=_cwd()):
        dest = await resolve_destination(
            svc, uid, query="把摘要保存成 abc.txt", explicit="My Drive")
    assert dest == DriveDestination(None, "", "My Drive")


async def test_root_prefixed_explicit_path_stays_in_the_personal_scope(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    await _mk(svc, uid, "Documents")
    with _Turn("把摘要保存成 abc.txt", cwd=_cwd()):
        dest = await resolve_destination(
            svc, uid, query="把摘要保存成 abc.txt", explicit="My Drive/Documents")
    assert dest == DriveDestination(None, "Documents", "My Drive / Documents")


# ── the declared working folder was refused upstream ───────────────────────────


async def test_declared_error_refuses_instead_of_falling_back(tmp_path):
    svc = make_drive(tmp_path)
    uid = uuid4()
    await _mk(svc, uid, "12100203")
    with _Turn("创建一个文件 abc.txt", error="the selected Cloud Drive folder is gone"):
        with pytest.raises(DestinationError, match="is gone"):
            await resolve_destination(
                svc, uid, query="创建一个文件 abc.txt", explicit="12100203")
