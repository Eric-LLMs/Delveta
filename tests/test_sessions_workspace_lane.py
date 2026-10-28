"""WorkSpace assistant session lane (``sessions.type = 2``).

A WorkSpace conversation is an ordinary server session tagged type 2 so it has its own
listing lane: ``GET /sessions?kind=workspace`` returns type 2 only, the chat lane returns
type 0 only, and the descriptive scope lives in ``sessions.meta``. These tests pin the
lane filter (in SQL and again at the route boundary) and the PATCH meta merge without a
real database.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from api.auth import AuthUser
from api.routers import sessions as sessions_module
from api.schemas import SessionRenameRequest
from core.infrastructure.db import UserRoleModel
from core.infrastructure.memory import list_sessions

USER = uuid.uuid4()
_FAKE_ROLE = UserRoleModel(role_id="user", role_name="User")


def _auth(user_id: uuid.UUID) -> AuthUser:
    return AuthUser(
        user_id=user_id,
        username="alice",
        display_name=None,
        role=_FAKE_ROLE,
        token_id=uuid.uuid4(),
    )


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []
        self.committed = False

    async def execute(self, stmt):
        self.statements.append(stmt)
        return _Result(self.rows)

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _row(title: str, type_: int, meta: dict | None = None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        created_at=None,
        summary=None,
        title=title,
        type=type_,
        meta=meta or {},
    )


def _factory(rows):
    holder = {}

    def factory():
        holder["session"] = _FakeSession(rows)
        return holder["session"]

    return factory, holder


@pytest.mark.asyncio
async def test_list_sessions_filters_by_kind_in_sql():
    factory, holder = _factory([_row("ws", 2)])
    await list_sessions(factory, USER, None, kind="workspace")
    sql = str(holder["session"].statements[0].compile(compile_kwargs={"literal_binds": True}))
    assert "sessions.type = 2" in sql


@pytest.mark.asyncio
async def test_list_sessions_chat_kind_filters_type_zero_and_returns_meta():
    row = _row("chat", 0, {"workspace_name": "My Drive"})
    factory, holder = _factory([row])
    out = await list_sessions(factory, USER)
    sql = str(holder["session"].statements[0].compile(compile_kwargs={"literal_binds": True}))
    assert "sessions.type = 0" in sql
    assert out[0]["type"] == 0
    assert out[0]["meta"] == {"workspace_name": "My Drive"}


@pytest.mark.asyncio
async def test_list_sessions_rejects_unknown_kind():
    factory, _ = _factory([])
    with pytest.raises(ValueError):
        await list_sessions(factory, USER, None, kind="research")


@pytest.mark.asyncio
async def test_workspace_route_passes_kind_and_keeps_only_type_two(monkeypatch):
    captured = {}

    async def _fake_list(_factory, _user_id, q, kind="chat"):
        captured["kind"] = kind
        return [
            {"id": "a", "created_at": None, "summary": None, "title": "ws", "type": 2},
            {"id": "b", "created_at": None, "summary": None, "title": "chat", "type": 0},
        ]

    monkeypatch.setattr(sessions_module, "list_sessions", _fake_list)
    result = await sessions_module.get_sessions(user=_auth(USER), q=None, kind="workspace")
    assert captured["kind"] == "workspace"
    assert [s["id"] for s in result["sessions"]] == ["a"]


@pytest.mark.asyncio
async def test_workspace_route_rejects_unknown_kind():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await sessions_module.get_sessions(user=_auth(USER), q=None, kind="research")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_patch_merges_meta_and_leaves_omitted_title(monkeypatch):
    sess = SimpleNamespace(
        id=uuid.uuid4(),
        user_id=USER,
        title="kept",
        meta={"workspace_name": "My Drive", "folder_path": "a"},
    )
    fake = _FakeSession([sess])
    fake.rows = [sess]
    monkeypatch.setattr(sessions_module, "SessionLocal", lambda: fake)

    out = await sessions_module.rename_session(
        sess.id,
        SessionRenameRequest(meta={"folder_path": None, "folder_name": None}),
        user=_auth(USER),
    )
    assert fake.committed is True
    assert sess.title == "kept"  # title omitted → untouched
    assert out["meta"] == {"workspace_name": "My Drive", "folder_path": None, "folder_name": None}


@pytest.mark.asyncio
async def test_patch_still_resets_title_when_sent_empty(monkeypatch):
    sess = SimpleNamespace(id=uuid.uuid4(), user_id=USER, title="old", meta={})
    fake = _FakeSession([sess])
    monkeypatch.setattr(sessions_module, "SessionLocal", lambda: fake)

    out = await sessions_module.rename_session(
        sess.id, SessionRenameRequest(title="  "), user=_auth(USER),
    )
    assert sess.title is None
    assert out["title"] is None
