"""Live E2E: the Cloud Drive working folder, from the /chat wire to the physical row.

Gated exactly like ``test_action_live.py``: skipped unless ``DELVETA_LIVE_TEST=1``
and a real PostgreSQL is reachable, and it boots its OWN uvicorn on port 8314 so
the running dev server on 8300 is never touched.

Two proven layers:

* **wire** — a real ``POST /chat/stream`` carrying ``ChatRequest.drive`` (the
  desktop client's selected folder) through the funnel's L0/ACTION lane into the
  real tool body, asserted as a COMMITTED ``public.folders`` row (direct asyncpg).
  No model is involved: the quoted-name extractor certifies the capability.
* **tool lane** — the real tool bodies (``create_file``, the summary save) over
  the app's own ``SessionLocal`` against the same PostgreSQL, with the turn
  context the chat assembles. Folder ids come from real rows, never fakes.

Nothing here asserts UI text; the scenario labels match the acceptance sheet.

    DELVETA_LIVE_TEST=1 python -m pytest tests/e2e/test_drive_cwd_live.py -v -m live
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from types import SimpleNamespace

import asyncpg
import httpx
import pytest
from agent import Context, ToolExecution, ToolRuntime
from agent.engine.context import _TURN_CTX, AgentTurn

from api.schemas import DriveContext
from api.tools import create_file_tool
from core.application.chat.context import _resolve_drive_cwd
from core.application.drive_service import DriveService
from core.infrastructure.db import SessionLocal, engine
from core.infrastructure.request_context import set_request_user

pytestmark = pytest.mark.live

PORT = 8314
BASE = f"http://127.0.0.1:{PORT}"
DB_DSN = os.environ.get(
    "DELVETA_LIVE_DB_DSN", "postgresql://delveta:delveta@localhost:15432/delveta"
)
LOGIN = {"username": "admin", "password": "pwd@Admin"}
ADMIN = "admin"


# ── live-stack gating ───────────────────────────────────────────────────────────
def _live_enabled() -> bool:
    return os.environ.get("DELVETA_LIVE_TEST") == "1"


async def _db_ok() -> bool:
    try:
        conn = await asyncio.wait_for(asyncpg.connect(DB_DSN), timeout=5)
        await conn.close()
        return True
    except Exception:  # noqa: BLE001 - absence of the live DB is a SKIP, not a failure
        return False


async def _wait_health() -> bool:
    async with httpx.AsyncClient(base_url=BASE, timeout=5.0) as c:
        for _ in range(60):
            try:
                if (await c.get("/health")).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    return False


@pytest.fixture(scope="module")
def live_server():
    """Boot a real uvicorn API over the real PG/Redis; SKIP when the stack is absent."""
    if not _live_enabled():
        pytest.skip("live stack disabled: set DELVETA_LIVE_TEST=1")
    if not asyncio.run(_db_ok()):
        pytest.skip(f"live PostgreSQL unreachable at {DB_DSN}")
    os.makedirs("logs", exist_ok=True)
    with open("logs/_e2e_drive_cwd.pytest.log", "w", encoding="utf-8") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "apps.api.main:app",
             "--port", str(PORT), "--log-level", "warning"],
            stdout=log, stderr=subprocess.STDOUT, env=os.environ,
        )
        try:
            if not asyncio.run(_wait_health()):
                pytest.skip("live API did not become healthy")
            yield BASE
        finally:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()


# ── fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
async def admin(live_server):
    """(user_id, auth_headers) for the seeded admin over the live stack."""
    async with httpx.AsyncClient(base_url=BASE, timeout=30.0) as c:
        r = await c.post("/auth/login", json=LOGIN)
        assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
        auth = {"Authorization": f"Bearer {r.json()['access_token']}"}
    conn = await asyncpg.connect(DB_DSN)
    try:
        uid = await conn.fetchval("select id from users where username = $1", ADMIN)
        assert uid is not None, "seed admin user missing"
    finally:
        await conn.close()
    return uid, auth


@pytest.fixture
async def drive():
    """The real DriveService over the app's SessionLocal (a fresh pool per test loop)."""
    await engine.dispose()
    svc = DriveService(SessionLocal)
    try:
        yield svc
    finally:
        await engine.dispose()


@pytest.fixture
async def conn():
    c = await asyncpg.connect(DB_DSN)
    try:
        yield c
    finally:
        await c.close()


def _tag() -> str:
    return str(uuid.uuid4().int % 900000 + 100000)


async def _root_folder(drive: DriveService, uid, name: str) -> dict:
    return await drive.create_folder(uid, None, None, name)


async def _cleanup(conn, drive: DriveService, uid, tag: str, *roots: str) -> None:
    """Delete this test's OWN rows (asset rows through the service, folders by path).

    The tag is unique per test, so ``name like %tag%`` can only match this test's
    assets. Every folder it names lives under one of ``roots``.
    """
    assets = await conn.fetch(
        "select id from public.assets where user_id = $1 and name like '%' || $2 || '%'",
        uid, tag)
    for asset in assets:
        try:
            await drive.delete_asset(uid, asset["id"])
        except Exception:  # noqa: BLE001 - already gone is fine during cleanup
            pass
    for root in roots:
        await conn.execute(
            "delete from public.folders where user_id = $1 "
            "and (path = $2 or path like $2 || '/%')", uid, root)


async def _fetch_row(conn, sql: str, *args, tries: int = 20):
    for _ in range(tries):
        row = await conn.fetchrow(sql, *args)
        if row:
            return row
        await asyncio.sleep(0.5)
    return None


async def _wire_turn(auth: dict, message: str, drive_ctx: dict | None) -> list[dict]:
    """Drive one real /chat/stream turn, approving any ASK; return the SSE events."""
    events: list[dict] = []
    payload: dict = {"message": message}
    if drive_ctx is not None:
        payload["drive"] = drive_ctx
    async with (
        httpx.AsyncClient(base_url=BASE, timeout=300.0) as c,
        c.stream("POST", "/chat/stream", headers=auth, json=payload) as r,
    ):
        assert r.status_code == 200, f"stream status {r.status_code}: {r.text}"
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            events.append(evt)
            if evt.get("type") == "approval-request":
                data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
                approval = data.get("approval_id")
                if approval:
                    await c.post(f"/approvals/{approval}", headers=auth,
                                 json={"allow": True})
    return events


def _cwd(row: dict, label: str) -> dict:
    return {"workspace_id": None, "folder_id": str(row["id"]),
            "path": row["path"], "label": label}


async def _run_tool(uid, mod, tool: str, args: dict, *, query: str,
                    cwd: dict | None = None, error: str = ""):
    """Execute a real tool body over the real DB with the chat's turn context."""
    runtime = ToolRuntime()
    ctx = Context()
    ctx.provide("session_factory", SessionLocal)
    mod.register(runtime, ctx, None)
    context: dict = {}
    if cwd is not None:
        context["drive_cwd"] = cwd
    if error:
        context["drive_error"] = error
    set_request_user(uid)
    token = _TURN_CTX.set(AgentTurn(user_msg=query, context=context))
    try:
        return await runtime.execute(ToolExecution("t1", tool, args))
    finally:
        _TURN_CTX.reset(token)
        set_request_user(None)


# ── (S2/S3) the wire: the selected folder is the write root ─────────────────────
async def test_wire_turn_writes_under_the_selected_folder(admin, drive, conn):
    uid, auth = admin
    tag = _tag()
    root = f"E2E工作区{tag}"
    folder = await _root_folder(drive, uid, root)
    child = f"子{tag}"
    try:
        events = await _wire_turn(
            auth, f'新建文件夹"{child}"',
            _cwd(folder, f"My Drive / {root}"))
        row = await _fetch_row(
            conn,
            "select id, path, workspace_id from public.folders "
            "where user_id = $1 and path = $2", uid, f"{root}/{child}",
        )
        assert row is not None, (
            f"no committed row under the selected folder; events={events}")
        assert row["workspace_id"] is None
        root_row = await conn.fetchrow(
            "select id from public.folders where user_id = $1 and path = $2", uid, root)
        assert root_row is not None and str(row["id"]) != str(root_row["id"])
        assert not await conn.fetchval(
            "select 1 from public.folders where user_id = $1 and path = $2", uid, child
        ), "the folder must NOT land at the My Drive root"
    finally:
        await _cleanup(conn, drive, uid, tag, root)


async def test_wire_switching_folders_keeps_the_write_location_in_sync(admin, drive, conn):
    uid, auth = admin
    tag = _tag()
    a_name, b_name = f"E2E甲{tag}", f"E2E乙{tag}"
    a = await _root_folder(drive, uid, a_name)
    b = await _root_folder(drive, uid, b_name)
    child = f"切换{tag}"
    try:
        await _wire_turn(auth, f'新建文件夹"{child}"', _cwd(a, f"My Drive / {a_name}"))
        row_a = await _fetch_row(
            conn, "select path from public.folders where user_id = $1 and path = $2",
            uid, f"{a_name}/{child}")
        await _wire_turn(auth, f'新建文件夹"{child}"', _cwd(b, f"My Drive / {b_name}"))
        row_b = await _fetch_row(
            conn, "select path from public.folders where user_id = $1 and path = $2",
            uid, f"{b_name}/{child}")
        assert row_a is not None, "first turn did not land under folder A"
        assert row_b is not None, "second turn did not land under folder B"
        assert str(row_a["path"]) == f"{a_name}/{child}"
        assert str(row_b["path"]) == f"{b_name}/{child}"
    finally:
        await _cleanup(conn, drive, uid, tag, a_name)
        await _cleanup(conn, drive, uid, tag, b_name)


async def test_wire_named_folder_overrides_the_working_folder(admin, drive, conn):
    """S6: '在 <name> 下面 …' wins for THIS turn; the cwd itself is untouched."""
    uid, auth = admin
    tag = _tag()
    cwd_name, docs_name = f"E2E工作区{tag}", f"Docs{tag}"
    cwd = await _root_folder(drive, uid, cwd_name)
    await _root_folder(drive, uid, docs_name)
    child = f"显式{tag}"
    try:
        await _wire_turn(
            auth, f'在 {docs_name} 下面新建文件夹"{child}"',
            _cwd(cwd, f"My Drive / {cwd_name}"))
        row = await _fetch_row(
            conn, "select path from public.folders where user_id = $1 and path = $2",
            uid, f"{docs_name}/{child}")
        assert row is not None, "the explicitly named folder did not receive the write"
        assert not await conn.fetchval(
            "select 1 from public.folders where user_id = $1 and path = $2",
            uid, f"{cwd_name}/{child}",
        ), "the working folder must not receive an explicitly redirected write"
    finally:
        await _cleanup(conn, drive, uid, tag, cwd_name)
        await _cleanup(conn, drive, uid, tag, docs_name)


# ── (S3) create_file lands in the working folder ────────────────────────────────
async def test_create_file_lands_in_the_working_folder(admin, drive, conn):
    uid, _ = admin
    tag = _tag()
    root = f"E2E文件{tag}"
    folder = await _root_folder(drive, uid, root)
    name = f"abc{tag}.txt"
    try:
        res = await _run_tool(
            uid, create_file_tool, "create_file",
            {"name": name, "content": "hello from the working folder"},
            query=f"创建一个文件 {name}",
            cwd={"workspace_id": None, "folder_id": str(folder["id"]),
                 "path": root, "label": f"My Drive / {root}"},
        )
        assert not res.is_error, res.error
        assert res.value == f"Created file 'My Drive / {root} / {name}' in your drive."
        row = await conn.fetchrow(
            "select folder_path, workspace_id, file_status from public.assets "
            "where user_id = $1 and name = $2", uid, name)
        assert row is not None and row["folder_path"] == root
        assert row["workspace_id"] is None
    finally:
        await _cleanup(conn, drive, uid, tag, root)


# ── (S4) the summary save lands in the working folder ───────────────────────────
async def test_summary_save_lands_in_the_working_folder(admin, drive, conn):
    from api.tools.toolkit import summary_auto

    uid, _ = admin
    tag = _tag()
    root = f"E2E摘要{tag}"
    folder = await _root_folder(drive, uid, root)
    name = f"abc{tag}.txt"
    ctx = SimpleNamespace(resolve=lambda key: SessionLocal)
    token = _TURN_CTX.set(AgentTurn(user_msg=f"把摘要保存成 {name}", context={
        "drive_cwd": {"workspace_id": None, "folder_id": str(folder["id"]),
                      "path": root, "label": f"My Drive / {root}"}}))
    set_request_user(uid)
    try:
        line = await summary_auto._save_to_drive(
            ctx, f"# 摘要 {tag}\n", name, stem="summary")
    finally:
        _TURN_CTX.reset(token)
        set_request_user(None)
    try:
        assert line == f"- My Drive / {root} / {name}", line
        row = await conn.fetchrow(
            "select id, folder_path from public.assets where user_id = $1 and name = $2",
            uid, name)
        assert row is not None and row["folder_path"] == root
        assert await drive.read_text(uid, row["id"]) == f"# 摘要 {tag}\n"
    finally:
        await _cleanup(conn, drive, uid, tag, root)


# ── (S5/S6) an explicit root wins for ONE operation only ────────────────────────
async def test_explicit_root_overrides_only_that_operation(admin, drive, conn):
    uid, _ = admin
    tag = _tag()
    root = f"E2E交替{tag}"
    folder = await _root_folder(drive, uid, root)
    cwd = {"workspace_id": None, "folder_id": str(folder["id"]),
           "path": root, "label": f"My Drive / {root}"}
    inside, outside, again = f"in{tag}.txt", f"root{tag}.txt", f"in2{tag}.txt"
    try:
        r1 = await _run_tool(uid, create_file_tool, "create_file",
                             {"name": inside, "content": "1"},
                             query=f"创建一个文件 {inside}", cwd=cwd)
        r2 = await _run_tool(uid, create_file_tool, "create_file",
                             {"name": outside, "content": "2"},
                             query=f"把内容保存到 My Drive 下面的 {outside}", cwd=cwd)
        r3 = await _run_tool(uid, create_file_tool, "create_file",
                             {"name": again, "content": "3"},
                             query=f"再创建一个文件 {again}", cwd=cwd)
        assert not (r1.is_error or r2.is_error or r3.is_error)
        rows = await conn.fetch(
            "select name, folder_path from public.assets "
            "where user_id = $1 and name = any($2::text[])",
            uid, [inside, outside, again])
        landed = {r["name"]: r["folder_path"] for r in rows}
        assert landed == {inside: root, outside: None, again: root}, landed
    finally:
        await _cleanup(conn, drive, uid, tag, root)


# ── (S8) an unreachable target folder is refused, never silently replaced ───────
async def test_refused_working_folder_writes_nothing(admin, drive, conn):
    uid, auth = admin
    tag = _tag()
    foreign = await conn.fetchval(
        "select id from public.folders where user_id <> $1 and path = '数据分析' limit 1", uid)
    assert foreign is not None, "fixture folder 数据分析 (another user) missing"

    body = SimpleNamespace(drive=DriveContext(
        folder_id=foreign, path="数据分析", label="Someone / 数据分析"))
    cwd, error = await _resolve_drive_cwd(body, drive, uid)
    assert cwd is None and "no longer available" in error

    name = f"拒绝{tag}.txt"
    res = await _run_tool(uid, create_file_tool, "create_file",
                          {"name": name, "content": "x"},
                          query=f"创建一个文件 {name}", error=error)
    assert res.is_error and "no longer available" in str(res.error.message)
    assert await conn.fetchval(
        "select 1 from public.assets where user_id = $1 and name = $2", uid, name) is None

    child = f"拒绝{tag}"
    events = await _wire_turn(
        auth, f'新建文件夹"{child}"',
        {"workspace_id": None, "folder_id": str(foreign), "path": "数据分析",
         "label": "Someone / 数据分析"})
    assert await conn.fetchval(
        "select 1 from public.folders where user_id = $1 and path like $2",
        uid, f"%/{child}") is None, f"a refused turn wrote a folder; events={events}"
    assert await conn.fetchval(
        "select 1 from public.folders where user_id = $1 and path = $2", uid, child) is None
