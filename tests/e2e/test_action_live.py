"""A-2 Live real-stack ACTION twin (Phase 4-D) — real physical-row persistence.

The pytest twin of ``scripts/e2e_live_action.py``: it drives the SAME production
L0 -> ACTION -> ToolRuntime -> real-tool-body chain over a real uvicorn API and
asserts the effect as a COMMITTED PostgreSQL row (direct ``asyncpg`` query, not an
in-process recorder), for both P0 capabilities:

    create_folder  -> public.folders (root: workspace_id IS NULL)
    add_term       -> public.terms   (bound to a script-created caller-owned domain)

Live-stack discipline. Every test is ``@pytest.mark.live`` and is SKIPPED unless
``DELVETA_LIVE_TEST=1`` AND a real PostgreSQL is reachable; the shared server
fixture skips if the API cannot boot. This keeps ``pytest tests`` green on a
machine with no container stack, while an operator with the core stack up
(``docker compose up -d postgres redis``) gets a genuine end-to-end proof:

    DELVETA_LIVE_TEST=1 python -m pytest tests/e2e/test_action_live.py -v -m live

Each test deletes only its OWN rows (unique per-run names) so the DB is left clean.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid

import asyncpg
import httpx
import pytest

pytestmark = pytest.mark.live

# A distinct port from the script's 8312 so the two can never collide if a stale
# uvicorn lingers while the operator runs the suite.
PORT = 8313
BASE = f"http://127.0.0.1:{PORT}"
DB_DSN = os.environ.get(
    "DELVETA_LIVE_DB_DSN", "postgresql://delveta:delveta@localhost:15432/delveta"
)
LOGIN = {"username": "admin", "password": "pwd@Admin"}


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
    with open("logs/_e2e_live_action.pytest.log", "w", encoding="utf-8") as log:
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


# ── real-chain drivers ──────────────────────────────────────────────────────────
async def _login(conn: asyncpg.Connection) -> tuple:
    async with httpx.AsyncClient(base_url=BASE, timeout=30.0) as c:
        r = await c.post("/auth/login", json=LOGIN)
        assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
        auth = {"Authorization": f"Bearer {r.json()['access_token']}"}
    uid = await conn.fetchval("select id from users where username = 'admin'")
    assert uid is not None, "seed admin user missing"
    return uid, auth


async def _sse_turn(auth: dict, msg: str) -> dict:
    """Drive one /chat/stream turn over the wire, approving any ASK; return the trace."""
    approval_seen = None
    async with (
        httpx.AsyncClient(base_url=BASE, timeout=300.0) as c,
        c.stream("POST", "/chat/stream", headers=auth, json={"message": msg}) as r,
    ):
        assert r.status_code == 200, f"stream status {r.status_code}"
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "approval-request":
                d = evt.get("data") if isinstance(evt.get("data"), dict) else {}
                approval_seen = d.get("approval_id")
                await c.post(f"/approvals/{approval_seen}", headers=auth,
                             json={"allow": True})
    assert approval_seen is not None, "an ASK must surface for a WRITE capability"
    return {"approval": approval_seen}


async def _fetch_row(conn: asyncpg.Connection, sql: str, *args, tries: int = 20):
    """Poll for a committed row — the turn commits during stream FINALIZE (after done)."""
    for _ in range(tries):
        row = await conn.fetchrow(sql, *args)
        if row:
            return row
        await asyncio.sleep(0.5)
    return None


async def _create_domain(auth: dict, name: str) -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=30.0) as c:
        r = await c.post("/domains", headers=auth, json={"name": name})
        assert r.status_code in (200, 201), f"domain create failed: {r.status_code} {r.text}"


# ── tests ───────────────────────────────────────────────────────────────────────
async def test_create_folder_persists_physical_row(live_server):
    """`新建文件夹"X"` -> L0 -> ACTION -> real body -> a COMMITTED folders row (root)."""
    suf = uuid.uuid4().int % 900000 + 100000
    folder = f"E2E归档{suf}"
    conn = await asyncpg.connect(DB_DSN)
    uid = None
    try:
        uid, auth = await _login(conn)
        await _sse_turn(auth, f'新建文件夹"{folder}"')
        row = await _fetch_row(
            conn,
            "select id, path, user_id, workspace_id from public.folders "
            "where user_id = $1 and path = $2", uid, folder,
        )
        assert row is not None, f"no committed folder row for {folder!r}"
        assert str(row["user_id"]) == str(uid)
        assert row["workspace_id"] is None, "a My Drive root folder must have NULL workspace_id"
    finally:
        if uid is not None:
            await conn.execute(
                "delete from public.folders where user_id = $1 and path = $2", uid, folder)
        await conn.close()


async def test_add_term_persists_physical_row(live_server):
    """`把"T"加入D词汇库` -> L0 -> ACTION -> real body -> a COMMITTED terms row."""
    suf = uuid.uuid4().int % 900000 + 100000
    domain = f"E2E验收{suf}"
    term = f"keystone{suf}"
    conn = await asyncpg.connect(DB_DSN)
    uid = domain_id = None
    try:
        uid, auth = await _login(conn)
        await _create_domain(auth, domain)
        domain_id = await conn.fetchval(
            "select id from public.domains where user_id = $1 and name = $2", uid, domain)
        assert domain_id is not None, f"fixture domain {domain!r} not persisted"
        await _sse_turn(auth, f'把"{term}"加入{domain}词汇库')
        row = await _fetch_row(
            conn,
            "select id, word, definition, domain_id from public.terms "
            "where domain_id = $1 and word = $2", domain_id, term,
        )
        assert row is not None, f"no committed term row for {term!r}"
        assert str(row["domain_id"]) == str(domain_id)
    finally:
        if domain_id is not None:
            await conn.execute("delete from public.terms where word = $1", term)
        if uid is not None:
            await conn.execute(
                "delete from public.domains where user_id = $1 and name = $2", uid, domain)
        await conn.close()
