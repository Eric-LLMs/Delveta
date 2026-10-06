"""A-2 Live real-stack ACTION end-to-end (Phase 4-D): physical-row verification.

Unlike the hermetic A-1 harness (``tests/e2e/``), which fakes the outer world
(DB = memory, retrieval/web = doubles), this script rides the REAL assembly with
only the CORE stack up (PostgreSQL + Redis + a real uvicorn API):

    HTTP /chat/stream -> TurnOrchestrator.resolve_plan
      -> understanding (L0 DIRECT_TOOLS extractor certifies the tool + args)
      -> intent_funnel.route (L0-certified turn -> returned unchanged)
      -> build_execution_plan -> ACTION
      -> ActionExecutor -> chat._run_tool -> ToolRuntime (sandbox ASK -> approval)
      -> REAL tool body -> REAL PostgreSQL

Two P0 capabilities, each certified by the production L0 grammar (no registry
publish, so no embedding / LLM dependency — core stack only):

  ① create_folder  root My Drive (workspace_id = NULL), name de-duped by _unique_name
  ② add_term       into a script-created, caller-owned vocabulary domain

The physical-row assertion is a DIRECT asyncpg query against the real DB (NOT an
in-process recorder): the proof is a COMMITTED ROW with its real UUID. Script-owned
rows (the folder, the term, and the domain) are physically deleted in ``finally`` so
the dev DB is left unpolluted (append-only audit/telemetry rows, if any, stay).

Exit code 0 = every hard leg passed.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import asyncpg
import httpx

PORT = 8312
BASE = f"http://127.0.0.1:{PORT}"
# asyncpg wants a bare postgresql:// DSN (the app's SQLAlchemy URL carries +asyncpg).
DB_DSN = os.environ.get(
    "DELVETA_LIVE_DB_DSN", "postgresql://delveta:delveta@localhost:15432/delveta"
)
LOGIN = {"username": "admin", "password": "pwd@Admin"}

SUF = uuid.uuid4().int % 900000 + 100000
# Names deliberately avoid the vocab keywords (词库/词汇/单词/生词本): the L0
# add_term grammar stops its domain slot lazily at the first such keyword, so a
# domain containing one would split early and the leftover would trip the
# compound-demand guard (abstain). "E2E验收"/"E2E归档" keep the slot unambiguous.
FOLDER = f"E2E归档{SUF}"
DOMAIN = f"E2E验收{SUF}"
TERM = f"keystone{SUF}"
MSG_FOLDER = f'新建文件夹"{FOLDER}"'
MSG_TERM = f'把"{TERM}"加入{DOMAIN}词汇库'

FAILS: list[str] = []


def hard(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'BAD'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


@contextmanager
def _backend(port: int) -> Iterator[subprocess.Popen]:
    """Spawn a real uvicorn API; a SYNC ctx-manager so the blocking ``open``/``Popen``
    never sit inside the async body (the app itself is driven asynchronously below)."""
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/_e2e_live_action.backend.{port}.log", "w", encoding="utf-8") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "apps.api.main:app",
             "--port", str(port), "--log-level", "info"],
            stdout=log, stderr=subprocess.STDOUT, env=os.environ,
        )
        try:
            yield child
        finally:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()


async def fetch_row(conn: asyncpg.Connection, sql: str, *args, tries: int = 20, delay: float = 0.5):
    """Poll for a committed row — the turn commits during stream FINALIZE (after done)."""
    for _ in range(tries):
        row = await conn.fetchrow(sql, *args)
        if row:
            return row
        await asyncio.sleep(delay)
    return None


async def wait_up(client: httpx.AsyncClient) -> bool:
    for _ in range(60):
        try:
            if (await client.get("/health")).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1.0)
    return False


async def sse_turn(client: httpx.AsyncClient, auth: dict, msg: str) -> dict:
    """One authenticated /chat/stream turn over the wire; approves any ASK."""
    t0 = time.perf_counter()
    answer, session, approval = "", None, None
    async with client.stream(
        "POST", "/chat/stream", headers=auth, json={"message": msg}, timeout=300.0
    ) as r:
        assert r.status_code == 200, f"stream status {r.status_code}"
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            typ, data = evt.get("type"), evt.get("data")
            if typ == "approval-request":
                d = data if isinstance(data, dict) else {}
                approval = d.get("approval_id")
                print(f"      ASK: {d.get('name')} {d.get('arguments')}", flush=True)
                rr = await client.post(f"/approvals/{approval}", headers=auth, json={"allow": True})
                print(f"      approval resolved -> {rr.status_code}", flush=True)
            elif typ == "content":
                if isinstance(data, str):
                    answer += data
                elif isinstance(data, dict):
                    answer += data.get("content") or data.get("delta") or ""
            elif typ == "done" and isinstance(data, dict):
                session = data.get("session_id")
    return {"ms": (time.perf_counter() - t0) * 1000, "answer": answer,
            "session": session, "approval": approval}


async def teardown(conn: asyncpg.Connection, uid) -> dict:
    """Physically delete only THIS run's rows (unique names) and report what was removed.

    Order matters: terms reference domains, so delete the term before the domain. All
    three deletes are scoped to the script's own unique identifiers so no pre-existing
    user data can be touched.
    """
    removed = {"terms": 0, "folders": 0, "domains": 0}
    try:
        removed["terms"] = int((await conn.execute(
            "delete from public.terms where word = $1", TERM)).split()[-1])
        removed["folders"] = int((await conn.execute(
            "delete from public.folders where user_id = $1 and path = $2", uid, FOLDER)).split()[-1])
        removed["domains"] = int((await conn.execute(
            "delete from public.domains where user_id = $1 and name = $2", uid, DOMAIN)).split()[-1])
    except Exception as exc:  # noqa: BLE001 - teardown must never mask the real result
        print(f"      teardown warning: {exc}", flush=True)
    return removed


async def main() -> int:
    print(f"live A-2 target: {BASE}  db={DB_DSN.split('@')[-1]}")
    print(f"fixtures: FOLDER={FOLDER!r} DOMAIN={DOMAIN!r} TERM={TERM!r}")
    conn: asyncpg.Connection | None = None
    uid = None
    with _backend(PORT) as child:
        try:
            conn = await asyncpg.connect(DB_DSN)
            async with httpx.AsyncClient(base_url=BASE, timeout=30.0) as c:
                up = await wait_up(c)
                hard("A0 backend up", up, f"(pid {child.pid}, port {PORT})")
                if not up:
                    return 1

                r = await c.post("/auth/login", json=LOGIN)
                hard("A1 login", r.status_code == 200, str(r.status_code))
                if r.status_code != 200:
                    return 1
                auth = {"Authorization": f"Bearer {r.json()['access_token']}"}
                uid = await conn.fetchval("select id from users where username = 'admin'")
                hard("A2 admin user id", uid is not None, str(uid))
                if uid is None:
                    return 1

                # ── ② fixture domain: add_term needs a caller-visible domain ───
                r = await c.post("/domains", headers=auth, json={"name": DOMAIN})
                hard("A3 fixture domain created over REST", r.status_code in (200, 201), str(r.status_code))
                domain_id = await conn.fetchval(
                    "select id from public.domains where user_id = $1 and name = $2", uid, DOMAIN)
                hard("A4 fixture domain persisted", domain_id is not None, str(domain_id))
                if domain_id is None:
                    return 1

                # ── ① create_folder via the real L0 -> ACTION chain ────────────
                print(f"      turn 1 message: {MSG_FOLDER!r}", flush=True)
                tr = await sse_turn(c, auth, MSG_FOLDER)
                print(f"      turn 1 {tr['ms']:.0f}ms answer={tr['answer'][:80]!r}", flush=True)
                hard("B1 create_folder ASK surfaced + allowed", tr["approval"] is not None)
                folder_row = await fetch_row(
                    conn,
                    "select id, path, user_id, workspace_id from public.folders "
                    "where user_id = $1 and path = $2", uid, FOLDER)
                hard("B2 create_folder physical row", folder_row is not None,
                     str(dict(folder_row) if folder_row else None))
                if folder_row:
                    print(f"      physical row: id={folder_row['id']} path={folder_row['path']!r} "
                          f"user_id={folder_row['user_id']} workspace_id={folder_row['workspace_id']}",
                          flush=True)
                    hard("B3 root folder (workspace_id IS NULL)", folder_row["workspace_id"] is None)

                # ── ② add_term via the real L0 -> ACTION chain ─────────────────
                print(f"      turn 2 message: {MSG_TERM!r}", flush=True)
                tr = await sse_turn(c, auth, MSG_TERM)
                print(f"      turn 2 {tr['ms']:.0f}ms answer={tr['answer'][:80]!r}", flush=True)
                hard("C1 add_term ASK surfaced + allowed", tr["approval"] is not None)
                term_row = await fetch_row(
                    conn,
                    "select id, word, definition, domain_id from public.terms "
                    "where domain_id = $1 and word = $2", domain_id, TERM)
                hard("C2 add_term physical row", term_row is not None,
                     str(dict(term_row) if term_row else None))
                if term_row:
                    print(f"      physical row: id={term_row['id']} word={term_row['word']!r} "
                          f"definition={term_row['definition']!r} domain_id={term_row['domain_id']}",
                          flush=True)
                    hard("C3 term bound to the fixture domain",
                         str(term_row["domain_id"]) == str(domain_id))
        finally:
            removed = {"terms": 0, "folders": 0, "domains": 0}
            if conn is not None and uid is not None:
                removed = await teardown(conn, uid)
            if conn is not None:
                # Post-cleanup proof: re-query must find nothing of ours.
                left = await conn.fetchval(
                    "select (select count(*) from public.folders where user_id=$1 and path=$2) "
                    "+ (select count(*) from public.terms where word=$3) "
                    "+ (select count(*) from public.domains where user_id=$1 and name=$4)",
                    uid, FOLDER, TERM, DOMAIN) if uid is not None else -1
                print(f"\ncleanup removed: {removed}; rows left for this run: {left}", flush=True)
                hard("Z1 DB unpolluted by this run", left == 0, f"left={left}")
                await conn.close()

    print(f"\nA-2 LIVE {'FAIL: ' + ','.join(FAILS) if FAILS else 'PASS'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
