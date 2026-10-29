"""Session-list pagination: bounded pages, exact cursors, pinned batches.

Complements ``test_history_pagination.py`` (messages inside one session) with the
sidebar's list contract: ``GET /sessions`` returns at most ``limit`` sessions, ``before``
walks strictly older, equal timestamps neither repeat nor skip a row, search and the lane
filter page the same way, and ``?ids=`` is bounded and owner-scoped.

Real-PostgreSQL module — it asserts true SQL behaviour, not a fake's.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import event, text

from api.auth import AuthUser
from api.auth import require_user as auth_require_user
from api.routers import sessions as sessions_module
from api.session_list_cursor import (
    SessionCursorError,
    decode_session_cursor,
    encode_session_cursor,
)
from core.infrastructure.memory import (
    list_sessions,
    list_sessions_by_ids,
    list_sessions_page,
)
from tests._pg_fixtures import pg  # noqa: F401  (pytest fixture)

BASE = datetime(2026, 10, 1, tzinfo=timezone.utc)


async def _user(pg) -> uuid.UUID:
    uid = uuid.uuid4()
    async with pg.maker() as s:
        await s.execute(
            text("INSERT INTO users (id, username) VALUES (:i, :u)"),
            {"i": uid, "u": f"u_{uid.hex[:12]}"},
        )
        await s.commit()
    return uid


async def _sessions(pg, uid, rows: list[tuple[datetime, int]]) -> list[uuid.UUID]:
    """Insert ``(created_at, type)`` sessions; returns their ids in insertion order."""
    ids = [uuid.uuid4() for _ in rows]
    if rows:
        async with pg.maker() as s:
            await s.execute(
                text(
                    "INSERT INTO sessions (id, user_id, type, created_at, title)"
                    " VALUES (:i, :u, :t, :c, :h)"
                ),
                [
                    {"i": sid, "u": uid, "t": type_, "c": ts, "h": f"s{n}"}
                    for n, (sid, (ts, type_)) in enumerate(zip(ids, rows))
                ],
            )
            await s.commit()
    return ids


async def _walk(pg, uid, limit: int, *, q=None, kind="chat") -> tuple[list[dict], list[dict]]:
    """Page through every session; returns (pages, rows in list order — newest first)."""
    pages, cursor = [], None
    for _ in range(500):  # hard stop so a paging bug fails instead of hanging
        rows, has_more = await list_sessions_page(
            pg.maker, uid, q, kind=kind, limit=limit, before=cursor
        )
        pages.append({"sessions": rows, "has_more": has_more, "before": cursor})
        if not has_more:
            break
        oldest = rows[-1]
        cursor = (datetime.fromisoformat(oldest["created_at"]), uuid.UUID(oldest["id"]))
    else:
        raise AssertionError("pagination did not terminate")
    # Pages arrive newest-first and each page is ordered newest-first, so concatenating in
    # arrival order reproduces the unpaginated list exactly.
    seen = [row for page in pages for row in page["sessions"]]
    return pages, seen


# ── page shape ────────────────────────────────────────────────────────────────


async def test_first_page_is_bounded_and_reports_more(pg):
    uid = await _user(pg)
    await _sessions(pg, uid, [(BASE + timedelta(seconds=i), 0) for i in range(120)])

    rows, has_more = await list_sessions_page(pg.maker, uid, limit=50)
    assert len(rows) == 50
    assert has_more is True
    assert rows[0]["title"] == "s119"  # newest first
    assert rows[-1]["title"] == "s70"


async def test_second_page_is_strictly_older_and_disjoint(pg):
    uid = await _user(pg)
    await _sessions(pg, uid, [(BASE + timedelta(seconds=i), 0) for i in range(120)])

    first, _ = await list_sessions_page(pg.maker, uid, limit=50)
    oldest = first[-1]
    second, _ = await list_sessions_page(
        pg.maker, uid, limit=50, before=(datetime.fromisoformat(oldest["created_at"]),
                                         uuid.UUID(oldest["id"]))
    )
    assert len(second) == 50
    assert all(
        (row["created_at"], row["id"]) < (oldest["created_at"], oldest["id"])
        for row in second
    )
    assert not ({r["id"] for r in first} & {r["id"] for r in second})


async def test_walk_reproduces_the_unpaginated_list(pg):
    uid = await _user(pg)
    await _sessions(pg, uid, [(BASE + timedelta(seconds=i), 0) for i in range(60)])

    expected = [s["id"] for s in await list_sessions(pg.maker, uid)]
    for limit in (1, 2, 7, 50):
        pages, seen = await _walk(pg, uid, limit)
        assert all(len(p["sessions"]) <= limit for p in pages)
        assert [s["id"] for s in seen] == expected, f"limit={limit}"
        assert len({s["id"] for s in seen}) == len(expected)


async def test_equal_timestamps_page_without_gaps_or_repeats(pg):
    """Sessions created in one transaction share created_at — id is the tie-break."""
    uid = await _user(pg)
    ts = BASE + timedelta(seconds=5)
    await _sessions(pg, uid, [(ts, 0) for _ in range(60)])

    expected = [s["id"] for s in await list_sessions(pg.maker, uid)]
    for limit in (1, 7, 13):
        pages, seen = await _walk(pg, uid, limit)
        assert all(len(p["sessions"]) <= limit for p in pages)
        assert [s["id"] for s in seen] == expected, f"limit={limit}"
        assert len({s["id"] for s in seen}) == 60


async def test_last_page_reports_no_more_and_empty_list_is_clean(pg):
    uid = await _user(pg)
    await _sessions(pg, uid, [(BASE + timedelta(seconds=i), 0) for i in range(30)])

    rows, has_more = await list_sessions_page(pg.maker, uid, limit=50)
    assert len(rows) == 30
    assert has_more is False

    empty_uid = await _user(pg)
    rows, has_more = await list_sessions_page(pg.maker, empty_uid, limit=50)
    assert rows == []
    assert has_more is False


# ── filters ───────────────────────────────────────────────────────────────────


async def test_search_pages_and_keeps_the_filter(pg):
    uid = await _user(pg)
    rows = []
    for i in range(40):
        ts = BASE + timedelta(seconds=i)
        rows.append((ts, 0))
    ids = await _sessions(pg, uid, rows)
    # Match by title on every third session, and one by message text.
    async with pg.maker() as s:
        await s.execute(
            text("UPDATE sessions SET title = 'alpha ' || title WHERE id = ANY(:ids)"),
            {"ids": [str(ids[i]) for i in range(0, 40, 3)]},
        )
        await s.execute(
            text(
                "INSERT INTO messages (id, user_id, session_id, role, text, created_at)"
                " VALUES (:i, :u, :s, 'user', 'alpha in the body', :c)"
            ),
            {"i": uuid.uuid4(), "u": uid, "s": ids[1], "c": BASE},
        )
        await s.commit()

    expected = [s["id"] for s in await list_sessions(pg.maker, uid, "alpha")]
    assert len(expected) == 15  # 14 title matches (every third of 40) + 1 body-only match
    for limit in (1, 3, 50):
        pages, seen = await _walk(pg, uid, limit, q="alpha")
        assert all(len(p["sessions"]) <= limit for p in pages)
        assert [s["id"] for s in seen] == expected, f"limit={limit}"
        assert all(s.get("snippet") for s in seen if s["id"] == str(ids[1]))


async def test_lane_filter_keeps_lanes_disjoint(pg):
    uid = await _user(pg)
    await _sessions(
        pg, uid, [(BASE + timedelta(seconds=i), 0) for i in range(10)]
        + [(BASE + timedelta(seconds=100 + i), 1) for i in range(3)]
        + [(BASE + timedelta(seconds=200 + i), 2) for i in range(4)]
    )

    chat, _ = await list_sessions_page(pg.maker, uid, kind="chat", limit=50)
    ws, _ = await list_sessions_page(pg.maker, uid, kind="workspace", limit=50)
    assert {s["type"] for s in chat} == {0}
    assert {s["type"] for s in ws} == {2}
    assert not ({s["id"] for s in chat} & {s["id"] for s in ws})
    with pytest.raises(ValueError):
        await list_sessions_page(pg.maker, uid, kind="research")


async def test_pagination_is_scoped_to_one_user(pg):
    alice, bob = await _user(pg), await _user(pg)
    await _sessions(pg, alice, [(BASE + timedelta(seconds=i), 0) for i in range(20)])
    await _sessions(pg, bob, [(BASE + timedelta(seconds=i), 0) for i in range(20)])

    a, _ = await list_sessions_page(pg.maker, alice, limit=10)
    b, _ = await list_sessions_page(pg.maker, bob, limit=10)
    assert len(a) == len(b) == 10
    assert not ({s["id"] for s in a} & {s["id"] for s in b})


# ── pinned batch ──────────────────────────────────────────────────────────────


async def test_ids_batch_returns_only_the_owners_rows(pg):
    alice, bob = await _user(pg), await _user(pg)
    a_ids = await _sessions(pg, alice, [(BASE + timedelta(seconds=i), 0) for i in range(5)])
    b_ids = await _sessions(pg, bob, [(BASE + timedelta(seconds=i), 0) for i in range(5)])

    wanted = [a_ids[0], a_ids[3], b_ids[1], uuid.uuid4()]
    rows = await list_sessions_by_ids(pg.maker, alice, wanted)
    assert {s["id"] for s in rows} == {str(a_ids[0]), str(a_ids[3])}


async def test_ids_batch_is_scoped_to_the_lane(pg):
    uid = await _user(pg)
    ids = await _sessions(pg, uid, [(BASE, 0), (BASE + timedelta(seconds=1), 2)])

    chat = await list_sessions_by_ids(pg.maker, uid, ids, kind="chat")
    ws = await list_sessions_by_ids(pg.maker, uid, ids, kind="workspace")
    assert [s["id"] for s in chat] == [str(ids[0])]
    assert [s["id"] for s in ws] == [str(ids[1])]


# ── SQL shape ────────────────────────────────────────────────────────────────


async def test_page_queries_carry_a_sql_limit(pg):
    uid = await _user(pg)
    await _sessions(pg, uid, [(BASE + timedelta(seconds=i), 0) for i in range(80)])

    statements: list[str] = []

    def _capture(conn, cursor, statement, *a):
        statements.append(statement)

    event.listen(pg.engine.sync_engine, "before_cursor_execute", _capture)
    try:
        first, _ = await list_sessions_page(pg.maker, uid, limit=10)
        await list_sessions_page(
            pg.maker,
            uid,
            limit=10,
            before=(datetime.fromisoformat(first[-1]["created_at"]), uuid.UUID(first[-1]["id"])),
        )
    finally:
        event.remove(pg.engine.sync_engine, "before_cursor_execute", _capture)

    session_reads = [s for s in statements if "FROM sessions" in s]
    assert session_reads
    assert all("LIMIT" in s.upper() or " IN (" in s.upper() for s in session_reads), "\n\n".join(
        session_reads
    )


async def test_legacy_list_sessions_still_reads_every_row(pg):
    uid = await _user(pg)
    await _sessions(pg, uid, [(BASE + timedelta(seconds=i), 0) for i in range(120)])
    assert len(await list_sessions(pg.maker, uid)) == 120


# ── cursor wire format ────────────────────────────────────────────────────────


def test_cursor_roundtrip_keeps_the_composite_position():
    ts = datetime(2026, 10, 29, 2, 31, 55, 795372, tzinfo=timezone.utc)
    sid = uuid.uuid4()
    assert decode_session_cursor(encode_session_cursor((ts, sid))) == (ts, sid)
    assert encode_session_cursor(None) is None


@pytest.mark.parametrize("raw", ["", "not-base64!!", "MTIz", "MjAyNi0xMC0yOVQwMDowMDowMHw="])
def test_malformed_cursor_is_rejected(raw):
    with pytest.raises(SessionCursorError):
        decode_session_cursor(raw)


# ── route ────────────────────────────────────────────────────────────────────


def _app(pg, user_id, monkeypatch) -> FastAPI:
    app = FastAPI()
    app.include_router(sessions_module.router)
    monkeypatch.setattr(sessions_module, "SessionLocal", pg.maker)
    app.dependency_overrides[auth_require_user] = lambda: AuthUser(
        user_id=user_id, username="u", display_name="u", role=None, token_id=None
    )
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_route_default_page_is_50_and_walks_with_the_cursor(pg, monkeypatch):
    uid = await _user(pg)
    await _sessions(pg, uid, [(BASE + timedelta(seconds=i), 0) for i in range(120)])

    async with _client(_app(pg, uid, monkeypatch)) as client:
        first = (await client.get("/sessions")).json()
        assert len(first["sessions"]) == 50
        assert first["has_more"] is True
        assert first["next_before"]

        second = (
            await client.get("/sessions", params={"before": first["next_before"]})
        ).json()
        assert len(second["sessions"]) == 50
        oldest = first["sessions"][-1]
        assert all(
            (s["created_at"], s["id"]) < (oldest["created_at"], oldest["id"])
            for s in second["sessions"]
        )

        third = (
            await client.get(
                "/sessions", params={"before": second["next_before"], "limit": 100}
            )
        ).json()
        assert third["has_more"] is False
        assert third["next_before"] is None
        assert len(first["sessions"]) + len(second["sessions"]) + len(third["sessions"]) == 120


async def test_route_ids_batch_is_capped_validated_and_owner_scoped(pg, monkeypatch):
    alice, bob = await _user(pg), await _user(pg)
    a_ids = await _sessions(pg, alice, [(BASE + timedelta(seconds=i), 0) for i in range(5)])
    b_ids = await _sessions(pg, bob, [(BASE + timedelta(seconds=i), 0) for i in range(5)])

    async with _client(_app(pg, alice, monkeypatch)) as client:
        got = (
            await client.get("/sessions", params={"ids": f"{a_ids[0]},{b_ids[0]}"})
        ).json()
        assert [s["id"] for s in got["sessions"]] == [str(a_ids[0])]
        assert got["has_more"] is False

        many = ",".join(str(uuid.uuid4()) for _ in range(51))
        assert (await client.get("/sessions", params={"ids": many})).status_code == 400
        assert (await client.get("/sessions", params={"ids": "nope"})).status_code == 400
        assert (
            await client.get(
                "/sessions", params={"ids": str(a_ids[0]), "before": "x"}
            )
        ).status_code == 400


async def test_route_rejects_a_bad_cursor_and_bad_limit(pg, monkeypatch):
    uid = await _user(pg)
    await _sessions(pg, uid, [(BASE + timedelta(seconds=i), 0) for i in range(5)])

    async with _client(_app(pg, uid, monkeypatch)) as client:
        assert (await client.get("/sessions", params={"before": "garbage"})).status_code == 400
        assert (await client.get("/sessions", params={"limit": 0})).status_code == 422
        assert (await client.get("/sessions", params={"limit": 201})).status_code == 422
        assert (await client.get("/sessions", params={"kind": "research"})).status_code == 400
