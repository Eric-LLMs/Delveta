"""Session-history pagination: bounded pages, exact cursors, no duplicates or gaps.

Complements ``test_message_order.py`` (why the order exists) with the paging contract:
the database returns at most ``limit`` rows per page, ``before`` walks strictly older, a
page never splits a timestamp group in the wrong space, and walking a whole transcript
reproduces the unpaginated order exactly.

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
from api.session_history_cursor import CursorError, decode_cursor, encode_cursor
from core.infrastructure.memory import load_session_detail
from tests._pg_fixtures import pg  # noqa: F401  (pytest fixture)

BASE = datetime(2026, 10, 1, tzinfo=timezone.utc)


async def _seed(pg, rows: list[tuple[str, datetime]]) -> tuple[uuid.UUID, uuid.UUID]:
    """Insert ``(role, created_at)`` rows into a fresh session; returns (user_id, session_id)."""
    uid, sid = uuid.uuid4(), uuid.uuid4()
    async with pg.maker() as s:
        await s.execute(
            text("INSERT INTO users (id, username) VALUES (:i, :u)"),
            {"i": uid, "u": f"u_{uid.hex[:12]}"},
        )
        await s.execute(
            text("INSERT INTO sessions (id, user_id, type) VALUES (:i, :u, 0)"),
            {"i": sid, "u": uid},
        )
        if rows:
            await s.execute(
                text(
                    "INSERT INTO messages (id, user_id, session_id, role, text, created_at)"
                    " VALUES (:i, :u, :s, :r, :t, :c)"
                ),
                [
                    {"i": uuid.uuid4(), "u": uid, "s": sid, "r": role, "t": f"{role} {n}", "c": ts}
                    for n, (role, ts) in enumerate(rows)
                ],
            )
        await s.commit()
    return uid, sid


async def _walk(pg, sid, limit: int) -> tuple[list[dict], list[dict]]:
    """Page through the whole transcript; returns (pages newest-first, transcript oldest-first)."""
    pages, cursor = [], None
    for _ in range(500):  # hard stop so a paging bug fails instead of hanging
        detail = await load_session_detail(pg.maker, sid, limit=limit, before=cursor)
        pages.append(detail)
        if not detail["has_more"]:
            break
        assert detail["next_before"] != cursor, "cursor did not advance"
        cursor = detail["next_before"]
        assert cursor is not None
    else:
        raise AssertionError("pagination did not terminate")
    # Pages arrive newest-first; each page is oldest-first internally.
    seen = [m for page in reversed(pages) for m in page["messages"]]
    return pages, seen


# ── page shape ────────────────────────────────────────────────────────────────


async def test_first_page_is_bounded_and_reports_more(pg):
    rows = [("user" if i % 2 == 0 else "assistant", BASE + timedelta(seconds=i)) for i in range(120)]
    _, sid = await _seed(pg, rows)

    detail = await load_session_detail(pg.maker, sid, limit=50)
    assert len(detail["messages"]) == 50
    assert detail["has_more"] is True
    assert detail["next_before"] is not None
    # Newest page: the last 50 of 120.
    assert detail["messages"][-1]["content"] == "assistant 119"
    assert detail["messages"][0]["content"] == "user 70"


async def test_second_page_only_returns_older_messages(pg):
    rows = [("user" if i % 2 == 0 else "assistant", BASE + timedelta(seconds=i)) for i in range(120)]
    _, sid = await _seed(pg, rows)

    first = await load_session_detail(pg.maker, sid, limit=50)
    second = await load_session_detail(pg.maker, sid, limit=50, before=first["next_before"])
    assert len(second["messages"]) == 50
    oldest_first_page = first["messages"][0]["created_at"]
    assert all(m["created_at"] < oldest_first_page for m in second["messages"])
    assert not ({m["id"] for m in first["messages"]} & {m["id"] for m in second["messages"]})


async def test_walk_reproduces_the_unpaginated_transcript(pg):
    # Mixed shapes: unique timestamps, a 3-row legacy tie, and a long gap.
    rows = []
    for i in range(40):
        rows.append(("user" if i % 2 == 0 else "assistant", BASE + timedelta(seconds=i)))
    tie = BASE + timedelta(seconds=40)
    rows += [("user", tie), ("tool", tie), ("assistant", tie)]
    for i in range(40):
        rows.append(("user" if i % 2 == 0 else "assistant", BASE + timedelta(seconds=50 + i)))
    _, sid = await _seed(pg, rows)

    expected = (await load_session_detail(pg.maker, sid))["messages"]
    for limit in (1, 2, 7, 50):
        pages, seen = await _walk(pg, sid, limit)
        for page in pages:
            assert len(page["messages"]) <= limit
        assert [m["id"] for m in seen] == [m["id"] for m in expected], f"limit={limit}"
        assert len({m["id"] for m in seen}) == len(expected)


async def test_timestamp_group_is_never_split_in_the_wrong_order(pg):
    """Groups of 3 identical timestamps straddle the page boundary at several limits."""
    rows = []
    for g in range(30):
        ts = BASE + timedelta(seconds=g)
        rows += [("user", ts), ("tool", ts), ("assistant", ts)]
    _, sid = await _seed(pg, rows)

    expected = [m["id"] for m in (await load_session_detail(pg.maker, sid))["messages"]]
    for limit in (4, 5, 7, 8):
        _, seen = await _walk(pg, sid, limit)
        assert [m["id"] for m in seen] == expected, f"limit={limit}"


async def test_single_group_larger_than_limit_pages_in_display_order(pg):
    """Bulk-imported legacy data: every row shares one timestamp — the page must stay bounded."""
    ts = BASE + timedelta(seconds=1)
    rows = [("user" if i % 2 == 0 else "assistant", ts) for i in range(120)]
    _, sid = await _seed(pg, rows)

    expected = [m["id"] for m in (await load_session_detail(pg.maker, sid))["messages"]]
    pages, seen = await _walk(pg, sid, 7)
    assert all(len(p["messages"]) <= 7 for p in pages)
    assert len(pages) > 1  # it really paged inside the group
    assert [m["id"] for m in seen] == expected
    assert len({m["id"] for m in seen}) == 120


async def test_pagination_is_scoped_to_one_session(pg):
    rows = [("user" if i % 2 == 0 else "assistant", BASE + timedelta(seconds=i)) for i in range(60)]
    _, sid_a = await _seed(pg, rows)
    _, sid_b = await _seed(pg, rows)  # same timestamps, different session

    page_a = (await load_session_detail(pg.maker, sid_a, limit=25))["messages"]
    page_b = (await load_session_detail(pg.maker, sid_b, limit=25))["messages"]
    ids_a = {m["id"] for m in page_a}
    ids_b = {m["id"] for m in page_b}
    assert len(ids_a) == len(ids_b) == 25
    assert ids_a.isdisjoint(ids_b)
    assert all(m["content"] for m in page_a + page_b)


async def test_empty_and_short_sessions_report_no_more(pg):
    _, empty_sid = await _seed(pg, [])
    short_rows = [("user" if i % 2 == 0 else "assistant", BASE + timedelta(seconds=i)) for i in range(30)]
    _, short_sid = await _seed(pg, short_rows)

    empty = await load_session_detail(pg.maker, empty_sid, limit=50)
    assert empty["messages"] == []
    assert empty["has_more"] is False
    assert empty["next_before"] is None

    short = await load_session_detail(pg.maker, short_sid, limit=50)
    assert len(short["messages"]) == 30
    assert short["has_more"] is False
    assert short["next_before"] is None
    assert short["messages"][0]["content"] == "user 0"


async def test_page_queries_are_bounded_by_sql_limit(pg):
    rows = [("user" if i % 2 == 0 else "assistant", BASE + timedelta(seconds=i)) for i in range(80)]
    _, sid = await _seed(pg, rows)

    statements: list[str] = []

    def _capture(conn, cursor, statement, *a):
        statements.append(statement)

    event.listen(pg.engine.sync_engine, "before_cursor_execute", _capture)
    try:
        first = await load_session_detail(pg.maker, sid, limit=10)
        await load_session_detail(pg.maker, sid, limit=10, before=first["next_before"])
    finally:
        event.remove(pg.engine.sync_engine, "before_cursor_execute", _capture)

    # Every newest-first message read is a paged read: it must carry SQL LIMIT. (The count
    # probe and the attachment join are exempt: one is an aggregate, the other is bounded by
    # the page's own ids.)
    ordered_reads = [
        s for s in statements
        if "FROM messages" in s and "DESC" in s.upper() and "LIMIT" not in s.upper()
    ]
    assert not ordered_reads, "\n\n".join(ordered_reads)
    assert any("LIMIT" in s.upper() and "FROM messages" in s for s in statements)
    # The page rows themselves never come back unbounded: no plain session-wide SELECT.
    session_wide = [
        s for s in statements
        if "FROM messages" in s
        and "session_id = " in s
        and "LIMIT" not in s.upper()
        and "count(" not in s.lower()
        and " IN (" not in s.upper()
    ]
    assert not session_wide, "\n\n".join(session_wide)


# ── cursor wire format ────────────────────────────────────────────────────────


def test_cursor_roundtrip_keeps_the_composite_position():
    ts = datetime(2026, 10, 29, 2, 31, 55, 795372, tzinfo=timezone.utc)
    mid = uuid.uuid4()
    for before in ((ts, 0, mid), (ts, None, None)):
        assert decode_cursor(encode_cursor(before)) == before
    assert encode_cursor(None) is None


@pytest.mark.parametrize("raw", ["", "not-base64!!", "MTIz", "MjAyNi0xMC0yOVQwMDowMDowMHwyfA=="])
def test_malformed_cursor_is_rejected(raw):
    with pytest.raises(CursorError):
        decode_cursor(raw)


# ── route: default limit, cursor handling, tenancy ───────────────────────────


def _app(pg, user_id, monkeypatch) -> FastAPI:
    app = FastAPI()
    app.include_router(sessions_module.router)

    async def _backfill(*a, **kw):  # the flag backfill is unrelated to paging
        return None

    monkeypatch.setattr(sessions_module, "SessionLocal", pg.maker)
    monkeypatch.setattr(sessions_module, "_backfill_imported_rag", _backfill)
    app.dependency_overrides[auth_require_user] = lambda: AuthUser(
        user_id=user_id, username="u", display_name="u", role=None, token_id=None
    )
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_route_default_page_is_50_and_walks_with_the_cursor(pg, monkeypatch):
    rows = [("user" if i % 2 == 0 else "assistant", BASE + timedelta(seconds=i)) for i in range(120)]
    uid, sid = await _seed(pg, rows)

    async with _client(_app(pg, uid, monkeypatch)) as client:
        first = (await client.get(f"/sessions/{sid}")).json()
        assert len(first["messages"]) == 50
        assert first["has_more"] is True
        assert first["next_before"]

        second = (
            await client.get(f"/sessions/{sid}", params={"before": first["next_before"]})
        ).json()
        assert len(second["messages"]) == 50
        oldest_first = datetime.fromisoformat(first["messages"][0]["created_at"])
        assert all(
            datetime.fromisoformat(m["created_at"]) < oldest_first for m in second["messages"]
        )

        third = (
            await client.get(
                f"/sessions/{sid}", params={"before": second["next_before"], "limit": 100}
            )
        ).json()
        assert third["has_more"] is False
        assert third["next_before"] is None
        assert len(first["messages"]) + len(second["messages"]) + len(third["messages"]) == 120


async def test_route_rejects_a_bad_cursor_and_another_users_session(pg, monkeypatch):
    rows = [("user", BASE + timedelta(seconds=i)) for i in range(5)]
    uid, sid = await _seed(pg, rows)

    async with _client(_app(pg, uid, monkeypatch)) as client:
        assert (
            await client.get(f"/sessions/{sid}", params={"before": "garbage"})
        ).status_code == 400
        assert (await client.get(f"/sessions/{sid}", params={"limit": 0})).status_code == 422
        assert (await client.get(f"/sessions/{uuid.uuid4()}")).status_code == 404

    async with _client(_app(pg, uuid.uuid4(), monkeypatch)) as stranger:
        assert (await stranger.get(f"/sessions/{sid}")).status_code == 404
