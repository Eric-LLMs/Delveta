"""Message-order regression tests: history reads and the per-turn write batch.

Root cause this guards against: a turn's user + assistant rows are written by ONE
multi-row INSERT in ONE transaction, so ``now()`` — the transaction start time — gave both
rows the SAME ``created_at``. The history read then broke the tie on ``messages.id``, a
random UUID, so the assistant reply could be returned (and rendered) before the question.

Two invariants:
- writes: one batch gets strictly increasing ``created_at`` in VALUES order;
- reads: every conversation query orders by ``(created_at, turn rank, id)``, where the rank
  is user → tool → assistant, which is the real order inside one turn's timestamp.

The real-PostgreSQL module exercises both end to end (and skips when no server is reachable).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.dialects import postgresql

from core.infrastructure.db import MessageModel
from core.infrastructure.memory import (
    SessionMemoryStore,
    _batch_timestamps,
    load_session_detail,
    load_session_messages,
    message_order_by,
)
from tests._pg_fixtures import pg  # noqa: F401  (pytest fixture)
from tests._memory_v2_fakes import Db, factory

# ── ordering expression (no DB) ────────────────────────────────────────────────────


def test_message_order_by_is_created_at_then_turn_rank_then_id():
    sql = str(
        select(MessageModel).order_by(*message_order_by()).compile(dialect=postgresql.dialect())
    )
    order = sql.split("ORDER BY", 1)[1]
    assert order.index("messages.created_at") < order.index("CASE") < order.index("messages.id")


async def test_history_reads_use_the_total_order():
    """Both resume paths order by the turn-aware key, not by raw (created_at, id)."""
    db = Db(rows=[])
    sid = uuid.uuid4()

    await load_session_messages(factory(db), sid)
    await load_session_detail(factory(db), sid)

    message_selects = [
        s for s in db.statements
        if str(s).lower().startswith("select") and "from messages" in str(s).lower()
        and "order by" in str(s).lower()
    ]
    assert len(message_selects) == 2
    for stmt in message_selects:
        order = str(stmt.compile(dialect=postgresql.dialect())).split("ORDER BY", 1)[1]
        assert "CASE" in order  # turn rank resolves same-timestamp rows
        assert order.index("created_at") < order.index("CASE")


def test_batch_timestamps_are_strictly_increasing():
    stamps = _batch_timestamps(3)
    offsets = [s.right.value for s in stamps]
    assert offsets == [
        timedelta(0),
        timedelta(microseconds=1),
        timedelta(microseconds=2),
    ]


async def test_turn_batch_insert_carries_ordered_timestamps():
    """One flush = one INSERT whose rows get now(), now()+1µs, … in VALUES order."""
    db = Db(insert_n=2, insert_roles=["user", "assistant"])
    sid = uuid.uuid4()
    mem = SessionMemoryStore(
        factory(db), embedder=None, llm=None, session_id=sid, user_id=uuid.uuid4()
    )
    await mem.append_message("user", "question")
    await mem.append_message("assistant", "answer")

    await mem.flush_writes()

    inserts = [s for s in db.statements if str(s).lower().startswith("insert into messages")]
    assert len(inserts) == 1  # the turn is still ONE batch INSERT
    compiled = inserts[0].compile(dialect=postgresql.dialect())
    offsets = [v for k, v in compiled.params.items() if k.startswith("now_")]
    assert offsets == [timedelta(0), timedelta(microseconds=1)]


# ── real PostgreSQL: history recovery end to end ──────────────────────────────────


async def _seed_user(maker) -> uuid.UUID:
    uid = uuid.uuid4()
    async with maker() as s:
        await s.execute(
            text("INSERT INTO users (id, username) VALUES (:i, :u)"),
            {"i": uid, "u": f"u_{uid.hex[:12]}"},
        )
        await s.commit()
    return uid


async def _seed_session(maker, user_id: uuid.UUID) -> uuid.UUID:
    sid = uuid.uuid4()
    async with maker() as s:
        await s.execute(
            text("INSERT INTO sessions (id, user_id) VALUES (:i, :u)"),
            {"i": sid, "u": user_id},
        )
        await s.commit()
    return sid


async def test_legacy_tied_turn_reads_user_before_assistant(pg):
    """History written before the write-side fix: equal timestamps, assistant UUID sorts
    first. The read order must still put the question first, and must be stable."""
    user_id = await _seed_user(pg.maker)
    session_id = await _seed_session(pg.maker, user_id)
    ts = datetime.now(timezone.utc)
    # Deliberately inverted UUIDs: the OLD order was (created_at, id), which put the reply
    # first here. This is the exact shape of the reported WorkSpace session.
    question_id = uuid.UUID(int=(1 << 128) - 1)
    answer_id = uuid.UUID(int=1)
    async with pg.maker() as s:
        await s.execute(
            text(
                "INSERT INTO messages (id, user_id, session_id, role, text, created_at)"
                " VALUES (:qid, :uid, :sid, 'user', 'hi, introduce yourself', :ts),"
                "        (:aid, :uid, :sid, 'assistant', 'Hi! I am Delveta.', :ts)"
            ),
            {"qid": question_id, "aid": answer_id, "uid": user_id, "sid": session_id, "ts": ts},
        )
        await s.commit()

    first = await load_session_detail(pg.maker, session_id)
    second = await load_session_detail(pg.maker, session_id)
    assert [m["role"] for m in first["messages"]] == ["user", "assistant"]
    assert [m["content"] for m in first["messages"]] == ["hi, introduce yourself", "Hi! I am Delveta."]
    # Restore is deterministic: a reopen (page refresh) yields the same order.
    assert [m["id"] for m in second["messages"]] == [m["id"] for m in first["messages"]]

    model_ctx = await load_session_messages(pg.maker, session_id)
    assert [m["role"] for m in model_ctx] == ["user", "assistant"]


async def test_write_batch_persists_ordered_history(pg):
    """New turns: one flush of [user, assistant], then a reload puts the question first."""
    user_id = await _seed_user(pg.maker)
    session_id = await _seed_session(pg.maker, user_id)
    mem = SessionMemoryStore(
        pg.maker, embedder=None, llm=None, session_id=session_id, user_id=user_id
    )
    await mem.append_message("user", "question one")
    await mem.append_message("assistant", "answer one")
    written = await mem.flush_writes()
    assert [r["role"] for r in written] == ["user", "assistant"]

    detail = await load_session_detail(pg.maker, session_id)
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    stamps = [datetime.fromisoformat(m["created_at"]) for m in detail["messages"]]
    assert stamps[0] < stamps[1]  # a real intra-turn order, not a UUID coin flip


async def test_consecutive_turns_keep_order_without_duplicates(pg):
    """Two sends (two flushes) + two reloads: u1,a1,u2,a2 — no duplication, no loss."""
    user_id = await _seed_user(pg.maker)
    session_id = await _seed_session(pg.maker, user_id)
    mem = SessionMemoryStore(
        pg.maker, embedder=None, llm=None, session_id=session_id, user_id=user_id
    )
    for q, a in (("q1", "a1"), ("q2", "a2")):
        await mem.append_message("user", q)
        await mem.append_message("assistant", a)
        await mem.flush_writes()

    detail = await load_session_detail(pg.maker, session_id)
    roles = [m["role"] for m in detail["messages"]]
    contents = [m["content"] for m in detail["messages"]]
    ids = [m["id"] for m in detail["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert contents == ["q1", "a1", "q2", "a2"]
    assert len(set(ids)) == 4  # nothing merged or duplicated

    reopened = await load_session_detail(pg.maker, session_id)
    assert [m["id"] for m in reopened["messages"]] == ids
