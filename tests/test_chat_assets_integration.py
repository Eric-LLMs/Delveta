"""Chat-asset CAS/GC integration tests against a REAL PostgreSQL (design §9).

These are the concurrency/crash/upgrade proofs the fake-backed ``test_chat_assets.py`` cannot
give: the advisory-lock protocol, file-first crash recovery, the reference recheck under
``ref_count`` drift, the logical-delete → background-GC cycle, and a from-scratch migration of
the whole chain. Every test runs against a **throwaway database** created on the configured
server and dropped afterwards — the dev database is never touched (its URL only supplies the
server + credentials). The module skips when no server is reachable.

Run directly:  python -m pytest tests/test_chat_assets_integration.py -q
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.application.chat_asset_service import ChatAssetService
from core.infrastructure.db import (
    AssetModel,
    ChatAttachmentModel,
    GlobalObjectModel,
)
from core.infrastructure.drive_repositories import (
    SqlAssetAclRepository,
    SqlAssetRepository,
    SqlChatAttachmentRepository,
    SqlGlobalObjectRepository,
)
from core.infrastructure.storage import LocalStorage, object_key
# Throwaway-database plumbing lives in a shared module so other integration modules
# (e.g. message-order recovery) use the same isolated fixture. ``pg`` is a pytest
# fixture imported into this module's namespace.
from tests._pg_fixtures import (  # noqa: F401
    MIGRATIONS,
    _PgNamespace,
    _admin_dsn,
    _apply_files,
    _asyncpg_test_dsn,
    _create_database,
    _drop_database,
    _test_url,
    pg,
)


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


async def _seed_message(maker, user_id, session_id) -> uuid.UUID:
    mid = uuid.uuid4()
    async with maker() as s:
        await s.execute(
            text(
                "INSERT INTO messages (id, user_id, session_id, role, text)"
                " VALUES (:i, :u, :s, 'user', 'hi')"
            ),
            {"i": mid, "u": user_id, "s": session_id},
        )
        await s.commit()
    return mid


def _objects(pg) -> SqlGlobalObjectRepository:
    return SqlGlobalObjectRepository(pg.maker, storage=pg.storage, engine=pg.engine)


# ── 1. fresh install runs the whole chain ──────────────────────────────────────────

async def test_fresh_database_applies_full_migration_chain(pg):
    """An empty database migrates end to end (0001–0024) and matches the ORM (design §10.5)."""
    async with pg.engine.connect() as conn:
        applied = set(
            (await conn.execute(text("SELECT version FROM schema_migrations"))).scalars().all()
        )
    expected = {p.stem for p in MIGRATIONS.glob("*.sql")}
    assert applied == expected, f"missing: {expected - applied}, extra: {applied - expected}"
    assert "0024_chat_assets" in applied

    # ORM models resolve against the migrated schema (column/type agreement).
    async with pg.maker() as s:
        await s.execute(select(AssetModel).limit(1))
        await s.execute(select(ChatAttachmentModel).limit(1))
        await s.execute(select(GlobalObjectModel).limit(1))
    # 0024's guard removed the retired column.
    async with pg.engine.connect() as conn:
        col = await conn.execute(
            text(
                "SELECT 1 FROM information_schema.columns"
                " WHERE table_name='messages' AND column_name='attach_asset_id'"
            )
        )
        assert col.first() is None


# ── 2. create/increment ∥ GC on the same SHA (Invariant A / I1, I3) ────────────────

async def test_same_sha_reupload_racing_gc_never_yields_a_live_row_without_file(pg):
    objects = _objects(pg)
    for _ in range(20):
        sha = uuid.uuid4().hex
        data = b"payload-%s" % sha.encode()
        key = object_key(sha)
        async with pg.maker() as s:
            s.add(
                GlobalObjectModel(
                    sha256=sha, size=len(data), storage_key=key, mime_type="application/octet-stream",
                    ref_count=0, state="pending_delete", pending_since=datetime.now(timezone.utc),
                )
            )
            await s.commit()
        await pg.storage.put(key, data)  # GC-reclaimable object, file present

        # GC and a same-SHA re-upload race for the object.
        await asyncio.gather(
            objects.collect(sha),
            objects.upsert_and_increment(sha, len(data), key, "application/octet-stream", data=data),
        )

        row = await objects.get(sha)
        assert row is not None, "GC deleted the object the upload just revived"
        assert row.state == "live" and row.ref_count == 1
        assert await pg.storage.get(key) == data, "live row with a missing file (Invariant A)"


# ── 3. GC crash recovery (Invariant D) ─────────────────────────────────────────────

async def test_gc_recovers_when_crashed_after_file_delete(pg):
    objects = _objects(pg)
    sha, data, key = uuid.uuid4().hex, b"crashed-after", None
    key = object_key(sha)
    async with pg.maker() as s:
        s.add(GlobalObjectModel(sha256=sha, size=len(data), storage_key=key, ref_count=1, state="live"))
        await s.commit()
    await pg.storage.put(key, data)

    await objects.free(sha)                 # → pending_delete
    await pg.storage.delete(key)            # simulate the crash point AFTER the file op

    result = await objects.sweep_pending(grace_seconds=0)
    assert result["collected"] == 1
    async with pg.maker() as s:
        assert await s.get(GlobalObjectModel, sha) is None


async def test_gc_recovers_when_crashed_before_file_delete(pg):
    objects = _objects(pg)
    sha, data = uuid.uuid4().hex, b"crashed-before"
    key = object_key(sha)
    async with pg.maker() as s:
        s.add(GlobalObjectModel(sha256=sha, size=len(data), storage_key=key, ref_count=1, state="live"))
        await s.commit()
    await pg.storage.put(key, data)

    await objects.free(sha)                 # → pending_delete, file still present

    result = await objects.sweep_pending(grace_seconds=0)
    assert result["collected"] == 1
    assert await pg.storage.get(key) is None
    async with pg.maker() as s:
        assert await s.get(GlobalObjectModel, sha) is None


# ── 4. ref_count drift must not mis-collect a shared object (§9.6, Invariant B) ─────

async def test_collect_refuses_to_delete_object_a_live_asset_still_references(pg):
    objects = _objects(pg)
    user_id = await _seed_user(pg.maker)
    sha, data = uuid.uuid4().hex, b"shared"
    key = object_key(sha)
    async with pg.maker() as s:
        s.add(GlobalObjectModel(sha256=sha, size=len(data), storage_key=key, ref_count=1, state="live"))
        await s.commit()
    await pg.storage.put(key, data)

    # A live drive asset references the object...
    async with pg.maker() as s:
        s.add(AssetModel(user_id=user_id, name="drift.txt", scope="drive", object_sha256=sha, file_status="READY"))
        await s.commit()
    # ...but the object shows ref_count=0 / pending_delete (deliberate drift).
    async with pg.maker() as s:
        obj = await s.get(GlobalObjectModel, sha)
        obj.ref_count = 0
        obj.state = "pending_delete"
        obj.pending_since = datetime.now(timezone.utc)
        await s.commit()

    assert await objects.collect(sha) is None            # refused
    row = await objects.get(sha)
    assert row is not None and row.state == "live"       # revived by the recheck
    assert await pg.storage.get(key) == data             # bytes untouched


# ── 5. session delete: owned retired logically, reference dropped (§8) ─────────────

async def test_session_delete_retires_owned_logically_and_gc_reclaims_later(pg):
    objects = _objects(pg)
    assets = SqlAssetRepository(pg.maker)
    attachments = SqlChatAttachmentRepository(pg.maker)
    acl = SqlAssetAclRepository(pg.maker)
    svc = ChatAssetService(
        pg.maker, storage=pg.storage, objects=objects, assets=assets,
        attachments=attachments, acl=acl,
    )

    user_id = await _seed_user(pg.maker)
    session_id = await _seed_session(pg.maker, user_id)
    message_id = await _seed_message(pg.maker, user_id, session_id)

    owned = await svc.create_owned_draft(user_id, "shot.png", b"owned-bytes", "image/png")
    owned_asset_id = owned.asset_id
    sha = (await assets.get(owned_asset_id)).object_sha256
    await attachments.bind([owned.id], user_id, session_id, message_id)

    drive_asset = await assets.create(user_id, "doc.txt", scope="drive", file_status="READY")
    ref = await svc.mount_reference_draft(user_id, drive_asset.id)
    await attachments.bind([ref.id], user_id, session_id, message_id)

    await svc.delete_for_session(session_id)

    # Owned: LOGICAL delete — row present but soft-deleted; object pending, file still there.
    owned_row = await assets.get(owned_asset_id)
    assert owned_row is not None and owned_row.deleted_at is not None
    assert await pg.storage.get(object_key(sha)) == b"owned-bytes"
    # Reference: link gone, drive asset untouched.
    assert await attachments.get(ref.id) is None
    assert await assets.get(drive_asset.id) is not None

    # Background GC after the grace window does the physics.
    result = await svc.gc(grace_seconds=0)
    assert result["assets_reaped"] == 1
    assert await assets.get(owned_asset_id) is None
    assert await pg.storage.get(object_key(sha)) is None
    async with pg.maker() as s:
        assert await s.get(GlobalObjectModel, sha) is None


# ── 5b. viewer reference persist→restore: card survives reopen, ACL re-checked (§6.3) ──

async def test_viewer_reference_survives_reload_and_rechecks_acl(pg):
    """A viewer-mounted reference binds in the message tx and comes back on reopen (§6.3).

    Persist → restore (scenario 5): ``load_session_detail`` rebuilds the bubble from the
    ``chat_attachments`` authority table — never a transient in-memory chip — so a reopened
    session renders the attachment card. Reading that reference re-checks the drive ACL at
    read time, so revoking the grant revokes chat access too (§5.2) while the bytes are
    untouched (a reference frees no drive asset).
    """
    from core.infrastructure.memory import load_session_detail

    objects = _objects(pg)
    assets = SqlAssetRepository(pg.maker)
    attachments = SqlChatAttachmentRepository(pg.maker)
    acl = SqlAssetAclRepository(pg.maker)
    svc = ChatAssetService(
        pg.maker, storage=pg.storage, objects=objects, assets=assets,
        attachments=attachments, acl=acl,
    )

    owner_id = await _seed_user(pg.maker)
    viewer_id = await _seed_user(pg.maker)          # the user asking about the doc
    session_id = await _seed_session(pg.maker, viewer_id)
    message_id = await _seed_message(pg.maker, viewer_id, session_id)

    # A drive file with real bytes, owned by ``owner`` and shared read to ``viewer``.
    sha, data = uuid.uuid4().hex, b"on-screen-pdf-bytes"
    key = object_key(sha)
    async with pg.maker() as s:
        s.add(GlobalObjectModel(sha256=sha, size=len(data), storage_key=key,
                                mime_type="application/pdf", ref_count=1, state="live"))
        await s.commit()
    await pg.storage.put(key, data)
    doc = await assets.create(
        owner_id, "on-screen.pdf", scope="drive", file_status="READY",
        object_sha256=sha, mime_type="application/pdf",
    )
    await acl.grant(doc.id, viewer_id, "read")

    # Open the doc + explicitly name it → mount a reference draft, bound at Send (same tx).
    ref = await svc.mount_reference_draft(viewer_id, doc.id, source="viewer")
    assert ref is not None and ref.ownership == "reference"
    await attachments.bind([ref.id], viewer_id, session_id, message_id)

    # Reopen: the card is rebuilt from the authority table, not a live viewer chip.
    detail = await load_session_detail(pg.maker, session_id)
    assert len(detail["messages"]) == 1
    assert detail["messages"][0]["attach"] == {
        "asset_id": str(doc.id), "name": "on-screen.pdf", "mime": "application/pdf",
    }

    # Readable while the shared grant holds…
    assert await svc.read_bytes(ref.id, session_id) == data

    # …but a revoked grant revokes chat access at read time (§5.2 recheck); bytes intact.
    await acl.revoke(doc.id, viewer_id)
    assert await svc.read_bytes(ref.id, session_id) is None
    assert await pg.storage.get(key) == data              # a reference frees no drive asset
    assert await assets.get_active(doc.id) is not None


# ── 6. legacy upgrade: bidirectional carry, atomic rollback on failure (§10.3) ─────

@pytest.fixture
async def pg_pre0024(tmp_path):
    """A throwaway database migrated only through 0023 (the pre-chat-assets schema)."""
    import asyncpg

    try:
        probe = await asyncpg.connect(_admin_dsn(), timeout=5)
        await probe.close()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    dbname = f"delveta_test_{uuid.uuid4().hex[:12]}"
    await _create_database(dbname)
    engine = create_async_engine(_test_url(dbname), pool_size=5)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    conn = await asyncpg.connect(_asyncpg_test_dsn(dbname))
    try:
        files = sorted(
            p for p in MIGRATIONS.glob("*.sql") if p.stem < "0024_chat_assets"
        )
        await _apply_files(conn, files)
        yield _PgNamespace(engine=engine, maker=maker, storage=LocalStorage(tmp_path)), conn
    finally:
        await conn.close()
        await engine.dispose()
        await _drop_database(dbname)


async def test_legacy_attachment_upgrade_carries_rows(pg_pre0024):
    """0024 carries classifiable legacy attachments into chat_attachments, then drops the col."""
    ns, conn = pg_pre0024
    user_id, asset_id, session_id, message_id = (uuid.uuid4() for _ in range(4))
    await conn.execute("INSERT INTO users (id, username) VALUES ($1, $2)", user_id, f"u_{user_id.hex[:8]}")
    await conn.execute("INSERT INTO sessions (id, user_id) VALUES ($1, $2)", session_id, user_id)
    # Pre-0024 assets have no ``scope`` column; 0024 adds it defaulting to 'drive', so every
    # carried legacy attachment is a 'reference' (an 'owned' chat asset cannot pre-exist 0024).
    await conn.execute(
        "INSERT INTO assets (id, user_id, name, file_status) VALUES ($1, $2, 'a.bin', 'READY')",
        asset_id, user_id,
    )
    await conn.execute(
        "INSERT INTO messages (id, user_id, session_id, role, text, attach_asset_id)"
        " VALUES ($1, $2, $3, 'user', 'hi', $4)",
        message_id, user_id, session_id, asset_id,
    )

    await _apply_files(conn, [MIGRATIONS / "0024_chat_assets.sql"])

    row = await conn.fetchrow(
        "SELECT asset_id, user_id, session_id, message_id, ownership, status FROM chat_attachments"
    )
    assert row["asset_id"] == asset_id
    assert row["message_id"] == message_id
    assert row["ownership"] == "reference"   # carried from a (defaulted) scope='drive' asset
    assert row["status"] == "bound"
    assert await conn.fetchval(
        "SELECT 1 FROM information_schema.columns"
        " WHERE table_name='messages' AND column_name='attach_asset_id'"
    ) is None


async def test_legacy_attachment_upgrade_rolls_back_on_unclassifiable_row(pg_pre0024):
    """An unclassifiable legacy attachment aborts 0024 atomically — nothing is dropped."""
    ns, conn = pg_pre0024
    owner_id, other_id, session_id, message_id, asset_id = (uuid.uuid4() for _ in range(5))
    await conn.execute("INSERT INTO users (id, username) VALUES ($1, $2)", owner_id, f"u_{owner_id.hex[:8]}")
    await conn.execute("INSERT INTO users (id, username) VALUES ($1, $2)", other_id, f"u_{other_id.hex[:8]}")
    await conn.execute("INSERT INTO sessions (id, user_id) VALUES ($1, $2)", session_id, other_id)
    # A cross-user attachment: the asset exists (FK is satisfied) but belongs to another user,
    # which 0024 classifies as unclassifiable — it must abort rather than guess.
    await conn.execute(
        "INSERT INTO assets (id, user_id, name, file_status) VALUES ($1, $2, 'a.bin', 'READY')",
        asset_id, owner_id,
    )
    await conn.execute(
        "INSERT INTO messages (id, user_id, session_id, role, text, attach_asset_id)"
        " VALUES ($1, $2, $3, 'user', 'hi', $4)",
        message_id, other_id, session_id, asset_id,
    )

    with pytest.raises(Exception):
        await _apply_files(conn, [MIGRATIONS / "0024_chat_assets.sql"])

    # Rolled back: the column survives and no partial table was left behind.
    assert await conn.fetchval(
        "SELECT 1 FROM information_schema.columns"
        " WHERE table_name='messages' AND column_name='attach_asset_id'"
    ) == 1
    assert await conn.fetchval("SELECT to_regclass('public.chat_attachments')") is None
