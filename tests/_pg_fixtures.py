"""Throwaway-PostgreSQL plumbing shared by the integration test modules.

Every fixture here creates a **brand-new database** on the server configured in
``settings.database_url``, applies the migration chain, and drops the database afterwards —
the dev database is never touched (its URL only supplies server + credentials). The ``pg``
fixture skips the calling module when no server is reachable.

Imported fixtures: a module gets one by importing it, e.g.
``from tests._pg_fixtures import pg  # noqa: F401``.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import settings
from core.infrastructure.db import init_db
from core.infrastructure.storage import LocalStorage

MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


def _admin_dsn() -> str:
    base = make_url(settings.database_url)
    # ``str(URL)`` masks the password; render explicitly for the asyncpg DSN.
    return (
        base.set(database="postgres")
        .render_as_string(hide_password=False)
        .replace("postgresql+asyncpg://", "postgresql://", 1)
    )


def _test_url(dbname: str) -> str:
    return (
        make_url(settings.database_url)
        .set(database=dbname)
        .render_as_string(hide_password=False)
    )


def _asyncpg_test_dsn(dbname: str) -> str:
    return _test_url(dbname).replace("postgresql+asyncpg://", "postgresql://", 1)


async def _create_database(dbname: str) -> None:
    import asyncpg

    admin = await asyncpg.connect(_admin_dsn(), timeout=5)
    try:
        await admin.execute(f'CREATE DATABASE "{dbname}"')
    finally:
        await admin.close()


async def _drop_database(dbname: str) -> None:
    import asyncpg

    admin = await asyncpg.connect(_admin_dsn(), timeout=5)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{dbname}" WITH (FORCE)')
    finally:
        await admin.close()


async def _apply_files(conn, files: list[Path]) -> None:
    """Apply migration files in order, restoring ``search_path`` per file (mirrors init_db)."""
    for path in files:
        async with conn.transaction():
            await conn.execute("SELECT pg_catalog.set_config('search_path', 'public', false)")
            await conn.execute(path.read_text(encoding="utf-8"))


@pytest.fixture
async def pg(tmp_path):
    """A freshly migrated throwaway database + engine/sessionmaker (full chain applied)."""
    import asyncpg

    try:
        probe = await asyncpg.connect(_admin_dsn(), timeout=5)
        await probe.close()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    dbname = f"delveta_test_{uuid.uuid4().hex[:12]}"
    await _create_database(dbname)
    engine = create_async_engine(_test_url(dbname), pool_size=5)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        await init_db(_test_url(dbname))
        yield _PgNamespace(engine=engine, maker=maker, storage=LocalStorage(tmp_path))
    finally:
        await engine.dispose()
        await _drop_database(dbname)


class _PgNamespace:
    def __init__(self, *, engine, maker, storage):
        self.engine = engine
        self.maker = maker
        self.storage = storage
