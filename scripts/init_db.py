"""Initialize the database by applying the migration chain (migrations/*.sql).

Equivalent to init_db inside the FastAPI lifespan, split out for convenient script/CI use.
The runner applies every not-yet-registered migration file in sorted order, so a fresh
install must run the full chain to reach the final schema; ``0001_init.sql`` alone is only
the chain's starting point, not a standalone final schema.
Usage: python scripts/init_db.py
"""
import asyncio

from core.infrastructure.db import init_db


async def main() -> None:
    await init_db()
    print("✓ 数据库已初始化(canonical schema 已应用)")


if __name__ == "__main__":
    asyncio.run(main())
