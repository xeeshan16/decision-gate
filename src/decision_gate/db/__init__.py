"""Persistence: schema, engine and repositories."""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool, StaticPool

from decision_gate.db.models import (
    Base,
    CalibratorRecord,
    DecisionRecord,
    DriftCheckRecord,
    OutcomeRecord,
    ReviewItemRecord,
)


def make_engine(url: str, *, single_connection: bool = False) -> AsyncEngine:
    """Create an async engine. Nothing connects until the first query.

    ``single_connection`` keeps one SQLite connection open for the engine's life instead of
    opening one per session. That is much faster for a long sequential job on one event
    loop (the simulator), and wrong for anything concurrent. PostgreSQL ignores it.
    """
    if url.startswith("sqlite"):
        # SQLite is for tests and quick local runs. A fresh connection per use keeps it
        # safe across the event loops that test clients create.
        engine = create_async_engine(url, poolclass=StaticPool if single_connection else NullPool)
        # SQLite ignores foreign keys unless asked, per connection; PostgreSQL always checks.
        event.listen(engine.sync_engine, "connect", _sqlite_foreign_keys)
        if single_connection:
            event.listen(engine.sync_engine, "connect", _sqlite_fast_writes)
        return engine
    return create_async_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10)


def _sqlite_fast_writes(dbapi_connection: Any, _record: Any) -> None:
    # Write-ahead log with fsync only at checkpoints: safe against an application crash,
    # and a lost transaction on power loss is acceptable for a batch job's working file.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def _sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:  # type: ignore[type-arg]
    return async_sessionmaker(engine, expire_on_commit=False)


__all__ = [
    "Base",
    "CalibratorRecord",
    "DecisionRecord",
    "DriftCheckRecord",
    "OutcomeRecord",
    "ReviewItemRecord",
    "make_engine",
    "make_sessionmaker",
]
