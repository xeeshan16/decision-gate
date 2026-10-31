"""Persistence: schema, engine and repositories."""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from decision_gate.db.models import Base, DecisionRecord, OutcomeRecord


def make_engine(url: str) -> AsyncEngine:
    """Create an async engine. Nothing connects until the first query."""
    if url.startswith("sqlite"):
        # SQLite is for tests and quick local runs. A fresh connection per use keeps it
        # safe across the event loops that test clients create.
        engine = create_async_engine(url, poolclass=NullPool)
        # SQLite ignores foreign keys unless asked, per connection; PostgreSQL always checks.
        event.listen(engine.sync_engine, "connect", _sqlite_foreign_keys)
        return engine
    return create_async_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10)


def _sqlite_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:  # type: ignore[type-arg]
    return async_sessionmaker(engine, expire_on_commit=False)


__all__ = ["Base", "DecisionRecord", "OutcomeRecord", "make_engine", "make_sessionmaker"]
