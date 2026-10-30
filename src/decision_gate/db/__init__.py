"""Persistence: schema, engine and repositories."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from decision_gate.db.models import Base, DecisionRecord


def make_engine(url: str) -> AsyncEngine:
    """Create an async engine. Nothing connects until the first query."""
    if url.startswith("sqlite"):
        # SQLite is for tests and quick local runs. A fresh connection per use keeps it
        # safe across the event loops that test clients create.
        return create_async_engine(url, poolclass=NullPool)
    return create_async_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:  # type: ignore[type-arg]
    return async_sessionmaker(engine, expire_on_commit=False)


__all__ = ["Base", "DecisionRecord", "make_engine", "make_sessionmaker"]
