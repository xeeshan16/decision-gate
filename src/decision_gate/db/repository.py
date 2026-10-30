"""Data access for decisions."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from decision_gate.db.models import DecisionRecord


class DecisionRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, decision_id: uuid.UUID) -> DecisionRecord | None:
        async with self._sessions() as session:
            return await session.get(DecisionRecord, decision_id)

    async def get_by_idempotency_key(self, key: str) -> DecisionRecord | None:
        async with self._sessions() as session:
            result = await session.execute(
                select(DecisionRecord).where(DecisionRecord.idempotency_key == key)
            )
            return result.scalar_one_or_none()

    async def add(self, record: DecisionRecord) -> tuple[DecisionRecord, bool]:
        """Insert a decision. Returns ``(stored, created)``.

        If another request already stored a decision with the same idempotency key (a
        concurrent retry), the unique constraint rejects this insert and the existing row is
        returned with ``created=False``.
        """
        async with self._sessions() as session:
            session.add(record)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                if record.idempotency_key is None:
                    raise
                existing = await self.get_by_idempotency_key(record.idempotency_key)
                if existing is None:  # pragma: no cover - the conflicting row vanished
                    raise
                return existing, False
        return record, True
