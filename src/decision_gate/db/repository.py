"""Data access for decisions and their outcomes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from decision_gate.db.models import DecisionRecord, OutcomeRecord


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


@dataclass(frozen=True, slots=True)
class LabelStats:
    """How much ground truth one question version has, and how the raw scores compare."""

    question_id: str
    question_version: int
    decisions: int
    labeled: int
    correct: int
    mean_confidence: float | None
    """Mean raw confidence over the labeled decisions only, so it compares with accuracy."""

    @property
    def label_rate(self) -> float:
        return self.labeled / self.decisions if self.decisions else 0.0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.labeled if self.labeled else None


class OutcomeRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, decision_id: uuid.UUID) -> OutcomeRecord | None:
        async with self._sessions() as session:
            return await session.get(OutcomeRecord, decision_id)

    async def add(self, record: OutcomeRecord) -> tuple[OutcomeRecord, bool]:
        """Insert an outcome. Returns ``(stored, created)``.

        The decision id is the primary key, so when two deliveries of the same outcome race,
        exactly one insert wins and the other gets the winner's row with ``created=False``.
        """
        async with self._sessions() as session:
            session.add(record)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await self.get(record.decision_id)
                if existing is None:
                    # Not a duplicate: the decision row does not exist (foreign key).
                    raise
                return existing, False
        return record, True

    async def labeled(
        self,
        question_id: str,
        question_version: int,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[tuple[DecisionRecord, OutcomeRecord]]:
        """Decisions of one question version joined to their outcomes, oldest first.

        This is the training and evaluation set for calibration and drift: only decisions
        whose ground truth is known. The window applies to the decision time.
        """
        query = (
            select(DecisionRecord, OutcomeRecord)
            .join(OutcomeRecord, OutcomeRecord.decision_id == DecisionRecord.id)
            .where(
                DecisionRecord.question_id == question_id,
                DecisionRecord.question_version == question_version,
            )
            .order_by(DecisionRecord.created_at, DecisionRecord.id)
        )
        if since is not None:
            query = query.where(DecisionRecord.created_at >= since)
        if until is not None:
            query = query.where(DecisionRecord.created_at < until)
        if limit is not None:
            query = query.limit(limit)
        async with self._sessions() as session:
            result = await session.execute(query)
            return [(decision, outcome) for decision, outcome in result.all()]

    async def stats(self, question_id: str | None = None) -> list[LabelStats]:
        """Per question version: decisions, how many are labeled, and raw accuracy."""
        labeled = OutcomeRecord.decision_id.is_not(None)
        query = (
            select(
                DecisionRecord.question_id,
                DecisionRecord.question_version,
                func.count(DecisionRecord.id),
                func.count(OutcomeRecord.decision_id),
                func.coalesce(func.sum(case((OutcomeRecord.correct, 1), else_=0)), 0),
                func.avg(case((labeled, DecisionRecord.confidence), else_=None)),
            )
            .outerjoin(OutcomeRecord, OutcomeRecord.decision_id == DecisionRecord.id)
            .group_by(DecisionRecord.question_id, DecisionRecord.question_version)
            .order_by(DecisionRecord.question_id, DecisionRecord.question_version)
        )
        if question_id is not None:
            query = query.where(DecisionRecord.question_id == question_id)
        async with self._sessions() as session:
            result = await session.execute(query)
            return [
                LabelStats(
                    question_id=qid,
                    question_version=int(version),
                    decisions=int(decisions),
                    labeled=int(n_labeled),
                    correct=int(n_correct),
                    mean_confidence=None if mean_conf is None else float(mean_conf),
                )
                for qid, version, decisions, n_labeled, n_correct, mean_conf in result.all()
            ]
