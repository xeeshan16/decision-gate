"""Data access for shadow decisions (candidate backends answering on the side)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import String, and_, case, cast, func, not_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from decision_gate.db.models import DecisionRecord, OutcomeRecord, ShadowDecisionRecord

SHADOW_STATUSES = ("ok", "error", "timeout")
SHADOW_MODES = ("live", "replay")


@dataclass(frozen=True, slots=True)
class PairedRow:
    """One labeled decision answered by both the primary backend and a candidate."""

    decision_id: uuid.UUID
    created_at: datetime
    outcome: Any
    primary_answer: Any
    primary_confidence: float
    primary_probabilities: dict[str, float] | None
    primary_correct: bool
    primary_latency_ms: float
    candidate_answer: Any
    candidate_confidence: float
    candidate_probabilities: dict[str, float] | None
    candidate_latency_ms: float
    candidate_model_version: str | None


@dataclass(frozen=True, slots=True)
class ShadowSummary:
    """How much a candidate has answered for one question version, and how it went."""

    question_id: str
    question_version: int
    candidate: str
    total: int
    ok: int
    errors: int
    timeouts: int
    labeled: int
    """``ok`` answers whose primary decision has an outcome, i.e. comparable ones."""
    last_at: datetime | None

    @property
    def failure_rate(self) -> float:
        return (self.errors + self.timeouts) / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class ReplayTarget:
    decision_id: uuid.UUID
    input: dict[str, Any]


class ShadowRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, decision_id: uuid.UUID, candidate: str) -> ShadowDecisionRecord | None:
        query = select(ShadowDecisionRecord).where(
            ShadowDecisionRecord.decision_id == decision_id,
            ShadowDecisionRecord.candidate == candidate,
        )
        async with self._sessions() as session:
            return (await session.execute(query)).scalar_one_or_none()

    async def for_decision(self, decision_id: uuid.UUID) -> list[ShadowDecisionRecord]:
        query = (
            select(ShadowDecisionRecord)
            .where(ShadowDecisionRecord.decision_id == decision_id)
            .order_by(ShadowDecisionRecord.candidate)
        )
        async with self._sessions() as session:
            return list((await session.execute(query)).scalars().all())

    async def add(self, record: ShadowDecisionRecord) -> tuple[ShadowDecisionRecord, bool]:
        """Insert a shadow answer; a second answer by the same candidate keeps the first."""
        async with self._sessions() as session:
            session.add(record)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await self.get(record.decision_id, record.candidate)
                if existing is None:
                    # Not a duplicate: the decision row does not exist (foreign key).
                    raise
                return existing, False
        return record, True

    async def paired_rows(
        self,
        question_id: str,
        question_version: int,
        candidate: str,
        *,
        mode: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[PairedRow]:
        """Labeled decisions with an ``ok`` answer from ``candidate``, oldest decision first."""
        shadow = ShadowDecisionRecord
        query = (
            select(
                DecisionRecord.id,
                DecisionRecord.created_at,
                OutcomeRecord.outcome,
                DecisionRecord.answer,
                DecisionRecord.confidence,
                DecisionRecord.probabilities,
                OutcomeRecord.correct,
                DecisionRecord.latency_ms,
                shadow.answer,
                shadow.confidence,
                shadow.probabilities,
                shadow.latency_ms,
                shadow.model_version,
            )
            .join(shadow, shadow.decision_id == DecisionRecord.id)
            .join(OutcomeRecord, OutcomeRecord.decision_id == DecisionRecord.id)
            .where(
                DecisionRecord.question_id == question_id,
                DecisionRecord.question_version == question_version,
                shadow.candidate == candidate,
                shadow.status == "ok",
            )
            .order_by(DecisionRecord.created_at, DecisionRecord.id)
        )
        if mode is not None:
            query = query.where(shadow.mode == mode)
        if since is not None:
            query = query.where(DecisionRecord.created_at >= since)
        if until is not None:
            query = query.where(DecisionRecord.created_at < until)
        async with self._sessions() as session:
            result = await session.execute(query)
            return [
                PairedRow(
                    decision_id=decision_id,
                    created_at=created_at,
                    outcome=outcome,
                    primary_answer=answer,
                    primary_confidence=float(confidence),
                    primary_probabilities=probabilities,
                    primary_correct=bool(correct),
                    primary_latency_ms=float(latency),
                    candidate_answer=c_answer,
                    candidate_confidence=float(c_confidence),
                    candidate_probabilities=c_probabilities,
                    candidate_latency_ms=float(c_latency),
                    candidate_model_version=c_model,
                )
                for (
                    decision_id,
                    created_at,
                    outcome,
                    answer,
                    confidence,
                    probabilities,
                    correct,
                    latency,
                    c_answer,
                    c_confidence,
                    c_probabilities,
                    c_latency,
                    c_model,
                ) in result.all()
            ]

    async def summaries(
        self,
        *,
        question_id: str | None = None,
        question_version: int | None = None,
        candidate: str | None = None,
        mode: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[ShadowSummary]:
        """Counts per question version and candidate. The window is on decision time."""
        shadow = ShadowDecisionRecord
        labeled = and_(shadow.status == "ok", OutcomeRecord.decision_id.is_not(None))
        query = (
            select(
                shadow.question_id,
                shadow.question_version,
                shadow.candidate,
                func.count(shadow.id),
                func.sum(case((shadow.status == "ok", 1), else_=0)),
                func.sum(case((shadow.status == "error", 1), else_=0)),
                func.sum(case((shadow.status == "timeout", 1), else_=0)),
                func.sum(case((labeled, 1), else_=0)),
                func.max(shadow.created_at),
            )
            .join(DecisionRecord, DecisionRecord.id == shadow.decision_id)
            .outerjoin(OutcomeRecord, OutcomeRecord.decision_id == shadow.decision_id)
            .group_by(shadow.question_id, shadow.question_version, shadow.candidate)
            .order_by(shadow.question_id, shadow.question_version, shadow.candidate)
        )
        if question_id is not None:
            query = query.where(shadow.question_id == question_id)
        if question_version is not None:
            query = query.where(shadow.question_version == question_version)
        if candidate is not None:
            query = query.where(shadow.candidate == candidate)
        if mode is not None:
            query = query.where(shadow.mode == mode)
        if since is not None:
            query = query.where(DecisionRecord.created_at >= since)
        if until is not None:
            query = query.where(DecisionRecord.created_at < until)
        async with self._sessions() as session:
            result = await session.execute(query)
            return [
                ShadowSummary(
                    question_id=qid,
                    question_version=int(version),
                    candidate=name,
                    total=int(total),
                    ok=int(ok or 0),
                    errors=int(errors or 0),
                    timeouts=int(timeouts or 0),
                    labeled=int(n_labeled or 0),
                    last_at=last_at,
                )
                for (
                    qid,
                    version,
                    name,
                    total,
                    ok,
                    errors,
                    timeouts,
                    n_labeled,
                    last_at,
                ) in result.all()
            ]

    async def replay_targets(
        self,
        question_id: str,
        question_version: int,
        candidate: str,
        *,
        labeled_only: bool = True,
        limit: int = 1000,
    ) -> tuple[list[ReplayTarget], int]:
        """Stored decisions the candidate has not answered yet, oldest first.

        Returns the targets and how many otherwise eligible decisions were skipped because
        their input was not stored (``DGATE_STORE_INPUTS=false``).
        """
        answered = (
            select(ShadowDecisionRecord.id)
            .where(
                ShadowDecisionRecord.decision_id == DecisionRecord.id,
                ShadowDecisionRecord.candidate == candidate,
            )
            .exists()
        )
        base = select(DecisionRecord.id).where(
            DecisionRecord.question_id == question_id,
            DecisionRecord.question_version == question_version,
            ~answered,
        )
        if labeled_only:
            base = base.join(OutcomeRecord, OutcomeRecord.decision_id == DecisionRecord.id)
        # A decision stored without its input holds JSON null, not SQL NULL.
        no_input = or_(DecisionRecord.input.is_(None), cast(DecisionRecord.input, String) == "null")
        targets_query = (
            base.add_columns(DecisionRecord.input)
            .where(not_(no_input))
            .order_by(DecisionRecord.created_at, DecisionRecord.id)
            .limit(limit)
        )
        without_input = select(func.count()).select_from(base.where(no_input).subquery())
        async with self._sessions() as session:
            rows = (await session.execute(targets_query)).all()
            skipped = int((await session.execute(without_input)).scalar_one())
        return [ReplayTarget(decision_id=row[0], input=row[1]) for row in rows], skipped
