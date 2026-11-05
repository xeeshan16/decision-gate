"""Data access for decisions and their outcomes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, case, func, not_, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from decision_gate.db.models import (
    CalibratorRecord,
    DecisionRecord,
    OutcomeRecord,
    ReviewItemRecord,
    RoutingRuleRecord,
)

UNRESOLVED = ("open", "claimed")
"""Review item statuses that are still in the queue."""


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

    async def add(
        self, record: DecisionRecord, review: ReviewItemRecord | None = None
    ) -> tuple[DecisionRecord, bool]:
        """Insert a decision, and its review item if it has one. Returns ``(stored, created)``.

        Both rows commit together, so a decision a person must handle can never be stored
        without its place in the review queue. If another request already stored a decision
        with the same idempotency key (a concurrent retry), the unique constraint rejects
        this insert and the existing row is returned with ``created=False``.
        """
        async with self._sessions() as session:
            session.add(record)
            try:
                if review is not None:
                    # The review item references the decision, so the decision goes first.
                    await session.flush()
                    session.add(review)
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


@dataclass(frozen=True, slots=True)
class LabeledRow:
    """The columns of one labeled decision that calibration needs, without ORM overhead."""

    question_id: str
    question_version: int
    backend: str
    model_version: str
    confidence: float
    correct: bool
    probabilities: dict[str, float] | None
    outcome: Any
    answer: Any = None
    created_at: datetime | None = None
    """Decision time, so jobs can split labeled data chronologically."""


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
        A review item still waiting for this decision is superseded in the same transaction:
        the truth is known, so nobody needs to review it any more.
        """
        async with self._sessions() as session:
            session.add(record)
            try:
                await session.flush()
                await session.execute(
                    update(ReviewItemRecord)
                    .where(
                        ReviewItemRecord.decision_id == record.decision_id,
                        ReviewItemRecord.status.in_(UNRESOLVED),
                    )
                    .values(
                        status="superseded",
                        resolved_at=datetime.now(UTC),
                        note=f"outcome recorded by {record.source}"[:1000],
                    )
                )
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

    async def labeled_rows(
        self,
        *,
        question_id: str | None = None,
        question_version: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[LabeledRow]:
        """Labeled decisions as plain rows, grouped by question version, oldest first.

        Selects only the columns the metrics read, so a report over many thousands of
        decisions does not build ORM objects. The window applies to the decision time.
        """
        query = (
            select(
                DecisionRecord.question_id,
                DecisionRecord.question_version,
                DecisionRecord.backend,
                DecisionRecord.model_version,
                DecisionRecord.confidence,
                OutcomeRecord.correct,
                DecisionRecord.probabilities,
                OutcomeRecord.outcome,
                DecisionRecord.answer,
                DecisionRecord.created_at,
            )
            .join(OutcomeRecord, OutcomeRecord.decision_id == DecisionRecord.id)
            .order_by(
                DecisionRecord.question_id,
                DecisionRecord.question_version,
                DecisionRecord.created_at,
                DecisionRecord.id,
            )
        )
        if question_id is not None:
            query = query.where(DecisionRecord.question_id == question_id)
        if question_version is not None:
            query = query.where(DecisionRecord.question_version == question_version)
        if since is not None:
            query = query.where(DecisionRecord.created_at >= since)
        if until is not None:
            query = query.where(DecisionRecord.created_at < until)
        async with self._sessions() as session:
            result = await session.execute(query)
            return [
                LabeledRow(
                    question_id=qid,
                    question_version=int(version),
                    backend=backend,
                    model_version=model_version,
                    confidence=float(confidence),
                    correct=bool(correct),
                    probabilities=probabilities,
                    outcome=outcome,
                    answer=answer,
                    created_at=created_at,
                )
                for (
                    qid,
                    version,
                    backend,
                    model_version,
                    confidence,
                    correct,
                    probabilities,
                    outcome,
                    answer,
                    created_at,
                ) in result.all()
            ]

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


class ActivationConflictError(RuntimeError):
    """Another activation for the same question version won a concurrent race."""


class CalibratorRepository:
    """The versioned calibrator registry."""

    MAX_VERSION_ATTEMPTS = 5

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, calibrator_id: uuid.UUID) -> CalibratorRecord | None:
        async with self._sessions() as session:
            return await session.get(CalibratorRecord, calibrator_id)

    async def active(self, question_id: str, question_version: int) -> CalibratorRecord | None:
        async with self._sessions() as session:
            result = await session.execute(
                select(CalibratorRecord).where(
                    CalibratorRecord.question_id == question_id,
                    CalibratorRecord.question_version == question_version,
                    CalibratorRecord.status == "active",
                )
            )
            return result.scalar_one_or_none()

    async def find(
        self,
        *,
        question_id: str | None = None,
        question_version: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[CalibratorRecord]:
        """Newest version first within each question version."""
        query = (
            select(CalibratorRecord)
            .order_by(
                CalibratorRecord.question_id,
                CalibratorRecord.question_version,
                CalibratorRecord.version.desc(),
            )
            .limit(limit)
        )
        if question_id is not None:
            query = query.where(CalibratorRecord.question_id == question_id)
        if question_version is not None:
            query = query.where(CalibratorRecord.question_version == question_version)
        if status is not None:
            query = query.where(CalibratorRecord.status == status)
        async with self._sessions() as session:
            result = await session.execute(query)
            return list(result.scalars().all())

    async def add(self, record: CalibratorRecord) -> CalibratorRecord:
        """Store a new fit with the next version number for its question version.

        If ``record.status`` is ``active``, the current active calibrator is retired in
        the same transaction. Version numbers are taken as ``max + 1``; when two fits race
        for the same number the unique constraint rejects one, which retries.
        """
        for _attempt in range(self.MAX_VERSION_ATTEMPTS):
            async with self._sessions() as session:
                latest = await session.scalar(
                    select(func.max(CalibratorRecord.version)).where(
                        CalibratorRecord.question_id == record.question_id,
                        CalibratorRecord.question_version == record.question_version,
                    )
                )
                record.version = int(latest or 0) + 1
                if record.status == "active":
                    record.activated_at = datetime.now(UTC)
                    await self._retire_active(session, record.question_id, record.question_version)
                session.add(record)
                try:
                    await session.commit()
                except IntegrityError:
                    # The rolled-back record is transient again and is re-added next try.
                    await session.rollback()
                    continue
                return record
        raise ActivationConflictError(  # pragma: no cover - needs repeated lost races
            f"could not store a calibrator for {record.question_key} after "
            f"{self.MAX_VERSION_ATTEMPTS} attempts (concurrent fits)"
        )

    async def activate(self, calibrator_id: uuid.UUID) -> CalibratorRecord | None:
        """Make one stored calibrator the active one for its question version.

        Works for any status, so an operator can roll back to an older calibrator.
        Returns ``None`` for an unknown id.
        """
        async with self._sessions() as session:
            record = await session.get(CalibratorRecord, calibrator_id)
            if record is None:
                return None
            if record.status == "active":
                return record
            await self._retire_active(session, record.question_id, record.question_version)
            record.status = "active"
            record.activated_at = datetime.now(UTC)
            try:
                await session.commit()
            except IntegrityError as exc:  # pragma: no cover - needs an exact race
                await session.rollback()
                raise ActivationConflictError(
                    f"another calibrator for {record.question_key} was activated concurrently"
                ) from exc
            return record

    @staticmethod
    async def _retire_active(session: AsyncSession, question_id: str, version: int) -> None:
        await session.execute(
            update(CalibratorRecord)
            .where(
                CalibratorRecord.question_id == question_id,
                CalibratorRecord.question_version == version,
                CalibratorRecord.status == "active",
            )
            .values(status="retired")
        )
        await session.flush()


class RoutingRuleRepository:
    """Routing rules: the thresholds each question version is routed with, kept for audit."""

    MAX_ATTEMPTS = 5

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, rule_id: uuid.UUID) -> RoutingRuleRecord | None:
        async with self._sessions() as session:
            return await session.get(RoutingRuleRecord, rule_id)

    async def active(self, question_id: str, question_version: int) -> RoutingRuleRecord | None:
        async with self._sessions() as session:
            result = await session.execute(
                select(RoutingRuleRecord).where(
                    RoutingRuleRecord.question_id == question_id,
                    RoutingRuleRecord.question_version == question_version,
                    RoutingRuleRecord.status == "active",
                )
            )
            return result.scalar_one_or_none()

    async def find(
        self,
        *,
        question_id: str | None = None,
        question_version: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RoutingRuleRecord]:
        """Newest rule first within each question version."""
        query = (
            select(RoutingRuleRecord)
            .order_by(
                RoutingRuleRecord.question_id,
                RoutingRuleRecord.question_version,
                RoutingRuleRecord.created_at.desc(),
            )
            .limit(limit)
        )
        if question_id is not None:
            query = query.where(RoutingRuleRecord.question_id == question_id)
        if question_version is not None:
            query = query.where(RoutingRuleRecord.question_version == question_version)
        if status is not None:
            query = query.where(RoutingRuleRecord.status == status)
        async with self._sessions() as session:
            result = await session.execute(query)
            return list(result.scalars().all())

    async def add_active(self, record: RoutingRuleRecord) -> RoutingRuleRecord:
        """Store a rule as the active one, retiring the current one in the same transaction.

        Two concurrent refreshes can both retire the old rule and insert; the partial unique
        index rejects the second insert, which then retries and retires the first.
        """
        record.status = "active"
        for _attempt in range(self.MAX_ATTEMPTS):
            async with self._sessions() as session:
                await session.execute(
                    update(RoutingRuleRecord)
                    .where(
                        RoutingRuleRecord.question_id == record.question_id,
                        RoutingRuleRecord.question_version == record.question_version,
                        RoutingRuleRecord.status == "active",
                    )
                    .values(status="retired")
                )
                session.add(record)
                try:
                    await session.commit()
                except IntegrityError:  # pragma: no cover - needs an exact race
                    await session.rollback()
                    continue
                return record
        raise ActivationConflictError(  # pragma: no cover - needs repeated lost races
            f"could not store a routing rule for {record.question_key} after "
            f"{self.MAX_ATTEMPTS} attempts (concurrent refreshes)"
        )


class ReviewNotFoundError(LookupError):
    """No review item has this id."""


class ReviewConflictError(RuntimeError):
    """The item is not in a state that allows this action (taken, closed, resolved)."""


def _available(now: datetime) -> Any:
    """Unresolved items nobody holds: open, or claimed under a lease that has run out."""
    return or_(
        ReviewItemRecord.status == "open",
        and_(ReviewItemRecord.status == "claimed", ReviewItemRecord.claim_expires_at <= now),
    )


def _aware(moment: datetime | None) -> datetime | None:
    # SQLite drops the offset; values are stored in UTC.
    if moment is None or moment.tzinfo is not None:
        return moment
    return moment.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ReviewStatRow:
    """The columns of one review item that queue statistics need."""

    question_id: str
    question_version: int
    tier: str
    status: str
    created_at: datetime
    due_at: datetime
    claim_expires_at: datetime | None
    resolved_at: datetime | None
    verdict: str | None
    agreed: bool | None


class ReviewRepository:
    """The human review queue.

    Every state change is a single conditional ``UPDATE`` (or, for resolution, one
    transaction holding a row lock), so two reviewers racing for the same item cannot both
    win: the loser's update matches no row and is reported as a conflict.
    """

    MAX_CLAIM_ATTEMPTS = 5

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, item_id: uuid.UUID) -> tuple[ReviewItemRecord, DecisionRecord] | None:
        async with self._sessions() as session:
            result = await session.execute(
                select(ReviewItemRecord, DecisionRecord)
                .join(DecisionRecord, DecisionRecord.id == ReviewItemRecord.decision_id)
                .where(ReviewItemRecord.id == item_id)
            )
            row = result.one_or_none()
            return None if row is None else (row[0], row[1])

    async def for_decision(self, decision_id: uuid.UUID) -> ReviewItemRecord | None:
        async with self._sessions() as session:
            result = await session.execute(
                select(ReviewItemRecord).where(ReviewItemRecord.decision_id == decision_id)
            )
            return result.scalar_one_or_none()

    async def find(
        self,
        *,
        now: datetime,
        state: str | None = None,
        question_id: str | None = None,
        tier: str | None = None,
        claimed_by: str | None = None,
        overdue: bool | None = None,
        limit: int = 100,
    ) -> list[tuple[ReviewItemRecord, DecisionRecord]]:
        """Items with their decisions, earliest deadline first.

        ``state`` is ``available`` (open, or claimed under an expired lease), ``claimed``
        (held under a live lease), ``resolved`` or ``superseded``. ``overdue`` compares
        unresolved items with ``now`` and resolved ones with their resolution time.
        """
        query = (
            select(ReviewItemRecord, DecisionRecord)
            .join(DecisionRecord, DecisionRecord.id == ReviewItemRecord.decision_id)
            .order_by(ReviewItemRecord.due_at, ReviewItemRecord.created_at, ReviewItemRecord.id)
            .limit(limit)
        )
        if state == "available":
            query = query.where(_available(now))
        elif state == "claimed":
            query = query.where(
                ReviewItemRecord.status == "claimed", ReviewItemRecord.claim_expires_at > now
            )
        elif state is not None:
            query = query.where(ReviewItemRecord.status == state)
        if question_id is not None:
            query = query.where(ReviewItemRecord.question_id == question_id)
        if tier is not None:
            query = query.where(ReviewItemRecord.tier == tier)
        if claimed_by is not None:
            query = query.where(ReviewItemRecord.claimed_by == claimed_by)
        if overdue is not None:
            late = or_(
                and_(ReviewItemRecord.status.in_(UNRESOLVED), ReviewItemRecord.due_at < now),
                and_(
                    ReviewItemRecord.status == "resolved",
                    ReviewItemRecord.resolved_at > ReviewItemRecord.due_at,
                ),
            )
            query = query.where(late if overdue else not_(late))
        async with self._sessions() as session:
            result = await session.execute(query)
            return [(item, decision) for item, decision in result.all()]

    async def claim_next(
        self,
        reviewer: str,
        *,
        now: datetime,
        lease_until: datetime,
        question_id: str | None = None,
        tier: str | None = None,
    ) -> ReviewItemRecord | None:
        """Claim the available item with the earliest deadline, or return ``None``.

        On PostgreSQL the candidate row is locked with ``SKIP LOCKED``, so concurrent
        reviewers each get a different item instead of queueing on the same one. The
        conditional update is the real guard (and the only one on SQLite): if another
        reviewer took the candidate first, it matches nothing and the next one is tried.
        """
        for _attempt in range(self.MAX_CLAIM_ATTEMPTS):
            async with self._sessions() as session:
                query = (
                    select(ReviewItemRecord.id)
                    .where(_available(now))
                    .order_by(
                        ReviewItemRecord.due_at, ReviewItemRecord.created_at, ReviewItemRecord.id
                    )
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                if question_id is not None:
                    query = query.where(ReviewItemRecord.question_id == question_id)
                if tier is not None:
                    query = query.where(ReviewItemRecord.tier == tier)
                candidate = await session.scalar(query)
                if candidate is None:
                    return None
                if await self._take(session, candidate, reviewer, now, lease_until):
                    await session.commit()
                    return await session.get(ReviewItemRecord, candidate, populate_existing=True)
                await session.rollback()  # pragma: no cover - needs an exact race
        return None  # pragma: no cover - needs repeated lost races

    async def claim(
        self, item_id: uuid.UUID, reviewer: str, *, now: datetime, lease_until: datetime
    ) -> ReviewItemRecord:
        """Claim one item. Claiming an item you already hold renews the lease."""
        async with self._sessions() as session:
            if await self._take(session, item_id, reviewer, now, lease_until):
                await session.commit()
                item = await session.get(ReviewItemRecord, item_id, populate_existing=True)
                assert item is not None
                return item
            item = await session.get(ReviewItemRecord, item_id)
            if item is None:
                raise ReviewNotFoundError(f"unknown review item {item_id}")
            raise ReviewConflictError(_unavailable(item))

    @staticmethod
    async def _take(
        session: AsyncSession,
        item_id: uuid.UUID,
        reviewer: str,
        now: datetime,
        lease_until: datetime,
    ) -> bool:
        renew = and_(
            ReviewItemRecord.status == "claimed",
            ReviewItemRecord.claimed_by == reviewer,
        )
        result = await session.execute(
            update(ReviewItemRecord)
            .where(ReviewItemRecord.id == item_id, or_(_available(now), renew))
            .values(
                status="claimed",
                claimed_by=reviewer,
                claimed_at=now,
                claim_expires_at=lease_until,
                # A renewal by the same holder is not a new claim.
                claims=case(
                    (renew, ReviewItemRecord.claims),
                    else_=ReviewItemRecord.claims + 1,
                ),
            )
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))

    async def release(self, item_id: uuid.UUID, reviewer: str) -> ReviewItemRecord:
        """Give an item back to the queue. Only its current holder can release it."""
        async with self._sessions() as session:
            result = await session.execute(
                update(ReviewItemRecord)
                .where(
                    ReviewItemRecord.id == item_id,
                    ReviewItemRecord.status == "claimed",
                    ReviewItemRecord.claimed_by == reviewer,
                )
                .values(status="open", claimed_by=None, claimed_at=None, claim_expires_at=None)
                .execution_options(synchronize_session=False)
            )
            if getattr(result, "rowcount", 0):
                await session.commit()
                item = await session.get(ReviewItemRecord, item_id, populate_existing=True)
                assert item is not None
                return item
            item = await session.get(ReviewItemRecord, item_id)
            if item is None:
                raise ReviewNotFoundError(f"unknown review item {item_id}")
            raise ReviewConflictError(
                f"review item {item_id} is not claimed by {reviewer}"
                if item.status in UNRESOLVED
                else _unavailable(item)
            )

    async def resolve(
        self,
        item_id: uuid.UUID,
        reviewer: str,
        *,
        now: datetime,
        outcome: OutcomeRecord,
        verdict: str,
        note: str | None = None,
    ) -> tuple[ReviewItemRecord, OutcomeRecord, bool]:
        """Resolve an item and record its outcome in one transaction.

        Returns ``(item, outcome, created)``; ``created`` is false when the same reviewer
        resolves the same item with the same answer again (a retry). Anyone may resolve an
        item that nobody holds under a live lease; an item someone else holds, an item
        already resolved differently, and a superseded item are conflicts.
        """
        async with self._sessions() as session:
            item = await session.get(ReviewItemRecord, item_id, with_for_update=True)
            if item is None:
                raise ReviewNotFoundError(f"unknown review item {item_id}")
            existing = await session.get(OutcomeRecord, item.decision_id)
            if item.status == "resolved":
                if item.resolved_by == reviewer and item.resolution == outcome.outcome:
                    assert existing is not None
                    return item, existing, False
                raise ReviewConflictError(_unavailable(item))
            if item.status == "superseded":
                raise ReviewConflictError(_unavailable(item))
            held = _aware(item.claim_expires_at)
            if item.status == "claimed" and item.claimed_by != reviewer and held and held > now:
                raise ReviewConflictError(
                    f"review item {item_id} is claimed by {item.claimed_by} until "
                    f"{held.isoformat()}"
                )
            if existing is not None:  # pragma: no cover - the outcome would have superseded it
                raise ReviewConflictError(f"decision {item.decision_id} already has an outcome")
            session.add(outcome)
            item.status = "resolved"
            item.resolved_by = reviewer
            item.resolved_at = now
            item.resolution = outcome.outcome
            item.verdict = verdict
            item.agreed = outcome.correct
            item.note = note
            if item.claimed_by != reviewer:
                # Resolved without a claim (or after someone's lease ran out): record who
                # actually did the work.
                item.claimed_by = reviewer
                item.claimed_at = item.claimed_at or now
                item.claims += 1
            item.claim_expires_at = None
            try:
                await session.commit()
            except IntegrityError as exc:  # pragma: no cover - needs an exact race
                await session.rollback()
                raise ReviewConflictError(
                    f"an outcome for decision {item.decision_id} was recorded concurrently"
                ) from exc
            return item, outcome, True

    async def stat_rows(
        self, *, question_id: str | None = None, since: datetime | None = None
    ) -> list[ReviewStatRow]:
        query = select(
            ReviewItemRecord.question_id,
            ReviewItemRecord.question_version,
            ReviewItemRecord.tier,
            ReviewItemRecord.status,
            ReviewItemRecord.created_at,
            ReviewItemRecord.due_at,
            ReviewItemRecord.claim_expires_at,
            ReviewItemRecord.resolved_at,
            ReviewItemRecord.verdict,
            ReviewItemRecord.agreed,
        ).order_by(
            ReviewItemRecord.question_id, ReviewItemRecord.question_version, ReviewItemRecord.tier
        )
        if question_id is not None:
            query = query.where(ReviewItemRecord.question_id == question_id)
        if since is not None:
            query = query.where(ReviewItemRecord.created_at >= since)
        async with self._sessions() as session:
            result = await session.execute(query)
            rows: list[ReviewStatRow] = []
            for row in result.all():
                created, due, expires, resolved = (_aware(v) for v in row[4:8])
                assert created is not None
                assert due is not None
                rows.append(
                    ReviewStatRow(
                        question_id=row[0],
                        question_version=int(row[1]),
                        tier=row[2],
                        status=row[3],
                        created_at=created,
                        due_at=due,
                        claim_expires_at=expires,
                        resolved_at=resolved,
                        verdict=row[8],
                        agreed=row[9],
                    )
                )
            return rows

    async def unqueued(self, limit: int = 1000) -> list[DecisionRecord]:
        """Verify and human decisions with no review item and no outcome, oldest first.

        These were stored before the review queue existed; ``backfill`` enqueues them.
        """
        query = (
            select(DecisionRecord)
            .outerjoin(ReviewItemRecord, ReviewItemRecord.decision_id == DecisionRecord.id)
            .outerjoin(OutcomeRecord, OutcomeRecord.decision_id == DecisionRecord.id)
            .where(
                DecisionRecord.tier.in_(("verify", "human")),
                ReviewItemRecord.id.is_(None),
                OutcomeRecord.decision_id.is_(None),
            )
            .order_by(DecisionRecord.created_at, DecisionRecord.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            result = await session.execute(query)
            return list(result.scalars().all())

    async def add_many(self, items: list[ReviewItemRecord]) -> int:
        """Insert items, skipping any whose decision was enqueued concurrently."""
        added = 0
        for item in items:
            async with self._sessions() as session:
                session.add(item)
                try:
                    await session.commit()
                except IntegrityError:  # pragma: no cover - needs a concurrent backfill
                    await session.rollback()
                    continue
                added += 1
        return added


def _unavailable(item: ReviewItemRecord) -> str:
    if item.status == "resolved":
        return f"review item {item.id} was already resolved by {item.resolved_by}"
    if item.status == "superseded":
        return f"review item {item.id} was superseded: {item.note or 'outcome recorded'}"
    expires = _aware(item.claim_expires_at)
    until = f" until {expires.isoformat()}" if expires else ""
    return f"review item {item.id} is claimed by {item.claimed_by}{until}"
