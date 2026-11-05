"""The human review queue: where ``verify`` and ``human`` decisions are worked.

Routing decides *who* handles a decision; this module is where people handle it. Every
decision a person must see becomes a review item with an SLA deadline (shorter for
``verify``, where a person only confirms an answer, than for ``human``, where they decide
from scratch). Reviewers claim items under a lease, so an item abandoned mid-review goes
back to the queue on its own, and resolve them with an answer.

A resolution is recorded as the decision's outcome in the same transaction. That closes
the loop: the labels people produce while working the queue are the labels the next
calibrator refit and threshold refresh learn from. It also means review labels are the
ground truth, so the queue keeps the evidence an auditor needs: who resolved what, when,
against which deadline, and whether they agreed with the backend.
"""

from __future__ import annotations

import math
import statistics
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from decision_gate.bounds import interval
from decision_gate.db.models import DecisionRecord, OutcomeRecord, ReviewItemRecord
from decision_gate.db.repository import (
    ReviewNotFoundError,
    ReviewRepository,
    ReviewStatRow,
)
from decision_gate.questions import Answer, QuestionCatalog

ReviewTier = Literal["verify", "human"]
REVIEW_TIERS: tuple[ReviewTier, ...] = ("verify", "human")
ReviewState = Literal["available", "claimed", "resolved", "superseded"]
REVIEW_STATES: tuple[ReviewState, ...] = ("available", "claimed", "resolved", "superseded")
Verdict = Literal["confirmed", "corrected", "decided"]

REVIEW_SOURCE = "review"
"""The ``source`` of outcomes that come from resolved review items."""
MAX_NOTE_LENGTH = 1000

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(moment: datetime) -> datetime:
    """Treat naive timestamps (SQLite) as UTC."""
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


class InvalidResolutionError(ValueError):
    """The resolution is not a valid answer, or asks for something the item cannot do."""


@dataclass(frozen=True, slots=True)
class ReviewSla:
    """How long each tier may wait, and how long a claim lasts before it lapses."""

    verify_minutes: float = 60.0
    human_minutes: float = 240.0
    claim_minutes: float = 15.0

    def __post_init__(self) -> None:
        for name in ("verify_minutes", "human_minutes", "claim_minutes"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number of minutes")

    def minutes_for(self, tier: str) -> float:
        return self.verify_minutes if tier == "verify" else self.human_minutes

    def due(self, tier: str, created_at: datetime) -> datetime:
        return created_at + timedelta(minutes=self.minutes_for(tier))

    def new_item(self, decision: DecisionRecord) -> ReviewItemRecord | None:
        """The review item for a decision, or ``None`` if nobody needs to look at it."""
        if decision.tier not in REVIEW_TIERS:
            return None
        created = as_utc(decision.created_at)
        return ReviewItemRecord(
            id=uuid.uuid4(),
            decision_id=decision.id,
            question_id=decision.question_id,
            question_version=decision.question_version,
            tier=decision.tier,
            status="open",
            created_at=created,
            due_at=self.due(decision.tier, created),
            claims=0,
        )


def item_state(item: ReviewItemRecord, now: datetime) -> ReviewState:
    """``claimed`` only while the lease is live; an expired claim is ``available``."""
    if item.status in ("resolved", "superseded"):
        return item.status  # type: ignore[return-value]
    expires = item.claim_expires_at
    if item.status == "claimed" and expires is not None and as_utc(expires) > now:
        return "claimed"
    return "available"


def is_overdue(item: ReviewItemRecord | ReviewStatRow, now: datetime) -> bool:
    """Unresolved past its deadline, or resolved after it. Superseded items never are."""
    due = as_utc(item.due_at)
    if item.status == "resolved":
        return item.resolved_at is not None and as_utc(item.resolved_at) > due
    if item.status == "superseded":
        return False
    return now > due


class ReviewService:
    def __init__(
        self,
        catalog: QuestionCatalog,
        repository: ReviewRepository,
        sla: ReviewSla,
        *,
        clock: Clock = utc_now,
    ) -> None:
        self._catalog = catalog
        self._repository = repository
        self.sla = sla
        self._clock = clock

    def now(self) -> datetime:
        return as_utc(self._clock())

    def _lease(self, now: datetime, minutes: float | None) -> datetime:
        return now + timedelta(minutes=minutes or self.sla.claim_minutes)

    async def claim_next(
        self,
        reviewer: str,
        *,
        question_id: str | None = None,
        tier: ReviewTier | None = None,
        lease_minutes: float | None = None,
    ) -> tuple[ReviewItemRecord, DecisionRecord] | None:
        """Claim the most urgent available item (earliest deadline first)."""
        now = self.now()
        item = await self._repository.claim_next(
            reviewer,
            now=now,
            lease_until=self._lease(now, lease_minutes),
            question_id=question_id,
            tier=tier,
        )
        return None if item is None else await self._with_decision(item.id)

    async def claim(
        self, item_id: uuid.UUID, reviewer: str, *, lease_minutes: float | None = None
    ) -> tuple[ReviewItemRecord, DecisionRecord]:
        now = self.now()
        await self._repository.claim(
            item_id, reviewer, now=now, lease_until=self._lease(now, lease_minutes)
        )
        return await self._with_decision(item_id)

    async def release(
        self, item_id: uuid.UUID, reviewer: str
    ) -> tuple[ReviewItemRecord, DecisionRecord]:
        await self._repository.release(item_id, reviewer)
        return await self._with_decision(item_id)

    async def resolve(
        self,
        item_id: uuid.UUID,
        reviewer: str,
        *,
        outcome: Answer | None = None,
        confirm: bool = False,
        note: str | None = None,
    ) -> tuple[ReviewItemRecord, DecisionRecord, OutcomeRecord, bool]:
        """Resolve an item with an answer (or ``confirm`` the backend's) and record it as
        the decision's outcome. Returns ``(item, decision, outcome, created)``."""
        if confirm == (outcome is not None):
            raise InvalidResolutionError("give exactly one of 'outcome' or 'confirm'")
        found = await self._repository.get(item_id)
        if found is None:
            raise ReviewNotFoundError(f"unknown review item {item_id}")
        item, decision = found
        try:
            question = self._catalog.get(decision.question_id, decision.question_version)
        except KeyError as exc:
            raise InvalidResolutionError(
                f"question {decision.question_key} is no longer in the catalog"
            ) from exc
        try:
            proposed: Answer = decision.answer if confirm or outcome is None else outcome
            value = question.validate_answer(proposed)
        except ValueError as exc:
            raise InvalidResolutionError(str(exc)) from exc
        correct = question.is_correct(decision.answer, value)
        verdict: Verdict = "decided"
        if item.tier == "verify":
            verdict = "confirmed" if correct else "corrected"
        now = self.now()
        record = OutcomeRecord(
            decision_id=decision.id,
            outcome=value,
            correct=correct,
            source=REVIEW_SOURCE,
            observed_at=now,
        )
        stored, stored_outcome, created = await self._repository.resolve(
            item_id,
            reviewer,
            now=now,
            outcome=record,
            verdict=verdict,
            note=note[:MAX_NOTE_LENGTH] if note else None,
        )
        return stored, decision, stored_outcome, created

    async def get(self, item_id: uuid.UUID) -> tuple[ReviewItemRecord, DecisionRecord]:
        return await self._with_decision(item_id)

    async def _with_decision(self, item_id: uuid.UUID) -> tuple[ReviewItemRecord, DecisionRecord]:
        found = await self._repository.get(item_id)
        if found is None:
            raise ReviewNotFoundError(f"unknown review item {item_id}")
        return found

    async def find(self, **filters: Any) -> list[tuple[ReviewItemRecord, DecisionRecord]]:
        return await self._repository.find(now=self.now(), **filters)

    async def stats(
        self, *, question_id: str | None = None, since: datetime | None = None
    ) -> list[QueueStats]:
        rows = await self._repository.stat_rows(question_id=question_id, since=since)
        return summarize(rows, self.now())

    async def backfill(self, batch: int = 1000) -> int:
        """Enqueue verify and human decisions stored before the queue existed.

        Their deadlines count from the decision time, so old decisions arrive overdue:
        that is the truth about how long they have waited.
        """
        total = 0
        while True:
            decisions = await self._repository.unqueued(batch)
            items = [item for d in decisions if (item := self.sla.new_item(d)) is not None]
            added = await self._repository.add_many(items)
            total += added
            if len(decisions) < batch or added == 0:
                return total


@dataclass(slots=True)
class QueueStats:
    """Queue health for one question version and tier."""

    question_id: str
    question_version: int
    tier: str
    available: int = 0
    claimed: int = 0
    resolved: int = 0
    superseded: int = 0
    overdue: int = 0
    """Unresolved items past their deadline right now."""
    oldest_waiting_minutes: float | None = None
    resolved_within_sla: int = 0
    resolution_minutes: list[float] = field(default_factory=list)
    confirmed: int = 0
    corrected: int = 0
    agreed: int = 0

    @property
    def sla_hit_rate(self) -> float | None:
        return self.resolved_within_sla / self.resolved if self.resolved else None

    @property
    def median_minutes(self) -> float | None:
        return statistics.median(self.resolution_minutes) if self.resolution_minutes else None

    @property
    def p90_minutes(self) -> float | None:
        return _percentile(self.resolution_minutes, 0.9)

    @property
    def agreement_rate(self) -> float | None:
        """How often the backend's answer was right, by the reviewers' answers."""
        return self.agreed / self.resolved if self.resolved else None

    @property
    def agreement_interval(self) -> tuple[float, float] | None:
        """95% Wilson interval for the agreement rate."""
        if not self.resolved:
            return None
        return interval(self.agreed, self.resolved, 0.95, "wilson")


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least ``q`` of values at or below."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def summarize(rows: Iterable[ReviewStatRow], now: datetime) -> list[QueueStats]:
    groups: dict[tuple[str, int, str], QueueStats] = {}
    for row in rows:
        key = (row.question_id, row.question_version, row.tier)
        stats = groups.get(key)
        if stats is None:
            stats = groups[key] = QueueStats(*key)
        if row.status == "resolved":
            stats.resolved += 1
            assert row.resolved_at is not None
            stats.resolution_minutes.append((row.resolved_at - row.created_at).total_seconds() / 60)
            if not is_overdue(row, now):
                stats.resolved_within_sla += 1
            stats.confirmed += row.verdict == "confirmed"
            stats.corrected += row.verdict == "corrected"
            stats.agreed += bool(row.agreed)
            continue
        if row.status == "superseded":
            stats.superseded += 1
            continue
        live = row.status == "claimed" and row.claim_expires_at and row.claim_expires_at > now
        if live:
            stats.claimed += 1
        else:
            stats.available += 1
        if now > row.due_at:
            stats.overdue += 1
        waited = (now - row.created_at).total_seconds() / 60
        if stats.oldest_waiting_minutes is None or waited > stats.oldest_waiting_minutes:
            stats.oldest_waiting_minutes = waited
    return [groups[key] for key in sorted(groups)]


__all__ = [
    "REVIEW_SOURCE",
    "REVIEW_STATES",
    "REVIEW_TIERS",
    "InvalidResolutionError",
    "QueueStats",
    "ReviewService",
    "ReviewSla",
    "ReviewState",
    "ReviewTier",
    "Verdict",
    "as_utc",
    "is_overdue",
    "item_state",
    "summarize",
    "utc_now",
]
