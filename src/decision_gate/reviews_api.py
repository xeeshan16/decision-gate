"""HTTP endpoints for the human review queue: list, claim, release and resolve."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable
from datetime import datetime
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from decision_gate.db.models import DecisionRecord, ReviewItemRecord
from decision_gate.db.repository import ReviewConflictError, ReviewNotFoundError
from decision_gate.questions import Answer
from decision_gate.reviews import (
    MAX_NOTE_LENGTH,
    InvalidResolutionError,
    QueueStats,
    ReviewService,
    ReviewState,
    ReviewTier,
    Verdict,
    as_utc,
    is_overdue,
    item_state,
)

Reviewer = Annotated[
    str, Field(min_length=1, max_length=200, pattern=r"\S", description="Who is acting.")
]
LeaseMinutes = Annotated[
    float | None,
    Field(default=None, gt=0, le=480, description="Claim length. Defaults to the server's."),
]

T = TypeVar("T")


def _opt(moment: datetime | None) -> datetime | None:
    return None if moment is None else as_utc(moment)


class ReviewDecisionOut(BaseModel):
    """What a reviewer needs to see to handle the decision."""

    id: uuid.UUID
    answer: Answer = Field(description="The backend's proposed answer.")
    confidence: float
    calibrated_confidence: float | None
    answer_set: list[Answer] | None = Field(
        description="Conformal answer set, when the policy asks for one."
    )
    routing_reason: str | None
    input: dict[str, Any] | None = Field(description="Null when inputs are not stored.")
    probabilities: dict[str, float] | None
    backend: str
    model_version: str
    created_at: datetime


class ReviewItemOut(BaseModel):
    id: uuid.UUID
    decision_id: uuid.UUID
    question: str
    question_version: int
    tier: ReviewTier
    state: ReviewState = Field(
        description="available (open, or the claim lapsed), claimed, resolved or superseded."
    )
    created_at: datetime
    due_at: datetime = Field(description="SLA deadline.")
    overdue: bool = Field(
        description="Unresolved past the deadline, or resolved after it. Superseded: false."
    )
    claims: int = Field(description="Times claimed; above one means a lease lapsed or was let go.")
    claimed_by: str | None
    claimed_at: datetime | None
    claim_expires_at: datetime | None
    resolved_by: str | None
    resolved_at: datetime | None
    resolution: Answer | None = Field(description="The reviewer's answer (the outcome).")
    verdict: Verdict | None = Field(
        description="verify items: confirmed or corrected. human items: decided."
    )
    agreed: bool | None = Field(description="Whether the backend's answer was correct.")
    note: str | None
    decision: ReviewDecisionOut

    @classmethod
    def build(
        cls, item: ReviewItemRecord, decision: DecisionRecord, now: datetime
    ) -> ReviewItemOut:
        return cls(
            id=item.id,
            decision_id=item.decision_id,
            question=item.question_id,
            question_version=item.question_version,
            tier=item.tier,
            state=item_state(item, now),
            created_at=as_utc(item.created_at),
            due_at=as_utc(item.due_at),
            overdue=is_overdue(item, now),
            claims=item.claims,
            claimed_by=item.claimed_by,
            claimed_at=_opt(item.claimed_at),
            claim_expires_at=_opt(item.claim_expires_at),
            resolved_by=item.resolved_by,
            resolved_at=_opt(item.resolved_at),
            resolution=item.resolution,
            verdict=item.verdict,
            agreed=item.agreed,
            note=item.note,
            decision=ReviewDecisionOut(
                id=decision.id,
                answer=decision.answer,
                confidence=decision.confidence,
                calibrated_confidence=decision.calibrated_confidence,
                answer_set=decision.answer_set,
                routing_reason=decision.routing_reason,
                input=decision.input,
                probabilities=decision.probabilities,
                backend=decision.backend,
                model_version=decision.model_version,
                created_at=as_utc(decision.created_at),
            ),
        )


class ClaimNextIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: Reviewer
    question: str | None = Field(default=None, description="Only items for this question.")
    tier: ReviewTier | None = Field(default=None, description="Only verify or human items.")
    lease_minutes: LeaseMinutes = None


class ClaimIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: Reviewer
    lease_minutes: LeaseMinutes = None


class ReleaseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: Reviewer


class ResolveIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: Reviewer
    outcome: Answer | None = Field(
        default=None, description="The correct answer, in the question's answer type."
    )
    confirm: bool = Field(
        default=False, description="The backend's answer is correct (instead of 'outcome')."
    )
    note: str | None = Field(default=None, max_length=MAX_NOTE_LENGTH)

    @model_validator(mode="after")
    def _one_answer(self) -> ResolveIn:
        if self.confirm == (self.outcome is not None):
            raise ValueError("give exactly one of 'outcome' or 'confirm: true'")
        return self


class ResolveOut(BaseModel):
    item: ReviewItemOut
    correct: bool = Field(description="Whether the backend's answer matched the resolution.")


class QueueStatsOut(BaseModel):
    question: str
    question_version: int
    tier: ReviewTier
    available: int
    claimed: int
    resolved: int
    superseded: int = Field(description="Left the queue because an outcome arrived elsewhere.")
    overdue: int = Field(description="Unresolved items past their deadline now.")
    oldest_waiting_minutes: float | None
    sla_hit_rate: float | None = Field(description="Share of resolved items resolved on time.")
    median_minutes: float | None = Field(description="Median minutes from decision to resolution.")
    p90_minutes: float | None
    confirmed: int
    corrected: int
    agreement_rate: float | None = Field(
        description="Share of resolved items where the backend's answer was correct."
    )
    agreement_interval: tuple[float, float] | None = Field(description="95% Wilson interval.")

    @classmethod
    def from_stats(cls, stats: QueueStats) -> QueueStatsOut:
        return cls(
            question=stats.question_id,
            question_version=stats.question_version,
            tier=stats.tier,
            available=stats.available,
            claimed=stats.claimed,
            resolved=stats.resolved,
            superseded=stats.superseded,
            overdue=stats.overdue,
            oldest_waiting_minutes=stats.oldest_waiting_minutes,
            sla_hit_rate=stats.sla_hit_rate,
            median_minutes=stats.median_minutes,
            p90_minutes=stats.p90_minutes,
            confirmed=stats.confirmed,
            corrected=stats.corrected,
            agreement_rate=stats.agreement_rate,
            agreement_interval=stats.agreement_interval,
        )


ERRORS: dict[int | str, dict[str, Any]] = {
    404: {"description": "Unknown review item."},
    409: {"description": "Claimed by someone else, already resolved, or superseded."},
}


def reviews_router(reviews: ReviewService) -> APIRouter:
    api = APIRouter(prefix="/v1/reviews", tags=["reviews"])

    async def guarded(action: Awaitable[T]) -> T:
        try:
            return await action
        except ReviewNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ReviewConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidResolutionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def out(found: tuple[ReviewItemRecord, DecisionRecord]) -> ReviewItemOut:
        return ReviewItemOut.build(found[0], found[1], reviews.now())

    @api.get("")
    async def list_reviews(
        state: ReviewState | None = None,
        question: str | None = None,
        tier: ReviewTier | None = None,
        claimed_by: str | None = None,
        overdue: bool | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> dict[str, list[ReviewItemOut]]:
        """Review items, earliest deadline first."""
        found = await reviews.find(
            state=state,
            question_id=question,
            tier=tier,
            claimed_by=claimed_by,
            overdue=overdue,
            limit=limit,
        )
        now = reviews.now()
        return {"items": [ReviewItemOut.build(item, d, now) for item, d in found]}

    @api.get("/stats")
    async def review_stats(
        question: str | None = None,
        since: Annotated[
            datetime | None, Query(description="Only items created at or after this time.")
        ] = None,
    ) -> dict[str, Any]:
        """Queue health per question version and tier: backlog, SLA and agreement."""
        stats = await reviews.stats(question_id=question, since=since)
        return {
            "sla_minutes": {
                "verify": reviews.sla.verify_minutes,
                "human": reviews.sla.human_minutes,
                "claim": reviews.sla.claim_minutes,
            },
            "queues": [QueueStatsOut.from_stats(s) for s in stats],
        }

    @api.post(
        "/claim",
        response_model=ReviewItemOut,
        responses={204: {"description": "Nothing is waiting."}},
    )
    async def claim_next(request: ClaimNextIn) -> Any:
        """Claim the available item with the earliest deadline."""
        found = await reviews.claim_next(
            request.reviewer,
            question_id=request.question,
            tier=request.tier,
            lease_minutes=request.lease_minutes,
        )
        if found is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        return out(found)

    @api.post("/backfill")
    async def backfill() -> dict[str, int]:
        """Enqueue verify and human decisions recorded before the queue existed."""
        return {"enqueued": await reviews.backfill()}

    @api.get("/{item_id}", responses={404: ERRORS[404]})
    async def get_review(item_id: uuid.UUID) -> ReviewItemOut:
        return out(await guarded(reviews.get(item_id)))

    @api.post("/{item_id}/claim", responses=ERRORS)
    async def claim(item_id: uuid.UUID, request: ClaimIn) -> ReviewItemOut:
        """Claim this item, or renew your own claim on it."""
        return out(
            await guarded(
                reviews.claim(item_id, request.reviewer, lease_minutes=request.lease_minutes)
            )
        )

    @api.post("/{item_id}/release", responses=ERRORS)
    async def release(item_id: uuid.UUID, request: ReleaseIn) -> ReviewItemOut:
        """Give the item back to the queue without resolving it."""
        return out(await guarded(reviews.release(item_id, request.reviewer)))

    @api.post(
        "/{item_id}/resolve",
        status_code=status.HTTP_201_CREATED,
        responses={
            **ERRORS,
            200: {"description": "The same resolution was already recorded."},
            422: {"description": "Not a valid answer to the decision's question."},
        },
    )
    async def resolve(item_id: uuid.UUID, request: ResolveIn, response: Response) -> ResolveOut:
        """Resolve the item and record the answer as the decision's outcome."""
        item, decision, outcome, created = await guarded(
            reviews.resolve(
                item_id,
                request.reviewer,
                outcome=request.outcome,
                confirm=request.confirm,
                note=request.note,
            )
        )
        if not created:
            response.status_code = status.HTTP_200_OK
        return ResolveOut(
            item=ReviewItemOut.build(item, decision, reviews.now()), correct=outcome.correct
        )

    return api


__all__ = ["ClaimIn", "ClaimNextIn", "ResolveIn", "ReviewItemOut", "reviews_router"]
