"""HTTP endpoints for shadow mode: status, per-decision answers, replay and comparison."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from decision_gate.calibrators import CalibrationError
from decision_gate.db.models import ShadowDecisionRecord
from decision_gate.db.shadow import ShadowRepository, ShadowSummary
from decision_gate.questions import Answer, QuestionCatalog
from decision_gate.refit import MethodChoice
from decision_gate.reviews import as_utc
from decision_gate.routing import RoutingConfig
from decision_gate.shadow import (
    DEFAULT_ALPHA,
    DEFAULT_HOLDOUT,
    DEFAULT_MAX_FAILURE_RATE,
    DEFAULT_MIN_COVERAGE_GAIN,
    DEFAULT_MIN_PAIRED,
    ShadowCompareError,
    ShadowComparison,
    compare_stored,
)
from decision_gate.shadow_runner import ShadowRunner

Mode = Literal["live", "replay"]


class ShadowSummaryOut(BaseModel):
    question: str
    question_version: int
    candidate: str
    total: int
    ok: int
    errors: int
    timeouts: int
    failure_rate: float
    labeled: int = Field(description="Successful answers whose decision has an outcome.")
    last_at: datetime | None

    @classmethod
    def from_summary(cls, s: ShadowSummary) -> ShadowSummaryOut:
        return cls(
            question=s.question_id,
            question_version=s.question_version,
            candidate=s.candidate,
            total=s.total,
            ok=s.ok,
            errors=s.errors,
            timeouts=s.timeouts,
            failure_rate=s.failure_rate,
            labeled=s.labeled,
            last_at=as_utc(s.last_at) if s.last_at else None,
        )


class ShadowAnswerOut(BaseModel):
    id: uuid.UUID
    decision_id: uuid.UUID
    candidate: str
    mode: Mode
    status: Literal["ok", "error", "timeout"]
    answer: Answer | None
    confidence: float | None
    probabilities: dict[str, float] | None
    backend: str | None
    model_version: str | None
    latency_ms: float
    error: str | None
    created_at: datetime

    @classmethod
    def from_record(cls, r: ShadowDecisionRecord) -> ShadowAnswerOut:
        return cls(
            id=r.id,
            decision_id=r.decision_id,
            candidate=r.candidate,
            mode=r.mode,
            status=r.status,
            answer=r.answer,
            confidence=r.confidence,
            probabilities=r.probabilities,
            backend=r.backend,
            model_version=r.model_version,
            latency_ms=r.latency_ms,
            error=r.error,
            created_at=as_utc(r.created_at),
        )


class ReplayIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    version: int | None = Field(default=None, ge=1, description="Defaults to the latest.")
    limit: int = Field(default=1000, ge=1, le=100_000)
    labeled_only: bool = Field(
        default=True, description="Only decisions with an outcome, i.e. comparable ones."
    )
    concurrency: int = Field(default=8, ge=1, le=64)


def comparison_out(c: ShadowComparison) -> dict[str, Any]:
    """A JSON-ready view of a comparison."""
    out = asdict(c)
    out["question"] = out.pop("question_id")
    out["candidate_report"] = out.pop("challenger")
    if c.accuracy is not None:
        out["accuracy"]["interval"] = list(c.accuracy.interval)
    for side in ("primary", "candidate_report"):
        if out[side] is not None:
            out[side]["model_versions"] = list(out[side]["model_versions"])
    out["reasons"] = list(c.reasons)
    return out


def shadow_router(
    catalog: QuestionCatalog,
    repository: ShadowRepository,
    routing: RoutingConfig,
    runner: ShadowRunner | None,
) -> APIRouter:
    api = APIRouter(prefix="/v1/shadow", tags=["shadow"])

    def question_or_404(question_id: str, version: int | None) -> Any:
        try:
            return catalog.get(question_id, version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    @api.get("/status")
    async def shadow_status(question: str | None = None) -> dict[str, Any]:
        """The configured candidate, its in-process counters and stored answer counts."""
        summaries = await repository.summaries(question_id=question)
        config: dict[str, Any] = {"enabled": runner is not None}
        if runner is not None:
            config.update(
                candidate=runner.candidate,
                backend=runner.backend.name,
                sample_rate=runner.sample_rate,
                questions=sorted(runner.questions) if runner.questions else None,
                timeout_s=runner.timeout_s,
                max_pending=runner.max_pending,
                pending=runner.pending,
                counters=asdict(runner.counters),
            )
        return {
            "shadow": config,
            "candidates": [ShadowSummaryOut.from_summary(s) for s in summaries],
        }

    @api.get("/decisions/{decision_id}")
    async def shadow_answers(decision_id: uuid.UUID) -> dict[str, list[ShadowAnswerOut]]:
        """Every candidate's answer to one decision."""
        records = await repository.for_decision(decision_id)
        return {"answers": [ShadowAnswerOut.from_record(r) for r in records]}

    @api.post(
        "/replay",
        responses={
            404: {"description": "Unknown question."},
            409: {"description": "Shadow mode is not configured."},
        },
    )
    async def replay(request: ReplayIn) -> dict[str, Any]:
        """Run the candidate over stored decisions it has not answered, oldest first.

        Gives a new candidate labeled history to be compared on at once. Needs stored
        inputs (``DGATE_STORE_INPUTS``); decisions without one are counted and skipped.
        """
        if runner is None:
            raise HTTPException(
                status_code=409, detail="shadow mode is not configured (DGATE_SHADOW_BACKEND)"
            )
        question = question_or_404(request.question, request.version)
        result = await runner.replay(
            question,
            limit=request.limit,
            labeled_only=request.labeled_only,
            concurrency=request.concurrency,
        )
        out = asdict(result)
        out["question"] = out.pop("question_id")
        return out

    @api.get(
        "/compare",
        responses={
            404: {"description": "Unknown question."},
            422: {"description": "Invalid options, or no candidate named or configured."},
        },
    )
    async def compare(
        question: str,
        version: Annotated[int | None, Query(ge=1)] = None,
        candidate: Annotated[
            str | None, Query(description="Defaults to the configured candidate.")
        ] = None,
        mode: Mode | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        holdout: Annotated[float, Query(gt=0, lt=1)] = DEFAULT_HOLDOUT,
        method: MethodChoice = "auto",
        min_paired: Annotated[int, Query(ge=40)] = DEFAULT_MIN_PAIRED,
        alpha: Annotated[float, Query(gt=0, lt=1)] = DEFAULT_ALPHA,
        min_coverage_gain: Annotated[float, Query(ge=0, lt=1)] = DEFAULT_MIN_COVERAGE_GAIN,
        max_failure_rate: Annotated[float, Query(ge=0, le=1)] = DEFAULT_MAX_FAILURE_RATE,
    ) -> dict[str, Any]:
        """Compare the candidate with the primary backend on paired, labeled decisions.

        Both are calibrated on the older part and scored on the newer part: accuracy with
        McNemar's exact test, and auto-tier coverage under the question's routing policy.
        The verdict is ``promote``, ``keep`` or ``insufficient``, with the reasons.
        """
        target = question_or_404(question, version)
        name = candidate or (runner.candidate if runner is not None else None)
        if name is None:
            raise HTTPException(
                status_code=422, detail="name a candidate; shadow mode is not configured"
            )
        try:
            result = await compare_stored(
                repository,
                target,
                name,
                policy=routing.policy_for(target.id),
                mode=mode,
                since=since,
                until=until,
                holdout=holdout,
                method=method,
                min_paired=min_paired,
                alpha=alpha,
                min_coverage_gain=min_coverage_gain,
                max_failure_rate=max_failure_rate,
            )
        except (ShadowCompareError, CalibrationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return comparison_out(result)

    return api


__all__ = ["ReplayIn", "comparison_out", "shadow_router"]
