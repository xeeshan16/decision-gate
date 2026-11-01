"""FastAPI application factory."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, get_args

from fastapi import FastAPI, Header, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from decision_gate import __version__
from decision_gate.backends import DecisionBackend
from decision_gate.backends.base import BackendError
from decision_gate.backends.factory import build_backend
from decision_gate.calibration import CalibrationReport, GroupField, build_reports
from decision_gate.catalog import load_catalog
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import DecisionRecord, OutcomeRecord
from decision_gate.db.repository import DecisionRepository, LabelStats, OutcomeRepository
from decision_gate.metrics import BinStrategy, ReliabilityBin
from decision_gate.questions import Answer, QuestionCatalog
from decision_gate.service import (
    BackendTimeoutError,
    DecisionService,
    IdempotencyConflictError,
    InvalidOutcomeError,
    OutcomeConflictError,
    OutcomeService,
    UnknownDecisionError,
)

MAX_OUTCOME_BATCH = 1000


def _utc(moment: datetime) -> datetime:
    # SQLite drops the offset; values are stored in UTC.
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Question id, e.g. 'refund.eligible'.")
    version: int | None = Field(default=None, ge=1, description="Defaults to the latest.")
    input: dict[str, Any] = Field(description="What the question is asked about.")
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Retries with the same key return the original decision.",
    )


class OutcomeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: uuid.UUID
    outcome: Answer = Field(description="The true answer, in the question's answer type.")
    source: str = Field(
        default="api", min_length=1, max_length=64, description="Where the label came from."
    )
    observed_at: datetime | None = Field(
        default=None, description="When the truth became known. Defaults to now."
    )


class OutcomeOut(BaseModel):
    decision_id: uuid.UUID
    outcome: Answer
    correct: bool = Field(description="Whether the decision's answer matched the outcome.")
    source: str
    observed_at: datetime
    recorded_at: datetime

    @classmethod
    def from_record(cls, record: OutcomeRecord) -> OutcomeOut:
        return cls(
            decision_id=record.decision_id,
            outcome=record.outcome,
            correct=record.correct,
            source=record.source,
            observed_at=_utc(record.observed_at),
            recorded_at=_utc(record.recorded_at),
        )


class OutcomeBatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcomes: list[OutcomeIn] = Field(min_length=1, max_length=MAX_OUTCOME_BATCH)


BatchStatus = Literal["created", "duplicate", "conflict", "unknown_decision", "invalid"]


class OutcomeBatchItem(BaseModel):
    decision_id: uuid.UUID
    status: BatchStatus
    detail: str | None = None
    correct: bool | None = None


class OutcomeBatchOut(BaseModel):
    results: list[OutcomeBatchItem]
    counts: dict[str, int]


class LabelStatsOut(BaseModel):
    question: str
    question_version: int
    decisions: int
    labeled: int
    label_rate: float
    accuracy: float | None = Field(description="Raw accuracy over labeled decisions.")
    mean_confidence: float | None = Field(
        description="Mean raw confidence over the same labeled decisions."
    )
    confidence_gap: float | None = Field(
        description="mean_confidence - accuracy. Positive means the backend is overconfident."
    )

    @classmethod
    def from_stats(cls, stats: LabelStats) -> LabelStatsOut:
        accuracy = stats.accuracy
        gap = (
            stats.mean_confidence - accuracy
            if accuracy is not None and stats.mean_confidence is not None
            else None
        )
        return cls(
            question=stats.question_id,
            question_version=stats.question_version,
            decisions=stats.decisions,
            labeled=stats.labeled,
            label_rate=stats.label_rate,
            accuracy=accuracy,
            mean_confidence=stats.mean_confidence,
            confidence_gap=gap,
        )


class ReliabilityBinOut(BaseModel):
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None
    gap: float | None = Field(description="mean_confidence - accuracy; positive = overconfident.")

    @classmethod
    def from_bin(cls, b: ReliabilityBin) -> ReliabilityBinOut:
        return cls(
            lower=b.lower,
            upper=b.upper,
            count=b.count,
            mean_confidence=b.mean_confidence,
            accuracy=b.accuracy,
            gap=b.gap,
        )


class CalibrationReportOut(BaseModel):
    question: str
    question_version: int
    backend: str | None = Field(description="Null unless grouped by backend.")
    model_version: str | None = Field(description="Null unless grouped by model_version.")
    count: int = Field(description="Labeled decisions in this group.")
    accuracy: float
    mean_confidence: float
    overconfidence: float = Field(description="mean_confidence - accuracy.")
    ece: float = Field(description="Expected calibration error over the bins.")
    ece_interval: tuple[float, float] | None = Field(
        description="95% bootstrap interval for the ECE, when requested."
    )
    mce: float = Field(description="Largest per-bin |accuracy - confidence|.")
    brier: float = Field(description="Top-label Brier score.")
    log_loss: float = Field(description="Top-label log loss (nats).")
    multiclass_brier: float | None = Field(
        description="Brier score of the full distribution, when every decision has one."
    )
    multiclass_log_loss: float | None
    bins: list[ReliabilityBinOut]

    @classmethod
    def from_report(cls, report: CalibrationReport) -> CalibrationReportOut:
        m = report.metrics
        return cls(
            question=report.question_id,
            question_version=report.question_version,
            backend=report.backend,
            model_version=report.model_version,
            count=m.count,
            accuracy=m.accuracy,
            mean_confidence=m.mean_confidence,
            overconfidence=m.overconfidence,
            ece=m.ece,
            ece_interval=m.ece_interval,
            mce=m.mce,
            brier=m.brier,
            log_loss=m.log_loss,
            multiclass_brier=m.multiclass_brier,
            multiclass_log_loss=m.multiclass_log_loss,
            bins=[ReliabilityBinOut.from_bin(b) for b in m.bins],
        )


class CalibrationMetricsOut(BaseModel):
    n_bins: int
    strategy: BinStrategy
    group_by: list[GroupField]
    reports: list[CalibrationReportOut]


class DecisionOut(BaseModel):
    id: uuid.UUID
    question: str
    question_version: int
    answer: Answer
    confidence: float = Field(description="Raw, uncalibrated confidence from the backend.")
    probabilities: dict[str, float] | None
    backend: str
    model_version: str
    latency_ms: float
    idempotency_key: str | None
    created_at: datetime
    outcome: OutcomeOut | None = Field(
        default=None, description="Ground truth, once it has been recorded."
    )

    @classmethod
    def from_record(
        cls, record: DecisionRecord, outcome: OutcomeRecord | None = None
    ) -> DecisionOut:
        return cls(
            id=record.id,
            question=record.question_id,
            question_version=record.question_version,
            answer=record.answer,
            confidence=record.confidence,
            probabilities=record.probabilities,
            backend=record.backend,
            model_version=record.model_version,
            latency_ms=record.latency_ms,
            idempotency_key=record.idempotency_key,
            created_at=_utc(record.created_at),
            outcome=OutcomeOut.from_record(outcome) if outcome is not None else None,
        )


def create_app(
    settings: Settings | None = None,
    *,
    catalog: QuestionCatalog | None = None,
    backend: DecisionBackend | None = None,
    engine: AsyncEngine | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if catalog is None:
        catalog = (
            load_catalog(settings.questions_file) if settings.questions_file else QuestionCatalog()
        )
    backend = backend or build_backend(settings)
    engine = engine or make_engine(settings.database_url)
    sessions = make_sessionmaker(engine)
    repository = DecisionRepository(sessions)
    outcome_repository = OutcomeRepository(sessions)
    outcome_service = OutcomeService(catalog, repository, outcome_repository)
    service = DecisionService(
        backend,
        repository,
        timeout_s=settings.backend_timeout_s,
        store_inputs=settings.store_inputs,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        close = getattr(backend, "aclose", None)
        if close is not None:
            await close()
        await engine.dispose()

    app = FastAPI(
        title="decision-gate",
        version=__version__,
        summary="Calibrated thresholds and human-in-the-loop routing for AI decision models.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.backend = backend
    app.state.engine = engine
    app.state.outcome_service = outcome_service

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz() -> JSONResponse:
        """Readiness: configuration loaded and the database answers."""
        body: dict[str, Any] = {
            "status": "ready",
            "version": __version__,
            "backend": backend.name,
            "questions": len(catalog.questions),
            "database": "ok",
        }
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception:
            body.update(status="not ready", database="unavailable")
            return JSONResponse(body, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return JSONResponse(body)

    @app.get("/v1/questions", tags=["questions"])
    async def list_questions() -> dict[str, Any]:
        return {"questions": [q.model_dump(mode="json") for q in catalog.questions]}

    @app.get("/v1/questions/{question_id}", tags=["questions"])
    async def get_question(question_id: str, version: int | None = None) -> dict[str, Any]:
        try:
            question = catalog.get(question_id, version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        return question.model_dump(mode="json")

    @app.post(
        "/v1/decide",
        tags=["decisions"],
        status_code=status.HTTP_201_CREATED,
        responses={
            200: {"description": "Idempotent replay of an earlier decision."},
            404: {"description": "Unknown question."},
            409: {"description": "Idempotency key reused for a different request."},
            502: {"description": "The backend failed or returned an invalid decision."},
            504: {"description": "The backend timed out."},
        },
    )
    async def decide(
        request: DecideRequest,
        response: Response,
        idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
    ) -> DecisionOut:
        """Ask the backend, record the decision and return it."""
        try:
            question = catalog.get(request.question, request.version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        key = request.idempotency_key or idempotency_key
        try:
            record, created = await service.decide(question, request.input, idempotency_key=key)
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except BackendTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if not created:
            response.status_code = status.HTTP_200_OK
        return DecisionOut.from_record(record)

    @app.get("/v1/decisions/{decision_id}", tags=["decisions"])
    async def get_decision(decision_id: uuid.UUID) -> DecisionOut:
        record = await repository.get(decision_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown decision")
        return DecisionOut.from_record(record, await outcome_repository.get(decision_id))

    @app.post(
        "/v1/outcomes",
        tags=["outcomes"],
        status_code=status.HTTP_201_CREATED,
        responses={
            200: {"description": "The same outcome was already recorded."},
            404: {"description": "Unknown decision."},
            409: {"description": "A different outcome was already recorded."},
            422: {"description": "Not a valid answer to the decision's question."},
        },
    )
    async def record_outcome(request: OutcomeIn, response: Response) -> OutcomeOut:
        """Record the ground truth for a decision. Safe to retry."""
        try:
            record, created = await outcome_service.record(
                request.decision_id,
                request.outcome,
                source=request.source,
                observed_at=request.observed_at,
            )
        except UnknownDecisionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except OutcomeConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidOutcomeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not created:
            response.status_code = status.HTTP_200_OK
        return OutcomeOut.from_record(record)

    @app.post("/v1/outcomes/batch", tags=["outcomes"])
    async def record_outcomes(request: OutcomeBatchIn) -> OutcomeBatchOut:
        """Record up to 1,000 outcomes, e.g. a backfill. Each item succeeds or fails alone."""
        results: list[OutcomeBatchItem] = []
        for item in request.outcomes:
            results.append(await _record_one(item))
        counts = dict.fromkeys(get_args(BatchStatus), 0)
        for result in results:
            counts[result.status] += 1
        return OutcomeBatchOut(results=results, counts=counts)

    async def _record_one(item: OutcomeIn) -> OutcomeBatchItem:
        def failed(kind: BatchStatus, exc: Exception) -> OutcomeBatchItem:
            return OutcomeBatchItem(decision_id=item.decision_id, status=kind, detail=str(exc))

        try:
            record, created = await outcome_service.record(
                item.decision_id, item.outcome, source=item.source, observed_at=item.observed_at
            )
        except UnknownDecisionError as exc:
            return failed("unknown_decision", exc)
        except OutcomeConflictError as exc:
            return failed("conflict", exc)
        except InvalidOutcomeError as exc:
            return failed("invalid", exc)
        return OutcomeBatchItem(
            decision_id=item.decision_id,
            status="created" if created else "duplicate",
            correct=record.correct,
        )

    @app.get("/v1/outcomes/stats", tags=["outcomes"])
    async def outcome_stats(question: str | None = None) -> dict[str, list[LabelStatsOut]]:
        """Label coverage and raw accuracy against raw confidence, per question version."""
        stats = await outcome_repository.stats(question)
        return {"questions": [LabelStatsOut.from_stats(s) for s in stats]}

    @app.get("/v1/calibration/metrics", tags=["calibration"])
    async def calibration_metrics(
        question: str | None = None,
        version: Annotated[int | None, Query(ge=1)] = None,
        group_by: Annotated[list[GroupField] | None, Query()] = None,
        bins: Annotated[int, Query(ge=1, le=100)] = 10,
        strategy: BinStrategy = "uniform",
        since: datetime | None = None,
        until: datetime | None = None,
        bootstrap: Annotated[
            int, Query(ge=0, le=1000, description="Resamples for a 95% ECE interval.")
        ] = 0,
        min_count: Annotated[int, Query(ge=1, description="Skip smaller groups.")] = 1,
    ) -> CalibrationMetricsOut:
        """ECE, MCE, Brier, log loss and reliability bins over labeled decisions.

        One report per question version, split further by ``group_by=backend`` and/or
        ``group_by=model_version``. The time window applies to the decision time.
        """
        fields = list(dict.fromkeys(group_by or []))
        rows = await outcome_repository.labeled_rows(
            question_id=question, question_version=version, since=since, until=until
        )
        reports = build_reports(
            rows,
            group_by=fields,
            n_bins=bins,
            strategy=strategy,
            bootstrap=bootstrap,
            min_count=min_count,
        )
        return CalibrationMetricsOut(
            n_bins=bins,
            strategy=strategy,
            group_by=fields,
            reports=[CalibrationReportOut.from_report(r) for r in reports],
        )

    return app
