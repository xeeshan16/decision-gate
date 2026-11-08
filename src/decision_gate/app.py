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
from decision_gate.alerts import AlertSink
from decision_gate.backends import DecisionBackend
from decision_gate.backends.base import BackendError
from decision_gate.backends.factory import build_backend, build_shadow_backend
from decision_gate.calibration import CalibrationReport, GroupField, build_reports
from decision_gate.calibrators import CalibrationError
from decision_gate.catalog import load_catalog
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import CalibratorRecord, DecisionRecord, OutcomeRecord
from decision_gate.db.repository import (
    ActivationConflictError,
    CalibratorRepository,
    DecisionRepository,
    DriftCheckRepository,
    LabelStats,
    OutcomeRepository,
    ReviewRepository,
    RoutingRuleRepository,
)
from decision_gate.db.shadow import ShadowRepository
from decision_gate.drift_api import drift_router
from decision_gate.drift_monitor import DriftMonitor
from decision_gate.gating_api import gating_router
from decision_gate.metrics import BinStrategy, ReliabilityBin
from decision_gate.questions import Answer, QuestionCatalog
from decision_gate.refit import (
    DEFAULT_HOLDOUT,
    DEFAULT_MIN_COUNT,
    MethodChoice,
    RefitJob,
    RefitOutcome,
    RefitResult,
)
from decision_gate.reviews import Clock, ReviewService, utc_now
from decision_gate.reviews_api import reviews_router
from decision_gate.routing import Router, RoutingConfig, RoutingRefresher, Tier, load_routing
from decision_gate.routing_api import routing_router
from decision_gate.selection import GateAnalyzer
from decision_gate.service import (
    BackendTimeoutError,
    DecisionService,
    IdempotencyConflictError,
    InvalidOutcomeError,
    OutcomeConflictError,
    OutcomeService,
    UnknownDecisionError,
)
from decision_gate.shadow_api import shadow_router
from decision_gate.shadow_runner import ShadowRunner

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


CalibratorStatus = Literal["active", "candidate", "rejected", "retired"]


class CalibratorOut(BaseModel):
    id: uuid.UUID
    question: str
    question_version: int
    version: int = Field(description="Fit number within this question version (1, 2, ...).")
    method: str
    params: dict[str, Any]
    status: CalibratorStatus
    reason: str
    train_count: int
    holdout_count: int
    trained_from: datetime | None
    trained_until: datetime | None
    evaluation: dict[str, Any] = Field(
        description="Held-out metrics for the raw scores, every candidate and the incumbent."
    )
    created_at: datetime
    activated_at: datetime | None

    @classmethod
    def from_record(cls, record: CalibratorRecord) -> CalibratorOut:
        return cls(
            id=record.id,
            question=record.question_id,
            question_version=record.question_version,
            version=record.version,
            method=record.method,
            params=record.params,
            status=record.status,
            reason=record.reason,
            train_count=record.train_count,
            holdout_count=record.holdout_count,
            trained_from=_utc(record.trained_from) if record.trained_from else None,
            trained_until=_utc(record.trained_until) if record.trained_until else None,
            evaluation=record.evaluation,
            created_at=_utc(record.created_at),
            activated_at=_utc(record.activated_at) if record.activated_at else None,
        )


class RefitIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    version: int | None = Field(default=None, ge=1, description="Defaults to the latest.")
    method: MethodChoice = Field(
        default="auto", description="'auto' tries every method that fits the data."
    )
    holdout: float = Field(
        default=DEFAULT_HOLDOUT, gt=0, lt=1, description="Newest fraction held out."
    )
    min_count: int = Field(default=DEFAULT_MIN_COUNT, ge=2, le=1_000_000)
    since: datetime | None = None
    until: datetime | None = None
    activate: bool = Field(
        default=True, description="Activate the new fit if it beats raw and the active one."
    )


class RefitOut(BaseModel):
    question: str
    question_version: int
    outcome: RefitOutcome
    reason: str
    labeled: int
    calibrator: CalibratorOut | None

    @classmethod
    def from_result(cls, result: RefitResult) -> RefitOut:
        return cls(
            question=result.question_id,
            question_version=result.question_version,
            outcome=result.outcome,
            reason=result.reason,
            labeled=result.labeled,
            calibrator=CalibratorOut.from_record(result.record) if result.record else None,
        )


class DecisionOut(BaseModel):
    id: uuid.UUID
    question: str
    question_version: int
    answer: Answer
    confidence: float = Field(description="Raw, uncalibrated confidence from the backend.")
    calibrated_confidence: float | None = Field(
        description="Probability that the answer is correct, from the active calibrator."
    )
    tier: Tier | None = Field(
        description="auto: use the answer; verify: a person confirms it; human: a person "
        "decides. Null for decisions recorded before routing existed."
    )
    reason: str | None = Field(description="Why the decision got its tier.")
    answer_set: list[Answer] | None = Field(
        description="Conformal answer set for decisions a person handles, when configured: "
        "contains the true answer with the policy's coverage rate."
    )
    routing_rule_id: uuid.UUID | None
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
            calibrated_confidence=record.calibrated_confidence,
            tier=record.tier,
            reason=record.routing_reason,
            answer_set=record.answer_set,
            routing_rule_id=record.routing_rule_id,
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
    routing: RoutingConfig | None = None,
    clock: Clock = utc_now,
    alerter: AlertSink | None = None,
    shadow_backend: DecisionBackend | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if catalog is None:
        catalog = (
            load_catalog(settings.questions_file) if settings.questions_file else QuestionCatalog()
        )
    if routing is None:
        routing = load_routing(settings.routing_file) if settings.routing_file else RoutingConfig()
    backend = backend or build_backend(settings)
    engine = engine or make_engine(settings.database_url)
    sessions = make_sessionmaker(engine)
    repository = DecisionRepository(sessions)
    outcome_repository = OutcomeRepository(sessions)
    outcome_service = OutcomeService(catalog, repository, outcome_repository, clock=clock)
    calibrator_repository = CalibratorRepository(sessions)
    refit_job = RefitJob(outcome_repository, calibrator_repository)
    rule_repository = RoutingRuleRepository(sessions)
    analyzer = GateAnalyzer(outcome_repository, calibrator_repository)
    router = Router(
        routing,
        rule_repository,
        calibrator_repository,
        cache_ttl_s=settings.routing_cache_ttl_s,
    )
    shadow_repository = ShadowRepository(sessions)
    shadow_backend = shadow_backend or build_shadow_backend(settings)
    shadow: ShadowRunner | None = None
    if shadow_backend is not None:
        shadow = ShadowRunner(
            shadow_backend,
            shadow_repository,
            candidate=settings.shadow_candidate or shadow_backend.name,
            timeout_s=settings.shadow_timeout_s,
            sample_rate=settings.shadow_sample_rate,
            questions=settings.shadow_question_ids,
            max_pending=settings.shadow_max_pending,
        )
    service = DecisionService(
        backend,
        repository,
        timeout_s=settings.backend_timeout_s,
        store_inputs=settings.store_inputs,
        router=router,
        review_sla=settings.review_sla,
        shadow=shadow,
        clock=clock,
    )
    review_service = ReviewService(
        catalog, ReviewRepository(sessions), settings.review_sla, clock=clock
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if shadow is not None:
            await shadow.drain(timeout_s=settings.shadow_timeout_s)
            await shadow.aclose()
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
    app.state.router = router
    app.state.shadow = shadow

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

    @app.post(
        "/v1/calibrators/refit",
        tags=["calibrators"],
        responses={
            404: {"description": "Unknown question."},
            422: {"description": "Invalid options, e.g. temperature on a score question."},
        },
    )
    async def refit_calibrator(request: RefitIn) -> RefitOut:
        """Fit calibrators on labeled decisions and activate the best one if it helps.

        The newest ``holdout`` fraction of labeled decisions is held out. The best method
        by held-out log loss is stored as a new version; it is activated only if it beats
        both the raw confidences and the active calibrator on that held-out data.
        """
        version = request.version
        if version is None:
            try:
                version = catalog.get(request.question).version
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        try:
            result = await refit_job.refit(
                request.question,
                version,
                method=request.method,
                holdout=request.holdout,
                min_count=request.min_count,
                since=request.since,
                until=request.until,
                activate=request.activate,
            )
        except CalibrationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        router.invalidate()
        return RefitOut.from_result(result)

    @app.get("/v1/calibrators", tags=["calibrators"])
    async def list_calibrators(
        question: str | None = None,
        version: Annotated[int | None, Query(ge=1)] = None,
        status: CalibratorStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> dict[str, list[CalibratorOut]]:
        """The calibrator registry, newest fit first within each question version."""
        records = await calibrator_repository.find(
            question_id=question, question_version=version, status=status, limit=limit
        )
        return {"calibrators": [CalibratorOut.from_record(r) for r in records]}

    @app.get("/v1/calibrators/{calibrator_id}", tags=["calibrators"])
    async def get_calibrator(calibrator_id: uuid.UUID) -> CalibratorOut:
        record = await calibrator_repository.get(calibrator_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown calibrator")
        return CalibratorOut.from_record(record)

    @app.post(
        "/v1/calibrators/{calibrator_id}/activate",
        tags=["calibrators"],
        responses={
            404: {"description": "Unknown calibrator."},
            409: {"description": "A concurrent activation won."},
        },
    )
    async def activate_calibrator(calibrator_id: uuid.UUID) -> CalibratorOut:
        """Make this calibrator the active one, e.g. to roll back. Retires the current one."""
        try:
            record = await calibrator_repository.activate(calibrator_id)
        except ActivationConflictError as exc:  # pragma: no cover - needs an exact race
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="unknown calibrator")
        router.invalidate()
        return CalibratorOut.from_record(record)

    drift_repository = DriftCheckRepository(sessions)
    monitor = DriftMonitor(
        routing,
        decisions=repository,
        outcomes=outcome_repository,
        calibrators=calibrator_repository,
        rules=rule_repository,
        checks=drift_repository,
        analyzer=analyzer,
        alerter=alerter if alerter is not None else settings.drift_alerter(),
        clock=clock,
    )
    app.include_router(reviews_router(review_service))
    app.include_router(gating_router(catalog, analyzer))
    app.include_router(
        routing_router(
            catalog,
            router,
            RoutingRefresher(
                routing,
                analyzer,
                calibrator_repository,
                rule_repository,
                drift_repository,
                clock=clock,
            ),
            rule_repository,
        )
    )
    app.include_router(drift_router(catalog, monitor, drift_repository, router))
    app.include_router(shadow_router(catalog, shadow_repository, routing, shadow))
    return app
