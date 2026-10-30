"""FastAPI application factory."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from decision_gate import __version__
from decision_gate.backends import DecisionBackend
from decision_gate.backends.base import BackendError
from decision_gate.backends.factory import build_backend
from decision_gate.catalog import load_catalog
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import DecisionRecord
from decision_gate.db.repository import DecisionRepository
from decision_gate.questions import Answer, QuestionCatalog
from decision_gate.service import BackendTimeoutError, DecisionService, IdempotencyConflictError


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

    @classmethod
    def from_record(cls, record: DecisionRecord) -> DecisionOut:
        created = record.created_at
        if created.tzinfo is None:  # SQLite drops the offset; values are stored in UTC.
            created = created.replace(tzinfo=UTC)
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
            created_at=created,
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
    repository = DecisionRepository(make_sessionmaker(engine))
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
        return DecisionOut.from_record(record)

    return app
