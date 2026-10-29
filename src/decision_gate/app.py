"""FastAPI application factory."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from decision_gate import __version__
from decision_gate.backends import DecisionBackend, FakeBackend
from decision_gate.catalog import load_catalog
from decision_gate.config import Settings
from decision_gate.questions import QuestionCatalog


def build_backend(settings: Settings) -> DecisionBackend:
    if settings.backend == "fake":
        return FakeBackend(overconfidence=settings.fake_overconfidence, seed=settings.fake_seed)
    raise ValueError(f"unsupported backend {settings.backend!r}")  # pragma: no cover


def create_app(
    settings: Settings | None = None,
    *,
    catalog: QuestionCatalog | None = None,
    backend: DecisionBackend | None = None,
) -> FastAPI:
    settings = settings or Settings()
    if catalog is None:
        catalog = (
            load_catalog(settings.questions_file) if settings.questions_file else QuestionCatalog()
        )
    backend = backend or build_backend(settings)

    app = FastAPI(
        title="decision-gate",
        version=__version__,
        summary="Calibrated thresholds and human-in-the-loop routing for AI decision models.",
    )
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.backend = backend

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz() -> dict[str, Any]:
        """Readiness: configuration loaded and a backend is available."""
        return {
            "status": "ready",
            "version": __version__,
            "backend": backend.name,
            "questions": len(catalog.questions),
        }

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

    return app
