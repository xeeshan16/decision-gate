"""HTTP endpoints for threshold selection and conformal prediction sets."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from decision_gate.bounds import BoundMethod
from decision_gate.conformal import DEFAULT_ALPHA, ConformalEvaluation, ConformalMethod
from decision_gate.questions import QuestionCatalog
from decision_gate.selection import (
    DEFAULT_SELECTION_HOLDOUT,
    AnyQuestion,
    ConformalReport,
    DataBasis,
    GateAnalyzer,
    SelectionError,
    ThresholdReport,
)
from decision_gate.thresholds import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MIN_ACCEPTED,
    DEFAULT_TARGET_PRECISION,
    ThresholdPoint,
    curve,
)


class _WindowIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    version: int | None = Field(default=None, ge=1, description="Defaults to the latest.")
    holdout: float = Field(
        default=DEFAULT_SELECTION_HOLDOUT,
        ge=0,
        lt=1,
        description="Newest fraction of eligible decisions held out to check the result.",
    )
    since: datetime | None = None
    until: datetime | None = None
    use_calibrator: bool = Field(
        default=True,
        description="Score with the active calibrator (skipping its training decisions).",
    )


class ThresholdSelectIn(_WindowIn):
    target_precision: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description=f"Minimum precision of automated decisions (default "
        f"{DEFAULT_TARGET_PRECISION}).",
    )
    max_error_rate: float | None = Field(
        default=None, gt=0, lt=1, description="Alternative to target_precision: 1 - precision."
    )
    confidence: float = Field(default=DEFAULT_CONFIDENCE, gt=0, lt=1)
    bound: BoundMethod = "clopper-pearson"
    min_accepted: int = Field(default=DEFAULT_MIN_ACCEPTED, ge=1, le=1_000_000)
    curve_points: int = Field(default=10, ge=0, le=100)

    @model_validator(mode="after")
    def _one_target(self) -> ThresholdSelectIn:
        if self.target_precision is not None and self.max_error_rate is not None:
            raise ValueError("give target_precision or max_error_rate, not both")
        return self

    @property
    def target(self) -> float:
        if self.max_error_rate is not None:
            return 1.0 - self.max_error_rate
        return self.target_precision or DEFAULT_TARGET_PRECISION


class ConformalFitIn(_WindowIn):
    alpha: float = Field(
        default=DEFAULT_ALPHA, gt=0, lt=1, description="Allowed miss rate: coverage >= 1 - alpha."
    )
    method: ConformalMethod = "lac"


class BasisOut(BaseModel):
    question: str
    question_version: int
    labeled: int = Field(description="Labeled decisions in the requested window.")
    excluded_training: int = Field(
        description="Skipped because the active calibrator was trained on them."
    )
    reused_training: int = Field(
        description="Calibrator training decisions kept because the calibrator preserves "
        "their order, so reusing them cannot bias the result."
    )
    selection_count: int
    holdout_count: int
    score_source: str
    calibrator_id: uuid.UUID | None
    calibrator_version: int | None

    @classmethod
    def from_basis(cls, basis: DataBasis) -> BasisOut:
        return cls(
            question=basis.question_id,
            question_version=basis.question_version,
            labeled=basis.labeled,
            excluded_training=basis.excluded_training,
            reused_training=basis.reused_training,
            selection_count=basis.selection_count,
            holdout_count=basis.holdout_count,
            score_source=basis.score_source,
            calibrator_id=basis.calibrator.id if basis.calibrator else None,
            calibrator_version=basis.calibrator.version if basis.calibrator else None,
        )


class ThresholdPointOut(BaseModel):
    threshold: float
    accepted: int
    correct: int
    coverage: float
    precision: float
    error_rate: float
    lower_bound: float

    @classmethod
    def from_point(cls, point: ThresholdPoint) -> ThresholdPointOut:
        return cls(
            threshold=point.threshold,
            accepted=point.accepted,
            correct=point.correct,
            coverage=point.coverage,
            precision=point.precision,
            error_rate=point.error_rate,
            lower_bound=point.lower_bound,
        )


class HoldoutCheckOut(BaseModel):
    total: int
    accepted: int
    correct: int
    coverage: float
    precision: float | None


class ThresholdOut(BaseModel):
    basis: BasisOut
    target_precision: float
    max_error_rate: float
    confidence: float
    bound: BoundMethod
    threshold: float | None = Field(
        description="Automate decisions scoring at least this; null means automate none."
    )
    selected: ThresholdPointOut | None
    stopped_at: ThresholdPointOut | None
    tested: int
    candidates: int
    curve: list[ThresholdPointOut]
    holdout: HoldoutCheckOut | None
    reason: str

    @classmethod
    def from_report(cls, report: ThresholdReport, request: ThresholdSelectIn) -> ThresholdOut:
        selection = report.selection
        check = report.check

        def point(value: ThresholdPoint | None) -> ThresholdPointOut | None:
            return ThresholdPointOut.from_point(value) if value is not None else None

        return cls(
            basis=BasisOut.from_basis(report.basis),
            target_precision=request.target,
            max_error_rate=1.0 - request.target,
            confidence=request.confidence,
            bound=request.bound,
            threshold=selection.threshold if selection else None,
            selected=point(selection.selected) if selection else None,
            stopped_at=point(selection.stopped_at) if selection else None,
            tested=selection.tested if selection else 0,
            candidates=len(selection.candidates) if selection else 0,
            curve=[ThresholdPointOut.from_point(p) for p in curve(selection, request.curve_points)]
            if selection
            else [],
            holdout=HoldoutCheckOut(
                total=check.total,
                accepted=check.accepted,
                correct=check.correct,
                coverage=check.coverage,
                precision=check.precision,
            )
            if check
            else None,
            reason=selection.reason if selection else "no labeled decisions to select on",
        )


class ConformalEvaluationOut(BaseModel):
    count: int
    coverage: float
    mean_size: float
    singleton_rate: float
    singleton_accuracy: float | None
    empty_rate: float
    size_counts: dict[int, int]

    @classmethod
    def from_evaluation(cls, evaluation: ConformalEvaluation) -> ConformalEvaluationOut:
        return cls(
            count=evaluation.count,
            coverage=evaluation.coverage,
            mean_size=evaluation.mean_size,
            singleton_rate=evaluation.singleton_rate,
            singleton_accuracy=evaluation.singleton_accuracy,
            empty_rate=evaluation.empty_rate,
            size_counts=evaluation.size_counts,
        )


class ConformalOut(BaseModel):
    basis: BasisOut
    method: ConformalMethod
    alpha: float
    target_coverage: float
    quantile: float | None = Field(description="Null when every label is in every set.")
    trivial: bool
    labels: list[str]
    calibration_count: int
    params: dict[str, Any] | None = Field(
        description="Everything needed to rebuild the predictor and form sets."
    )
    holdout: ConformalEvaluationOut | None
    reason: str

    @classmethod
    def from_report(cls, report: ConformalReport, request: ConformalFitIn) -> ConformalOut:
        predictor = report.predictor
        return cls(
            basis=BasisOut.from_basis(report.basis),
            method=request.method,
            alpha=request.alpha,
            target_coverage=1.0 - request.alpha,
            quantile=None if predictor is None or predictor.is_trivial else predictor.quantile,
            trivial=predictor is None or predictor.is_trivial,
            labels=list(predictor.labels) if predictor else [],
            calibration_count=predictor.calibration_count if predictor else 0,
            params=predictor.params() if predictor else None,
            holdout=ConformalEvaluationOut.from_evaluation(report.evaluation)
            if report.evaluation
            else None,
            reason=report.reason,
        )


def gating_router(catalog: QuestionCatalog, analyzer: GateAnalyzer) -> APIRouter:
    router = APIRouter(tags=["gating"])

    def question_for(question_id: str, version: int | None) -> AnyQuestion:
        try:
            return catalog.get(question_id, version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    errors: dict[int | str, dict[str, Any]] = {
        404: {"description": "Unknown question."},
        422: {"description": "Invalid options, or the data cannot support the request."},
    }

    @router.post("/v1/thresholds/select", responses=errors)
    async def select_threshold(request: ThresholdSelectIn) -> ThresholdOut:
        """Pick the lowest automation threshold that meets a precision target with confidence.

        Candidates are tested from the highest threshold down (fixed-sequence testing), and
        each must have a one-sided lower confidence bound on precision at or above the
        target. The newest ``holdout`` fraction is kept back to check the result.
        """
        question = question_for(request.question, request.version)
        try:
            report = await analyzer.thresholds(
                question,
                target_precision=request.target,
                confidence=request.confidence,
                bound=request.bound,
                min_accepted=request.min_accepted,
                holdout=request.holdout,
                since=request.since,
                until=request.until,
                use_calibrator=request.use_calibrator,
            )
        except SelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ThresholdOut.from_report(report, request)

    @router.post("/v1/conformal/fit", responses=errors)
    async def fit_conformal(request: ConformalFitIn) -> ConformalOut:
        """Calibrate split-conformal prediction sets for a choice or predicate question.

        The sets contain the true label with probability at least ``1 - alpha`` on traffic
        exchangeable with the calibration decisions. The newest ``holdout`` fraction checks
        the coverage and set sizes actually achieved.
        """
        question = question_for(request.question, request.version)
        try:
            report = await analyzer.conformal(
                question,
                alpha=request.alpha,
                method=request.method,
                holdout=request.holdout,
                since=request.since,
                until=request.until,
                use_calibrator=request.use_calibrator,
            )
        except SelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ConformalOut.from_report(report, request)

    return router


__all__ = [
    "ConformalFitIn",
    "ConformalOut",
    "ThresholdOut",
    "ThresholdSelectIn",
    "gating_router",
]
