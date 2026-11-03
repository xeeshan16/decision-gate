"""Threshold selection and conformal calibration over the labeled decisions in the database.

Both analyses use the *calibrated* view of each decision when a calibrator is active, and
both must not reuse the calibrator's own training decisions: a score function fitted on a
decision looks better on it than it will on new traffic, which would make the threshold
or the conformal quantile optimistic. So when a calibrator is used, only decisions made
after its training window (``trained_until``) are eligible.

The eligible decisions, oldest first, are then split by time: the older part selects the
threshold (or calibrates the conformal quantile) and the newer ``holdout`` part checks
what it actually achieves. With ``holdout=0`` everything is used for selection and there
is no check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from decision_gate.backends.base import answer_key
from decision_gate.bounds import BoundMethod
from decision_gate.calibrators import (
    AnyCalibrator,
    CalibrationError,
    TemperatureScaler,
    load_calibrator,
)
from decision_gate.conformal import (
    DEFAULT_ALPHA,
    ConformalError,
    ConformalEvaluation,
    ConformalMethod,
    ConformalPredictor,
    distribution_matrix,
    evaluate,
    fit_conformal,
    min_calibration_size,
)
from decision_gate.db.models import CalibratorRecord
from decision_gate.db.repository import CalibratorRepository, LabeledRow, OutcomeRepository
from decision_gate.questions import ChoiceQuestion, PredicateQuestion, ScoreQuestion
from decision_gate.refit import calibration_data
from decision_gate.thresholds import (
    DEFAULT_CONFIDENCE,
    DEFAULT_MIN_ACCEPTED,
    DEFAULT_TARGET_PRECISION,
    ThresholdCheck,
    ThresholdError,
    ThresholdSelection,
    apply_threshold,
    select_threshold,
)

AnyQuestion = PredicateQuestion | ChoiceQuestion | ScoreQuestion
DEFAULT_SELECTION_HOLDOUT = 0.3


class SelectionError(ValueError):
    """The request cannot be served with this question or data."""


@dataclass(frozen=True, slots=True)
class DataBasis:
    """Which decisions an analysis used and how they were scored."""

    question_id: str
    question_version: int
    labeled: int
    """Labeled decisions in the requested window."""
    excluded_training: int
    """Of those, decisions skipped because the calibrator was trained on them."""
    selection_count: int
    holdout_count: int
    calibrator: CalibratorRecord | None

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"

    @property
    def score_source(self) -> str:
        if self.calibrator is None:
            return "raw"
        return f"calibrator v{self.calibrator.version} ({self.calibrator.method})"


@dataclass(frozen=True, slots=True)
class ThresholdReport:
    basis: DataBasis
    selection: ThresholdSelection | None
    """``None`` when there were no decisions to select on."""
    check: ThresholdCheck | None
    """The selected threshold on the held-out decisions, when both exist."""


@dataclass(frozen=True, slots=True)
class ConformalReport:
    basis: DataBasis
    predictor: ConformalPredictor | None
    evaluation: ConformalEvaluation | None
    reason: str


def split_point(count: int, holdout: float) -> int:
    """How many of ``count`` chronologically ordered decisions are used for selection."""
    if not 0.0 <= holdout < 1.0:
        raise SelectionError("holdout must be within [0, 1)")
    if holdout == 0.0 or count == 0:
        return count
    return count - max(1, math.ceil(count * holdout))


def calibrated_distributions(
    rows: Sequence[LabeledRow], calibrator: AnyCalibrator | None
) -> list[dict[str, float]]:
    """Each row's distribution, temperature-scaled when the calibrator rescales them.

    Platt and isotonic calibrators only recalibrate the decided answer's probability, not
    the distribution, so the raw distribution is used with them.
    """
    result: list[dict[str, float]] = []
    for row in rows:
        if row.probabilities is None:
            raise SelectionError(
                "conformal prediction needs a probability distribution on every decision"
            )
        distribution = dict(row.probabilities)
        if isinstance(calibrator, TemperatureScaler):
            distribution = calibrator.scale(distribution)
        result.append(distribution)
    return result


class GateAnalyzer:
    def __init__(self, outcomes: OutcomeRepository, calibrators: CalibratorRepository) -> None:
        self._outcomes = outcomes
        self._calibrators = calibrators

    async def _rows(
        self,
        question: AnyQuestion,
        *,
        since: datetime | None,
        until: datetime | None,
        use_calibrator: bool,
    ) -> tuple[list[LabeledRow], int, CalibratorRecord | None, AnyCalibrator | None]:
        rows = await self._outcomes.labeled_rows(
            question_id=question.id,
            question_version=question.version,
            since=since,
            until=until,
        )
        record = (
            await self._calibrators.active(question.id, question.version)
            if use_calibrator
            else None
        )
        if record is None:
            return rows, 0, None, None
        try:
            calibrator = load_calibrator(record.method, record.params)
        except CalibrationError as exc:
            raise SelectionError(f"the active calibrator cannot be loaded: {exc}") from exc
        if record.trained_until is not None:
            cutoff = record.trained_until
            eligible = [r for r in rows if r.created_at is None or r.created_at > cutoff]
        else:  # pragma: no cover - the refit job always records the window
            eligible = rows
        return eligible, len(rows) - len(eligible), record, calibrator

    async def thresholds(
        self,
        question: AnyQuestion,
        *,
        target_precision: float = DEFAULT_TARGET_PRECISION,
        confidence: float = DEFAULT_CONFIDENCE,
        bound: BoundMethod = "clopper-pearson",
        min_accepted: int = DEFAULT_MIN_ACCEPTED,
        holdout: float = DEFAULT_SELECTION_HOLDOUT,
        since: datetime | None = None,
        until: datetime | None = None,
        use_calibrator: bool = True,
    ) -> ThresholdReport:
        """Select the automation threshold for one question version."""
        rows, excluded, record, calibrator = await self._rows(
            question, since=since, until=until, use_calibrator=use_calibrator
        )
        n_select = split_point(len(rows), holdout)
        basis = DataBasis(
            question_id=question.id,
            question_version=question.version,
            labeled=len(rows) + excluded,
            excluded_training=excluded,
            selection_count=n_select,
            holdout_count=len(rows) - n_select,
            calibrator=record,
        )
        if n_select == 0:
            return ThresholdReport(basis=basis, selection=None, check=None)
        data = calibration_data(rows)
        if calibrator is None:
            scores = data.confidences
        else:
            try:
                scores = calibrator.predict(data)
            except CalibrationError as exc:
                raise SelectionError(
                    f"the active calibrator cannot score these decisions: {exc}"
                ) from exc
        try:
            selection = select_threshold(
                scores[:n_select],
                data.correct[:n_select],
                target_precision=target_precision,
                confidence=confidence,
                bound=bound,
                min_accepted=min_accepted,
            )
        except ThresholdError as exc:
            raise SelectionError(str(exc)) from exc
        check = None
        if selection.threshold is not None and n_select < len(rows):
            check = apply_threshold(selection.threshold, scores[n_select:], data.correct[n_select:])
        return ThresholdReport(basis=basis, selection=selection, check=check)

    async def conformal(
        self,
        question: AnyQuestion,
        *,
        alpha: float = DEFAULT_ALPHA,
        method: ConformalMethod = "lac",
        holdout: float = DEFAULT_SELECTION_HOLDOUT,
        since: datetime | None = None,
        until: datetime | None = None,
        use_calibrator: bool = True,
    ) -> ConformalReport:
        """Calibrate a split-conformal predictor for a choice or predicate question."""
        if isinstance(question, ScoreQuestion):
            raise SelectionError(
                "conformal prediction sets need a label set; score questions have none"
            )
        if not 0.0 < alpha < 1.0:
            raise SelectionError("alpha must be strictly between 0 and 1")
        labels: tuple[str, ...] = (
            question.options if isinstance(question, ChoiceQuestion) else ("false", "true")
        )
        rows, excluded, record, calibrator = await self._rows(
            question, since=since, until=until, use_calibrator=use_calibrator
        )
        n_fit = split_point(len(rows), holdout)
        basis = DataBasis(
            question_id=question.id,
            question_version=question.version,
            labeled=len(rows) + excluded,
            excluded_training=excluded,
            selection_count=n_fit,
            holdout_count=len(rows) - n_fit,
            # Only a temperature scaler changes the distributions conformal sets use.
            calibrator=record if isinstance(calibrator, TemperatureScaler) else None,
        )
        if n_fit == 0:
            return ConformalReport(
                basis=basis, predictor=None, evaluation=None, reason="no labeled decisions"
            )
        try:
            matrix = distribution_matrix(
                calibrated_distributions(rows, calibrator),
                labels,
            )
            index = {label: i for i, label in enumerate(labels)}
            truths = np.array([index[answer_key(row.outcome)] for row in rows], dtype=np.intp)
            predictor = fit_conformal(
                matrix[:n_fit], truths[:n_fit], labels, alpha=alpha, method=method
            )
        except (ConformalError, KeyError) as exc:
            raise SelectionError(f"cannot build prediction sets: {exc}") from exc
        evaluation = (
            evaluate(predictor, matrix[n_fit:], truths[n_fit:]) if n_fit < len(rows) else None
        )
        if predictor.is_trivial:
            reason = (
                f"{n_fit} calibration decisions are too few for alpha={alpha} "
                f"(need {min_calibration_size(alpha)}); every label is in every set"
            )
        else:
            reason = (
                f"{method} sets with quantile {predictor.quantile:.4f} cover the true label "
                f"with probability >= {1 - alpha:.0%}"
            )
        return ConformalReport(
            basis=basis, predictor=predictor, evaluation=evaluation, reason=reason
        )


__all__ = [
    "DEFAULT_SELECTION_HOLDOUT",
    "ConformalReport",
    "DataBasis",
    "GateAnalyzer",
    "SelectionError",
    "ThresholdReport",
    "calibrated_distributions",
    "split_point",
]
