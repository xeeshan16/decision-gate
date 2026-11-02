"""The refit job: fit calibrators on labeled decisions and promote one only if it helps.

For one question version the job:

1. loads every labeled decision (optionally within a decision-time window), oldest first;
2. splits them by time: the oldest ``1 - holdout`` fraction trains, the newest fraction is
   held out. A chronological split is deliberately pessimistic: it measures how well a
   calibrator fitted on the past does on decisions it has never seen, which is how it
   will be used;
3. fits every candidate method on the training split and scores each, the raw scores and
   the currently active calibrator on the held-out split (ECE, Brier, log loss);
4. picks the candidate with the lowest held-out log loss (a proper scoring rule; Brier
   breaks ties) and stores it. It becomes ``active`` only if it beats both the raw scores
   and the active calibrator on that same held-out data; otherwise it is stored as
   ``rejected`` (or ``candidate`` when activation was not requested) with the reason.

The stored calibrator is the one fitted on the training split, i.e. exactly the model the
held-out numbers describe. Nothing is stored when there are too few labels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, get_args

import numpy as np

from decision_gate.backends.base import answer_key
from decision_gate.calibrators import (
    METHODS,
    AnyCalibrator,
    CalibrationData,
    CalibrationError,
    Method,
    fit_calibrator,
    load_calibrator,
)
from decision_gate.db.models import CalibratorRecord
from decision_gate.db.repository import CalibratorRepository, LabeledRow, OutcomeRepository
from decision_gate.metrics import calibration_metrics

MethodChoice = Literal["auto", "platt", "isotonic", "temperature"]
METHOD_CHOICES: tuple[MethodChoice, ...] = get_args(MethodChoice)
RefitOutcome = Literal["activated", "candidate", "rejected", "skipped"]

DEFAULT_HOLDOUT = 0.3
DEFAULT_MIN_COUNT = 100
# Isotonic regression has as many free steps as the data allow; below this many training
# decisions it mostly memorises noise, so "auto" leaves it out.
ISOTONIC_MIN_TRAIN = 200
EVAL_BINS = 10


def calibration_data(rows: Sequence[LabeledRow]) -> CalibrationData:
    """Arrays for the calibrators. Distributions are kept only if every row has one."""
    confidences = np.array([row.confidence for row in rows], dtype=np.float64)
    correct = np.array([float(row.correct) for row in rows], dtype=np.float64)
    if rows and all(
        row.probabilities is not None
        and isinstance(row.outcome, bool | str)
        and isinstance(row.answer, bool | str)
        for row in rows
    ):
        return CalibrationData(
            confidences,
            correct,
            distributions=tuple(row.probabilities or {} for row in rows),
            answers=tuple(answer_key(row.answer) for row in rows),
            truths=tuple(answer_key(row.outcome) for row in rows),
        )
    return CalibrationData(confidences, correct)


def score(predicted: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Held-out metrics for one set of probabilities."""
    m = calibration_metrics(predicted, correct, n_bins=EVAL_BINS)
    return {
        "ece": m.ece,
        "brier": m.brier,
        "log_loss": m.log_loss,
        "mean_confidence": m.mean_confidence,
        "accuracy": m.accuracy,
    }


def candidate_methods(choice: MethodChoice, train: CalibrationData) -> list[Method]:
    if choice != "auto":
        if choice == "temperature" and not train.has_distributions:
            raise CalibrationError(
                "temperature scaling needs a probability distribution on every decision; "
                "score questions and backends without distributions can use platt or isotonic"
            )
        return [choice]
    methods: list[Method] = ["platt"]
    if len(train) >= ISOTONIC_MIN_TRAIN:
        methods.append("isotonic")
    if train.has_distributions:
        methods.append("temperature")
    return methods


def _better(a: dict[str, float], b: dict[str, float]) -> bool:
    """Strictly lower held-out log loss, or equal log loss and lower Brier."""
    if not math.isclose(a["log_loss"], b["log_loss"], rel_tol=0.0, abs_tol=1e-12):
        return a["log_loss"] < b["log_loss"]
    return a["brier"] < b["brier"] - 1e-12


@dataclass(frozen=True, slots=True)
class RefitResult:
    question_id: str
    question_version: int
    outcome: RefitOutcome
    reason: str
    labeled: int
    record: CalibratorRecord | None = None

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"


class RefitJob:
    def __init__(self, outcomes: OutcomeRepository, calibrators: CalibratorRepository) -> None:
        self._outcomes = outcomes
        self._calibrators = calibrators

    async def refit(
        self,
        question_id: str,
        question_version: int,
        *,
        method: MethodChoice = "auto",
        holdout: float = DEFAULT_HOLDOUT,
        min_count: int = DEFAULT_MIN_COUNT,
        since: datetime | None = None,
        until: datetime | None = None,
        activate: bool = True,
    ) -> RefitResult:
        if method not in METHOD_CHOICES:
            raise CalibrationError(f"unknown method {method!r}; choose from {METHOD_CHOICES}")
        if not 0.0 < holdout < 1.0:
            raise CalibrationError("holdout must be strictly between 0 and 1")
        if min_count < 2:
            raise CalibrationError("min_count must be at least 2")
        rows = await self._outcomes.labeled_rows(
            question_id=question_id, question_version=question_version, since=since, until=until
        )
        # labeled_rows orders by question then decision time, so one question version's
        # rows are already oldest first.
        result_base: dict[str, Any] = {
            "question_id": question_id,
            "question_version": question_version,
            "labeled": len(rows),
        }
        if len(rows) < min_count:
            return RefitResult(
                **result_base,
                outcome="skipped",
                reason=f"{len(rows)} labeled decisions, fewer than min_count={min_count}",
            )
        n_holdout = max(1, math.ceil(len(rows) * holdout))
        n_train = len(rows) - n_holdout
        data = calibration_data(rows)
        train, test = data.subset(slice(0, n_train)), data.subset(slice(n_train, None))
        methods = candidate_methods(method, train)

        raw = score(test.confidences, test.correct)
        fitted: dict[Method, AnyCalibrator] = {}
        scores: dict[str, dict[str, float]] = {}
        for name in methods:
            calibrator = fit_calibrator(name, train)
            fitted[name] = calibrator
            scores[name] = score(calibrator.predict(test), test.correct)
        best = min(methods, key=lambda name: (scores[name]["log_loss"], scores[name]["brier"]))

        incumbent_record = await self._calibrators.active(question_id, question_version)
        incumbent: dict[str, Any] | None = None
        if incumbent_record is not None:
            try:
                current = load_calibrator(incumbent_record.method, incumbent_record.params)
                incumbent = {
                    "id": str(incumbent_record.id),
                    "version": incumbent_record.version,
                    "method": incumbent_record.method,
                    **score(current.predict(test), test.correct),
                }
            except CalibrationError as exc:
                # E.g. a temperature scaler while some held-out decisions lack a
                # distribution. It cannot be compared, so it does not block a new one.
                incumbent = {
                    "id": str(incumbent_record.id),
                    "version": incumbent_record.version,
                    "method": incumbent_record.method,
                    "error": str(exc),
                }

        beats_raw = _better(scores[best], raw)
        beats_incumbent = (
            incumbent is None or "error" in incumbent or _better(scores[best], incumbent)
        )
        ll = scores[best]["log_loss"]
        if not beats_raw:
            status, reason = (
                "rejected",
                f"{best} held-out log loss {ll:.4f} does not beat raw {raw['log_loss']:.4f}",
            )
        elif not beats_incumbent:
            assert incumbent is not None
            status, reason = (
                "rejected",
                f"{best} held-out log loss {ll:.4f} does not beat active "
                f"v{incumbent['version']} ({incumbent['method']}) {incumbent['log_loss']:.4f}",
            )
        else:
            vs = f"raw {raw['log_loss']:.4f}"
            if incumbent is not None and "log_loss" in incumbent:
                vs += f", active v{incumbent['version']} {incumbent['log_loss']:.4f}"
            status = "active" if activate else "candidate"
            reason = f"{best} held-out log loss {ll:.4f} beats {vs}"

        record = CalibratorRecord(
            question_id=question_id,
            question_version=question_version,
            method=best,
            params=fitted[best].params(),
            status=status,
            reason=reason[:500],
            train_count=n_train,
            holdout_count=n_holdout,
            trained_from=rows[0].created_at,
            trained_until=rows[n_train - 1].created_at,
            evaluation={
                "split": "chronological",
                "holdout_fraction": holdout,
                "bins": EVAL_BINS,
                "raw": raw,
                "calibrated": scores[best],
                "candidates": scores,
                "incumbent": incumbent,
            },
        )
        stored = await self._calibrators.add(record)
        outcome: RefitOutcome = "activated" if status == "active" else status  # type: ignore[assignment]
        return RefitResult(**result_base, outcome=outcome, reason=reason, record=stored)

    async def refit_all(self, **options: Any) -> list[RefitResult]:
        """Refit every question version that has at least one labeled decision."""
        results: list[RefitResult] = []
        for stats in await self._outcomes.stats():
            if stats.labeled == 0:
                continue
            results.append(await self.refit(stats.question_id, stats.question_version, **options))
        return results


__all__ = [
    "DEFAULT_HOLDOUT",
    "DEFAULT_MIN_COUNT",
    "ISOTONIC_MIN_TRAIN",
    "METHODS",
    "METHOD_CHOICES",
    "MethodChoice",
    "RefitJob",
    "RefitOutcome",
    "RefitResult",
    "calibration_data",
    "candidate_methods",
]
