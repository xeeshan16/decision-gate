"""Shadow comparisons: is a candidate backend better than the one in production?

A candidate (a new model, a new prompt version, another vendor) answers the same inputs as
the primary backend without affecting any response (see :mod:`decision_gate.shadow_runner`).
Once outcomes arrive for those decisions, both models can be scored on exactly the same
inputs and the same ground truth. This module does that scoring.

Raw confidences are not comparable across models: a new model may simply be more
overconfident. What matters is how each model performs *after calibration* and what that
means for routing. So, for each side separately and identically:

1. the paired, labeled decisions are ordered by decision time and split: the oldest
   ``1 - holdout`` fraction trains a calibrator, the newest fraction is held out (the same
   chronological split the refit job uses, for the same reason);
2. the calibrator is fitted on the training split only (``auto`` picks a method with an
   inner chronological split of the training data, never by looking at the held-out set);
3. on the held-out split it reports raw and calibrated ECE, Brier and log loss, and runs
   the same fixed-sequence threshold selection the router uses under the question's
   routing policy, which says how much of the traffic each model could automate within
   the error budget.

Two things are compared on the held-out pairs:

* **Accuracy**, with McNemar's exact test. Both models answered the same items, so the
  test only looks at the items where exactly one of them was right. This is far more
  sensitive than comparing two independent accuracy figures.
* **Auto-tier coverage** at the policy's error budget. Each side's threshold is selected
  and evaluated on the same held-out data, which is optimistic for both in the same way;
  the difference is what is compared, not the absolute numbers.

Per-answer Brier and log loss are reported but do not decide anything: each model
forecasts the correctness of *its own* answer, so the two scores describe different events
(a model that is always wrong and says so scores perfectly).

The verdict is ``promote`` only if the candidate fails rarely enough, is not significantly
less accurate, does not automate meaningfully less, and is either significantly more
accurate or automates meaningfully more. Otherwise ``keep``; ``insufficient`` when there
are too few paired labels to say.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from statistics import NormalDist
from typing import Any, Literal

import numpy as np

from decision_gate.backends.base import AnyQuestion
from decision_gate.bounds import regularized_incomplete_beta
from decision_gate.calibrators import CalibrationData, CalibrationError, Method, fit_calibrator
from decision_gate.db.repository import LabeledRow
from decision_gate.db.shadow import PairedRow, ShadowRepository, ShadowSummary
from decision_gate.refit import MethodChoice, calibration_data, candidate_methods, score
from decision_gate.routing import RoutingPolicy
from decision_gate.thresholds import ThresholdError, select_threshold

Verdict = Literal["promote", "keep", "insufficient"]

DEFAULT_HOLDOUT = 0.5
DEFAULT_MIN_PAIRED = 200
DEFAULT_ALPHA = 0.05
DEFAULT_MIN_COVERAGE_GAIN = 0.02
DEFAULT_MAX_FAILURE_RATE = 0.01
# Fewest training decisions to fit a calibrator on, and fewest for an inner split.
MIN_TRAIN = 20
INNER_HOLDOUT = 0.3


class ShadowCompareError(ValueError):
    """The comparison options are invalid."""


# --- Statistics --------------------------------------------------------------------------


def binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``."""
    if n < 0 or not 0.0 <= p <= 1.0:
        raise ValueError("need n >= 0 and p within [0, 1]")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return regularized_incomplete_beta(1.0 - p, n - k, k + 1)


def mcnemar_exact(primary_only: int, candidate_only: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts.

    Under the null hypothesis (both models equally accurate), each item where exactly one
    model is right is equally likely to favour either, so the smaller count is
    ``Binomial(b + c, 1/2)``.
    """
    if primary_only < 0 or candidate_only < 0:
        raise ValueError("counts must be non-negative")
    discordant = primary_only + candidate_only
    if discordant == 0:
        return 1.0
    return min(1.0, 2.0 * binomial_cdf(min(primary_only, candidate_only), discordant, 0.5))


def paired_difference_interval(
    primary_only: int, candidate_only: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wald interval for ``accuracy(candidate) - accuracy(primary)`` on paired items."""
    if total < 1:
        raise ValueError("total must be at least 1")
    if primary_only + candidate_only > total:
        raise ValueError("discordant counts cannot exceed the total")
    delta = (candidate_only - primary_only) / total
    variance = (primary_only + candidate_only) / total - delta * delta
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    half = z * math.sqrt(max(variance, 0.0) / total)
    return max(-1.0, delta - half), min(1.0, delta + half)


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


# --- Reports -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AutoTier:
    """What threshold selection on one side's calibrated held-out scores gives."""

    threshold: float | None
    coverage: float
    accepted: int
    error_rate: float | None
    error_upper_bound: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class SideReport:
    name: str
    model_versions: tuple[str, ...]
    method: str
    calibrator: dict[str, Any]
    raw: dict[str, float]
    calibrated: dict[str, float]
    auto: AutoTier
    latency_p50_ms: float | None
    latency_p95_ms: float | None


@dataclass(frozen=True, slots=True)
class PairedAccuracy:
    total: int
    both_correct: int
    primary_only: int
    """Items the primary got right and the candidate got wrong."""
    candidate_only: int
    both_wrong: int
    delta: float
    """``accuracy(candidate) - accuracy(primary)``."""
    interval: tuple[float, float]
    p_value: float


@dataclass(frozen=True, slots=True)
class Reliability:
    total: int
    failures: int
    rate: float


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    question_id: str
    question_version: int
    candidate: str
    verdict: Verdict
    reasons: tuple[str, ...]
    paired: int
    train: int
    holdout: int
    options: dict[str, Any]
    reliability: Reliability | None = None
    accuracy: PairedAccuracy | None = None
    primary: SideReport | None = None
    challenger: SideReport | None = None
    """The candidate's report (``candidate`` is its name)."""
    policy: dict[str, Any] = field(default_factory=dict)

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"


# --- Comparison --------------------------------------------------------------------------


def _side_rows(
    question: AnyQuestion, rows: Sequence[PairedRow]
) -> tuple[list[LabeledRow], list[LabeledRow]]:
    primary: list[LabeledRow] = []
    challenger: list[LabeledRow] = []
    for row in rows:
        outcome = question.validate_answer(row.outcome)
        primary.append(
            LabeledRow(
                question_id=question.id,
                question_version=question.version,
                backend="primary",
                model_version="",
                confidence=row.primary_confidence,
                correct=row.primary_correct,
                probabilities=row.primary_probabilities,
                outcome=outcome,
                answer=row.primary_answer,
                created_at=row.created_at,
            )
        )
        challenger.append(
            LabeledRow(
                question_id=question.id,
                question_version=question.version,
                backend="candidate",
                model_version=row.candidate_model_version or "",
                confidence=row.candidate_confidence,
                correct=question.is_correct(row.candidate_answer, outcome),
                probabilities=row.candidate_probabilities,
                outcome=outcome,
                answer=row.candidate_answer,
                created_at=row.created_at,
            )
        )
    return primary, challenger


def _choose_method(choice: MethodChoice, train: CalibrationData) -> Method:
    methods = candidate_methods(choice, train)
    if len(methods) == 1:
        return methods[0]
    cut = len(train) - max(1, math.ceil(len(train) * INNER_HOLDOUT))
    inner_train, inner_test = train.subset(slice(0, cut)), train.subset(slice(cut, None))
    best: tuple[float, float, Method] | None = None
    for method in candidate_methods(choice, inner_train):
        try:
            predicted = fit_calibrator(method, inner_train).predict(inner_test)
        except CalibrationError:  # pragma: no cover - fall back to the methods that fit
            continue
        s = score(predicted, inner_test.correct)
        key = (s["log_loss"], s["brier"], method)
        if best is None or key < best:
            best = key
    return best[2] if best is not None else "platt"


def _auto_tier(scores: np.ndarray, correct: np.ndarray, policy: RoutingPolicy) -> AutoTier:
    try:
        selection = select_threshold(
            scores,
            correct,
            target_precision=policy.target_precision,
            confidence=policy.confidence,
            bound=policy.bound,
            min_accepted=policy.min_accepted,
        )
    except ThresholdError as exc:  # pragma: no cover - inputs are validated upstream
        return AutoTier(None, 0.0, 0, None, None, str(exc))
    chosen = selection.selected
    if selection.threshold is None or chosen is None:
        return AutoTier(None, 0.0, 0, None, None, selection.reason)
    return AutoTier(
        threshold=selection.threshold,
        coverage=chosen.coverage,
        accepted=chosen.accepted,
        error_rate=chosen.error_rate,
        error_upper_bound=1.0 - chosen.lower_bound,
        reason=selection.reason,
    )


def _side(
    name: str,
    rows: Sequence[LabeledRow],
    latencies: Sequence[float],
    n_train: int,
    method: MethodChoice,
    policy: RoutingPolicy,
) -> SideReport:
    data = calibration_data(rows)
    train, test = data.subset(slice(0, n_train)), data.subset(slice(n_train, None))
    chosen = _choose_method(method, train)
    calibrator = fit_calibrator(chosen, train)
    calibrated = calibrator.predict(test)
    return SideReport(
        name=name,
        model_versions=tuple(sorted({r.model_version for r in rows if r.model_version})),
        method=chosen,
        calibrator=calibrator.params(),
        raw=score(test.confidences, test.correct),
        calibrated=score(calibrated, test.correct),
        auto=_auto_tier(calibrated, test.correct, policy),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
    )


def compare(
    question: AnyQuestion,
    candidate: str,
    rows: Sequence[PairedRow],
    *,
    summary: ShadowSummary | None = None,
    policy: RoutingPolicy | None = None,
    holdout: float = DEFAULT_HOLDOUT,
    method: MethodChoice = "auto",
    min_paired: int = DEFAULT_MIN_PAIRED,
    alpha: float = DEFAULT_ALPHA,
    min_coverage_gain: float = DEFAULT_MIN_COVERAGE_GAIN,
    max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE,
) -> ShadowComparison:
    """Compare the primary backend with ``candidate`` on paired labeled decisions.

    ``rows`` must be ordered by decision time (as :meth:`ShadowRepository.paired_rows`
    returns them). ``summary`` gives the candidate's failure counts for the reliability
    gate; without it reliability is not checked.
    """
    if not 0.0 < holdout < 1.0:
        raise ShadowCompareError("holdout must be strictly between 0 and 1")
    if not 0.0 < alpha < 1.0:
        raise ShadowCompareError("alpha must be strictly between 0 and 1")
    if not 0.0 <= min_coverage_gain < 1.0:
        raise ShadowCompareError("min_coverage_gain must be within [0, 1)")
    if not 0.0 <= max_failure_rate <= 1.0:
        raise ShadowCompareError("max_failure_rate must be within [0, 1]")
    if min_paired < 2 * MIN_TRAIN:
        raise ShadowCompareError(f"min_paired must be at least {2 * MIN_TRAIN}")
    policy = policy or RoutingPolicy()
    options: dict[str, Any] = {
        "holdout": holdout,
        "method": method,
        "min_paired": min_paired,
        "alpha": alpha,
        "min_coverage_gain": min_coverage_gain,
        "max_failure_rate": max_failure_rate,
    }
    reliability = (
        Reliability(summary.total, summary.errors + summary.timeouts, summary.failure_rate)
        if summary is not None
        else None
    )
    base: dict[str, Any] = {
        "question_id": question.id,
        "question_version": question.version,
        "candidate": candidate,
        "options": options,
        "reliability": reliability,
        "policy": policy.model_dump(mode="json"),
    }
    n = len(rows)
    n_holdout = max(1, math.ceil(n * holdout)) if n else 0
    n_train = n - n_holdout
    if n < min_paired or n_train < MIN_TRAIN:
        return ShadowComparison(
            **base,
            verdict="insufficient",
            reasons=(
                f"{n} paired labeled decisions; at least {min_paired} are needed "
                "(the candidate must answer decisions whose outcomes are known)",
            ),
            paired=n,
            train=max(n_train, 0),
            holdout=n_holdout,
        )
    if method == "temperature" and not all(
        r.primary_probabilities is not None and r.candidate_probabilities is not None for r in rows
    ):
        raise ShadowCompareError(
            "temperature scaling needs a probability distribution from both backends on every "
            "decision; use platt, isotonic or auto"
        )

    primary_rows, challenger_rows = _side_rows(question, rows)
    held = rows[n_train:]
    primary = _side(
        "primary",
        primary_rows,
        [r.primary_latency_ms for r in held],
        n_train,
        method,
        policy,
    )
    challenger = _side(
        candidate,
        challenger_rows,
        [r.candidate_latency_ms for r in held],
        n_train,
        method,
        policy,
    )

    p_ok = np.array([r.correct for r in primary_rows[n_train:]], dtype=bool)
    c_ok = np.array([r.correct for r in challenger_rows[n_train:]], dtype=bool)
    primary_only = int(np.sum(p_ok & ~c_ok))
    candidate_only = int(np.sum(~p_ok & c_ok))
    accuracy = PairedAccuracy(
        total=n_holdout,
        both_correct=int(np.sum(p_ok & c_ok)),
        primary_only=primary_only,
        candidate_only=candidate_only,
        both_wrong=int(np.sum(~p_ok & ~c_ok)),
        delta=(candidate_only - primary_only) / n_holdout,
        interval=paired_difference_interval(primary_only, candidate_only, n_holdout, 1.0 - alpha),
        p_value=mcnemar_exact(primary_only, candidate_only),
    )
    verdict, reasons = _verdict(
        accuracy, primary, challenger, reliability, alpha, min_coverage_gain, max_failure_rate
    )
    return ShadowComparison(
        **base,
        verdict=verdict,
        reasons=reasons,
        paired=n,
        train=n_train,
        holdout=n_holdout,
        accuracy=accuracy,
        primary=primary,
        challenger=challenger,
    )


async def compare_stored(
    repository: ShadowRepository,
    question: AnyQuestion,
    candidate: str,
    *,
    policy: RoutingPolicy | None = None,
    mode: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    **options: Any,
) -> ShadowComparison:
    """Load the paired rows and failure counts for ``candidate`` and :func:`compare` them."""
    rows = await repository.paired_rows(
        question.id, question.version, candidate, mode=mode, since=since, until=until
    )
    summaries = await repository.summaries(
        question_id=question.id,
        question_version=question.version,
        candidate=candidate,
        mode=mode,
        since=since,
        until=until,
    )
    return compare(
        question,
        candidate,
        rows,
        summary=summaries[0] if summaries else None,
        policy=policy,
        **options,
    )


def _verdict(
    accuracy: PairedAccuracy,
    primary: SideReport,
    challenger: SideReport,
    reliability: Reliability | None,
    alpha: float,
    min_gain: float,
    max_failure_rate: float,
) -> tuple[Verdict, tuple[str, ...]]:
    blockers: list[str] = []
    gains: list[str] = []
    notes: list[str] = []

    if reliability is not None and reliability.rate > max_failure_rate:
        blockers.append(
            f"the candidate failed {reliability.failures} of {reliability.total} calls "
            f"({reliability.rate:.1%}, limit {max_failure_rate:.1%})"
        )

    significant = accuracy.p_value < alpha
    delta = f"{accuracy.delta:+.1%} accuracy (McNemar p={accuracy.p_value:.3g})"
    if significant and accuracy.delta < 0:
        blockers.append(f"significantly less accurate: {delta}")
    elif significant and accuracy.delta > 0:
        gains.append(f"significantly more accurate: {delta}")
    else:
        notes.append(f"no significant accuracy difference: {delta}")

    gain = challenger.auto.coverage - primary.auto.coverage
    coverage = (
        f"auto coverage {challenger.auto.coverage:.1%} vs {primary.auto.coverage:.1%} "
        f"at the error budget"
    )
    if gain < -min_gain:
        blockers.append(f"automates less: {coverage}")
    elif gain >= min_gain and gain > 0:
        gains.append(f"automates more: {coverage}")
    else:
        notes.append(f"similar automation: {coverage}")

    if blockers:
        return "keep", tuple(blockers + gains + notes)
    if gains:
        return "promote", tuple(gains + notes)
    return "keep", ("no meaningful improvement", *notes)


__all__ = [
    "AutoTier",
    "PairedAccuracy",
    "Reliability",
    "ShadowCompareError",
    "ShadowComparison",
    "SideReport",
    "Verdict",
    "binomial_cdf",
    "compare",
    "compare_stored",
    "mcnemar_exact",
    "paired_difference_interval",
    "percentile",
]
