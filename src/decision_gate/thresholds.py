"""Threshold selection: the lowest confidence at which decisions can be automated safely.

A decision is *accepted* (automated) when its score, the calibrated probability that the
answer is correct, is at least the threshold. The goal is the threshold that automates as
much as possible while the precision of accepted decisions (equivalently ``1 - error
rate``) stays at or above a target, *with statistical confidence*, not just on average
over the sample.

Picking "the lowest threshold whose observed precision meets the target" overfits: with
many candidate thresholds, some will look good by chance. Instead the candidates are
tested as a **fixed sequence**, from the highest threshold down:

* for each candidate, the null hypothesis is "precision of decisions at or above this
  threshold is below the target";
* it is rejected when the one-sided lower confidence bound on precision (Clopper-Pearson
  by default) is at least the target;
* the search stops at the first candidate that cannot be rejected, and the last rejected
  candidate is selected.

The order of candidates depends only on the scores, never on the labels, so fixed-sequence
testing controls the family-wise error rate at ``1 - confidence`` without any multiplicity
correction. The guarantee: if the labeled decisions are an i.i.d. sample from the traffic
the threshold will be used on, then with probability at least ``confidence`` the selected
threshold's precision on that traffic is at least the target. It says nothing about
traffic that has drifted away from the sample; that is what drift monitoring is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from decision_gate.bounds import BOUND_METHODS, BoundMethod, lower_bound

FloatArray = npt.NDArray[np.float64]

DEFAULT_TARGET_PRECISION = 0.95
DEFAULT_CONFIDENCE = 0.95
DEFAULT_MIN_ACCEPTED = 100
DEFAULT_MAX_CANDIDATES = 100


class ThresholdError(ValueError):
    """Invalid inputs or options for threshold selection."""


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    """What happens if decisions with ``score >= threshold`` are accepted."""

    threshold: float
    accepted: int
    correct: int
    coverage: float
    """Fraction of all decisions that would be accepted."""
    precision: float
    """Observed fraction of accepted decisions that are correct."""
    lower_bound: float
    """One-sided lower confidence bound on the precision."""

    @property
    def error_rate(self) -> float:
        return 1.0 - self.precision


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    target_precision: float
    confidence: float
    bound: BoundMethod
    total: int
    threshold: float | None
    """``None`` when no candidate met the target: nothing should be automated."""
    selected: ThresholdPoint | None
    stopped_at: ThresholdPoint | None
    """The first candidate that failed the test (``None`` if every candidate passed)."""
    tested: int
    candidates: tuple[ThresholdPoint, ...]
    """Every candidate threshold, highest first, including the untested ones."""
    reason: str

    @property
    def max_error_rate(self) -> float:
        return 1.0 - self.target_precision


def _validate(scores: npt.ArrayLike, correct: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if s.ndim != 1 or y.shape != s.shape:
        raise ThresholdError("scores and correct must be 1-D and the same length")
    if s.size == 0:
        raise ThresholdError("at least one labeled decision is needed")
    if not np.all(np.isfinite(s)) or np.any((s < 0.0) | (s > 1.0)):
        raise ThresholdError("scores must be finite and within [0, 1]")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ThresholdError("correct must contain only 0 and 1 (or booleans)")
    return s, y


def candidate_thresholds(
    scores: npt.ArrayLike,
    correct: npt.ArrayLike,
    *,
    min_accepted: int = DEFAULT_MIN_ACCEPTED,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[tuple[float, int, int]]:
    """``(threshold, accepted, correct)`` for each candidate, highest threshold first.

    Candidates are the distinct scores that accept at least ``min_accepted`` decisions.
    This also sets where the test sequence starts, which matters: started on a handful of
    decisions, one unlucky error fails the first test and ends the search with nothing
    selected (29/30 correct has a 95% lower bound of 0.85). When there are more than
    ``max_candidates`` the list is thinned evenly by acceptance count, always keeping the
    lowest. Both rules look only at scores, never at labels.
    """
    if min_accepted < 1:
        raise ThresholdError("min_accepted must be at least 1")
    if max_candidates < 1:
        raise ThresholdError("max_candidates must be at least 1")
    s, y = _validate(scores, correct)
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    cum_correct = np.cumsum(y_sorted)
    # The last position of each run of equal scores: accepting at that score accepts
    # everything up to and including it.
    ends = np.flatnonzero(np.append(s_sorted[1:] != s_sorted[:-1], True))
    eligible = ends[ends + 1 >= min_accepted]
    if eligible.size > max_candidates:
        keep = np.unique(np.linspace(0, eligible.size - 1, max_candidates).round().astype(int))
        eligible = eligible[keep]
    return [(float(s_sorted[i]), int(i + 1), int(cum_correct[i])) for i in eligible]


def select_threshold(
    scores: npt.ArrayLike,
    correct: npt.ArrayLike,
    *,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    confidence: float = DEFAULT_CONFIDENCE,
    bound: BoundMethod = "clopper-pearson",
    min_accepted: int = DEFAULT_MIN_ACCEPTED,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> ThresholdSelection:
    """The lowest threshold that passes a fixed-sequence test of the precision target."""
    if not 0.0 < target_precision < 1.0:
        raise ThresholdError("target_precision must be strictly between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ThresholdError("confidence must be strictly between 0 and 1")
    if bound not in BOUND_METHODS:
        raise ThresholdError(f"unknown bound {bound!r}; choose from {list(BOUND_METHODS)}")
    raw = candidate_thresholds(
        scores, correct, min_accepted=min_accepted, max_candidates=max_candidates
    )
    total = int(np.asarray(scores).size)
    points = tuple(
        ThresholdPoint(
            threshold=t,
            accepted=n,
            correct=k,
            coverage=n / total,
            precision=k / n,
            lower_bound=lower_bound(k, n, confidence, bound),
        )
        for t, n, k in raw
    )
    selected: ThresholdPoint | None = None
    stopped_at: ThresholdPoint | None = None
    tested = 0
    for point in points:
        tested += 1
        if point.lower_bound >= target_precision:
            selected = point
            continue
        stopped_at = point
        break

    level = f"{confidence:.0%}"
    if not points:
        reason = (
            f"{total} labeled decisions; no threshold accepts at least min_accepted={min_accepted}"
        )
    elif selected is None:
        assert stopped_at is not None
        reason = (
            f"even the highest threshold {stopped_at.threshold:.4f} has a {level} lower "
            f"bound of {stopped_at.lower_bound:.4f} on precision "
            f"({stopped_at.correct}/{stopped_at.accepted}), below the target "
            f"{target_precision:.4f}"
        )
    else:
        reason = (
            f"accept score >= {selected.threshold:.4f}: {selected.coverage:.1%} of decisions, "
            f"precision {selected.precision:.4f} ({selected.correct}/{selected.accepted}), "
            f"{level} lower bound {selected.lower_bound:.4f} >= target {target_precision:.4f}"
        )
        if stopped_at is not None:
            reason += f"; the next candidate {stopped_at.threshold:.4f} failed"
    return ThresholdSelection(
        target_precision=target_precision,
        confidence=confidence,
        bound=bound,
        total=total,
        threshold=selected.threshold if selected else None,
        selected=selected,
        stopped_at=stopped_at,
        tested=tested,
        candidates=points,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class ThresholdCheck:
    """A threshold applied to decisions it was not selected on."""

    threshold: float
    total: int
    accepted: int
    correct: int

    @property
    def coverage(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    @property
    def precision(self) -> float | None:
        return self.correct / self.accepted if self.accepted else None


def apply_threshold(
    threshold: float,
    scores: npt.ArrayLike,
    correct: npt.ArrayLike,
) -> ThresholdCheck:
    """Coverage and realised precision of ``score >= threshold`` on other decisions."""
    if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise ThresholdError("threshold must be within [0, 1]")
    s, y = _validate(scores, correct)
    accepted = s >= threshold
    return ThresholdCheck(
        threshold=threshold,
        total=int(s.size),
        accepted=int(accepted.sum()),
        correct=int(y[accepted].sum()),
    )


def curve(selection: ThresholdSelection, points: int = 10) -> list[ThresholdPoint]:
    """An evenly spaced subset of the candidates for display, always with the selection."""
    candidates = list(selection.candidates)
    if points < 1 or not candidates:
        return []
    if len(candidates) > points:
        picks = np.unique(np.linspace(0, len(candidates) - 1, points).round().astype(int))
        chosen = [candidates[i] for i in picks]
    else:
        chosen = candidates
    for extra in (selection.selected, selection.stopped_at):
        if extra is not None and extra not in chosen:
            chosen.append(extra)
    return sorted(chosen, key=lambda p: -p.threshold)


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MIN_ACCEPTED",
    "DEFAULT_TARGET_PRECISION",
    "ThresholdCheck",
    "ThresholdError",
    "ThresholdPoint",
    "ThresholdSelection",
    "apply_threshold",
    "candidate_thresholds",
    "curve",
    "select_threshold",
]
