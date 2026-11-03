"""Split-conformal prediction sets for choice (and predicate) questions.

Instead of a single answer, a conformal predictor returns a *set* of labels that contains
the true label with probability at least ``1 - alpha``, whatever the model and however
badly it is calibrated. The only assumption is that the calibration decisions and the
future decisions are exchangeable (e.g. i.i.d. from the same traffic).

Split conformal, given ``n`` labeled calibration decisions:

1. compute a nonconformity score ``s_i`` for each decision's *true* label;
2. take ``q`` = the ``ceil((n + 1)(1 - alpha))``-th smallest score (infinite when that
   rank exceeds ``n``: too few decisions to promise this coverage, so every label is in);
3. for a new decision, the set is every label whose score is at most ``q``.

Then ``P(true label in set) >= 1 - alpha``, and when scores have no ties it is also at
most ``1 - alpha + 1 / (n + 1)``. The guarantee is marginal (on average over decisions),
not per decision.

Two scores are implemented:

* ``lac`` (least ambiguous classifier): ``1 - p(label)``. Gives the smallest sets on
  average, but can return an empty set when the model is unsure of everything.
* ``aps`` (adaptive prediction sets, non-randomised): this label's probability plus that
  of every strictly more probable label. Sets grow on hard inputs (a flat distribution) and
  shrink on easy ones; the most probable label is always included, so they are never
  empty. On average they are larger than ``lac``.

For routing, a singleton set is a confident answer, a larger set narrows down what a
human reviewer has to consider, and an empty set means "send to a human".
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt

ConformalMethod = Literal["lac", "aps"]
CONFORMAL_METHODS: tuple[ConformalMethod, ...] = get_args(ConformalMethod)
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

DEFAULT_ALPHA = 0.1
# Two scores within this distance count as equal when comparing to the quantile, so a
# label scored by the same arithmetic as a calibration decision is never excluded by
# floating-point noise.
_TOLERANCE = 1e-12


class ConformalError(ValueError):
    """Invalid inputs or options for conformal prediction."""


def distribution_matrix(
    distributions: Sequence[Mapping[str, float]], labels: Sequence[str]
) -> FloatArray:
    """One row per decision, one column per label, each row normalised to sum to 1.

    A label missing from a distribution has probability 0. Keys outside ``labels`` or rows
    with no probability mass are errors.
    """
    if not labels:
        raise ConformalError("at least one label is needed")
    column = {label: i for i, label in enumerate(labels)}
    if len(column) != len(labels):
        raise ConformalError("labels must be unique")
    matrix = np.zeros((len(distributions), len(labels)), dtype=np.float64)
    for i, distribution in enumerate(distributions):
        for label, p in distribution.items():
            if label not in column:
                raise ConformalError(f"label {label!r} is not one of {list(labels)}")
            value = float(p)
            if not math.isfinite(value) or value < 0.0:
                raise ConformalError("probabilities must be finite and non-negative")
            matrix[i, column[label]] = value
    totals = matrix.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ConformalError("every distribution needs some probability mass")
    normalised: FloatArray = matrix / totals[:, None]
    return normalised


def label_scores(probabilities: FloatArray, method: ConformalMethod) -> FloatArray:
    """The nonconformity score of every label for every decision (same shape as input)."""
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ConformalError("probabilities must be a non-empty 2-D matrix")
    if method == "lac":
        result: FloatArray = 1.0 - probabilities
        return result
    if method == "aps":
        # For label j: p_j plus the mass of every strictly more probable label. Tied
        # labels share a score, so a set takes all of them or none of them.
        above = probabilities[:, None, :] > probabilities[:, :, None]
        aps: FloatArray = probabilities + (above * probabilities[:, None, :]).sum(axis=2)
        return aps
    raise ConformalError(f"unknown method {method!r}; choose from {list(CONFORMAL_METHODS)}")


def conformal_quantile(scores: FloatArray, alpha: float) -> float:
    """The ``ceil((n + 1)(1 - alpha))``-th smallest score, or ``inf`` if that rank > n."""
    if not 0.0 < alpha < 1.0:
        raise ConformalError("alpha must be strictly between 0 and 1")
    if scores.ndim != 1 or scores.size == 0:
        raise ConformalError("at least one calibration score is needed")
    n = int(scores.size)
    rank = math.ceil((n + 1) * (1.0 - alpha) - 1e-9)
    if rank > n:
        return math.inf
    return float(np.sort(scores)[rank - 1])


def min_calibration_size(alpha: float) -> int:
    """The fewest calibration decisions that give a finite quantile at this ``alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ConformalError("alpha must be strictly between 0 and 1")
    n = 1
    while math.ceil((n + 1) * (1.0 - alpha) - 1e-9) > n:
        n += 1
    return n


@dataclass(frozen=True, slots=True)
class ConformalPredictor:
    method: ConformalMethod
    alpha: float
    quantile: float
    """``inf`` when there were too few calibration decisions (every label is in the set)."""
    labels: tuple[str, ...]
    calibration_count: int

    @property
    def is_trivial(self) -> bool:
        return math.isinf(self.quantile)

    def set_mask(self, probabilities: FloatArray) -> npt.NDArray[np.bool_]:
        """Which labels are in each decision's prediction set."""
        if probabilities.shape[-1] != len(self.labels):
            raise ConformalError("probabilities must have one column per label")
        mask: npt.NDArray[np.bool_] = (
            label_scores(probabilities, self.method) <= self.quantile + _TOLERANCE
        )
        if self.method == "aps":
            # The most probable label's APS score is its own probability, so a very
            # confident decision could otherwise get an empty set. Adding it only grows
            # sets, which keeps the coverage guarantee.
            mask |= probabilities >= probabilities.max(axis=1, keepdims=True)
        return mask

    def predict_set(self, probabilities: Mapping[str, float]) -> list[str]:
        """The prediction set for one decision, most probable label first."""
        row = distribution_matrix([probabilities], self.labels)
        mask = self.set_mask(row)[0]
        order = np.argsort(-row[0], kind="stable")
        return [self.labels[i] for i in order if mask[i]]

    def params(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            # JSON has no infinity; ``None`` means "every label".
            "quantile": None if self.is_trivial else self.quantile,
            "labels": list(self.labels),
            "calibration_count": self.calibration_count,
        }

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> ConformalPredictor:
        quantile = params["quantile"]
        method = params["method"]
        if method not in CONFORMAL_METHODS:
            raise ConformalError(f"unknown method {method!r}")
        return cls(
            method=method,
            alpha=float(params["alpha"]),
            quantile=math.inf if quantile is None else float(quantile),
            labels=tuple(params["labels"]),
            calibration_count=int(params["calibration_count"]),
        )


def fit_conformal(
    probabilities: FloatArray,
    truths: IntArray,
    labels: Sequence[str],
    *,
    alpha: float = DEFAULT_ALPHA,
    method: ConformalMethod = "lac",
) -> ConformalPredictor:
    """Calibrate a split-conformal predictor on decisions with known true labels."""
    if method not in CONFORMAL_METHODS:
        raise ConformalError(f"unknown method {method!r}; choose from {list(CONFORMAL_METHODS)}")
    if probabilities.ndim != 2 or probabilities.shape != (truths.size, len(labels)):
        raise ConformalError("probabilities must be (decisions x labels), one truth per row")
    if truths.size == 0:
        raise ConformalError("at least one calibration decision is needed")
    if np.any((truths < 0) | (truths >= len(labels))):
        raise ConformalError("truths must index into labels")
    scores = label_scores(probabilities, method)[np.arange(truths.size), truths]
    return ConformalPredictor(
        method=method,
        alpha=alpha,
        quantile=conformal_quantile(scores, alpha),
        labels=tuple(labels),
        calibration_count=int(truths.size),
    )


@dataclass(frozen=True, slots=True)
class ConformalEvaluation:
    """How a predictor's sets behaved on decisions it was not calibrated on."""

    count: int
    coverage: float
    """Fraction of sets containing the true label."""
    mean_size: float
    singleton_rate: float
    singleton_accuracy: float | None
    """Of the singleton sets, the fraction that are the true label."""
    empty_rate: float
    size_counts: dict[int, int]


def evaluate(
    predictor: ConformalPredictor, probabilities: FloatArray, truths: IntArray
) -> ConformalEvaluation:
    if truths.size == 0:
        raise ConformalError("at least one decision is needed to evaluate")
    mask = predictor.set_mask(probabilities)
    sizes = mask.sum(axis=1)
    covered = mask[np.arange(truths.size), truths]
    singles = sizes == 1
    return ConformalEvaluation(
        count=int(truths.size),
        coverage=float(covered.mean()),
        mean_size=float(sizes.mean()),
        singleton_rate=float(singles.mean()),
        singleton_accuracy=float(covered[singles].mean()) if singles.any() else None,
        empty_rate=float((sizes == 0).mean()),
        size_counts=dict(sorted(Counter(int(s) for s in sizes).items())),
    )


__all__ = [
    "CONFORMAL_METHODS",
    "DEFAULT_ALPHA",
    "ConformalError",
    "ConformalEvaluation",
    "ConformalMethod",
    "ConformalPredictor",
    "conformal_quantile",
    "distribution_matrix",
    "evaluate",
    "fit_conformal",
    "label_scores",
    "min_calibration_size",
]
