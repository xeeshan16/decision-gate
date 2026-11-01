"""Calibration metrics: how far a model's confidence is from how often it is right.

Every question kind reduces to the same pair per labeled decision: the *confidence* ``c``
the backend gave its own answer, and whether that answer turned out *correct* (``y`` is 0
or 1). Most metrics here are defined on that "top-label" pair, so they mean the same
thing for predicate, choice and score questions:

* **Reliability bins** group decisions by confidence and compare, in each bin, the mean
  confidence with the observed accuracy. Plotted, this is the reliability diagram.
* **ECE** (expected calibration error) is the sample-weighted mean of
  ``|accuracy - confidence|`` over the bins; **MCE** is the largest such gap.
* **Brier score** is the mean squared error ``mean((c - y)^2)``. Unlike ECE it does not
  depend on a binning choice, and it rewards both calibration and sharpness.
* **Log loss** is ``-mean(y log c + (1 - y) log(1 - c))``. It punishes confident mistakes
  much harder than the Brier score does.

For questions with a full distribution over a finite label set (predicate and choice
answers with ``probabilities``) the multi-class Brier score and log loss look at the
whole distribution, not just the top answer.

Everything is plain numpy and deterministic (the bootstrap takes an explicit seed).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BinStrategy = Literal["uniform", "quantile"]
Confidences = Sequence[float] | FloatArray
Correctness = Sequence[bool] | npt.NDArray[np.bool_] | FloatArray
Distribution = Sequence[Sequence[float]] | FloatArray
Labels = Sequence[int] | npt.NDArray[np.integer]

# Keeps log loss finite when a probability is exactly 0 or 1.
LOG_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lower: float
    """Lower edge. Uniform bins are ``(lower, upper]``; the first one also includes 0."""
    upper: float
    count: int
    mean_confidence: float | None
    """``None`` for an empty bin."""
    accuracy: float | None

    @property
    def gap(self) -> float | None:
        """``mean_confidence - accuracy``: positive means overconfident in this bin."""
        if self.mean_confidence is None or self.accuracy is None:
            return None
        return self.mean_confidence - self.accuracy


def _as_pairs(confidences: Confidences, correct: Correctness) -> tuple[FloatArray, FloatArray]:
    c = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if c.ndim != 1 or y.ndim != 1:
        raise ValueError("confidences and correct must be one-dimensional")
    if c.shape != y.shape:
        raise ValueError(f"confidences and correct differ in length ({c.shape[0]} != {y.shape[0]})")
    if c.size == 0:
        raise ValueError("at least one labeled decision is needed")
    if not np.all(np.isfinite(c)) or np.any((c < 0.0) | (c > 1.0)):
        raise ValueError("confidences must be finite and within [0, 1]")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("correct must contain only booleans or 0/1")
    return c, y


def brier_score(confidences: Confidences, correct: Correctness) -> float:
    """Mean squared error between confidence and correctness. 0 is perfect."""
    c, y = _as_pairs(confidences, correct)
    return float(np.mean((c - y) ** 2))


def log_loss(confidences: Confidences, correct: Correctness) -> float:
    """Binary cross-entropy of confidence against correctness, in nats."""
    c, y = _as_pairs(confidences, correct)
    c = np.clip(c, LOG_EPSILON, 1.0 - LOG_EPSILON)
    return float(-np.mean(y * np.log(c) + (1.0 - y) * np.log1p(-c)))


def _bin_members(
    c: FloatArray, n_bins: int, strategy: BinStrategy
) -> tuple[npt.NDArray[np.intp], FloatArray]:
    """Bin index for each sample, and the ``n_bins + 1`` bin edges."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        # Right-closed bins: a confidence exactly on an inner edge belongs to the lower bin.
        index = np.searchsorted(edges[1:-1], c, side="left")
        return index.astype(np.intp), edges
    if strategy == "quantile":
        # Equal-mass bins: sort and cut into n_bins groups whose sizes differ by at most
        # one. Ties on a cut may land in different bins; edges are the members' extremes.
        order = np.argsort(c, kind="stable")
        index = np.empty(c.size, dtype=np.intp)
        for b, members in enumerate(np.array_split(order, n_bins)):
            index[members] = b
        return index, np.array([], dtype=np.float64)
    raise ValueError(f"unknown bin strategy {strategy!r}")


def reliability_bins(
    confidences: Confidences,
    correct: Correctness,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> list[ReliabilityBin]:
    """The reliability diagram as a list of bins, lowest confidence first.

    ``uniform`` uses ``n_bins`` equal-width bins over [0, 1] and keeps empty bins, so
    diagrams from different models line up. ``quantile`` puts (nearly) the same number of
    decisions in each bin, which gives steadier estimates when confidences cluster; it
    returns at most ``n_bins`` bins, never an empty one.
    """
    c, y = _as_pairs(confidences, correct)
    index, edges = _bin_members(c, n_bins, strategy)
    counts = np.bincount(index, minlength=n_bins)
    conf_sums = np.bincount(index, weights=c, minlength=n_bins)
    hit_sums = np.bincount(index, weights=y, minlength=n_bins)
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        n = int(counts[b])
        if strategy == "uniform":
            lower, upper = float(edges[b]), float(edges[b + 1])
        elif n == 0:
            continue  # fewer samples than bins
        else:
            members = c[index == b]
            lower, upper = float(members.min()), float(members.max())
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=n,
                mean_confidence=float(conf_sums[b] / n) if n else None,
                accuracy=float(hit_sums[b] / n) if n else None,
            )
        )
    return bins


def _ece_from_bins(bins: Sequence[ReliabilityBin], total: int) -> tuple[float, float]:
    ece = 0.0
    mce = 0.0
    for b in bins:
        if b.gap is None:
            continue
        ece += b.count / total * abs(b.gap)
        mce = max(mce, abs(b.gap))
    return ece, mce


def expected_calibration_error(
    confidences: Confidences,
    correct: Correctness,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> float:
    """Sample-weighted mean of ``|accuracy - confidence|`` over reliability bins."""
    bins = reliability_bins(confidences, correct, n_bins=n_bins, strategy=strategy)
    return _ece_from_bins(bins, sum(b.count for b in bins))[0]


def maximum_calibration_error(
    confidences: Confidences,
    correct: Correctness,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
) -> float:
    """The largest ``|accuracy - confidence|`` over non-empty reliability bins."""
    bins = reliability_bins(confidences, correct, n_bins=n_bins, strategy=strategy)
    return _ece_from_bins(bins, sum(b.count for b in bins))[1]


def ece_interval(
    confidences: Confidences,
    correct: Correctness,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    resamples: int = 200,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap interval for the ECE.

    ECE from a few hundred labels is noisy, and it is biased upwards on small samples
    (even a perfectly calibrated model shows a gap in every bin). The interval makes that
    visible before anyone acts on a single number. Deterministic for a given ``seed``.
    """
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    c, y = _as_pairs(confidences, correct)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for r in range(resamples):
        pick = rng.integers(0, c.size, size=c.size)
        estimates[r] = expected_calibration_error(
            c[pick], y[pick], n_bins=n_bins, strategy=strategy
        )
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _as_distribution(
    probabilities: Distribution, labels: Labels
) -> tuple[FloatArray, npt.NDArray[np.intp]]:
    p = np.asarray(probabilities, dtype=np.float64)
    t = np.asarray(labels)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("probabilities must be an (n, k) array with n >= 1 and k >= 2")
    if t.shape != (p.shape[0],):
        raise ValueError("labels must have one entry per row of probabilities")
    if not np.issubdtype(t.dtype, np.integer) or np.any((t < 0) | (t >= p.shape[1])):
        raise ValueError("labels must be column indices into probabilities")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probabilities must sum to 1")
    return p, t.astype(np.intp)


def multiclass_brier_score(
    probabilities: Distribution,
    labels: Labels,
) -> float:
    """Mean over rows of ``sum_k (p_k - [k == label])^2``. Ranges from 0 to 2."""
    p, t = _as_distribution(probabilities, labels)
    onehot = np.zeros_like(p)
    onehot[np.arange(t.size), t] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def multiclass_log_loss(
    probabilities: Distribution,
    labels: Labels,
) -> float:
    """Mean negative log probability of the true label, in nats."""
    p, t = _as_distribution(probabilities, labels)
    return float(-np.mean(np.log(np.clip(p[np.arange(t.size), t], LOG_EPSILON, 1.0))))


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Every metric for one set of labeled decisions."""

    count: int
    accuracy: float
    mean_confidence: float
    ece: float
    mce: float
    brier: float
    log_loss: float
    bins: tuple[ReliabilityBin, ...]
    n_bins: int
    strategy: BinStrategy
    ece_interval: tuple[float, float] | None = None
    multiclass_brier: float | None = None
    multiclass_log_loss: float | None = None

    @property
    def overconfidence(self) -> float:
        """``mean_confidence - accuracy``: positive means the model claims too much."""
        return self.mean_confidence - self.accuracy


def calibration_metrics(
    confidences: Confidences,
    correct: Correctness,
    *,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    bootstrap: int = 0,
    seed: int = 0,
    probabilities: Distribution | None = None,
    labels: Labels | None = None,
) -> CalibrationMetrics:
    """Compute all metrics at once. ``bootstrap > 0`` adds a 95% interval for the ECE."""
    c, y = _as_pairs(confidences, correct)
    bins = reliability_bins(c, y, n_bins=n_bins, strategy=strategy)
    ece, mce = _ece_from_bins(bins, c.size)
    multi_brier = multi_loss = None
    if probabilities is not None or labels is not None:
        if probabilities is None or labels is None:
            raise ValueError("pass both probabilities and labels, or neither")
        p, t = _as_distribution(probabilities, labels)
        if t.size != c.size:
            raise ValueError("probabilities must have one row per decision")
        multi_brier = multiclass_brier_score(p, t)
        multi_loss = multiclass_log_loss(p, t)
    interval = (
        ece_interval(c, y, n_bins=n_bins, strategy=strategy, resamples=bootstrap, seed=seed)
        if bootstrap > 0
        else None
    )
    return CalibrationMetrics(
        count=int(c.size),
        accuracy=float(np.mean(y)),
        mean_confidence=float(np.mean(c)),
        ece=ece,
        mce=mce,
        brier=brier_score(c, y),
        log_loss=log_loss(c, y),
        bins=tuple(bins),
        n_bins=n_bins,
        strategy=strategy,
        ece_interval=interval,
        multiclass_brier=multi_brier,
        multiclass_log_loss=multi_loss,
    )
