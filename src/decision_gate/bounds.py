"""Confidence bounds for a binomial proportion, implemented without scipy.

Threshold selection needs a statement like "with 95% confidence, at least 97% of the
decisions above this threshold are correct". That is a one-sided lower confidence bound on
a binomial proportion: ``k`` correct out of ``n`` accepted decisions.

Two bounds are offered:

* **Clopper-Pearson** (the default) inverts the binomial test exactly. Its coverage is at
  least the nominal level for every true proportion and sample size, so it is
  conservative. The bound is a beta-distribution quantile, computed here by bisection on
  the regularized incomplete beta function.
* **Wilson** (score interval) is a closed-form approximation. It is tighter and its
  *average* coverage is close to nominal, but for some proportions and sample sizes it
  falls below it, so it is the less safe choice for an error budget.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Literal, get_args

BoundMethod = Literal["clopper-pearson", "wilson"]
BOUND_METHODS: tuple[BoundMethod, ...] = get_args(BoundMethod)

_CF_MAX_ITERATIONS = 500
_CF_EPSILON = 1e-15
_TINY = 1e-300
_BISECTION_TOLERANCE = 1e-13


def _continued_fraction(a: float, b: float, x: float) -> float:
    """The continued fraction for the incomplete beta function, by the modified Lentz method.

    Converges quickly for ``x < (a + 1) / (a + b + 2)``; the caller uses the symmetry
    ``I_x(a, b) = 1 - I_{1-x}(b, a)`` to stay in that region.
    """
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d
    for m in range(1, _CF_MAX_ITERATIONS + 1):
        m2 = 2 * m
        # Even step.
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        h *= d * c
        # Odd step.
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _TINY else _TINY)
        c = 1.0 + aa / c
        c = c if abs(c) > _TINY else _TINY
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPSILON:
            return h
    raise ArithmeticError(  # pragma: no cover - needs parameters far beyond binomial use
        f"incomplete beta continued fraction did not converge for a={a}, b={b}, x={x}"
    )


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """``I_x(a, b)``: the CDF of a Beta(a, b) distribution at ``x``."""
    if not (a > 0 and b > 0):
        raise ValueError("a and b must be positive")
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be within [0, 1]")
    if x == 0.0 or x == 1.0:
        return x
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _continued_fraction(b, a, 1.0 - x) / b


def beta_quantile(p: float, a: float, b: float) -> float:
    """The ``p`` quantile of Beta(a, b), by bisection (the CDF is strictly increasing)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be within [0, 1]")
    if p in (0.0, 1.0):
        return p
    lo, hi = 0.0, 1.0
    while hi - lo > _BISECTION_TOLERANCE:
        mid = (lo + hi) / 2.0
        if regularized_incomplete_beta(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _check(successes: int, trials: int, confidence: float) -> None:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be within [0, trials]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")


def _wilson(successes: int, trials: int, z: float) -> tuple[float, float]:
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    return max(0.0, center - half), min(1.0, center + half)


def lower_bound(
    successes: int,
    trials: int,
    confidence: float = 0.95,
    method: BoundMethod = "clopper-pearson",
) -> float:
    """One-sided lower confidence bound: ``P(true p >= bound) >= confidence``."""
    _check(successes, trials, confidence)
    if method == "clopper-pearson":
        if successes == 0:
            return 0.0
        return beta_quantile(1.0 - confidence, successes, trials - successes + 1)
    if method == "wilson":
        return _wilson(successes, trials, NormalDist().inv_cdf(confidence))[0]
    raise ValueError(f"unknown bound {method!r}; choose from {list(BOUND_METHODS)}")


def upper_bound(
    successes: int,
    trials: int,
    confidence: float = 0.95,
    method: BoundMethod = "clopper-pearson",
) -> float:
    """One-sided upper confidence bound: ``P(true p <= bound) >= confidence``."""
    _check(successes, trials, confidence)
    # The upper bound on p is one minus the lower bound on the failure proportion.
    return 1.0 - lower_bound(trials - successes, trials, confidence, method)


def interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
    method: BoundMethod = "clopper-pearson",
) -> tuple[float, float]:
    """Two-sided interval, splitting the miss probability equally between the tails."""
    _check(successes, trials, confidence)
    one_sided = 1.0 - (1.0 - confidence) / 2.0
    return (
        lower_bound(successes, trials, one_sided, method),
        upper_bound(successes, trials, one_sided, method),
    )


__all__ = [
    "BOUND_METHODS",
    "BoundMethod",
    "beta_quantile",
    "interval",
    "lower_bound",
    "regularized_incomplete_beta",
    "upper_bound",
]
