"""Binomial confidence bounds and fixed-sequence threshold selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from decision_gate.bounds import (
    BoundMethod,
    beta_quantile,
    interval,
    lower_bound,
    regularized_incomplete_beta,
    upper_bound,
)
from decision_gate.thresholds import (
    ThresholdError,
    apply_threshold,
    candidate_thresholds,
    curve,
    select_threshold,
)

# --- incomplete beta and quantiles ------------------------------------------------------


@pytest.mark.parametrize(("n", "k", "p"), [(10, 3, 0.2), (25, 20, 0.7), (60, 1, 0.01), (7, 7, 0.9)])
def test_incomplete_beta_matches_the_binomial_tail(n: int, k: int, p: float) -> None:
    # For integers, I_p(k, n - k + 1) = P(X >= k) with X ~ Binomial(n, p).
    tail = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))
    assert regularized_incomplete_beta(p, k, n - k + 1) == pytest.approx(tail, abs=1e-12)


def test_incomplete_beta_reference_values_and_edges() -> None:
    # Reference values from an independent implementation (scipy.stats.beta.cdf).
    assert regularized_incomplete_beta(0.3, 2.5, 4.0) == pytest.approx(0.3521975859067672)
    assert regularized_incomplete_beta(0.9, 30, 2) == pytest.approx(0.1695646331008649)
    assert regularized_incomplete_beta(0.0, 2, 3) == 0.0
    assert regularized_incomplete_beta(1.0, 2, 3) == 1.0
    # Symmetry: I_x(a, b) = 1 - I_{1-x}(b, a).
    assert regularized_incomplete_beta(0.35, 3.2, 1.7) == pytest.approx(
        1 - regularized_incomplete_beta(0.65, 1.7, 3.2)
    )


def test_beta_quantile_inverts_the_cdf() -> None:
    for p in (0.01, 0.05, 0.5, 0.95):
        x = beta_quantile(p, 4, 3)
        assert regularized_incomplete_beta(x, 4, 3) == pytest.approx(p, abs=1e-10)
    assert beta_quantile(0.05, 4, 3) == pytest.approx(0.2713383725197525)
    assert beta_quantile(0.0, 2, 2) == 0.0
    assert beta_quantile(1.0, 2, 2) == 1.0


@pytest.mark.parametrize(
    ("x", "a", "b", "message"),
    [(0.5, 0, 1, "positive"), (0.5, 1, -1, "positive"), (1.5, 1, 1, "within")],
)
def test_incomplete_beta_rejects_bad_arguments(x: float, a: float, b: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        regularized_incomplete_beta(x, a, b)


def test_beta_quantile_rejects_bad_probability() -> None:
    with pytest.raises(ValueError, match="within"):
        beta_quantile(1.2, 1, 1)


# --- Clopper-Pearson and Wilson ---------------------------------------------------------


def test_clopper_pearson_textbook_values() -> None:
    # 5 of 10, 95% two-sided: the interval R's binom.test reports.
    lo, hi = interval(5, 10)
    assert lo == pytest.approx(0.1870860284, abs=1e-9)
    assert hi == pytest.approx(0.8129139716, abs=1e-9)
    # All failures or all successes have closed forms: (alpha/2)^(1/n).
    assert interval(0, 10) == pytest.approx((0.0, 1 - 0.025 ** (1 / 10)))
    assert interval(10, 10) == pytest.approx((0.025 ** (1 / 10), 1.0))
    # One-sided "rule of three" shape: 0 of 30 gives an upper bound of 1 - 0.05^(1/30).
    assert upper_bound(0, 30) == pytest.approx(1 - 0.05 ** (1 / 30))
    assert lower_bound(30, 30) == pytest.approx(0.05 ** (1 / 30))
    assert lower_bound(0, 30) == 0.0


def test_wilson_textbook_values() -> None:
    # p = 0.5, n = 10, z = 1.95996: centre 0.5, half-width 0.263407.
    lo, hi = interval(5, 10, method="wilson")
    assert lo == pytest.approx(0.2365931, abs=1e-6)
    assert hi == pytest.approx(0.7634069, abs=1e-6)
    assert lower_bound(0, 10, method="wilson") == 0.0
    assert upper_bound(10, 10, method="wilson") == 1.0


def test_bounds_bracket_the_estimate_and_wilson_is_tighter() -> None:
    for k, n in [(97, 100), (1, 3), (450, 500)]:
        cp, wilson = lower_bound(k, n), lower_bound(k, n, method="wilson")
        assert cp <= wilson <= k / n <= upper_bound(k, n)
    # Higher confidence widens the bound.
    assert lower_bound(97, 100, 0.99) < lower_bound(97, 100, 0.95) < lower_bound(97, 100, 0.8)


def _exact_coverage(n: int, p: float, method: BoundMethod, confidence: float = 0.95) -> float:
    """P(lower bound <= p) with k ~ Binomial(n, p), summed exactly over k."""
    return math.fsum(
        math.comb(n, k) * p**k * (1 - p) ** (n - k)
        for k in range(n + 1)
        if lower_bound(k, n, confidence, method) <= p
    )


def test_clopper_pearson_coverage_is_never_below_nominal() -> None:
    for p in np.linspace(0.02, 0.98, 25):
        assert _exact_coverage(40, float(p), "clopper-pearson") >= 0.95 - 1e-12


def test_wilson_coverage_dips_below_nominal_somewhere() -> None:
    # The reason Clopper-Pearson is the default for an error budget.
    worst = min(_exact_coverage(40, float(p), "wilson") for p in np.linspace(0.02, 0.98, 49))
    assert worst < 0.95


@pytest.mark.parametrize(
    ("k", "n", "confidence", "message"),
    [(1, 0, 0.95, "trials"), (5, 4, 0.95, "successes"), (-1, 4, 0.95, "successes"),
     (1, 4, 1.0, "confidence"), (1, 4, 0.0, "confidence")],
)  # fmt: skip
def test_bounds_validate_inputs(k: int, n: int, confidence: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        lower_bound(k, n, confidence)


def test_unknown_bound_method() -> None:
    with pytest.raises(ValueError, match="unknown bound"):
        lower_bound(1, 2, method="agresti")  # type: ignore[arg-type]


# --- candidate thresholds ---------------------------------------------------------------

SCORES = [0.9, 0.7, 0.8, 0.9, 0.7, 0.7]
CORRECT = [True, False, True, True, True, False]


def test_candidates_are_distinct_scores_highest_first_with_ties_grouped() -> None:
    assert candidate_thresholds(SCORES, CORRECT, min_accepted=1) == [
        (0.9, 2, 2),
        (0.8, 3, 3),
        (0.7, 6, 4),
    ]
    # Thresholds accepting fewer than min_accepted decisions are skipped.
    assert candidate_thresholds(SCORES, CORRECT, min_accepted=3) == [(0.8, 3, 3), (0.7, 6, 4)]
    assert candidate_thresholds(SCORES, CORRECT, min_accepted=7) == []


def test_candidates_are_thinned_evenly_keeping_the_lowest() -> None:
    scores = np.linspace(1.0, 0.0, 1000, endpoint=False)
    correct = np.ones(1000)
    thinned = candidate_thresholds(scores, correct, min_accepted=1, max_candidates=10)
    assert len(thinned) == 10
    assert thinned[0][1] == 1
    assert thinned[-1][1] == 1000
    assert [t for t, _, _ in thinned] == sorted((t for t, _, _ in thinned), reverse=True)


@pytest.mark.parametrize(
    ("scores", "correct", "options", "message"),
    [
        ([0.5], [True, False], {}, "same length"),
        ([], [], {}, "at least one"),
        ([1.5], [True], {}, "within"),
        ([float("nan")], [True], {}, "finite"),
        ([0.5], [2], {}, "only 0 and 1"),
        ([0.5], [True], {"min_accepted": 0}, "min_accepted"),
        ([0.5], [True], {"max_candidates": 0}, "max_candidates"),
    ],
)
def test_candidates_validate_inputs(
    scores: list[float], correct: list[float], options: dict[str, int], message: str
) -> None:
    with pytest.raises(ThresholdError, match=message):
        candidate_thresholds(scores, correct, **options)


# --- fixed-sequence selection -----------------------------------------------------------


def test_selection_stops_at_the_first_failure() -> None:
    # Lower bounds (95%, one-sided): 2/2 -> 0.2236, 3/3 -> 0.3684, 4/6 -> 0.2713.
    # With target 0.3 the highest candidate already fails, so nothing is selected even
    # though 0.8 alone would pass: that is the price of not cherry-picking.
    result = select_threshold(SCORES, CORRECT, target_precision=0.3, min_accepted=1)
    assert result.threshold is None
    assert result.selected is None
    assert result.stopped_at is not None
    assert result.stopped_at.threshold == 0.9
    assert result.stopped_at.lower_bound == pytest.approx(0.05**0.5)
    assert result.tested == 1
    assert "even the highest threshold" in result.reason


def test_selection_picks_the_last_passing_candidate() -> None:
    result = select_threshold(SCORES, CORRECT, target_precision=0.3, min_accepted=3)
    assert result.threshold == 0.8
    assert result.selected is not None
    assert result.selected.lower_bound == pytest.approx(0.05 ** (1 / 3))
    assert result.selected.coverage == 0.5
    assert result.selected.precision == 1.0
    assert result.selected.error_rate == 0.0
    assert result.stopped_at is not None
    assert result.stopped_at.threshold == 0.7
    assert result.stopped_at.lower_bound == pytest.approx(0.2713383725197525)
    assert result.tested == 2
    assert result.max_error_rate == pytest.approx(0.7)
    assert "the next candidate 0.7000 failed" in result.reason


def test_selection_when_every_candidate_passes() -> None:
    result = select_threshold(SCORES, CORRECT, target_precision=0.25, min_accepted=3)
    assert result.threshold == 0.7
    assert result.stopped_at is None
    assert result.selected is not None
    assert result.selected.coverage == 1.0
    assert "failed" not in result.reason


def test_selection_with_too_few_decisions() -> None:
    result = select_threshold(SCORES, CORRECT)
    assert result.threshold is None
    assert result.candidates == ()
    assert result.tested == 0
    assert "min_accepted=100" in result.reason


def test_wilson_selects_a_threshold_at_least_as_low() -> None:
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.5, 1.0, 3000)
    correct = rng.uniform(size=3000) < scores
    cp = select_threshold(scores, correct, target_precision=0.9)
    wilson = select_threshold(scores, correct, target_precision=0.9, bound="wilson")
    assert cp.threshold is not None
    assert wilson.threshold is not None
    assert wilson.threshold <= cp.threshold


def test_selected_threshold_meets_the_target_at_the_promised_rate() -> None:
    # A calibrated model: P(correct | score s) = s, scores uniform on [0.5, 1). The true
    # precision of "accept s >= t" is the mean score above t, (1 + t) / 2.
    rng = np.random.default_rng(11)
    target, failures, runs, selected = 0.9, 0, 100, 0
    for _ in range(runs):
        scores = rng.uniform(0.5, 1.0, 1500)
        correct = rng.uniform(size=1500) < scores
        result = select_threshold(
            scores, correct, target_precision=target, min_accepted=100, max_candidates=40
        )
        if result.threshold is None:
            continue
        selected += 1
        failures += (1 + result.threshold) / 2 < target
    assert selected > 90
    # At most 5% in expectation; allow for sampling noise over 100 runs.
    assert failures / runs <= 0.1


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"target_precision": 1.0}, "target_precision"),
        ({"target_precision": 0.0}, "target_precision"),
        ({"confidence": 1.0}, "confidence"),
        ({"bound": "normal"}, "unknown bound"),
    ],
)
def test_selection_validates_options(options: dict[str, object], message: str) -> None:
    with pytest.raises(ThresholdError, match=message):
        select_threshold(SCORES, CORRECT, **options)  # type: ignore[arg-type]


# --- applying a threshold and the display curve -----------------------------------------


def test_apply_threshold_counts_coverage_and_precision() -> None:
    check = apply_threshold(0.8, SCORES, CORRECT)
    assert (check.total, check.accepted, check.correct) == (6, 3, 3)
    assert check.coverage == 0.5
    assert check.precision == 1.0
    none = apply_threshold(0.95, SCORES, CORRECT)
    assert none.accepted == 0
    assert none.precision is None
    with pytest.raises(ThresholdError, match="threshold"):
        apply_threshold(1.5, SCORES, CORRECT)


def test_curve_samples_candidates_and_keeps_the_decision_points() -> None:
    rng = np.random.default_rng(5)
    scores = rng.uniform(0.5, 1.0, 2000)
    correct = rng.uniform(size=2000) < scores
    result = select_threshold(scores, correct, target_precision=0.9)
    points = curve(result, points=5)
    assert result.selected in points
    assert result.stopped_at in points
    assert 5 <= len(points) <= 7
    assert [p.threshold for p in points] == sorted((p.threshold for p in points), reverse=True)
    small = select_threshold(SCORES, CORRECT, target_precision=0.3, min_accepted=1)
    assert curve(small, points=10) == list(small.candidates)
    assert curve(small, points=0) == []
