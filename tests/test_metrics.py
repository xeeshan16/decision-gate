"""Calibration metrics against fixtures worked out by hand (the arithmetic is in comments)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from decision_gate.metrics import (
    ReliabilityBin,
    brier_score,
    calibration_metrics,
    ece_interval,
    expected_calibration_error,
    log_loss,
    maximum_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
    reliability_bins,
)

# Five decisions: confidence of the answer given, and whether it was right.
CONF = [0.9, 0.9, 0.8, 0.6, 0.3]
HIT = [True, False, True, True, False]


def test_brier_score_by_hand() -> None:
    # (0.1^2 + 0.9^2 + 0.2^2 + 0.4^2 + 0.3^2) / 5 = (0.01 + 0.81 + 0.04 + 0.16 + 0.09) / 5
    assert brier_score(CONF, HIT) == pytest.approx(1.11 / 5)


def test_log_loss_by_hand() -> None:
    # -(ln 0.9 + ln 0.1 + ln 0.8 + ln 0.6 + ln 0.7) / 5
    expected = -(math.log(0.9) + math.log(0.1) + math.log(0.8) + math.log(0.6) + math.log(0.7))
    assert log_loss(CONF, HIT) == pytest.approx(expected / 5)
    assert log_loss(CONF, HIT) == pytest.approx(0.6997179, abs=1e-7)


def test_log_loss_stays_finite_at_the_extremes() -> None:
    assert math.isfinite(log_loss([1.0, 0.0], [False, True]))
    assert log_loss([1.0, 0.0], [True, False]) == pytest.approx(0.0, abs=1e-9)


def test_uniform_bins_by_hand() -> None:
    # Edges 0, .2, .4, .6, .8, 1 and right-closed bins, so 0.6 lands in (0.4, 0.6].
    bins = reliability_bins(CONF, HIT, n_bins=5)
    assert [b.lower for b in bins] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert [b.upper for b in bins] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert [b.count for b in bins] == [0, 1, 1, 1, 2]
    assert bins[0] == ReliabilityBin(0.0, 0.2, 0, None, None)
    assert bins[0].gap is None
    assert [b.mean_confidence for b in bins[1:]] == pytest.approx([0.3, 0.6, 0.8, 0.9])
    assert [b.accuracy for b in bins[1:]] == pytest.approx([0.0, 1.0, 1.0, 0.5])
    assert [b.gap for b in bins[1:]] == pytest.approx([0.3, -0.4, -0.2, 0.4])


def test_ece_and_mce_by_hand() -> None:
    # ECE = (1*0.3 + 1*0.4 + 1*0.2 + 2*0.4) / 5 = 0.34; MCE is the largest gap, 0.4.
    assert expected_calibration_error(CONF, HIT, n_bins=5) == pytest.approx(0.34)
    assert maximum_calibration_error(CONF, HIT, n_bins=5) == pytest.approx(0.4)


def test_quantile_bins_by_hand() -> None:
    # Sorted: .3 .6 .8 | .9 .9. First bin: conf 1.7/3, accuracy 2/3, |gap| 0.1.
    # Second: conf 0.9, accuracy 0.5, |gap| 0.4. ECE = (3*0.1 + 2*0.4) / 5 = 0.22.
    bins = reliability_bins(CONF, HIT, n_bins=2, strategy="quantile")
    assert [(b.lower, b.upper, b.count) for b in bins] == [(0.3, 0.8, 3), (0.9, 0.9, 2)]
    assert bins[0].mean_confidence == pytest.approx(1.7 / 3)
    assert bins[0].accuracy == pytest.approx(2 / 3)
    assert expected_calibration_error(CONF, HIT, n_bins=2, strategy="quantile") == (
        pytest.approx(0.22)
    )


def test_quantile_bins_skip_empty_bins_when_samples_are_few() -> None:
    bins = reliability_bins([0.2, 0.7], [False, True], n_bins=5, strategy="quantile")
    assert [b.count for b in bins] == [1, 1]


def test_edges_belong_to_the_lower_bin_and_zero_to_the_first() -> None:
    bins = reliability_bins([0.0, 0.5, 0.5000001, 1.0], [False, True, True, True], n_bins=2)
    assert [b.count for b in bins] == [2, 2]


def test_one_bin_ece_is_the_overall_gap() -> None:
    assert expected_calibration_error(CONF, HIT, n_bins=1) == pytest.approx(abs(0.7 - 0.6))


def test_a_calibrated_model_has_zero_ece() -> None:
    # Confidence 0.75 and three in four right: no gap. Brier = 0.75*0.25^2 + 0.25*0.75^2.
    conf = [0.75] * 8
    hit = [True, True, True, False] * 2
    assert expected_calibration_error(conf, hit) == pytest.approx(0.0)
    assert brier_score(conf, hit) == pytest.approx(0.1875)


def test_multiclass_scores_by_hand() -> None:
    probs = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]]
    labels = [0, 2]
    # Row 1: 0.3^2 + 0.2^2 + 0.1^2 = 0.14. Row 2: 0.1^2 + 0.6^2 + 0.7^2 = 0.86. Mean 0.5.
    assert multiclass_brier_score(probs, labels) == pytest.approx(0.5)
    # -(ln 0.7 + ln 0.3) / 2
    assert multiclass_log_loss(probs, labels) == pytest.approx(-(math.log(0.7) + math.log(0.3)) / 2)


def test_binary_distribution_brier_is_twice_the_top_label_brier() -> None:
    # For two classes, sum_k (p_k - y_k)^2 = 2 (p_true - 1)^2 on every row.
    probs = [[0.9, 0.1], [0.4, 0.6]]
    labels = [0, 1]
    assert multiclass_brier_score(probs, labels) == pytest.approx(
        2 * brier_score([0.9, 0.6], [True, True])
    )


def test_calibration_metrics_collects_everything() -> None:
    m = calibration_metrics(
        CONF, HIT, n_bins=5, probabilities=[[0.9, 0.1]] * 5, labels=[0, 1, 0, 0, 1]
    )
    assert m.count == 5
    assert m.accuracy == pytest.approx(0.6)
    assert m.mean_confidence == pytest.approx(0.7)
    assert m.overconfidence == pytest.approx(0.1)
    assert m.ece == pytest.approx(0.34)
    assert m.mce == pytest.approx(0.4)
    assert m.brier == pytest.approx(0.222)
    assert m.log_loss == pytest.approx(0.6997179, abs=1e-7)
    assert len(m.bins) == 5
    assert (m.n_bins, m.strategy) == (5, "uniform")
    assert m.ece_interval is None
    # Rows labeled 0 score 0.02, rows labeled 1 score 1.62: (3*0.02 + 2*1.62) / 5 = 0.66.
    assert m.multiclass_brier == pytest.approx(0.66)
    assert m.multiclass_log_loss == pytest.approx(-(3 * math.log(0.9) + 2 * math.log(0.1)) / 5)


def test_calibration_metrics_without_a_distribution() -> None:
    m = calibration_metrics(CONF, HIT)
    assert m.multiclass_brier is None
    assert m.multiclass_log_loss is None


def test_bootstrap_interval_is_deterministic_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(7)
    conf = rng.uniform(0.5, 1.0, size=400)
    hit = rng.uniform(size=400) < conf  # calibrated by construction
    low, high = ece_interval(conf, hit, resamples=100, seed=3)
    assert (low, high) == ece_interval(conf, hit, resamples=100, seed=3)
    assert 0.0 <= low <= expected_calibration_error(conf, hit) <= high < 0.1
    m = calibration_metrics(conf, hit, bootstrap=100, seed=3)
    assert m.ece_interval == (low, high)


def test_overconfidence_is_detected_on_a_large_sample() -> None:
    rng = np.random.default_rng(11)
    truth = rng.uniform(0.5, 0.9, size=5000)
    hit = rng.uniform(size=5000) < truth
    claimed = np.minimum(truth + 0.08, 0.999)
    m = calibration_metrics(claimed, hit, bootstrap=50)
    assert m.overconfidence == pytest.approx(0.08, abs=0.02)
    assert m.ece_interval is not None
    assert m.ece_interval[0] > 0.04


@pytest.mark.parametrize(
    ("conf", "hit", "message"),
    [
        ([], [], "at least one"),
        ([0.5, 0.6], [True], "differ in length"),
        ([[0.5]], [[True]], "one-dimensional"),
        ([1.5], [True], "within \\[0, 1\\]"),
        ([float("nan")], [True], "finite"),
        ([0.5], [2], "booleans or 0/1"),
    ],
)
def test_bad_pairs_are_rejected(conf: list[object], hit: list[object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        brier_score(conf, hit)  # type: ignore[arg-type]


def test_bad_binning_and_bootstrap_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        reliability_bins(CONF, HIT, n_bins=0)
    with pytest.raises(ValueError, match="strategy"):
        reliability_bins(CONF, HIT, strategy="log")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resamples"):
        ece_interval(CONF, HIT, resamples=0)
    with pytest.raises(ValueError, match="level"):
        ece_interval(CONF, HIT, level=1.0)


@pytest.mark.parametrize(
    ("probs", "labels", "message"),
    [
        ([[1.0]], [0], "\\(n, k\\)"),
        ([], [], "\\(n, k\\)"),
        ([[0.5, 0.5]], [0, 1], "one entry per row"),
        ([[0.5, 0.5]], [2], "column indices"),
        ([[0.5, 0.5]], [0.0], "column indices"),
        ([[1.5, -0.5]], [0], "within \\[0, 1\\]"),
        ([[0.5, 0.4]], [0], "sum to 1"),
    ],
)
def test_bad_distributions_are_rejected(
    probs: list[list[float]], labels: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        multiclass_brier_score(probs, labels)  # type: ignore[arg-type]


def test_distribution_arguments_must_come_together_and_match() -> None:
    with pytest.raises(ValueError, match="both"):
        calibration_metrics(CONF, HIT, probabilities=[[0.5, 0.5]] * 5)
    with pytest.raises(ValueError, match="one row per decision"):
        calibration_metrics(CONF, HIT, probabilities=[[0.5, 0.5]], labels=[0])
