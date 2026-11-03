"""Split-conformal prediction sets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from decision_gate.conformal import (
    ConformalError,
    ConformalPredictor,
    conformal_quantile,
    distribution_matrix,
    evaluate,
    fit_conformal,
    label_scores,
    min_calibration_size,
)

LABELS = ("billing", "technical", "account")


def test_quantile_uses_the_finite_sample_rank() -> None:
    scores = np.arange(1, 11) / 10.0  # 0.1 .. 1.0
    # alpha 0.2: rank ceil(11 * 0.8) = 9 -> the 9th smallest.
    assert conformal_quantile(scores, 0.2) == pytest.approx(0.9)
    # alpha 0.1: rank ceil(11 * 0.9) = 10 -> the largest.
    assert conformal_quantile(scores, 0.1) == pytest.approx(1.0)
    # alpha 0.05: rank 11 > 10 -> not enough data, every label is in.
    assert conformal_quantile(scores, 0.05) == math.inf
    # Order of the input does not matter.
    assert conformal_quantile(scores[::-1].copy(), 0.2) == pytest.approx(0.9)


def test_min_calibration_size() -> None:
    assert min_calibration_size(0.1) == 9
    assert min_calibration_size(0.05) == 19
    assert min_calibration_size(0.2) == 4
    assert conformal_quantile(np.zeros(9), 0.1) == 0.0
    assert conformal_quantile(np.zeros(8), 0.1) == math.inf


def test_label_scores_by_hand() -> None:
    p = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    np.testing.assert_allclose(label_scores(p, "lac"), [[0.5, 0.7, 0.8], [0.6, 0.6, 0.8]])
    # aps: own probability plus every strictly more probable label; ties share a score.
    np.testing.assert_allclose(label_scores(p, "aps"), [[0.5, 0.8, 1.0], [0.4, 0.4, 1.0]])


def test_distribution_matrix_normalises_and_fills_missing_labels() -> None:
    matrix = distribution_matrix([{"billing": 2.0, "account": 2.0}, {"technical": 1.0}], LABELS)
    np.testing.assert_allclose(matrix, [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])


@pytest.mark.parametrize(
    ("distributions", "labels", "message"),
    [
        ([{"refunds": 1.0}], LABELS, "not one of"),
        ([{"billing": 0.0}], LABELS, "probability mass"),
        ([{"billing": -0.1, "account": 1.0}], LABELS, "non-negative"),
        ([{"billing": float("nan")}], LABELS, "finite"),
        ([{"billing": 1.0}], ("billing", "billing"), "unique"),
        ([{"billing": 1.0}], (), "at least one label"),
    ],
)
def test_distribution_matrix_rejects_bad_input(
    distributions: list[dict[str, float]], labels: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ConformalError, match=message):
        distribution_matrix(distributions, labels)


def _hand_case() -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.1, 0.8],
            [0.4, 0.35, 0.25],
        ]
    )
    truths = np.array([0, 1, 1, 2, 2], dtype=np.intp)
    return probabilities, truths


def test_fit_and_predict_sets_by_hand() -> None:
    probabilities, truths = _hand_case()
    # LAC scores of the true labels: 0.3, 0.7, 0.5, 0.2, 0.75. With alpha 0.34 the rank is
    # ceil(6 * 0.66) = 4, so the quantile is the 4th smallest, 0.7.
    predictor = fit_conformal(probabilities, truths, LABELS, alpha=0.34)
    assert predictor.quantile == pytest.approx(0.7)
    assert predictor.calibration_count == 5
    assert not predictor.is_trivial
    # Labels with p >= 0.3, most probable first.
    assert predictor.predict_set({"billing": 0.6, "technical": 0.3, "account": 0.1}) == [
        "billing",
        "technical",
    ]
    assert predictor.predict_set({"billing": 0.25, "technical": 0.25, "account": 0.5}) == [
        "account"
    ]
    # A looser alpha (rank ceil(6 * 0.5) = 3, quantile 0.5) keeps labels with p >= 0.5, so
    # a model unsure of everything gets an empty set.
    loose = fit_conformal(probabilities, truths, LABELS, alpha=0.5)
    assert loose.quantile == pytest.approx(0.5)
    assert loose.predict_set({"billing": 0.4, "technical": 0.3, "account": 0.3}) == []
    assert loose.predict_set({"billing": 0.5, "technical": 0.3, "account": 0.2}) == ["billing"]


def test_aps_sets_by_hand() -> None:
    probabilities, truths = _hand_case()
    # APS scores of the true labels: 0.7, 0.9, 0.5, 0.8, 1.0 -> 4th smallest is 0.9.
    predictor = fit_conformal(probabilities, truths, LABELS, alpha=0.34, method="aps")
    assert predictor.quantile == pytest.approx(0.9)
    assert predictor.predict_set({"billing": 0.6, "technical": 0.3, "account": 0.1}) == [
        "billing",
        "technical",
    ]
    # billing's own score is 0.95 > 0.9, but the top label is always kept: never empty.
    assert predictor.predict_set({"billing": 0.95, "technical": 0.03, "account": 0.02}) == [
        "billing"
    ]
    flat = np.full((1, 3), 1 / 3)
    assert predictor.set_mask(flat).all()


def test_trivial_predictor_includes_every_label() -> None:
    probabilities, truths = _hand_case()
    predictor = fit_conformal(probabilities, truths, LABELS, alpha=0.1)
    assert predictor.is_trivial
    assert set(predictor.predict_set({"billing": 1.0})) == set(LABELS)


def test_params_round_trip_including_infinity() -> None:
    probabilities, truths = _hand_case()
    for alpha in (0.34, 0.1):
        predictor = fit_conformal(probabilities, truths, LABELS, alpha=alpha, method="aps")
        params = predictor.params()
        assert ConformalPredictor.from_params(params) == predictor
    assert fit_conformal(probabilities, truths, LABELS, alpha=0.1).params()["quantile"] is None
    with pytest.raises(ConformalError, match="unknown method"):
        ConformalPredictor.from_params({**params, "method": "raps"})


def test_evaluate_by_hand() -> None:
    probabilities, truths = _hand_case()
    predictor = fit_conformal(probabilities, truths, LABELS, alpha=0.34)
    result = evaluate(predictor, probabilities, truths)
    # Sets (p >= 0.3): {0}, {0, 1}, {1, 2}, {2}, {0, 1}. The last misses label 2.
    assert result.count == 5
    assert result.coverage == pytest.approx(0.8)
    assert result.mean_size == pytest.approx(8 / 5)
    assert result.singleton_rate == pytest.approx(0.4)
    assert result.singleton_accuracy == 1.0
    assert result.empty_rate == 0.0
    assert result.size_counts == {1: 2, 2: 3}
    loose = fit_conformal(probabilities, truths, LABELS, alpha=0.5)
    flat = np.full((2, 3), 1 / 3)
    empty = evaluate(loose, flat, np.array([0, 1], dtype=np.intp))
    assert empty.empty_rate == 1.0
    assert empty.singleton_accuracy is None
    with pytest.raises(ConformalError, match="at least one"):
        evaluate(predictor, probabilities[:0], truths[:0])


def _synthetic(rng: np.random.Generator, n: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Random distributions with the truth drawn from a sharpened version of each one.

    The model is miscalibrated on purpose: conformal coverage must hold anyway.
    """
    probabilities = rng.dirichlet(np.full(k, 0.6), size=n)
    sharp = probabilities**2
    sharp /= sharp.sum(axis=1, keepdims=True)
    truths = np.array([rng.choice(k, p=row) for row in sharp], dtype=np.intp)
    return probabilities, truths


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_marginal_coverage_holds_on_a_miscalibrated_model(method: str) -> None:
    rng = np.random.default_rng(42)
    labels = tuple(f"l{i}" for i in range(5))
    alpha, n_cal, coverages = 0.1, 300, []
    for _ in range(150):
        p, t = _synthetic(rng, n_cal + 300)
        predictor = fit_conformal(p[:n_cal], t[:n_cal], labels, alpha=alpha, method=method)  # type: ignore[arg-type]
        coverages.append(evaluate(predictor, p[n_cal:], t[n_cal:]).coverage)
    mean = float(np.mean(coverages))
    # Expected coverage is at least 1 - alpha; the mean of 150 runs of 300 test decisions
    # has a standard error near 0.0014. LAC is also at most 1 - alpha + 1 / (n + 1); APS
    # always keeps the top label, which makes it conservative, so it has no upper bound.
    assert mean >= 1 - alpha - 0.006
    if method == "lac":
        assert mean <= 1 - alpha + 1 / (n_cal + 1) + 0.006


def test_aps_adapts_set_size_to_difficulty() -> None:
    rng = np.random.default_rng(7)
    p, t = _synthetic(rng, 2000)
    labels = tuple(f"l{i}" for i in range(5))
    predictor = fit_conformal(p[:1000], t[:1000], labels, alpha=0.1, method="aps")
    sizes = predictor.set_mask(p[1000:]).sum(axis=1)
    top = p[1000:].max(axis=1)
    # Confident inputs get smaller sets than unsure ones.
    assert sizes[top > 0.8].mean() < sizes[top < 0.4].mean()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"method": "raps"}, "unknown method"),
    ],
)
def test_fit_validates_options(kwargs: dict[str, object], message: str) -> None:
    probabilities, truths = _hand_case()
    with pytest.raises(ConformalError, match=message):
        fit_conformal(probabilities, truths, LABELS, **kwargs)  # type: ignore[arg-type]


def test_fit_validates_shapes() -> None:
    probabilities, truths = _hand_case()
    with pytest.raises(ConformalError, match="one truth per row"):
        fit_conformal(probabilities, truths[:3], LABELS)
    with pytest.raises(ConformalError, match="at least one"):
        fit_conformal(probabilities[:0], truths[:0], LABELS)
    with pytest.raises(ConformalError, match="index into"):
        fit_conformal(probabilities, np.array([0, 1, 2, 3, 0], dtype=np.intp), LABELS)
    with pytest.raises(ConformalError, match="one column per label"):
        fit_conformal(probabilities, truths, LABELS).set_mask(probabilities[:, :2])
    with pytest.raises(ConformalError, match="non-empty 2-D"):
        label_scores(np.zeros(3), "lac")
    with pytest.raises(ConformalError, match="unknown method"):
        label_scores(probabilities, "raps")  # type: ignore[arg-type]
    with pytest.raises(ConformalError, match="at least one calibration score"):
        conformal_quantile(np.zeros(0), 0.1)
    with pytest.raises(ConformalError, match="alpha"):
        min_calibration_size(1.0)
