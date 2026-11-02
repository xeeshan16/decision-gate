"""Calibrators: hand-computed fixtures, optimality conditions and recovery of known distortions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from decision_gate.calibrators import (
    METHODS,
    CalibrationData,
    CalibrationError,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
    fit_calibrator,
    load_calibrator,
    pool_adjacent_violators,
    temperature_nll,
)


def pairs(confidences: list[float], correct: list[int]) -> CalibrationData:
    return CalibrationData.from_pairs(confidences, [bool(y) for y in correct])


def with_distributions(
    distributions: list[dict[str, float]], answers: list[str], truths: list[str]
) -> CalibrationData:
    confidences = [d[a] for d, a in zip(distributions, answers, strict=True)]
    correct = [float(a == t) for a, t in zip(answers, truths, strict=True)]
    return CalibrationData(
        np.array(confidences),
        np.array(correct),
        distributions=tuple(distributions),
        answers=tuple(answers),
        truths=tuple(truths),
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


class TestCalibrationData:
    def test_validation(self) -> None:
        with pytest.raises(CalibrationError, match="same length"):
            CalibrationData(np.array([0.5, 0.6]), np.array([1.0]))
        with pytest.raises(CalibrationError, match="at least one"):
            CalibrationData(np.array([]), np.array([]))
        with pytest.raises(CalibrationError, match="within"):
            pairs([1.2], [1])
        with pytest.raises(CalibrationError, match="within"):
            pairs([math.nan], [1])
        with pytest.raises(CalibrationError, match="0 and 1"):
            CalibrationData(np.array([0.5]), np.array([0.5]))
        with pytest.raises(CalibrationError, match="all be given"):
            CalibrationData(np.array([0.5]), np.array([1.0]), distributions=({"a": 0.5},))
        with pytest.raises(CalibrationError, match="all be given"):
            CalibrationData(
                np.array([0.5]),
                np.array([1.0]),
                distributions=({"a": 0.5},),
                answers=("a", "b"),
                truths=("a",),
            )

    def test_subset_by_slice_and_index(self) -> None:
        data = with_distributions(
            [{"a": 0.6, "b": 0.4}, {"a": 0.3, "b": 0.7}, {"a": 0.9, "b": 0.1}],
            ["a", "b", "a"],
            ["a", "a", "b"],
        )
        head = data.subset(slice(0, 2))
        assert len(head) == 2
        assert head.answers == ("a", "b")
        picked = data.subset(np.array([2, 0]))
        assert picked.confidences.tolist() == [0.9, 0.6]
        assert picked.truths == ("b", "a")
        plain = pairs([0.1, 0.2], [0, 1]).subset(np.array([1]))
        assert plain.distributions is None
        assert not plain.has_distributions


class TestPoolAdjacentViolators:
    def test_hand_fixtures(self) -> None:
        fitted, blocks = pool_adjacent_violators(np.array([1.0, 0.0, 1.0, 1.0]), np.ones(4))
        assert fitted.tolist() == [0.5, 0.5, 1.0, 1.0]
        assert blocks.tolist() == [0, 0, 1, 2]
        # 3 and 1 pool to 2; the next 2 is not a violation, so it stays its own block.
        fitted, blocks = pool_adjacent_violators(np.array([3.0, 1.0, 2.0]), np.ones(3))
        assert fitted.tolist() == [2.0, 2.0, 2.0]
        assert blocks.tolist() == [0, 0, 1]
        # 3 and 0 pool to 1.5, which is now below the 2 before it, so all three pool.
        fitted, _ = pool_adjacent_violators(np.array([2.0, 3.0, 0.0]), np.ones(3))
        assert fitted.tolist() == pytest.approx([5 / 3] * 3)
        # Without the cascade, 1 stays on its own: [1, 1.5, 1.5].
        fitted, _ = pool_adjacent_violators(np.array([1.0, 3.0, 0.0]), np.ones(3))
        assert fitted.tolist() == [1.0, 1.5, 1.5]

    def test_weights(self) -> None:
        fitted, _ = pool_adjacent_violators(np.array([1.0, 0.0]), np.array([3.0, 1.0]))
        assert fitted.tolist() == [0.75, 0.75]

    def test_result_is_monotone_and_least_squares_mean_preserving(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.random(500)
        weights = rng.integers(1, 5, 500).astype(np.float64)
        fitted, _ = pool_adjacent_violators(values, weights)
        assert np.all(np.diff(fitted) >= -1e-12)
        # Pooling replaces values by weighted block means, so the weighted sum is kept.
        assert float(fitted @ weights) == pytest.approx(float(values @ weights))


class TestIsotonic:
    def test_hand_fixture(self) -> None:
        data = pairs([0.1, 0.2, 0.3, 0.4], [1, 0, 1, 1])
        iso = IsotonicCalibrator.fit(data, floor=0.0)
        assert iso.x == (0.1, 0.2, 0.3, 0.4)
        assert iso.y == (0.5, 0.5, 1.0, 1.0)
        out = iso.transform(np.array([0.05, 0.15, 0.25, 0.35, 0.99]))
        assert out.tolist() == pytest.approx([0.5, 0.5, 0.75, 1.0, 1.0])

    def test_floor_keeps_away_from_certainty(self) -> None:
        iso = IsotonicCalibrator.fit(pairs([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]))
        assert iso.calibrate(0.4, None, "x") == pytest.approx(0.999)
        assert iso.calibrate(0.1, None, "x") == pytest.approx(0.001)

    def test_ties_are_one_point_at_their_mean(self) -> None:
        iso = IsotonicCalibrator.fit(pairs([0.5, 0.5, 0.5, 0.9], [1, 0, 0, 1]), floor=0.0)
        assert iso.x == (0.5, 0.9)
        assert iso.y == pytest.approx((1 / 3, 1.0))

    def test_flat_blocks_keep_only_their_ends(self) -> None:
        iso = IsotonicCalibrator.fit(pairs([0.1, 0.2, 0.3, 0.4, 0.5], [1, 0, 0, 0, 1]), floor=0)
        # 1,0,0,0 pools to 0.25 over [0.1, 0.4]; the inner knots 0.2 and 0.3 are redundant.
        assert iso.x == (0.1, 0.4, 0.5)
        assert iso.y == (0.25, 0.25, 1.0)

    def test_monotone_on_noisy_data(self) -> None:
        rng = np.random.default_rng(1)
        c = rng.random(2000)
        y = (rng.random(2000) < c**2).astype(np.float64)
        iso = IsotonicCalibrator.fit(CalibrationData.from_pairs(c, y))
        grid = iso.transform(np.linspace(0, 1, 101))
        assert np.all(np.diff(grid) >= 0)
        assert grid[0] >= 0.001
        assert grid[-1] <= 0.999

    def test_invalid_knots(self) -> None:
        with pytest.raises(CalibrationError, match="non-empty"):
            IsotonicCalibrator(x=(), y=())
        with pytest.raises(CalibrationError, match="non-decreasing"):
            IsotonicCalibrator(x=(0.2, 0.1), y=(0.1, 0.2))
        with pytest.raises(CalibrationError, match="non-decreasing"):
            IsotonicCalibrator(x=(0.1, 0.2), y=(0.3, 0.2))
        with pytest.raises(CalibrationError, match="floor"):
            IsotonicCalibrator(x=(0.1,), y=(0.1,), floor=0.5)


class TestPlatt:
    def test_constant_confidence_fits_the_smoothed_base_rate(self) -> None:
        # Three right, one wrong at the same confidence. Platt's targets are
        # (3 + 1) / (3 + 2) = 0.8 and 1 / (1 + 2) = 1/3, and with one distinct input the
        # best prediction is their mean: (3 * 0.8 + 1/3) / 4 = 41/60.
        platt = PlattCalibrator.fit(pairs([0.7, 0.7, 0.7, 0.7], [1, 1, 1, 0]))
        assert platt.calibrate(0.7, None, "x") == pytest.approx(41 / 60, abs=1e-8)

    def test_gradient_is_zero_at_the_fit(self) -> None:
        rng = np.random.default_rng(3)
        c = rng.uniform(0.05, 0.95, 300)
        y = (rng.random(300) < c**1.5).astype(np.float64)
        platt = PlattCalibrator.fit(CalibrationData(c, y))
        x = np.log(c / (1 - c))
        positives, negatives = y.sum(), y.size - y.sum()
        t = np.where(y == 1, (positives + 1) / (positives + 2), 1 / (negatives + 2))
        residual = platt.transform(c) - t
        assert abs(residual.mean()) < 1e-8
        assert abs((residual * x).mean()) < 1e-8

    def test_recovers_a_known_overconfidence(self) -> None:
        rng = np.random.default_rng(11)
        z = rng.normal(0.0, 1.5, 20_000)
        reported = sigmoid(2.0 * z)  # twice as far from 0.5 in log-odds as the truth
        correct = (rng.random(z.size) < sigmoid(z)).astype(np.float64)
        platt = PlattCalibrator.fit(CalibrationData(reported, correct))
        assert platt.a == pytest.approx(0.5, abs=0.03)
        assert platt.b == pytest.approx(0.0, abs=0.03)

    def test_calibrated_input_stays_close_to_identity(self) -> None:
        rng = np.random.default_rng(5)
        c = rng.uniform(0.02, 0.98, 20_000)
        correct = (rng.random(c.size) < c).astype(np.float64)
        platt = PlattCalibrator.fit(CalibrationData(c, correct))
        assert platt.a == pytest.approx(1.0, abs=0.05)
        assert platt.b == pytest.approx(0.0, abs=0.05)

    def test_separable_data_gives_finite_bounded_predictions(self) -> None:
        platt = PlattCalibrator.fit(pairs([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]))
        assert math.isfinite(platt.a)
        assert math.isfinite(platt.b)
        # Targets are 3/4 and 1/4, so the fit cannot claim more than that by much.
        assert 0.5 < platt.calibrate(0.9, None, "x") < 0.8
        assert 0.2 < platt.calibrate(0.1, None, "x") < 0.5

    def test_all_one_class(self) -> None:
        platt = PlattCalibrator.fit(pairs([0.6, 0.7, 0.8], [1, 1, 1]))
        # Target (3 + 1) / (3 + 2) = 0.8 for every decision.
        assert platt.transform(np.array([0.6, 0.7, 0.8])).mean() == pytest.approx(0.8, abs=1e-6)

    def test_extreme_confidences_are_clipped(self) -> None:
        platt = PlattCalibrator(a=1.0, b=0.0)
        out = platt.transform(np.array([0.0, 1.0]))
        assert out.tolist() == pytest.approx([1e-6, 1 - 1e-6])


class TestTemperature:
    def test_hand_fixture(self) -> None:
        scaler = TemperatureScaler(temperature=2.0)
        # p ** (1/2): sqrt(0.8) : sqrt(0.2) = 2 : 1.
        assert scaler.scale({"a": 0.8, "b": 0.2}) == pytest.approx({"a": 2 / 3, "b": 1 / 3})
        log_p = np.log(np.array([[0.8, 0.2]]))
        truth = np.array([0], dtype=np.intp)
        assert temperature_nll(log_p, truth, 1.0) == pytest.approx(-math.log(0.8))
        assert temperature_nll(log_p, truth, 2.0) == pytest.approx(math.log(1.5))

    def test_identity_and_ranking(self) -> None:
        dist = {"a": 0.5, "b": 0.3, "c": 0.2}
        assert TemperatureScaler().scale(dist) == pytest.approx(dist)
        for t in (0.3, 3.0):
            scaled = TemperatureScaler(temperature=t).scale(dist)
            assert sum(scaled.values()) == pytest.approx(1.0)
            assert sorted(scaled, key=scaled.__getitem__) == ["c", "b", "a"]

    def test_fit_is_a_minimum(self) -> None:
        rng = np.random.default_rng(2)
        distributions, answers, truths = [], [], []
        for _ in range(400):
            p = rng.dirichlet([1.0, 1.0, 1.0])
            dist = dict(zip("abc", p.tolist(), strict=True))
            distributions.append(dist)
            answers.append(max(dist, key=dist.__getitem__))
            truths.append(str(rng.choice(list("abc"), p=p)))
        data = with_distributions(distributions, answers, truths)
        fitted = TemperatureScaler.fit(data).temperature
        log_p = np.log(np.array([[d[k] for k in "abc"] for d in distributions]))
        truth = np.array(["abc".index(t) for t in truths], dtype=np.intp)
        best = temperature_nll(log_p, truth, fitted)
        assert best <= temperature_nll(log_p, truth, fitted * 1.01)
        assert best <= temperature_nll(log_p, truth, fitted / 1.01)

    def test_recovers_a_known_temperature(self) -> None:
        rng = np.random.default_rng(13)
        logits = rng.normal(0.0, 1.5, (8000, 4))
        true_p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        reported = np.exp(2 * logits) / np.exp(2 * logits).sum(axis=1, keepdims=True)
        labels = "wxyz"
        distributions = [dict(zip(labels, row.tolist(), strict=True)) for row in reported]
        answers = [labels[int(i)] for i in reported.argmax(axis=1)]
        truths = [labels[int(rng.choice(4, p=p))] for p in true_p]
        scaler = TemperatureScaler.fit(with_distributions(distributions, answers, truths))
        assert scaler.temperature == pytest.approx(2.0, rel=0.05)

    def test_always_right_sharpens_to_the_bound(self) -> None:
        data = with_distributions([{"a": 0.6, "b": 0.4}] * 5, ["a"] * 5, ["a"] * 5)
        assert TemperatureScaler.fit(data).temperature == pytest.approx(0.05, rel=1e-4)

    def test_predict_and_calibrate(self) -> None:
        data = with_distributions(
            [{"a": 0.8, "b": 0.2}, {"a": 0.2, "b": 0.8}], ["a", "b"], ["a", "a"]
        )
        scaler = TemperatureScaler(temperature=2.0)
        assert scaler.predict(data).tolist() == pytest.approx([2 / 3, 2 / 3])
        assert scaler.calibrate(0.8, {"a": 0.8, "b": 0.2}, "a") == pytest.approx(2 / 3)

    def test_errors(self) -> None:
        plain = pairs([0.6], [1])
        with pytest.raises(CalibrationError, match="distribution"):
            TemperatureScaler.fit(plain)
        with pytest.raises(CalibrationError, match="distribution"):
            TemperatureScaler().predict(plain)
        with pytest.raises(CalibrationError, match="distribution"):
            TemperatureScaler().calibrate(0.6, None, "a")
        with pytest.raises(CalibrationError, match="not in the distribution"):
            TemperatureScaler().calibrate(0.6, {"a": 0.6, "b": 0.4}, "c")
        with pytest.raises(CalibrationError, match="empty"):
            TemperatureScaler().scale({})
        for bad in (0.0, -1.0, math.inf):
            with pytest.raises(CalibrationError, match="positive"):
                TemperatureScaler(temperature=bad)


class TestRegistryHelpers:
    @pytest.mark.parametrize("method", METHODS)
    def test_params_round_trip(self, method: str) -> None:
        rng = np.random.default_rng(0)
        distributions = [
            dict(zip("ab", p.tolist(), strict=True)) for p in rng.dirichlet([2, 2], 50)
        ]
        answers = [max(d, key=d.__getitem__) for d in distributions]
        truths = [str(rng.choice(["a", "b"])) for _ in distributions]
        data = with_distributions(distributions, answers, truths)
        fitted = fit_calibrator(method, data)  # type: ignore[arg-type]
        assert fitted.method == method
        rebuilt = load_calibrator(method, fitted.params())
        assert rebuilt == fitted
        assert rebuilt.predict(data).tolist() == pytest.approx(fitted.predict(data).tolist())

    def test_unknown_method_and_bad_params(self) -> None:
        with pytest.raises(CalibrationError, match="unknown method"):
            fit_calibrator("beta", pairs([0.5], [1]))  # type: ignore[arg-type]
        with pytest.raises(CalibrationError, match="unknown method"):
            load_calibrator("beta", {})
        with pytest.raises(CalibrationError, match="invalid platt parameters"):
            load_calibrator("platt", {"a": 1.0})
        with pytest.raises(CalibrationError, match="invalid temperature parameters"):
            load_calibrator("temperature", {"temperature": "hot"})
        with pytest.raises(CalibrationError, match="non-decreasing"):
            load_calibrator("isotonic", {"x": [0.2, 0.1], "y": [0.1, 0.2]})

    def test_isotonic_params_default_floor(self) -> None:
        iso = load_calibrator("isotonic", {"x": [0.1, 0.9], "y": [0.2, 0.8]})
        assert isinstance(iso, IsotonicCalibrator)
        assert iso.floor == 1e-3
