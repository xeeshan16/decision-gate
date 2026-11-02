"""Calibrators: maps from a backend's raw confidence to a probability that can be trusted.

A calibrator is fitted on labeled decisions (raw confidence, was the answer correct) and
then applied to new decisions. Three methods are implemented, numpy only:

* **Platt scaling** fits ``p = sigmoid(a * logit(c) + b)`` by Newton's method on the
  cross-entropy, with Platt's smoothed targets so that perfectly separable data still
  gives finite parameters. Two parameters, so it needs little data, but it can only
  stretch or shift the confidence curve.
* **Isotonic regression** fits the best non-decreasing step function with the
  pool-adjacent-violators algorithm, then interpolates linearly between steps. It can fix
  any monotone distortion but needs more data and overfits small samples.
* **Temperature scaling** divides the log-probabilities of a full distribution by one
  temperature ``T`` and renormalises. ``T > 1`` softens an overconfident model, ``T < 1``
  sharpens an underconfident one. The ranking of answers never changes, so the decided
  answer stays the same; only its probability moves. It needs a distribution over the
  options, so it applies to predicate and choice questions only.

Every calibrator reports the calibrated probability of the decided answer being correct,
the same "top-label" quantity the metrics module measures. Parameters round-trip through
plain JSON (``params`` / ``from_params``) so fitted calibrators can be stored and versioned.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, get_args

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Method = Literal["platt", "isotonic", "temperature"]
METHODS: tuple[Method, ...] = get_args(Method)

# Raw confidences are clipped to this distance from 0 and 1 before taking the logit.
LOGIT_EPSILON = 1e-6
# Probabilities in a distribution are clipped to this before taking the log.
LOG_EPSILON = 1e-12


class CalibrationError(ValueError):
    """The data cannot be used to fit or apply a calibrator."""


@dataclass(frozen=True, slots=True)
class CalibrationData:
    """Labeled decisions in the shape calibrators need.

    ``confidences`` and ``correct`` are always present. ``distributions`` (the predicted
    probability of every option), ``answers`` (the key of the decided answer) and
    ``truths`` (the key of the true answer) are present only when every decision has a
    distribution, which temperature scaling requires.
    """

    confidences: FloatArray
    correct: FloatArray
    distributions: tuple[Mapping[str, float], ...] | None = None
    answers: tuple[str, ...] | None = None
    truths: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.confidences.ndim != 1 or self.correct.shape != self.confidences.shape:
            raise CalibrationError("confidences and correct must be 1-D and the same length")
        if self.confidences.size == 0:
            raise CalibrationError("at least one labeled decision is needed")
        if not np.all(np.isfinite(self.confidences)) or np.any(
            (self.confidences < 0.0) | (self.confidences > 1.0)
        ):
            raise CalibrationError("confidences must be finite and within [0, 1]")
        if np.any((self.correct != 0.0) & (self.correct != 1.0)):
            raise CalibrationError("correct must contain only 0 and 1")
        parts = (self.distributions, self.answers, self.truths)
        if any(p is not None for p in parts) and any(
            p is None or len(p) != len(self) for p in parts
        ):
            raise CalibrationError(
                "distributions, answers and truths must all be given, one per decision"
            )

    def __len__(self) -> int:
        return int(self.confidences.size)

    @property
    def has_distributions(self) -> bool:
        return self.distributions is not None

    @classmethod
    def from_pairs(
        cls, confidences: Sequence[float] | FloatArray, correct: Sequence[bool] | FloatArray
    ) -> CalibrationData:
        return cls(np.asarray(confidences, dtype=np.float64), np.asarray(correct, np.float64))

    def subset(self, indices: npt.NDArray[np.intp] | slice) -> CalibrationData:
        def pick(values: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
            if values is None:
                return None
            if isinstance(indices, slice):
                return values[indices]
            return tuple(values[i] for i in indices)

        return CalibrationData(
            confidences=self.confidences[indices],
            correct=self.correct[indices],
            distributions=pick(self.distributions),
            answers=pick(self.answers),
            truths=pick(self.truths),
        )


class Calibrator(Protocol):
    method: ClassVar[Method]

    def predict(self, data: CalibrationData) -> FloatArray:
        """Calibrated probability that each decided answer is correct."""
        ...

    def calibrate(
        self, confidence: float, probabilities: Mapping[str, float] | None, answer: str
    ) -> float:
        """Calibrated probability for a single decision (``answer`` is its answer key)."""
        ...

    def params(self) -> dict[str, Any]:
        """JSON-serialisable parameters; ``from_params(params())`` rebuilds the calibrator."""
        ...


def _logit(confidences: FloatArray) -> FloatArray:
    c = np.clip(confidences, LOGIT_EPSILON, 1.0 - LOGIT_EPSILON)
    return np.asarray(np.log(c) - np.log1p(-c), dtype=np.float64)


def _sigmoid(x: FloatArray) -> FloatArray:
    # Split by sign so neither branch overflows.
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    z = np.exp(x[~positive])
    out[~positive] = z / (1.0 + z)
    return out


def _cross_entropy(p: FloatArray, t: FloatArray) -> float:
    p = np.clip(p, LOG_EPSILON, 1.0 - LOG_EPSILON)
    return float(-np.mean(t * np.log(p) + (1.0 - t) * np.log1p(-p)))


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """``p = sigmoid(a * logit(c) + b)``. ``a = 1, b = 0`` is the identity."""

    a: float = 1.0
    b: float = 0.0
    method: ClassVar[Method] = "platt"

    @classmethod
    def fit(
        cls, data: CalibrationData, *, max_iter: int = 100, tol: float = 1e-10
    ) -> PlattCalibrator:
        """Newton's method with a backtracking line search, starting from the identity.

        Targets are Platt's smoothed labels, ``(N+ + 1) / (N+ + 2)`` for correct answers and
        ``1 / (N- + 2)`` for wrong ones: they are a Bayesian prior that keeps ``a`` and ``b``
        finite when the data are separable or all one class.
        """
        x = _logit(data.confidences)
        y = data.correct
        positives = float(y.sum())
        negatives = float(y.size) - positives
        t = np.where(y == 1.0, (positives + 1.0) / (positives + 2.0), 1.0 / (negatives + 2.0))
        design = np.column_stack([x, np.ones_like(x)])
        theta = np.array([1.0, 0.0])
        loss = _cross_entropy(_sigmoid(design @ theta), t)
        for _ in range(max_iter):
            p = _sigmoid(design @ theta)
            gradient = design.T @ (p - t) / y.size
            weights = p * (1.0 - p)
            hessian = (design.T * weights) @ design / y.size + 1e-12 * np.eye(2)
            step = np.linalg.solve(hessian, gradient)
            scale = 1.0
            while scale > 1e-8:
                candidate = theta - scale * step
                candidate_loss = _cross_entropy(_sigmoid(design @ candidate), t)
                if candidate_loss <= loss:
                    break
                scale /= 2.0
            else:  # pragma: no cover - only reached at the optimum, up to rounding
                break
            theta, loss = candidate, candidate_loss
            if float(np.max(np.abs(scale * step))) < tol:
                break
        return cls(a=float(theta[0]), b=float(theta[1]))

    def transform(self, confidences: FloatArray) -> FloatArray:
        return _sigmoid(self.a * _logit(np.asarray(confidences, dtype=np.float64)) + self.b)

    def predict(self, data: CalibrationData) -> FloatArray:
        return self.transform(data.confidences)

    def calibrate(
        self, confidence: float, probabilities: Mapping[str, float] | None, answer: str
    ) -> float:
        return float(self.transform(np.array([confidence]))[0])

    def params(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> PlattCalibrator:
        return cls(a=float(params["a"]), b=float(params["b"]))


def pool_adjacent_violators(
    values: FloatArray, weights: FloatArray
) -> tuple[FloatArray, npt.NDArray[np.intp]]:
    """Weighted least-squares non-decreasing fit to ``values`` (already in x order).

    Returns the fitted value per input and, per input, the index of the block it was
    pooled into. Runs in linear time with a stack of blocks.
    """
    block_sum: list[float] = []
    block_weight: list[float] = []
    block_start: list[int] = []
    for i, (v, w) in enumerate(zip(values.tolist(), weights.tolist(), strict=True)):
        block_sum.append(v * w)
        block_weight.append(w)
        block_start.append(i)
        # Merge while the last block's mean is below the one before it (a violation).
        while (
            len(block_sum) > 1
            and block_sum[-1] / block_weight[-1] < block_sum[-2] / block_weight[-2]
        ):
            s, w_last = block_sum.pop(), block_weight.pop()
            block_start.pop()
            block_sum[-1] += s
            block_weight[-1] += w_last
    fitted = np.empty_like(values)
    block_of = np.empty(values.size, dtype=np.intp)
    ends = [*block_start[1:], values.size]
    for b, (start, end) in enumerate(zip(block_start, ends, strict=True)):
        fitted[start:end] = block_sum[b] / block_weight[b]
        block_of[start:end] = b
    return fitted, block_of


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """A non-decreasing piecewise-linear map through ``(x, y)`` knots.

    Outside the fitted range the end values are held constant. Outputs are kept within
    ``[floor, 1 - floor]`` so a calibrated decision never claims certainty from a finite
    sample (a probability of exactly 0 or 1 would also make log loss infinite).
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    floor: float = 1e-3
    method: ClassVar[Method] = "isotonic"

    def __post_init__(self) -> None:
        if not self.x or len(self.x) != len(self.y):
            raise CalibrationError("isotonic knots must be non-empty and paired")
        if any(b < a for a, b in zip(self.x, self.x[1:], strict=False)) or any(
            b < a for a, b in zip(self.y, self.y[1:], strict=False)
        ):
            raise CalibrationError("isotonic knots must be non-decreasing")
        if not 0.0 <= self.floor < 0.5:
            raise CalibrationError("floor must be within [0, 0.5)")

    @classmethod
    def fit(cls, data: CalibrationData, *, floor: float = 1e-3) -> IsotonicCalibrator:
        order = np.argsort(data.confidences, kind="stable")
        c, y = data.confidences[order], data.correct[order]
        # Equal confidences are one point (their mean accuracy, weighted by count), so the
        # fitted function is well defined at every observed confidence.
        unique, start = np.unique(c, return_index=True)
        counts = np.diff(np.append(start, c.size)).astype(np.float64)
        means = np.add.reduceat(y, start) / counts
        fitted, block_of = pool_adjacent_violators(means, counts)
        # Keep the first and last confidence of each block: the map is flat inside a block
        # and linear between neighbouring blocks.
        boundaries = np.flatnonzero(np.diff(block_of)) + 1
        firsts = np.concatenate([[0], boundaries])
        lasts = np.concatenate([boundaries - 1, [unique.size - 1]])
        keep = np.unique(np.concatenate([firsts, lasts]))
        return cls(
            x=tuple(float(v) for v in unique[keep]),
            y=tuple(float(v) for v in fitted[keep]),
            floor=floor,
        )

    def transform(self, confidences: FloatArray) -> FloatArray:
        mapped = np.interp(np.asarray(confidences, dtype=np.float64), self.x, self.y)
        return np.clip(mapped, self.floor, 1.0 - self.floor)

    def predict(self, data: CalibrationData) -> FloatArray:
        return self.transform(data.confidences)

    def calibrate(
        self, confidence: float, probabilities: Mapping[str, float] | None, answer: str
    ) -> float:
        return float(self.transform(np.array([confidence]))[0])

    def params(self) -> dict[str, Any]:
        return {"x": list(self.x), "y": list(self.y), "floor": self.floor}

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> IsotonicCalibrator:
        return cls(
            x=tuple(float(v) for v in params["x"]),
            y=tuple(float(v) for v in params["y"]),
            floor=float(params.get("floor", 1e-3)),
        )


def _log_matrix(
    distributions: Sequence[Mapping[str, float]], extra: Sequence[str]
) -> tuple[FloatArray, dict[str, int]]:
    """Log-probabilities as a matrix over the union of labels; missing labels get log(eps)."""
    labels = sorted({k for d in distributions for k in d} | set(extra))
    column = {label: i for i, label in enumerate(labels)}
    matrix = np.full((len(distributions), len(labels)), math.log(LOG_EPSILON))
    for i, dist in enumerate(distributions):
        for label, p in dist.items():
            matrix[i, column[label]] = math.log(max(float(p), LOG_EPSILON))
    return matrix, column


def _softmax_rows(z: FloatArray) -> FloatArray:
    shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return np.asarray(e / e.sum(axis=1, keepdims=True), dtype=np.float64)


def temperature_nll(log_p: FloatArray, truth: npt.NDArray[np.intp], temperature: float) -> float:
    """Mean negative log-likelihood of the true labels after scaling by ``temperature``."""
    z = log_p / temperature
    shifted = z - z.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_norm - shifted[np.arange(truth.size), truth]))


@dataclass(frozen=True, slots=True)
class TemperatureScaler:
    """Rescale a distribution: ``p_i ∝ p_i ** (1 / T)``. ``T = 1`` is the identity."""

    temperature: float = 1.0
    method: ClassVar[Method] = "temperature"

    MIN_TEMPERATURE: ClassVar[float] = 0.05
    MAX_TEMPERATURE: ClassVar[float] = 20.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature > 0):
            raise CalibrationError("temperature must be a positive finite number")

    @classmethod
    def fit(cls, data: CalibrationData, *, tol: float = 1e-9) -> TemperatureScaler:
        """Minimise the NLL of the true labels over the inverse temperature.

        The NLL is convex in ``beta = 1 / T`` (its second derivative is the variance of the
        scaled log-probabilities under the scaled distribution), so a golden-section search
        over ``beta`` in ``[1/20, 1/0.05]`` finds the global minimum.
        """
        if data.distributions is None or data.truths is None:
            raise CalibrationError("temperature scaling needs a distribution for every decision")
        log_p, column = _log_matrix(data.distributions, data.truths)
        truth = np.array([column[t] for t in data.truths], dtype=np.intp)

        def loss(beta: float) -> float:
            return temperature_nll(log_p, truth, 1.0 / beta)

        lo, hi = 1.0 / cls.MAX_TEMPERATURE, 1.0 / cls.MIN_TEMPERATURE
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        fc, fd = loss(c), loss(d)
        while hi - lo > tol:
            if fc <= fd:
                hi, d, fd = d, c, fc
                c = hi - ratio * (hi - lo)
                fc = loss(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + ratio * (hi - lo)
                fd = loss(d)
        return cls(temperature=2.0 / (lo + hi))

    def scale(self, probabilities: Mapping[str, float]) -> dict[str, float]:
        """The rescaled distribution, over the same keys."""
        if not probabilities:
            raise CalibrationError("cannot scale an empty distribution")
        keys = list(probabilities)
        z = np.array([[math.log(max(float(probabilities[k]), LOG_EPSILON)) for k in keys]])
        scaled = _softmax_rows(z / self.temperature)[0]
        return {k: float(p) for k, p in zip(keys, scaled, strict=True)}

    def predict(self, data: CalibrationData) -> FloatArray:
        if data.distributions is None or data.answers is None:
            raise CalibrationError("temperature scaling needs a distribution for every decision")
        return np.array(
            [
                self.calibrate(0.0, dist, answer)
                for dist, answer in zip(data.distributions, data.answers, strict=True)
            ],
            dtype=np.float64,
        )

    def calibrate(
        self, confidence: float, probabilities: Mapping[str, float] | None, answer: str
    ) -> float:
        if probabilities is None:
            raise CalibrationError("temperature scaling needs the decision's distribution")
        scaled = self.scale(probabilities)
        if answer not in scaled:
            raise CalibrationError(f"answer {answer!r} is not in the distribution")
        return scaled[answer]

    def params(self) -> dict[str, Any]:
        return {"temperature": self.temperature}

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> TemperatureScaler:
        return cls(temperature=float(params["temperature"]))


AnyCalibrator = PlattCalibrator | IsotonicCalibrator | TemperatureScaler
_CLASSES: dict[str, type[AnyCalibrator]] = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "temperature": TemperatureScaler,
}


def fit_calibrator(method: Method, data: CalibrationData) -> AnyCalibrator:
    """Fit a calibrator of the given method."""
    try:
        cls = _CLASSES[method]
    except KeyError:
        raise CalibrationError(f"unknown method {method!r}; choose from {list(METHODS)}") from None
    return cls.fit(data)


def load_calibrator(method: str, params: Mapping[str, Any]) -> AnyCalibrator:
    """Rebuild a stored calibrator from its method and parameters."""
    try:
        cls = _CLASSES[method]
    except KeyError:
        raise CalibrationError(f"unknown method {method!r}; choose from {list(METHODS)}") from None
    try:
        return cls.from_params(params)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CalibrationError):
            raise
        raise CalibrationError(f"invalid {method} parameters: {exc}") from exc
