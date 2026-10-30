"""A deterministic stand-in for a real decision model.

The fake backend lets the whole system run in tests, CI and demos with no network and no
API keys, while still behaving like a model whose confidence needs calibrating:

* The same question version and input always produce the same answer and confidence.
* Each item gets a hidden "true" probability ``q`` of being answered correctly, and the
  answer is correct with exactly that probability (over the hash of the input).
* The *reported* confidence is ``q`` pushed away from chance by ``overconfidence``
  (a multiplier on the log-odds distance from the chance level).
  With ``overconfidence=1`` the backend is perfectly calibrated; values above 1 reproduce
  the familiar overconfident model, values below 1 an underconfident one.

The ground truth comes from an *oracle*. By default it is derived from the input hash, so
it is stable but arbitrary. Simulations pass their own oracle so that the truth matches
the synthetic dataset they generate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable

from decision_gate.backends.base import (
    AnyQuestion,
    DecisionInput,
    RawDecision,
    clip_probability,
    spread_distribution,
)
from decision_gate.questions import Answer, ChoiceQuestion, PredicateQuestion, ScoreQuestion

Oracle = Callable[[AnyQuestion, DecisionInput], Answer]


def _canonical(data: DecisionInput) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


class _Stream:
    """Deterministic uniform numbers in [0, 1) derived from a seed string."""

    def __init__(self, seed: str) -> None:
        self._seed = seed.encode()
        self._counter = 0

    def next(self) -> float:
        digest = hashlib.sha256(self._seed + self._counter.to_bytes(4, "big")).digest()
        self._counter += 1
        return int.from_bytes(digest[:8], "big") / 2**64


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def hash_oracle(question: AnyQuestion, data: DecisionInput) -> Answer:
    """Default ground truth: a stable pseudo-random answer per question version and input."""
    u = _Stream(f"oracle|{question.key}|{_canonical(data)}").next()
    if isinstance(question, PredicateQuestion):
        return u < 0.5
    if isinstance(question, ChoiceQuestion):
        return question.options[int(u * len(question.options))]
    span = question.max_value - question.min_value
    return question.min_value + u * span


class FakeBackend:
    """Deterministic, configurable fake model. See the module docstring."""

    def __init__(
        self,
        *,
        overconfidence: float = 1.0,
        min_skill: float | None = None,
        oracle: Oracle = hash_oracle,
        seed: str = "decision-gate",
        model_version: str = "fake-1",
    ) -> None:
        if overconfidence <= 0 or not math.isfinite(overconfidence):
            raise ValueError("overconfidence must be a positive finite number")
        if min_skill is not None and not 0.0 < min_skill < 1.0:
            raise ValueError("min_skill must be within (0, 1)")
        self._overconfidence = overconfidence
        self._min_skill = min_skill
        self._oracle = oracle
        self._seed = seed
        self._model_version = model_version

    @property
    def name(self) -> str:
        return "fake"

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        return self.decide_sync(question, data)

    def decide_sync(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        stream = _Stream(f"{self._seed}|{question.key}|{_canonical(data)}")
        truth = question.validate_answer(self._oracle(question, data))

        # Probability of answering correctly. It never drops below chance for the
        # question, otherwise the "model" would be worse than guessing.
        floor = self._min_skill if self._min_skill is not None else self._chance(question)
        true_p = floor + (1.0 - floor) * stream.next()
        correct = stream.next() < true_p

        # Distort in log-odds space around the chance level, so an overconfident model
        # overstates every answer that is better than a guess (and vice versa).
        anchor = _logit(self._chance(question))
        distorted = anchor + (_logit(clip_probability(true_p)) - anchor) * self._overconfidence
        reported = clip_probability(_sigmoid(distorted))
        answer = truth if correct else self._wrong_answer(question, truth, stream)
        return RawDecision(
            answer=answer,
            confidence=reported,
            probabilities=spread_distribution(question, answer, reported),
            backend=self.name,
            model_version=self._model_version,
        )

    @staticmethod
    def _chance(question: AnyQuestion) -> float:
        if isinstance(question, ChoiceQuestion):
            return 1.0 / len(question.options)
        if isinstance(question, PredicateQuestion):
            return 0.5
        # A score question has no natural chance level; start at "coin flip" as well.
        return 0.5

    @staticmethod
    def _wrong_answer(question: AnyQuestion, truth: Answer, stream: _Stream) -> Answer:
        if isinstance(question, PredicateQuestion):
            return not truth
        if isinstance(question, ChoiceQuestion):
            others = [o for o in question.options if o != truth]
            return others[int(stream.next() * len(others))]
        assert isinstance(question, ScoreQuestion)
        assert isinstance(truth, float)
        # Miss by more than the tolerance, staying inside the allowed range.
        miss = question.tolerance * (1.5 + stream.next())
        below, above = truth - miss, truth + miss
        if below >= question.min_value and (above > question.max_value or stream.next() < 0.5):
            return below
        if above <= question.max_value:
            return above
        # Range too narrow to miss on either side by that much: use the far end.
        lo_gap, hi_gap = truth - question.min_value, question.max_value - truth
        return question.min_value if lo_gap > hi_gap else question.max_value
