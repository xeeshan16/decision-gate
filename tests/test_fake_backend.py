from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from decision_gate.backends import DecisionBackend, FakeBackend, RawDecision
from decision_gate.backends.base import AnyQuestion, answer_key, clip_probability
from decision_gate.backends.fake import hash_oracle
from decision_gate.questions import Answer, ChoiceQuestion, PredicateQuestion, ScoreQuestion


def _inputs(n: int) -> list[dict[str, Any]]:
    return [{"ticket": i, "text": f"synthetic ticket {i}"} for i in range(n)]


def test_satisfies_protocol() -> None:
    assert isinstance(FakeBackend(), DecisionBackend)


def test_deterministic(predicate: PredicateQuestion) -> None:
    a = FakeBackend().decide_sync(predicate, {"x": 1, "y": [1, 2]})
    b = FakeBackend().decide_sync(predicate, {"y": [1, 2], "x": 1})  # key order irrelevant
    assert a == b


def test_seed_changes_answers(choice: ChoiceQuestion) -> None:
    a = [FakeBackend(seed="s1").decide_sync(choice, d) for d in _inputs(50)]
    b = [FakeBackend(seed="s2").decide_sync(choice, d) for d in _inputs(50)]
    assert a != b


def test_async_matches_sync(choice: ChoiceQuestion) -> None:
    backend = FakeBackend()
    assert asyncio.run(backend.decide(choice, {"a": 1})) == backend.decide_sync(choice, {"a": 1})


@pytest.mark.parametrize("fixture", ["predicate", "choice", "score"])
def test_answers_are_valid(fixture: str, request: pytest.FixtureRequest) -> None:
    question: AnyQuestion = request.getfixturevalue(fixture)
    backend = FakeBackend(overconfidence=3.0)
    for data in _inputs(300):
        decision = backend.decide_sync(question, data)
        question.validate_answer(decision.answer)
        assert 0.0 < decision.confidence < 1.0
        if decision.probabilities is not None:
            assert math.isclose(sum(decision.probabilities.values()), 1.0)
            assert decision.probabilities[answer_key(decision.answer)] == pytest.approx(
                decision.confidence
            )


def _accuracy_and_mean_confidence(
    backend: FakeBackend, question: AnyQuestion, n: int = 4000
) -> tuple[float, float]:
    correct = 0
    total_conf = 0.0
    for data in _inputs(n):
        decision = backend.decide_sync(question, data)
        correct += question.is_correct(decision.answer, hash_oracle(question, data))
        total_conf += decision.confidence
    return correct / n, total_conf / n


@pytest.mark.parametrize("fixture", ["predicate", "choice", "score"])
def test_calibrated_when_overconfidence_is_one(
    fixture: str, request: pytest.FixtureRequest
) -> None:
    question: AnyQuestion = request.getfixturevalue(fixture)
    accuracy, confidence = _accuracy_and_mean_confidence(FakeBackend(overconfidence=1.0), question)
    # 4000 Bernoulli draws: the standard error is under 0.008, so 0.03 is a safe margin.
    assert abs(accuracy - confidence) < 0.03


@pytest.mark.parametrize("fixture", ["predicate", "choice"])
def test_overconfident_backend_overstates(fixture: str, request: pytest.FixtureRequest) -> None:
    question: AnyQuestion = request.getfixturevalue(fixture)
    accuracy, confidence = _accuracy_and_mean_confidence(FakeBackend(overconfidence=3.0), question)
    assert confidence - accuracy > 0.08


def test_underconfident_backend_understates(predicate: PredicateQuestion) -> None:
    accuracy, confidence = _accuracy_and_mean_confidence(FakeBackend(overconfidence=0.4), predicate)
    assert accuracy - confidence > 0.08


def test_min_skill_raises_accuracy(choice: ChoiceQuestion) -> None:
    accuracy, _ = _accuracy_and_mean_confidence(FakeBackend(min_skill=0.9), choice, n=2000)
    assert accuracy > 0.9


def test_custom_oracle_is_followed(choice: ChoiceQuestion) -> None:
    def oracle(question: AnyQuestion, data: Any) -> Answer:
        return "c"

    backend = FakeBackend(oracle=oracle, min_skill=0.99)
    answers = [backend.decide_sync(choice, d).answer for d in _inputs(200)]
    assert answers.count("c") > 190


def test_invalid_oracle_answer_rejected(choice: ChoiceQuestion) -> None:
    backend = FakeBackend(oracle=lambda q, d: "not-an-option")
    with pytest.raises(ValueError, match="one of"):
        backend.decide_sync(choice, {})


def test_wrong_score_answers_miss_by_more_than_tolerance(score: ScoreQuestion) -> None:
    backend = FakeBackend(overconfidence=1.0)
    misses = 0
    for data in _inputs(1000):
        decision = backend.decide_sync(score, data)
        truth = hash_oracle(score, data)
        assert isinstance(truth, float)
        assert isinstance(decision.answer, float)
        if decision.answer != truth:
            misses += 1
            assert abs(decision.answer - truth) > score.tolerance
    assert misses > 100


def test_score_miss_in_narrow_range_uses_far_end() -> None:
    # Range 10 with tolerance 4: a miss of at least 6 cannot fit on both sides of 5.
    question = ScoreQuestion(id="narrow", prompt="p", min_value=0, max_value=10, tolerance=4)
    backend = FakeBackend(oracle=lambda q, d: 5.0, min_skill=0.01)
    answers = {backend.decide_sync(question, d).answer for d in _inputs(200)}
    assert answers <= {0.0, 10.0, 5.0}
    assert answers & {0.0, 10.0}


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_rejects_bad_overconfidence(value: float) -> None:
    with pytest.raises(ValueError, match="overconfidence"):
        FakeBackend(overconfidence=value)


@pytest.mark.parametrize("value", [0.0, 1.0, 1.5])
def test_rejects_bad_min_skill(value: float) -> None:
    with pytest.raises(ValueError, match="min_skill"):
        FakeBackend(min_skill=value)


def test_clip_probability() -> None:
    assert 0.0 < clip_probability(0.0) < 1e-5
    assert 1.0 - 1e-5 < clip_probability(1.0) < 1.0
    with pytest.raises(ValueError, match="finite"):
        clip_probability(math.nan)


def test_raw_decision_rejects_bad_distribution() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        RawDecision(
            answer=True,
            confidence=0.7,
            probabilities={"true": 0.7, "false": 0.7},
            backend="x",
            model_version="1",
        )
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        RawDecision(
            answer=True,
            confidence=0.7,
            probabilities={"true": 1.5, "false": -0.5},
            backend="x",
            model_version="1",
        )


def test_answer_key() -> None:
    assert answer_key(True) == "true"
    assert answer_key(False) == "false"
    assert answer_key("billing") == "billing"
