from __future__ import annotations

import math

import pytest
from pydantic import TypeAdapter, ValidationError

from decision_gate.questions import (
    ChoiceQuestion,
    PredicateQuestion,
    Question,
    QuestionCatalog,
    ScoreQuestion,
)


@pytest.mark.parametrize("qid", ["ab", "refund.eligible", "a-b_c.d9", "x" * 64])
def test_valid_ids(qid: str) -> None:
    assert PredicateQuestion(id=qid, prompt="p").id == qid


@pytest.mark.parametrize("qid", ["a", "Refund", "1abc", "abc.", "has space", "x" * 65, ""])
def test_invalid_ids(qid: str) -> None:
    with pytest.raises(ValidationError, match="id must be"):
        PredicateQuestion(id=qid, prompt="p")


def test_key_includes_version() -> None:
    assert PredicateQuestion(id="q1", version=3, prompt="p").key == "q1@v3"


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        PredicateQuestion.model_validate({"id": "q1", "prompt": "p", "opts": 1})


def test_discriminated_union_parses_each_kind() -> None:
    adapter: TypeAdapter[PredicateQuestion | ChoiceQuestion | ScoreQuestion] = TypeAdapter(Question)
    assert isinstance(
        adapter.validate_python({"id": "q1", "kind": "predicate", "prompt": "p"}), PredicateQuestion
    )
    parsed = adapter.validate_python(
        {"id": "q2", "kind": "choice", "prompt": "p", "options": ["x", "y"]}
    )
    assert isinstance(parsed, ChoiceQuestion)
    parsed = adapter.validate_python(
        {
            "id": "q3",
            "kind": "score",
            "prompt": "p",
            "min_value": 0,
            "max_value": 1,
            "tolerance": 0.1,
        }
    )
    assert isinstance(parsed, ScoreQuestion)
    with pytest.raises(ValidationError):
        adapter.validate_python({"id": "q4", "kind": "ranking", "prompt": "p"})


class TestPredicate:
    def test_answers(self, predicate: PredicateQuestion) -> None:
        assert predicate.validate_answer(True) is True
        assert predicate.is_correct(True, True)
        assert not predicate.is_correct(False, True)

    @pytest.mark.parametrize("bad", ["true", 1, 0.0])
    def test_non_bool_rejected(self, predicate: PredicateQuestion, bad: object) -> None:
        with pytest.raises(ValueError, match="true or false"):
            predicate.validate_answer(bad)  # type: ignore[arg-type]


class TestChoice:
    def test_answers(self, choice: ChoiceQuestion) -> None:
        assert choice.validate_answer("b") == "b"
        assert choice.is_correct("c", "c")
        assert not choice.is_correct("a", "c")

    def test_unknown_option_rejected(self, choice: ChoiceQuestion) -> None:
        with pytest.raises(ValueError, match="one of"):
            choice.validate_answer("z")

    def test_outcome_validated_too(self, choice: ChoiceQuestion) -> None:
        with pytest.raises(ValueError, match="one of"):
            choice.is_correct("a", "nope")

    @pytest.mark.parametrize(
        ("options", "message"),
        [(("a",), "at least 2"), (("a", "a"), "unique"), (("a", " "), "non-empty")],
    )
    def test_bad_options(self, options: tuple[str, ...], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            ChoiceQuestion(id="q1", prompt="p", options=options)


class TestScore:
    def test_tolerance_is_inclusive(self, score: ScoreQuestion) -> None:
        assert score.is_correct(50, 55)
        assert score.is_correct(50.0, 45.0)
        assert not score.is_correct(50, 55.001)

    @pytest.mark.parametrize("bad", [True, "5", math.nan, math.inf, -1, 101])
    def test_bad_answers(self, score: ScoreQuestion, bad: object) -> None:
        with pytest.raises(ValueError, match="score"):
            score.validate_answer(bad)  # type: ignore[arg-type]

    def test_int_answers_become_float(self, score: ScoreQuestion) -> None:
        assert isinstance(score.validate_answer(7), float)

    @pytest.mark.parametrize(
        ("lo", "hi", "tol", "message"),
        [
            (5, 5, 1, "below"),
            (10, 0, 1, "below"),
            (0, 10, 10, "smaller than"),
            (0, math.inf, 1, "finite"),
        ],
    )
    def test_bad_ranges(self, lo: float, hi: float, tol: float, message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            ScoreQuestion(id="q1", prompt="p", min_value=lo, max_value=hi, tolerance=tol)


class TestCatalog:
    def test_get_latest_and_specific_version(self) -> None:
        v1 = PredicateQuestion(id="q1", version=1, prompt="old")
        v2 = PredicateQuestion(id="q1", version=2, prompt="new")
        catalog = QuestionCatalog(questions=(v1, v2))
        assert catalog.get("q1").prompt == "new"
        assert catalog.get("q1", 1).prompt == "old"

    def test_missing(self) -> None:
        catalog = QuestionCatalog()
        with pytest.raises(KeyError, match="unknown question 'q1'"):
            catalog.get("q1")
        with pytest.raises(KeyError, match="version 2"):
            catalog.get("q1", 2)

    def test_duplicate_keys_rejected(self) -> None:
        q = PredicateQuestion(id="q1", prompt="p")
        with pytest.raises(ValidationError, match="duplicate question q1@v1"):
            QuestionCatalog(questions=(q, q))
