from __future__ import annotations

from pathlib import Path

import pytest

from decision_gate.catalog import load_catalog
from decision_gate.questions import (
    ChoiceQuestion,
    PredicateQuestion,
    QuestionCatalog,
    ScoreQuestion,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def example_catalog() -> QuestionCatalog:
    return load_catalog(EXAMPLES / "questions.yaml")


@pytest.fixture
def predicate() -> PredicateQuestion:
    return PredicateQuestion(id="refund.eligible", prompt="Eligible?")


@pytest.fixture
def choice() -> ChoiceQuestion:
    return ChoiceQuestion(id="ticket.queue", prompt="Which queue?", options=("a", "b", "c", "d"))


@pytest.fixture
def score() -> ScoreQuestion:
    return ScoreQuestion(
        id="ticket.minutes", prompt="Minutes?", min_value=0, max_value=100, tolerance=5
    )
