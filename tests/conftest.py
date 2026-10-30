from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from decision_gate.catalog import load_catalog
from decision_gate.db import migrate
from decision_gate.questions import (
    ChoiceQuestion,
    PredicateQuestion,
    QuestionCatalog,
    ScoreQuestion,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# Integration tests against real PostgreSQL run when this is set (CI sets it); otherwise
# the same tests run against SQLite only.
POSTGRES_URL = os.environ.get("DGATE_TEST_DATABASE_URL")


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    """A migrated, empty SQLite database file."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'dgate.db'}"
    migrate.upgrade(url)
    return url


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def database_url(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[str]:
    """A migrated, empty database: SQLite always, PostgreSQL when configured."""
    if request.param == "sqlite":
        url = f"sqlite+aiosqlite:///{tmp_path / 'dgate.db'}"
        migrate.upgrade(url)
        yield url
        return
    if not POSTGRES_URL:
        pytest.skip("set DGATE_TEST_DATABASE_URL to run PostgreSQL integration tests")
    migrate.downgrade(POSTGRES_URL, "base")
    migrate.upgrade(POSTGRES_URL)
    yield POSTGRES_URL


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
