"""Outcomes end to end, against SQLite and (when configured) real PostgreSQL."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from decision_gate.app import create_app
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import OutcomeRecord
from decision_gate.db.repository import DecisionRepository, OutcomeRepository
from decision_gate.questions import ChoiceQuestion, QuestionCatalog
from decision_gate.service import (
    InvalidOutcomeError,
    OutcomeConflictError,
    OutcomeService,
    UnknownDecisionError,
)


@pytest.fixture
def client(database_url: str, example_catalog: QuestionCatalog) -> Iterator[TestClient]:
    settings = Settings(database_url=database_url)
    with TestClient(create_app(settings, catalog=example_catalog)) as client:
        yield client


def decide(client: TestClient, question: str, data: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/v1/decide", json={"question": question, "input": data})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def other_label(answer: str) -> str:
    return next(o for o in ("billing", "technical", "account") if o != answer)


def test_record_outcome_scores_the_decision(client: TestClient) -> None:
    decision = decide(client, "refund.eligible", {"order": 1})
    response = client.post(
        "/v1/outcomes",
        json={"decision_id": decision["id"], "outcome": decision["answer"], "source": "crm"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["correct"] is True
    assert body["source"] == "crm"
    assert body["outcome"] == decision["answer"]
    assert body["observed_at"].endswith(("Z", "+00:00"))

    # The decision now carries its outcome.
    fetched = client.get(f"/v1/decisions/{decision['id']}").json()
    assert fetched["outcome"] == body


def test_wrong_answer_is_scored_incorrect(client: TestClient) -> None:
    decision = decide(client, "ticket.queue", {"subject": "refund"})
    response = client.post(
        "/v1/outcomes",
        json={"decision_id": decision["id"], "outcome": other_label(decision["answer"])},
    )
    assert response.status_code == 201
    assert response.json()["correct"] is False
    assert response.json()["source"] == "api"


def test_score_outcome_uses_tolerance(client: TestClient) -> None:
    decision = decide(client, "ticket.handling_minutes", {"subject": "reset"})
    answer = decision["answer"]
    near = answer + 10 if answer + 10 <= 240 else answer - 10
    response = client.post("/v1/outcomes", json={"decision_id": decision["id"], "outcome": near})
    assert response.json()["correct"] is True

    other = decide(client, "ticket.handling_minutes", {"subject": "other"})
    a2 = other["answer"]
    far = a2 + 100 if a2 + 100 <= 240 else a2 - 100
    response = client.post("/v1/outcomes", json={"decision_id": other["id"], "outcome": far})
    assert response.json()["correct"] is False


def test_replaying_the_same_outcome_is_idempotent(client: TestClient) -> None:
    decision = decide(client, "refund.eligible", {"order": 2})
    payload = {"decision_id": decision["id"], "outcome": True}
    first = client.post("/v1/outcomes", json=payload)
    second = client.post("/v1/outcomes", json={**payload, "source": "kafka"})
    assert (first.status_code, second.status_code) == (201, 200)
    # The stored row is returned unchanged, including its original source.
    assert second.json() == first.json()


def test_a_different_outcome_conflicts(client: TestClient) -> None:
    decision = decide(client, "refund.eligible", {"order": 3})
    client.post("/v1/outcomes", json={"decision_id": decision["id"], "outcome": True})
    response = client.post("/v1/outcomes", json={"decision_id": decision["id"], "outcome": False})
    assert response.status_code == 409
    assert "already has outcome" in response.json()["detail"]


def test_unknown_decision_is_404(client: TestClient) -> None:
    response = client.post("/v1/outcomes", json={"decision_id": str(uuid.uuid4()), "outcome": 1})
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("question", "bad"),
    [("refund.eligible", "yes"), ("ticket.queue", "sales"), ("ticket.handling_minutes", 999)],
)
def test_invalid_outcome_is_422(client: TestClient, question: str, bad: Any) -> None:
    decision = decide(client, question, {"x": 1})
    response = client.post("/v1/outcomes", json={"decision_id": decision["id"], "outcome": bad})
    assert response.status_code == 422
    assert decision["question"] in response.json()["detail"]


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        ("2026-10-30T12:00:00+05:00", datetime(2026, 10, 30, 7, 0, tzinfo=UTC)),
        ("2026-10-30T12:00:00", datetime(2026, 10, 30, 12, 0, tzinfo=UTC)),  # naive = UTC
    ],
)
def test_observed_at_is_normalised_to_utc(
    client: TestClient, sent: str, expected: datetime
) -> None:
    decision = decide(client, "refund.eligible", {"order": 4})
    response = client.post(
        "/v1/outcomes",
        json={"decision_id": decision["id"], "outcome": True, "observed_at": sent},
    )
    assert datetime.fromisoformat(response.json()["observed_at"]) == expected


def test_batch_reports_each_item(client: TestClient) -> None:
    a = decide(client, "refund.eligible", {"order": 10})
    b = decide(client, "refund.eligible", {"order": 11})
    client.post("/v1/outcomes", json={"decision_id": b["id"], "outcome": True})
    missing = str(uuid.uuid4())
    response = client.post(
        "/v1/outcomes/batch",
        json={
            "outcomes": [
                {"decision_id": a["id"], "outcome": a["answer"]},
                {"decision_id": b["id"], "outcome": True},
                {"decision_id": b["id"], "outcome": False},
                {"decision_id": missing, "outcome": True},
                {"decision_id": a["id"], "outcome": "maybe"},
            ]
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [r["status"] for r in body["results"]] == [
        "created",
        "duplicate",
        "conflict",
        "unknown_decision",
        "invalid",
    ]
    assert body["results"][0]["correct"] is True
    assert body["results"][3]["detail"].startswith("unknown decision")
    assert body["counts"] == {
        "created": 1,
        "duplicate": 1,
        "conflict": 1,
        "unknown_decision": 1,
        "invalid": 1,
    }


def test_batch_limits(client: TestClient) -> None:
    assert client.post("/v1/outcomes/batch", json={"outcomes": []}).status_code == 422
    too_many = [{"decision_id": str(uuid.uuid4()), "outcome": True}] * 1001
    assert client.post("/v1/outcomes/batch", json={"outcomes": too_many}).status_code == 422


def test_stats_compare_raw_confidence_with_accuracy(client: TestClient) -> None:
    decisions = [decide(client, "ticket.queue", {"n": i}) for i in range(6)]
    decide(client, "refund.eligible", {"n": 0})  # never labeled
    # Label four: three right, one wrong.
    for d in decisions[:3]:
        client.post("/v1/outcomes", json={"decision_id": d["id"], "outcome": d["answer"]})
    wrong = decisions[3]
    client.post(
        "/v1/outcomes",
        json={"decision_id": wrong["id"], "outcome": other_label(wrong["answer"])},
    )

    body = client.get("/v1/outcomes/stats").json()["questions"]
    by_question = {s["question"]: s for s in body}
    refund, queue = by_question["refund.eligible"], by_question["ticket.queue"]
    assert refund == {
        "question": "refund.eligible",
        "question_version": 1,
        "decisions": 1,
        "labeled": 0,
        "label_rate": 0.0,
        "accuracy": None,
        "mean_confidence": None,
        "confidence_gap": None,
    }
    assert queue["decisions"] == 6
    assert queue["labeled"] == 4
    assert queue["label_rate"] == pytest.approx(4 / 6)
    assert queue["accuracy"] == pytest.approx(0.75)
    mean_conf = sum(d["confidence"] for d in decisions[:4]) / 4
    assert queue["mean_confidence"] == pytest.approx(mean_conf)
    assert queue["confidence_gap"] == pytest.approx(mean_conf - 0.75)

    filtered = client.get("/v1/outcomes/stats", params={"question": "ticket.queue"}).json()
    assert [s["question"] for s in filtered["questions"]] == ["ticket.queue"]


# --- repository and service, below the API -------------------------------------------


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _seed(database_url: str, catalog: QuestionCatalog, n: int) -> list[uuid.UUID]:
    settings = Settings(database_url=database_url)
    app = create_app(settings, catalog=catalog)
    with TestClient(app) as client:
        return [uuid.UUID(decide(client, "ticket.queue", {"n": i})["id"]) for i in range(n)]


def test_labeled_joins_and_windows(database_url: str, example_catalog: QuestionCatalog) -> None:
    ids = run(_seed(database_url, example_catalog, 4))

    async def scenario() -> None:
        engine = make_engine(database_url)
        try:
            sessions = make_sessionmaker(engine)
            decisions, outcomes = DecisionRepository(sessions), OutcomeRepository(sessions)
            service = OutcomeService(example_catalog, decisions, outcomes)
            for decision_id in ids[:3]:
                record = await decisions.get(decision_id)
                assert record is not None
                await service.record(decision_id, record.answer, source="test")

            pairs = await outcomes.labeled("ticket.queue", 1)
            assert [d.id for d, _ in pairs] == ids[:3]
            assert all(o.correct for _, o in pairs)
            assert await outcomes.labeled("ticket.queue", 2) == []
            assert len(await outcomes.labeled("ticket.queue", 1, limit=2)) == 2

            first = pairs[0][0].created_at
            future = datetime.now(UTC) + timedelta(days=1)
            assert await outcomes.labeled("ticket.queue", 1, since=future) == []
            assert len(await outcomes.labeled("ticket.queue", 1, until=future)) == 3
            assert await outcomes.labeled("ticket.queue", 1, until=first) == []
        finally:
            await engine.dispose()

    run(scenario())


def test_service_errors(database_url: str, example_catalog: QuestionCatalog) -> None:
    ids = run(_seed(database_url, example_catalog, 1))

    async def scenario() -> None:
        engine = make_engine(database_url)
        try:
            sessions = make_sessionmaker(engine)
            decisions, outcomes = DecisionRepository(sessions), OutcomeRepository(sessions)
            service = OutcomeService(example_catalog, decisions, outcomes)
            with pytest.raises(UnknownDecisionError):
                await service.record(uuid.uuid4(), "billing", source="t")
            with pytest.raises(InvalidOutcomeError, match="one of"):
                await service.record(ids[0], "sales", source="t")

            # A catalog that no longer has the question version cannot score the outcome.
            renamed = QuestionCatalog(
                questions=(
                    ChoiceQuestion(id="ticket.queue", version=2, prompt="?", options=("a", "b")),
                )
            )
            stale = OutcomeService(renamed, decisions, outcomes)
            with pytest.raises(InvalidOutcomeError, match="no longer in the catalog"):
                await stale.record(ids[0], "a", source="t")

            _, created = await service.record(ids[0], "billing", source="t")
            assert created
            with pytest.raises(OutcomeConflictError):
                await service.record(ids[0], "technical", source="t")

            # The database refuses an orphan outcome, on SQLite too (foreign keys are on).
            orphan = OutcomeRecord(
                decision_id=uuid.uuid4(),
                outcome="billing",
                correct=True,
                source="t",
                observed_at=datetime.now(UTC),
            )
            with pytest.raises(IntegrityError):
                await outcomes.add(orphan)
        finally:
            await engine.dispose()

    run(scenario())


def test_concurrent_duplicate_outcomes_store_one_row(
    database_url: str, example_catalog: QuestionCatalog
) -> None:
    ids = run(_seed(database_url, example_catalog, 1))

    async def scenario() -> list[bool]:
        engine = make_engine(database_url)
        try:
            sessions = make_sessionmaker(engine)
            service = OutcomeService(
                example_catalog, DecisionRepository(sessions), OutcomeRepository(sessions)
            )
            results = await asyncio.gather(
                *(service.record(ids[0], "billing", source=f"r{i}") for i in range(5))
            )
            return [created for _, created in results]
        finally:
            await engine.dispose()

    created = run(scenario())
    assert created.count(True) == 1
    assert created.count(False) == 4
