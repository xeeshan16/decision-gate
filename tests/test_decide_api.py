"""The decide flow end to end, against SQLite and (when configured) real PostgreSQL."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from decision_gate.app import create_app
from decision_gate.backends import BackendError, FakeBackend, RawDecision
from decision_gate.backends.base import AnyQuestion, DecisionInput
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import DecisionRecord
from decision_gate.db.repository import DecisionRepository
from decision_gate.questions import ChoiceQuestion, PredicateQuestion, QuestionCatalog
from decision_gate.service import DecisionService, IdempotencyConflictError, input_sha256

ORDER = {"order_age_days": 12, "item": "headphones"}


class FailingBackend:
    name = "failing"

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        raise BackendError("failing: upstream returned 500")


class SlowBackend:
    name = "slow"

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        await asyncio.sleep(5)
        raise AssertionError("unreachable")  # pragma: no cover


class CountingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        self.calls += 1
        await asyncio.sleep(0.05)  # widen the race window for concurrent retries
        return await super().decide(question, data)


def make_client(database_url: str, catalog: QuestionCatalog, **kwargs: Any) -> Iterator[TestClient]:
    settings_kwargs = {
        k: kwargs.pop(k) for k in ("store_inputs", "backend_timeout_s") if k in kwargs
    }
    settings = Settings(database_url=database_url, **settings_kwargs)
    with TestClient(create_app(settings, catalog=catalog, **kwargs)) as client:
        yield client


@pytest.fixture
def client(database_url: str, example_catalog: QuestionCatalog) -> Iterator[TestClient]:
    yield from make_client(database_url, example_catalog)


def count_rows(database_url: str) -> int:
    async def run() -> int:
        engine = make_engine(database_url)
        try:
            async with engine.connect() as connection:
                result = await connection.execute(select(func.count()).select_from(DecisionRecord))
                return int(result.scalar_one())
        finally:
            await engine.dispose()

    return asyncio.run(run())


def test_decide_records_and_returns_decision(client: TestClient, database_url: str) -> None:
    response = client.post("/v1/decide", json={"question": "refund.eligible", "input": ORDER})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["question"] == "refund.eligible"
    assert body["question_version"] == 1
    assert isinstance(body["answer"], bool)
    assert 0 < body["confidence"] < 1
    assert set(body["probabilities"]) == {"true", "false"}
    assert body["backend"] == "fake"
    assert body["latency_ms"] >= 0
    assert body["created_at"].endswith(("Z", "+00:00"))

    # Same answer as the fake gives directly: the service does not alter raw decisions.
    question = client.app.state.catalog.get("refund.eligible")  # type: ignore[attr-defined]
    direct = FakeBackend(overconfidence=2.0).decide_sync(question, ORDER)
    assert body["answer"] == direct.answer
    assert body["confidence"] == pytest.approx(direct.confidence)

    fetched = client.get(f"/v1/decisions/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == body
    assert count_rows(database_url) == 1


@pytest.mark.parametrize(
    ("question", "data", "check"),
    [
        ("ticket.queue", {"subject": "card declined"}, lambda a: isinstance(a, str)),
        ("ticket.handling_minutes", {"subject": "reset"}, lambda a: 0 <= a <= 240),
    ],
)
def test_decide_other_question_kinds(
    client: TestClient, question: str, data: dict[str, Any], check: Any
) -> None:
    body = client.post("/v1/decide", json={"question": question, "input": data}).json()
    assert check(body["answer"])
    if question == "ticket.handling_minutes":
        assert body["probabilities"] is None


def test_idempotent_replay(client: TestClient, database_url: str) -> None:
    request = {"question": "ticket.queue", "input": {"subject": "x"}, "idempotency_key": "k-1"}
    first = client.post("/v1/decide", json=request)
    second = client.post("/v1/decide", json=request)
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json() == first.json()
    assert count_rows(database_url) == 1


def test_idempotency_key_header(client: TestClient) -> None:
    request = {"question": "ticket.queue", "input": {"subject": "y"}}
    first = client.post("/v1/decide", json=request, headers={"Idempotency-Key": "h-1"})
    second = client.post("/v1/decide", json=request, headers={"Idempotency-Key": "h-1"})
    assert first.json()["idempotency_key"] == "h-1"
    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["id"] == first.json()["id"]


def test_idempotency_key_reuse_for_other_input_conflicts(client: TestClient) -> None:
    base = {"question": "ticket.queue", "idempotency_key": "k-2"}
    assert client.post("/v1/decide", json={**base, "input": {"a": 1}}).status_code == 201
    response = client.post("/v1/decide", json={**base, "input": {"a": 2}})
    assert response.status_code == 409
    other_question = {**base, "question": "refund.eligible", "input": {"a": 1}}
    assert client.post("/v1/decide", json=other_question).status_code == 409


def test_decide_specific_version(database_url: str) -> None:
    catalog = QuestionCatalog(
        questions=(
            PredicateQuestion(id="qq", version=1, prompt="old"),
            ChoiceQuestion(id="qq", version=2, prompt="new", options=("x", "y")),
        )
    )
    for client in make_client(database_url, catalog):
        latest = client.post("/v1/decide", json={"question": "qq", "input": {}}).json()
        pinned = client.post(
            "/v1/decide", json={"question": "qq", "version": 1, "input": {}}
        ).json()
        assert latest["question_version"] == 2
        assert latest["answer"] in {"x", "y"}
        assert pinned["question_version"] == 1
        assert isinstance(pinned["answer"], bool)


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"question": "nope", "input": {}}, 404),
        ({"question": "ticket.queue", "version": 7, "input": {}}, 404),
        ({"question": "ticket.queue", "input": [1, 2]}, 422),
        ({"question": "ticket.queue"}, 422),
        ({"question": "ticket.queue", "input": {}, "extra": 1}, 422),
        ({"question": "ticket.queue", "input": {}, "idempotency_key": ""}, 422),
    ],
)
def test_decide_rejects_bad_requests(
    client: TestClient, payload: dict[str, Any], status: int
) -> None:
    assert client.post("/v1/decide", json=payload).status_code == status


def test_unknown_decision_is_404(client: TestClient) -> None:
    assert client.get(f"/v1/decisions/{uuid.uuid4()}").status_code == 404
    assert client.get("/v1/decisions/not-a-uuid").status_code == 422


def test_backend_failure_is_502_and_not_recorded(
    database_url: str, example_catalog: QuestionCatalog
) -> None:
    for client in make_client(database_url, example_catalog, backend=FailingBackend()):
        response = client.post("/v1/decide", json={"question": "ticket.queue", "input": {}})
        assert response.status_code == 502
        assert "upstream returned 500" in response.json()["detail"]
    assert count_rows(database_url) == 0


def test_backend_timeout_is_504(database_url: str, example_catalog: QuestionCatalog) -> None:
    for client in make_client(
        database_url, example_catalog, backend=SlowBackend(), backend_timeout_s=0.05
    ):
        response = client.post("/v1/decide", json={"question": "ticket.queue", "input": {}})
        assert response.status_code == 504
        assert "no answer within 0.05s" in response.json()["detail"]


def test_store_inputs_false_keeps_only_the_hash(
    database_url: str, example_catalog: QuestionCatalog
) -> None:
    for client in make_client(database_url, example_catalog, store_inputs=False):
        body = client.post("/v1/decide", json={"question": "refund.eligible", "input": ORDER})
        decision_id = uuid.UUID(body.json()["id"])

    async def load() -> DecisionRecord | None:
        engine = make_engine(database_url)
        try:
            return await DecisionRepository(make_sessionmaker(engine)).get(decision_id)
        finally:
            await engine.dispose()

    record = asyncio.run(load())
    assert record is not None
    assert record.input is None
    assert record.input_sha256 == input_sha256(ORDER)
    assert record.question_key == "refund.eligible@v1"


def test_input_hash_ignores_key_order() -> None:
    assert input_sha256({"a": 1, "b": 2}) == input_sha256({"b": 2, "a": 1})
    assert input_sha256({"a": 1}) != input_sha256({"a": 2})


async def test_concurrent_retries_store_one_decision(
    database_url: str, predicate: PredicateQuestion
) -> None:
    engine = make_engine(database_url)
    backend = CountingBackend()
    service = DecisionService(backend, DecisionRepository(make_sessionmaker(engine)), timeout_s=5.0)
    try:
        results = await asyncio.gather(
            *(service.decide(predicate, ORDER, idempotency_key="race") for _ in range(4))
        )
        ids = {record.id for record, _ in results}
        assert len(ids) == 1
        assert sum(created for _, created in results) == 1
        assert backend.calls == 4  # all raced past the lookup; the constraint decided

        with pytest.raises(IdempotencyConflictError):
            await service.decide(predicate, {"other": True}, idempotency_key="race")
    finally:
        await engine.dispose()


async def test_concurrent_conflicting_retry_is_rejected(
    database_url: str, predicate: PredicateQuestion
) -> None:
    engine = make_engine(database_url)
    service = DecisionService(
        CountingBackend(), DecisionRepository(make_sessionmaker(engine)), timeout_s=5.0
    )
    try:
        results = await asyncio.gather(
            service.decide(predicate, {"n": 1}, idempotency_key="race-2"),
            service.decide(predicate, {"n": 2}, idempotency_key="race-2"),
            return_exceptions=True,
        )
        assert sum(isinstance(r, IdempotencyConflictError) for r in results) == 1
    finally:
        await engine.dispose()


async def test_duplicate_primary_key_without_idempotency_key_raises(database_url: str) -> None:
    from sqlalchemy.exc import IntegrityError

    engine = make_engine(database_url)
    repository = DecisionRepository(make_sessionmaker(engine))

    def record(decision_id: uuid.UUID) -> DecisionRecord:
        return DecisionRecord(
            id=decision_id,
            question_id="q",
            question_version=1,
            input_sha256="0" * 64,
            answer=True,
            confidence=0.5,
            backend="fake",
            model_version="fake-1",
            latency_ms=1.0,
        )

    try:
        decision_id = uuid.uuid4()
        _, created = await repository.add(record(decision_id))
        assert created
        with pytest.raises(IntegrityError):
            await repository.add(record(decision_id))
    finally:
        await engine.dispose()
