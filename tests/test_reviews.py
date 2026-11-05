"""The human review queue: enqueueing, claims and leases, resolution, SLAs, stats, CLI."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_gate.app import create_app
from decision_gate.backends import FakeBackend
from decision_gate.cli import app as cli
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import DecisionRecord, OutcomeRecord, ReviewItemRecord
from decision_gate.db.repository import (
    DecisionRepository,
    ReviewNotFoundError,
    ReviewRepository,
    ReviewStatRow,
)
from decision_gate.questions import QuestionCatalog
from decision_gate.reviews import (
    InvalidResolutionError,
    ReviewService,
    ReviewSla,
    _percentile,
    is_overdue,
    item_state,
    summarize,
)
from tests.conftest import EXAMPLES
from tests.test_refit import _insert, run

runner = CliRunner()
QUESTIONS = str(EXAMPLES / "questions.yaml")


class FakeClock:
    """A clock the test moves by hand; starts at the real time so stored rows line up."""

    def __init__(self) -> None:
        self.now = datetime.now(UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: float) -> None:
        self.now += timedelta(minutes=minutes)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client(
    database_url: str, example_catalog: QuestionCatalog, clock: FakeClock
) -> Iterator[TestClient]:
    # No routing file: every decision goes to a human, so every decision is enqueued.
    app = create_app(
        Settings(database_url=database_url),
        catalog=example_catalog,
        backend=FakeBackend(),
        clock=clock,
    )
    with TestClient(app) as test_client:
        yield test_client


def decide(client: TestClient, question: str = "refund.eligible", item: int = 1) -> dict[str, Any]:
    response = client.post("/v1/decide", json={"question": question, "input": {"item": item}})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def make_decision(
    tier: str | None,
    *,
    question: str = "refund.eligible",
    answer: Any = True,
    created_at: datetime | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        id=uuid.uuid4(),
        question_id=question,
        question_version=1,
        input_sha256="0" * 64,
        input={"item": 1},
        answer=answer,
        confidence=0.7,
        probabilities=None,
        backend="fake",
        model_version="m",
        latency_ms=0.1,
        created_at=created_at or datetime.now(UTC),
        tier=tier,
        calibrated_confidence=0.6,
        routing_reason="test",
    )


def enqueue(database_url: str, decision: DecisionRecord, sla: ReviewSla | None = None) -> None:
    """Store a decision and its review item the way the decide flow does."""

    async def work() -> None:
        engine = make_engine(database_url)
        try:
            repository = DecisionRepository(make_sessionmaker(engine))
            await repository.add(decision, (sla or ReviewSla()).new_item(decision))
        finally:
            await engine.dispose()

    run(work())


def items(client: TestClient, **params: Any) -> list[dict[str, Any]]:
    response = client.get("/v1/reviews", params=params)
    assert response.status_code == 200, response.text
    found: list[dict[str, Any]] = response.json()["items"]
    return found


def claim_next(client: TestClient, reviewer: str, **extra: Any) -> dict[str, Any]:
    response = client.post("/v1/reviews/claim", json={"reviewer": reviewer, **extra})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def resolve(client: TestClient, item_id: str, **payload: Any) -> Any:
    return client.post(f"/v1/reviews/{item_id}/resolve", json=payload)


def as_dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


# --- Enqueueing --------------------------------------------------------------------------


def test_human_decisions_are_enqueued_with_their_deadline(client: TestClient) -> None:
    decision = decide(client)
    assert decision["tier"] == "human"
    [item] = items(client)
    assert item["decision_id"] == decision["id"]
    assert (item["tier"], item["state"], item["claims"], item["overdue"]) == (
        "human",
        "available",
        0,
        False,
    )
    assert as_dt(item["due_at"]) - as_dt(item["created_at"]) == timedelta(minutes=240)
    assert as_dt(item["created_at"]) == as_dt(decision["created_at"])
    assert item["decision"]["input"] == {"item": 1}
    assert item["decision"]["answer"] == decision["answer"]


def test_a_replayed_decision_is_enqueued_once(client: TestClient) -> None:
    payload = {"question": "refund.eligible", "input": {"item": 1}, "idempotency_key": "k"}
    assert client.post("/v1/decide", json=payload).status_code == 201
    assert client.post("/v1/decide", json=payload).status_code == 200
    assert len(items(client)) == 1


def test_only_verify_and_human_decisions_get_items() -> None:
    sla = ReviewSla(verify_minutes=30, human_minutes=120)
    created = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)
    assert sla.new_item(make_decision("auto")) is None
    assert sla.new_item(make_decision(None)) is None
    verify = sla.new_item(make_decision("verify", created_at=created))
    human = sla.new_item(make_decision("human", created_at=created))
    assert verify is not None
    assert human is not None
    assert verify.due_at == created + timedelta(minutes=30)
    assert human.due_at == created + timedelta(minutes=120)
    assert (verify.status, verify.claims) == ("open", 0)


@pytest.mark.parametrize("field", ["verify_minutes", "human_minutes", "claim_minutes"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_sla_minutes_must_be_positive(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ReviewSla(**{field: value})


def test_settings_build_the_sla() -> None:
    settings = Settings(
        review_verify_sla_minutes=10, review_human_sla_minutes=20, review_claim_minutes=5
    )
    assert settings.review_sla == ReviewSla(10, 20, 5)


# --- Claims and leases -------------------------------------------------------------------


def test_claims_go_earliest_deadline_first(database_url: str, client: TestClient) -> None:
    now = datetime.now(UTC)
    human = make_decision("human", created_at=now - timedelta(minutes=100))  # due in 140m
    verify = make_decision("verify", created_at=now)  # due in 60m
    late = make_decision("human", created_at=now - timedelta(minutes=300))  # overdue
    for decision in (human, verify, late):
        enqueue(database_url, decision)

    order = [claim_next(client, f"r{i}")["decision_id"] for i in range(3)]
    assert order == [str(late.id), str(verify.id), str(human.id)]
    response = client.post("/v1/reviews/claim", json={"reviewer": "r9"})
    assert response.status_code == 204
    assert response.content == b""


def test_claim_filters_by_question_and_tier(database_url: str, client: TestClient) -> None:
    enqueue(database_url, make_decision("verify"))
    enqueue(database_url, make_decision("human", question="ticket.queue", answer="billing"))
    assert len(items(client, question="ticket.queue", tier="human")) == 1
    assert items(client, question="ticket.queue", tier="verify") == []
    got = claim_next(client, "ann", tier="human")
    assert got["question"] == "ticket.queue"
    again = client.post("/v1/reviews/claim", json={"reviewer": "ann", "tier": "human"})
    assert again.status_code == 204
    none = client.post("/v1/reviews/claim", json={"reviewer": "ann", "question": "ticket.queue"})
    assert none.status_code == 204
    assert claim_next(client, "bob", question="refund.eligible")["tier"] == "verify"


def test_a_claimed_item_is_held_until_its_lease_runs_out(
    client: TestClient, clock: FakeClock
) -> None:
    decide(client)
    item = claim_next(client, "ann")
    assert (item["state"], item["claimed_by"], item["claims"]) == ("claimed", "ann", 1)
    assert as_dt(item["claim_expires_at"]) == clock.now + timedelta(minutes=15)

    taken = client.post(f"/v1/reviews/{item['id']}/claim", json={"reviewer": "bob"})
    assert taken.status_code == 409
    assert "claimed by ann" in taken.json()["detail"]
    assert items(client, state="available") == []
    assert len(items(client, state="claimed", claimed_by="ann")) == 1

    clock.advance(16)
    [lapsed] = items(client, state="available")
    assert lapsed["state"] == "available"
    stolen = client.post(f"/v1/reviews/{item['id']}/claim", json={"reviewer": "bob"})
    assert stolen.status_code == 200
    assert (stolen.json()["claimed_by"], stolen.json()["claims"]) == ("bob", 2)


def test_renewing_your_own_claim_extends_the_lease(client: TestClient, clock: FakeClock) -> None:
    decide(client)
    item = claim_next(client, "ann", lease_minutes=5)
    clock.advance(4)
    renewed = client.post(
        f"/v1/reviews/{item['id']}/claim", json={"reviewer": "ann", "lease_minutes": 30}
    )
    assert renewed.status_code == 200
    body = renewed.json()
    assert body["claims"] == 1
    assert as_dt(body["claim_expires_at"]) == clock.now + timedelta(minutes=30)


def test_release_returns_the_item_to_the_queue(client: TestClient) -> None:
    decide(client)
    item = claim_next(client, "ann")
    wrong = client.post(f"/v1/reviews/{item['id']}/release", json={"reviewer": "bob"})
    assert wrong.status_code == 409
    assert "not claimed by bob" in wrong.json()["detail"]
    released = client.post(f"/v1/reviews/{item['id']}/release", json={"reviewer": "ann"})
    assert released.status_code == 200
    assert (released.json()["state"], released.json()["claimed_by"]) == ("available", None)
    again = client.post(f"/v1/reviews/{item['id']}/release", json={"reviewer": "ann"})
    assert again.status_code == 409


def test_unknown_items_are_404(client: TestClient) -> None:
    missing = uuid.uuid4()
    assert client.get(f"/v1/reviews/{missing}").status_code == 404
    for action in ("claim", "release"):
        response = client.post(f"/v1/reviews/{missing}/{action}", json={"reviewer": "ann"})
        assert response.status_code == 404
    assert resolve(client, str(missing), reviewer="ann", confirm=True).status_code == 404


def test_requests_are_validated(client: TestClient) -> None:
    bad = [
        {"reviewer": ""},
        {"reviewer": "   "},
        {"reviewer": "ann", "tier": "auto"},
        {"reviewer": "ann", "lease_minutes": 0},
        {"reviewer": "ann", "lease_minutes": 481},
        {"reviewer": "ann", "extra": 1},
    ]
    for payload in bad:
        assert client.post("/v1/reviews/claim", json=payload).status_code == 422, payload
    decide(client)
    [item] = items(client)
    for payload in (
        {"reviewer": "ann"},
        {"reviewer": "ann", "confirm": True, "outcome": True},
        {"reviewer": "ann", "confirm": False},
    ):
        assert resolve(client, item["id"], **payload).status_code == 422, payload


def test_concurrent_claims_never_share_an_item(database_url: str) -> None:
    for _ in range(6):
        enqueue(database_url, make_decision("human"))

    async def work() -> list[uuid.UUID | None]:
        engine = make_engine(database_url)
        try:
            repository = ReviewRepository(make_sessionmaker(engine))
            now = datetime.now(UTC)
            lease = now + timedelta(minutes=15)
            claimed = await asyncio.gather(
                *(repository.claim_next(f"r{i}", now=now, lease_until=lease) for i in range(10))
            )
            return [None if c is None else c.id for c in claimed]
        finally:
            await engine.dispose()

    got = run(work())
    taken = [item for item in got if item is not None]
    assert len(taken) == 6
    assert len(set(taken)) == 6


# --- Resolution --------------------------------------------------------------------------


def test_confirming_a_verify_item_records_the_outcome(
    database_url: str, client: TestClient
) -> None:
    decision = make_decision("verify", answer=True)
    enqueue(database_url, decision)
    item = claim_next(client, "ann")
    response = resolve(client, item["id"], reviewer="ann", confirm=True, note="checked order")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["correct"] is True
    resolved = body["item"]
    assert (resolved["state"], resolved["verdict"], resolved["agreed"]) == (
        "resolved",
        "confirmed",
        True,
    )
    assert (resolved["resolved_by"], resolved["resolution"], resolved["note"]) == (
        "ann",
        True,
        "checked order",
    )
    assert resolved["claim_expires_at"] is None

    stored = client.get(f"/v1/decisions/{decision.id}").json()["outcome"]
    assert (stored["outcome"], stored["correct"], stored["source"]) == (True, True, "review")
    assert client.get("/v1/outcomes/stats").json()["questions"][0]["labeled"] == 1


def test_correcting_an_answer_marks_the_backend_wrong(
    database_url: str, client: TestClient
) -> None:
    enqueue(database_url, make_decision("verify", answer=True))
    enqueue(database_url, make_decision("human", question="ticket.queue", answer="billing"))
    verify = claim_next(client, "ann", tier="verify")
    body = resolve(client, verify["id"], reviewer="ann", outcome=False).json()
    assert (body["correct"], body["item"]["verdict"], body["item"]["agreed"]) == (
        False,
        "corrected",
        False,
    )
    human = claim_next(client, "ann", tier="human")
    body = resolve(client, human["id"], reviewer="ann", outcome="billing").json()
    assert (body["correct"], body["item"]["verdict"]) == (True, "decided")


def test_resolving_is_idempotent_for_the_same_answer(client: TestClient) -> None:
    decision = decide(client)
    item = claim_next(client, "ann")
    first = resolve(client, item["id"], reviewer="ann", outcome=decision["answer"])
    assert first.status_code == 201
    again = resolve(client, item["id"], reviewer="ann", outcome=decision["answer"])
    assert again.status_code == 200
    assert again.json()["item"]["resolved_at"] == first.json()["item"]["resolved_at"]

    changed = resolve(client, item["id"], reviewer="ann", outcome=not decision["answer"])
    assert changed.status_code == 409
    assert "already resolved by ann" in changed.json()["detail"]
    other = resolve(client, item["id"], reviewer="bob", outcome=decision["answer"])
    assert other.status_code == 409
    assert client.post(f"/v1/reviews/{item['id']}/claim", json={"reviewer": "bob"}).status_code == (
        409
    )


def test_someone_elses_live_claim_blocks_resolution(client: TestClient, clock: FakeClock) -> None:
    decide(client)
    item = claim_next(client, "ann")
    blocked = resolve(client, item["id"], reviewer="bob", confirm=True)
    assert blocked.status_code == 409
    assert "claimed by ann" in blocked.json()["detail"]
    clock.advance(20)  # ann's lease lapses; bob did the work, so bob may resolve
    done = resolve(client, item["id"], reviewer="bob", confirm=True)
    assert done.status_code == 201
    assert (done.json()["item"]["claimed_by"], done.json()["item"]["claims"]) == ("bob", 2)


def test_an_unclaimed_item_can_be_resolved_directly(client: TestClient) -> None:
    decide(client)
    [item] = items(client)
    done = resolve(client, item["id"], reviewer="ann", confirm=True)
    assert done.status_code == 201
    body = done.json()["item"]
    assert (body["claimed_by"], body["claims"]) == ("ann", 1)
    assert body["claimed_at"] == body["resolved_at"]


def test_an_invalid_answer_is_rejected(client: TestClient) -> None:
    decide(client, question="ticket.queue")
    [item] = items(client)
    response = resolve(client, item["id"], reviewer="ann", outcome="nonsense")
    assert response.status_code == 422
    assert items(client)[0]["state"] == "available"


def test_an_outcome_from_elsewhere_supersedes_the_item(client: TestClient) -> None:
    decision = decide(client)
    item = claim_next(client, "ann")
    recorded = client.post(
        "/v1/outcomes",
        json={"decision_id": decision["id"], "outcome": True, "source": "chargeback-feed"},
    )
    assert recorded.status_code == 201
    [superseded] = items(client, state="superseded")
    assert superseded["note"] == "outcome recorded by chargeback-feed"
    assert superseded["overdue"] is False
    assert superseded["resolved_at"] is not None
    blocked = resolve(client, item["id"], reviewer="ann", confirm=True)
    assert blocked.status_code == 409
    assert "superseded" in blocked.json()["detail"]
    assert client.post("/v1/reviews/claim", json={"reviewer": "ann"}).status_code == 204
    release = client.post(f"/v1/reviews/{item['id']}/release", json={"reviewer": "ann"})
    assert release.status_code == 409


def test_a_question_dropped_from_the_catalog_cannot_be_resolved(database_url: str) -> None:
    decision = make_decision("human", question="retired.question")
    enqueue(database_url, decision)

    async def work() -> None:
        engine = make_engine(database_url)
        try:
            repository = ReviewRepository(make_sessionmaker(engine))
            service = ReviewService(QuestionCatalog(), repository, ReviewSla())
            item = await repository.for_decision(decision.id)
            assert item is not None
            with pytest.raises(InvalidResolutionError, match="no longer in the catalog"):
                await service.resolve(item.id, "ann", confirm=True)
            with pytest.raises(InvalidResolutionError, match="exactly one"):
                await service.resolve(item.id, "ann")
            with pytest.raises(ReviewNotFoundError):
                await service.resolve(uuid.uuid4(), "ann", confirm=True)
            assert await repository.for_decision(uuid.uuid4()) is None
            outcome = OutcomeRecord(
                decision_id=decision.id,
                outcome=True,
                correct=True,
                source="review",
                observed_at=datetime.now(UTC),
            )
            with pytest.raises(ReviewNotFoundError):
                await repository.resolve(
                    uuid.uuid4(),
                    "ann",
                    now=datetime.now(UTC),
                    outcome=outcome,
                    verdict="decided",
                )
        finally:
            await engine.dispose()

    run(work())


# --- SLAs and statistics -----------------------------------------------------------------


def test_overdue_items_are_flagged_and_filterable(client: TestClient, clock: FakeClock) -> None:
    decide(client, item=1)
    decide(client, item=2)
    first = claim_next(client, "ann")
    resolve(client, first["id"], reviewer="ann", confirm=True)
    clock.advance(241)
    late = items(client, overdue=True)
    assert [i["state"] for i in late] == ["available"]
    on_time = items(client, overdue=False)
    assert [i["state"] for i in on_time] == ["resolved"]

    second = claim_next(client, "bob")
    resolve(client, second["id"], reviewer="bob", confirm=True)
    [resolved_late] = items(client, overdue=True)
    assert (resolved_late["state"], resolved_late["overdue"]) == ("resolved", True)


def test_stats_report_backlog_sla_and_agreement(
    database_url: str, client: TestClient, clock: FakeClock
) -> None:
    now = clock.now
    for i in range(4):
        enqueue(
            database_url,
            make_decision("verify", answer=True, created_at=now - timedelta(microseconds=i)),
        )
    enqueue(database_url, make_decision("human", question="ticket.queue", answer="billing"))
    enqueue(database_url, make_decision("human", question="ticket.queue", answer="billing"))

    clock.advance(10)
    a = claim_next(client, "ann", tier="verify")
    resolve(client, a["id"], reviewer="ann", confirm=True)  # 10 minutes, on time
    clock.advance(20)
    b = claim_next(client, "ann", tier="verify")
    resolve(client, b["id"], reviewer="ann", outcome=False)  # 30 minutes, corrected
    clock.advance(50)
    c = claim_next(client, "ann", tier="verify")
    resolve(client, c["id"], reviewer="ann", confirm=True)  # 80 minutes: late
    claim_next(client, "bob", tier="human")  # held, not resolved

    body = client.get("/v1/reviews/stats").json()
    assert body["sla_minutes"] == {"verify": 60.0, "human": 240.0, "claim": 15.0}
    queues = {(q["question"], q["tier"]): q for q in body["queues"]}
    verify = queues[("refund.eligible", "verify")]
    assert (verify["available"], verify["claimed"], verify["resolved"]) == (1, 0, 3)
    assert verify["overdue"] == 1  # the fourth item has waited 80 minutes
    assert verify["oldest_waiting_minutes"] == pytest.approx(80, abs=0.01)
    assert verify["sla_hit_rate"] == pytest.approx(2 / 3)
    assert verify["median_minutes"] == pytest.approx(30, abs=0.01)
    assert verify["p90_minutes"] == pytest.approx(80, abs=0.01)
    assert (verify["confirmed"], verify["corrected"]) == (2, 1)
    assert verify["agreement_rate"] == pytest.approx(2 / 3)
    low, high = verify["agreement_interval"]
    assert 0 < low < 2 / 3 < high < 1
    human = queues[("ticket.queue", "human")]
    assert (human["available"], human["claimed"], human["overdue"]) == (1, 1, 0)
    assert human["sla_hit_rate"] is None
    assert human["agreement_interval"] is None

    only = client.get("/v1/reviews/stats", params={"question": "ticket.queue"}).json()
    assert [q["question"] for q in only["queues"]] == ["ticket.queue"]
    later = (clock.now + timedelta(days=1)).isoformat()
    assert client.get("/v1/reviews/stats", params={"since": later}).json()["queues"] == []


def test_summaries_from_rows() -> None:
    now = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)

    def row(status: str, created_min: float, **kw: Any) -> ReviewStatRow:
        created = now - timedelta(minutes=created_min)
        return ReviewStatRow(
            question_id="q",
            question_version=1,
            tier="human",
            status=status,
            created_at=created,
            due_at=created + timedelta(minutes=60),
            claim_expires_at=kw.get("expires"),
            resolved_at=kw.get("resolved"),
            verdict=kw.get("verdict"),
            agreed=kw.get("agreed"),
        )

    [stats] = summarize(
        [
            row("open", 90),
            row("claimed", 10, expires=now - timedelta(minutes=1)),  # lapsed: available
            row("claimed", 5, expires=now + timedelta(minutes=5)),
            row("superseded", 30),
            row("resolved", 50, resolved=now, verdict="decided", agreed=True),
        ],
        now,
    )
    assert (stats.available, stats.claimed, stats.superseded, stats.resolved) == (2, 1, 1, 1)
    assert stats.overdue == 1
    assert stats.oldest_waiting_minutes == pytest.approx(90)
    assert stats.sla_hit_rate == 1.0
    assert stats.agreement_rate == 1.0
    assert summarize([], now) == []


def test_percentile_is_nearest_rank() -> None:
    assert _percentile([], 0.9) is None
    assert _percentile([5.0], 0.9) == 5.0
    assert _percentile([float(v) for v in range(1, 11)], 0.9) == 9.0
    assert _percentile([float(v) for v in range(1, 11)], 0.0) == 1.0


def test_item_state_and_overdue_helpers() -> None:
    now = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    item = ReviewItemRecord(
        status="claimed",
        created_at=now,
        due_at=(now + timedelta(minutes=5)).replace(tzinfo=None),
        claim_expires_at=None,
    )
    assert item_state(item, now) == "available"
    assert is_overdue(item, now) is False
    item.status = "resolved"
    item.resolved_at = None
    assert item_state(item, now) == "resolved"
    assert is_overdue(item, now) is False


# --- Backfill ----------------------------------------------------------------------------


def test_backfill_enqueues_decisions_from_before_the_queue(
    database_url: str, client: TestClient
) -> None:
    old = datetime.now(UTC) - timedelta(days=1)
    verify = make_decision("verify", created_at=old)
    human = make_decision("human", created_at=old)
    auto = make_decision("auto", created_at=old)
    legacy = make_decision(None, created_at=old)
    labeled = make_decision("human", created_at=old)
    run(_insert(database_url, [verify, human, auto, legacy, labeled]))
    run(
        _insert(
            database_url,
            [
                OutcomeRecord(
                    decision_id=labeled.id,
                    outcome=True,
                    correct=True,
                    source="test",
                    observed_at=old,
                )
            ],
        )
    )
    assert client.post("/v1/reviews/backfill").json() == {"enqueued": 2}
    assert client.post("/v1/reviews/backfill").json() == {"enqueued": 0}
    found = items(client)
    assert {i["decision_id"] for i in found} == {str(verify.id), str(human.id)}
    assert all(i["overdue"] for i in found)  # their deadlines counted from the decision


def test_backfill_works_in_batches(database_url: str) -> None:
    old = datetime.now(UTC) - timedelta(hours=1)
    run(_insert(database_url, [make_decision("human", created_at=old) for _ in range(5)]))

    async def work() -> int:
        engine = make_engine(database_url)
        try:
            repository = ReviewRepository(make_sessionmaker(engine))
            service = ReviewService(QuestionCatalog(), repository, ReviewSla())
            return await service.backfill(batch=2)
        finally:
            await engine.dispose()

    assert run(work()) == 5


# --- CLI ---------------------------------------------------------------------------------


def invoke(*args: str, database_url: str) -> Any:
    return runner.invoke(cli, ["reviews", *args, "--url", database_url])


def test_cli_queue_workflow(database_url: str) -> None:
    enqueue(database_url, make_decision("verify", answer=True))
    enqueue(database_url, make_decision("human", question="ticket.queue", answer="billing"))
    queue_decision = make_decision("human", question="ticket.queue", answer="technical")
    queue_decision.answer_set = ["technical", "account"]
    queue_decision.input = None  # inputs not stored: only the hash is kept
    enqueue(database_url, queue_decision)

    listed = invoke("list", database_url=database_url)
    assert listed.exit_code == 0, listed.output
    assert listed.output.count("available") == 3

    claimed = invoke("claim", "--reviewer", "ann", "--tier", "verify", database_url=database_url)
    assert claimed.exit_code == 0, claimed.output
    assert "claimed by ann" in claimed.output
    assert 'input: {"item": 1}' in claimed.output
    verify_id = claimed.output.split()[0]

    resolved = runner.invoke(
        cli,
        [
            "reviews",
            "resolve",
            verify_id,
            "--reviewer",
            "ann",
            "--confirm",
            "-q",
            QUESTIONS,
            "--url",
            database_url,
        ],
    )
    assert resolved.exit_code == 0, resolved.output
    assert "[confirmed], backend right" in resolved.output

    target = invoke(
        "claim", "--reviewer", "bob", "--question", "ticket.queue", database_url=database_url
    )
    assert target.exit_code == 0, target.output
    item_id = target.output.split()[0]
    assert "answer set" not in target.output  # the billing decision has no set

    released = invoke("release", item_id, "--reviewer", "bob", database_url=database_url)
    assert released.exit_code == 0, released.output
    assert released.output.startswith("released")

    again = invoke(
        "claim", item_id, "--reviewer", "bob", "--lease", "30", database_url=database_url
    )
    assert again.exit_code == 0, again.output
    bare = runner.invoke(
        cli,
        [
            "reviews",
            "resolve",
            item_id,
            "--reviewer",
            "bob",
            "--outcome",
            "account",  # a bare word is taken as a string answer
            "-q",
            QUESTIONS,
            "--url",
            database_url,
        ],
    )
    assert bare.exit_code == 0, bare.output
    assert '"account" [decided], backend wrong' in bare.output

    last = invoke("claim", "--reviewer", "cy", database_url=database_url)
    assert "answer set: " in last.output
    assert "input:" not in last.output
    empty = invoke("claim", "--reviewer", "cy", database_url=database_url)
    assert empty.output.strip() == "nothing is waiting"

    done = invoke("list", "--state", "resolved", database_url=database_url)
    assert done.output.count("resolved by") == 2
    assert "-> true [confirmed]" in done.output
    late = invoke("list", "--overdue", database_url=database_url)
    assert late.output.strip() == "no review items match"

    table = invoke("stats", database_url=database_url)
    assert table.exit_code == 0, table.output
    assert "SLA: verify 60m, human 4.0h, claims last 15m" in table.output
    assert "refund.eligible@v1 verify" in table.output
    as_json = invoke("stats", "--json", "--question", "ticket.queue", database_url=database_url)
    queues = json.loads(as_json.output)["queues"]
    assert [q["resolved"] for q in queues] == [1]


def test_cli_errors(database_url: str) -> None:
    decision = make_decision("human")
    enqueue(database_url, decision)
    item_id = invoke("list", database_url=database_url).output.split()[0]
    assert invoke("claim", item_id, "--reviewer", "ann", database_url=database_url).exit_code == 0

    taken = invoke("claim", item_id, "--reviewer", "bob", database_url=database_url)
    assert taken.exit_code == 1
    assert "claimed by ann" in taken.output
    bad_id = invoke("release", "not-a-uuid", "--reviewer", "ann", database_url=database_url)
    assert bad_id.exit_code == 2
    missing = invoke("release", str(uuid.uuid4()), "--reviewer", "ann", database_url=database_url)
    assert missing.exit_code == 1
    bad_tier = invoke("list", "--tier", "auto", database_url=database_url)
    assert bad_tier.exit_code == 2
    bad_state = invoke("list", "--state", "done", database_url=database_url)
    assert bad_state.exit_code == 2
    both = runner.invoke(
        cli,
        ["reviews", "resolve", item_id, "--reviewer", "ann", "--confirm", "--outcome", "true"],
    )
    assert both.exit_code == 2
    invalid = runner.invoke(
        cli,
        [
            "reviews",
            "resolve",
            item_id,
            "--reviewer",
            "ann",
            "--outcome",
            '"maybe"',
            "-q",
            QUESTIONS,
            "--url",
            database_url,
        ],
    )
    assert invalid.exit_code == 1
    assert "error:" in invalid.output


def test_cli_backfill_and_empty_stats(database_url: str) -> None:
    empty = invoke("stats", database_url=database_url)
    assert empty.output.strip() == "no review items"
    run(_insert(database_url, [make_decision("verify")]))
    result = invoke("backfill", database_url=database_url)
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "enqueued 1 decision(s)"
