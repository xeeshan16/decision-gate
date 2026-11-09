"""Metrics, tracing, the circuit breaker and rate limits, through the HTTP API."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from prometheus_client.parser import text_string_to_metric_families

from decision_gate.app import create_app
from decision_gate.backends import BackendError, FakeBackend, RawDecision
from decision_gate.backends.base import AnyQuestion, DecisionInput
from decision_gate.config import Settings
from decision_gate.db.repository import DecisionRepository
from decision_gate.questions import QuestionCatalog
from decision_gate.service import DecisionService
from decision_gate.telemetry import Telemetry, build_tracer_provider, configure_tracing

ORDER = {"order_age_days": 12, "item": "headphones"}
DECIDE = {"question": "refund.eligible", "input": ORDER}


class StepClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class SwitchBackend(FakeBackend):
    """The fake backend, until it is told to fail (or hang)."""

    def __init__(self) -> None:
        super().__init__()
        self.mode = "ok"
        self.calls = 0

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        self.calls += 1
        if self.mode == "fail":
            raise BackendError("fake: upstream returned 500")
        if self.mode == "hang":
            await asyncio.sleep(5)
        return await super().decide(question, data)


def sample(client: TestClient, name: str, **labels: str) -> float:
    """The value of one sample on /metrics (0 if absent)."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for family in text_string_to_metric_families(response.text):
        for s in family.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                return float(s.value)
    return 0.0


def make_client(sqlite_url: str, catalog: QuestionCatalog, **kwargs: Any) -> Iterator[TestClient]:
    settings_keys = {
        "backend_timeout_s",
        "circuit_breaker_enabled",
        "circuit_failure_threshold",
        "circuit_reset_s",
        "rate_limit_per_s",
        "rate_limit_burst",
        "rate_limit_key_header",
        "metrics_enabled",
    }
    settings = Settings(
        database_url=sqlite_url, **{k: kwargs.pop(k) for k in settings_keys & kwargs.keys()}
    )
    with TestClient(create_app(settings, catalog=catalog, **kwargs)) as client:
        yield client


@pytest.fixture
def clock() -> StepClock:
    return StepClock()


@pytest.fixture
def backend() -> SwitchBackend:
    return SwitchBackend()


@pytest.fixture
def client(
    sqlite_url: str, example_catalog: QuestionCatalog, backend: SwitchBackend, clock: StepClock
) -> Iterator[TestClient]:
    yield from make_client(
        sqlite_url,
        example_catalog,
        backend=backend,
        monotonic=clock,
        circuit_failure_threshold=2,
        circuit_reset_s=30,
        backend_timeout_s=0.2,
    )


def test_metrics_count_requests_decisions_and_backend_calls(client: TestClient) -> None:
    assert client.post("/v1/decide", json=DECIDE).status_code == 201
    replay = client.post("/v1/decide", json=DECIDE, headers={"Idempotency-Key": "k1"})
    assert client.post("/v1/decide", json=DECIDE, headers={"Idempotency-Key": "k1"}).json() == (
        replay.json()
    )
    decision_id = replay.json()["id"]
    assert client.get(f"/v1/decisions/{decision_id}").status_code == 200
    assert client.get("/no/such/path").status_code == 404

    # No routing policy, so everything goes to a human.
    assert (
        sample(
            client,
            "dgate_decisions_total",
            question="refund.eligible",
            tier="human",
            replay="false",
        )
        == 2
    )
    assert sample(client, "dgate_decisions_total", tier="human", replay="true") == 1
    assert sample(client, "dgate_backend_calls_total", backend="fake", result="ok") == 2
    assert sample(client, "dgate_backend_latency_seconds_count", backend="fake") == 2
    # Routes are templates, never raw ids; unknown paths collapse into one label.
    assert (
        sample(client, "dgate_http_requests_total", method="POST", route="/v1/decide", status="201")
        == 2
    )
    assert (
        sample(client, "dgate_http_requests_total", method="POST", route="/v1/decide", status="200")
        == 1
    )
    assert (
        sample(
            client,
            "dgate_http_requests_total",
            method="GET",
            route="/v1/decisions/{decision_id}",
            status="200",
        )
        == 1
    )
    assert sample(client, "dgate_http_requests_total", route="unmatched", status="404") == 1
    assert (
        sample(
            client, "dgate_http_request_duration_seconds_count", method="POST", route="/v1/decide"
        )
        == 3
    )
    assert sample(client, "dgate_build_info") == 1
    assert sample(client, "dgate_circuit_state", backend="fake") == 0


def test_outcomes_are_counted_by_source(client: TestClient) -> None:
    decision_id = client.post("/v1/decide", json=DECIDE).json()["id"]
    outcome = {"decision_id": decision_id, "outcome": True}
    assert client.post("/v1/outcomes", json=outcome).status_code == 201
    assert client.post("/v1/outcomes", json=outcome).status_code == 200
    other = client.post("/v1/decide", json={**DECIDE, "input": {"order_age_days": 40}}).json()
    batch = {"outcomes": [{"decision_id": other["id"], "outcome": False, "source": "crm"}]}
    assert client.post("/v1/outcomes/batch", json=batch).json()["counts"]["created"] == 1
    assert sample(client, "dgate_outcomes_recorded_total", source="api", created="true") == 1
    assert sample(client, "dgate_outcomes_recorded_total", source="api", created="false") == 1
    assert sample(client, "dgate_outcomes_recorded_total", source="crm", created="true") == 1


def test_circuit_opens_answers_503_and_recovers_after_a_probe(
    client: TestClient, backend: SwitchBackend, clock: StepClock
) -> None:
    stored = client.post("/v1/decide", json=DECIDE, headers={"Idempotency-Key": "before"})
    assert stored.status_code == 201
    backend.mode = "fail"
    for _ in range(2):
        response = client.post("/v1/decide", json=DECIDE)
        assert response.status_code == 502
        assert response.json() == {
            "detail": "fake: upstream returned 500",
            "fallback_tier": "human",
        }
    assert client.get("/readyz").json()["circuit"] == "open"
    calls = backend.calls

    clock.now += 10
    rejected = client.post("/v1/decide", json=DECIDE)
    assert rejected.status_code == 503
    assert rejected.headers["retry-after"] == "20"
    assert rejected.json()["fallback_tier"] == "human"
    assert "circuit open after 2 consecutive failures" in rejected.json()["detail"]
    assert backend.calls == calls  # the failing backend is left alone

    # A replay of a stored decision never reaches the backend, so it still works.
    replay = client.post("/v1/decide", json=DECIDE, headers={"Idempotency-Key": "before"})
    assert replay.status_code == 200
    assert replay.json() == stored.json()

    assert sample(client, "dgate_circuit_state", backend="fake") == 2
    assert sample(client, "dgate_backend_calls_total", result="error") == 2
    assert sample(client, "dgate_backend_calls_total", result="rejected") == 1
    assert sample(client, "dgate_backend_latency_seconds_count") == 3  # 1 ok + 2 errors

    clock.now += 20
    assert client.get("/readyz").json()["circuit"] == "half_open"
    backend.mode = "ok"
    assert client.post("/v1/decide", json=DECIDE).status_code == 201
    assert client.get("/readyz").json()["circuit"] == "closed"
    assert sample(client, "dgate_circuit_state", backend="fake") == 0
    assert sample(client, "dgate_circuit_transitions_total", to="open") == 1
    assert sample(client, "dgate_circuit_transitions_total", to="half_open") == 1
    assert sample(client, "dgate_circuit_transitions_total", to="closed") == 1


def test_timeouts_count_against_the_circuit(client: TestClient, backend: SwitchBackend) -> None:
    backend.mode = "hang"
    for _ in range(2):
        response = client.post("/v1/decide", json=DECIDE)
        assert response.status_code == 504
        assert response.json()["fallback_tier"] == "human"
    assert client.post("/v1/decide", json=DECIDE).status_code == 503
    assert sample(client, "dgate_backend_calls_total", result="timeout") == 2


def test_breaker_can_be_disabled(
    sqlite_url: str, example_catalog: QuestionCatalog, backend: SwitchBackend
) -> None:
    backend.mode = "fail"
    for client in make_client(
        sqlite_url, example_catalog, backend=backend, circuit_breaker_enabled=False
    ):
        assert client.get("/readyz").json()["circuit"] == "disabled"
        statuses = {client.post("/v1/decide", json=DECIDE).status_code for _ in range(8)}
        assert statuses == {502}
        assert backend.calls == 8


def test_rate_limit_per_caller(
    sqlite_url: str, example_catalog: QuestionCatalog, clock: StepClock
) -> None:
    for client in make_client(
        sqlite_url,
        example_catalog,
        monotonic=clock,
        rate_limit_per_s=0.5,
        rate_limit_burst=2,
        rate_limit_key_header="X-Client-Id",
    ):
        a = {"X-Client-Id": "team-a"}
        assert [
            client.post("/v1/decide", json=DECIDE, headers=a).status_code for _ in range(2)
        ] == [201, 201]
        limited = client.post("/v1/decide", json=DECIDE, headers=a)
        assert limited.status_code == 429
        assert limited.headers["retry-after"] == "2"
        assert limited.json() == {"detail": "rate limit exceeded", "retry_after_s": 2.0}
        # Another caller has its own bucket; without the header, the address is the key.
        assert (
            client.post("/v1/decide", json=DECIDE, headers={"X-Client-Id": "team-b"}).status_code
            == 201
        )
        assert client.post("/v1/decide", json=DECIDE).status_code == 201
        # Reads are not limited.
        assert client.get("/v1/questions", headers=a).status_code == 200
        clock.now += 2
        assert client.post("/v1/decide", json=DECIDE, headers=a).status_code == 201
        assert sample(client, "dgate_rate_limited_total", route="/v1/decide") == 1
        assert sample(client, "dgate_http_requests_total", status="429") == 1


def test_rate_limit_by_client_address(
    sqlite_url: str, example_catalog: QuestionCatalog, clock: StepClock
) -> None:
    for client in make_client(
        sqlite_url, example_catalog, monotonic=clock, rate_limit_per_s=1, rate_limit_burst=1
    ):
        assert client.post("/v1/decide", json=DECIDE).status_code == 201
        # The header is not configured as the key, so it does not buy a new bucket.
        other = client.post("/v1/decide", json=DECIDE, headers={"X-Client-Id": "other"})
        assert other.status_code == 429


async def test_a_cancelled_backend_call_is_counted_as_cancelled(
    example_catalog: QuestionCatalog, backend: SwitchBackend
) -> None:
    backend.mode = "hang"
    telemetry = Telemetry()
    repository: Any = object()  # never reached: the call is cancelled first
    service = DecisionService(
        backend, DecisionRepository(repository), timeout_s=10, telemetry=telemetry
    )
    task = asyncio.create_task(service.decide(example_catalog.get("refund.eligible"), ORDER))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    calls = telemetry.registry.get_sample_value(
        "dgate_backend_calls_total", {"backend": "fake", "result": "cancelled"}
    )
    assert calls == 1
    assert telemetry.registry.get_sample_value(
        "dgate_backend_latency_seconds_count", {"backend": "fake"}
    ) in (None, 0)


def test_metrics_endpoint_can_be_disabled(
    sqlite_url: str, example_catalog: QuestionCatalog
) -> None:
    for client in make_client(sqlite_url, example_catalog, metrics_enabled=False):
        assert client.get("/metrics").status_code == 404


def test_decide_is_traced_and_continues_the_callers_trace(
    sqlite_url: str, example_catalog: QuestionCatalog
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry = Telemetry(tracer_provider=provider)
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    for client in make_client(sqlite_url, example_catalog, telemetry=telemetry):
        response = client.post(
            "/v1/decide",
            json=DECIDE,
            headers={"traceparent": f"00-{trace_id}-00f067aa0ba902b7-01"},
        )
        assert response.status_code == 201
        assert client.get("/no/such/path").status_code == 404
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert {
        "POST /v1/decide",
        "dgate.decide",
        "dgate.backend.decide",
        "dgate.route",
        "dgate.store",
        "GET /no/such/path",
    } <= spans.keys()
    server = spans["POST /v1/decide"]
    assert server.kind is SpanKind.SERVER
    assert server.attributes is not None
    assert server.attributes["http.route"] == "/v1/decide"
    assert server.attributes["http.response.status_code"] == 201
    decide = spans["dgate.decide"]
    assert decide.attributes is not None
    assert decide.attributes["dgate.question.id"] == "refund.eligible"
    assert decide.attributes["dgate.tier"] == "human"
    assert decide.attributes["dgate.decision.id"] == response.json()["id"]
    assert decide.attributes["dgate.replay"] is False
    assert spans["dgate.backend.decide"].attributes == {"dgate.backend": "fake"}
    # One trace, continued from the caller's traceparent, with the steps nested.
    for name in ("POST /v1/decide", "dgate.decide", "dgate.backend.decide", "dgate.store"):
        assert format(spans[name].context.trace_id, "032x") == trace_id
    assert decide.parent is not None
    assert decide.parent.span_id == server.context.span_id
    store_parent = spans["dgate.store"].parent
    assert store_parent is not None
    assert store_parent.span_id == decide.context.span_id


def test_server_errors_mark_the_span(
    sqlite_url: str, example_catalog: QuestionCatalog, backend: SwitchBackend
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    backend.mode = "fail"
    for client in make_client(
        sqlite_url, example_catalog, backend=backend, telemetry=Telemetry(tracer_provider=provider)
    ):
        assert client.post("/v1/decide", json=DECIDE).status_code == 502
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert not spans["POST /v1/decide"].status.is_ok
    assert not spans["dgate.backend.decide"].status.is_ok  # the exception is recorded


def test_build_tracer_provider_names_the_service() -> None:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider(Settings(otel_service_name="gate-eu"), exporter)
    with provider.get_tracer("t").start_as_current_span("x"):
        pass
    provider.force_flush()
    (span,) = exporter.get_finished_spans()
    assert span.resource.attributes["service.name"] == "gate-eu"
    provider.shutdown()


def test_tracing_is_off_unless_enabled() -> None:
    assert configure_tracing(Settings()) is None


def test_blank_rate_limit_settings_mean_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DGATE_RATE_LIMIT_PER_S", "")
    monkeypatch.setenv("DGATE_RATE_LIMIT_KEY_HEADER", "")
    settings = Settings()
    assert settings.rate_limit_per_s is None
    assert settings.rate_limit_key_header is None
