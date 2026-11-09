"""Prometheus metrics and OpenTelemetry tracing for the service.

Metrics live in a :class:`Telemetry` object with its own registry, so every app (and every
test) has an isolated set and ``/metrics`` exposes exactly what this process measured.

Tracing uses only the OpenTelemetry *API*, which is a no-op until a tracer provider is
installed. :func:`configure_tracing` installs the SDK with an OTLP/HTTP exporter when
``DGATE_OTEL_ENABLED`` is set; it needs the ``otel`` extra. The exporter reads the
standard ``OTEL_EXPORTER_OTLP_*`` variables (endpoint, headers), so nothing
project-specific is needed to point it at a collector.

Label values are kept low-cardinality on purpose: question ids come from the catalog,
routes are the matched path *templates* (``/v1/decisions/{decision_id}``), never raw
paths, and an unmatched path is reported as ``unmatched``.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from fastapi import Request, Response
from opentelemetry import trace
from opentelemetry.propagate import extract
from opentelemetry.trace import SpanKind, Status, StatusCode
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from decision_gate import __version__
from decision_gate.resilience import CircuitState

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter

    from decision_gate.config import Settings

BackendResult = Literal["ok", "error", "timeout", "rejected", "cancelled"]

# Decisions are dominated by the backend call: tens of milliseconds for a local model to
# tens of seconds for a slow LLM, so the buckets reach further than the defaults.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

TRACER_NAME = "decision_gate"


class Telemetry:
    """The service's metrics and its tracer."""

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        *,
        tracer_provider: trace.TracerProvider | None = None,
    ) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self.tracer = trace.get_tracer(TRACER_NAME, __version__, tracer_provider=tracer_provider)
        r = self.registry
        Info("dgate_build", "Build information.", registry=r).info({"version": __version__})
        self.http_requests = Counter(
            "dgate_http_requests",
            "HTTP requests by route template and status code.",
            ["method", "route", "status"],
            registry=r,
        )
        self.http_duration = Histogram(
            "dgate_http_request_duration_seconds",
            "HTTP request duration by route template.",
            ["method", "route"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.decisions = Counter(
            "dgate_decisions",
            "Decisions by question and tier; replay=true for idempotent replays.",
            ["question", "tier", "replay"],
            registry=r,
        )
        self.backend_calls = Counter(
            "dgate_backend_calls",
            "Backend calls by result: ok, error, timeout, cancelled, or rejected by the breaker.",
            ["backend", "result"],
            registry=r,
        )
        self.backend_latency = Histogram(
            "dgate_backend_latency_seconds",
            "Backend call duration, for calls that were made (not rejected).",
            ["backend"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.circuit_state = Gauge(
            "dgate_circuit_state",
            "Circuit breaker state per backend: 0 closed, 1 half-open, 2 open.",
            ["backend"],
            registry=r,
        )
        self.circuit_transitions = Counter(
            "dgate_circuit_transitions",
            "Circuit breaker state changes.",
            ["backend", "to"],
            registry=r,
        )
        self.rate_limited = Counter(
            "dgate_rate_limited",
            "Requests rejected with 429 by the rate limiter.",
            ["route"],
            registry=r,
        )
        self.outcomes = Counter(
            "dgate_outcomes_recorded",
            "Outcomes recorded over HTTP by source; created=false for duplicates.",
            ["source", "created"],
            registry=r,
        )

    # Recording helpers, so callers do not depend on label names.

    def record_backend_call(self, backend: str, result: BackendResult, seconds: float) -> None:
        self.backend_calls.labels(backend, result).inc()
        if result not in ("rejected", "cancelled"):
            self.backend_latency.labels(backend).observe(seconds)

    def record_decision(self, question: str, tier: str | None, *, replay: bool) -> None:
        self.decisions.labels(question, tier or "none", "true" if replay else "false").inc()

    def record_outcome(self, source: str, *, created: bool) -> None:
        self.outcomes.labels(source, "true" if created else "false").inc()

    def circuit_listener(self, backend: str) -> Callable[[CircuitState, CircuitState], None]:
        """A breaker ``on_state_change`` callback that keeps the gauge and counter current."""
        self.circuit_state.labels(backend).set(CircuitState.CLOSED.gauge_value)

        def listen(_: CircuitState, new: CircuitState) -> None:
            self.circuit_state.labels(backend).set(new.gauge_value)
            self.circuit_transitions.labels(backend, new.value).inc()

        return listen

    def exposition(self) -> Response:
        return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)

    async def http_middleware(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Count and time every request, inside a server span that continues the caller's
        trace (W3C ``traceparent``) when there is one."""
        started = time.perf_counter()
        method = request.method
        context = extract(dict(request.headers))
        with self.tracer.start_as_current_span(
            f"{method} {request.url.path}",
            context=context,
            kind=SpanKind.SERVER,
            attributes={"http.request.method": method, "url.path": request.url.path},
        ) as span:
            status_code = 500
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            finally:
                route = _route_template(request)
                if route != "unmatched":
                    span.update_name(f"{method} {route}")
                    span.set_attribute("http.route", route)
                span.set_attribute("http.response.status_code", status_code)
                if status_code >= 500:
                    span.set_status(Status(StatusCode.ERROR))
                self.http_requests.labels(method, route, str(status_code)).inc()
                self.http_duration.labels(method, route).observe(time.perf_counter() - started)


def _route_template(request: Request) -> str:
    route: Any = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


def build_tracer_provider(
    settings: Settings, exporter: SpanExporter | None = None
) -> SDKTracerProvider:
    """An SDK tracer provider for this service, exporting with OTLP/HTTP by default.

    Raises :class:`RuntimeError` with an installation hint if the ``otel`` extra is
    missing.
    """
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise RuntimeError(
            "DGATE_OTEL_ENABLED needs the OpenTelemetry SDK: pip install 'decision-gate[otel]'"
        ) from exc
    if exporter is None:  # pragma: no cover - needs a collector; tests pass an exporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": settings.otel_service_name, "service.version": __version__}
        )
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def configure_tracing(settings: Settings) -> SDKTracerProvider | None:
    """Install the global tracer provider when tracing is enabled; returns it (or None)."""
    if not settings.otel_enabled:
        return None
    provider = build_tracer_provider(settings)  # pragma: no cover - needs a collector
    trace.set_tracer_provider(provider)  # pragma: no cover
    return provider  # pragma: no cover
