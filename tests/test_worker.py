"""The maintenance worker: step order, error isolation, the pass lock, the loop and the CLI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import text
from typer.testing import CliRunner

from decision_gate.cli import app as cli
from decision_gate.db import make_engine
from decision_gate.questions import QuestionCatalog
from decision_gate.routing import RoutingConfig, parse_routing
from decision_gate.worker import (
    ADVISORY_LOCK_KEY,
    AdvisoryLock,
    MaintenanceWorker,
    NoLock,
    PassReport,
    StepResult,
    build_worker,
    pass_lock,
)
from tests.conftest import EXAMPLES, POSTGRES_URL
from tests.test_refit import BASE, run, seed_fake

runner = CliRunner()

POLICY = {"max_error_rate": 0.1, "verify_max_error_rate": 0.25, "verify_sample_rate": 0.1}
NOW = BASE + timedelta(hours=2)


@pytest.fixture
def routing() -> RoutingConfig:
    return parse_routing(
        {
            "defaults": {"min_labeled": 200},
            "questions": {"refund.eligible": POLICY, "ticket.queue": POLICY},
        }
    )


async def _pass(
    database_url: str, catalog: QuestionCatalog, routing: RoutingConfig, **kwargs: Any
) -> PassReport:
    engine = make_engine(database_url)
    try:
        gate = build_worker(engine, catalog.questions, routing, clock=lambda: NOW, **kwargs)
        return await gate.run_once()
    finally:
        await engine.dispose()


def test_a_pass_refits_then_refreshes_then_checks_drift(
    database_url: str, example_catalog: QuestionCatalog, routing: RoutingConfig
) -> None:
    seed_fake(database_url, example_catalog, "refund.eligible", 600)

    report = run(_pass(database_url, example_catalog, routing))

    assert report.skipped is None
    assert report.errors == []
    steps = [(s.step, s.question) for s in report.steps]
    # Refit covers every question; routing and drift only the routed ones; strictly in order.
    assert [s for s, _ in steps] == ["refit"] * 3 + ["routing"] * 2 + ["drift"] * 2
    assert {q for s, q in steps if s == "routing"} == {"refund.eligible@v1", "ticket.queue@v1"}
    by_key = {(s.step, s.question): s for s in report.steps}
    assert by_key["refit", "refund.eligible@v1"].status == "activated"
    assert by_key["refit", "ticket.queue@v1"].status == "skipped"
    refreshed = by_key["routing", "refund.eligible@v1"]
    assert refreshed.status == "refreshed"
    # The refresh ran on the calibrator the refit just activated, so it found a threshold.
    assert not refreshed.detail.startswith("auto none")
    assert by_key["routing", "ticket.queue@v1"].detail.startswith("auto none")
    assert by_key["drift", "refund.eligible@v1"].status in ("ok", "insufficient")
    assert report.finished_at == NOW

    payload = report.as_dict()
    assert payload["errors"] == 0
    assert payload["drifting"] == []
    assert len(payload["steps"]) == 7

    # The next pass keeps the active calibrator unless a refit beats it.
    again = run(_pass(database_url, example_catalog, routing))
    assert {(s.step, s.question): s.status for s in again.steps}["refit", "refund.eligible@v1"] in (
        "rejected",
        "activated",
        "candidate",
    )


class _Boom:
    """A refit job that fails for one question and works for the rest."""

    def __init__(self, failing: str) -> None:
        self.failing = failing
        self.calls: list[str] = []

    async def refit(self, question_id: str, version: int, **_: Any) -> Any:
        self.calls.append(question_id)
        if question_id == self.failing:
            raise RuntimeError("database went away")

        class _Result:
            outcome = "skipped"
            reason = "too few labels"

        return _Result()


class _Refresher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def refresh(self, question: Any) -> Any:
        self.calls.append(question.id)

        class _Rule:
            auto_threshold = 0.9
            verify_threshold = None
            reason = "selected"

        class _Result:
            rule = _Rule()

        return _Result()


class _Monitor:
    def __init__(self, status: str = "drift") -> None:
        self.status = status

    def monitored(self, question: Any) -> bool:
        return bool(question.id == "refund.eligible")

    async def check(self, question: Any, *, now: datetime) -> Any:
        class _Check:
            status = self.status
            action = "tightened"
            action_reason = "auto threshold raised"

        return _Check()


def _stub_worker(
    catalog: QuestionCatalog, routing: RoutingConfig, **kwargs: Any
) -> tuple[MaintenanceWorker, _Boom, _Refresher]:
    refit, refresher = _Boom("ticket.queue"), _Refresher()
    worker = MaintenanceWorker(
        catalog.questions,
        routing,
        refit=refit,  # type: ignore[arg-type]
        refresher=refresher,  # type: ignore[arg-type]
        monitor=kwargs.pop("monitor", _Monitor()),
        clock=lambda: NOW,
        **kwargs,
    )
    return worker, refit, refresher


def test_one_failing_question_does_not_stop_the_pass(
    example_catalog: QuestionCatalog, routing: RoutingConfig
) -> None:
    worker, refit, refresher = _stub_worker(example_catalog, routing)

    report = run(worker.run_once())

    assert refit.calls == ["refund.eligible", "ticket.queue", "ticket.handling_minutes"]
    assert refresher.calls == ["refund.eligible", "ticket.queue"]
    [error] = report.errors
    assert error == StepResult(
        "refit", "ticket.queue@v1", "error", "RuntimeError: database went away"
    )
    assert report.drifting == ["refund.eligible@v1"]
    drift = report.steps[-1]
    assert drift.detail == "action tightened: auto threshold raised"
    assert report.steps[3].detail == "auto 0.9000, verify none: selected"


def test_a_held_lock_skips_the_pass(
    example_catalog: QuestionCatalog, routing: RoutingConfig
) -> None:
    class _Held:
        def hold(self) -> Any:
            import contextlib

            @contextlib.asynccontextmanager
            async def held() -> AsyncIterator[bool]:
                yield False

            return held()

    worker, refit, _ = _stub_worker(example_catalog, routing, lock=_Held())

    report = run(worker.run_once())

    assert report.skipped == "another worker is running a pass"
    assert report.steps == []
    assert refit.calls == []
    assert report.as_dict()["skipped"] == report.skipped


def test_sqlite_needs_no_lock(sqlite_url: str) -> None:
    engine = make_engine(sqlite_url)
    try:
        assert isinstance(pass_lock(engine), NoLock)
    finally:
        run(engine.dispose())


@pytest.mark.postgres
@pytest.mark.skipif(not POSTGRES_URL, reason="needs DGATE_TEST_DATABASE_URL")
def test_postgres_advisory_lock_lets_one_worker_run(
    example_catalog: QuestionCatalog, routing: RoutingConfig
) -> None:
    assert POSTGRES_URL is not None

    async def scenario() -> tuple[bool, bool, bool]:
        engine = make_engine(POSTGRES_URL)
        other = make_engine(POSTGRES_URL)
        try:
            lock = pass_lock(engine)
            assert isinstance(lock, AdvisoryLock)
            async with lock.hold() as first:
                # Another session (another worker replica) cannot take it meanwhile.
                async with other.connect() as conn:
                    taken = (
                        await conn.execute(
                            text("SELECT pg_try_advisory_lock(:k)"), {"k": ADVISORY_LOCK_KEY}
                        )
                    ).scalar_one()
                async with AdvisoryLock(other).hold() as second:
                    pass
            async with lock.hold() as after:
                pass
            return bool(first), bool(taken) or bool(second), bool(after)
        finally:
            await engine.dispose()
            await other.dispose()

    first, other_got_it, after = asyncio.run(scenario())
    assert first is True
    assert other_got_it is False
    assert after is True  # freed when the pass ends


def test_run_forever_repeats_until_stopped(
    example_catalog: QuestionCatalog, routing: RoutingConfig
) -> None:
    worker, refit, _ = _stub_worker(example_catalog, routing, monitor=_Monitor("ok"))
    seen: list[PassReport] = []

    async def scenario() -> int:
        stop = asyncio.Event()

        def on_report(report: PassReport) -> None:
            seen.append(report)
            if len(seen) == 3:
                stop.set()

        return await worker.run_forever(0.001, stop, on_report)

    assert asyncio.run(scenario()) == 3
    assert len(seen) == 3
    assert refit.calls.count("refund.eligible") == 3
    assert all(r.drifting == [] for r in seen)


def test_run_forever_without_a_callback_stops_on_the_event(
    example_catalog: QuestionCatalog, routing: RoutingConfig
) -> None:
    worker, _, _ = _stub_worker(example_catalog, routing)

    async def scenario() -> int:
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run_forever(60.0, stop))
        await asyncio.sleep(0.05)
        stop.set()  # wakes the wait at once rather than after 60 s
        return await asyncio.wait_for(task, timeout=5)

    assert asyncio.run(scenario()) == 1


def _cli(sqlite_url: str, *extra: str) -> Any:
    return runner.invoke(
        cli,
        [
            "worker",
            "--once",
            "-q",
            str(EXAMPLES / "questions.yaml"),
            "-r",
            str(EXAMPLES / "routing.yaml"),
            "--url",
            sqlite_url,
            *extra,
        ],
    )


def test_worker_cli_runs_one_pass(sqlite_url: str, example_catalog: QuestionCatalog) -> None:
    seed_fake(sqlite_url, example_catalog, "refund.eligible", 300)

    result = _cli(sqlite_url, "--min-count", "50")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["errors"] == 0
    statuses = {(s["step"], s["question"]): s["status"] for s in payload["steps"]}
    assert statuses["refit", "refund.eligible@v1"] == "activated"
    assert statuses["routing", "ticket.handling_minutes@v1"] == "refreshed"


@pytest.mark.parametrize(("make", "code"), [("error", 1), ("drift", 3)])
def test_worker_cli_exit_codes(
    sqlite_url: str, monkeypatch: pytest.MonkeyPatch, make: str, code: int
) -> None:
    def fake_build(*_: Any, **__: Any) -> Any:
        class _Gate:
            async def run_once(self) -> PassReport:
                report = PassReport(started_at=NOW, finished_at=NOW)
                status = "error" if make == "error" else "drift"
                step = "refit" if make == "error" else "drift"
                report.steps.append(StepResult(step, "refund.eligible@v1", status, "x"))
                return report

        return _Gate()

    monkeypatch.setattr("decision_gate.worker.build_worker", fake_build)

    result = _cli(sqlite_url)

    assert result.exit_code == code, result.output


def test_noop_lock_always_runs() -> None:
    async def scenario() -> bool:
        async with NoLock().hold() as got:
            return got

    assert asyncio.run(scenario()) is True
