"""The maintenance worker: refit, routing refresh and drift checks, in that order, on a timer.

Each pass runs three steps over the question catalog:

1. **Refit** every question version's calibrator on its labels. A new calibrator is only
   activated when it beats the active one on held-out data.
2. **Refresh routing** for every routed question, so thresholds are selected on the
   newest active calibrator. The refresher already keeps any drift tightening in force.
3. **Check drift** for every monitored question, last, so a tightening lands on the rule
   just stored rather than being overwritten by a refresh later in the same pass.

A failure in one question's step is recorded and the pass moves on: one bad question must
not stop maintenance for the others. With several worker replicas on PostgreSQL, a
session-level advisory lock makes sure only one of them runs a pass at a time; the others
skip that pass. SQLite (tests, single process) needs no lock.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from decision_gate.drift_monitor import DriftMonitor
from decision_gate.refit import RefitJob
from decision_gate.routing import RoutingConfig, RoutingRefresher
from decision_gate.selection import AnyQuestion

logger = logging.getLogger(__name__)

# Arbitrary but fixed: "dgate" in ASCII. Every worker against one database uses the same key.
ADVISORY_LOCK_KEY = 0x6467617465


class PassLock(Protocol):
    def hold(self) -> contextlib.AbstractAsyncContextManager[bool]:
        """Yield True if this worker may run the pass, False if another one holds it."""


class NoLock:
    """For SQLite and single-process runs: every pass may run."""

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator[bool]:
        yield True


class AdvisoryLock:
    """A PostgreSQL session advisory lock, taken without waiting and freed after the pass."""

    def __init__(self, engine: AsyncEngine, key: int = ADVISORY_LOCK_KEY) -> None:
        self._engine = engine
        self._key = key

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator[bool]:
        async with self._engine.connect() as conn:
            got = bool(
                (
                    await conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": self._key})
                ).scalar_one()
            )
            # Session locks are not tied to a transaction; end the implicit one so the
            # connection does not sit "idle in transaction" for the whole pass.
            await conn.commit()
            try:
                yield got
            finally:
                if got:
                    await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": self._key})
                    await conn.commit()


def pass_lock(engine: AsyncEngine) -> PassLock:
    return AdvisoryLock(engine) if engine.dialect.name == "postgresql" else NoLock()


@dataclass(slots=True)
class StepResult:
    step: str
    """``refit``, ``routing`` or ``drift``."""
    question: str
    status: str
    """The step's own outcome (e.g. ``activated``, ``refreshed``, ``drift``) or ``error``."""
    detail: str


@dataclass(slots=True)
class PassReport:
    started_at: datetime
    finished_at: datetime | None = None
    skipped: str | None = None
    """Why the pass did not run (another worker holds the lock), else ``None``."""
    steps: list[StepResult] = field(default_factory=list)

    @property
    def errors(self) -> list[StepResult]:
        return [s for s in self.steps if s.status == "error"]

    @property
    def drifting(self) -> list[str]:
        return [s.question for s in self.steps if s.step == "drift" and s.status == "drift"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "skipped": self.skipped,
            "errors": len(self.errors),
            "drifting": self.drifting,
            "steps": [
                {"step": s.step, "question": s.question, "status": s.status, "detail": s.detail}
                for s in self.steps
            ],
        }


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MaintenanceWorker:
    def __init__(
        self,
        questions: Sequence[AnyQuestion],
        routing: RoutingConfig,
        *,
        refit: RefitJob,
        refresher: RoutingRefresher,
        monitor: DriftMonitor,
        lock: PassLock | None = None,
        refit_options: dict[str, Any] | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._questions = list(questions)
        self._routing = routing
        self._refit = refit
        self._refresher = refresher
        self._monitor = monitor
        self._lock = lock or NoLock()
        self._refit_options = dict(refit_options or {})
        self._clock = clock

    async def run_once(self) -> PassReport:
        report = PassReport(started_at=self._clock())
        async with self._lock.hold() as acquired:
            if not acquired:
                report.skipped = "another worker is running a pass"
            else:
                await self._pass(report)
        report.finished_at = self._clock()
        return report

    async def _pass(self, report: PassReport) -> None:
        for q in self._questions:
            await self._step(report, "refit", q, self._do_refit(q))
        for q in self._questions:
            if self._routing.policy_for(q.id) is not None:
                await self._step(report, "routing", q, self._do_refresh(q))
        for q in self._questions:
            if self._monitor.monitored(q):
                await self._step(report, "drift", q, self._do_drift(q))

    async def _step(
        self, report: PassReport, step: str, q: AnyQuestion, work: Awaitable[tuple[str, str]]
    ) -> None:
        try:
            status, detail = await work
        except Exception as exc:  # one question's failure must not stop the pass
            logger.exception("worker %s failed for %s", step, q.key)
            status, detail = "error", f"{type(exc).__name__}: {exc}"
        report.steps.append(StepResult(step, q.key, status, detail))

    async def _do_refit(self, q: AnyQuestion) -> tuple[str, str]:
        result = await self._refit.refit(q.id, q.version, **self._refit_options)
        return result.outcome, result.reason

    async def _do_refresh(self, q: AnyQuestion) -> tuple[str, str]:
        rule = (await self._refresher.refresh(q)).rule
        auto = "none" if rule.auto_threshold is None else f"{rule.auto_threshold:.4f}"
        verify = "none" if rule.verify_threshold is None else f"{rule.verify_threshold:.4f}"
        return "refreshed", f"auto {auto}, verify {verify}: {rule.reason}"

    async def _do_drift(self, q: AnyQuestion) -> tuple[str, str]:
        check = await self._monitor.check(q, now=self._clock())
        detail = f"action {check.action}"
        if check.action not in ("none", "skipped") and check.action_reason:
            detail += f": {check.action_reason}"
        return check.status, detail

    async def run_forever(
        self,
        interval_s: float,
        stop: asyncio.Event,
        on_report: Callable[[PassReport], None] | None = None,
    ) -> int:
        """Run a pass, then wait ``interval_s`` (or until ``stop``), and repeat.

        Returns the number of passes started. A pass in progress is finished before
        stopping, so a rule is never half written.
        """
        passes = 0
        while not stop.is_set():
            report = await self.run_once()
            passes += 1
            if on_report is not None:
                on_report(report)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
        return passes


def build_worker(
    engine: AsyncEngine,
    questions: Sequence[AnyQuestion],
    routing: RoutingConfig,
    *,
    alerter: Any = None,
    refit_options: dict[str, Any] | None = None,
    clock: Callable[[], datetime] = _utcnow,
) -> MaintenanceWorker:
    """Wire the worker to one database, the way ``dgate worker`` runs it."""
    from decision_gate.db import make_sessionmaker
    from decision_gate.db.repository import (
        CalibratorRepository,
        DecisionRepository,
        DriftCheckRepository,
        OutcomeRepository,
        RoutingRuleRepository,
    )
    from decision_gate.selection import GateAnalyzer

    sessions = make_sessionmaker(engine)
    outcomes = OutcomeRepository(sessions)
    calibrators = CalibratorRepository(sessions)
    rules = RoutingRuleRepository(sessions)
    checks = DriftCheckRepository(sessions)
    analyzer = GateAnalyzer(outcomes, calibrators)
    return MaintenanceWorker(
        questions,
        routing,
        refit=RefitJob(outcomes, calibrators),
        refresher=RoutingRefresher(routing, analyzer, calibrators, rules, checks, clock=clock),
        monitor=DriftMonitor(
            routing,
            decisions=DecisionRepository(sessions),
            outcomes=outcomes,
            calibrators=calibrators,
            rules=rules,
            checks=checks,
            analyzer=analyzer,
            alerter=alerter,
            clock=clock,
        ),
        lock=pass_lock(engine),
        refit_options=refit_options,
        clock=clock,
    )
