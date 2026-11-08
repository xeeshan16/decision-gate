"""Run the real gate on synthetic traffic, on a simulated clock.

Nothing here re-implements the gate. The simulation drives the same components the API
uses, against a real database: :class:`~decision_gate.service.DecisionService` with the
:class:`~decision_gate.routing.Router`, the review queue, outcome ingestion, the refit job,
routing refreshes and the drift monitor. Only the clock, the traffic, the model and the
people are simulated.

The timeline has three phases:

1. **history**: the gate is new. With no calibrator, every decision goes to a person,
   and their answers become the first labels. At the end of the phase a calibrator is
   fitted and the first routing rule selected.
2. **steady**: the same traffic, now routed. People handle the verify and human tiers;
   the true answer for a share of auto decisions arrives a few days later. The drift
   monitor runs every night, and the calibrator is refitted and routing refreshed on a
   schedule.
3. **drift**: the population changes (see :mod:`decision_gate.simulator.datasets`) and
   nobody tells the gate.

Because the simulation knows the truth of every decision, it records what production
cannot see: the auto tier's real error rate, day by day.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import func, select

from decision_gate.alerts import AlertSink
from decision_gate.backends.base import AnyQuestion
from decision_gate.db import make_engine, make_sessionmaker, migrate
from decision_gate.db.models import DecisionRecord
from decision_gate.db.repository import (
    CalibratorRepository,
    DecisionRepository,
    DriftCheckRepository,
    OutcomeRepository,
    ReviewRepository,
    RoutingRuleRepository,
)
from decision_gate.drift_monitor import DriftMonitor
from decision_gate.questions import Answer, ChoiceQuestion, QuestionCatalog
from decision_gate.refit import RefitJob
from decision_gate.reviews import ReviewService, ReviewSla
from decision_gate.routing import Router, RoutingConfig, RoutingRefresher, parse_routing
from decision_gate.selection import GateAnalyzer
from decision_gate.service import DecisionService, OutcomeService
from decision_gate.simulator.datasets import Dataset, Item, Segment, make_dataset
from decision_gate.simulator.model import Profile, SimulatedModel, default_profiles

DAY = timedelta(days=1)
DEFAULT_START = datetime(2026, 1, 5, tzinfo=UTC)
OUTCOME_SOURCE = "simulator"
REVIEWER = "sim-reviewer"
ID_NAMESPACE = uuid.UUID("5b1f3c2e-9d4a-4f6b-8a71-2c0e7d9b6a13")

# Routing used when the caller gives none: the example policies for the two simulated
# questions, with the example drift settings.
DEFAULT_ROUTING: dict[str, Any] = {
    "defaults": {
        "confidence": 0.95,
        "min_labeled": 200,
        "min_accepted": 100,
        "verify_sample_rate": 0.05,
        "require_calibrator": True,
    },
    "questions": {
        "refund.eligible": {
            "max_error_rate": 0.10,
            "verify_max_error_rate": 0.25,
            "conformal_alpha": 0.10,
        },
        "ticket.queue": {
            "max_error_rate": 0.05,
            "verify_max_error_rate": 0.20,
            "conformal_alpha": 0.10,
        },
    },
    "drift": {
        "window_hours": 168,
        "reference_hours": 672,
        "min_decisions": 200,
        "min_labeled": 100,
        "min_auto_labeled": 30,
        "max_score_psi": 0.25,
        "max_label_psi": 0.25,
        "max_ece": 0.05,
        "action": "tighten",
    },
}


def default_routing() -> RoutingConfig:
    return parse_routing(DEFAULT_ROUTING, source="<simulator defaults>")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    datasets: tuple[str, ...] = ("refund", "ticket")
    seed: int = 7
    per_day: int = 300
    history_days: int = 21
    steady_days: int = 14
    drift_days: int = 14
    overconfidence: float = 2.0
    reviewer_accuracy: float = 1.0
    outcome_rate: float = 0.1
    outcome_delay_days: float = 3.0
    refit_every_days: int = 7
    start: datetime = DEFAULT_START

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("choose at least one dataset")
        if self.per_day < 1:
            raise ValueError("per_day must be at least 1")
        if self.history_days < 1:
            raise ValueError("history_days must be at least 1")
        if self.steady_days < 0 or self.drift_days < 0:
            raise ValueError("steady_days and drift_days cannot be negative")
        if self.overconfidence <= 0:
            raise ValueError("overconfidence must be positive")
        if not 0.0 <= self.reviewer_accuracy <= 1.0:
            raise ValueError("reviewer_accuracy must be within [0, 1]")
        if not 0.0 <= self.outcome_rate <= 1.0:
            raise ValueError("outcome_rate must be within [0, 1]")
        if self.outcome_delay_days < 0:
            raise ValueError("outcome_delay_days cannot be negative")
        if self.refit_every_days < 0:
            raise ValueError("refit_every_days cannot be negative")
        if self.start.tzinfo is None:
            raise ValueError("start must be timezone-aware")

    @property
    def days(self) -> int:
        return self.history_days + self.steady_days + self.drift_days

    def phase(self, day: int) -> str:
        """Phase of a zero-based day."""
        if day < self.history_days:
            return "history"
        if day < self.history_days + self.steady_days:
            return "steady"
        return "drift"

    def maintenance_day(self, day: int) -> bool:
        """Whether a calibrator refit and routing refresh run at the end of ``day``."""
        since_first = day + 1 - self.history_days
        if since_first == 0:
            return True
        return (
            since_first > 0
            and self.refit_every_days > 0
            and since_first % self.refit_every_days == 0
        )


class SimClock:
    """A settable clock, handed to every component that asks what time it is."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


@dataclass(slots=True)
class DecisionTrace:
    """What the simulation knows about one decision, truth included."""

    day: int
    segment: Segment
    tier: str
    confidence: float
    calibrated: float | None
    correct: bool
    labeled_by: str | None


@dataclass(slots=True)
class Event:
    day: int
    question: str
    kind: str
    detail: dict[str, Any]


@dataclass(slots=True)
class QuestionRun:
    question: AnyQuestion
    traces: list[DecisionTrace] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    # Thresholds in force at the start of each day: (auto, verify).
    thresholds: list[tuple[float | None, float | None]] = field(default_factory=list)
    checks: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationResult:
    config: SimulationConfig
    routing: RoutingConfig
    runs: dict[str, QuestionRun]
    alerts: list[tuple[str, dict[str, Any]]]
    database_url: str | None


class _RecordingAlerter:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send(self, event: str, payload: dict[str, Any]) -> None:
        self.sent.append((event, payload))


def _wrong_answer(question: AnyQuestion, truth: Answer, rng: np.random.Generator) -> Answer:
    if isinstance(question, ChoiceQuestion):
        others = [o for o in question.options if o != truth]
        return others[int(rng.integers(len(others)))]
    return not truth


class Simulation:
    """One simulated run. Build it, then ``await run()``."""

    def __init__(
        self,
        config: SimulationConfig,
        routing: RoutingConfig | None = None,
        *,
        database_url: str | None = None,
        profiles: dict[Segment, Profile] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.config = config
        self.routing = routing or default_routing()
        self.datasets: list[Dataset] = [make_dataset(name) for name in config.datasets]
        self.catalog = QuestionCatalog(questions=tuple(d.question for d in self.datasets))
        self._database_url = database_url
        self._profiles = profiles or default_profiles(config.overconfidence)
        self._progress = progress
        self._truths: dict[str, Answer] = {}
        self._segments: dict[str, Segment] = {}
        self._next_id = uuid.uuid4()

    async def run(self) -> SimulationResult:
        if self._database_url is not None:
            return await self._run_on(self._database_url, keep=True)
        with tempfile.TemporaryDirectory(prefix="dgate-sim-") as tmp:
            url = f"sqlite+aiosqlite:///{Path(tmp) / 'simulation.db'}"
            return await self._run_on(url, keep=False)

    async def _run_on(self, database_url: str, *, keep: bool) -> SimulationResult:
        # Alembic's env runs its own event loop, so it cannot run on this one.
        await asyncio.to_thread(migrate.upgrade, database_url)
        engine = make_engine(database_url, single_connection=True)
        try:
            sessions = make_sessionmaker(engine)
            async with sessions() as session:
                existing = await session.scalar(
                    select(func.count(DecisionRecord.id)).where(
                        DecisionRecord.question_id.in_([q.id for q in self.catalog.questions])
                    )
                )
            if existing:
                raise ValueError(
                    f"the database already holds {existing} decisions for the simulated "
                    "questions; simulate into an empty database"
                )
            alerter = _RecordingAlerter()
            runs = await self._simulate(sessions, alerter)
        finally:
            await engine.dispose()
        return SimulationResult(
            config=self.config,
            routing=self.routing,
            runs=runs,
            alerts=alerter.sent,
            database_url=database_url if keep else None,
        )

    async def _simulate(self, sessions: Any, alerter: AlertSink) -> dict[str, QuestionRun]:
        cfg = self.config
        clock = SimClock(cfg.start)
        decisions = DecisionRepository(sessions)
        outcomes = OutcomeRepository(sessions)
        calibrators = CalibratorRepository(sessions)
        rules = RoutingRuleRepository(sessions)
        checks = DriftCheckRepository(sessions)
        analyzer = GateAnalyzer(outcomes, calibrators)
        # Rules only change at the end of a day, and the cache is cleared right after.
        router = Router(self.routing, rules, calibrators, cache_ttl_s=1e9)
        sla = ReviewSla()
        model = SimulatedModel(self._truths, self._segments, self._profiles)
        # Decision ids decide which decisions the router samples for verification, so
        # they are derived from the seed and the item: the same seed gives the same run.
        service = DecisionService(
            model,
            decisions,
            timeout_s=10.0,
            router=router,
            review_sla=sla,
            clock=clock,
            new_id=lambda: self._next_id,
        )
        outcome_service = OutcomeService(self.catalog, decisions, outcomes, clock=clock)
        reviews = ReviewService(self.catalog, ReviewRepository(sessions), sla, clock=clock)
        refit = RefitJob(outcomes, calibrators)
        refresher = RoutingRefresher(
            self.routing, analyzer, calibrators, rules, checks, clock=clock
        )
        monitor = DriftMonitor(
            self.routing,
            decisions=decisions,
            outcomes=outcomes,
            calibrators=calibrators,
            rules=rules,
            checks=checks,
            analyzer=analyzer,
            alerter=alerter,
            clock=clock,
        )

        runs = {d.question.id: QuestionRun(d.question) for d in self.datasets}
        pending: list[tuple[datetime, Any, Answer]] = []
        step = DAY / cfg.per_day
        for day in range(cfg.days):
            day_start = cfg.start + day * DAY
            day_end = day_start + DAY
            drifted = cfg.phase(day) == "drift"
            for index, dataset in enumerate(self.datasets):
                question = dataset.question
                run = runs[question.id]
                active = await rules.active(question.id, question.version)
                run.thresholds.append(
                    (active.auto_threshold, active.verify_threshold) if active else (None, None)
                )
                rng = np.random.default_rng([cfg.seed, index, day])
                for i in range(cfg.per_day):
                    item = dataset.generate(
                        rng, f"{dataset.name}-{day:03d}-{i:05d}", drifted=drifted
                    )
                    moment = day_start + step * (i + 0.5)
                    trace = await self._decide(
                        service, reviews, sla, clock, question, item, moment, rng, pending, day
                    )
                    run.traces.append(trace)
            # Truth that has arrived by the end of the day.
            clock.now = day_end
            due = [p for p in pending if p[0] <= day_end]
            pending = [p for p in pending if p[0] > day_end]
            for observed_at, decision_id, truth in due:
                await outcome_service.record(
                    decision_id, truth, source=OUTCOME_SOURCE, observed_at=observed_at
                )

            if cfg.maintenance_day(day):
                await self._maintain(refit, refresher, runs, day)
                router.invalidate()
            if day >= cfg.history_days:
                for question in self.catalog.questions:
                    if not monitor.monitored(question):
                        continue
                    check = await monitor.check(question, now=day_end)
                    runs[question.id].checks[day] = {
                        "status": check.status,
                        "action": check.action,
                        "action_reason": check.action_reason,
                        "signals": {s["name"]: s["status"] for s in check.signals},
                        "messages": {s["name"]: s["message"] for s in check.signals},
                    }
                    if check.status == "drift" or check.action not in ("none", "skipped"):
                        runs[question.id].events.append(
                            Event(day, question.key, "drift_check", runs[question.id].checks[day])
                        )
                router.invalidate()

            if self._progress is not None:
                self._progress(day + 1, cfg.days)
        return runs

    async def _decide(
        self,
        service: DecisionService,
        reviews: ReviewService,
        sla: ReviewSla,
        clock: SimClock,
        question: AnyQuestion,
        item: Item,
        moment: datetime,
        rng: np.random.Generator,
        pending: list[tuple[datetime, Any, Answer]],
        day: int,
    ) -> DecisionTrace:
        item_id = str(item.data["id"])
        self._truths[item_id] = item.truth
        self._segments[item_id] = item.segment
        self._next_id = uuid.uuid5(ID_NAMESPACE, f"{self.config.seed}|{item_id}")
        clock.now = moment
        record, _created = await service.decide(question, item.data)
        tier = record.tier or "human"
        correct = question.is_correct(record.answer, item.truth)
        labeled_by: str | None = None
        if tier in ("verify", "human"):
            # A reviewer picks the item up well within its deadline and answers it.
            # Items are resolved as they arrive, so the one just queued is the only one.
            claimed = await reviews.claim_next(REVIEWER, question_id=question.id)
            assert claimed is not None
            review_item, _decision = claimed
            clock.now = moment + timedelta(
                minutes=sla.minutes_for(tier) * float(rng.uniform(0.1, 0.9))
            )
            answer = (
                item.truth
                if rng.random() < self.config.reviewer_accuracy
                else _wrong_answer(question, item.truth, rng)
            )
            await reviews.resolve(review_item.id, REVIEWER, outcome=answer)
            labeled_by = "review"
        elif rng.random() < self.config.outcome_rate:
            observed = moment + timedelta(days=self.config.outcome_delay_days)
            pending.append((observed, record.id, item.truth))
            labeled_by = "outcome"
        return DecisionTrace(
            day=day,
            segment=item.segment,
            tier=tier,
            confidence=record.confidence,
            calibrated=record.calibrated_confidence,
            correct=correct,
            labeled_by=labeled_by,
        )

    async def _maintain(
        self,
        refit: RefitJob,
        refresher: RoutingRefresher,
        runs: dict[str, QuestionRun],
        day: int,
    ) -> None:
        for question in self.catalog.questions:
            run = runs[question.id]
            result = await refit.refit(question.id, question.version)
            run.events.append(
                Event(
                    day,
                    question.key,
                    "refit",
                    {
                        "outcome": result.outcome,
                        "reason": result.reason,
                        "labeled": result.labeled,
                        "method": result.record.method if result.record else None,
                        "version": result.record.version if result.record else None,
                        "evaluation": result.record.evaluation if result.record else None,
                    },
                )
            )
            if self.routing.policy_for(question.id) is None:
                continue
            refreshed = (await refresher.refresh(question)).rule
            run.events.append(
                Event(
                    day,
                    question.key,
                    "routing",
                    {
                        "auto_threshold": refreshed.auto_threshold,
                        "verify_threshold": refreshed.verify_threshold,
                        "reason": refreshed.reason,
                        "labeled": refreshed.labeled,
                    },
                )
            )


async def simulate(
    config: SimulationConfig,
    routing: RoutingConfig | None = None,
    *,
    database_url: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SimulationResult:
    return await Simulation(config, routing, database_url=database_url, progress=progress).run()


def export_items(dataset_name: str, *, seed: int, count: int, drifted: bool) -> Sequence[Item]:
    """Items of one dataset, generated the way a simulation generates its first day."""
    dataset = make_dataset(dataset_name)
    rng = np.random.default_rng([seed, 0, 0])
    return [dataset.generate(rng, f"{dataset.name}-{i:06d}", drifted=drifted) for i in range(count)]


__all__ = [
    "DEFAULT_ROUTING",
    "DecisionTrace",
    "Event",
    "QuestionRun",
    "SimClock",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "default_routing",
    "export_items",
    "simulate",
]
