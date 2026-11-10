"""The simulator: datasets, the simulated model, the end-to-end run, reports and CLI."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from decision_gate.cli import app as cli
from decision_gate.routing import parse_routing
from decision_gate.simulator.datasets import (
    QUEUES,
    RefundDataset,
    TicketDataset,
    make_dataset,
    refund_eligible,
    refund_window,
)
from decision_gate.simulator.engine import (
    DEFAULT_ROUTING,
    DecisionTrace,
    QuestionRun,
    SimClock,
    SimulationConfig,
    SimulationResult,
    default_routing,
    export_items,
    simulate,
)
from decision_gate.simulator.model import Profile, SimulatedModel, default_profiles
from decision_gate.simulator.report import (
    AutoError,
    analyze,
    analyze_question,
    auto_error,
    render_markdown,
    render_text,
    summarize,
)

runner = CliRunner()

# Small windows and sample sizes, so a few simulated days exercise every component.
SMALL_ROUTING: dict[str, Any] = {
    "defaults": {"min_labeled": 100, "min_accepted": 30, "verify_sample_rate": 0.1},
    "questions": {
        "refund.eligible": {"max_error_rate": 0.10, "verify_max_error_rate": 0.25},
        "ticket.queue": {
            "max_error_rate": 0.10,
            "verify_max_error_rate": 0.30,
            "conformal_alpha": 0.10,
        },
    },
    "drift": {
        "window_hours": 48,
        "reference_hours": 96,
        "min_decisions": 50,
        "min_labeled": 30,
        "min_auto_labeled": 10,
    },
}
SMALL = SimulationConfig(
    per_day=60, history_days=3, steady_days=2, drift_days=3, seed=3, refit_every_days=2
)


def small_routing() -> Any:
    return parse_routing(SMALL_ROUTING)


# --- Datasets -----------------------------------------------------------------------------


class TestRefundPolicy:
    @pytest.mark.parametrize(
        ("category", "reason", "opened", "window"),
        [
            ("grocery", "damaged", False, 7),
            ("grocery", "changed_mind", False, None),
            ("digital", "changed_mind", False, 14),
            ("digital", "damaged", True, None),
            ("marketplace", "late", True, 14),
            ("apparel", "wrong_item", True, 60),
            ("home", "changed_mind", False, 30),
        ],
    )
    def test_windows(self, category: str, reason: str, opened: bool, window: int | None) -> None:
        assert refund_window(category, reason, opened) == window

    def test_eligibility_is_inclusive_of_the_last_day(self) -> None:
        assert refund_eligible("home", "late", False, 30)
        assert not refund_eligible("home", "late", False, 31)
        assert not refund_eligible("grocery", "late", False, 0)


def _items(dataset: Any, n: int, *, drifted: bool, seed: int = 1) -> list[Any]:
    rng = np.random.default_rng(seed)
    return [dataset.generate(rng, f"x-{i}", drifted=drifted) for i in range(n)]


class TestDatasets:
    def test_same_seed_same_items(self) -> None:
        for dataset in (RefundDataset(), TicketDataset()):
            assert _items(dataset, 50, drifted=False) == _items(dataset, 50, drifted=False)
            assert _items(dataset, 50, drifted=False) != _items(dataset, 50, drifted=False, seed=2)

    def test_refund_truth_follows_the_policy(self) -> None:
        for item in _items(RefundDataset(), 2000, drifted=True):
            d = item.data
            assert item.truth == refund_eligible(
                d["category"], d["reason"], d["opened"], d["order_age_days"]
            )
            assert (item.segment == "novel") == (d["category"] == "marketplace")

    def test_refund_drift_changes_the_population(self) -> None:
        normal = _items(RefundDataset(), 3000, drifted=False)
        drifted = _items(RefundDataset(), 3000, drifted=True)
        assert all(i.segment != "novel" for i in normal)
        assert Counter(i.segment for i in drifted)["novel"] > 500
        assert np.mean([i.data["order_age_days"] for i in drifted]) > 1.4 * np.mean(
            [i.data["order_age_days"] for i in normal]
        )
        # Older orders and a 14-day window: fewer refunds are eligible.
        assert np.mean([i.truth for i in drifted]) < np.mean([i.truth for i in normal]) - 0.1

    def test_ticket_items(self) -> None:
        normal = _items(TicketDataset(), 3000, drifted=False)
        drifted = _items(TicketDataset(), 3000, drifted=True)
        assert {i.truth for i in normal} == set(QUEUES)
        assert all(i.segment != "novel" for i in normal)
        novel = [i for i in drifted if i.segment == "novel"]
        assert novel
        assert all(i.truth == "technical" and "5.0" in i.data["subject"] for i in novel)
        share = Counter(i.truth for i in drifted)["technical"] / len(drifted)
        assert 0.50 < share < 0.60
        assert set(normal[0].data) == {"id", "subject", "channel", "plan"}

    def test_unknown_dataset(self) -> None:
        with pytest.raises(ValueError, match="unknown dataset"):
            make_dataset("payroll")


# --- Simulated model ----------------------------------------------------------------------


class TestModel:
    def test_profiles_validate(self) -> None:
        with pytest.raises(ValueError, match="skill"):
            Profile(skill=1.0, overconfidence=1.0)
        with pytest.raises(ValueError, match="overconfidence"):
            Profile(skill=0.5, overconfidence=0.0)
        with pytest.raises(ValueError, match="no profile"):
            SimulatedModel({}, {}, {"easy": Profile(0.5, 1.0)})

    def test_skill_depends_on_the_segment(self) -> None:
        dataset = TicketDataset()
        items = _items(dataset, 4000, drifted=True)
        truths = {i.data["id"]: i.truth for i in items}
        segments = {i.data["id"]: i.segment for i in items}
        model = SimulatedModel(truths, segments)
        accuracy: dict[str, list[bool]] = {"easy": [], "hard": [], "novel": []}
        confidence: dict[str, list[float]] = {"easy": [], "hard": [], "novel": []}
        for item in items:
            raw = model.decide_sync(dataset.question, item.data)
            assert raw.backend == "simulated"
            assert raw.model_version == "sim-1"
            accuracy[item.segment].append(raw.answer == item.truth)
            confidence[item.segment].append(raw.confidence)
        easy, hard, novel = (float(np.mean(accuracy[s])) for s in ("easy", "hard", "novel"))
        assert easy > hard > novel
        # ...while the novel segment is no less sure of itself.
        assert np.mean(confidence["novel"]) > np.mean(accuracy["novel"]) + 0.3
        # Deterministic.
        again = model.decide_sync(dataset.question, items[0].data)
        assert again == model.decide_sync(dataset.question, items[0].data)

    def test_unknown_input(self) -> None:
        model = SimulatedModel({}, {})
        with pytest.raises(KeyError, match="no truth"):
            asyncio.run(model.decide(RefundDataset().question, {"id": "nope"}))

    def test_default_profiles_scale_with_overconfidence(self) -> None:
        profiles = default_profiles(3.0)
        assert profiles["easy"].overconfidence == 3.0
        assert profiles["novel"].overconfidence == 7.5


# --- Configuration ------------------------------------------------------------------------


class TestConfig:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("datasets", (), "at least one"),
            ("per_day", 0, "per_day"),
            ("history_days", 0, "history_days"),
            ("drift_days", -1, "negative"),
            ("overconfidence", 0.0, "overconfidence"),
            ("reviewer_accuracy", 1.5, "reviewer_accuracy"),
            ("outcome_rate", -0.1, "outcome_rate"),
            ("outcome_delay_days", -1.0, "outcome_delay_days"),
            ("refit_every_days", -1, "refit_every_days"),
            ("start", datetime(2026, 1, 1), "timezone"),
        ],
    )
    def test_invalid(self, field: str, value: Any, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            SimulationConfig(**{field: value})

    def test_phases_and_maintenance(self) -> None:
        cfg = SimulationConfig(history_days=3, steady_days=2, drift_days=2, refit_every_days=2)
        assert cfg.days == 7
        assert [cfg.phase(d) for d in range(7)] == ["history"] * 3 + ["steady"] * 2 + ["drift"] * 2
        assert [d for d in range(7) if cfg.maintenance_day(d)] == [2, 4, 6]
        once = SimulationConfig(history_days=3, refit_every_days=0)
        assert [d for d in range(10) if once.maintenance_day(d)] == [2]

    def test_clock_and_default_routing(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        clock = SimClock(start)
        assert clock() == start
        routing = default_routing()
        assert routing.policy_for("ticket.queue").max_error_rate == 0.05  # type: ignore[union-attr]
        assert DEFAULT_ROUTING["drift"]["action"] == "tighten"


# --- End to end ---------------------------------------------------------------------------


def run_small(database_url: str | None = None, **overrides: Any) -> SimulationResult:
    config = dataclasses.replace(SMALL, **overrides)
    return asyncio.run(simulate(config, small_routing(), database_url=database_url))


@pytest.fixture(scope="module")
def small_result() -> SimulationResult:
    return run_small()


class TestSimulation:
    def test_runs_the_whole_gate(self, small_result: SimulationResult) -> None:
        cfg = small_result.config
        assert small_result.database_url is None
        for run in small_result.runs.values():
            assert len(run.traces) == cfg.per_day * cfg.days
            assert len(run.thresholds) == cfg.days
            history = [t for t in run.traces if t.day < cfg.history_days]
            # No calibrator yet: everything goes to a person, who labels it.
            assert {t.tier for t in history} == {"human"}
            assert all(t.labeled_by == "review" for t in history)
            assert all(t.calibrated is None for t in history)
            kinds = Counter(e.kind for e in run.events)
            assert kinds["refit"] == kinds["routing"] == 3  # end of days 3, 5 and 7
            routed = [t for t in run.traces if t.day >= cfg.history_days]
            assert all(t.calibrated is not None for t in routed)
            # A drift check ran every night after the history.
            assert sorted(run.checks) == list(range(cfg.history_days, cfg.days))

    def test_same_seed_same_run(self, small_result: SimulationResult) -> None:
        again = run_small()
        for qid, run in small_result.runs.items():
            assert [(t.tier, t.correct) for t in run.traces] == [
                (t.tier, t.correct) for t in again.runs[qid].traces
            ]

    def test_into_a_kept_database(self, database_url: str) -> None:
        result = run_small(database_url, history_days=2, steady_days=1, drift_days=0, per_day=40)
        assert result.database_url == database_url
        # The database now holds the decisions; a second run refuses to mix with them.
        with pytest.raises(ValueError, match="empty database"):
            run_small(database_url, history_days=1, steady_days=0, drift_days=0, per_day=5)

    def test_unreliable_reviewers_and_delayed_outcomes(self) -> None:
        result = run_small(
            per_day=40,
            history_days=3,
            steady_days=2,
            drift_days=0,
            reviewer_accuracy=0.0,
            outcome_rate=1.0,
            outcome_delay_days=0.0,
        )
        for run in result.runs.values():
            assert {t.labeled_by for t in run.traces if t.tier == "auto"} <= {"outcome"}
        # Every review answer was wrong, so every history decision the model got right
        # was labeled as a mistake.
        report = analyze(result)
        for q in report["questions"]:
            assert q["findings"]["drift_starts_day"] is None
            assert "auto_error_before_action" not in q["findings"]
        assert "drift from day" not in render_text(report)

    def test_questions_without_a_policy_are_not_monitored(self) -> None:
        routing = parse_routing({"questions": {"ticket.queue": {"max_error_rate": 0.1}}})
        config = SimulationConfig(per_day=20, history_days=2, steady_days=1, drift_days=0)
        result = asyncio.run(simulate(config, routing))
        refund = result.runs["refund.eligible"]
        assert refund.checks == {}
        assert {e.kind for e in refund.events} == {"refit"}
        assert {t.tier for t in refund.traces} == {"human"}


# --- Reports ------------------------------------------------------------------------------


def trace(day: int, tier: str, correct: bool, calibrated: float | None = 0.9) -> DecisionTrace:
    return DecisionTrace(day, "easy", tier, 0.95, calibrated, correct, None)


class TestReport:
    def test_auto_error(self) -> None:
        assert AutoError(0, 0).rate is None
        assert AutoError(0, 0).interval is None
        stats = auto_error(
            [trace(0, "auto", True), trace(0, "auto", False), trace(0, "human", False)]
        )
        assert (stats.decisions, stats.errors, stats.rate) == (2, 1, 0.5)
        low, high = stats.interval  # type: ignore[misc]
        assert low < 0.5 < high

    def test_summarize(self) -> None:
        assert summarize([])["accuracy"] is None
        summary = summarize([trace(0, "auto", True)] * 60)
        assert summary["tiers"]["auto"] == 1.0
        assert summary["ece_raw"] == pytest.approx(0.05)
        assert summary["ece_calibrated"] == pytest.approx(0.1)

    def test_analysis_and_rendering(self, small_result: SimulationResult) -> None:
        report = analyze(small_result)
        text = render_text(report)
        markdown = render_markdown(report)
        json.dumps(report, default=str)
        for q in report["questions"]:
            assert q["question"] in text
            assert f"`{q['question']}`" in markdown
            assert set(q["phases"]) == {"history", "steady", "drift"}
            assert len(q["daily"]) == small_result.config.days
            assert q["findings"]["drift_starts_day"] == 6
        assert "calibrator v1 activated" in text
        assert "| phase |" in markdown

    def test_findings(self) -> None:
        cfg = SimulationConfig(per_day=2, history_days=1, steady_days=2, drift_days=3)
        run = QuestionRun(RefundDataset().question)
        run.thresholds = [(None, None)] + [(0.8, 0.5)] * 5
        for day in range(cfg.days):
            tier = "human" if day == 0 else "auto"
            run.traces += [trace(day, tier, True), trace(day, tier, day < 4)]
        ok = {"status": "ok", "action": "none", "signals": {}, "action_reason": ""}
        drift = {"status": "drift", "signals": {"ece": "drift"}, "action_reason": "raised"}
        run.checks = {
            1: {**drift, "action": "alert"},  # a false alarm before the drift
            2: ok,
            3: ok,
            4: {**drift, "action": "tightened"},
            5: {**drift, "action": "already_tightened"},
        }
        result = SimulationResult(cfg, small_routing(), {"refund.eligible": run}, [], None)
        findings = analyze_question(run, result)["findings"]
        assert findings["false_alarm_days"] == [2]
        assert findings["drift_starts_day"] == 4
        assert (findings["detected_day"], findings["action_day"]) == (5, 5)
        assert findings["detection_delay_days"] == 2
        assert findings["auto_error_before_action"]["decisions"] == 4
        assert findings["auto_error_after_action"]["errors"] == 1
        text = render_text(analyze(result))
        assert "drift reported before any drift on days [2]" in text
        assert "auto tier tightened at the end of day 5" in text

    def test_undetected_drift_and_no_policy(self) -> None:
        cfg = SimulationConfig(per_day=2, history_days=1, steady_days=0, drift_days=1)
        run = QuestionRun(TicketDataset().question)
        run.traces = [trace(0, "human", True), trace(1, "verify", True, None)]
        routing = parse_routing({})
        result = SimulationResult(cfg, routing, {"ticket.queue": run}, [], "sqlite:///kept.db")
        report = analyze(result)
        text = render_text(report)
        assert "no policy" in text
        assert "not detected" in text
        assert "database kept: sqlite:///kept.db" in text


# --- CLI ----------------------------------------------------------------------------------

TINY = ["--per-day", "40", "--history-days", "3", "--steady-days", "1", "--drift-days", "1"]


class TestCli:
    def test_run_writes_reports(self, tmp_path: Path) -> None:
        routing = tmp_path / "routing.yaml"
        routing.write_text(yaml.safe_dump(SMALL_ROUTING))
        md, js = tmp_path / "report.md", tmp_path / "report.json"
        result = runner.invoke(
            cli,
            [
                *("simulate", "run", *TINY, "-d", "refund", "-r", str(routing)),
                *("--markdown", str(md), "--json", str(js), "--refit-every", "0"),
                *("--diagrams", str(tmp_path / "svg")),
            ],
        )
        assert result.exit_code == 0, result.output
        # Only the history phase (120 decisions) reaches the 50 a diagram needs.
        assert [p.name for p in (tmp_path / "svg").iterdir()] == ["refund.eligible-history.svg"]
        assert "refund.eligible@v1" in result.stdout
        assert "ticket.queue" not in result.stdout
        assert md.read_text().startswith("# decision-gate simulation report")
        assert json.loads(js.read_text())["config"]["datasets"] == ["refund"]

    def test_run_with_default_routing_quietly(self) -> None:
        result = runner.invoke(cli, ["simulate", "run", *TINY, "-d", "ticket", "--quiet"])
        assert result.exit_code == 0, result.output
        assert "simulated day" not in result.output

    def test_run_errors(self, tmp_path: Path) -> None:
        bad = runner.invoke(cli, ["simulate", "run", "-d", "payroll"])
        assert bad.exit_code == 2
        assert "unknown dataset" in bad.output
        url = f"sqlite+aiosqlite:///{tmp_path / 'sim.db'}"
        first = runner.invoke(cli, ["simulate", "run", *TINY, "--database-url", url, "--quiet"])
        assert first.exit_code == 0, first.output
        assert f"database kept: {url}" in first.stdout
        second = runner.invoke(cli, ["simulate", "run", *TINY, "--database-url", url, "--quiet"])
        assert second.exit_code == 1
        assert "empty database" in second.output

    def test_export(self) -> None:
        result = runner.invoke(cli, ["simulate", "export", "ticket", "-n", "5", "--drifted"])
        assert result.exit_code == 0, result.output
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(lines) == 5
        assert set(lines[0]) == {"input", "truth", "segment"}
        assert [i.data for i in export_items("ticket", seed=7, count=5, drifted=True)] == [
            line["input"] for line in lines
        ]
        bad = runner.invoke(cli, ["simulate", "export", "payroll"])
        assert bad.exit_code == 2


@pytest.mark.parametrize(
    ("decisions", "errors", "verdict"),
    [
        (0, 0, "nothing automated"),
        (1000, 20, "within the 5% budget"),
        (1000, 52, "not distinguishable from the 5% budget"),
        (1000, 90, "ABOVE the 5% budget"),
    ],
)
def test_verdicts_allow_for_sampling_noise(decisions: int, errors: int, verdict: str) -> None:
    from decision_gate.simulator.report import _verdict

    assert _verdict(AutoError(decisions, errors).as_dict(), 0.05) == verdict


def test_a_suspended_auto_tier_automates_nothing() -> None:
    cfg = SimulationConfig(per_day=1, history_days=1, steady_days=0, drift_days=2)
    run = QuestionRun(RefundDataset().question)
    run.traces = [trace(0, "human", True), trace(1, "auto", False), trace(2, "verify", True)]
    run.checks = {1: {"status": "drift", "action": "suspended", "signals": {}, "action_reason": ""}}
    result = SimulationResult(cfg, small_routing(), {"refund.eligible": run}, [], None)
    text = render_text(analyze(result))
    assert "auto tier suspended at the end of day 2" in text
    assert "drift, after the action: nothing automated" in text
