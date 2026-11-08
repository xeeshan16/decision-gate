"""Drift monitoring: PSI, the drift policy, each signal, the monitor, alerts, API and CLI."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_gate.alerts import AlertError, WebhookAlerter, sign
from decision_gate.app import create_app
from decision_gate.backends import FakeBackend
from decision_gate.backends.fake import hash_oracle
from decision_gate.cli import app as cli
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import DecisionRecord, DriftCheckRecord, OutcomeRecord
from decision_gate.db.repository import (
    CalibratorRepository,
    DecisionRepository,
    DriftCheckRepository,
    LabeledRow,
    OutcomeRepository,
    RoutingRuleRepository,
)
from decision_gate.drift import (
    PSI_EPSILON,
    DriftConfig,
    DriftPolicy,
    Signal,
    categorical_psi,
    overall_status,
    parse_drift,
    population_stability_index,
)
from decision_gate.drift_monitor import (
    EVENT_DETECTED,
    EVENT_RECOVERED,
    DriftMonitor,
    _Scores,
    alert_payload,
    auto_error_signal,
    ece_signal,
    label_psi_signal,
    score_psi_signal,
)
from decision_gate.questions import PredicateQuestion, QuestionCatalog
from decision_gate.routing import (
    Router,
    RoutingConfig,
    RoutingConfigError,
    RoutingPolicy,
    RoutingRefresher,
    parse_routing,
)
from decision_gate.selection import GateAnalyzer, SelectionError
from tests.conftest import EXAMPLES
from tests.test_refit import BASE, _insert, run
from tests.test_selection import refit

runner = CliRunner()

QUESTION = "refund.eligible"
POLICY = {"max_error_rate": 0.1, "verify_max_error_rate": 0.25, "verify_sample_rate": 0.1}
# Small windows so tests can lay out reference and current traffic within a few hours.
DRIFT = {"window_hours": 1.5, "reference_hours": 2.0, "min_decisions": 200, "min_labeled": 100}
REFERENCE_N = 3000
CURRENT_START = 7200  # seconds after BASE: two hours later
CHECK_AT = BASE + timedelta(hours=3)
SELECTED_AT = BASE + timedelta(seconds=REFERENCE_N)


def config(**drift: Any) -> RoutingConfig:
    return parse_routing(
        {
            "defaults": {"min_labeled": 200},
            "questions": {QUESTION: POLICY, "ticket.handling_minutes": {"max_error_rate": 0.2}},
            "drift": {**DRIFT, **drift},
        }
    )


def flipped_oracle(question: Any, data: Any) -> Any:
    """Ground truth as a broken model sees it: wrong on 40% of inputs."""
    truth = hash_oracle(question, data)
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).digest()
    return (not truth) if digest[0] < 102 else truth


def seed(
    database_url: str,
    catalog: QuestionCatalog,
    n: int,
    *,
    start: int = 0,
    question_id: str = QUESTION,
    backend: FakeBackend | None = None,
    labeled: bool = True,
    truth: Any = hash_oracle,
) -> None:
    """``n`` decisions one second apart from ``BASE + start`` seconds, with outcomes."""
    question = catalog.get(question_id)
    backend = backend or FakeBackend(overconfidence=2.0)
    decisions: list[Any] = []
    outcomes: list[Any] = []
    for i in range(start, start + n):
        data = {"item": i}
        raw = backend.decide_sync(question, data)
        decision_id = uuid.uuid4()
        decisions.append(
            DecisionRecord(
                id=decision_id,
                question_id=question.id,
                question_version=question.version,
                input_sha256="0" * 64,
                input=data,
                answer=raw.answer,
                confidence=raw.confidence,
                probabilities=raw.probabilities,
                backend=raw.backend,
                model_version=raw.model_version,
                latency_ms=0.1,
                created_at=BASE + timedelta(seconds=i),
            )
        )
        if labeled:
            outcome = truth(question, data)
            outcomes.append(
                OutcomeRecord(
                    decision_id=decision_id,
                    outcome=outcome,
                    correct=question.is_correct(raw.answer, outcome),
                    source="test",
                    observed_at=BASE + timedelta(seconds=i),
                )
            )
    run(_insert(database_url, decisions))
    if outcomes:
        run(_insert(database_url, outcomes))


class RecordingAlerter:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.fail = fail

    async def send(self, event: str, payload: dict[str, Any]) -> None:
        if self.fail:
            raise AlertError("receiver down")
        self.sent.append((event, payload))


async def _with_monitor(database_url: str, routing: RoutingConfig, alerter: Any, work: Any) -> Any:
    engine = make_engine(database_url)
    try:
        sessions = make_sessionmaker(engine)
        outcomes = OutcomeRepository(sessions)
        calibrators = CalibratorRepository(sessions)
        rules = RoutingRuleRepository(sessions)
        checks = DriftCheckRepository(sessions)
        analyzer = GateAnalyzer(outcomes, calibrators)
        monitor = DriftMonitor(
            routing,
            decisions=DecisionRepository(sessions),
            outcomes=outcomes,
            calibrators=calibrators,
            rules=rules,
            checks=checks,
            analyzer=analyzer,
            alerter=alerter,
        )
        # Rules are selected right after the reference traffic, as they would be live.
        refresher = RoutingRefresher(
            routing, analyzer, calibrators, rules, checks, clock=lambda: SELECTED_AT
        )
        router = Router(routing, rules, calibrators, cache_ttl_s=0)
        return await work(monitor, refresher, router, rules, checks)
    finally:
        await engine.dispose()


def with_monitor(
    database_url: str, work: Any, routing: RoutingConfig | None = None, alerter: Any = None
) -> Any:
    return run(_with_monitor(database_url, routing or config(), alerter, work))


def prepare(database_url: str, catalog: QuestionCatalog, routing: RoutingConfig) -> Any:
    """Reference traffic, a fitted calibrator and a routing rule selected on it."""
    seed(database_url, catalog, REFERENCE_N)
    refit(database_url, QUESTION)

    async def work(_m: Any, refresher: Any, *_: Any) -> Any:
        return (await refresher.refresh(catalog.get(QUESTION))).rule

    rule = with_monitor(database_url, work, routing)
    assert rule.auto_threshold is not None
    return rule


def check(
    database_url: str,
    catalog: QuestionCatalog,
    *,
    at: datetime = CHECK_AT,
    routing: RoutingConfig | None = None,
    alerter: Any = None,
) -> DriftCheckRecord:
    async def work(monitor: DriftMonitor, *_: Any) -> DriftCheckRecord:
        return await monitor.check(catalog.get(QUESTION), now=at)

    result: DriftCheckRecord = with_monitor(database_url, work, routing, alerter)
    return result


def signals(record: DriftCheckRecord) -> dict[str, str]:
    return {s["name"]: s["status"] for s in record.signals}


def broken_backend() -> FakeBackend:
    return FakeBackend(overconfidence=2.0, oracle=flipped_oracle)


def slightly_worse_oracle(question: Any, data: Any) -> Any:
    """A model that is wrong about 4% more often: 10 of every 256 inputs flipped."""
    truth = hash_oracle(question, data)
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).digest()
    return (not truth) if digest[0] < 10 else truth


# --- Population stability index ----------------------------------------------------------


class TestPsi:
    def test_hand_computed_two_bins(self) -> None:
        reference = [0.1 * k for k in range(1, 11)]  # median 0.55
        current = [0.2, 0.3, 0.4, 0.5]  # all at or below the median
        result = population_stability_index(reference, current, bins=2)
        expected = (1.0 - 0.5) * math.log(1.0 / 0.5) + (PSI_EPSILON - 0.5) * math.log(
            PSI_EPSILON / 0.5
        )
        assert result.value == pytest.approx(expected)
        assert [b.label for b in result.bins] == ["(-inf, 0.5500]", "(0.5500, inf]"]
        assert (result.reference_count, result.current_count) == (10, 4)

    def test_same_distribution_is_near_zero_and_shift_is_large(self) -> None:
        rng = np.random.default_rng(1)
        reference = rng.beta(5, 2, 5000)
        same = rng.beta(5, 2, 5000)
        shifted = rng.beta(2, 2, 5000)
        assert population_stability_index(reference, same).value < 0.02
        assert population_stability_index(reference, shifted).value > 0.25

    def test_monotone_transform_leaves_psi_unchanged(self) -> None:
        rng = np.random.default_rng(2)
        reference = rng.uniform(0.5, 1.0, 1000)
        current = rng.uniform(0.6, 1.0, 1000)
        raw = population_stability_index(reference, current).value

        def squash(x: np.ndarray) -> np.ndarray:
            return np.asarray(1 / (1 + np.exp(-3 * (x - 0.7))))

        assert population_stability_index(squash(reference), squash(current)).value == (
            pytest.approx(raw)
        )

    def test_out_of_range_scores_land_in_the_open_bins(self) -> None:
        result = population_stability_index([0.4, 0.5, 0.6, 0.7], [0.0, 1.0], bins=2)
        shares = [b.current_share for b in result.bins]
        assert shares == [0.5, 0.5]

    def test_ties_merge_bins(self) -> None:
        result = population_stability_index([0.9] * 50 + [0.5] * 50, [0.9] * 10, bins=10)
        assert len(result.bins) < 11
        assert result.as_dict()["bins"][0]["contribution"] == pytest.approx(
            result.bins[0].contribution
        )

    @pytest.mark.parametrize(
        ("reference", "current", "bins", "match"),
        [
            ([0.1], [0.2], 1, "bins must be at least 2"),
            ([], [0.2], 10, "non-empty"),
            ([[0.1]], [0.2], 10, "non-empty"),
            ([0.1, math.nan], [0.2], 10, "finite"),
        ],
    )
    def test_rejects_bad_input(self, reference: Any, current: Any, bins: int, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            population_stability_index(reference, current, bins=bins)

    def test_categorical_hand_computed(self) -> None:
        result = categorical_psi({"true": 50, "false": 50}, {"true": 80, "false": 20})
        expected = (0.2 - 0.5) * math.log(0.2 / 0.5) + (0.8 - 0.5) * math.log(0.8 / 0.5)
        assert result.value == pytest.approx(expected)
        assert [b.label for b in result.bins] == ["false", "true"]

    def test_categorical_new_label_counts(self) -> None:
        result = categorical_psi({"a": 10}, {"a": 5, "b": 5})
        b = next(x for x in result.bins if x.label == "b")
        assert b.reference_share == 0.0
        assert b.contribution > 1.0

    @pytest.mark.parametrize(
        ("reference", "current", "match"),
        [({"a": -1, "b": 2}, {"a": 1}, "negative"), ({"a": 0}, {"a": 1}, "at least one")],
    )
    def test_categorical_rejects_bad_counts(
        self, reference: dict[str, int], current: dict[str, int], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            categorical_psi(reference, current)


def test_overall_status() -> None:
    ok = Signal("ece", "ok", "")
    drift = Signal("score_psi", "drift", "")
    short = Signal("label_psi", "insufficient", "")
    assert overall_status([ok, short]) == "ok"
    assert overall_status([ok, drift, short]) == "drift"
    assert overall_status([short]) == "insufficient"
    assert overall_status([]) == "insufficient"
    assert drift.as_dict()["name"] == "score_psi"


# --- Policy ------------------------------------------------------------------------------


class TestPolicy:
    def test_drift_section_merges_shared_fields(self) -> None:
        routing = parse_routing(
            {
                "questions": {"a.b": {}},
                "drift": {"window_hours": 24, "questions": {"a.b": {"max_ece": 0.1}}},
            }
        )
        assert routing.drift.policy_for("a.b").window_hours == 24
        assert routing.drift.policy_for("a.b").max_ece == 0.1
        assert routing.drift.policy_for("x.y").max_ece == 0.05
        assert parse_drift(None) == DriftConfig()
        assert parse_drift({"questions": None}).default == DriftPolicy()

    def test_drift_settings_do_not_change_policy_fingerprints(self) -> None:
        plain = parse_routing({"questions": {"a.b": POLICY}})
        tuned = parse_routing({"questions": {"a.b": POLICY}, "drift": {"max_ece": 0.2}})
        a, b = plain.policy_for("a.b"), tuned.policy_for("a.b")
        assert a is not None
        assert b is not None
        assert a.fingerprint() == b.fingerprint()

    def test_unknown_drift_question_is_reported(self, example_catalog: QuestionCatalog) -> None:
        routing = parse_routing({"drift": {"questions": {"no.such": {}}}})
        assert routing.unknown_questions(example_catalog) == ["no.such"]

    @pytest.mark.parametrize(
        ("raw", "match"),
        [
            ({"drift": [1]}, "must be a mapping"),
            ({"drift": {"questions": [1]}}, "must be mappings|must be a mapping"),
            ({"drift": {"max_ece": 2}}, "invalid drift policy"),
            ({"drift": {"bogus": 1}}, "invalid drift policy"),
            ({"drift": {"min_auto_labeled": 500, "min_labeled": 100}}, "invalid drift policy"),
            ({"drift": {"questions": {"a.b": {"action": "panic"}}}}, "invalid drift policy"),
        ],
    )
    def test_invalid_sections(self, raw: dict[str, Any], match: str) -> None:
        with pytest.raises(RoutingConfigError, match=match):
            parse_routing(raw)

    def test_example_file_has_a_drift_section(self) -> None:
        from decision_gate.routing import load_routing

        routing = load_routing(EXAMPLES / "routing.yaml")
        assert routing.drift.default.action == "tighten"


# --- Signals -----------------------------------------------------------------------------


def row(outcome: Any, correct: bool = True) -> LabeledRow:
    return LabeledRow(
        question_id=QUESTION,
        question_version=1,
        backend="fake",
        model_version="m",
        confidence=0.8,
        correct=correct,
        probabilities=None,
        outcome=outcome,
    )


POLICY_SMALL = DriftPolicy(min_decisions=10, min_labeled=20, min_auto_labeled=10)


class TestSignals:
    def test_score_psi(self) -> None:
        few = score_psi_signal([0.5] * 5, [0.5] * 50, POLICY_SMALL)
        assert few.status == "insufficient"
        rng = np.random.default_rng(3)
        reference = rng.uniform(0.5, 1, 500).tolist()
        same = score_psi_signal(reference, rng.uniform(0.5, 1, 500).tolist(), POLICY_SMALL)
        assert same.status == "ok"
        moved = score_psi_signal(reference, rng.uniform(0.9, 1, 500).tolist(), POLICY_SMALL)
        assert moved.status == "drift"
        assert moved.detail["current_mean"] > moved.detail["reference_mean"]
        assert ">" in moved.message

    def test_label_psi(self, predicate: Any, score: Any) -> None:
        assert label_psi_signal(score, [], [], POLICY_SMALL).message.startswith("not measured")
        assert label_psi_signal(predicate, [row(True)] * 5, [], POLICY_SMALL).status == (
            "insufficient"
        )
        reference = [row(True)] * 50 + [row(False)] * 50
        steady = label_psi_signal(predicate, reference, reference, POLICY_SMALL)
        assert steady.status == "ok"
        assert steady.value == pytest.approx(0.0)
        shifted = [row(True)] * 90 + [row(False)] * 10
        result = label_psi_signal(predicate, reference, shifted, POLICY_SMALL)
        assert result.status == "drift"
        assert "'false' 50.0% -> 10.0%" in result.message

    def test_ece(self) -> None:
        missing = _Scores(None, np.array([]), "none", None, "no active calibrator")
        assert ece_signal(missing, POLICY_SMALL).message == "no scores: no active calibrator"
        few = _Scores(np.full(5, 0.8), np.ones(5), "raw", None)
        assert ece_signal(few, POLICY_SMALL).status == "insufficient"
        rng = np.random.default_rng(4)
        p = rng.uniform(0.5, 1.0, 2000)
        calibrated = _Scores(p, (rng.uniform(size=2000) < p).astype(float), "raw", None)
        assert ece_signal(calibrated, POLICY_SMALL).status == "ok"
        overconfident = _Scores(p, (rng.uniform(size=2000) < p - 0.3).astype(float), "raw", None)
        result = ece_signal(overconfident, POLICY_SMALL)
        assert result.status == "drift"
        assert result.detail["interval"][0] > POLICY_SMALL.max_ece

    def test_auto_error(self) -> None:
        policy = RoutingPolicy(max_error_rate=0.1)
        scores = np.linspace(0.5, 1.0, 200)
        good = _Scores(scores, np.ones(200), "raw", None)

        def rule(threshold: float | None) -> Any:
            return type("R", (), {"auto_threshold": threshold})()

        assert auto_error_signal(good, None, policy, POLICY_SMALL).status == "insufficient"
        assert (
            "no auto threshold" in auto_error_signal(good, rule(None), policy, POLICY_SMALL).message
        )
        missing = _Scores(None, np.ones(200), "none", None, "x")
        assert auto_error_signal(missing, rule(0.9), policy, POLICY_SMALL).message == (
            "no scores: x"
        )
        assert auto_error_signal(good, rule(0.999), policy, POLICY_SMALL).status == "insufficient"
        ok = auto_error_signal(good, rule(0.8), policy, POLICY_SMALL)
        assert (ok.status, ok.value) == ("ok", 0.0)
        # Above 0.8, every other decision is wrong: 50% errors, far over a 10% budget.
        correct = np.where(np.arange(200) % 2 == 0, 1.0, 0.0)
        bad = auto_error_signal(
            _Scores(scores, correct, "raw", None), rule(0.8), policy, POLICY_SMALL
        )
        assert bad.status == "drift"
        assert bad.detail["error_lower_bound"] > 0.1
        # 2 errors in 80 (2.5%) is within budget even though the point estimate is non-zero.
        few_errors = np.ones(200)
        few_errors[-2:] = 0.0
        within = auto_error_signal(
            _Scores(scores, few_errors, "raw", None), rule(0.8), policy, POLICY_SMALL
        )
        assert within.status == "ok"


# --- The monitor against a database ------------------------------------------------------


class TestMonitor:
    def test_steady_traffic_is_ok(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        prepare(database_url, example_catalog, config())
        seed(database_url, example_catalog, 3000, start=CURRENT_START)
        alerter = RecordingAlerter()
        record = check(database_url, example_catalog, alerter=alerter)
        assert record.status == "ok", record.signals
        assert set(signals(record).values()) == {"ok"}
        assert (record.action, record.alert) == ("none", None)
        assert alerter.sent == []
        assert record.summary["current"]["decisions"] == 3000
        assert record.summary["current"]["labeled"] == 3000
        assert record.summary["rule_current"] is True
        # The reference ends where the rule was selected, before the current window starts.
        assert record.reference_end == SELECTED_AT < record.window_start

    def test_broken_model_tightens_routing_and_alerts(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        old = prepare(database_url, example_catalog, config())
        seed(
            database_url,
            example_catalog,
            3000,
            start=CURRENT_START,
            backend=broken_backend(),
        )
        alerter = RecordingAlerter()
        record = check(database_url, example_catalog, alerter=alerter)
        assert record.status == "drift"
        assert signals(record)["auto_error"] == "drift"
        assert signals(record)["ece"] == "drift"
        assert record.action in ("tightened", "suspended")
        assert record.previous_rule_id == old.id
        assert record.rule_id is not None
        assert (record.alert, record.alert_event) == ("sent", EVENT_DETECTED)
        event, payload = alerter.sent[0]
        assert event == EVENT_DETECTED
        assert payload["check_id"] == str(record.id)
        assert payload["rule_id"] == str(record.rule_id)
        assert "drift in" in payload["text"]

        async def work(_m: Any, _r: Any, router: Router, rules: Any, checks: Any) -> Any:
            question = example_catalog.get(QUESTION)
            active = await rules.active(QUESTION, 1)
            retired = await rules.get(old.id)
            route = await router.route(
                question, uuid.uuid4(), True, 0.999, {"true": 0.999, "false": 0.001}
            )
            return active, retired, route, await checks.get(record.id)

        active, retired, route, stored = with_monitor(database_url, work)
        assert active.id == record.rule_id
        assert active.drift_check_id == record.id
        assert active.policy_fingerprint == old.policy_fingerprint
        assert active.calibrator_id == old.calibrator_id
        assert retired.status == "retired"
        assert stored.alert == "sent"
        if record.action == "suspended":
            assert active.auto_threshold is None
            assert route.tier != "auto"
            assert route.tier == "verify" or "suspended after drift check" in route.reason
        else:
            assert active.auto_threshold > old.auto_threshold

    def test_drift_persists_without_a_second_alert_then_recovers(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        prepare(database_url, example_catalog, config())
        seed(database_url, example_catalog, 3000, start=CURRENT_START, backend=broken_backend())
        alerter = RecordingAlerter()
        first = check(database_url, example_catalog, alerter=alerter)
        assert first.status == "drift"
        # Ten minutes later the same traffic is still in the window.
        second = check(
            database_url, example_catalog, at=CHECK_AT + timedelta(minutes=10), alerter=alerter
        )
        assert second.status == "drift"
        assert second.action in ("already_tightened", "not_tightened")
        assert second.alert is None
        assert len(alerter.sent) == 1

        # The model is fixed: a later window holds only healthy traffic.
        later = CURRENT_START + 4 * 3600
        seed(database_url, example_catalog, 3000, start=later)
        recovered = check(
            database_url,
            example_catalog,
            at=BASE + timedelta(seconds=later + 3000),
            routing=config(window_hours=0.9),
            alerter=alerter,
        )
        assert recovered.status == "ok", recovered.signals
        assert recovered.alert_event == EVENT_RECOVERED
        assert alerter.sent[-1][1]["text"].endswith(
            "run a routing refresh to re-select thresholds on recent labels"
        )

    def test_refresh_keeps_a_suspension_until_drift_clears(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        prepare(database_url, example_catalog, config())
        # Unlabeled traffic from a much more confident model: the auto tier is suspended,
        # while every label (all from before the shift) would still support a threshold.
        seed(
            database_url,
            example_catalog,
            1000,
            start=CURRENT_START,
            backend=FakeBackend(overconfidence=6.0),
            labeled=False,
        )
        record = check(database_url, example_catalog)
        assert record.action == "suspended"

        async def refresh(_m: Any, refresher: Any, *_: Any) -> Any:
            return (await refresher.refresh(example_catalog.get(QUESTION))).rule

        refreshed = with_monitor(database_url, refresh)
        assert refreshed.drift_check_id == record.id
        assert refreshed.auto_threshold is None
        assert refreshed.reason.startswith("auto threshold suspended: drift check")

        async def low_score(_m: Any, _r: Any, router: Router, *_: Any) -> Any:
            return await router.route(
                example_catalog.get(QUESTION), uuid.uuid4(), True, 0.5, {"true": 0.5, "false": 0.5}
            )

        route = with_monitor(database_url, low_score)
        assert route.tier == "human"
        assert route.reason.startswith(f"the auto tier is suspended after drift check {record.id}")

        # Once a check says ok, a refresh may loosen again.
        later = CURRENT_START + 4 * 3600
        seed(database_url, example_catalog, 3000, start=later)
        ok = check(
            database_url,
            example_catalog,
            at=BASE + timedelta(seconds=later + 3000),
            routing=config(window_hours=0.9),
        )
        assert ok.status == "ok", ok.signals
        loosened = with_monitor(database_url, refresh)
        assert loosened.drift_check_id is None
        assert loosened.auto_threshold is not None

    def test_refresh_keeps_a_raised_threshold(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        old = prepare(database_url, example_catalog, config())
        check_id = uuid.uuid4()

        async def work(_m: Any, refresher: Any, _r: Any, rules: Any, checks: Any) -> Any:
            drifted = DriftCheckRecord(
                id=check_id,
                question_id=QUESTION,
                question_version=1,
                status="drift",
                checked_at=CHECK_AT,
                window_start=CHECK_AT,
                window_end=CHECK_AT,
                reference_start=CHECK_AT,
                reference_end=CHECK_AT,
                policy={},
                signals=[],
                summary={},
                action="tightened",
                action_reason="test",
                previous_rule_id=old.id,
            )
            raised = type(old)(
                id=uuid.uuid4(),
                question_id=QUESTION,
                question_version=1,
                policy_fingerprint=old.policy_fingerprint,
                policy=old.policy,
                calibrator_id=old.calibrator_id,
                auto_threshold=0.9999,
                verify_threshold=old.verify_threshold,
                labeled=old.labeled,
                eligible=old.eligible,
                reason="raised",
                evaluation={},
                drift_check_id=check_id,
            )
            await checks.add(drifted, raised)
            return (await refresher.refresh(example_catalog.get(QUESTION))).rule

        refreshed = with_monitor(database_url, work)
        assert refreshed.auto_threshold == 0.9999
        assert refreshed.drift_check_id == check_id
        assert refreshed.reason.startswith("auto threshold kept at 0.9999")
        assert refreshed.verify_threshold is None or refreshed.verify_threshold <= 0.9999

    def test_score_shift_without_labels_suspends_the_auto_tier(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        prepare(database_url, example_catalog, config())
        seed(
            database_url,
            example_catalog,
            1000,
            start=CURRENT_START,
            backend=FakeBackend(overconfidence=6.0),
            labeled=False,
        )
        record = check(database_url, example_catalog)
        assert signals(record) == {
            "score_psi": "drift",
            "label_psi": "insufficient",
            "ece": "insufficient",
            "auto_error": "insufficient",
        }
        assert record.action == "suspended"
        assert "fewer than min_labeled=200" in record.action_reason

    def test_window_labels_can_vindicate_the_threshold(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        # A label-mix shift alone, with a model that still performs: the window's labels
        # support the current threshold, so nothing is tightened.
        prepare(database_url, example_catalog, config())
        seed(database_url, example_catalog, 3000, start=CURRENT_START)
        record = check(database_url, example_catalog, routing=config(max_label_psi=1e-9))
        assert signals(record)["label_psi"] == "drift"
        assert record.action == "not_tightened"
        assert "support the current auto threshold" in record.action_reason
        assert record.rule_id is None

    def test_a_slightly_worse_model_gets_a_higher_threshold(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        old = prepare(database_url, example_catalog, config())
        worse = FakeBackend(overconfidence=2.0, oracle=slightly_worse_oracle)
        seed(database_url, example_catalog, 3000, start=CURRENT_START, backend=worse)
        # Only the (forced) outcome-mix signal fires; the window's own labels then pick a
        # stricter threshold than the one in force.
        record = check(database_url, example_catalog, routing=config(max_label_psi=1e-9))
        assert record.action == "tightened", record.action_reason
        assert "re-selected on 3000 labeled decisions in the window" in record.action_reason

        async def work(_m: Any, _r: Any, router: Router, rules: Any, _c: Any) -> Any:
            active = await rules.active(QUESTION, 1)
            between = (old.auto_threshold + active.auto_threshold) / 2
            route = await router.route(
                example_catalog.get(QUESTION),
                uuid.uuid4(),
                True,
                0.5,
                {"true": 0.5, "false": 0.5},
            )
            return active, between, route

        active, _between, route = with_monitor(database_url, work)
        assert active.auto_threshold > old.auto_threshold
        assert active.verify_threshold <= active.auto_threshold
        assert active.reason.startswith("drift (label_psi)")
        assert route.rule_id == active.id

    def test_a_calibrator_that_cannot_score_is_reported(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        from tests.test_selection import insert_calibrator

        seed(database_url, example_catalog, 500, start=CURRENT_START)
        insert_calibrator(database_url, QUESTION, "platt", {"bogus": 1})
        record = check(database_url, example_catalog)
        ece = next(s for s in record.signals if s["name"] == "ece")
        assert ece["status"] == "insufficient"
        assert "cannot score" in ece["message"]

    def test_window_selection_errors_suspend(
        self,
        database_url: str,
        example_catalog: QuestionCatalog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        prepare(database_url, example_catalog, config())
        seed(database_url, example_catalog, 3000, start=CURRENT_START, backend=broken_backend())

        async def boom(*_: Any, **__: Any) -> Any:
            raise SelectionError("scores unavailable")

        monkeypatch.setattr(GateAnalyzer, "thresholds", boom)
        record = check(database_url, example_catalog)
        assert record.action == "suspended"
        assert "no threshold can be selected on the window: scores unavailable" in (
            record.action_reason
        )

    def test_alert_only_policy_leaves_routing_alone(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        old = prepare(database_url, example_catalog, config())
        seed(database_url, example_catalog, 3000, start=CURRENT_START, backend=broken_backend())
        alerter = RecordingAlerter(fail=True)
        record = check(
            database_url, example_catalog, routing=config(action="alert"), alerter=alerter
        )
        assert record.status == "drift"
        assert (record.action, record.rule_id) == ("none", None)
        assert "routing is unchanged" in record.action_reason
        assert (record.alert, record.alert_error) == ("failed", "receiver down")

        async def active(_m: Any, _r: Any, _ro: Any, rules: Any, _c: Any) -> Any:
            return await rules.active(QUESTION, 1)

        assert with_monitor(database_url, active).id == old.id

    def test_without_an_alerter_the_alert_is_unconfigured(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        prepare(database_url, example_catalog, config())
        seed(database_url, example_catalog, 3000, start=CURRENT_START, backend=broken_backend())
        assert check(database_url, example_catalog).alert == "unconfigured"

    def test_no_rule_means_nothing_to_tighten(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        routing = config()
        seed(database_url, example_catalog, REFERENCE_N)
        seed(
            database_url,
            example_catalog,
            1000,
            start=CURRENT_START,
            backend=FakeBackend(overconfidence=6.0),
        )
        record = check(database_url, example_catalog, routing=routing)
        assert record.status == "drift"
        assert record.action == "skipped"
        assert signals(record)["ece"] == "insufficient"  # no calibrator, and one is required
        assert record.summary["scores"] == "none"
        assert record.reference_end == record.window_start

    def test_raw_scores_when_no_calibrator_is_required(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        routing = parse_routing(
            {
                "questions": {QUESTION: {**POLICY, "require_calibrator": False}},
                "drift": DRIFT,
            }
        )
        seed(database_url, example_catalog, REFERENCE_N)
        seed(database_url, example_catalog, 3000, start=CURRENT_START)
        record = check(database_url, example_catalog, routing=routing)
        assert record.summary["scores"] == "raw confidence"
        # The fake is overconfident (2.0), and raw scores show it.
        assert signals(record)["ece"] == "drift"

    def test_disabled_and_unrouted_questions(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        disabled = config(questions={QUESTION: {"enabled": False}})
        with pytest.raises(SelectionError, match="disabled"):
            check(database_url, example_catalog, routing=disabled)

        async def work(monitor: DriftMonitor, *_: Any) -> Any:
            assert not monitor.monitored(example_catalog.get("ticket.queue"))
            with pytest.raises(SelectionError, match="no routing policy"):
                await monitor.check(example_catalog.get("ticket.queue"), now=CHECK_AT)
            return await monitor.check_all(example_catalog.questions, now=CHECK_AT)

        records = with_monitor(database_url, work, disabled)
        assert [r.question_id for r in records] == ["ticket.handling_minutes"]
        assert records[0].status == "insufficient"
        assert signals(records[0])["label_psi"] == "insufficient"

    def test_a_rule_changed_mid_check_is_not_overwritten(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        old = prepare(database_url, example_catalog, config())

        async def work(_m: Any, refresher: Any, _r: Any, rules: Any, checks: Any) -> Any:
            newer = (await refresher.refresh(example_catalog.get(QUESTION))).rule
            stale = DriftCheckRecord(
                id=uuid.uuid4(),
                question_id=QUESTION,
                question_version=1,
                status="drift",
                checked_at=CHECK_AT,
                window_start=CHECK_AT,
                window_end=CHECK_AT,
                reference_start=CHECK_AT,
                reference_end=CHECK_AT,
                policy={},
                signals=[],
                summary={},
                action="suspended",
                action_reason="",
                previous_rule_id=old.id,
            )
            tightened = type(old)(
                id=uuid.uuid4(),
                question_id=QUESTION,
                question_version=1,
                policy_fingerprint=old.policy_fingerprint,
                policy=old.policy,
                calibrator_id=old.calibrator_id,
                labeled=0,
                eligible=0,
                reason="x",
                evaluation={},
            )
            stored, installed = await checks.add(stale, tightened)
            return newer, stored, installed, await rules.active(QUESTION, 1)

        newer, stored, installed, active = with_monitor(database_url, work)
        assert not installed
        assert stored.action == "skipped"
        assert active.id == newer.id


class _Rules:
    def __init__(self, active: Any) -> None:
        self._active = active

    async def active(self, *_: Any) -> Any:
        return self._active


class _Checks:
    def __init__(self, latest: Any) -> None:
        self._latest = latest

    async def latest(self, *_: Any) -> Any:
        return self._latest


def _rule(auto: float | None, *, check_id: uuid.UUID | None = None, cal: Any = None) -> Any:
    from decision_gate.db.models import RoutingRuleRecord

    return RoutingRuleRecord(
        id=uuid.uuid4(),
        question_id=QUESTION,
        question_version=1,
        policy_fingerprint="f",
        policy={},
        calibrator_id=cal,
        auto_threshold=auto,
        verify_threshold=None,
        labeled=0,
        eligible=0,
        reason="selected",
        evaluation={},
        drift_check_id=check_id,
    )


class TestRespectDrift:
    def respect(self, active: Any, latest: Any, new: Any, *, with_checks: bool = True) -> Any:
        refresher = RoutingRefresher(
            config(),
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            _Rules(active),  # type: ignore[arg-type]
            _Checks(latest) if with_checks else None,  # type: ignore[arg-type]
        )
        run(refresher._respect_drift(PredicateQuestion(id=QUESTION, prompt="?"), new))
        return new

    def test_branches(self) -> None:
        cid = uuid.uuid4()
        drifting = type("C", (), {"status": "drift", "id": cid})()
        ok = type("C", (), {"status": "ok", "id": cid})()
        # No drift repository, no active rule, an untightened rule, or drift cleared: as is.
        assert self.respect(None, drifting, _rule(0.8), with_checks=False).auto_threshold == 0.8
        assert self.respect(None, drifting, _rule(0.8)).auto_threshold == 0.8
        assert self.respect(_rule(0.9), drifting, _rule(0.8)).auto_threshold == 0.8
        assert self.respect(_rule(0.9, check_id=cid), ok, _rule(0.8)).auto_threshold == 0.8
        assert self.respect(_rule(0.9, check_id=cid), None, _rule(0.8)).auto_threshold == 0.8
        # A different calibrator: the floor does not apply.
        other = self.respect(_rule(0.9, check_id=cid, cal=uuid.uuid4()), drifting, _rule(0.8))
        assert (other.auto_threshold, other.drift_check_id) == (0.8, None)
        # The refresh found no threshold itself: it stays off, and remembers the drift.
        off = self.respect(_rule(0.9, check_id=cid), drifting, _rule(None))
        assert (off.auto_threshold, off.drift_check_id, off.reason) == (None, cid, "selected")
        # The refresh is already stricter: kept, with the drift link.
        stricter = self.respect(_rule(0.9, check_id=cid), drifting, _rule(0.95))
        assert (stricter.auto_threshold, stricter.drift_check_id) == (0.95, cid)
        # Looser: raised to the floor.
        raised = self.respect(_rule(0.9, check_id=cid), drifting, _rule(0.8))
        assert raised.auto_threshold == 0.9
        assert raised.reason.startswith("auto threshold kept at 0.9000")


def test_alert_payload_for_recovery() -> None:
    record = DriftCheckRecord(
        id=uuid.uuid4(),
        question_id=QUESTION,
        question_version=2,
        status="ok",
        checked_at=CHECK_AT,
        window_start=CHECK_AT,
        window_end=CHECK_AT,
        reference_start=CHECK_AT,
        reference_end=CHECK_AT,
        signals=[{"name": "ece", "status": "ok", "value": 0.01, "limit": 0.05, "message": "m"}],
        action="none",
        action_reason="no drift",
        previous_rule_id=None,
        rule_id=None,
    )
    payload = alert_payload(EVENT_RECOVERED, record)
    assert payload["text"].startswith("refund.eligible@v2: no drift any more")
    assert payload["signals"][0]["name"] == "ece"
    assert payload["checked_at"] == CHECK_AT.isoformat()


# --- Webhook -----------------------------------------------------------------------------


def alerter_with(handler: Any, **kwargs: Any) -> WebhookAlerter:
    return WebhookAlerter(
        "https://alerts.example/hook",
        transport=httpx.MockTransport(handler),
        backoff_s=0.0,
        **kwargs,
    )


class TestWebhook:
    def test_signed_delivery(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(204)

        run(alerter_with(handler, secret="s3cret").send("drift.detected", {"a": 1}))
        request = seen[0]
        body = request.content
        expected = hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        assert request.headers["X-Dgate-Signature"] == f"sha256={expected}"
        assert request.headers["X-Dgate-Signature"] == sign("s3cret", body)
        assert request.headers["X-Dgate-Event"] == "drift.detected"
        assert json.loads(body) == {"a": 1}

    def test_unsigned_without_a_secret(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200)

        run(alerter_with(handler).send("e", {}))
        assert "X-Dgate-Signature" not in seen[0].headers

    def test_retries_server_errors_with_one_delivery_id(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(503 if len(seen) < 3 else 200)

        run(alerter_with(handler, attempts=3).send("e", {}))
        assert len(seen) == 3
        assert len({r.headers["X-Dgate-Delivery"] for r in seen}) == 1

    def test_client_errors_are_not_retried(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(404)

        with pytest.raises(AlertError, match="after 1 attempt"):
            run(alerter_with(handler, attempts=3).send("e", {}))
        assert len(seen) == 1

    def test_connection_errors_exhaust_the_attempts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(AlertError, match=r"after 2 attempt.*ConnectError"):
            run(alerter_with(handler, attempts=2).send("e", {}))

    def test_rejects_zero_attempts(self) -> None:
        with pytest.raises(ValueError, match="attempts"):
            WebhookAlerter("https://x", attempts=0)

    def test_settings_build_the_alerter(self) -> None:
        assert Settings(drift_webhook_url="").drift_alerter() is None
        alerter = Settings(
            drift_webhook_url="https://hooks.example/x", drift_webhook_secret="k"
        ).drift_alerter()
        assert alerter is not None
        assert alerter.url == "https://hooks.example/x"
        assert Settings(drift_webhook_url="https://h").drift_alerter() is not None


# --- API ---------------------------------------------------------------------------------


@pytest.fixture
def api(database_url: str, example_catalog: QuestionCatalog) -> Iterator[TestClient]:
    app = create_app(
        Settings(database_url=database_url, routing_cache_ttl_s=0),
        catalog=example_catalog,
        backend=FakeBackend(),
        routing=config(),
        clock=lambda: CHECK_AT,
        alerter=RecordingAlerter(),
    )
    with TestClient(app) as client:
        yield client


class TestApi:
    def test_check_status_history_and_rules(
        self, api: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        status = api.get("/v1/drift/status").json()["questions"]
        assert {q["question"]: q["latest"] for q in status} == {
            "refund.eligible": None,
            "ticket.handling_minutes": None,
        }
        prepare(database_url, example_catalog, config())
        seed(database_url, example_catalog, 3000, start=CURRENT_START, backend=broken_backend())

        response = api.post("/v1/drift/check", json={"question": QUESTION})
        assert response.status_code == 200, response.text
        [body] = response.json()["checks"]
        assert body["status"] == "drift"
        assert body["alert"] == "sent"
        assert body["checked_at"].startswith("2026-10-01T03:00:00")

        all_checks = api.post("/v1/drift/check", json={}).json()["checks"]
        assert {c["question"] for c in all_checks} == {QUESTION, "ticket.handling_minutes"}

        listed = api.get("/v1/drift/checks", params={"question": QUESTION, "version": 1}).json()[
            "checks"
        ]
        assert len(listed) == 2
        assert api.get("/v1/drift/checks", params={"status": "drift"}).json()["checks"]
        one = api.get(f"/v1/drift/checks/{body['id']}").json()
        assert one["rule_id"] == body["rule_id"]
        latest = api.get("/v1/drift/status").json()["questions"]
        assert any(q["latest"] and q["latest"]["status"] == "drift" for q in latest)

        rules = api.get("/v1/routing/rules", params={"question": QUESTION}).json()["rules"]
        assert rules[0]["drift_check_id"] == body["id"]

    def test_errors(self, api: TestClient) -> None:
        assert api.post("/v1/drift/check", json={"question": "nope"}).status_code == 404
        response = api.post("/v1/drift/check", json={"question": "ticket.queue"})
        assert response.status_code == 404
        assert "no routing policy" in response.json()["detail"]
        assert api.get(f"/v1/drift/checks/{uuid.uuid4()}").status_code == 404
        assert api.post("/v1/drift/check", json={"bogus": 1}).status_code == 422

    def test_selection_errors_are_422(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(*_: Any, **__: Any) -> Any:
            raise SelectionError("cannot analyse")

        monkeypatch.setattr(DriftMonitor, "check_all", boom)
        response = api.post("/v1/drift/check", json={})
        assert response.status_code == 422
        assert response.json()["detail"] == "cannot analyse"


# --- CLI ---------------------------------------------------------------------------------


def write_routing(tmp_path: Path, **drift: Any) -> Path:
    path = tmp_path / "routing.json"
    path.write_text(
        json.dumps(
            {
                "defaults": {"min_labeled": 200},
                "questions": {QUESTION: POLICY},
                "drift": {**DRIFT, **drift},
            }
        )
    )
    return path


class TestCli:
    def args(self, database_url: str, routing: Path, *extra: str) -> list[str]:
        return [
            "drift",
            "check",
            *extra,
            "--routing",
            str(routing),
            "-q",
            str(EXAMPLES / "questions.yaml"),
            "--url",
            database_url,
        ]

    def test_check_exit_codes_and_list(
        self,
        database_url: str,
        example_catalog: QuestionCatalog,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("DGATE_DRIFT_WEBHOOK_URL", raising=False)
        routing = write_routing(tmp_path)
        prepare(database_url, example_catalog, config())
        seed(database_url, example_catalog, 3000, start=CURRENT_START)
        result = runner.invoke(
            cli, self.args(database_url, routing, QUESTION, "--at", CHECK_AT.isoformat())
        )
        assert result.exit_code == 0, result.output
        assert "refund.eligible@v1: OK" in result.output
        assert "auto_error" in result.output

        seed(database_url, example_catalog, 3000, start=4 * 3600, backend=broken_backend())
        later = (BASE + timedelta(hours=5)).replace(tzinfo=None).isoformat()
        result = runner.invoke(cli, self.args(database_url, routing, "--at", later))
        assert result.exit_code == 3, result.output
        assert "DRIFT" in result.output
        assert "alert drift.detected unconfigured" in result.output
        assert "-> auto" in result.output

        listed = runner.invoke(cli, ["drift", "list", "--url", database_url])
        assert listed.exit_code == 0
        assert listed.output.count("refund.eligible@v1") == 2
        verbose = runner.invoke(
            cli, ["drift", "list", "-v", "--status", "drift", "--url", database_url]
        )
        assert "score_psi" in verbose.output

    def test_failed_webhook_is_shown(
        self,
        database_url: str,
        example_catalog: QuestionCatalog,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DGATE_DRIFT_WEBHOOK_URL", "http://127.0.0.1:9/hook")
        monkeypatch.setenv("DGATE_DRIFT_WEBHOOK_ATTEMPTS", "1")
        monkeypatch.setenv("DGATE_DRIFT_WEBHOOK_TIMEOUT_S", "2")
        prepare(database_url, example_catalog, config())
        seed(database_url, example_catalog, 3000, start=CURRENT_START, backend=broken_backend())
        result = runner.invoke(
            cli,
            self.args(database_url, write_routing(tmp_path), "--at", CHECK_AT.isoformat()),
        )
        assert result.exit_code == 3, result.output
        assert "alert drift.detected failed" in result.output
        assert "alert error: webhook delivery failed after 1 attempt(s)" in result.output

    def test_cli_errors(self, database_url: str, tmp_path: Path) -> None:
        routing = write_routing(tmp_path)
        bad = runner.invoke(cli, self.args(database_url, routing, "--at", "yesterday"))
        assert bad.exit_code == 2
        assert "ISO 8601" in bad.output
        disabled = write_routing(tmp_path, enabled=False)
        result = runner.invoke(cli, self.args(database_url, disabled, QUESTION))
        assert result.exit_code == 1
        assert "disabled" in result.output
        assert runner.invoke(cli, self.args(database_url, disabled)).output.startswith(
            "no monitored questions"
        )
        empty = runner.invoke(cli, ["drift", "list", "--url", database_url])
        assert empty.output.strip() == "no drift checks"
        wrong = runner.invoke(cli, ["drift", "list", "--status", "meh", "--url", database_url])
        assert wrong.exit_code == 2
