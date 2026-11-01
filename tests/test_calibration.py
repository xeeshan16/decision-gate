"""Calibration reports: grouping logic, the HTTP endpoint and the CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_gate.app import create_app
from decision_gate.backends import FakeBackend
from decision_gate.calibration import build_reports
from decision_gate.cli import app as cli
from decision_gate.config import Settings
from decision_gate.db.repository import LabeledRow
from decision_gate.metrics import calibration_metrics
from decision_gate.questions import QuestionCatalog

runner = CliRunner()


def row(
    confidence: float,
    correct: bool,
    *,
    question: str = "q",
    version: int = 1,
    backend: str = "fake",
    model: str = "m1",
    probabilities: dict[str, float] | None = None,
    outcome: Any = True,
) -> LabeledRow:
    return LabeledRow(
        question_id=question,
        question_version=version,
        backend=backend,
        model_version=model,
        confidence=confidence,
        correct=correct,
        probabilities=probabilities,
        outcome=outcome,
    )


class TestBuildReports:
    def test_one_report_per_question_version_by_default(self) -> None:
        rows = [
            row(0.9, True, version=2),
            row(0.8, False, model="m2"),
            row(0.7, True),
            row(0.6, True, question="a"),
        ]
        reports = build_reports(rows, n_bins=5)
        assert [(r.question_key, r.metrics.count) for r in reports] == [
            ("a@v1", 1),
            ("q@v1", 2),
            ("q@v2", 1),
        ]
        assert all(r.backend is None and r.model_version is None for r in reports)
        assert reports[1].metrics == calibration_metrics([0.8, 0.7], [False, True], n_bins=5)

    def test_split_by_backend_and_model_version(self) -> None:
        rows = [
            row(0.9, True, model="m2"),
            row(0.8, False, backend="http"),
            row(0.7, True),
            row(0.6, False, model="m2"),
        ]
        by_model = build_reports(rows, group_by=["model_version"])
        assert [(r.backend, r.model_version, r.metrics.count) for r in by_model] == [
            (None, "m1", 2),
            (None, "m2", 2),
        ]
        both = build_reports(rows, group_by=["backend", "model_version"])
        assert [(r.backend, r.model_version, r.metrics.count) for r in both] == [
            ("fake", "m1", 1),
            ("fake", "m2", 2),
            ("http", "m1", 1),
        ]

    def test_min_count_drops_small_groups(self) -> None:
        rows = [row(0.9, True), row(0.8, True), row(0.7, True, question="a")]
        assert [r.question_key for r in build_reports(rows, min_count=2)] == ["q@v1"]

    def test_multiclass_scores_use_the_full_distribution(self) -> None:
        rows = [
            row(0.7, True, probabilities={"a": 0.7, "b": 0.2, "c": 0.1}, outcome="a"),
            # The true label 'c' carries 0.3; key order in the mapping does not matter.
            row(0.6, False, probabilities={"c": 0.3, "b": 0.6, "a": 0.1}, outcome="c"),
        ]
        [report] = build_reports(rows)
        assert report.metrics.multiclass_brier == pytest.approx(0.5)  # see test_metrics

    def test_predicate_outcomes_map_to_true_false_keys(self) -> None:
        rows = [
            row(0.8, True, probabilities={"true": 0.8, "false": 0.2}, outcome=True),
            row(0.9, False, probabilities={"true": 0.1, "false": 0.9}, outcome=True),
        ]
        [report] = build_reports(rows)
        # Row 1: 2 * 0.2^2 = 0.08. Row 2: 2 * 0.9^2 = 1.62. Mean 0.85.
        assert report.metrics.multiclass_brier == pytest.approx(0.85)

    def test_a_label_the_backend_never_mentioned_counts_as_zero(self) -> None:
        rows = [row(0.9, False, probabilities={"a": 0.9, "b": 0.1}, outcome="z")]
        [report] = build_reports(rows)
        # 0.9^2 + 0.1^2 + 1^2
        assert report.metrics.multiclass_brier == pytest.approx(1.82)
        assert report.metrics.multiclass_log_loss is not None
        assert report.metrics.multiclass_log_loss > 20  # log(1e-12) territory

    def test_no_distribution_scores_when_any_row_lacks_one(self) -> None:
        rows = [
            row(0.7, True, probabilities={"a": 0.7, "b": 0.3}, outcome="a"),
            row(0.6, True, outcome="a"),
        ]
        assert build_reports(rows)[0].metrics.multiclass_brier is None
        scores = [row(0.5, True, outcome=12.0, probabilities={"a": 1.0})]
        assert build_reports(scores)[0].metrics.multiclass_brier is None

    def test_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="cannot group by"):
            build_reports([row(0.5, True)], group_by=["region"])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="min_count"):
            build_reports([row(0.5, True)], min_count=0)

    def test_no_rows_no_reports(self) -> None:
        assert build_reports([]) == []


def decide(client: TestClient, question: str, data: dict[str, Any]) -> dict[str, Any]:
    response = client.post("/v1/decide", json={"question": question, "input": data})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def label(client: TestClient, decision: dict[str, Any], outcome: Any) -> None:
    response = client.post("/v1/outcomes", json={"decision_id": decision["id"], "outcome": outcome})
    assert response.status_code == 201, response.text


def wrong(decision: dict[str, Any]) -> Any:
    answer = decision["answer"]
    if isinstance(answer, bool):
        return not answer
    if isinstance(answer, str):
        return "other" if answer != "other" else "billing"
    return answer + 100 if answer + 100 <= 240 else answer - 100


@pytest.fixture
def labeled_db(
    database_url: str, example_catalog: QuestionCatalog
) -> Iterator[tuple[TestClient, dict[str, list[dict[str, Any]]]]]:
    """Decisions from two fake model versions, every third one answered wrongly."""
    settings = Settings(database_url=database_url)
    decisions: dict[str, list[dict[str, Any]]] = {}
    old = create_app(settings, catalog=example_catalog, backend=FakeBackend(model_version="m1"))
    with TestClient(old) as client:
        for question in ("refund.eligible", "ticket.queue", "ticket.handling_minutes"):
            for i in range(12):
                decision = decide(client, question, {"item": i})
                label(client, decision, wrong(decision) if i % 3 == 0 else decision["answer"])
                decisions.setdefault(question, []).append(decision)
        unlabeled = decide(client, "refund.eligible", {"item": "no label"})
        assert unlabeled["outcome"] is None
    new = create_app(settings, catalog=example_catalog, backend=FakeBackend(model_version="m2"))
    with TestClient(new) as client:
        for i in range(6):
            decision = decide(client, "refund.eligible", {"item": f"new-{i}"})
            label(client, decision, decision["answer"])
            decisions.setdefault("refund.eligible:m2", []).append(decision)
        yield client, decisions


def test_endpoint_matches_the_metrics_module(
    labeled_db: tuple[TestClient, dict[str, list[dict[str, Any]]]],
) -> None:
    client, decisions = labeled_db
    response = client.get("/v1/calibration/metrics", params={"bins": 5})
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["n_bins"], body["strategy"], body["group_by"]) == (5, "uniform", [])
    by_question = {r["question"]: r for r in body["reports"]}
    assert list(by_question) == ["refund.eligible", "ticket.handling_minutes", "ticket.queue"]

    refund = by_question["refund.eligible"]
    assert refund["count"] == 18  # the unlabeled decision is left out
    expected_rows = decisions["refund.eligible"] + decisions["refund.eligible:m2"]
    expected = calibration_metrics(
        [d["confidence"] for d in expected_rows],
        [i % 3 != 0 for i in range(12)] + [True] * 6,
        n_bins=5,
    )
    assert refund["ece"] == pytest.approx(expected.ece)
    assert refund["brier"] == pytest.approx(expected.brier)
    assert refund["log_loss"] == pytest.approx(expected.log_loss)
    assert refund["accuracy"] == pytest.approx(14 / 18)
    assert refund["overconfidence"] == pytest.approx(expected.overconfidence)
    assert [b["count"] for b in refund["bins"]] == [b.count for b in expected.bins]
    assert refund["multiclass_brier"] is not None
    assert refund["ece_interval"] is None

    assert by_question["ticket.queue"]["multiclass_log_loss"] is not None
    assert by_question["ticket.handling_minutes"]["multiclass_brier"] is None
    assert by_question["ticket.handling_minutes"]["accuracy"] == pytest.approx(8 / 12)


def test_endpoint_filters_groups_and_bootstraps(
    labeled_db: tuple[TestClient, dict[str, list[dict[str, Any]]]],
) -> None:
    client, _ = labeled_db
    params: dict[str, Any] = {
        "question": "refund.eligible",
        "version": 1,
        "group_by": ["model_version", "backend", "model_version"],
        "strategy": "quantile",
        "bins": 3,
        "bootstrap": 20,
    }
    body = client.get("/v1/calibration/metrics", params=params).json()
    assert body["group_by"] == ["model_version", "backend"]
    reports = body["reports"]
    assert [(r["backend"], r["model_version"], r["count"]) for r in reports] == [
        ("fake", "m1", 12),
        ("fake", "m2", 6),
    ]
    assert all(len(r["bins"]) == 3 for r in reports)
    assert all(r["ece_interval"][0] <= r["ece_interval"][1] for r in reports)

    skipped = client.get(
        "/v1/calibration/metrics",
        params={"question": "refund.eligible", "group_by": "model_version", "min_count": 10},
    ).json()
    assert [r["model_version"] for r in skipped["reports"]] == ["m1"]


def test_endpoint_time_window(
    labeled_db: tuple[TestClient, dict[str, list[dict[str, Any]]]],
) -> None:
    client, _ = labeled_db
    now = datetime.now(UTC)
    future = client.get(
        "/v1/calibration/metrics", params={"since": (now + timedelta(hours=1)).isoformat()}
    ).json()
    assert future["reports"] == []
    window = {
        "since": (now - timedelta(hours=1)).isoformat(),
        "until": (now + timedelta(hours=1)).isoformat(),
    }
    current = client.get("/v1/calibration/metrics", params=window).json()
    assert sum(r["count"] for r in current["reports"]) == 18 + 12 + 12


def test_endpoint_validation(
    labeled_db: tuple[TestClient, dict[str, list[dict[str, Any]]]],
) -> None:
    client, _ = labeled_db
    for params in ({"group_by": "region"}, {"bins": 0}, {"strategy": "log"}, {"bootstrap": -1}):
        assert client.get("/v1/calibration/metrics", params=params).status_code == 422
    unknown = client.get("/v1/calibration/metrics", params={"question": "nope"})
    assert unknown.json()["reports"] == []


class TestCli:
    def test_report_table_and_bins(
        self, labeled_db: tuple[TestClient, dict[str, list[dict[str, Any]]]], database_url: str
    ) -> None:
        args = ["calibration", "report", "--url", database_url, "--bins", "5"]
        result = runner.invoke(cli, [*args, "--by", "model_version", "--show-bins"])
        assert result.exit_code == 0, result.output
        lines = result.output.splitlines()
        assert lines[0].split()[:3] == ["question", "model_version", "n"]
        table = [line.split() for line in lines[1:5]]
        assert [(cells[0], cells[1], cells[2]) for cells in table] == [
            ("refund.eligible@v1", "m1", "12"),
            ("refund.eligible@v1", "m2", "6"),
            ("ticket.handling_minutes@v1", "m1", "12"),
            ("ticket.queue@v1", "m1", "12"),
        ]
        assert "refund.eligible@v1 m1: reliability (uniform bins)" in result.output
        assert "0.00-0.20" in result.output

    def test_report_json_with_bootstrap(
        self, labeled_db: tuple[TestClient, dict[str, list[dict[str, Any]]]], database_url: str
    ) -> None:
        args = ["calibration", "report", "--url", database_url, "--question", "ticket.queue"]
        result = runner.invoke(
            cli, [*args, "--json", "--bootstrap", "10", "--strategy", "quantile"]
        )
        assert result.exit_code == 0, result.output
        body = json.loads(result.output)
        assert body["group_by"] == []
        [report] = body["reports"]
        assert report["count"] == 12
        assert len(report["ece_interval"]) == 2

    def test_report_table_shows_the_interval(
        self, labeled_db: tuple[TestClient, dict[str, list[dict[str, Any]]]], database_url: str
    ) -> None:
        args = ["calibration", "report", "--url", database_url, "--bootstrap", "10"]
        result = runner.invoke(cli, [*args, "--version", "1", "--by", "backend", "--by", "backend"])
        assert result.exit_code == 0, result.output
        first = result.output.splitlines()[1].split()
        assert first[:3] == ["refund.eligible@v1", "fake", "18"]
        low, high = first[6].split("-")
        assert float(low) <= float(high)

    def test_report_window_and_empty(
        self, labeled_db: tuple[TestClient, dict[str, list[dict[str, Any]]]], database_url: str
    ) -> None:
        later = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
        args = ["calibration", "report", "--url", database_url, "--since", later]
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
        assert result.output.strip() == "no labeled decisions match"
        earlier = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
        args = ["calibration", "report", "--url", database_url, "--since", earlier]
        result = runner.invoke(cli, [*args, "--until", later, "--min-count", "13"])
        assert result.output.splitlines()[1].split()[:2] == ["refund.eligible@v1", "18"]
        assert len(result.output.splitlines()) == 2

    @pytest.mark.parametrize(
        "extra", [["--by", "region"], ["--strategy", "log"]], ids=["by", "strategy"]
    )
    def test_bad_options(self, sqlite_url: str, extra: list[str]) -> None:
        result = runner.invoke(cli, ["calibration", "report", "--url", sqlite_url, *extra])
        assert result.exit_code == 2
