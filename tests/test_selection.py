"""Threshold selection and conformal sets over stored decisions: analyzer, HTTP and CLI."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_gate.app import create_app
from decision_gate.backends import FakeBackend
from decision_gate.cli import app as cli
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import CalibratorRecord, DecisionRecord, OutcomeRecord
from decision_gate.db.repository import CalibratorRepository, LabeledRow, OutcomeRepository
from decision_gate.questions import QuestionCatalog
from decision_gate.refit import RefitJob
from decision_gate.selection import (
    GateAnalyzer,
    SelectionError,
    calibrated_distributions,
    split_point,
)
from decision_gate.thresholds import select_threshold
from tests.test_refit import BASE, _insert, run, seed_fake

runner = CliRunner()


async def _analyze(database_url: str, work: Any) -> Any:
    engine = make_engine(database_url)
    try:
        sessions = make_sessionmaker(engine)
        outcomes, calibrators = OutcomeRepository(sessions), CalibratorRepository(sessions)
        return await work(GateAnalyzer(outcomes, calibrators), RefitJob(outcomes, calibrators))
    finally:
        await engine.dispose()


def analyze(database_url: str, work: Any) -> Any:
    return run(_analyze(database_url, work))


def refit(database_url: str, question: str) -> CalibratorRecord:
    async def work(_: GateAnalyzer, job: RefitJob) -> Any:
        return await job.refit(question, 1)

    result = analyze(database_url, work)
    assert result.outcome == "activated", result.reason
    record: CalibratorRecord = result.record
    return record


def insert_calibrator(database_url: str, question: str, method: str, params: Any) -> None:
    record = CalibratorRecord(
        question_id=question,
        question_version=1,
        version=1,
        method=method,
        params=params,
        status="active",
        reason="test",
        train_count=1,
        holdout_count=1,
        trained_from=BASE - timedelta(days=2),
        trained_until=BASE - timedelta(days=1),
        evaluation={},
    )
    run(_insert(database_url, [record]))


def insert_decision(
    database_url: str, question: str, answer: Any, outcome: Any, probabilities: Any
) -> None:
    decision_id = uuid.uuid4()
    decision = DecisionRecord(
        id=decision_id,
        question_id=question,
        question_version=1,
        input_sha256="0" * 64,
        input=None,
        answer=answer,
        confidence=0.7,
        probabilities=probabilities,
        backend="fake",
        model_version="m",
        latency_ms=0.1,
        created_at=BASE,
    )
    outcome_record = OutcomeRecord(
        decision_id=decision_id,
        outcome=outcome,
        correct=answer == outcome,
        source="test",
        observed_at=BASE,
    )
    run(_insert(database_url, [decision]))
    run(_insert(database_url, [outcome_record]))


@pytest.fixture
def client(database_url: str, example_catalog: QuestionCatalog) -> Iterator[TestClient]:
    app = create_app(
        Settings(database_url=database_url), catalog=example_catalog, backend=FakeBackend()
    )
    with TestClient(app) as test_client:
        yield test_client


def test_split_point() -> None:
    assert split_point(10, 0.3) == 7
    assert split_point(10, 0.0) == 10
    assert split_point(1, 0.3) == 0
    assert split_point(0, 0.3) == 0
    with pytest.raises(SelectionError, match="holdout"):
        split_point(10, 1.0)


def test_calibrated_distributions_need_a_distribution() -> None:
    row = LabeledRow("q", 1, "fake", "m", 0.5, True, None, "a", "a")
    with pytest.raises(SelectionError, match="distribution on every decision"):
        calibrated_distributions([row], None)


class TestThresholdsApi:
    def test_raw_scores_match_direct_selection(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 1500)
        response = client.post(
            "/v1/thresholds/select",
            json={"question": "refund.eligible", "target_precision": 0.9, "curve_points": 5},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        basis = body["basis"]
        assert basis["labeled"] == 1500
        assert (basis["selection_count"], basis["holdout_count"]) == (1050, 450)
        assert (basis["score_source"], basis["calibrator_id"]) == ("raw", None)
        assert basis["excluded_training"] == 0

        async def rows(analyzer: GateAnalyzer, _: RefitJob) -> list[LabeledRow]:
            return await analyzer._outcomes.labeled_rows(question_id="refund.eligible")

        labeled = analyze(database_url, rows)[:1050]
        direct = select_threshold(
            [r.confidence for r in labeled], [r.correct for r in labeled], target_precision=0.9
        )
        assert direct.threshold is not None
        assert body["threshold"] == direct.threshold
        assert body["selected"]["accepted"] == direct.selected.accepted  # type: ignore[union-attr]
        assert body["selected"]["lower_bound"] >= 0.9
        assert body["target_precision"] == 0.9
        assert body["max_error_rate"] == pytest.approx(0.1)
        assert body["bound"] == "clopper-pearson"
        assert body["tested"] >= 1
        assert body["candidates"] >= body["tested"]
        assert 5 <= len(body["curve"]) <= 7
        holdout = body["holdout"]
        assert holdout["total"] == 450
        assert 0 < holdout["accepted"] < 450
        assert holdout["precision"] > 0.8
        assert "accept score >=" in body["reason"]

    def test_with_the_active_calibrator(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 1500)
        record = refit(database_url, "refund.eligible")
        assert record.method == "temperature"
        body = client.post(
            "/v1/thresholds/select",
            json={"question": "refund.eligible", "max_error_rate": 0.1, "min_accepted": 50},
        ).json()
        basis = body["basis"]
        # The refit trained on the oldest 70% (1050); only the 450 after it are eligible.
        assert basis["excluded_training"] == 1050
        assert (basis["selection_count"], basis["holdout_count"]) == (315, 135)
        assert basis["score_source"] == "calibrator v1 (temperature)"
        assert basis["calibrator_id"] == str(record.id)
        assert basis["calibrator_version"] == 1
        assert body["target_precision"] == pytest.approx(0.9)
        assert 1 <= len(body["curve"]) <= 12  # 10 by default, plus the decision points
        raw = client.post(
            "/v1/thresholds/select",
            json={"question": "refund.eligible", "use_calibrator": False, "holdout": 0},
        ).json()
        assert raw["basis"]["score_source"] == "raw"
        assert raw["basis"]["excluded_training"] == 0
        assert raw["holdout"] is None

    def test_nothing_to_select_on(self, client: TestClient) -> None:
        body = client.post("/v1/thresholds/select", json={"question": "ticket.queue"}).json()
        assert body["threshold"] is None
        assert body["selected"] is None
        assert body["candidates"] == 0
        assert body["reason"] == "no labeled decisions to select on"

    def test_errors(self, client: TestClient, database_url: str) -> None:
        assert client.post("/v1/thresholds/select", json={"question": "nope"}).status_code == 404
        both = client.post(
            "/v1/thresholds/select",
            json={"question": "ticket.queue", "target_precision": 0.9, "max_error_rate": 0.1},
        )
        assert both.status_code == 422
        assert "not both" in both.text
        for payload in (
            {"question": "ticket.queue", "holdout": 1},
            {"question": "ticket.queue", "bound": "normal"},
            {"question": "ticket.queue", "target_precision": 1},
        ):
            assert client.post("/v1/thresholds/select", json=payload).status_code == 422
        # An active temperature scaler cannot score a decision without a distribution.
        insert_calibrator(database_url, "ticket.queue", "temperature", {"temperature": 2.0})
        insert_decision(database_url, "ticket.queue", "billing", "billing", None)
        bad = client.post("/v1/thresholds/select", json={"question": "ticket.queue", "holdout": 0})
        assert bad.status_code == 422
        assert "cannot score these decisions" in bad.json()["detail"]

    def test_unloadable_calibrator(self, client: TestClient, database_url: str) -> None:
        insert_calibrator(database_url, "refund.eligible", "platt", {"slope": 1})
        bad = client.post("/v1/thresholds/select", json={"question": "refund.eligible"})
        assert bad.status_code == 422
        assert "cannot be loaded" in bad.json()["detail"]


class TestConformalApi:
    def test_choice_question_sets(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.queue", 1000)
        response = client.post("/v1/conformal/fit", json={"question": "ticket.queue", "alpha": 0.1})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["basis"]["score_source"] == "raw"
        assert (body["basis"]["selection_count"], body["basis"]["holdout_count"]) == (700, 300)
        assert body["labels"] == list(example_catalog.get("ticket.queue").options)  # type: ignore[union-attr]
        assert body["calibration_count"] == 700
        assert body["trivial"] is False
        assert 0 < body["quantile"] < 1
        assert body["target_coverage"] == pytest.approx(0.9)
        assert body["params"]["quantile"] == body["quantile"]
        holdout = body["holdout"]
        assert holdout["count"] == 300
        assert holdout["coverage"] >= 0.85
        assert sum(holdout["size_counts"].values()) == 300
        assert "cover the true label with probability >= 90%" in body["reason"]
        aps = client.post(
            "/v1/conformal/fit",
            json={"question": "ticket.queue", "method": "aps", "holdout": 0},
        ).json()
        assert aps["method"] == "aps"
        assert aps["holdout"] is None
        assert aps["calibration_count"] == 1000

    def test_temperature_scaled_predicate_sets(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 1500)
        record = refit(database_url, "refund.eligible")
        body = client.post("/v1/conformal/fit", json={"question": "refund.eligible"}).json()
        assert body["labels"] == ["false", "true"]
        assert body["basis"]["calibrator_id"] == str(record.id)
        assert body["basis"]["excluded_training"] == 1050
        assert body["calibration_count"] == 315

    def test_platt_calibrator_leaves_distributions_alone(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.queue", 400)
        insert_calibrator(database_url, "ticket.queue", "platt", {"a": 0.5, "b": 0.0})
        body = client.post("/v1/conformal/fit", json={"question": "ticket.queue"}).json()
        assert body["basis"]["score_source"] == "raw"
        assert body["basis"]["calibrator_id"] is None

    def test_trivial_and_empty(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        empty = client.post("/v1/conformal/fit", json={"question": "ticket.queue"}).json()
        assert empty["trivial"] is True
        assert empty["params"] is None
        assert empty["reason"] == "no labeled decisions"
        seed_fake(database_url, example_catalog, "ticket.queue", 8)
        few = client.post(
            "/v1/conformal/fit", json={"question": "ticket.queue", "holdout": 0}
        ).json()
        assert few["trivial"] is True
        assert few["quantile"] is None
        assert few["params"]["quantile"] is None
        assert "need 9" in few["reason"]

    def test_errors(self, client: TestClient, database_url: str) -> None:
        assert client.post("/v1/conformal/fit", json={"question": "nope"}).status_code == 404
        score = client.post("/v1/conformal/fit", json={"question": "ticket.handling_minutes"})
        assert score.status_code == 422
        assert "score questions" in score.json()["detail"]
        for payload in (
            {"question": "ticket.queue", "alpha": 0},
            {"question": "ticket.queue", "method": "raps"},
        ):
            assert client.post("/v1/conformal/fit", json=payload).status_code == 422
        insert_decision(database_url, "ticket.queue", "billing", "refunds", {"billing": 1.0})
        bad = client.post("/v1/conformal/fit", json={"question": "ticket.queue", "holdout": 0})
        assert bad.status_code == 422
        assert "cannot build prediction sets" in bad.json()["detail"]

    def test_analyzer_validates_alpha(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        async def work(analyzer: GateAnalyzer, _: RefitJob) -> Any:
            return await analyzer.conformal(example_catalog.get("ticket.queue"), alpha=1.0)

        with pytest.raises(SelectionError, match="alpha"):
            analyze(database_url, work)

    def test_analyzer_validates_threshold_options(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 20)

        async def work(analyzer: GateAnalyzer, _: RefitJob) -> Any:
            question = example_catalog.get("refund.eligible")
            return await analyzer.thresholds(question, target_precision=1.0)

        with pytest.raises(SelectionError, match="target_precision"):
            analyze(database_url, work)


class TestCli:
    @pytest.fixture(autouse=True)
    def _catalog(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DGATE_QUESTIONS_FILE", "examples/questions.yaml")

    def test_thresholds_select(self, database_url: str, example_catalog: QuestionCatalog) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 1500)
        args = ["--url", database_url]
        text = runner.invoke(
            cli,
            ["thresholds", "select", "refund.eligible", "--target-precision", "0.9", "--curve",
             "4", *args],
        )  # fmt: skip
        assert text.exit_code == 0, text.output
        lines = text.output.splitlines()
        assert lines[0].startswith("refund.eligible@v1: 1500 labeled, scores from raw")
        assert lines[1].startswith("accept score >=")
        assert lines[2].startswith("held out: ")
        assert "threshold" in lines[3]
        assert any(line.endswith("<- selected") for line in lines)
        as_json = runner.invoke(
            cli, ["thresholds", "select", "refund.eligible", "--max-error-rate", "0.1", "--json",
                  *args]
        )  # fmt: skip
        assert as_json.exit_code == 0, as_json.output
        payload = json.loads(as_json.output)
        assert payload["target_precision"] == pytest.approx(0.9)
        assert payload["curve"]

    def test_thresholds_with_calibrator_and_nothing_selected(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.handling_minutes", 1500)
        refit(database_url, "ticket.handling_minutes")
        result = runner.invoke(
            cli,
            ["thresholds", "select", "ticket.handling_minutes", "--target-precision", "0.999",
             "--min-accepted", "20", "--url", database_url],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        assert "skipped (calibrator training data)" in result.output
        assert "even the highest threshold" in result.output
        assert "held out:" not in result.output

    def test_thresholds_errors(self, database_url: str) -> None:
        args = ["--url", database_url]
        both = runner.invoke(
            cli,
            ["thresholds", "select", "ticket.queue", "--target-precision", "0.9",
             "--max-error-rate", "0.1", *args],
        )  # fmt: skip
        assert both.exit_code == 2
        assert "not both" in both.output
        bound = runner.invoke(cli, ["thresholds", "select", "ticket.queue", "--bound", "x", *args])
        assert bound.exit_code == 2
        unknown = runner.invoke(cli, ["thresholds", "select", "nope", *args])
        assert unknown.exit_code == 1
        assert "unknown question" in unknown.output
        insert_calibrator(database_url, "refund.eligible", "platt", {"slope": 1})
        broken = runner.invoke(cli, ["thresholds", "select", "refund.eligible", *args])
        assert broken.exit_code == 1
        assert "cannot be loaded" in broken.output

    def test_conformal_fit(self, database_url: str, example_catalog: QuestionCatalog) -> None:
        seed_fake(database_url, example_catalog, "ticket.queue", 600)
        args = ["--url", database_url]
        text = runner.invoke(cli, ["conformal", "fit", "ticket.queue", "--alpha", "0.2", *args])
        assert text.exit_code == 0, text.output
        assert "lac sets with quantile" in text.output
        assert "held out: coverage" in text.output
        as_json = runner.invoke(
            cli, ["conformal", "fit", "ticket.queue", "--method", "aps", "--json", *args]
        )
        assert as_json.exit_code == 0, as_json.output
        assert json.loads(as_json.output)["method"] == "aps"
        no_holdout = runner.invoke(
            cli, ["conformal", "fit", "ticket.queue", "--holdout", "0", "--raw", *args]
        )
        assert no_holdout.exit_code == 0
        assert "held out:" not in no_holdout.output

    def test_conformal_errors(self, database_url: str) -> None:
        args = ["--url", database_url]
        bad_method = runner.invoke(
            cli, ["conformal", "fit", "ticket.queue", "--method", "x", *args]
        )
        assert bad_method.exit_code == 2
        score = runner.invoke(cli, ["conformal", "fit", "ticket.handling_minutes", *args])
        assert score.exit_code == 1
        assert "score questions" in score.output


def test_fake_gate_runs_without_a_database() -> None:
    result = runner.invoke(
        cli, ["fake-gate", "-q", "examples/questions.yaml", "-n", "3000", "--target-precision",
              "0.9"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "refund.eligible@v1  (calibrator: temperature" in result.output
    assert "ticket.handling_minutes@v1  (calibrator: platt" in result.output
    assert "  calibrated  threshold" in result.output
    assert "  lac         sets for 90% coverage" in result.output
    # Score questions have no label set, so no conformal lines follow them.
    tail = result.output.split("ticket.handling_minutes@v1")[1]
    assert "sets for" not in tail


def test_fake_gate_reports_when_nothing_qualifies() -> None:
    result = runner.invoke(
        cli, ["fake-gate", "-q", "examples/questions.yaml", "-n", "400", "--target-precision",
              "0.999"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "no threshold:" in result.output


def test_fake_gate_bad_catalog(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("questions: [{id: x}]\n")
    result = runner.invoke(cli, ["fake-gate", "-q", str(path)])
    assert result.exit_code == 1
    assert "error:" in result.output
