"""The calibrator registry, the refit job, and their HTTP and CLI surfaces."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from typer.testing import CliRunner

from decision_gate.app import create_app
from decision_gate.backends import FakeBackend
from decision_gate.backends.fake import hash_oracle
from decision_gate.calibrators import CalibrationError, load_calibrator
from decision_gate.cli import app as cli
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import CalibratorRecord, DecisionRecord, OutcomeRecord
from decision_gate.db.repository import CalibratorRepository, LabeledRow, OutcomeRepository
from decision_gate.questions import QuestionCatalog
from decision_gate.refit import RefitJob, calibration_data, candidate_methods

runner = CliRunner()
BASE = datetime(2026, 10, 1, tzinfo=UTC)


def run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


async def with_repos[T](
    database_url: str,
    work: Callable[[RefitJob, CalibratorRepository], Awaitable[T]],
) -> T:
    engine = make_engine(database_url)
    try:
        sessions = make_sessionmaker(engine)
        calibrators = CalibratorRepository(sessions)
        return await work(RefitJob(OutcomeRepository(sessions), calibrators), calibrators)
    finally:
        await engine.dispose()


async def _insert(database_url: str, records: list[Any]) -> None:
    engine = make_engine(database_url)
    try:
        async with make_sessionmaker(engine)() as session:
            session.add_all(records)
            await session.commit()
    finally:
        await engine.dispose()


def seed_fake(
    database_url: str,
    catalog: QuestionCatalog,
    question_id: str,
    n: int,
    *,
    overconfidence: float = 2.0,
    start: int = 0,
) -> None:
    """``n`` labeled fake decisions, one second apart, with the fake's own ground truth."""
    question = catalog.get(question_id)
    backend = FakeBackend(overconfidence=overconfidence)
    decisions: list[Any] = []
    outcomes: list[Any] = []
    for i in range(start, start + n):
        data = {"item": i}
        raw = backend.decide_sync(question, data)
        truth = hash_oracle(question, data)
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
        outcomes.append(
            OutcomeRecord(
                decision_id=decision_id,
                outcome=truth,
                correct=question.is_correct(raw.answer, truth),
                source="test",
                observed_at=BASE + timedelta(seconds=i, hours=1),
            )
        )
    run(_insert(database_url, decisions))
    run(_insert(database_url, outcomes))


def seed_rows(
    database_url: str, rows: list[tuple[float, bool]], *, question: str = "refund.eligible"
) -> None:
    """Hand-made predicate decisions with given confidence and correctness, in order."""
    decisions: list[Any] = []
    outcomes: list[Any] = []
    for i, (confidence, correct) in enumerate(rows):
        decision_id = uuid.uuid4()
        decisions.append(
            DecisionRecord(
                id=decision_id,
                question_id=question,
                question_version=1,
                input_sha256="0" * 64,
                input=None,
                answer=True,
                confidence=confidence,
                probabilities={"true": confidence, "false": 1 - confidence},
                backend="fake",
                model_version="m",
                latency_ms=0.1,
                created_at=BASE + timedelta(seconds=i),
            )
        )
        outcomes.append(
            OutcomeRecord(
                decision_id=decision_id,
                outcome=correct,
                correct=correct,
                source="test",
                observed_at=BASE + timedelta(seconds=i),
            )
        )
    run(_insert(database_url, decisions))
    run(_insert(database_url, outcomes))


class TestCalibrationData:
    def test_distributions_only_when_every_row_has_one(self) -> None:
        def row(probabilities: dict[str, float] | None, outcome: Any, answer: Any) -> LabeledRow:
            return LabeledRow("q", 1, "fake", "m", 0.7, True, probabilities, outcome, answer)

        full = calibration_data([row({"true": 0.7, "false": 0.3}, True, True)])
        assert full.answers == ("true",)
        assert full.truths == ("true",)
        assert not calibration_data([row(None, True, True)]).has_distributions
        assert not calibration_data([row({"a": 1.0}, 12.0, 12.0)]).has_distributions

    def test_candidate_methods(self) -> None:
        small = calibration_data(
            [LabeledRow("q", 1, "f", "m", 0.7, True, {"a": 0.7, "b": 0.3}, "a", "a")] * 10
        )
        assert candidate_methods("auto", small) == ["platt", "temperature"]
        assert candidate_methods("isotonic", small) == ["isotonic"]
        big = calibration_data([LabeledRow("q", 1, "f", "m", 0.7, True, None, 1.0, 1.0)] * 200)
        assert candidate_methods("auto", big) == ["platt", "isotonic"]
        with pytest.raises(CalibrationError, match="temperature scaling needs"):
            candidate_methods("temperature", big)


class TestRefitJob:
    def test_too_few_labels_is_skipped_without_a_record(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 20)

        async def work(job: RefitJob, calibrators: CalibratorRepository) -> Any:
            result = await job.refit("refund.eligible", 1, min_count=50)
            return result, await calibrators.find()

        result, stored = run(with_repos(database_url, work))
        assert (result.outcome, result.labeled, result.record) == ("skipped", 20, None)
        assert "fewer than min_count=50" in result.reason
        assert stored == []

    def test_overconfident_backend_gets_an_active_calibrator(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.queue", 400)

        async def work(job: RefitJob, calibrators: CalibratorRepository) -> Any:
            result = await job.refit("ticket.queue", 1)
            return result, await calibrators.active("ticket.queue", 1)

        result, active = run(with_repos(database_url, work))
        assert result.outcome == "activated"
        record = result.record
        assert record is not None
        assert active is not None
        assert active.id == record.id
        assert (record.version, record.status, record.train_count, record.holdout_count) == (
            1,
            "active",
            280,
            120,
        )
        assert record.activated_at is not None
        ev = record.evaluation
        assert set(ev["candidates"]) == {"platt", "isotonic", "temperature"}
        assert ev["split"] == "chronological"
        assert ev["incumbent"] is None
        assert ev["calibrated"] == ev["candidates"][record.method]
        # The overconfident fake loses log loss and ECE once calibrated.
        assert ev["calibrated"]["log_loss"] < ev["raw"]["log_loss"]
        assert ev["calibrated"]["ece"] < ev["raw"]["ece"]
        # Training data ends before the held-out data starts (one decision per second).
        trained_until = record.trained_until.replace(tzinfo=UTC)
        assert trained_until == BASE + timedelta(seconds=279)
        assert "beats raw" in record.reason
        if record.method == "temperature":
            # Overconfidence 2 in log-odds is temperature 2 for the fake's distributions.
            assert record.params["temperature"] == pytest.approx(2.0, rel=0.25)

    def test_score_questions_never_try_temperature(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.handling_minutes", 150)

        async def work(job: RefitJob, _: CalibratorRepository) -> Any:
            return await job.refit("ticket.handling_minutes", 1)

        result = run(with_repos(database_url, work))
        assert result.record is not None
        # 105 training decisions: too few for isotonic under "auto".
        assert list(result.record.evaluation["candidates"]) == ["platt"]

        async def explicit(job: RefitJob, _: CalibratorRepository) -> Any:
            return await job.refit("ticket.handling_minutes", 1, method="temperature")

        with pytest.raises(CalibrationError, match="temperature scaling needs"):
            run(with_repos(database_url, explicit))

    def test_a_fit_no_better_than_the_active_one_is_rejected(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 300)

        async def work(job: RefitJob, calibrators: CalibratorRepository) -> Any:
            first = await job.refit("refund.eligible", 1, method="platt")
            second = await job.refit("refund.eligible", 1, method="platt")
            return first, second, await calibrators.active("refund.eligible", 1)

        first, second, active = run(with_repos(database_url, work))
        assert first.outcome == "activated"
        # Same data, same method: an identical fit does not strictly beat the incumbent.
        assert second.outcome == "rejected"
        assert second.record is not None
        assert second.record.version == 2
        assert "does not beat active v1 (platt)" in second.reason
        assert second.record.evaluation["incumbent"]["version"] == 1
        assert active is not None
        assert first.record is not None
        assert active.id == first.record.id

    def test_a_fit_worse_than_raw_is_rejected(self, database_url: str) -> None:
        # Always right during training, always wrong on the newest 30%: calibrating
        # towards "always right" makes the held-out decisions worse than the raw 0.9.
        seed_rows(database_url, [(0.9, True)] * 70 + [(0.9, False)] * 30)

        async def work(job: RefitJob, calibrators: CalibratorRepository) -> Any:
            result = await job.refit("refund.eligible", 1)
            return result, await calibrators.active("refund.eligible", 1)

        result, active = run(with_repos(database_url, work))
        assert result.outcome == "rejected"
        assert "does not beat raw" in result.reason
        assert result.record is not None
        assert result.record.status == "rejected"
        assert active is None

    def test_no_activate_stores_a_candidate_then_better_fits_replace_it(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 300)

        async def work(job: RefitJob, calibrators: CalibratorRepository) -> Any:
            candidate = await job.refit("refund.eligible", 1, activate=False)
            nothing_active = await calibrators.active("refund.eligible", 1)
            assert candidate.record is not None
            activated = await calibrators.activate(candidate.record.id)
            again = await calibrators.activate(candidate.record.id)
            return candidate, nothing_active, activated, again

        candidate, nothing_active, activated, again = run(with_repos(database_url, work))
        assert candidate.outcome == "candidate"
        assert nothing_active is None
        assert activated.status == "active"
        assert activated.activated_at is not None
        assert again.id == activated.id
        assert again.status == "active"

    def test_an_incumbent_that_cannot_be_scored_does_not_block(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.handling_minutes", 300)
        # A temperature scaler cannot score decisions without distributions.
        bogus = CalibratorRecord(
            question_id="ticket.handling_minutes",
            question_version=1,
            method="temperature",
            params={"temperature": 2.0},
            status="active",
            reason="test",
            train_count=0,
            holdout_count=0,
            evaluation={},
        )

        async def work(job: RefitJob, calibrators: CalibratorRepository) -> Any:
            await calibrators.add(bogus)
            return await job.refit("ticket.handling_minutes", 1)

        result = run(with_repos(database_url, work))
        assert result.outcome == "activated", result.reason
        assert result.record is not None
        incumbent = result.record.evaluation["incumbent"]
        assert incumbent["method"] == "temperature"
        assert "distribution" in incumbent["error"]
        assert result.record.version == 2

    def test_a_fit_that_beats_the_active_one_replaces_it(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 300)
        # The identity calibrator scores exactly like the raw confidences.
        identity = CalibratorRecord(
            question_id="refund.eligible",
            question_version=1,
            method="platt",
            params={"a": 1.0, "b": 0.0},
            status="active",
            reason="test",
            train_count=0,
            holdout_count=0,
            evaluation={},
        )

        async def work(job: RefitJob, calibrators: CalibratorRepository) -> Any:
            await calibrators.add(identity)
            result = await job.refit("refund.eligible", 1)
            return result, await calibrators.find(question_id="refund.eligible")

        result, stored = run(with_repos(database_url, work))
        assert result.outcome == "activated"
        assert ", active v1 " in result.reason
        assert [(r.version, r.status) for r in stored] == [(2, "active"), (1, "retired")]

    def test_refit_all_and_window(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 150)
        seed_fake(database_url, example_catalog, "ticket.queue", 30)
        unlabeled = DecisionRecord(
            question_id="ticket.handling_minutes",
            question_version=1,
            input_sha256="0" * 64,
            answer=10.0,
            confidence=0.5,
            backend="fake",
            model_version="m",
            latency_ms=0.1,
        )
        run(_insert(database_url, [unlabeled]))

        async def work(job: RefitJob, _: CalibratorRepository) -> Any:
            everything = await job.refit_all(min_count=100)
            windowed = await job.refit(
                "refund.eligible", 1, until=BASE + timedelta(seconds=50), min_count=100
            )
            return everything, windowed

        everything, windowed = run(with_repos(database_url, work))
        assert [(r.question_key, r.outcome != "skipped") for r in everything] == [
            ("refund.eligible@v1", True),
            ("ticket.queue@v1", False),
        ]
        assert (windowed.outcome, windowed.labeled) == ("skipped", 50)

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"method": "beta"}, "unknown method"),
            ({"holdout": 0.0}, "holdout"),
            ({"holdout": 1.0}, "holdout"),
            ({"min_count": 1}, "min_count"),
        ],
    )
    def test_bad_options(self, sqlite_url: str, options: dict[str, Any], message: str) -> None:
        async def work(job: RefitJob, _: CalibratorRepository) -> Any:
            return await job.refit("refund.eligible", 1, **options)

        with pytest.raises(CalibrationError, match=message):
            run(with_repos(sqlite_url, work))


class TestRegistry:
    def record(self, status: str = "candidate", question: str = "q") -> CalibratorRecord:
        return CalibratorRecord(
            question_id=question,
            question_version=1,
            method="platt",
            params={"a": 1.0, "b": 0.0},
            status=status,
            reason="test",
            train_count=1,
            holdout_count=1,
            evaluation={},
        )

    def test_versions_activation_and_filters(self, database_url: str) -> None:
        async def work(_: RefitJob, repo: CalibratorRepository) -> Any:
            one = await repo.add(self.record("active"))
            two = await repo.add(self.record("active"))
            three = await repo.add(self.record("rejected"))
            other = await repo.add(self.record("active", question="r"))
            rolled_back = await repo.activate(one.id)
            return (
                [one, two, three, other, rolled_back],
                await repo.find(),
                await repo.find(question_id="q", status="retired"),
                await repo.find(question_version=2),
                await repo.find(limit=1),
                await repo.get(uuid.uuid4()),
                await repo.activate(uuid.uuid4()),
            )

        stored, everything, retired, none, limited, missing, missing_activate = run(
            with_repos(database_url, work)
        )
        one, two, three, other, rolled_back = stored
        assert [r.version for r in (one, two, three, other)] == [1, 2, 3, 1]
        assert rolled_back.id == one.id
        assert rolled_back.status == "active"
        assert [(r.question_id, r.version, r.status) for r in everything] == [
            ("q", 3, "rejected"),
            ("q", 2, "retired"),
            ("q", 1, "active"),
            ("r", 1, "active"),
        ]
        assert [r.version for r in retired] == [2]
        assert none == []
        assert len(limited) == 1
        assert missing is None
        assert missing_activate is None
        rebuilt = load_calibrator(one.method, one.params)
        assert rebuilt.calibrate(0.7, None, "x") == pytest.approx(0.7)

    def test_the_database_allows_one_active_per_question_version(self, database_url: str) -> None:
        with pytest.raises(IntegrityError):
            run(_insert(database_url, [self.record("active"), self.record("active")]))

    def test_concurrent_fits_get_distinct_versions(self, database_url: str) -> None:
        if database_url.startswith("sqlite"):
            pytest.skip("SQLite serialises writers; the version race needs PostgreSQL")

        async def work(_: RefitJob, repo: CalibratorRepository) -> Any:
            stored = await asyncio.gather(*(repo.add(self.record()) for _ in range(4)))
            return sorted(r.version for r in stored)

        assert run(with_repos(database_url, work)) == [1, 2, 3, 4]


@pytest.fixture
def client(database_url: str, example_catalog: QuestionCatalog) -> Iterator[TestClient]:
    app = create_app(
        Settings(database_url=database_url), catalog=example_catalog, backend=FakeBackend()
    )
    with TestClient(app) as test_client:
        yield test_client


class TestApi:
    def test_refit_list_get_and_activate(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 300)
        response = client.post(
            "/v1/calibrators/refit", json={"question": "refund.eligible", "activate": False}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["question"], body["question_version"], body["outcome"]) == (
            "refund.eligible",
            1,
            "candidate",
        )
        calibrator = body["calibrator"]
        assert calibrator["status"] == "candidate"
        assert calibrator["trained_until"].startswith("2026-10-01T00:03:29")
        assert calibrator["activated_at"] is None

        listed = client.get("/v1/calibrators", params={"question": "refund.eligible"}).json()
        assert [c["id"] for c in listed["calibrators"]] == [calibrator["id"]]
        assert client.get("/v1/calibrators", params={"status": "active"}).json() == {
            "calibrators": []
        }

        fetched = client.get(f"/v1/calibrators/{calibrator['id']}")
        assert fetched.json()["evaluation"] == calibrator["evaluation"]
        activated = client.post(f"/v1/calibrators/{calibrator['id']}/activate")
        assert activated.status_code == 200
        assert activated.json()["status"] == "active"
        assert activated.json()["activated_at"] is not None

    def test_skipped_and_errors(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.handling_minutes", 10)
        skipped = client.post(
            "/v1/calibrators/refit", json={"question": "ticket.handling_minutes", "version": 1}
        ).json()
        assert (skipped["outcome"], skipped["labeled"], skipped["calibrator"]) == (
            "skipped",
            10,
            None,
        )
        assert client.post("/v1/calibrators/refit", json={"question": "nope"}).status_code == 404
        bad = client.post(
            "/v1/calibrators/refit",
            json={"question": "ticket.handling_minutes", "method": "temperature", "min_count": 2},
        )
        assert bad.status_code == 422
        assert "temperature scaling needs" in bad.json()["detail"]
        for payload in ({"question": "q", "holdout": 1}, {"question": "q", "method": "beta"}):
            assert client.post("/v1/calibrators/refit", json=payload).status_code == 422
        unknown = uuid.uuid4()
        assert client.get(f"/v1/calibrators/{unknown}").status_code == 404
        assert client.post(f"/v1/calibrators/{unknown}/activate").status_code == 404
        assert client.get("/v1/calibrators", params={"status": "great"}).status_code == 422


class TestCli:
    def test_refit_list_show_activate(
        self, database_url: str, example_catalog: QuestionCatalog, monkeypatch: Any
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.queue", 300)
        monkeypatch.setenv("DGATE_QUESTIONS_FILE", "examples/questions.yaml")
        args = ["--url", database_url]
        refit = runner.invoke(cli, ["calibrators", "refit", "ticket.queue", "--no-activate", *args])
        assert refit.exit_code == 0, refit.output
        assert refit.output.startswith("ticket.queue@v1")
        assert "candidate" in refit.output
        calibrator_id = refit.output.strip().split()[-1].rstrip("]")

        listed = runner.invoke(cli, ["calibrators", "list", *args])
        assert listed.exit_code == 0, listed.output
        header, line = listed.output.splitlines()
        assert header.split()[:4] == ["question", "ver", "method", "status"]
        assert line.split()[:2] == ["ticket.queue@v1", "1"]
        assert line.split()[3] == "candidate"

        shown = runner.invoke(cli, ["calibrators", "show", calibrator_id, *args])
        assert shown.exit_code == 0, shown.output
        assert json.loads(shown.output)["id"] == calibrator_id

        activated = runner.invoke(cli, ["calibrators", "activate", calibrator_id, *args])
        assert activated.exit_code == 0, activated.output
        assert activated.output.startswith("activated ticket.queue@v1 v1")

        empty = runner.invoke(cli, ["calibrators", "list", "--status", "rejected", *args])
        assert empty.output.strip() == "no calibrators"

    def test_refit_all_json_and_options(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 200)
        args = ["calibrators", "refit", "--all", "--url", database_url]
        result = runner.invoke(cli, [*args, "--json", "--method", "platt", "--holdout", "0.25"])
        assert result.exit_code == 0, result.output
        [only] = json.loads(result.output)["results"]
        assert only["outcome"] == "activated"
        assert only["calibrator"]["method"] == "platt"
        assert only["calibrator"]["holdout_count"] == 50
        window = runner.invoke(
            cli, [*args, "--since", "2026-10-02T00:00:00", "--until", "2026-10-03T00:00:00"]
        )
        # --all picks question versions with labels; the window then leaves none to fit.
        assert "skipped    n=0" in window.output
        skipped = runner.invoke(
            cli,
            [
                "calibrators",
                "refit",
                "refund.eligible",
                "--version",
                "1",
                "--min-count",
                "500",
                "--url",
                database_url,
            ],
        )
        assert "skipped" in skipped.output

    def test_errors(self, sqlite_url: str, example_catalog: QuestionCatalog) -> None:
        base = ["calibrators", "refit", "--url", sqlite_url]
        assert runner.invoke(cli, base).exit_code == 2
        assert runner.invoke(cli, [*base, "q", "--all"]).exit_code == 2
        assert runner.invoke(cli, [*base, "--all", "--method", "beta"]).exit_code == 2
        assert runner.invoke(cli, [*base, "--all"]).output.strip() == "no labeled decisions"
        unknown = runner.invoke(cli, [*base, "nope", "-q", "examples/questions.yaml"])
        assert unknown.exit_code == 1
        seed_fake(sqlite_url, example_catalog, "ticket.handling_minutes", 5)
        bad = runner.invoke(cli, [*base, "--all", "--method", "temperature", "--min-count", "2"])
        assert bad.exit_code == 1
        assert "temperature scaling needs" in bad.output
        for command in ("show", "activate"):
            assert runner.invoke(cli, ["calibrators", command, "x"]).exit_code == 2
            missing = runner.invoke(
                cli, ["calibrators", command, str(uuid.uuid4()), "--url", sqlite_url]
            )
            assert missing.exit_code == 1


def test_fake_calibrate_prints_every_method() -> None:
    result = runner.invoke(
        cli, ["fake-calibrate", "-q", "examples/questions.yaml", "-n", "600", "--holdout", "0.5"]
    )
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split()[:2] == ["question", "method"]
    methods = [line.split()[0] if line.startswith(" ") else line.split()[1] for line in lines[1:]]
    # Predicate and choice: raw, platt, isotonic, temperature. Score: raw, platt, isotonic.
    assert methods == ["raw", "platt", "isotonic", "temperature"] * 2 + [
        "raw",
        "platt",
        "isotonic",
    ]
    assert "temperature=" in result.output
    assert "knots" in result.output
    assert runner.invoke(cli, ["fake-calibrate", "-q", "missing.yaml"]).exit_code == 1
