from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_gate import __version__
from decision_gate.app import create_app
from decision_gate.catalog import CatalogError, load_catalog, parse_catalog
from decision_gate.cli import app as cli
from decision_gate.config import Settings
from decision_gate.questions import QuestionCatalog
from tests.conftest import EXAMPLES

runner = CliRunner()


@pytest.fixture
def client(example_catalog: QuestionCatalog, sqlite_url: str) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(database_url=sqlite_url), catalog=example_catalog)) as c:
        yield c


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body == {
        "status": "ready",
        "version": __version__,
        "backend": "fake",
        "questions": 3,
        "database": "ok",
    }


def test_readyz_reports_unavailable_database(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/missing-dir/dgate.db"
    with TestClient(create_app(Settings(database_url=url))) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["database"] == "unavailable"


def test_list_and_get_questions(client: TestClient) -> None:
    ids = [q["id"] for q in client.get("/v1/questions").json()["questions"]]
    assert ids == ["refund.eligible", "ticket.queue", "ticket.handling_minutes"]
    body = client.get("/v1/questions/ticket.queue").json()
    assert body["kind"] == "choice"
    assert body["options"][0] == "billing"


def test_unknown_question_is_404(client: TestClient) -> None:
    response = client.get("/v1/questions/nope")
    assert response.status_code == 404
    assert "unknown question" in response.json()["detail"]
    assert client.get("/v1/questions/ticket.queue?version=9").status_code == 404


def test_app_loads_catalog_from_settings(sqlite_url: str) -> None:
    app = create_app(Settings(questions_file=EXAMPLES / "questions.yaml", database_url=sqlite_url))
    assert TestClient(app).get("/readyz").json()["questions"] == 3


def test_shutdown_closes_backend_client(sqlite_url: str) -> None:
    settings = Settings(database_url=sqlite_url, backend="http", http_url="http://decider")
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/readyz").json()["backend"] == "http"
    assert app.state.backend._client.is_closed


def test_app_without_catalog_is_empty() -> None:
    assert TestClient(create_app(Settings())).get("/v1/questions").json() == {"questions": []}


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DGATE_FAKE_OVERCONFIDENCE", "1.5")
    monkeypatch.setenv("DGATE_QUESTIONS_FILE", "/tmp/q.yaml")
    settings = Settings()
    assert settings.fake_overconfidence == 1.5
    assert settings.questions_file == Path("/tmp/q.yaml")


class TestCatalogLoading:
    def test_json_and_bare_list(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        path.write_text(json.dumps([{"id": "q1", "kind": "predicate", "prompt": "p"}]))
        assert load_catalog(path).get("q1").kind == "predicate"

    def test_empty_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "q.yaml"
        path.write_text("")
        assert load_catalog(path).questions == ()

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogError, match="cannot read"):
            load_catalog(tmp_path / "missing.yaml")

    def test_bad_syntax(self, tmp_path: Path) -> None:
        path = tmp_path / "q.yaml"
        path.write_text("questions: [")
        with pytest.raises(CatalogError, match="not valid YAML"):
            load_catalog(path)
        path = tmp_path / "q.json"
        path.write_text("{")
        with pytest.raises(CatalogError, match="not valid JSON"):
            load_catalog(path)

    def test_wrong_shape(self) -> None:
        with pytest.raises(CatalogError, match="expected a mapping"):
            parse_catalog("just text")

    def test_invalid_question(self) -> None:
        with pytest.raises(CatalogError, match="invalid question catalog"):
            parse_catalog({"questions": [{"id": "q1", "kind": "choice", "prompt": "p"}]})


class TestCli:
    def test_version(self) -> None:
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0
        assert result.output.strip() == __version__

    def test_questions_check(self) -> None:
        result = runner.invoke(cli, ["questions", "check", str(EXAMPLES / "questions.yaml")])
        assert result.exit_code == 0, result.output
        assert "ticket.queue@v1" in result.output
        assert "3 question(s) OK" in result.output

    def test_questions_check_fails_on_bad_file(self, tmp_path: Path) -> None:
        path = tmp_path / "q.yaml"
        path.write_text("questions: [{id: q1, kind: choice, prompt: p}]")
        result = runner.invoke(cli, ["questions", "check", str(path)])
        assert result.exit_code == 1

    def test_ask(self) -> None:
        args = ["ask", "ticket.queue", "-q", str(EXAMPLES / "questions.yaml")]
        result = runner.invoke(cli, [*args, "-i", '{"subject": "card declined"}'])
        assert result.exit_code == 0, result.output
        body = json.loads(result.output)
        assert body["question"] == "ticket.queue@v1"
        assert body["answer"] in {"billing", "technical", "account", "shipping", "other"}
        again = runner.invoke(cli, [*args, "-i", '{"subject": "card declined"}'])
        assert again.output == result.output

    @pytest.mark.parametrize(
        ("extra", "code"),
        [(["-i", "{"], 2), (["-i", "[1]"], 2)],
    )
    def test_ask_bad_input(self, extra: list[str], code: int) -> None:
        args = ["ask", "ticket.queue", "-q", str(EXAMPLES / "questions.yaml"), *extra]
        assert runner.invoke(cli, args).exit_code == code

    def test_ask_unknown_question(self) -> None:
        result = runner.invoke(cli, ["ask", "nope", "-q", str(EXAMPLES / "questions.yaml")])
        assert result.exit_code == 1

    def test_fake_gap_reports_overconfidence(self) -> None:
        args = ["fake-gap", "-q", str(EXAMPLES / "questions.yaml"), "-n", "2000"]
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output
        rows = [line.split() for line in result.output.splitlines()[1:]]
        assert [row[0] for row in rows] == [
            "refund.eligible@v1",
            "ticket.queue@v1",
            "ticket.handling_minutes@v1",
        ]
        assert all(float(row[3]) > 0.05 for row in rows)
        assert all(float(row[4]) > 0.05 for row in rows)  # ECE

    def test_fake_gap_calibrated(self) -> None:
        args = ["fake-gap", "-q", str(EXAMPLES / "questions.yaml"), "-n", "2000"]
        result = runner.invoke(cli, [*args, "--overconfidence", "1.0"])
        rows = [line.split() for line in result.output.splitlines()[1:]]
        assert all(abs(float(row[3])) < 0.04 for row in rows)
        assert all(float(row[4]) < 0.04 for row in rows)  # ECE
        overconfident = runner.invoke(cli, args).output.splitlines()[1:]
        # Calibration alone lowers the Brier score for the same answers.
        for calibrated, raw in zip(rows, overconfident, strict=True):
            assert float(calibrated[5]) < float(raw.split()[5])

    def test_fake_gap_bad_catalog(self, tmp_path: Path) -> None:
        result = runner.invoke(cli, ["fake-gap", "-q", str(tmp_path / "missing.yaml")])
        assert result.exit_code == 1
