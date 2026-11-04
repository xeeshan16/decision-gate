"""Routing tiers: policy files, rule refreshes, decide-time routing, the API and the CLI."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_gate.app import create_app
from decision_gate.backends import FakeBackend
from decision_gate.backends.fake import hash_oracle
from decision_gate.cli import app as cli
from decision_gate.config import Settings
from decision_gate.db.models import CalibratorRecord, RoutingRuleRecord
from decision_gate.questions import QuestionCatalog
from decision_gate.routing import (
    RoutingConfig,
    RoutingConfigError,
    RoutingPolicy,
    RoutingRefresher,
    load_routing,
    parse_routing,
    sample_unit,
)
from decision_gate.selection import SelectionError
from tests.conftest import EXAMPLES
from tests.test_refit import BASE, _insert, run, seed_fake
from tests.test_selection import insert_calibrator, insert_decision, refit

runner = CliRunner()

POLICY = {
    "max_error_rate": 0.1,
    "verify_max_error_rate": 0.25,
    "verify_sample_rate": 0.1,
    "conformal_alpha": 0.1,
}


def make_client(
    database_url: str,
    catalog: QuestionCatalog,
    routing: RoutingConfig,
    *,
    ttl: float = 0.0,
) -> TestClient:
    settings = Settings(database_url=database_url, routing_cache_ttl_s=ttl)
    return TestClient(create_app(settings, catalog=catalog, backend=FakeBackend(), routing=routing))


@pytest.fixture
def routing() -> RoutingConfig:
    return parse_routing(
        {
            "defaults": {"min_labeled": 200},
            "questions": {
                "refund.eligible": POLICY,
                "ticket.queue": {**POLICY, "max_error_rate": 0.05, "verify_max_error_rate": 0.2},
                "ticket.handling_minutes": {"max_error_rate": 0.2, "require_calibrator": False},
            },
        }
    )


@pytest.fixture
def client(
    database_url: str, example_catalog: QuestionCatalog, routing: RoutingConfig
) -> Iterator[TestClient]:
    with make_client(database_url, example_catalog, routing) as test_client:
        yield test_client


def decide(client: TestClient, question: str, item: int) -> dict[str, Any]:
    response = client.post("/v1/decide", json={"question": question, "input": {"item": item}})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def refresh(client: TestClient, question: str | None = None) -> list[dict[str, Any]]:
    payload = {} if question is None else {"question": question}
    response = client.post("/v1/routing/refresh", json=payload)
    assert response.status_code == 200, response.text
    rules: list[dict[str, Any]] = response.json()["rules"]
    return rules


def states(client: TestClient) -> dict[str, str]:
    body = client.get("/v1/routing/status").json()
    return {q["question"]: q["state"] for q in body["questions"]}


# --- Policy files ------------------------------------------------------------------------


class TestPolicies:
    def test_defaults_merge_into_every_policy(self) -> None:
        config = parse_routing(
            {
                "defaults": {"confidence": 0.9, "min_labeled": 50},
                "default": {"max_error_rate": 0.2},
                "questions": {"a.b": {"min_labeled": 10}, "c.d": None},
            }
        )
        a = config.policy_for("a.b")
        assert a is not None
        assert (a.confidence, a.min_labeled, a.max_error_rate) == (0.9, 10, 0.05)
        c = config.policy_for("c.d")
        assert c is not None
        assert c.min_labeled == 50
        unlisted = config.policy_for("x.y")
        assert unlisted is not None
        assert (unlisted.max_error_rate, unlisted.confidence) == (0.2, 0.9)
        assert parse_routing({"questions": {"a.b": {}}}).policy_for("x.y") is None
        assert parse_routing(None) == RoutingConfig()

    def test_fingerprint_tracks_every_field(self) -> None:
        policy = RoutingPolicy()
        assert policy.fingerprint() == RoutingPolicy().fingerprint()
        assert len(policy.fingerprint()) == 16
        assert policy.fingerprint() != RoutingPolicy(verify_sample_rate=0.1).fingerprint()
        assert policy.target_precision == pytest.approx(0.95)

    @pytest.mark.parametrize(
        ("raw", "match"),
        [
            ([1, 2], "expected a mapping"),
            ({"policies": {}}, "unknown keys"),
            ({"questions": ["a.b"]}, "must be mappings"),
            ({"questions": {"a.b": {"max_error_rate": 1.5}}}, "invalid routing policy"),
            ({"questions": {"a.b": {"bogus": 1}}}, "invalid routing policy"),
            (
                {"questions": {"a.b": {"max_error_rate": 0.1, "verify_max_error_rate": 0.1}}},
                "verify_max_error_rate must be larger",
            ),
        ],
    )
    def test_invalid(self, raw: Any, match: str) -> None:
        with pytest.raises(RoutingConfigError, match=match):
            parse_routing(raw)

    def test_load_files(self, tmp_path: Path, example_catalog: QuestionCatalog) -> None:
        config = load_routing(EXAMPLES / "routing.yaml")
        assert config.unknown_questions(example_catalog) == []
        assert config.policy_for("refund.eligible") is not None
        as_json = tmp_path / "routing.json"
        as_json.write_text(json.dumps({"questions": {"nope.q": {}}}))
        assert load_routing(as_json).unknown_questions(example_catalog) == ["nope.q"]
        broken = tmp_path / "broken.yaml"
        broken.write_text("questions: [unclosed")
        with pytest.raises(RoutingConfigError, match="not valid"):
            load_routing(broken)
        with pytest.raises(RoutingConfigError, match="cannot read"):
            load_routing(tmp_path / "missing.yaml")

    def test_sample_unit_is_deterministic_and_uniform(self) -> None:
        ids = [uuid.UUID(int=i) for i in range(20_000)]
        values = [sample_unit(i) for i in ids]
        assert values[:5] == [sample_unit(i) for i in ids[:5]]
        assert all(0.0 <= v < 1.0 for v in values)
        assert sum(values) / len(values) == pytest.approx(0.5, abs=0.01)
        assert sum(v < 0.1 for v in values) / len(values) == pytest.approx(0.1, abs=0.01)


# --- Routing end to end --------------------------------------------------------------------


class TestRouting:
    def test_no_policy_and_no_rule_go_to_a_human(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        with make_client(database_url, example_catalog, RoutingConfig()) as plain:
            body = decide(plain, "refund.eligible", 1)
            assert body["tier"] == "human"
            assert body["reason"] == "no routing policy covers refund.eligible"
            assert body["calibrated_confidence"] is None
            assert states(plain)["refund.eligible"] == "no_policy"

    def test_uncalibrated_question_goes_to_a_human(self, client: TestClient) -> None:
        body = decide(client, "refund.eligible", 1)
        assert (body["tier"], body["reason"]) == (
            "human",
            "refund.eligible@v1 is not calibrated yet",
        )
        [rule] = refresh(client, "refund.eligible")
        assert rule["auto_threshold"] is None
        assert rule["calibrator_id"] is None
        assert "no active calibrator" in rule["reason"]
        assert states(client)["refund.eligible"] == "current"

    def test_calibrated_routing_meets_the_budget(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 1500)
        calibrator = refit(database_url, "refund.eligible")
        assert states(client)["refund.eligible"] == "no_rule"
        assert decide(client, "refund.eligible", 1)["reason"].startswith("no routing rule")

        [rule] = refresh(client, "refund.eligible")
        assert rule["status"] == "active"
        assert rule["calibrator_id"] == str(calibrator.id)
        # Temperature scaling on two labels keeps the order, so all 1500 are eligible.
        assert (rule["labeled"], rule["eligible"]) == (1500, 1500)
        auto, verify = rule["auto_threshold"], rule["verify_threshold"]
        assert auto is not None
        assert verify is not None
        assert verify <= auto
        assert rule["evaluation"]["auto"]["lower_bound"] >= 0.9
        assert rule["evaluation"]["verify"]["lower_bound"] >= 0.75
        assert rule["conformal"]["labels"] == ["false", "true"]
        assert rule["policy"]["max_error_rate"] == 0.1
        assert states(client)["refund.eligible"] == "current"

        question = example_catalog.get("refund.eligible")
        tiers: Counter[str] = Counter()
        correct: Counter[str] = Counter()
        for item in range(10_000, 10_600):
            body = decide(client, "refund.eligible", item)
            tier, score = body["tier"], body["calibrated_confidence"]
            tiers[tier] += 1
            correct[tier] += question.is_correct(
                body["answer"], hash_oracle(question, {"item": item})
            )
            assert body["routing_rule_id"] == rule["id"]
            if tier == "auto":
                assert score >= auto
                assert body["answer_set"] is None
                assert "auto threshold" in body["reason"]
            elif tier == "verify":
                assert score >= verify
                assert body["answer_set"]
                assert "verif" in body["reason"]
            else:
                assert score < verify
                assert "below every threshold" in body["reason"]
        assert set(tiers) == {"auto", "verify", "human"}
        # 600 fresh decisions are a small sample; the guarantee is on the long-run rate.
        assert correct["auto"] / tiers["auto"] >= 0.85

    def test_verify_sampling_and_replay(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 1500)
        refit(database_url, "refund.eligible")
        [rule] = refresh(client, "refund.eligible")
        automatable = 0
        for item in range(20_000, 20_400):
            body = decide(client, "refund.eligible", item)
            if body["calibrated_confidence"] < rule["auto_threshold"]:
                continue
            automatable += 1
            # Sampling is a pure function of the decision id, so it can be recomputed.
            sampled = sample_unit(uuid.UUID(body["id"])) < 0.1
            assert body["tier"] == ("verify" if sampled else "auto")
            assert ("sampled for verification" in body["reason"]) == sampled
        assert automatable > 0

        first = client.post(
            "/v1/decide",
            json={"question": "refund.eligible", "input": {"item": 7}, "idempotency_key": "k"},
        ).json()
        # A new policy would change the tier of a new decision, but not of a replay.
        strict = parse_routing({"questions": {"refund.eligible": {"max_error_rate": 0.01}}})
        with make_client(database_url, example_catalog, strict) as other:
            replay = other.post(
                "/v1/decide",
                json={"question": "refund.eligible", "input": {"item": 7}, "idempotency_key": "k"},
            )
            assert replay.status_code == 200
            assert replay.json()["tier"] == first["tier"]
            assert replay.json()["reason"] == first["reason"]
            fresh = decide(other, "refund.eligible", 8)
            assert fresh["tier"] == "human"
            assert fresh["reason"] == "the routing policy changed since the rule was selected"
            assert states(other)["refund.eligible"] == "policy_changed"
        stored = client.get(f"/v1/decisions/{first['id']}").json()
        assert stored["tier"] == first["tier"]

    def test_a_new_calibrator_invalidates_the_rule(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 1500)
        refit(database_url, "refund.eligible")
        refresh(client, "refund.eligible")
        assert states(client)["refund.eligible"] == "current"
        seed_fake(database_url, example_catalog, "refund.eligible", 600, start=1500)
        response = client.post(
            "/v1/calibrators/refit",
            json={"question": "refund.eligible", "method": "platt"},
        )
        assert response.status_code == 200, response.text
        if response.json()["outcome"] != "activated":
            calibrator_id = response.json()["calibrator"]["id"]
            activated = client.post(f"/v1/calibrators/{calibrator_id}/activate")
            assert activated.status_code == 200
        assert states(client)["refund.eligible"] == "calibrator_changed"
        body = decide(client, "refund.eligible", 3)
        assert body["tier"] == "human"
        assert body["reason"] == "the active calibrator changed since the thresholds were selected"
        [rule] = refresh(client, "refund.eligible")
        assert states(client)["refund.eligible"] == "current"
        history = client.get("/v1/routing/rules", params={"question": "refund.eligible"}).json()
        assert [r["status"] for r in history["rules"]] == ["active", "retired"]
        assert history["rules"][0]["id"] == rule["id"]
        assert client.get(f"/v1/routing/rules/{rule['id']}").json()["id"] == rule["id"]
        assert client.get(f"/v1/routing/rules/{uuid.uuid4()}").status_code == 404
        retired = client.get("/v1/routing/rules", params={"status": "retired"}).json()
        assert len(retired["rules"]) == 1

    def test_choice_question_without_an_auto_threshold(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.queue", 1500)
        refit(database_url, "ticket.queue")
        [rule] = refresh(client, "ticket.queue")
        # Five labels: temperature scaling can reorder, so training decisions are skipped.
        assert rule["eligible"] == 450
        assert rule["auto_threshold"] is None
        assert "auto: none" in rule["reason"]
        tiers: Counter[str] = Counter()
        for item in range(30_000, 30_200):
            body = decide(client, "ticket.queue", item)
            tiers[body["tier"]] += 1
            assert body["tier"] != "auto"
            assert set(body["answer_set"]) <= set(example_catalog.get("ticket.queue").options)  # type: ignore[union-attr]
            if body["tier"] == "human":
                assert body["reason"].startswith("no auto threshold meets the 5.0% error budget")
        assert tiers["human"] > 0

    def test_too_few_labels(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.handling_minutes", 150)
        [rule] = refresh(client, "ticket.handling_minutes")
        assert rule["calibrator_id"] is None
        assert rule["auto_threshold"] is None
        assert "fewer than min_labeled=200" in rule["reason"]
        body = decide(client, "ticket.handling_minutes", 1)
        assert body["tier"] == "human"
        assert body["reason"] == "only 150 eligible labeled decisions (min_labeled 200)"

    def test_raw_confidence_when_no_calibrator_is_required(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        seed_fake(database_url, example_catalog, "ticket.handling_minutes", 1000)
        [rule] = refresh(client, "ticket.handling_minutes")
        assert rule["auto_threshold"] is not None
        assert rule["verify_threshold"] is None
        assert rule["conformal"] is None
        tiers: Counter[str] = Counter()
        for item in range(40_000, 40_200):
            body = decide(client, "ticket.handling_minutes", item)
            tiers[body["tier"]] += 1
            assert body["calibrated_confidence"] is None
            assert body["answer_set"] is None
            if body["tier"] == "auto":
                assert body["reason"].startswith("raw confidence")
        assert tiers["auto"] > 0
        assert tiers["human"] > 0

    def test_calibrator_problems_fail_closed(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        # Temperature scaling needs a distribution; score questions have none.
        insert_calibrator(
            database_url, "ticket.handling_minutes", "temperature", {"temperature": 2.0}
        )
        body = decide(client, "ticket.handling_minutes", 1)
        assert body["tier"] == "human"
        assert "cannot score this decision" in body["reason"]
        assert body["calibrated_confidence"] is None
        insert_calibrator(database_url, "refund.eligible", "platt", {"slope": 1})
        body = decide(client, "refund.eligible", 1)
        assert body["tier"] == "human"
        assert "cannot be loaded" in body["reason"]
        bad = client.post("/v1/routing/refresh", json={"question": "refund.eligible"})
        assert bad.status_code == 422

    def test_corrupt_answer_set_parameters_are_ignored(
        self, client: TestClient, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        insert_calibrator(
            database_url,
            "refund.eligible",
            "platt",
            {"a": 1.0, "b": 0.0},
            trained_until=BASE - timedelta(days=1),
        )
        calibrator_id = client.get("/v1/calibrators").json()["calibrators"][0]["id"]
        policy = parse_routing({"questions": {"refund.eligible": POLICY}}).policy_for(
            "refund.eligible"
        )
        assert policy is not None
        rule = RoutingRuleRecord(
            question_id="refund.eligible",
            question_version=1,
            status="active",
            policy_fingerprint=policy.fingerprint(),
            policy=policy.model_dump(mode="json"),
            calibrator_id=uuid.UUID(calibrator_id),
            auto_threshold=0.999,
            verify_threshold=0.998,
            conformal={"method": "nope"},
            labeled=1,
            eligible=1,
            reason="test",
            evaluation={},
        )
        run(_insert(database_url, [rule]))
        body = decide(client, "refund.eligible", 1)
        assert body["tier"] == "human"
        assert body["answer_set"] is None
        assert "below every threshold (0.9980)" in body["reason"]

    def test_answer_sets_that_cannot_be_built(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        loose = {"min_labeled": 1, "min_accepted": 1, "require_calibrator": False}
        config = parse_routing(
            {
                "defaults": {**loose, "conformal_alpha": 0.1},
                "questions": {"refund.eligible": {}, "ticket.queue": {}},
            }
        )
        # One labeled decision is too few for a finite conformal quantile: no sets stored.
        insert_decision(database_url, "refund.eligible", True, True, {"true": 0.7, "false": 0.3})
        # A decision without a distribution cannot be used for sets at all.
        insert_decision(database_url, "ticket.queue", "billing", "billing", None)
        with make_client(database_url, example_catalog, config) as client:
            rules = {r["question"]: r for r in refresh(client)}
            assert rules["refund.eligible"]["conformal"] is None
            assert "every label is in every set" in rules["refund.eligible"]["reason"]
            assert rules["ticket.queue"]["conformal"] is None
            assert "answer sets: none" in rules["ticket.queue"]["reason"]
            filtered = client.get("/v1/routing/rules", params={"version": 1, "limit": 1})
            assert len(filtered.json()["rules"]) == 1

    def test_mismatched_answer_set_parameters(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        policy = RoutingPolicy(require_calibrator=False, conformal_alpha=0.1)
        rule = RoutingRuleRecord(
            question_id="ticket.queue",
            question_version=1,
            status="active",
            policy_fingerprint=policy.fingerprint(),
            policy=policy.model_dump(mode="json"),
            calibrator_id=None,
            auto_threshold=None,
            verify_threshold=0.0,
            conformal={
                "method": "lac",
                "alpha": 0.1,
                "quantile": 0.5,
                "labels": ["x", "y"],
                "calibration_count": 10,
            },
            labeled=1,
            eligible=1,
            reason="test",
            evaluation={},
        )
        run(_insert(database_url, [rule]))
        config = RoutingConfig(questions={"ticket.queue": policy})
        with make_client(database_url, example_catalog, config) as client:
            # The stored labels do not match the question's: no set, but still routed.
            body = decide(client, "ticket.queue", 1)
            assert body["tier"] == "verify"
            assert body["answer_set"] is None

    def test_refresher_needs_a_policy(self, example_catalog: QuestionCatalog) -> None:
        refresher = RoutingRefresher(RoutingConfig(), None, None, None)  # type: ignore[arg-type]
        with pytest.raises(SelectionError, match="no routing policy covers"):
            run(refresher.refresh(example_catalog.get("refund.eligible")))

    def test_cache_is_used_and_invalidated(
        self, database_url: str, example_catalog: QuestionCatalog, routing: RoutingConfig
    ) -> None:
        with make_client(database_url, example_catalog, routing, ttl=60.0) as cached:
            assert decide(cached, "refund.eligible", 1)["reason"].endswith("not calibrated yet")
            # Activating behind the server's back is not seen until the cache expires...
            run(_insert(database_url, [_platt("refund.eligible")]))
            assert decide(cached, "refund.eligible", 2)["reason"].endswith("not calibrated yet")
            # ...but a refresh through the API clears it.
            refresh(cached, "refund.eligible")
            body = decide(cached, "refund.eligible", 3)
            assert body["calibrated_confidence"] is not None

    def test_refresh_errors_and_refresh_all(self, client: TestClient) -> None:
        assert client.post("/v1/routing/refresh", json={"question": "nope"}).status_code == 404
        rules = refresh(client)
        assert {r["question"] for r in rules} == {
            "refund.eligible",
            "ticket.queue",
            "ticket.handling_minutes",
        }

    def test_refresh_needs_a_policy(
        self, database_url: str, example_catalog: QuestionCatalog
    ) -> None:
        only = parse_routing({"questions": {"ticket.queue": {}}})
        with make_client(database_url, example_catalog, only) as one:
            missing = one.post("/v1/routing/refresh", json={"question": "refund.eligible"})
            assert missing.status_code == 404
            assert "no routing policy" in missing.json()["detail"]
            assert [r["question"] for r in refresh(one)] == ["ticket.queue"]
            status = one.get("/v1/routing/status").json()["questions"]
            assert {q["question"]: q["policy_fingerprint"] is None for q in status} == {
                "refund.eligible": True,
                "ticket.queue": False,
                "ticket.handling_minutes": True,
            }


def _platt(question: str) -> CalibratorRecord:
    return CalibratorRecord(
        question_id=question,
        question_version=1,
        version=1,
        method="platt",
        params={"a": 1.0, "b": 0.0},
        status="active",
        reason="test",
        train_count=1,
        holdout_count=1,
        trained_from=BASE,
        trained_until=BASE,
        evaluation={},
    )


# --- CLI -----------------------------------------------------------------------------------


class TestCli:
    questions = str(EXAMPLES / "questions.yaml")

    def test_check(self, tmp_path: Path) -> None:
        result = runner.invoke(
            cli, ["routing", "check", str(EXAMPLES / "routing.yaml"), "-q", self.questions]
        )
        assert result.exit_code == 0, result.output
        assert "refund.eligible@v1: [" in result.output
        assert "verify error <= 25.0%" in result.output
        assert "answer sets at 90% coverage" in result.output

        partial = tmp_path / "partial.yaml"
        partial.write_text(
            "questions:\n  ticket.queue: {require_calibrator: false}\n  typo.q: {}\n"
        )
        result = runner.invoke(cli, ["routing", "check", str(partial), "-q", self.questions])
        assert result.exit_code == 1
        assert "refund.eligible@v1: no policy" in result.output
        assert "raw confidence allowed" in result.output
        assert "['typo.q']" in result.output

        bad = tmp_path / "bad.yaml"
        bad.write_text("questions: {a.b: {max_error_rate: 2}}\n")
        result = runner.invoke(cli, ["routing", "check", str(bad), "-q", self.questions])
        assert result.exit_code == 1
        assert "invalid routing policy" in result.output

    def test_refresh(self, database_url: str, example_catalog: QuestionCatalog) -> None:
        seed_fake(database_url, example_catalog, "refund.eligible", 1500)
        refit(database_url, "refund.eligible")
        routing_file = str(EXAMPLES / "routing.yaml")
        args = ["-r", routing_file, "-q", self.questions, "--url", database_url]
        result = runner.invoke(cli, ["routing", "refresh", "refund.eligible", *args])
        assert result.exit_code == 0, result.output
        assert "refund.eligible@v1: auto 0." in result.output
        assert "answer sets yes (1500 of 1500 labeled decisions used)" in result.output

        result = runner.invoke(cli, ["routing", "refresh", *args])
        assert result.exit_code == 0, result.output
        assert result.output.count("@v1:") == 3
        assert "ticket.queue@v1: auto none, verify none, answer sets no" in result.output

        result = runner.invoke(cli, ["routing", "refresh", "nope", *args])
        assert result.exit_code == 1

        result = runner.invoke(
            cli,
            [
                "thresholds",
                "select",
                "refund.eligible",
                "-q",
                self.questions,
                "--url",
                database_url,
            ],
        )
        assert result.exit_code == 0, result.output
        assert "1050 reused (the calibrator preserves their order)" in result.output

    def test_refresh_errors(
        self, database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DGATE_ROUTING_FILE", raising=False)
        result = runner.invoke(
            cli, ["routing", "refresh", "-q", self.questions, "--url", database_url]
        )
        assert result.exit_code == 2
        assert "pass --routing" in result.output

        only = tmp_path / "only.yaml"
        only.write_text("questions:\n  ticket.queue: {}\n")
        args = ["-r", str(only), "-q", self.questions, "--url", database_url]
        result = runner.invoke(cli, ["routing", "refresh", "refund.eligible", *args])
        assert result.exit_code == 1
        assert "no routing policy covers refund.eligible" in result.output

        insert_calibrator(database_url, "ticket.queue", "platt", {"slope": 1})
        result = runner.invoke(cli, ["routing", "refresh", *args])
        assert result.exit_code == 1
        assert "cannot be loaded" in result.output

        monkeypatch.setenv("DGATE_ROUTING_FILE", str(tmp_path / "missing.yaml"))
        result = runner.invoke(
            cli, ["routing", "refresh", "-q", self.questions, "--url", database_url]
        )
        assert result.exit_code == 1
        assert "cannot read" in result.output
