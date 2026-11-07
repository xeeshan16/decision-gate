"""Shadow mode: the statistics, the runner, storage, the API and the CLI."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from decision_gate.app import create_app
from decision_gate.backends import FakeBackend, HTTPDecisionBackend, LLMJSONBackend
from decision_gate.backends.base import AnyQuestion, DecisionInput, RawDecision
from decision_gate.backends.factory import build_shadow_backend
from decision_gate.backends.fake import hash_oracle
from decision_gate.cli import app as cli
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.models import DecisionRecord, OutcomeRecord, ShadowDecisionRecord
from decision_gate.db.shadow import PairedRow, ShadowRepository, ShadowSummary
from decision_gate.questions import ChoiceQuestion, PredicateQuestion, QuestionCatalog
from decision_gate.routing import RoutingPolicy, sample_unit
from decision_gate.shadow import (
    ShadowCompareError,
    _verdict,
    binomial_cdf,
    compare,
    mcnemar_exact,
    paired_difference_interval,
    percentile,
)
from decision_gate.shadow_runner import ShadowRunner, shadow_sample_unit
from tests.conftest import EXAMPLES
from tests.test_refit import BASE, _insert, run, seed_fake

runner = CliRunner()
QUESTIONS = str(EXAMPLES / "questions.yaml")
ROUTING = str(EXAMPLES / "routing.yaml")


# --- Statistics ------------------------------------------------------------------------


def _binomial_cdf_exact(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


@pytest.mark.parametrize(("k", "n", "p"), [(0, 5, 0.5), (3, 10, 0.5), (7, 20, 0.3), (19, 20, 0.9)])
def test_binomial_cdf_matches_the_exact_sum(k: int, n: int, p: float) -> None:
    assert binomial_cdf(k, n, p) == pytest.approx(_binomial_cdf_exact(k, n, p), abs=1e-10)


def test_binomial_cdf_edges() -> None:
    assert binomial_cdf(-1, 10, 0.5) == 0.0
    assert binomial_cdf(10, 10, 0.5) == 1.0
    assert binomial_cdf(0, 0, 0.5) == 1.0
    with pytest.raises(ValueError, match="need n"):
        binomial_cdf(1, -1, 0.5)
    with pytest.raises(ValueError, match="need n"):
        binomial_cdf(1, 3, 1.5)


def test_mcnemar_exact_hand_computed() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    # min(b, c) = 1 of 10 discordant: 2 * (1 + 10) / 1024.
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert mcnemar_exact(9, 1) == pytest.approx(22 / 1024)
    # 0 of 6: 2 / 64.
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(5, 5) == 1.0
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar_exact(-1, 2)


def test_paired_difference_interval_hand_computed() -> None:
    low, high = paired_difference_interval(10, 20, 100, 0.95)
    half = 1.959963984540054 * math.sqrt((0.30 - 0.01) / 100)
    assert low == pytest.approx(0.1 - half)
    assert high == pytest.approx(0.1 + half)
    assert paired_difference_interval(0, 0, 50) == (0.0, 0.0)
    assert paired_difference_interval(0, 10, 10) == (1.0, 1.0)
    with pytest.raises(ValueError, match="total"):
        paired_difference_interval(0, 0, 0)
    with pytest.raises(ValueError, match="exceed"):
        paired_difference_interval(6, 6, 10)


def test_percentile() -> None:
    assert percentile([], 50) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


# --- Comparison on synthetic pairs -----------------------------------------------------


def synthetic_pairs(
    n: int,
    *,
    primary_skill: float,
    candidate_skill: float,
    seed: int = 7,
    distributions: bool = True,
    identical: bool = False,
) -> list[PairedRow]:
    """Paired rows for a predicate question: each side is right with its own skill.

    Each side reports a confidence correlated with whether it is right, like a model that
    knows something about its own errors.
    """
    rng = np.random.default_rng(seed)
    rows: list[PairedRow] = []
    for i in range(n):
        truth = bool(rng.random() < 0.5)
        p_ok = bool(rng.random() < primary_skill)
        c_ok = p_ok if identical else bool(rng.random() < candidate_skill)
        p_conf = float(np.clip(rng.beta(8, 2) if p_ok else rng.beta(4, 3), 0.51, 0.999))
        c_conf = (
            p_conf
            if identical
            else float(np.clip(rng.beta(8, 2) if c_ok else rng.beta(4, 3), 0.51, 0.999))
        )
        p_answer = truth if p_ok else not truth
        c_answer = truth if c_ok else not truth

        def dist(answer: bool, conf: float) -> dict[str, float] | None:
            if not distributions:
                return None
            p_true = conf if answer else 1.0 - conf
            return {"true": p_true, "false": 1.0 - p_true}

        rows.append(
            PairedRow(
                decision_id=uuid.uuid4(),
                created_at=BASE + timedelta(seconds=i),
                outcome=truth,
                primary_answer=p_answer,
                primary_confidence=p_conf,
                primary_probabilities=dist(p_answer, p_conf),
                primary_correct=p_ok,
                primary_latency_ms=10.0 + i % 7,
                candidate_answer=c_answer,
                candidate_confidence=c_conf,
                candidate_probabilities=dist(c_answer, c_conf),
                candidate_latency_ms=20.0 + i % 5,
                candidate_model_version="cand-1",
            )
        )
    return rows


LOOSE = RoutingPolicy(max_error_rate=0.15, min_accepted=30)


def test_compare_promotes_a_clearly_better_candidate(predicate: PredicateQuestion) -> None:
    rows = synthetic_pairs(1200, primary_skill=0.70, candidate_skill=0.85)
    result = compare(predicate, "cand", rows, policy=LOOSE)
    assert result.verdict == "promote", result.reasons
    assert result.paired == 1200
    assert result.train == 600
    assert result.holdout == 600
    assert result.accuracy is not None
    assert result.accuracy.delta > 0.08
    assert result.accuracy.p_value < 1e-6
    assert result.accuracy.interval[0] > 0
    assert (
        sum(
            (
                result.accuracy.both_correct,
                result.accuracy.primary_only,
                result.accuracy.candidate_only,
                result.accuracy.both_wrong,
            )
        )
        == 600
    )
    assert result.primary is not None
    assert result.challenger is not None
    assert result.challenger.name == "cand"
    assert result.challenger.model_versions == ("cand-1",)
    assert result.challenger.auto.coverage > result.primary.auto.coverage
    assert result.challenger.latency_p95_ms is not None
    assert result.challenger.latency_p95_ms > result.primary.latency_p95_ms  # type: ignore[operator]
    assert result.reasons[0].startswith("significantly more accurate")
    assert result.options["holdout"] == 0.5
    assert result.policy["max_error_rate"] == 0.15
    assert result.question_key == "refund.eligible@v1"


def test_compare_keeps_an_identical_candidate(predicate: PredicateQuestion) -> None:
    rows = synthetic_pairs(600, primary_skill=0.8, candidate_skill=0.8, identical=True)
    result = compare(predicate, "same", rows, policy=LOOSE)
    assert result.verdict == "keep"
    assert result.reasons[0] == "no meaningful improvement"
    assert result.accuracy is not None
    assert result.accuracy.p_value == 1.0
    assert result.accuracy.delta == 0.0
    assert result.primary is not None
    assert result.challenger is not None
    assert result.challenger.auto.coverage == result.primary.auto.coverage


def test_compare_keeps_a_worse_candidate(predicate: PredicateQuestion) -> None:
    rows = synthetic_pairs(1000, primary_skill=0.85, candidate_skill=0.70)
    result = compare(predicate, "worse", rows, policy=LOOSE, method="platt")
    assert result.verdict == "keep"
    assert result.reasons[0].startswith("significantly less accurate")
    assert result.primary is not None
    assert result.primary.method == "platt"


def test_compare_blocks_an_unreliable_candidate(predicate: PredicateQuestion) -> None:
    rows = synthetic_pairs(1000, primary_skill=0.70, candidate_skill=0.85)
    summary = ShadowSummary("refund.eligible", 1, "cand", 1100, 1000, 80, 20, 1000, None)
    result = compare(predicate, "cand", rows, summary=summary, policy=LOOSE)
    assert result.verdict == "keep"
    assert result.reasons[0].startswith("the candidate failed 100 of 1100 calls")
    assert result.reliability is not None
    assert result.reliability.rate == pytest.approx(100 / 1100)
    # Under a looser failure limit the same evidence promotes.
    relaxed = compare(predicate, "cand", rows, summary=summary, policy=LOOSE, max_failure_rate=0.1)
    assert relaxed.verdict == "promote"


def test_compare_needs_enough_pairs(predicate: PredicateQuestion) -> None:
    result = compare(predicate, "cand", synthetic_pairs(50, primary_skill=0.7, candidate_skill=0.9))
    assert result.verdict == "insufficient"
    assert "50 paired labeled decisions" in result.reasons[0]
    assert result.accuracy is None
    assert result.primary is None
    empty = compare(predicate, "cand", [])
    assert empty.verdict == "insufficient"
    assert empty.holdout == 0
    assert empty.train == 0
    # Enough pairs, but too few left to train on.
    rows = synthetic_pairs(200, primary_skill=0.7, candidate_skill=0.9)
    assert compare(predicate, "cand", rows, holdout=0.95).verdict == "insufficient"


def test_compare_without_distributions_uses_platt_or_isotonic(
    predicate: PredicateQuestion,
) -> None:
    rows = synthetic_pairs(800, primary_skill=0.7, candidate_skill=0.85, distributions=False)
    result = compare(predicate, "cand", rows, policy=LOOSE)
    assert result.primary is not None
    assert result.primary.method in {"platt", "isotonic"}
    isotonic = compare(predicate, "cand", rows, policy=LOOSE, method="isotonic")
    assert isotonic.challenger is not None
    assert isotonic.challenger.method == "isotonic"
    with pytest.raises(ShadowCompareError, match="temperature"):
        compare(predicate, "cand", rows, method="temperature")


def test_compare_validates_options(predicate: PredicateQuestion) -> None:
    rows: list[PairedRow] = []
    with pytest.raises(ShadowCompareError, match="holdout"):
        compare(predicate, "c", rows, holdout=1.0)
    with pytest.raises(ShadowCompareError, match="alpha"):
        compare(predicate, "c", rows, alpha=0.0)
    with pytest.raises(ShadowCompareError, match="min_coverage_gain"):
        compare(predicate, "c", rows, min_coverage_gain=1.0)
    with pytest.raises(ShadowCompareError, match="max_failure_rate"):
        compare(predicate, "c", rows, max_failure_rate=2.0)
    with pytest.raises(ShadowCompareError, match="min_paired"):
        compare(predicate, "c", rows, min_paired=10)


def test_verdict_blocks_a_candidate_that_automates_less(predicate: PredicateQuestion) -> None:
    rows = synthetic_pairs(1000, primary_skill=0.85, candidate_skill=0.70)
    result = compare(predicate, "c", rows, policy=LOOSE)
    assert result.accuracy is not None
    assert result.primary is not None
    assert result.challenger is not None
    # Same sides with the accuracy test made inconclusive: coverage alone decides.
    inconclusive = type(result.accuracy)(**{**_fields(result.accuracy), "p_value": 0.5})
    verdict, reasons = _verdict(
        inconclusive, result.primary, result.challenger, None, 0.05, 0.02, 0.01
    )
    assert verdict == "keep"
    assert reasons[0].startswith("automates less")
    swapped, why = _verdict(inconclusive, result.challenger, result.primary, None, 0.05, 0.02, 0.01)
    assert swapped == "promote"
    assert why[0].startswith("automates more")


def _fields(obj: Any) -> dict[str, Any]:
    return {name: getattr(obj, name) for name in obj.__slots__}


# --- The runner ------------------------------------------------------------------------


def test_shadow_sampling_is_independent_of_verify_sampling() -> None:
    ids = [uuid.UUID(int=i) for i in range(2000)]
    shadow = np.array([shadow_sample_unit(i) for i in ids])
    verify = np.array([sample_unit(i) for i in ids])
    assert 0.45 < shadow.mean() < 0.55
    assert abs(np.corrcoef(shadow, verify)[0, 1]) < 0.1
    # The decisions verify-sampled at 5% are shadowed at about the overall rate.
    assert 0.3 < np.mean(shadow[verify < 0.05] < 0.5) < 0.7


class SlowBackend:
    """Answers only when released, or never (to time out)."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    @property
    def name(self) -> str:
        return "slow"

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        await self.release.wait()
        return RawDecision(answer=True, confidence=0.9, backend="slow", model_version="s1")


class BrokenBackend:
    def __init__(self, answer: Any = None) -> None:
        self._answer = answer
        self.closed = False

    @property
    def name(self) -> str:
        return "broken"

    async def decide(self, question: AnyQuestion, data: DecisionInput) -> RawDecision:
        if self._answer is None:
            raise RuntimeError("model exploded " + "x" * 600)
        return RawDecision(answer=self._answer, confidence=0.9, backend="b", model_version="b1")

    async def aclose(self) -> None:
        self.closed = True


class FailingRepository(ShadowRepository):
    def __init__(self) -> None:
        pass

    async def add(self, record: ShadowDecisionRecord) -> tuple[ShadowDecisionRecord, bool]:
        raise RuntimeError("database down")


def test_runner_rejects_bad_options() -> None:
    backend = FakeBackend()
    with pytest.raises(ValueError, match="sample_rate"):
        ShadowRunner(backend, FailingRepository(), candidate="c", timeout_s=1, sample_rate=2)
    with pytest.raises(ValueError, match="max_pending"):
        ShadowRunner(backend, FailingRepository(), candidate="c", timeout_s=1, max_pending=0)
    with pytest.raises(ValueError, match="candidate"):
        ShadowRunner(backend, FailingRepository(), candidate="", timeout_s=1)


def test_runner_filters_questions_and_samples(predicate: PredicateQuestion) -> None:
    async def work() -> None:
        shadow = ShadowRunner(
            FakeBackend(),
            FailingRepository(),
            candidate="c",
            timeout_s=1,
            sample_rate=0.0,
            questions=["refund.eligible"],
        )
        assert shadow.shadows("refund.eligible")
        assert not shadow.shadows("ticket.queue")
        assert not shadow.submit(predicate, uuid.uuid4(), {"a": 1})
        assert shadow.counters.skipped == 1
        await shadow.drain()  # nothing in flight

    run(work())


def test_runner_drops_beyond_max_pending_and_drains(predicate: PredicateQuestion) -> None:
    async def work() -> None:
        backend = SlowBackend()
        shadow = ShadowRunner(
            backend, FailingRepository(), candidate="c", timeout_s=5, max_pending=2
        )
        assert shadow.submit(predicate, uuid.uuid4(), {"a": 1})
        assert shadow.submit(predicate, uuid.uuid4(), {"a": 2})
        assert not shadow.submit(predicate, uuid.uuid4(), {"a": 3})
        assert shadow.pending == 2
        assert shadow.counters.dropped == 1
        backend.release.set()
        await shadow.drain()
        assert shadow.pending == 0
        assert shadow.counters.completed == 2
        # The repository is down: answers are lost but nothing raises.
        assert shadow.counters.store_errors == 2

    run(work())


def test_runner_drain_timeout_cancels(predicate: PredicateQuestion) -> None:
    async def work() -> None:
        shadow = ShadowRunner(SlowBackend(), FailingRepository(), candidate="c", timeout_s=30)
        shadow.submit(predicate, uuid.uuid4(), {"a": 1})
        await shadow.drain(timeout_s=0.05)
        assert shadow.pending == 0

    run(work())


def test_runner_records_failures(predicate: PredicateQuestion, choice: ChoiceQuestion) -> None:
    async def work() -> None:
        decision = uuid.uuid4()
        slow = ShadowRunner(SlowBackend(), FailingRepository(), candidate="c", timeout_s=0.01)
        timed_out = await slow.answer(predicate, decision, {}, "live")
        assert timed_out.status == "timeout"
        assert timed_out.answer is None
        assert timed_out.error == "no answer within 0.01s"

        broken = ShadowRunner(BrokenBackend(), FailingRepository(), candidate="c", timeout_s=1)
        failed = await broken.answer(predicate, decision, {}, "live")
        assert failed.status == "error"
        assert failed.error is not None
        assert failed.error.startswith("RuntimeError: model")
        assert len(failed.error) == 500
        assert failed.error.endswith("...")

        invalid = ShadowRunner(
            BrokenBackend("nope"), FailingRepository(), candidate="c", timeout_s=1
        )
        bad = await invalid.answer(choice, decision, {}, "replay")
        assert bad.status == "error"
        assert "must be one of" in (bad.error or "")
        assert bad.mode == "replay"
        assert bad.latency_ms >= 0

        await broken._run(predicate, decision, {}, "live")
        assert broken.counters.failed == 1
        assert broken.counters.store_errors == 1
        await broken.aclose()
        assert isinstance(broken.backend, BrokenBackend)
        assert broken.backend.closed
        await slow.aclose()  # a backend without aclose

    run(work())


# --- Storage and replay ----------------------------------------------------------------


async def _repository(database_url: str) -> tuple[Any, ShadowRepository]:
    engine = make_engine(database_url)
    return engine, ShadowRepository(make_sessionmaker(engine))


def _decision(question: str = "refund.eligible", data: Any = None) -> DecisionRecord:
    return DecisionRecord(
        id=uuid.uuid4(),
        question_id=question,
        question_version=1,
        input_sha256="0" * 64,
        input=data,
        answer=True,
        confidence=0.8,
        probabilities=None,
        backend="fake",
        model_version="m",
        latency_ms=1.0,
        created_at=BASE,
    )


def test_repository_add_is_idempotent_per_candidate(database_url: str) -> None:
    decision = _decision(data={"a": 1})
    run(_insert(database_url, [decision]))

    def shadow(candidate: str, status: str = "ok") -> ShadowDecisionRecord:
        return ShadowDecisionRecord(
            decision_id=decision.id,
            candidate=candidate,
            mode="live",
            question_id="refund.eligible",
            question_version=1,
            status=status,
            answer=True if status == "ok" else None,
            confidence=0.7 if status == "ok" else None,
            latency_ms=2.0,
        )

    async def work() -> None:
        engine, repository = await _repository(database_url)
        try:
            first, created = await repository.add(shadow("a"))
            assert created
            again, created = await repository.add(shadow("a", "error"))
            assert not created
            assert again.id == first.id
            assert again.status == "ok"
            await repository.add(shadow("b", "timeout"))
            answers = await repository.for_decision(decision.id)
            assert [a.candidate for a in answers] == ["a", "b"]
            assert answers[0].question_key == "refund.eligible@v1"
            orphan = shadow("a")
            orphan.decision_id = uuid.uuid4()
            with pytest.raises(Exception, match=r"(?i)foreign key"):
                await repository.add(orphan)
            summaries = await repository.summaries(question_id="refund.eligible")
            assert [(s.candidate, s.ok, s.timeouts, s.labeled) for s in summaries] == [
                ("a", 1, 0, 0),
                ("b", 0, 1, 0),
            ]
            assert summaries[1].failure_rate == 1.0
            assert ShadowSummary("q", 1, "c", 0, 0, 0, 0, 0, None).failure_rate == 0.0
            filtered = await repository.summaries(
                question_version=1,
                candidate="a",
                mode="live",
                since=BASE - timedelta(hours=1),
                until=BASE + timedelta(hours=1),
            )
            assert len(filtered) == 1
            assert await repository.summaries(mode="replay") == []
        finally:
            await engine.dispose()

    run(work())


def test_replay_answers_stored_decisions_once(
    database_url: str, example_catalog: QuestionCatalog
) -> None:
    seed_fake(database_url, example_catalog, "refund.eligible", 30)
    # Two decisions without a stored input and one without an outcome.
    blind = [_decision(data=None), _decision(data=None)]
    unlabeled = _decision(data={"item": 999})
    run(_insert(database_url, [*blind, unlabeled]))
    outcomes = [
        OutcomeRecord(decision_id=d.id, outcome=True, correct=True, source="t", observed_at=BASE)
        for d in blind
    ]
    run(_insert(database_url, outcomes))
    question = example_catalog.get("refund.eligible")

    async def work() -> None:
        engine, repository = await _repository(database_url)
        try:
            shadow = ShadowRunner(
                FakeBackend(min_skill=0.8, model_version="fake-2"),
                repository,
                candidate="better",
                timeout_s=5,
            )
            first = await shadow.replay(question, limit=20, concurrency=4)
            assert (first.attempted, first.ok, first.failed) == (20, 20, 0)
            assert first.skipped_without_input == 2
            rest = await shadow.replay(question)
            assert (rest.attempted, rest.duplicates) == (10, 0)
            everything = await shadow.replay(question, labeled_only=False)
            assert everything.attempted == 1  # the unlabeled decision
            assert (await shadow.replay(question)).attempted == 0
            rows = await repository.paired_rows("refund.eligible", 1, "better", mode="replay")
            assert len(rows) == 30
            assert rows == sorted(rows, key=lambda r: r.created_at)
            assert {r.candidate_model_version for r in rows} == {"fake-2"}
            window = await repository.paired_rows(
                "refund.eligible",
                1,
                "better",
                since=BASE + timedelta(seconds=10),
                until=BASE + timedelta(seconds=20),
            )
            assert len(window) == 10
            assert await repository.paired_rows("refund.eligible", 1, "better", mode="live") == []
            with pytest.raises(ValueError, match="at least 1"):
                await shadow.replay(question, limit=0)
        finally:
            await engine.dispose()

    run(work())


def test_replay_counts_a_lost_race_as_duplicate(
    database_url: str, example_catalog: QuestionCatalog
) -> None:
    seed_fake(database_url, example_catalog, "refund.eligible", 3)
    question = example_catalog.get("refund.eligible")

    class RacingRepository(ShadowRepository):
        async def replay_targets(self, *args: Any, **kwargs: Any) -> Any:
            targets, skipped = await super().replay_targets(*args, **kwargs)
            return targets + targets[:1], skipped  # the same decision twice

    async def work() -> None:
        engine = make_engine(database_url)
        try:
            repository = RacingRepository(make_sessionmaker(engine))
            shadow = ShadowRunner(FakeBackend(), repository, candidate="c", timeout_s=5)
            result = await shadow.replay(question, concurrency=1)
            assert (result.attempted, result.ok, result.duplicates) == (4, 3, 1)
        finally:
            await engine.dispose()

    run(work())


# --- Settings and the backend factory --------------------------------------------------


def test_shadow_settings() -> None:
    off = Settings(shadow_backend="", shadow_name="")
    assert off.shadow_backend is None
    assert off.shadow_candidate is None
    assert build_shadow_backend(off) is None
    fake = Settings(
        shadow_backend="fake",
        shadow_questions=" refund.eligible, ,ticket.queue ",
        shadow_fake_min_skill=0.8,
    )
    assert fake.shadow_candidate == "fake"
    assert fake.shadow_question_ids == ["refund.eligible", "ticket.queue"]
    assert isinstance(build_shadow_backend(fake), FakeBackend)
    named = Settings(shadow_backend="fake", shadow_name="prompt-v3")
    assert named.shadow_candidate == "prompt-v3"
    http = Settings(shadow_backend="http", shadow_http_url="http://x", shadow_http_token="t")
    assert isinstance(build_shadow_backend(http), HTTPDecisionBackend)
    llm = Settings(
        shadow_backend="llm_json", shadow_llm_base_url="http://x/v1", shadow_llm_model="m"
    )
    assert isinstance(build_shadow_backend(llm), LLMJSONBackend)
    with pytest.raises(ValidationError, match="DGATE_SHADOW_HTTP_URL"):
        Settings(shadow_backend="http")
    with pytest.raises(ValidationError, match="DGATE_SHADOW_LLM_BASE_URL"):
        Settings(shadow_backend="llm_json", shadow_llm_base_url="http://x/v1")


# --- API -------------------------------------------------------------------------------


@pytest.fixture
def shadow_client(database_url: str, example_catalog: QuestionCatalog) -> Iterator[TestClient]:
    app = create_app(
        Settings(database_url=database_url, shadow_name="better"),
        catalog=example_catalog,
        backend=FakeBackend(overconfidence=2.0),
        shadow_backend=FakeBackend(min_skill=0.8, model_version="fake-2", seed="other"),
    )
    with TestClient(app) as client:
        yield client


def _drain(client: TestClient) -> None:
    shadow = client.app.state.shadow  # type: ignore[attr-defined]
    client.portal.call(shadow.drain)  # type: ignore[union-attr]


def test_api_shadows_live_decisions(
    shadow_client: TestClient, example_catalog: QuestionCatalog
) -> None:
    question = example_catalog.get("refund.eligible")
    ids = []
    for i in range(60):
        data = {"item": i}
        body = shadow_client.post(
            "/v1/decide",
            json={"question": "refund.eligible", "input": data, "idempotency_key": f"k{i}"},
        ).json()
        ids.append(body["id"])
        shadow_client.post(
            "/v1/outcomes", json={"decision_id": body["id"], "outcome": hash_oracle(question, data)}
        )
    # An idempotent replay is not shadowed again.
    replayed = shadow_client.post(
        "/v1/decide",
        json={"question": "refund.eligible", "input": {"item": 0}, "idempotency_key": "k0"},
    )
    assert replayed.status_code == 200
    _drain(shadow_client)

    status = shadow_client.get("/v1/shadow/status").json()
    assert status["shadow"]["enabled"] is True
    assert status["shadow"]["candidate"] == "better"
    assert status["shadow"]["counters"]["submitted"] == 60
    assert status["shadow"]["counters"]["completed"] == 60
    [summary] = status["candidates"]
    assert (summary["total"], summary["ok"], summary["labeled"]) == (60, 60, 60)
    assert shadow_client.get("/v1/shadow/status?question=ticket.queue").json()["candidates"] == []

    [answer] = shadow_client.get(f"/v1/shadow/decisions/{ids[0]}").json()["answers"]
    assert answer["candidate"] == "better"
    assert answer["mode"] == "live"
    assert answer["status"] == "ok"
    assert answer["model_version"] == "fake-2"

    small = shadow_client.get("/v1/shadow/compare", params={"question": "refund.eligible"})
    assert small.status_code == 200
    assert small.json()["verdict"] == "insufficient"
    report = shadow_client.get(
        "/v1/shadow/compare",
        params={"question": "refund.eligible", "min_paired": 40, "method": "platt"},
    ).json()
    assert report["verdict"] in {"promote", "keep"}
    assert report["question"] == "refund.eligible"
    assert report["candidate_report"]["model_versions"] == ["fake-2"]
    assert len(report["accuracy"]["interval"]) == 2
    assert report["reliability"]["failures"] == 0

    bad = shadow_client.get(
        "/v1/shadow/compare",
        params={"question": "ticket.handling_minutes", "method": "temperature", "min_paired": 40},
    )
    # No pairs for that question: too few to compare, before the method matters.
    assert bad.json()["verdict"] == "insufficient"
    assert shadow_client.get("/v1/shadow/compare", params={"question": "nope"}).status_code == 404


def test_api_replay_and_compare(
    database_url: str, example_catalog: QuestionCatalog, shadow_client: TestClient
) -> None:
    seed_fake(database_url, example_catalog, "refund.eligible", 400)
    seed_fake(database_url, example_catalog, "ticket.handling_minutes", 60)
    replay = shadow_client.post(
        "/v1/shadow/replay", json={"question": "refund.eligible", "limit": 1000}
    )
    assert replay.status_code == 200, replay.text
    body = replay.json()
    assert body["question"] == "refund.eligible"
    assert (body["attempted"], body["ok"]) == (400, 400)
    report = shadow_client.get(
        "/v1/shadow/compare",
        params={"question": "refund.eligible", "mode": "replay", "candidate": "better"},
    ).json()
    assert report["verdict"] == "promote", report["reasons"]
    assert report["accuracy"]["delta"] > 0
    assert shadow_client.post("/v1/shadow/replay", json={"question": "nope"}).status_code == 404

    shadow_client.post("/v1/shadow/replay", json={"question": "ticket.handling_minutes"})
    temperature = shadow_client.get(
        "/v1/shadow/compare",
        params={"question": "ticket.handling_minutes", "method": "temperature", "min_paired": 40},
    )
    assert temperature.status_code == 422
    assert "temperature" in temperature.json()["detail"]


def test_api_without_shadow_mode(database_url: str, example_catalog: QuestionCatalog) -> None:
    app = create_app(
        Settings(database_url=database_url), catalog=example_catalog, backend=FakeBackend()
    )
    with TestClient(app) as client:
        status = client.get("/v1/shadow/status").json()
        assert status == {"shadow": {"enabled": False}, "candidates": []}
        replay = client.post("/v1/shadow/replay", json={"question": "refund.eligible"})
        assert replay.status_code == 409
        compare_ = client.get("/v1/shadow/compare", params={"question": "refund.eligible"})
        assert compare_.status_code == 422
        named = client.get(
            "/v1/shadow/compare", params={"question": "refund.eligible", "candidate": "x"}
        )
        assert named.json()["verdict"] == "insufficient"


def test_app_builds_the_shadow_backend_from_settings(
    database_url: str, example_catalog: QuestionCatalog
) -> None:
    settings = Settings(database_url=database_url, shadow_backend="fake", shadow_sample_rate=0.5)
    app = create_app(settings, catalog=example_catalog, backend=FakeBackend())
    with TestClient(app) as client:
        config = client.get("/v1/shadow/status").json()["shadow"]
        assert config["candidate"] == "fake"
        assert config["sample_rate"] == 0.5


# --- CLI -------------------------------------------------------------------------------


def _seed_compared(database_url: str, catalog: QuestionCatalog, n: int = 400) -> None:
    seed_fake(database_url, catalog, "refund.eligible", n)
    question = catalog.get("refund.eligible")

    async def work() -> None:
        engine, repository = await _repository(database_url)
        try:
            shadow = ShadowRunner(
                FakeBackend(min_skill=0.8, model_version="fake-2", seed="other"),
                repository,
                candidate="better",
                timeout_s=5,
            )
            await shadow.replay(question, limit=n)
        finally:
            await engine.dispose()

    run(work())


def test_cli_status_compare_and_exit_codes(
    sqlite_url: str, example_catalog: QuestionCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DGATE_ROUTING_FILE", raising=False)
    monkeypatch.delenv("DGATE_SHADOW_NAME", raising=False)
    monkeypatch.delenv("DGATE_SHADOW_BACKEND", raising=False)
    empty = runner.invoke(cli, ["shadow", "status", "--url", sqlite_url])
    assert empty.exit_code == 0
    assert "no shadow answers" in empty.output

    _seed_compared(sqlite_url, example_catalog)
    status = runner.invoke(cli, ["shadow", "status", "--url", sqlite_url])
    assert status.exit_code == 0, status.output
    assert "better" in status.output
    assert "400" in status.output

    base = ["shadow", "compare", "refund.eligible", "-q", QUESTIONS, "--url", sqlite_url]
    no_name = runner.invoke(cli, base)
    assert no_name.exit_code == 2
    assert "--candidate" in no_name.output

    promoted = runner.invoke(cli, [*base, "--candidate", "better", "-r", ROUTING])
    assert promoted.exit_code == 0, promoted.output
    assert "PROMOTE" in promoted.output
    assert "accuracy delta" in promoted.output
    assert "candidate failures 0/400" in promoted.output

    monkeypatch.setenv("DGATE_SHADOW_NAME", "better")
    as_json = runner.invoke(cli, [*base, "--json", "--method", "platt"])
    assert as_json.exit_code == 0, as_json.output
    report = json.loads(as_json.output)
    assert report["verdict"] == "promote"
    assert report["candidate"] == "better"

    keep = runner.invoke(
        cli,
        [
            *base,
            "--candidate",
            "better",
            "--max-failure-rate",
            "0",
            "--min-paired",
            "40",
            "--holdout",
            "0.5",
            "--mode",
            "live",
        ],
    )
    assert keep.exit_code == 4
    assert "INSUFFICIENT" in keep.output

    bad_mode = runner.invoke(cli, [*base, "--mode", "sideways"])
    assert bad_mode.exit_code == 2
    bad_method = runner.invoke(cli, [*base, "--method", "magic"])
    assert bad_method.exit_code == 2

    seed_fake(sqlite_url, example_catalog, "ticket.handling_minutes", 60)
    score_q = example_catalog.get("ticket.handling_minutes")

    async def replay_scores() -> None:
        engine, repository = await _repository(sqlite_url)
        try:
            shadow = ShadowRunner(FakeBackend(), repository, candidate="better", timeout_s=5)
            await shadow.replay(score_q)
        finally:
            await engine.dispose()

    run(replay_scores())
    temperature = runner.invoke(
        cli,
        [
            "shadow",
            "compare",
            "ticket.handling_minutes",
            "-q",
            QUESTIONS,
            "--url",
            sqlite_url,
            "--method",
            "temperature",
            "--min-paired",
            "40",
        ],
    )
    assert temperature.exit_code == 1
    assert "temperature" in temperature.output


def test_cli_compare_keeps_a_worse_candidate(
    sqlite_url: str, example_catalog: QuestionCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DGATE_ROUTING_FILE", raising=False)
    seed_fake(sqlite_url, example_catalog, "refund.eligible", 400)
    question = example_catalog.get("refund.eligible")

    async def work() -> None:
        engine, repository = await _repository(sqlite_url)
        try:
            worse = FakeBackend(min_skill=0.5, overconfidence=3.0, model_version="w", seed="w")
            await ShadowRunner(worse, repository, candidate="worse", timeout_s=5).replay(question)
        finally:
            await engine.dispose()

    run(work())
    result = runner.invoke(
        cli,
        [
            "shadow",
            "compare",
            "refund.eligible",
            "--candidate",
            "worse",
            "-q",
            QUESTIONS,
            "--url",
            sqlite_url,
        ],
    )
    assert result.exit_code == 3, result.output
    assert "KEEP" in result.output


def test_cli_replay(
    sqlite_url: str, example_catalog: QuestionCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_fake(sqlite_url, example_catalog, "refund.eligible", 25)
    args = ["shadow", "replay", "refund.eligible", "-q", QUESTIONS, "--url", sqlite_url]
    monkeypatch.delenv("DGATE_SHADOW_BACKEND", raising=False)
    missing = runner.invoke(cli, args)
    assert missing.exit_code == 2
    assert "DGATE_SHADOW_BACKEND" in missing.output

    monkeypatch.setenv("DGATE_SHADOW_BACKEND", "fake")
    monkeypatch.setenv("DGATE_SHADOW_NAME", "cli-candidate")
    done = runner.invoke(cli, [*args, "--limit", "10", "--concurrency", "2"])
    assert done.exit_code == 0, done.output
    assert "cli-candidate: 10 replayed, 10 ok" in done.output
    rest = runner.invoke(cli, [*args, "--all"])
    assert "15 replayed" in rest.output
    unknown = runner.invoke(cli, ["shadow", "replay", "nope", "-q", QUESTIONS, "--url", sqlite_url])
    assert unknown.exit_code == 1


def test_decide_still_answers_when_the_candidate_fails(
    database_url: str, example_catalog: QuestionCatalog
) -> None:
    app = create_app(
        Settings(database_url=database_url, shadow_name="broken"),
        catalog=example_catalog,
        backend=FakeBackend(),
        shadow_backend=BrokenBackend(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/decide", json={"question": "refund.eligible", "input": {"item": 1}}
        )
        assert response.status_code == 201
        _drain(client)
        [summary] = client.get("/v1/shadow/status").json()["candidates"]
        assert summary["errors"] == 1
        assert summary["failure_rate"] == 1.0
    assert app.state.shadow.backend.closed  # closed on shutdown


def test_choice_question_pairs_score_the_candidate_against_the_truth(
    database_url: str, example_catalog: QuestionCatalog
) -> None:
    seed_fake(database_url, example_catalog, "ticket.queue", 120)
    question = example_catalog.get("ticket.queue")

    async def work() -> list[PairedRow]:
        engine, repository = await _repository(database_url)
        try:
            await ShadowRunner(
                FakeBackend(min_skill=0.9, seed="other"), repository, candidate="c", timeout_s=5
            ).replay(question)
            return await repository.paired_rows("ticket.queue", 1, "c")
        finally:
            await engine.dispose()

    rows = run(work())
    assert len(rows) == 120
    result = compare(question, "c", rows, min_paired=100, holdout=0.5)
    assert result.accuracy is not None
    candidate_right = sum(1 for r in rows[60:] if r.candidate_answer == r.outcome)
    assert result.accuracy.both_correct + result.accuracy.candidate_only == candidate_right
