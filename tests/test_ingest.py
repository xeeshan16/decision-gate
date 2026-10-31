"""Streaming outcome ingestion: JSON Lines, the Kafka source (with fakes) and the CLI."""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from decision_gate.app import create_app
from decision_gate.cli import app as cli
from decision_gate.config import Settings
from decision_gate.db import make_engine, make_sessionmaker
from decision_gate.db.repository import DecisionRepository, OutcomeRepository
from decision_gate.ingest import (
    IngestStats,
    JsonLinesSource,
    OutcomeIngestor,
    OutcomeSource,
    RejectReason,
    SourceMessage,
)
from decision_gate.kafka import KafkaOutcomeSource
from decision_gate.questions import QuestionCatalog
from decision_gate.service import OutcomeService

from .conftest import EXAMPLES

runner = CliRunner()


@pytest.fixture
def decisions(sqlite_url: str, example_catalog: QuestionCatalog) -> list[dict[str, Any]]:
    """Three recorded refund decisions."""
    app = create_app(Settings(database_url=sqlite_url), catalog=example_catalog)
    with TestClient(app) as client:
        return [
            client.post(
                "/v1/decide", json={"question": "refund.eligible", "input": {"order": i}}
            ).json()
            for i in range(3)
        ]


def event(decision: dict[str, Any], outcome: Any = None, **extra: Any) -> str:
    value = decision["answer"] if outcome is None else outcome
    return json.dumps({"decision_id": decision["id"], "outcome": value, **extra})


def ingest(url: str, catalog: QuestionCatalog, source: OutcomeSource, **kw: Any) -> IngestStats:
    async def run() -> IngestStats:
        engine = make_engine(url)
        try:
            sessions = make_sessionmaker(engine)
            service = OutcomeService(
                catalog, DecisionRepository(sessions), OutcomeRepository(sessions)
            )
            return await OutcomeIngestor(service, source).run(**kw)
        finally:
            await engine.dispose()

    return asyncio.run(run())


def stored_outcomes(url: str) -> dict[str, Any]:
    async def run() -> dict[str, Any]:
        engine = make_engine(url)
        try:
            repo = OutcomeRepository(make_sessionmaker(engine))
            stats = await repo.stats("refund.eligible")
            pairs = await repo.labeled("refund.eligible", 1)
            return {
                "labeled": stats[0].labeled,
                "sources": [o.source for _, o in pairs],
                "correct": [o.correct for _, o in pairs],
            }
        finally:
            await engine.dispose()

    return asyncio.run(run())


# --- JSON Lines ------------------------------------------------------------------------


def test_jsonl_ingest_records_rejects_and_commits(
    sqlite_url: str,
    example_catalog: QuestionCatalog,
    decisions: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    a, b, c = decisions
    lines = [
        event(a),
        "",
        event(b, not b["answer"], source="crm", observed_at="2026-10-30T09:00:00Z"),
        event(a),  # redelivery: duplicate, not an error
        event(a, not a["answer"]),  # conflicting label
        "not json",
        json.dumps({"decision_id": "nope", "outcome": True}),
        json.dumps({"decision_id": str(uuid.uuid4()), "outcome": True}),
        event(c, "maybe"),
    ]
    dlq = tmp_path / "dlq.jsonl"
    seen: list[str] = []
    source = JsonLinesSource(
        io.StringIO("\n".join(lines) + "\n"),
        dead_letter_path=dlq,
        on_reject=lambda m, r, d: seen.append(r),
    )
    stats = ingest(sqlite_url, example_catalog, source)

    assert stats.created == 2
    assert stats.duplicate == 1
    assert stats.rejected == {"conflict": 1, "malformed": 2, "unknown_decision": 1, "invalid": 1}
    assert stats.handled == 8
    assert source.committed == "line 9"  # every handled line is committed, rejects too
    assert seen == ["conflict", "malformed", "malformed", "unknown_decision", "invalid"]

    parked = [json.loads(line) for line in dlq.read_text().splitlines()]
    assert [p["position"] for p in parked] == ["line 5", "line 6", "line 7", "line 8", "line 9"]
    assert parked[1] == {
        "position": "line 6",
        "reason": "malformed",
        "detail": parked[1]["detail"],
        "payload": "not json",
    }
    assert parked[2]["detail"].startswith("decision_id:")

    stored = stored_outcomes(sqlite_url)
    assert stored["labeled"] == 2
    assert sorted(stored["sources"]) == ["crm", "jsonl"]
    assert sorted(stored["correct"]) == [False, True]


def test_max_messages_stops_early(
    sqlite_url: str, example_catalog: QuestionCatalog, decisions: list[dict[str, Any]]
) -> None:
    source = JsonLinesSource(io.StringIO("\n".join(event(d) for d in decisions)))
    stats = ingest(sqlite_url, example_catalog, source, max_messages=2)
    assert stats.created == 2
    assert source.committed == "line 2"
    assert ingest(sqlite_url, example_catalog, source, max_messages=0).handled == 0


def test_rejects_without_a_sink_are_only_counted(
    sqlite_url: str, example_catalog: QuestionCatalog, decisions: list[dict[str, Any]]
) -> None:
    source = JsonLinesSource(io.StringIO("{}\n"))
    assert ingest(sqlite_url, example_catalog, source).rejected == {"malformed": 1}


class BrokenSource:
    """A source whose database is gone: the error must surface and nothing is committed."""

    name = "broken"

    def __init__(self, decision: dict[str, Any]) -> None:
        self.payload = event(decision).encode()
        self.commits: list[str] = []
        self.dead: list[str] = []

    async def messages(self) -> AsyncIterator[SourceMessage]:
        yield SourceMessage(payload=self.payload, position="p0")

    async def commit(self, message: SourceMessage) -> None:
        self.commits.append(message.position)

    async def dead_letter(self, message: SourceMessage, reason: RejectReason, detail: str) -> None:
        self.dead.append(reason)  # pragma: no cover - must not happen


def test_transient_errors_propagate_without_commit(
    example_catalog: QuestionCatalog, decisions: list[dict[str, Any]], tmp_path: Path
) -> None:
    source = BrokenSource(decisions[0])
    missing_db = f"sqlite+aiosqlite:///{tmp_path / 'missing' / 'db.sqlite'}"
    with pytest.raises(Exception, match="unable to open database file"):
        ingest(missing_db, example_catalog, source)
    assert source.commits == []
    assert source.dead == []


# --- Kafka source, driven by in-memory fakes ------------------------------------------


@dataclass
class FakeRecord:
    topic: str
    partition: int
    offset: int
    value: bytes | None
    key: bytes | None = None


@dataclass
class FakeConsumer:
    records: list[FakeRecord]
    started: bool = False
    stopped: bool = False
    commits: list[dict[Any, int]] = field(default_factory=list)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def __aiter__(self) -> AsyncIterator[FakeRecord]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[FakeRecord]:
        for record in self.records:
            yield record

    async def commit(self, offsets: dict[Any, int]) -> None:
        self.commits.append(offsets)


@dataclass
class FakeProducer:
    sent: list[dict[str, Any]] = field(default_factory=list)
    started: bool = False
    stopped: bool = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send_and_wait(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> None:
        self.sent.append({"topic": topic, "value": value, "key": key, "headers": headers})


def test_kafka_source_commits_next_offset_and_dead_letters(
    sqlite_url: str, example_catalog: QuestionCatalog, decisions: list[dict[str, Any]]
) -> None:
    a, b, _ = decisions
    consumer = FakeConsumer(
        [
            FakeRecord("dgate.outcomes", 0, 41, event(a).encode(), key=b"k1"),
            FakeRecord("dgate.outcomes", 1, 7, None),  # tombstone
            FakeRecord("dgate.outcomes", 0, 42, event(b, source="ledger").encode()),
        ]
    )
    producer = FakeProducer()
    source = KafkaOutcomeSource(
        consumer,
        topic_partition=lambda t, p: (t, p),
        producer=producer,
        dead_letter_topic="dgate.outcomes.dlq",
    )

    async def run() -> IngestStats:
        async with source:
            assert consumer.started
            assert producer.started
            engine = make_engine(sqlite_url)
            try:
                sessions = make_sessionmaker(engine)
                service = OutcomeService(
                    example_catalog, DecisionRepository(sessions), OutcomeRepository(sessions)
                )
                return await OutcomeIngestor(service, source).run()
            finally:
                await engine.dispose()

    stats = asyncio.run(run())
    assert consumer.stopped
    assert producer.stopped
    assert (stats.created, stats.rejected) == (2, {"malformed": 1})
    assert consumer.commits == [
        {("dgate.outcomes", 0): 42},
        {("dgate.outcomes", 1): 8},
        {("dgate.outcomes", 0): 43},
    ]
    assert len(producer.sent) == 1
    parked = producer.sent[0]
    assert parked["topic"] == "dgate.outcomes.dlq"
    assert parked["value"] == b""
    headers = dict(parked["headers"])
    assert headers["dgate-reason"] == b"malformed"
    assert headers["dgate-origin"] == b"dgate.outcomes[1]@7"
    assert sorted(stored_outcomes(sqlite_url)["sources"]) == ["kafka", "ledger"]


def test_kafka_source_without_dead_letter_topic(
    sqlite_url: str, example_catalog: QuestionCatalog, decisions: list[dict[str, Any]]
) -> None:
    consumer = FakeConsumer([FakeRecord("t", 0, 0, b"garbage")])
    source = KafkaOutcomeSource(consumer, topic_partition=lambda t, p: (t, p), name="crm")
    assert source.name == "crm"
    asyncio.run(source.start())
    stats = ingest(sqlite_url, example_catalog, source)
    asyncio.run(source.stop())
    assert consumer.started
    assert consumer.stopped
    assert stats.rejected == {"malformed": 1}
    assert consumer.commits == [{("t", 0): 1}]


def pair(topic: str, partition: int) -> tuple[str, int]:
    return topic, partition


def test_kafka_source_config_must_be_consistent() -> None:
    with pytest.raises(ValueError, match="dead-letter topic needs a producer"):
        KafkaOutcomeSource(FakeConsumer([]), topic_partition=pair, dead_letter_topic="dlq")
    with pytest.raises(ValueError, match="dead-letter topic needs a producer"):
        KafkaOutcomeSource(FakeConsumer([]), topic_partition=pair, producer=FakeProducer())


def test_kafka_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().kafka_dead_letter_topic == "dgate.outcomes.dlq"
    monkeypatch.setenv("DGATE_KAFKA_DEAD_LETTER_TOPIC", "")
    assert Settings().kafka_dead_letter_topic is None


# --- CLI -------------------------------------------------------------------------------


@pytest.fixture
def events_file(decisions: list[dict[str, Any]], tmp_path: Path) -> Path:
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join([event(decisions[0]), event(decisions[1]), "oops"]) + "\n")
    return path


def test_cli_ingest_file(sqlite_url: str, events_file: Path, tmp_path: Path) -> None:
    dlq = tmp_path / "rejected.jsonl"
    args = ["outcomes", "ingest", str(events_file), "--url", sqlite_url]
    args += ["-q", str(EXAMPLES / "questions.yaml"), "--dead-letter", str(dlq)]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "created 2, duplicate 0, rejected 1" in result.output
    assert "malformed: 1" in result.output
    assert "line 3: malformed" in result.output  # warning on stderr (mixed by the runner)
    assert len(dlq.read_text().splitlines()) == 1

    # Re-running is safe, and --strict turns the rejected line into a failure.
    again = runner.invoke(cli, [*args, "--strict"])
    assert again.exit_code == 1
    assert "created 0, duplicate 2, rejected 1" in again.output


def test_cli_ingest_stdin_with_env_catalog(
    sqlite_url: str, events_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DGATE_QUESTIONS_FILE", str(EXAMPLES / "questions.yaml"))
    monkeypatch.setenv("DGATE_DATABASE_URL", sqlite_url)
    result = runner.invoke(cli, ["outcomes", "ingest", "-"], input=events_file.read_text())
    assert result.exit_code == 0, result.output
    assert "created 2" in result.output


def test_cli_ingest_errors(sqlite_url: str, tmp_path: Path) -> None:
    catalog = ["-q", str(EXAMPLES / "questions.yaml"), "--url", sqlite_url]
    missing = runner.invoke(cli, ["outcomes", "ingest", str(tmp_path / "nope.jsonl"), *catalog])
    assert missing.exit_code == 1
    assert "error:" in missing.output

    no_catalog = runner.invoke(cli, ["outcomes", "ingest", "-", "--url", sqlite_url], input="")
    assert no_catalog.exit_code == 2
    assert "DGATE_QUESTIONS_FILE" in no_catalog.output

    bad = tmp_path / "bad.yaml"
    bad.write_text("questions: [{id: X}]\n")
    bad_catalog = runner.invoke(cli, ["outcomes", "ingest", "-", "-q", str(bad)], input="")
    assert bad_catalog.exit_code == 1
