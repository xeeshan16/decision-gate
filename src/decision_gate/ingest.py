"""Streaming outcome ingestion.

Outcomes often live in another system (a CRM, a ticketing tool, a payments ledger) that
emits an event when the truth becomes known. :class:`OutcomeIngestor` reads those events
from any :class:`OutcomeSource` and records them through the same
:class:`~decision_gate.service.OutcomeService` the HTTP API uses.

Delivery is at-least-once. A message's position is committed only after it has been
handled, so a crash replays it; recording is idempotent, so a replay is harmless. Each
message ends up in exactly one of three places:

* **recorded**: stored, or already stored with the same value;
* **dead letter**: it can never succeed as sent (malformed, unknown decision, invalid
  value, conflicting label). It is handed to the source's dead-letter sink with a reason
  and then committed, so one bad event cannot block the stream;
* **not committed**: anything else (the database is down, a bug). The error propagates
  and the message is redelivered once the consumer restarts.

Two sources ship: :class:`JsonLinesSource` (a file or stdin, for backfills and demos) and
:class:`~decision_gate.kafka.KafkaOutcomeSource` (optional ``kafka`` extra).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from decision_gate.questions import Answer
from decision_gate.service import (
    InvalidOutcomeError,
    OutcomeConflictError,
    OutcomeService,
    UnknownDecisionError,
)

RejectReason = Literal["malformed", "unknown_decision", "invalid", "conflict"]


class OutcomeEvent(BaseModel):
    """The wire format of one outcome event (JSON)."""

    model_config = ConfigDict(extra="ignore")

    decision_id: uuid.UUID
    outcome: Answer
    source: str | None = Field(default=None, min_length=1, max_length=64)
    observed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SourceMessage:
    """One raw message and where it came from (for commits and error reports)."""

    payload: bytes
    position: str
    handle: Any = None
    """Source-specific data needed to commit this message, e.g. a Kafka record."""


@runtime_checkable
class OutcomeSource(Protocol):
    """Where outcome events come from."""

    @property
    def name(self) -> str:
        """Default ``source`` label for events that do not name one."""
        ...

    def messages(self) -> AsyncIterator[SourceMessage]: ...

    async def commit(self, message: SourceMessage) -> None:
        """Mark ``message`` and everything before it as handled."""
        ...

    async def dead_letter(self, message: SourceMessage, reason: RejectReason, detail: str) -> None:
        """Park a message that can never be recorded as sent."""
        ...


@dataclass
class IngestStats:
    created: int = 0
    duplicate: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def handled(self) -> int:
        return self.created + self.duplicate + sum(self.rejected.values())

    def reject(self, reason: RejectReason) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


class OutcomeIngestor:
    def __init__(self, service: OutcomeService, source: OutcomeSource) -> None:
        self._service = service
        self._source = source

    async def run(self, *, max_messages: int | None = None) -> IngestStats:
        """Consume until the source is exhausted or ``max_messages`` have been handled."""
        stats = IngestStats()
        if max_messages is not None and max_messages <= 0:
            return stats
        async for message in self._source.messages():
            await self.handle(message, stats)
            if max_messages is not None and stats.handled >= max_messages:
                break
        return stats

    async def handle(self, message: SourceMessage, stats: IngestStats) -> None:
        rejection = await self._record(message, stats)
        if rejection is not None:
            reason, detail = rejection
            stats.reject(reason)
            await self._source.dead_letter(message, reason, detail)
        await self._source.commit(message)

    async def _record(
        self, message: SourceMessage, stats: IngestStats
    ) -> tuple[RejectReason, str] | None:
        try:
            event = OutcomeEvent.model_validate_json(message.payload)
        except ValidationError as exc:
            return "malformed", _first_error(exc)
        try:
            _, created = await self._service.record(
                event.decision_id,
                event.outcome,
                source=event.source or self._source.name,
                observed_at=event.observed_at,
            )
        except UnknownDecisionError as exc:
            return "unknown_decision", str(exc)
        except InvalidOutcomeError as exc:
            return "invalid", str(exc)
        except OutcomeConflictError as exc:
            return "conflict", str(exc)
        if created:
            stats.created += 1
        else:
            stats.duplicate += 1
        return None


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    where = ".".join(str(part) for part in error["loc"]) or "message"
    return f"{where}: {error['msg']}"


class JsonLinesSource:
    """Outcome events as JSON Lines from a file or a text stream.

    Rejected lines go to ``dead_letter_path`` (as JSON Lines with the reason attached) when
    given, otherwise to the ``on_reject`` callback. Blank lines are skipped.
    """

    def __init__(
        self,
        stream: IO[str],
        *,
        name: str = "jsonl",
        dead_letter_path: Path | None = None,
        on_reject: Callable[[SourceMessage, RejectReason, str], None] | None = None,
    ) -> None:
        self._stream = stream
        self._name = name
        self._dead_letter_path = dead_letter_path
        self._on_reject = on_reject
        self.committed: str | None = None

    @property
    def name(self) -> str:
        return self._name

    async def messages(self) -> AsyncIterator[SourceMessage]:
        for number, line in enumerate(self._stream, start=1):
            if line.strip():
                yield SourceMessage(payload=line.encode(), position=f"line {number}")

    async def commit(self, message: SourceMessage) -> None:
        self.committed = message.position

    async def dead_letter(self, message: SourceMessage, reason: RejectReason, detail: str) -> None:
        if self._on_reject is not None:
            self._on_reject(message, reason, detail)
        if self._dead_letter_path is None:
            return
        entry = {
            "position": message.position,
            "reason": reason,
            "detail": detail,
            "payload": message.payload.decode(errors="replace").rstrip("\n"),
        }
        with self._dead_letter_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(entry) + "\n")
