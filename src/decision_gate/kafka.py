"""Kafka outcome source (optional: ``pip install 'decision-gate[kafka]'``).

Consumes outcome events from a topic with automatic offset commits turned off, and commits
each record's offset only after :class:`~decision_gate.ingest.OutcomeIngestor` has
handled it. Rejected events are published to a dead-letter topic, with the reason in the
record headers, before their offset is committed.

The consumer and producer are passed in, so tests drive this class with in-memory fakes;
:func:`from_settings` builds the real aiokafka clients.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from decision_gate.config import Settings
from decision_gate.ingest import RejectReason, SourceMessage


class KafkaRecord(Protocol):
    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes | None


class Consumer(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def __aiter__(self) -> AsyncIterator[Any]: ...
    async def commit(self, offsets: dict[Any, int]) -> None: ...


class Producer(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send_and_wait(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> Any: ...


class KafkaOutcomeSource:
    def __init__(
        self,
        consumer: Consumer,
        *,
        topic_partition: Callable[[str, int], Any],
        producer: Producer | None = None,
        dead_letter_topic: str | None = None,
        name: str = "kafka",
    ) -> None:
        if (producer is None) != (dead_letter_topic is None):
            raise ValueError("a dead-letter topic needs a producer, and the reverse")
        self._consumer = consumer
        self._producer = producer
        self._topic_partition = topic_partition
        self._dead_letter_topic = dead_letter_topic
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        await self._consumer.start()
        if self._producer is not None:
            await self._producer.start()

    async def stop(self) -> None:
        try:
            await self._consumer.stop()
        finally:
            if self._producer is not None:
                await self._producer.stop()

    async def __aenter__(self) -> KafkaOutcomeSource:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    async def messages(self) -> AsyncIterator[SourceMessage]:
        async for record in self._consumer:
            yield SourceMessage(
                payload=record.value or b"",
                position=f"{record.topic}[{record.partition}]@{record.offset}",
                handle=record,
            )

    async def commit(self, message: SourceMessage) -> None:
        record: KafkaRecord = message.handle
        # Kafka commits the offset of the *next* record to read.
        tp = self._topic_partition(record.topic, record.partition)
        await self._consumer.commit({tp: record.offset + 1})

    async def dead_letter(self, message: SourceMessage, reason: RejectReason, detail: str) -> None:
        if self._producer is None or self._dead_letter_topic is None:
            return
        record: KafkaRecord = message.handle
        await self._producer.send_and_wait(
            self._dead_letter_topic,
            value=message.payload,
            key=record.key,
            headers=[
                ("dgate-reason", reason.encode()),
                ("dgate-detail", detail.encode()[:1000]),
                ("dgate-origin", message.position.encode()),
            ],
        )


def from_settings(settings: Settings) -> KafkaOutcomeSource:  # pragma: no cover - needs a broker
    """Build a source with real aiokafka clients from ``DGATE_KAFKA_*`` settings."""
    try:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        from aiokafka.structs import TopicPartition
    except ImportError as exc:
        raise RuntimeError(
            "Kafka support needs the 'kafka' extra: pip install 'decision-gate[kafka]'"
        ) from exc

    consumer = AIOKafkaConsumer(
        settings.kafka_outcomes_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    producer = None
    if settings.kafka_dead_letter_topic:
        producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers)
    return KafkaOutcomeSource(
        consumer,
        topic_partition=TopicPartition,
        producer=producer,
        dead_letter_topic=settings.kafka_dead_letter_topic,
    )
