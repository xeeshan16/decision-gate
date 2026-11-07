"""Run a candidate backend in shadow: same inputs, no effect on any response.

After the primary decision is stored, :meth:`ShadowRunner.submit` schedules the candidate
on the same input as a background task and returns at once, so a slow or failing
candidate never adds latency to ``/v1/decide`` or turns a good response into an error.
Backpressure is explicit: at most ``max_pending`` shadow calls are in flight, and
decisions arriving beyond that are not shadowed (counted as ``dropped``) rather than
queued without bound.

Which decisions are shadowed is decided by a hash of the decision id with its own salt.
It is reproducible, and independent of verify sampling (which hashes the same id with a
different salt), so the shadowed decisions are not biased towards any routing tier.

:meth:`ShadowRunner.replay` answers *stored* decisions instead, oldest first, which gives
a candidate labeled history to be compared on straight away instead of waiting for new
traffic and new outcomes.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from decision_gate.backends.base import AnyQuestion, DecisionBackend
from decision_gate.db.models import ShadowDecisionRecord
from decision_gate.db.shadow import ShadowRepository

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 500


def shadow_sample_unit(decision_id: uuid.UUID) -> float:
    """A uniform number in [0, 1) from the decision id, independent of verify sampling."""
    digest = hashlib.sha256(b"dgate-shadow-sample:" + decision_id.bytes).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


@dataclass(slots=True)
class ShadowCounters:
    submitted: int = 0
    skipped: int = 0
    """Not sampled, or the question is not shadowed."""
    dropped: int = 0
    """Sampled, but ``max_pending`` calls were already in flight."""
    completed: int = 0
    failed: int = 0
    """The candidate errored or timed out (the failure is still stored)."""
    store_errors: int = 0
    """The shadow answer could not be stored."""


@dataclass(frozen=True, slots=True)
class ReplayResult:
    question_id: str
    question_version: int
    candidate: str
    attempted: int
    ok: int
    failed: int
    duplicates: int
    skipped_without_input: int


class ShadowRunner:
    def __init__(
        self,
        backend: DecisionBackend,
        repository: ShadowRepository,
        *,
        candidate: str,
        timeout_s: float,
        sample_rate: float = 1.0,
        questions: Collection[str] | None = None,
        max_pending: int = 100,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be within [0, 1]")
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        if not candidate or len(candidate) > 64:
            raise ValueError("candidate must be 1 to 64 characters")
        self._backend = backend
        self._repository = repository
        self.candidate = candidate
        self.timeout_s = timeout_s
        self.sample_rate = sample_rate
        self.questions = frozenset(questions) if questions else None
        self.max_pending = max_pending
        self.counters = ShadowCounters()
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def backend(self) -> DecisionBackend:
        return self._backend

    @property
    def pending(self) -> int:
        return len(self._tasks)

    def shadows(self, question_id: str) -> bool:
        return self.questions is None or question_id in self.questions

    def sampled(self, question_id: str, decision_id: uuid.UUID) -> bool:
        return self.shadows(question_id) and shadow_sample_unit(decision_id) < self.sample_rate

    def submit(
        self, question: AnyQuestion, decision_id: uuid.UUID, data: Mapping[str, Any]
    ) -> bool:
        """Schedule the candidate on this decision's input. Never blocks, never raises."""
        if not self.sampled(question.id, decision_id):
            self.counters.skipped += 1
            return False
        if len(self._tasks) >= self.max_pending:
            self.counters.dropped += 1
            return False
        self.counters.submitted += 1
        task = asyncio.create_task(self._run(question, decision_id, dict(data), "live"))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def drain(self, timeout_s: float | None = None) -> None:
        """Wait for in-flight shadow calls (on shutdown, and in tests)."""
        if not self._tasks:
            return
        tasks = list(self._tasks)
        _done, still_running = await asyncio.wait(tasks, timeout=timeout_s)
        for task in still_running:
            task.cancel()
        if still_running:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*still_running, return_exceptions=True)

    async def answer(
        self, question: AnyQuestion, decision_id: uuid.UUID, data: Mapping[str, Any], mode: str
    ) -> ShadowDecisionRecord:
        """Ask the candidate and build the record (``ok``, ``error`` or ``timeout``)."""
        record = ShadowDecisionRecord(
            id=uuid.uuid4(),
            decision_id=decision_id,
            candidate=self.candidate,
            mode=mode,
            question_id=question.id,
            question_version=question.version,
        )
        started = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                self._backend.decide(question, data), timeout=self.timeout_s
            )
            answer = question.validate_answer(raw.answer)
        except TimeoutError:
            record.status = "timeout"
            record.error = f"no answer within {self.timeout_s:g}s"
        except Exception as exc:  # any candidate failure is data, not a crash
            record.status = "error"
            record.error = _clip(f"{type(exc).__name__}: {exc}")
        else:
            record.status = "ok"
            record.answer = answer
            record.confidence = raw.confidence
            record.probabilities = raw.probabilities
            record.backend = raw.backend
            record.model_version = raw.model_version
        record.latency_ms = (time.perf_counter() - started) * 1000.0
        return record

    async def _run(
        self, question: AnyQuestion, decision_id: uuid.UUID, data: dict[str, Any], mode: str
    ) -> None:
        record = await self.answer(question, decision_id, data, mode)
        if record.status == "ok":
            self.counters.completed += 1
        else:
            self.counters.failed += 1
        try:
            await self._repository.add(record)
        except Exception:
            self.counters.store_errors += 1
            logger.exception("could not store shadow answer for decision %s", decision_id)

    async def replay(
        self,
        question: AnyQuestion,
        *,
        limit: int = 1000,
        labeled_only: bool = True,
        concurrency: int = 8,
    ) -> ReplayResult:
        """Answer stored decisions the candidate has not answered yet, oldest first."""
        if limit < 1 or concurrency < 1:
            raise ValueError("limit and concurrency must be at least 1")
        targets, skipped = await self._repository.replay_targets(
            question.id,
            question.version,
            self.candidate,
            labeled_only=labeled_only,
            limit=limit,
        )
        gate = asyncio.Semaphore(concurrency)
        # Candidate calls run concurrently; writes go one at a time. A replay is a batch
        # job, and concurrent writers only contend (SQLite even rejects them as locked).
        writes = asyncio.Lock()

        async def one(decision_id: uuid.UUID, data: dict[str, Any]) -> tuple[str, bool]:
            async with gate:
                record = await self.answer(question, decision_id, data, "replay")
            async with writes:
                _stored, created = await self._repository.add(record)
            return record.status, created

        results = await asyncio.gather(*(one(t.decision_id, t.input) for t in targets))
        created = [status for status, new in results if new]
        return ReplayResult(
            question_id=question.id,
            question_version=question.version,
            candidate=self.candidate,
            attempted=len(targets),
            ok=sum(1 for s in created if s == "ok"),
            failed=sum(1 for s in created if s != "ok"),
            duplicates=len(results) - len(created),
            skipped_without_input=skipped,
        )

    async def aclose(self) -> None:
        close = getattr(self._backend, "aclose", None)
        if close is not None:
            await close()


def _clip(text: str) -> str:
    return text if len(text) <= MAX_ERROR_LENGTH else text[: MAX_ERROR_LENGTH - 3] + "..."


__all__ = ["ReplayResult", "ShadowCounters", "ShadowRunner", "shadow_sample_unit"]
