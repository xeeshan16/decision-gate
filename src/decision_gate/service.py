"""The decide flow: ask the backend, validate, persist, answer.

Calibration and routing slot in here in later milestones; for now every decision is
recorded with the backend's raw confidence so that outcomes can be joined to it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

from decision_gate.backends.base import AnyQuestion, BackendError, DecisionBackend
from decision_gate.db.models import DecisionRecord
from decision_gate.db.repository import DecisionRepository


class BackendTimeoutError(BackendError):
    """The backend did not answer within the configured time limit."""


class IdempotencyConflictError(ValueError):
    """An idempotency key was reused for a different question or input."""


def input_sha256(data: Mapping[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class DecisionService:
    def __init__(
        self,
        backend: DecisionBackend,
        repository: DecisionRepository,
        *,
        timeout_s: float,
        store_inputs: bool = True,
    ) -> None:
        self._backend = backend
        self._repository = repository
        self._timeout_s = timeout_s
        self._store_inputs = store_inputs

    async def decide(
        self,
        question: AnyQuestion,
        data: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> tuple[DecisionRecord, bool]:
        """Return ``(decision, created)``; ``created`` is false for an idempotent replay."""
        digest = input_sha256(data)
        if idempotency_key is not None:
            existing = await self._repository.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                return self._check_replay(existing, question, digest), False

        started = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                self._backend.decide(question, data), timeout=self._timeout_s
            )
        except TimeoutError as exc:
            raise BackendTimeoutError(
                f"{self._backend.name}: no answer within {self._timeout_s:g}s"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        record = DecisionRecord(
            id=uuid.uuid4(),
            idempotency_key=idempotency_key,
            question_id=question.id,
            question_version=question.version,
            input_sha256=digest,
            input=dict(data) if self._store_inputs else None,
            answer=raw.answer,
            confidence=raw.confidence,
            probabilities=raw.probabilities,
            backend=raw.backend,
            model_version=raw.model_version,
            latency_ms=latency_ms,
        )
        stored, created = await self._repository.add(record)
        if not created:
            self._check_replay(stored, question, digest)
        return stored, created

    @staticmethod
    def _check_replay(record: DecisionRecord, question: AnyQuestion, digest: str) -> DecisionRecord:
        same = (
            record.question_id == question.id
            and record.question_version == question.version
            and record.input_sha256 == digest
        )
        if not same:
            raise IdempotencyConflictError(
                "idempotency key was already used for a different question or input"
            )
        return record
