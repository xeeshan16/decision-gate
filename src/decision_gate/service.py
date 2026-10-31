"""The decide and outcome flows.

``DecisionService`` asks the backend, validates, persists and answers. ``OutcomeService``
records the ground truth that arrives later and scores it against the stored decision.
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
from datetime import UTC, datetime
from typing import Any

from decision_gate.backends.base import AnyQuestion, BackendError, DecisionBackend
from decision_gate.db.models import DecisionRecord, OutcomeRecord
from decision_gate.db.repository import DecisionRepository, OutcomeRepository
from decision_gate.questions import Answer, QuestionCatalog


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


class UnknownDecisionError(LookupError):
    """The outcome refers to a decision that was never recorded."""


class InvalidOutcomeError(ValueError):
    """The outcome is not a valid answer to the decision's question."""


class OutcomeConflictError(ValueError):
    """A different outcome was already recorded for this decision."""


class OutcomeService:
    """Record ground truth for decisions, exactly once each.

    Recording is idempotent so that every delivery path can retry freely: an HTTP client
    after a timeout, a Kafka consumer after a rebalance. Sending the same outcome again
    returns the stored row; sending a *different* one is a conflict, because silently
    overwriting a label would rewrite the history that calibration was fitted on.
    """

    def __init__(
        self,
        catalog: QuestionCatalog,
        decisions: DecisionRepository,
        outcomes: OutcomeRepository,
    ) -> None:
        self._catalog = catalog
        self._decisions = decisions
        self._outcomes = outcomes

    async def record(
        self,
        decision_id: uuid.UUID,
        outcome: Answer,
        *,
        source: str,
        observed_at: datetime | None = None,
    ) -> tuple[OutcomeRecord, bool]:
        """Return ``(outcome, created)``; ``created`` is false for an idempotent replay."""
        decision = await self._decisions.get(decision_id)
        if decision is None:
            raise UnknownDecisionError(f"unknown decision {decision_id}")
        try:
            question = self._catalog.get(decision.question_id, decision.question_version)
        except KeyError as exc:
            raise InvalidOutcomeError(
                f"question {decision.question_key} is no longer in the catalog"
            ) from exc
        try:
            value = question.validate_answer(outcome)
        except ValueError as exc:
            raise InvalidOutcomeError(str(exc)) from exc

        existing = await self._outcomes.get(decision_id)
        if existing is None:
            record = OutcomeRecord(
                decision_id=decision_id,
                outcome=value,
                correct=question.is_correct(decision.answer, value),
                source=source,
                observed_at=_utc(observed_at) if observed_at else datetime.now(UTC),
            )
            existing, created = await self._outcomes.add(record)
            if created:
                return existing, True
        if question.validate_answer(existing.outcome) != value:
            raise OutcomeConflictError(
                f"decision {decision_id} already has outcome {existing.outcome!r}"
            )
        return existing, False


def _utc(moment: datetime) -> datetime:
    """Treat naive timestamps as UTC, and normalise aware ones to UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)
