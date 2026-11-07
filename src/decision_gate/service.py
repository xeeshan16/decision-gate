"""The decide and outcome flows.

``DecisionService`` asks the backend, validates, persists and answers. ``OutcomeService``
records the ground truth that arrives later and scores it against the stored decision.
Each decision is routed (auto, verify or human) before it is stored, and the tier is
stored with it, so an idempotent replay returns the same tier even if the thresholds
have changed since. Verify and human decisions are enqueued for review in the same
transaction.
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
from decision_gate.reviews import ReviewSla
from decision_gate.routing import Router
from decision_gate.shadow_runner import ShadowRunner

MAX_REASON_LENGTH = 500


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
        router: Router | None = None,
        review_sla: ReviewSla | None = None,
        shadow: ShadowRunner | None = None,
    ) -> None:
        self._backend = backend
        self._repository = repository
        self._timeout_s = timeout_s
        self._store_inputs = store_inputs
        self._router = router
        self._review_sla = review_sla
        self._shadow = shadow

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

        decision_id = uuid.uuid4()
        record = DecisionRecord(
            id=decision_id,
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
            created_at=datetime.now(UTC),
        )
        if self._router is not None:
            route = await self._router.route(
                question, decision_id, raw.answer, raw.confidence, raw.probabilities
            )
            record.tier = route.tier
            record.routing_reason = _clip(route.reason, MAX_REASON_LENGTH)
            record.calibrated_confidence = route.calibrated_confidence
            record.calibrator_id = route.calibrator_id
            record.routing_rule_id = route.rule_id
            record.answer_set = route.answer_set
        review = self._review_sla.new_item(record) if self._review_sla else None
        stored, created = await self._repository.add(record, review)
        if not created:
            self._check_replay(stored, question, digest)
        elif self._shadow is not None:
            # In the background, after the decision is stored (the shadow row references it).
            self._shadow.submit(question, stored.id, data)
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


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


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
