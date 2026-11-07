"""Database schema. Changes go through Alembic migrations in ``decision_gate.migrations``."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on PostgreSQL (indexable, compact); plain JSON elsewhere so tests can use SQLite.
JSONType = JSON().with_variant(JSONB(), "postgresql")

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class DecisionRecord(Base):
    """One raw decision proposed by a backend, as it was at decision time."""

    __tablename__ = "decisions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    idempotency_key: Mapped[str | None] = mapped_column(String(200), unique=True)
    question_id: Mapped[str] = mapped_column(String(64))
    question_version: Mapped[int] = mapped_column(Integer)
    input_sha256: Mapped[str] = mapped_column(String(64))
    input: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    answer: Mapped[Any] = mapped_column(JSONType)
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSONType)
    backend: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(200))
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), server_default=func.now()
    )
    # Routing, decided once and kept with the decision so a replay returns the same tier.
    # Null on decisions recorded before routing existed.
    tier: Mapped[str | None] = mapped_column(String(16))
    calibrated_confidence: Mapped[float | None] = mapped_column(Float)
    routing_reason: Mapped[str | None] = mapped_column(String(500))
    answer_set: Mapped[list[Any] | None] = mapped_column(JSONType)
    calibrator_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    routing_rule_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)

    __table_args__ = (
        # Calibration and drift jobs scan one question version over a time window.
        Index(None, "question_id", "question_version", "created_at"),
    )

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"


class OutcomeRecord(Base):
    """The ground truth for one decision, which usually arrives long after the decision.

    One outcome per decision: the decision id is the primary key, so a replayed outcome
    (an HTTP retry, a redelivered Kafka message) can never be counted twice. ``correct``
    is computed once, when the outcome is recorded, by the question's own rule.
    """

    __tablename__ = "outcomes"

    decision_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("decisions.id", ondelete="CASCADE"), primary_key=True
    )
    outcome: Mapped[Any] = mapped_column(JSONType)
    correct: Mapped[bool] = mapped_column(Boolean)
    source: Mapped[str] = mapped_column(String(64))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class CalibratorRecord(Base):
    """One fitted calibrator for a question version, kept forever for audit.

    ``version`` counts fits per question version (1, 2, ...). At most one calibrator per
    question version is ``active``; a partial unique index enforces that, so two
    concurrent activations cannot both win. The others are ``candidate`` (beat the raw
    scores but were not activated), ``rejected`` (did not beat the raw scores or the
    active calibrator on held-out decisions) or ``retired`` (replaced).
    """

    __tablename__ = "calibrators"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    question_id: Mapped[str] = mapped_column(String(64))
    question_version: Mapped[int] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer)
    method: Mapped[str] = mapped_column(String(32))
    params: Mapped[dict[str, Any]] = mapped_column(JSONType)
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(String(500))
    train_count: Mapped[int] = mapped_column(Integer)
    holdout_count: Mapped[int] = mapped_column(Integer)
    trained_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    """Decision time of the oldest training decision."""
    trained_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    """Decision time of the newest training decision; held-out decisions are all later."""
    evaluation: Mapped[dict[str, Any]] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), server_default=func.now()
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("question_id", "question_version", "version"),
        Index(
            "uq_calibrators_one_active",
            "question_id",
            "question_version",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
    )

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"


class RoutingRuleRecord(Base):
    """The thresholds one question version is routed with, as selected at one point in time.

    A rule is tied to the policy it was selected under (``policy_fingerprint``) and to the
    calibrator whose scores it was selected on (``calibrator_id``). If either changes, the
    rule no longer describes the scores it would be applied to, and the router sends every
    decision to a human until the rule is refreshed. Rules are kept for audit; at most one
    per question version is ``active``.
    """

    __tablename__ = "routing_rules"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    question_id: Mapped[str] = mapped_column(String(64))
    question_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    policy_fingerprint: Mapped[str] = mapped_column(String(64))
    policy: Mapped[dict[str, Any]] = mapped_column(JSONType)
    calibrator_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("calibrators.id", ondelete="SET NULL")
    )
    auto_threshold: Mapped[float | None] = mapped_column(Float)
    verify_threshold: Mapped[float | None] = mapped_column(Float)
    conformal: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    """Split-conformal predictor parameters, when the policy asks for answer sets."""
    labeled: Mapped[int] = mapped_column(Integer)
    eligible: Mapped[int] = mapped_column(Integer)
    """Labeled decisions the thresholds were selected on (excludes calibrator training)."""
    reason: Mapped[str] = mapped_column(String(1000))
    evaluation: Mapped[dict[str, Any]] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), server_default=func.now()
    )
    drift_check_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    """Set when a drift check tightened this rule (or a refresh kept that tightening)."""

    __table_args__ = (
        Index(None, "question_id", "question_version", "created_at"),
        Index(
            "uq_routing_rules_one_active",
            "question_id",
            "question_version",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
    )

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"


class ReviewItemRecord(Base):
    """One decision waiting for, held by, or handled by a person.

    Every ``verify`` and ``human`` decision gets exactly one review item, created in the
    same transaction as the decision (the decision id is unique here). ``status`` moves
    ``open`` -> ``claimed`` -> ``resolved``; a claim is a lease that expires, after which
    the item can be claimed again. If the decision's ground truth arrives some other way
    first (a later business event, a Kafka label), the item becomes ``superseded`` and
    leaves the queue. Resolving an item records its outcome in the same transaction, which
    is how reviews flow back into calibration.
    """

    __tablename__ = "review_items"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    decision_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("decisions.id", ondelete="CASCADE"), unique=True
    )
    question_id: Mapped[str] = mapped_column(String(64))
    question_version: Mapped[int] = mapped_column(Integer)
    tier: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    """The SLA deadline: resolve by then."""
    claims: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    """How many times the item was claimed; more than one means a lease ran out or was
    released."""
    claimed_by: Mapped[str | None] = mapped_column(String(200))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    claim_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_by: Mapped[str | None] = mapped_column(String(200))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution: Mapped[Any] = mapped_column(JSONType, nullable=True)
    """The answer the reviewer gave, which is also stored as the decision's outcome."""
    verdict: Mapped[str | None] = mapped_column(String(16))
    """``confirmed`` or ``corrected`` for verify items, ``decided`` for human items."""
    agreed: Mapped[bool | None] = mapped_column(Boolean)
    """Whether the backend's answer was correct by the reviewer's outcome."""
    note: Mapped[str | None] = mapped_column(String(1000))

    __table_args__ = (
        # Claiming the next item scans unresolved items by deadline.
        Index(None, "status", "due_at"),
        Index(None, "question_id", "question_version", "status"),
    )

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"


class DriftCheckRecord(Base):
    """One drift check of one question version: what was measured, and what it led to.

    ``signals`` holds each check's verdict, value, limit and detail. ``action`` says what
    the check did to routing: ``none``, ``tightened`` (a stricter rule replaced
    ``previous_rule_id`` with ``rule_id``), ``suspended`` (the new rule has no auto
    threshold), ``already_tightened``, ``not_tightened`` (the window's labels support the
    current threshold) or ``skipped`` (with the reason in ``action_reason``). The alert
    columns record whether the webhook was needed and delivered.
    """

    __tablename__ = "drift_checks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    question_id: Mapped[str] = mapped_column(String(64))
    question_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reference_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reference_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    policy: Mapped[dict[str, Any]] = mapped_column(JSONType)
    signals: Mapped[list[dict[str, Any]]] = mapped_column(JSONType)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONType)
    """Window sizes and label rates, for context."""
    action: Mapped[str] = mapped_column(String(32))
    action_reason: Mapped[str] = mapped_column(String(1000))
    previous_rule_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    rule_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    alert: Mapped[str | None] = mapped_column(String(16))
    """``None`` (no alert needed), ``sent``, ``failed`` or ``unconfigured``."""
    alert_event: Mapped[str | None] = mapped_column(String(32))
    alert_error: Mapped[str | None] = mapped_column(String(500))

    __table_args__ = (Index(None, "question_id", "question_version", "checked_at"),)

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"


class ShadowDecisionRecord(Base):
    """What a candidate backend answered for a decision it did not make.

    The primary backend's decision is what the caller got and what was routed; the
    candidate answered the same input on the side (``mode='live'``) or later, replaying
    stored inputs (``mode='replay'``). A candidate never affects a response. Its answer is
    scored against the primary decision's outcome when that arrives, so the two models are
    compared on exactly the same inputs and the same ground truth. ``status`` is ``ok``,
    ``error`` or ``timeout``; failures are kept because a candidate that fails more often
    is not ready to be promoted however good its answers are.
    """

    __tablename__ = "shadow_decisions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    decision_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("decisions.id", ondelete="CASCADE")
    )
    candidate: Mapped[str] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(16))
    question_id: Mapped[str] = mapped_column(String(64))
    question_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    answer: Mapped[Any] = mapped_column(JSONType, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSONType)
    backend: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(200))
    latency_ms: Mapped[float] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), server_default=func.now()
    )

    __table_args__ = (
        # One answer per candidate per decision, so retries and replays never double count.
        UniqueConstraint("decision_id", "candidate"),
        Index(None, "question_id", "question_version", "candidate"),
    )

    @property
    def question_key(self) -> str:
        return f"{self.question_id}@v{self.question_version}"
