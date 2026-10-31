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
    Uuid,
    func,
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
