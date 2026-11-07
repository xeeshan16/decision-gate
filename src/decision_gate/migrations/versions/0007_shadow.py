"""Add shadow decisions: candidate backends' answers to decisions they did not make.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "shadow_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("decision_id", sa.Uuid(), nullable=False),
        sa.Column("candidate", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),
        sa.Column("question_id", sa.String(64), nullable=False),
        sa.Column("question_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("answer", JSONType, nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("probabilities", JSONType, nullable=True),
        sa.Column("backend", sa.String(64), nullable=True),
        sa.Column("model_version", sa.String(200), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("error", sa.String(500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_shadow_decisions_decision_id_decisions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_shadow_decisions"),
        sa.UniqueConstraint(
            "decision_id", "candidate", name="uq_shadow_decisions_decision_id_candidate"
        ),
    )
    op.create_index(
        "ix_shadow_decisions_question_id_question_version_candidate",
        "shadow_decisions",
        ["question_id", "question_version", "candidate"],
    )


def downgrade() -> None:
    op.drop_index("ix_shadow_decisions_question_id_question_version_candidate", "shadow_decisions")
    op.drop_table("shadow_decisions")
