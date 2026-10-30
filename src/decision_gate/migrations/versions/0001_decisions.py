"""Create the decisions table.

Revision ID: 0001
Revises:
Create Date: 2026-10-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=True),
        sa.Column("question_id", sa.String(64), nullable=False),
        sa.Column("question_version", sa.Integer(), nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("input", JSONType, nullable=True),
        sa.Column("answer", JSONType, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("probabilities", JSONType, nullable=True),
        sa.Column("backend", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(200), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_decisions"),
        sa.UniqueConstraint("idempotency_key", name="uq_decisions_idempotency_key"),
    )
    op.create_index(
        "ix_decisions_question_id_question_version_created_at",
        "decisions",
        ["question_id", "question_version", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_decisions_question_id_question_version_created_at", "decisions")
    op.drop_table("decisions")
