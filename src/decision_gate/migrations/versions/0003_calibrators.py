"""Create the calibrators table (the versioned calibrator registry).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "calibrators",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(64), nullable=False),
        sa.Column("question_version", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("method", sa.String(32), nullable=False),
        sa.Column("params", JSONType, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("reason", sa.String(500), nullable=False),
        sa.Column("train_count", sa.Integer(), nullable=False),
        sa.Column("holdout_count", sa.Integer(), nullable=False),
        sa.Column("trained_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trained_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("evaluation", JSONType, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_calibrators"),
        sa.UniqueConstraint(
            "question_id",
            "question_version",
            "version",
            name="uq_calibrators_question_id_question_version_version",
        ),
    )
    # At most one active calibrator per question version.
    op.create_index(
        "uq_calibrators_one_active",
        "calibrators",
        ["question_id", "question_version"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        sqlite_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index("uq_calibrators_one_active", "calibrators")
    op.drop_table("calibrators")
