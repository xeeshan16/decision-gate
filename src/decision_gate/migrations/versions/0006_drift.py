"""Add drift checks, and link routing rules to the drift check that tightened them.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-06
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "drift_checks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(64), nullable=False),
        sa.Column("question_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reference_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reference_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("policy", JSONType, nullable=False),
        sa.Column("signals", JSONType, nullable=False),
        sa.Column("summary", JSONType, nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("action_reason", sa.String(1000), nullable=False),
        sa.Column("previous_rule_id", sa.Uuid(), nullable=True),
        sa.Column("rule_id", sa.Uuid(), nullable=True),
        sa.Column("alert", sa.String(16), nullable=True),
        sa.Column("alert_event", sa.String(32), nullable=True),
        sa.Column("alert_error", sa.String(500), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_drift_checks"),
    )
    op.create_index(
        "ix_drift_checks_question_id_question_version_checked_at",
        "drift_checks",
        ["question_id", "question_version", "checked_at"],
    )
    with op.batch_alter_table("routing_rules") as batch:
        batch.add_column(sa.Column("drift_check_id", sa.Uuid(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("routing_rules") as batch:
        batch.drop_column("drift_check_id")
    op.drop_index("ix_drift_checks_question_id_question_version_checked_at", "drift_checks")
    op.drop_table("drift_checks")
