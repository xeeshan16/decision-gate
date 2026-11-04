"""Add routing: the routing_rules table and each decision's tier.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-04
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def decision_columns() -> list[sa.Column[Any]]:
    return [
        sa.Column("tier", sa.String(16), nullable=True),
        sa.Column("calibrated_confidence", sa.Float(), nullable=True),
        sa.Column("routing_reason", sa.String(500), nullable=True),
        sa.Column("answer_set", JSONType, nullable=True),
        sa.Column("calibrator_id", sa.Uuid(), nullable=True),
        sa.Column("routing_rule_id", sa.Uuid(), nullable=True),
    ]


def upgrade() -> None:
    with op.batch_alter_table("decisions") as batch:
        for column in decision_columns():
            batch.add_column(column)
    op.create_table(
        "routing_rules",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(64), nullable=False),
        sa.Column("question_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("policy_fingerprint", sa.String(64), nullable=False),
        sa.Column("policy", JSONType, nullable=False),
        sa.Column("calibrator_id", sa.Uuid(), nullable=True),
        sa.Column("auto_threshold", sa.Float(), nullable=True),
        sa.Column("verify_threshold", sa.Float(), nullable=True),
        sa.Column("conformal", JSONType, nullable=True),
        sa.Column("labeled", sa.Integer(), nullable=False),
        sa.Column("eligible", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(1000), nullable=False),
        sa.Column("evaluation", JSONType, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["calibrator_id"],
            ["calibrators.id"],
            name="fk_routing_rules_calibrator_id_calibrators",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_routing_rules"),
    )
    op.create_index(
        "ix_routing_rules_question_id_question_version_created_at",
        "routing_rules",
        ["question_id", "question_version", "created_at"],
    )
    # At most one active rule per question version.
    op.create_index(
        "uq_routing_rules_one_active",
        "routing_rules",
        ["question_id", "question_version"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        sqlite_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index("uq_routing_rules_one_active", "routing_rules")
    op.drop_index("ix_routing_rules_question_id_question_version_created_at", "routing_rules")
    op.drop_table("routing_rules")
    with op.batch_alter_table("decisions") as batch:
        for column in reversed(decision_columns()):
            batch.drop_column(column.name)
