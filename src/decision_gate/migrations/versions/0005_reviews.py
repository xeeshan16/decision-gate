"""Add the human review queue: one review item per verify or human decision.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "review_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("decision_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(64), nullable=False),
        sa.Column("question_version", sa.Integer(), nullable=False),
        sa.Column("tier", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claims", sa.Integer(), server_default="0", nullable=False),
        sa.Column("claimed_by", sa.String(200), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(200), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution", JSONType, nullable=True),
        sa.Column("verdict", sa.String(16), nullable=True),
        sa.Column("agreed", sa.Boolean(), nullable=True),
        sa.Column("note", sa.String(1000), nullable=True),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_review_items_decision_id_decisions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_review_items"),
        sa.UniqueConstraint("decision_id", name="uq_review_items_decision_id"),
    )
    op.create_index("ix_review_items_status_due_at", "review_items", ["status", "due_at"])
    op.create_index(
        "ix_review_items_question_id_question_version_status",
        "review_items",
        ["question_id", "question_version", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_review_items_question_id_question_version_status", "review_items")
    op.drop_index("ix_review_items_status_due_at", "review_items")
    op.drop_table("review_items")
