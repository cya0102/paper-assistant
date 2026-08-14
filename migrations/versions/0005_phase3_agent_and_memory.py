"""Add durable interaction memory, notes, and user preferences.

Revision ID: 0005_phase3
Revises: 0004_phase2a
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0005_phase3"
down_revision: str | None = "0004_phase2a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[object], sa.Column[object]]:
    return (
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def upgrade() -> None:
    op.create_table(
        "interactions",
        sa.Column("interaction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("paper_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("topics_json", postgresql.JSONB(), nullable=False),
        sa.Column("interaction_type", sa.String(length=64), nullable=False),
        sa.Column("retrieved_chunk_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("answer_summary", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("interaction_id", name="pk_interactions"),
    )
    op.create_index("ix_interactions_user_created", "interactions", ["user_id", "created_at"])
    op.create_index("ix_interactions_session", "interactions", ["session_id"])
    op.create_table(
        "notes",
        sa.Column("note_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True)),
        sa.Column("section_id", postgresql.UUID(as_uuid=True)),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tags_json", postgresql.JSONB(), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.paper_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["section_id"], ["sections.section_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("note_id", name="pk_notes"),
    )
    op.create_index("ix_notes_user_project", "notes", ["user_id", "project_id"])
    op.create_table(
        "user_preferences",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("preference_key", sa.String(length=128), nullable=False),
        sa.Column("preference_value", postgresql.JSONB(), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("user_id", "preference_key", name="pk_user_preferences"),
    )


def downgrade() -> None:
    op.drop_table("user_preferences")
    op.drop_index("ix_notes_user_project", table_name="notes")
    op.drop_table("notes")
    op.drop_index("ix_interactions_session", table_name="interactions")
    op.drop_index("ix_interactions_user_created", table_name="interactions")
    op.drop_table("interactions")
