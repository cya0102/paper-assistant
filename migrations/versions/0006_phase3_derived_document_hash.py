"""Bind derived structure state to its canonical parsed document.

Revision ID: 0006_phase3
Revises: 0005_phase3

Existing rows intentionally remain NULL. The runtime treats NULL as stale, so
each existing PaperVersion rebuilds its structure and chunks once on the next
ingestion. This is safer than backfilling an ambiguous parser artifact.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0006_phase3"
down_revision: str | None = "0005_phase3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "derived_data_states",
        sa.Column("document_hash", sa.String(length=64), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_derived_data_states_document_hash_sha256"),
        "derived_data_states",
        "document_hash IS NULL OR document_hash ~ '^[0-9a-f]{64}$'",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_derived_data_states_document_hash_sha256"),
        "derived_data_states",
        type_="check",
    )
    op.drop_column("derived_data_states", "document_hash")
