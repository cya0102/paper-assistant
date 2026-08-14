"""Add Phase 1B PDF metadata and semantic identity columns.

Revision ID: 0002_phase1b
Revises: 0001_phase1
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002_phase1b"
down_revision: str | None = "0001_phase1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("normalized_title", sa.Text(), nullable=True))
    op.add_column(
        "papers",
        sa.Column(
            "normalized_authors_json",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column("paper_versions", sa.Column("content_hash", sa.String(length=64)))
    op.create_check_constraint(
        op.f("ck_paper_versions_content_hash_sha256"),
        "paper_versions",
        "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
    )
    op.create_index("ix_paper_versions_content_hash", "paper_versions", ["content_hash"])
    op.add_column(
        "paper_files",
        sa.Column("page_count", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "paper_files",
        sa.Column(
            "metadata_json",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        op.f("ck_paper_files_page_count_nonnegative"),
        "paper_files",
        "page_count >= 0",
    )
    op.alter_column("papers", "normalized_authors_json", server_default=None)
    op.alter_column("paper_files", "page_count", server_default=None)
    op.alter_column("paper_files", "metadata_json", server_default=None)


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_paper_files_page_count_nonnegative"), "paper_files", type_="check"
    )
    op.drop_column("paper_files", "metadata_json")
    op.drop_column("paper_files", "page_count")
    op.drop_index("ix_paper_versions_content_hash", table_name="paper_versions")
    op.drop_constraint(
        op.f("ck_paper_versions_content_hash_sha256"), "paper_versions", type_="check"
    )
    op.drop_column("paper_versions", "content_hash")
    op.drop_column("papers", "normalized_authors_json")
    op.drop_column("papers", "normalized_title")
