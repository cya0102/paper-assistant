"""Add the content-addressed Artifact catalog and Citation Manifest.

Revision ID: 0009_context_artifacts
Revises: 0008_research_graph
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0009_context_artifacts"
down_revision: str | None = "0008_research_graph"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_artifacts",
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True)),
        sa.Column("research_task_id", postgresql.UUID(as_uuid=True)),
        sa.Column("work_unit_id", postgresql.UUID(as_uuid=True)),
        sa.Column("tool_call_id", sa.String(length=255)),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=255), nullable=False),
        sa.Column("media_type", sa.String(length=128), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("storage_backend", sa.String(length=64), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_research_artifacts_content_hash_sha256"),
        ),
        sa.CheckConstraint(
            "byte_size >= 0", name=op.f("ck_research_artifacts_byte_size_nonnegative")
        ),
        sa.CheckConstraint(
            "token_estimate >= 0", name=op.f("ck_research_artifacts_token_estimate_nonnegative")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("artifact_id", name="pk_research_artifacts"),
        sa.UniqueConstraint(
            "project_id",
            "artifact_type",
            "schema_version",
            "content_hash",
            name="uq_research_artifacts_project_type_schema_hash",
        ),
    )
    op.create_index(
        "ix_research_artifacts_project_artifact",
        "research_artifacts",
        ["project_id", "artifact_id"],
    )
    op.create_index("ix_research_artifacts_created_by", "research_artifacts", ["created_by"])
    op.create_index("ix_research_artifacts_task", "research_artifacts", ["research_task_id"])
    op.create_index("ix_research_artifacts_work_unit", "research_artifacts", ["work_unit_id"])

    op.create_table(
        "artifact_citations",
        sa.Column("citation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("citation_label", sa.String(length=64), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_title", sa.Text(), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True)),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True)),
        sa.Column("element_id", postgresql.UUID(as_uuid=True)),
        sa.Column("page_start", sa.Integer()),
        sa.Column("page_end", sa.Integer()),
        sa.Column("evidence_hash", sa.String(length=64)),
        sa.CheckConstraint(
            "evidence_hash IS NULL OR evidence_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_artifact_citations_evidence_hash_sha256"),
        ),
        sa.CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) OR "
            "(page_start >= 1 AND page_end >= page_start)",
            name=op.f("ck_artifact_citations_page_range"),
        ),
        sa.ForeignKeyConstraint(
            ["artifact_id"], ["research_artifacts.artifact_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("citation_id", name="pk_artifact_citations"),
        sa.UniqueConstraint(
            "artifact_id",
            "citation_label",
            name="uq_artifact_citations_artifact_label",
        ),
    )
    op.create_index(
        "ix_artifact_citations_project_artifact",
        "artifact_citations",
        ["project_id", "artifact_id"],
    )
    op.create_index("ix_artifact_citations_label", "artifact_citations", ["citation_label"])
    op.create_index(
        "ix_artifact_citations_paper_version",
        "artifact_citations",
        ["project_id", "paper_id", "version_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_artifact_citations_paper_version", table_name="artifact_citations")
    op.drop_index("ix_artifact_citations_label", table_name="artifact_citations")
    op.drop_index("ix_artifact_citations_project_artifact", table_name="artifact_citations")
    op.drop_table("artifact_citations")
    op.drop_index("ix_research_artifacts_work_unit", table_name="research_artifacts")
    op.drop_index("ix_research_artifacts_task", table_name="research_artifacts")
    op.drop_index("ix_research_artifacts_created_by", table_name="research_artifacts")
    op.drop_index("ix_research_artifacts_project_artifact", table_name="research_artifacts")
    op.drop_table("research_artifacts")
