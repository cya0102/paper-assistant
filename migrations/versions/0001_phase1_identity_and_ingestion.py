"""Create Phase 1 identity and ingestion schema.

Revision ID: 0001_phase1
Revises: None
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_phase1"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("root_path", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("project_id", name="pk_projects"),
        sa.UniqueConstraint("root_path", name="uq_projects_root_path"),
    )
    op.create_table(
        "papers",
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("canonical_title", sa.Text()),
        sa.Column("short_name", sa.String(length=255)),
        sa.Column("acronym", sa.String(length=128)),
        sa.Column("aliases_json", postgresql.JSONB(), nullable=False),
        sa.Column("authors_json", postgresql.JSONB(), nullable=False),
        sa.Column("doi", sa.String(length=512)),
        sa.Column("arxiv_id", sa.String(length=128)),
        sa.Column("year", sa.Integer()),
        sa.Column("venue", sa.String(length=512)),
        sa.Column("abstract", sa.Text()),
        sa.Column("canonical_version_id", postgresql.UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("paper_id", name="pk_papers"),
    )
    op.create_index("ix_papers_doi", "papers", ["doi"])
    op.create_index("ix_papers_arxiv_id", "papers", ["arxiv_id"])
    op.create_table(
        "paper_versions",
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_label", sa.String(length=255)),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("source_identifier", sa.String(length=512)),
        sa.Column("parser_version", sa.String(length=128)),
        sa.Column("pipeline_status", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.paper_id"], ondelete="CASCADE", name="fk_versions_paper"),
        sa.PrimaryKeyConstraint("version_id", name="pk_paper_versions"),
        sa.UniqueConstraint("version_id", "paper_id", name="uq_paper_versions_version_paper"),
    )
    op.create_index("ix_paper_versions_paper_id", "paper_versions", ["paper_id"])
    op.create_foreign_key(
        "fk_papers_canonical_version",
        "papers",
        "paper_versions",
        ["canonical_version_id", "paper_id"],
        ["version_id", "paper_id"],
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_table(
        "paper_files",
        sa.Column("file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True)),
        sa.Column("version_id", postgresql.UUID(as_uuid=True)),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("file_hash", sa.String(length=64), nullable=False),
        sa.Column("content_hash", sa.String(length=64)),
        sa.Column("is_canonical", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("file_size >= 0", name=op.f("ck_paper_files_file_size_nonnegative")),
        sa.CheckConstraint("file_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_paper_files_file_hash_sha256")),
        sa.CheckConstraint(
            "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_paper_files_content_hash_sha256"),
        ),
        sa.CheckConstraint(
            "(paper_id IS NULL AND version_id IS NULL) OR (paper_id IS NOT NULL AND version_id IS NOT NULL)",
            name=op.f("ck_paper_files_paper_version_pair"),
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE", name="fk_files_project"),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            ondelete="SET NULL",
            name="fk_paper_files_version_paper",
        ),
        sa.PrimaryKeyConstraint("file_id", name="pk_paper_files"),
        sa.UniqueConstraint("project_id", "file_hash", name="uq_paper_files_project_hash"),
        sa.UniqueConstraint("file_id", "project_id", name="uq_paper_files_file_project"),
    )
    op.create_index("ix_paper_files_content_hash", "paper_files", ["content_hash"])
    op.create_index("ix_paper_files_paper_id", "paper_files", ["paper_id"])
    op.create_index("ix_paper_files_version_id", "paper_files", ["version_id"])
    op.create_table(
        "paper_file_locations",
        sa.Column("location_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("mtime_ns", sa.BigInteger(), nullable=False),
        sa.Column("presence_status", sa.String(length=32), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("mtime_ns >= 0", name=op.f("ck_paper_file_locations_mtime_nonnegative")),
        sa.ForeignKeyConstraint(
            ["file_id", "project_id"],
            ["paper_files.file_id", "paper_files.project_id"],
            ondelete="CASCADE",
            name="fk_file_locations_file_project",
        ),
        sa.PrimaryKeyConstraint("location_id", name="pk_paper_file_locations"),
        sa.UniqueConstraint("project_id", "relative_path", name="uq_file_locations_project_path"),
    )
    op.create_index("ix_file_locations_file_id", "paper_file_locations", ["file_id"])
    op.create_table(
        "parsed_documents",
        sa.Column("parsed_document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("parser_name", sa.String(length=255), nullable=False),
        sa.Column("parser_version", sa.String(length=128), nullable=False),
        sa.Column("document_json_path", sa.Text(), nullable=False),
        sa.Column("document_markdown_path", sa.Text(), nullable=False),
        sa.Column("assets_path", sa.Text(), nullable=False),
        sa.Column("document_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("document_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_parsed_documents_document_hash_sha256")),
        sa.ForeignKeyConstraint(["source_file_id"], ["paper_files.file_id"], ondelete="RESTRICT", name="fk_documents_file"),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            ondelete="CASCADE",
            name="fk_parsed_documents_version_paper",
        ),
        sa.PrimaryKeyConstraint("parsed_document_id", name="pk_parsed_documents"),
        sa.UniqueConstraint(
            "version_id", "parser_name", "parser_version", "schema_version",
            name="uq_parsed_documents_version_parser_schema",
        ),
    )
    op.create_index("ix_parsed_documents_paper_id", "parsed_documents", ["paper_id"])
    op.create_table(
        "ingestion_runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requested_paths_json", postgresql.JSONB(), nullable=False),
        sa.Column("recursive", sa.Boolean(), nullable=False),
        sa.Column("force_reindex", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("counters_json", postgresql.JSONB(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE", name="fk_runs_project"),
        sa.PrimaryKeyConstraint("run_id", name="pk_ingestion_runs"),
    )
    op.create_table(
        "ingestion_items",
        sa.Column("item_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("file_id", postgresql.UUID(as_uuid=True)),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True)),
        sa.Column("version_id", postgresql.UUID(as_uuid=True)),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("disposition", sa.String(length=64), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(length=64)),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["file_id"], ["paper_files.file_id"], ondelete="SET NULL", name="fk_items_file"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.paper_id"], ondelete="SET NULL", name="fk_items_paper"),
        sa.ForeignKeyConstraint(["run_id"], ["ingestion_runs.run_id"], ondelete="CASCADE", name="fk_items_run"),
        sa.ForeignKeyConstraint(["version_id"], ["paper_versions.version_id"], ondelete="SET NULL", name="fk_items_version"),
        sa.PrimaryKeyConstraint("item_id", name="pk_ingestion_items"),
        sa.UniqueConstraint("run_id", "relative_path", name="uq_ingestion_items_run_path"),
    )
    op.create_index("ix_ingestion_items_file_id", "ingestion_items", ["file_id"])
    op.create_index("ix_ingestion_items_stage", "ingestion_items", ["stage"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_items_stage", table_name="ingestion_items")
    op.drop_index("ix_ingestion_items_file_id", table_name="ingestion_items")
    op.drop_table("ingestion_items")
    op.drop_table("ingestion_runs")
    op.drop_index("ix_parsed_documents_paper_id", table_name="parsed_documents")
    op.drop_table("parsed_documents")
    op.drop_index("ix_file_locations_file_id", table_name="paper_file_locations")
    op.drop_table("paper_file_locations")
    op.drop_index("ix_paper_files_version_id", table_name="paper_files")
    op.drop_index("ix_paper_files_paper_id", table_name="paper_files")
    op.drop_index("ix_paper_files_content_hash", table_name="paper_files")
    op.drop_table("paper_files")
    op.drop_constraint("fk_papers_canonical_version", "papers", type_="foreignkey")
    op.drop_index("ix_paper_versions_paper_id", table_name="paper_versions")
    op.drop_table("paper_versions")
    op.drop_index("ix_papers_arxiv_id", table_name="papers")
    op.drop_index("ix_papers_doi", table_name="papers")
    op.drop_table("papers")
    op.drop_table("projects")
