"""Add pgvector hierarchical indexes and PostgreSQL full-text search.

Revision ID: 0004_phase2a
Revises: 0003_phase1c
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from paper_agent.storage.postgres.vector_type import VectorType

revision: str = "0004_phase2a"
down_revision: str | None = "0003_phase1c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[object], sa.Column[object]]:
    return (
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column(
        "papers",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple', coalesce(canonical_title, '') || ' ' || "
                "coalesce(short_name, '') || ' ' || coalesce(acronym, '') || ' ' || "
                "coalesce(doi, '') || ' ' || coalesce(arxiv_id, '') || ' ' || "
                "coalesce(venue, '') || ' ' || coalesce(abstract, ''))",
                persisted=True,
            ),
        ),
    )
    op.create_index("ix_papers_search_vector", "papers", ["search_vector"], postgresql_using="gin")
    op.add_column(
        "sections",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(section_path, ''))",
                persisted=True,
            ),
        ),
    )
    op.create_index("ix_sections_search_vector", "sections", ["search_vector"], postgresql_using="gin")
    op.add_column(
        "chunks",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple', coalesce(section_path, '') || ' ' || coalesce(text, ''))",
                persisted=True,
            ),
        ),
    )
    op.create_index("ix_chunks_search_vector", "chunks", ["search_vector"], postgresql_using="gin")

    op.create_table(
        "embedding_configs",
        sa.Column("embedding_version", sa.String(length=512), nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("provider_version", sa.String(length=128), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("embedding_version", name="pk_embedding_configs"),
    )
    op.create_table(
        "indexing_states",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("embedding_version", sa.String(length=512), nullable=False),
        sa.Column("index_version", sa.String(length=255), nullable=False),
        sa.Column("source_digest", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"], ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_indexing_states_version_paper", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["embedding_version"], ["embedding_configs.embedding_version"]),
        sa.PrimaryKeyConstraint("project_id", "version_id", name="pk_indexing_states"),
    )
    op.create_table(
        "paper_embeddings",
        sa.Column("embedding_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("embedding_version", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", VectorType(256), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"], ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_paper_embeddings_version_paper", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["embedding_version"], ["embedding_configs.embedding_version"]),
        sa.PrimaryKeyConstraint("embedding_id", name="pk_paper_embeddings"),
        sa.UniqueConstraint("project_id", "version_id", name="uq_paper_embeddings_project_version"),
    )
    op.create_table(
        "section_embeddings",
        sa.Column("embedding_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("embedding_version", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", VectorType(256), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"], ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_section_embeddings_version_paper", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["section_id"], ["sections.section_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["embedding_version"], ["embedding_configs.embedding_version"]),
        sa.PrimaryKeyConstraint("embedding_id", name="pk_section_embeddings"),
        sa.UniqueConstraint("project_id", "section_id", name="uq_section_embeddings_project_section"),
    )
    op.create_index("ix_section_embeddings_project_paper", "section_embeddings", ["project_id", "paper_id"])
    op.create_table(
        "chunk_embeddings",
        sa.Column("embedding_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("embedding_version", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", VectorType(256), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["project_id"], ["projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"], ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_chunk_embeddings_version_paper", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["section_id"], ["sections.section_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.chunk_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["embedding_version"], ["embedding_configs.embedding_version"]),
        sa.PrimaryKeyConstraint("embedding_id", name="pk_chunk_embeddings"),
        sa.UniqueConstraint("project_id", "chunk_id", name="uq_chunk_embeddings_project_chunk"),
    )
    op.create_index("ix_chunk_embeddings_project_section", "chunk_embeddings", ["project_id", "section_id"])
    op.execute(
        "CREATE INDEX ix_paper_embeddings_vector_hnsw ON paper_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX ix_section_embeddings_vector_hnsw ON section_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX ix_chunk_embeddings_vector_hnsw ON chunk_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_index("ix_chunk_embeddings_vector_hnsw", table_name="chunk_embeddings")
    op.drop_index("ix_section_embeddings_vector_hnsw", table_name="section_embeddings")
    op.drop_index("ix_paper_embeddings_vector_hnsw", table_name="paper_embeddings")
    op.drop_index("ix_chunk_embeddings_project_section", table_name="chunk_embeddings")
    op.drop_table("chunk_embeddings")
    op.drop_index("ix_section_embeddings_project_paper", table_name="section_embeddings")
    op.drop_table("section_embeddings")
    op.drop_table("paper_embeddings")
    op.drop_table("indexing_states")
    op.drop_table("embedding_configs")
    op.drop_index("ix_chunks_search_vector", table_name="chunks")
    op.drop_column("chunks", "search_vector")
    op.drop_index("ix_sections_search_vector", table_name="sections")
    op.drop_column("sections", "search_vector")
    op.drop_index("ix_papers_search_vector", table_name="papers")
    op.drop_column("papers", "search_vector")
