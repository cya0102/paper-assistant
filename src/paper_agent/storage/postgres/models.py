"""SQLAlchemy mappings for PostgreSQL source and ingestion state."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgresUUID
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from paper_agent.storage.postgres.vector_type import VectorType


EMBEDDING_DIMENSION = 256


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ProjectRow(TimestampMixin, Base):
    __tablename__ = "projects"

    project_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    root_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)


class PaperRow(TimestampMixin, Base):
    __tablename__ = "papers"
    __table_args__ = (
        ForeignKeyConstraint(
            ["canonical_version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_papers_canonical_version",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
        ),
        Index("ix_papers_doi", "doi"),
        Index("ix_papers_arxiv_id", "arxiv_id"),
        Index("ix_papers_search_vector", "search_vector", postgresql_using="gin"),
    )

    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    canonical_title: Mapped[str | None] = mapped_column(Text)
    normalized_title: Mapped[str | None] = mapped_column(Text)
    short_name: Mapped[str | None] = mapped_column(String(255))
    acronym: Mapped[str | None] = mapped_column(String(128))
    aliases_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    authors_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    normalized_authors_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    doi: Mapped[str | None] = mapped_column(String(512))
    arxiv_id: Mapped[str | None] = mapped_column(String(128))
    year: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(String(512))
    abstract: Mapped[str | None] = mapped_column(Text)
    canonical_version_id: Mapped[UUID | None] = mapped_column(PostgresUUID(as_uuid=True))
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', coalesce(canonical_title, '') || ' ' || "
            "coalesce(short_name, '') || ' ' || coalesce(acronym, '') || ' ' || "
            "coalesce(doi, '') || ' ' || coalesce(arxiv_id, '') || ' ' || "
            "coalesce(venue, '') || ' ' || coalesce(abstract, ''))",
            persisted=True,
        ),
    )


class PaperVersionRow(TimestampMixin, Base):
    __tablename__ = "paper_versions"
    __table_args__ = (
        UniqueConstraint("version_id", "paper_id", name="uq_paper_versions_version_paper"),
        Index("ix_paper_versions_paper_id", "paper_id"),
        Index("ix_paper_versions_content_hash", "content_hash"),
        CheckConstraint(
            "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
            name="content_hash_sha256",
        ),
    )

    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    paper_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False
    )
    version_label: Mapped[str | None] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_identifier: Mapped[str | None] = mapped_column(String(512))
    parser_version: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    pipeline_status: Mapped[str] = mapped_column(String(64), nullable=False)


class PaperFileRow(TimestampMixin, Base):
    __tablename__ = "paper_files"
    __table_args__ = (
        UniqueConstraint("project_id", "file_hash", name="uq_paper_files_project_hash"),
        UniqueConstraint("file_id", "project_id", name="uq_paper_files_file_project"),
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_paper_files_version_paper",
            ondelete="SET NULL",
        ),
        CheckConstraint("file_size >= 0", name="file_size_nonnegative"),
        CheckConstraint("page_count >= 0", name="page_count_nonnegative"),
        CheckConstraint("file_hash ~ '^[0-9a-f]{64}$'", name="file_hash_sha256"),
        CheckConstraint(
            "content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'",
            name="content_hash_sha256",
        ),
        CheckConstraint(
            "(paper_id IS NULL AND version_id IS NULL) OR "
            "(paper_id IS NOT NULL AND version_id IS NOT NULL)",
            name="paper_version_pair",
        ),
        Index("ix_paper_files_content_hash", "content_hash"),
        Index("ix_paper_files_paper_id", "paper_id"),
        Index("ix_paper_files_version_id", "version_id"),
    )

    file_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[UUID | None] = mapped_column(PostgresUUID(as_uuid=True))
    version_id: Mapped[UUID | None] = mapped_column(PostgresUUID(as_uuid=True))
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    is_canonical: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)


class FileLocationRow(TimestampMixin, Base):
    __tablename__ = "paper_file_locations"
    __table_args__ = (
        UniqueConstraint("project_id", "relative_path", name="uq_file_locations_project_path"),
        ForeignKeyConstraint(
            ["file_id", "project_id"],
            ["paper_files.file_id", "paper_files.project_id"],
            name="fk_file_locations_file_project",
            ondelete="CASCADE",
        ),
        CheckConstraint("mtime_ns >= 0", name="mtime_nonnegative"),
        Index("ix_file_locations_file_id", "file_id"),
    )

    location_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    file_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    mtime_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    presence_status: Mapped[str] = mapped_column(String(32), nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ParsedDocumentRow(Base):
    __tablename__ = "parsed_documents"
    __table_args__ = (
        UniqueConstraint(
            "version_id",
            "parser_name",
            "parser_version",
            "schema_version",
            name="uq_parsed_documents_version_parser_schema",
        ),
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_parsed_documents_version_paper",
            ondelete="CASCADE",
        ),
        CheckConstraint("document_hash ~ '^[0-9a-f]{64}$'", name="document_hash_sha256"),
        Index("ix_parsed_documents_paper_id", "paper_id"),
    )

    parsed_document_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    source_file_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("paper_files.file_id", ondelete="RESTRICT"), nullable=False
    )
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    parser_name: Mapped[str] = mapped_column(String(255), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(128), nullable=False)
    document_json_path: Mapped[str] = mapped_column(Text, nullable=False)
    document_markdown_path: Mapped[str] = mapped_column(Text, nullable=False)
    assets_path: Mapped[str] = mapped_column(Text, nullable=False)
    document_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SectionRow(TimestampMixin, Base):
    __tablename__ = "sections"
    __table_args__ = (
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_sections_version_paper",
            ondelete="CASCADE",
        ),
        UniqueConstraint("version_id", "section_order", name="uq_sections_version_order"),
        Index("ix_sections_version_parent", "version_id", "parent_section_id"),
        Index("ix_sections_search_vector", "search_vector", postgresql_using="gin"),
    )

    section_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    parent_section_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("sections.section_id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_title: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    section_order: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    source_heading_block_id: Mapped[str | None] = mapped_column(String(255))
    source_block_ids_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    structure_version: Mapped[str] = mapped_column(String(255), nullable=False)
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(section_path, ''))",
            persisted=True,
        ),
    )


class ElementRow(TimestampMixin, Base):
    __tablename__ = "elements"
    __table_args__ = (
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_elements_version_paper",
            ondelete="CASCADE",
        ),
        Index("ix_elements_version_section", "version_id", "section_id"),
        Index("ix_elements_type", "element_type"),
    )

    element_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    section_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("sections.section_id", ondelete="CASCADE"), nullable=False
    )
    element_type: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str | None] = mapped_column(String(255))
    caption: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_json: Mapped[dict[str, float] | None] = mapped_column(JSONB)
    source_block_ids_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    structure_version: Mapped[str] = mapped_column(String(255), nullable=False)


class SemanticGroupRow(TimestampMixin, Base):
    __tablename__ = "semantic_groups"
    __table_args__ = (
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_semantic_groups_version_paper",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "version_id", "group_order", name="uq_semantic_groups_version_order"
        ),
        Index("ix_semantic_groups_version_section", "version_id", "section_id"),
    )

    group_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    section_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("sections.section_id", ondelete="CASCADE"), nullable=False
    )
    group_order: Mapped[int] = mapped_column(Integer, nullable=False)
    group_type: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    source_block_ids_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    related_element_ids_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    structure_version: Mapped[str] = mapped_column(String(255), nullable=False)


class ChunkRow(TimestampMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_chunks_version_paper",
            ondelete="CASCADE",
        ),
        UniqueConstraint("version_id", "chunk_order", name="uq_chunks_version_order"),
        Index("ix_chunks_version_section", "version_id", "section_id"),
        Index("ix_chunks_chunking_version", "chunking_version"),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )

    chunk_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    section_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("sections.section_id", ondelete="CASCADE"), nullable=False
    )
    section_path: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_order: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    source_group_ids_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    source_block_ids_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    related_element_ids_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    chunking_version: Mapped[str] = mapped_column(String(255), nullable=False)
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', coalesce(section_path, '') || ' ' || coalesce(text, ''))",
            persisted=True,
        ),
    )


class DerivedDataStateRow(TimestampMixin, Base):
    __tablename__ = "derived_data_states"
    __table_args__ = (
        CheckConstraint(
            "document_hash IS NULL OR document_hash ~ '^[0-9a-f]{64}$'",
            name="document_hash_sha256",
        ),
    )

    version_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("paper_versions.version_id", ondelete="CASCADE"),
        primary_key=True,
    )
    structure_version: Mapped[str | None] = mapped_column(String(255))
    chunking_version: Mapped[str | None] = mapped_column(String(255))
    document_hash: Mapped[str | None] = mapped_column(String(64))


class EmbeddingConfigRow(Base):
    __tablename__ = "embedding_configs"

    embedding_version: Mapped[str] = mapped_column(String(512), primary_key=True)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_version: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class IndexingStateRow(TimestampMixin, Base):
    __tablename__ = "indexing_states"
    __table_args__ = (
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_indexing_states_version_paper",
            ondelete="CASCADE",
        ),
    )

    project_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("projects.project_id", ondelete="CASCADE"),
        primary_key=True,
    )
    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    embedding_version: Mapped[str] = mapped_column(
        String(512), ForeignKey("embedding_configs.embedding_version"), nullable=False
    )
    index_version: Mapped[str] = mapped_column(String(255), nullable=False)
    source_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class PaperEmbeddingRow(TimestampMixin, Base):
    __tablename__ = "paper_embeddings"
    __table_args__ = (
        UniqueConstraint("project_id", "version_id", name="uq_paper_embeddings_project_version"),
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_paper_embeddings_version_paper",
            ondelete="CASCADE",
        ),
        Index(
            "ix_paper_embeddings_vector_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    embedding_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    embedding_version: Mapped[str] = mapped_column(
        String(512), ForeignKey("embedding_configs.embedding_version"), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[tuple[float, ...]] = mapped_column(VectorType(EMBEDDING_DIMENSION), nullable=False)


class SectionEmbeddingRow(TimestampMixin, Base):
    __tablename__ = "section_embeddings"
    __table_args__ = (
        UniqueConstraint("project_id", "section_id", name="uq_section_embeddings_project_section"),
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_section_embeddings_version_paper",
            ondelete="CASCADE",
        ),
        Index("ix_section_embeddings_project_paper", "project_id", "paper_id"),
        Index(
            "ix_section_embeddings_vector_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    embedding_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    section_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("sections.section_id", ondelete="CASCADE"), nullable=False
    )
    embedding_version: Mapped[str] = mapped_column(
        String(512), ForeignKey("embedding_configs.embedding_version"), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[tuple[float, ...]] = mapped_column(VectorType(EMBEDDING_DIMENSION), nullable=False)


class ChunkEmbeddingRow(TimestampMixin, Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("project_id", "chunk_id", name="uq_chunk_embeddings_project_chunk"),
        ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_chunk_embeddings_version_paper",
            ondelete="CASCADE",
        ),
        Index("ix_chunk_embeddings_project_section", "project_id", "section_id"),
        Index(
            "ix_chunk_embeddings_vector_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    embedding_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    project_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    version_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    section_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("sections.section_id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("chunks.chunk_id", ondelete="CASCADE"), nullable=False
    )
    embedding_version: Mapped[str] = mapped_column(
        String(512), ForeignKey("embedding_configs.embedding_version"), nullable=False
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[tuple[float, ...]] = mapped_column(VectorType(EMBEDDING_DIMENSION), nullable=False)


class InteractionRow(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        Index("ix_interactions_user_created", "user_id", "created_at"),
        Index("ix_interactions_session", "session_id"),
    )

    interaction_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    session_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    paper_ids_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    topics_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    interaction_type: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieved_chunk_ids_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    answer_summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class NoteRow(TimestampMixin, Base):
    __tablename__ = "notes"
    __table_args__ = (Index("ix_notes_user_project", "user_id", "project_id"),)

    note_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), nullable=False)
    project_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("papers.paper_id", ondelete="CASCADE")
    )
    section_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("sections.section_id", ondelete="SET NULL")
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)


class UserPreferenceRow(TimestampMixin, Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True)
    preference_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    preference_value: Mapped[Any] = mapped_column(JSONB, nullable=False)


class IngestionRunRow(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False
    )
    requested_paths_json: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    recursive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    force_reindex: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    counters_json: Mapped[dict[str, int]] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IngestionItemRow(Base):
    __tablename__ = "ingestion_items"
    __table_args__ = (
        UniqueConstraint("run_id", "relative_path", name="uq_ingestion_items_run_path"),
        Index("ix_ingestion_items_file_id", "file_id"),
        Index("ix_ingestion_items_stage", "stage"),
    )

    item_id: Mapped[UUID] = mapped_column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("ingestion_runs.run_id", ondelete="CASCADE"), nullable=False
    )
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("paper_files.file_id", ondelete="SET NULL")
    )
    paper_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("papers.paper_id", ondelete="SET NULL")
    )
    version_id: Mapped[UUID | None] = mapped_column(
        PostgresUUID(as_uuid=True), ForeignKey("paper_versions.version_id", ondelete="SET NULL")
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    disposition: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
