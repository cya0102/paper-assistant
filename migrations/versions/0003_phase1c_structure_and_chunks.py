"""Add Phase 1C structure, semantic groups, and traceable chunks.

Revision ID: 0003_phase1c
Revises: 0002_phase1b
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0003_phase1c"
down_revision: str | None = "0002_phase1b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[object], sa.Column[object]]:
    return (
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def upgrade() -> None:
    op.create_table(
        "sections",
        sa.Column("section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("parent_section_id", postgresql.UUID(as_uuid=True)),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("normalized_title", sa.Text(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("section_order", sa.Integer(), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("source_heading_block_id", sa.String(length=255)),
        sa.Column("source_block_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("structure_version", sa.String(length=255), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"], ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_sections_version_paper", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_section_id"], ["sections.section_id"],
            name="fk_sections_parent_section_id_sections", ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("section_id", name="pk_sections"),
        sa.UniqueConstraint("version_id", "section_order", name="uq_sections_version_order"),
    )
    op.create_index("ix_sections_version_parent", "sections", ["version_id", "parent_section_id"])

    op.create_table(
        "elements",
        sa.Column("element_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("element_type", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=255)),
        sa.Column("caption", sa.Text()),
        sa.Column("content", sa.Text()),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("bbox_json", postgresql.JSONB()),
        sa.Column("source_block_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("structure_version", sa.String(length=255), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"], ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_elements_version_paper", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["section_id"], ["sections.section_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("element_id", name="pk_elements"),
    )
    op.create_index("ix_elements_version_section", "elements", ["version_id", "section_id"])
    op.create_index("ix_elements_type", "elements", ["element_type"])

    op.create_table(
        "semantic_groups",
        sa.Column("group_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("group_order", sa.Integer(), nullable=False),
        sa.Column("group_type", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("source_block_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("related_element_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("structure_version", sa.String(length=255), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"], ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_semantic_groups_version_paper", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["section_id"], ["sections.section_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("group_id", name="pk_semantic_groups"),
        sa.UniqueConstraint("version_id", "group_order", name="uq_semantic_groups_version_order"),
    )
    op.create_index("ix_semantic_groups_version_section", "semantic_groups", ["version_id", "section_id"])

    op.create_table(
        "chunks",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=False),
        sa.Column("chunk_order", sa.Integer(), nullable=False),
        sa.Column("chunk_type", sa.String(length=32), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("source_group_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("source_block_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("related_element_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("chunking_version", sa.String(length=255), nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"], ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_chunks_version_paper", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["section_id"], ["sections.section_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id", name="pk_chunks"),
        sa.UniqueConstraint("version_id", "chunk_order", name="uq_chunks_version_order"),
    )
    op.create_index("ix_chunks_version_section", "chunks", ["version_id", "section_id"])
    op.create_index("ix_chunks_chunking_version", "chunks", ["chunking_version"])

    op.create_table(
        "derived_data_states",
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("structure_version", sa.String(length=255)),
        sa.Column("chunking_version", sa.String(length=255)),
        *_timestamps(),
        sa.ForeignKeyConstraint(["version_id"], ["paper_versions.version_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("version_id", name="pk_derived_data_states"),
    )


def downgrade() -> None:
    op.drop_table("derived_data_states")
    op.drop_index("ix_chunks_chunking_version", table_name="chunks")
    op.drop_index("ix_chunks_version_section", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_semantic_groups_version_section", table_name="semantic_groups")
    op.drop_table("semantic_groups")
    op.drop_index("ix_elements_type", table_name="elements")
    op.drop_index("ix_elements_version_section", table_name="elements")
    op.drop_table("elements")
    op.drop_index("ix_sections_version_parent", table_name="sections")
    op.drop_table("sections")
