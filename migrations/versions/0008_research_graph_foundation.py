"""Add evidence-first Research Graph foundation.

Revision ID: 0008_research_graph
Revises: 0007_phase3
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0008_research_graph"
down_revision: str | None = "0007_phase3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> tuple[sa.Column[object], sa.Column[object]]:
    return (
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def _derivation_columns() -> tuple[sa.Column[object], ...]:
    return (
        sa.Column("extraction_method", sa.String(length=128), nullable=False),
        sa.Column("extractor_version", sa.String(length=255), nullable=False),
        sa.Column("schema_version", sa.String(length=255), nullable=False),
        sa.Column("model_name", sa.String(length=255)),
        sa.Column("prompt_version", sa.String(length=255)),
        sa.Column("source_document_hash", sa.String(length=64)),
        sa.Column("chunking_version", sa.String(length=255)),
        sa.Column("generation_key", sa.String(length=64), nullable=False),
    )


def upgrade() -> None:
    op.create_table(
        "paper_profiles",
        sa.Column("profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        *_derivation_columns(),
        sa.Column(
            "additional_attributes_json",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("superseded_by_profile_id", postgresql.UUID(as_uuid=True)),
        *_timestamps(),
        sa.CheckConstraint(
            "source_document_hash IS NULL OR source_document_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_paper_profiles_source_document_hash_sha256"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_paper_profiles_version_paper",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by_profile_id"],
            ["paper_profiles.profile_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("profile_id", name="pk_paper_profiles"),
    )
    op.create_index(
        "uq_paper_profiles_active_version",
        "paper_profiles",
        ["project_id", "paper_id", "version_id"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.create_index(
        "ix_paper_profiles_project_paper",
        "paper_profiles",
        ["project_id", "paper_id"],
    )

    op.create_table(
        "paper_profile_fields",
        sa.Column("field_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("profile_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("field_name", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("normalized_value", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        *_timestamps(),
        sa.CheckConstraint("ordinal >= 0", name=op.f("ck_paper_profile_fields_ordinal_nonnegative")),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name=op.f("ck_paper_profile_fields_confidence_range"),
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"], ["paper_profiles.profile_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("field_id", name="pk_paper_profile_fields"),
        sa.UniqueConstraint(
            "profile_id",
            "field_name",
            "ordinal",
            name="uq_profile_fields_profile_name_ordinal",
        ),
    )
    op.create_index("ix_profile_fields_name", "paper_profile_fields", ["field_name"])

    op.create_table(
        "claims",
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_type", sa.String(length=64), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("normalized_statement", sa.Text(), nullable=False),
        sa.Column("polarity", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        *_derivation_columns(),
        sa.Column("claim_key", sa.String(length=64), nullable=False),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("entailment_status", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("superseded_by_claim_id", postgresql.UUID(as_uuid=True)),
        *_timestamps(),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name=op.f("ck_claims_confidence_range")
        ),
        sa.CheckConstraint(
            "source_document_hash IS NULL OR source_document_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_claims_source_document_hash_sha256"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_claims_version_paper",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by_claim_id"], ["claims.claim_id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("claim_id", name="pk_claims"),
    )
    op.create_index(
        "uq_claims_active_key",
        "claims",
        ["project_id", "claim_key"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.create_index(
        "ix_claims_project_paper_type",
        "claims",
        ["project_id", "paper_id", "claim_type"],
    )

    op.create_table(
        "research_entities",
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.String(length=512), nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("normalization_status", sa.String(length=32), nullable=False),
        *_derivation_columns(),
        sa.Column(
            "attributes_json",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        *_timestamps(),
        sa.CheckConstraint(
            "source_document_hash IS NULL OR source_document_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_research_entities_source_document_hash_sha256"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("entity_id", name="pk_research_entities"),
        sa.UniqueConstraint(
            "project_id",
            "entity_type",
            "normalized_name",
            name="uq_research_entities_project_type_name",
        ),
    )
    op.create_index(
        "ix_research_entities_project_type",
        "research_entities",
        ["project_id", "entity_type"],
    )

    op.create_table(
        "research_entity_aliases",
        sa.Column("alias_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("normalized_alias", sa.String(length=512), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"], ["research_entities.entity_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("alias_id", name="pk_research_entity_aliases"),
        sa.UniqueConstraint(
            "entity_id",
            "normalized_alias",
            name="uq_research_entity_aliases_entity_alias",
        ),
    )
    op.create_index(
        "ix_research_entity_aliases_normalized",
        "research_entity_aliases",
        ["normalized_alias"],
    )

    op.create_table(
        "paper_relations",
        sa.Column("relation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation_type", sa.String(length=64), nullable=False),
        sa.Column("relation_key", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        *_derivation_columns(),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("superseded_by_relation_id", postgresql.UUID(as_uuid=True)),
        *_timestamps(),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name=op.f("ck_paper_relations_confidence_range"),
        ),
        sa.CheckConstraint(
            "source_document_hash IS NULL OR source_document_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_paper_relations_source_document_hash_sha256"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by_relation_id"],
            ["paper_relations.relation_id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("relation_id", name="pk_paper_relations"),
    )
    op.create_index(
        "uq_paper_relations_active_key",
        "paper_relations",
        ["project_id", "relation_key"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.create_index(
        "ix_paper_relations_project_type",
        "paper_relations",
        ["project_id", "relation_type"],
    )
    op.create_index(
        "ix_paper_relations_source",
        "paper_relations",
        ["project_id", "source_type", "source_id"],
    )
    op.create_index(
        "ix_paper_relations_target",
        "paper_relations",
        ["project_id", "target_type", "target_id"],
    )

    op.create_table(
        "evidence_links",
        sa.Column("evidence_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_id", postgresql.UUID(as_uuid=True), nullable=False),
        # Source IDs deliberately remain snapshot identifiers rather than FKs:
        # structure/chunk replacement must not erase historical provenance.
        sa.Column("section_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("element_id", postgresql.UUID(as_uuid=True)),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("source_block_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("evidence_text", sa.Text(), nullable=False),
        sa.Column("relation_to_target", sa.String(length=32), nullable=False),
        sa.Column("evidence_kind", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("evidence_key", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "page_start >= 1 AND page_end >= page_start",
            name=op.f("ck_evidence_links_page_range"),
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name=op.f("ck_evidence_links_confidence_range"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["version_id", "paper_id"],
            ["paper_versions.version_id", "paper_versions.paper_id"],
            name="fk_evidence_links_version_paper",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("evidence_id", name="pk_evidence_links"),
        sa.UniqueConstraint(
            "project_id",
            "target_type",
            "target_id",
            "evidence_key",
            name="uq_evidence_links_target_evidence",
        ),
    )
    op.create_index(
        "ix_evidence_links_target",
        "evidence_links",
        ["project_id", "target_type", "target_id"],
    )
    op.create_index(
        "ix_evidence_links_paper_version",
        "evidence_links",
        ["project_id", "paper_id", "version_id"],
    )
    op.create_index("ix_evidence_links_chunk", "evidence_links", ["chunk_id"])


def downgrade() -> None:
    op.drop_index("ix_evidence_links_chunk", table_name="evidence_links")
    op.drop_index("ix_evidence_links_paper_version", table_name="evidence_links")
    op.drop_index("ix_evidence_links_target", table_name="evidence_links")
    op.drop_table("evidence_links")
    op.drop_index("ix_paper_relations_target", table_name="paper_relations")
    op.drop_index("ix_paper_relations_source", table_name="paper_relations")
    op.drop_index("ix_paper_relations_project_type", table_name="paper_relations")
    op.drop_index("uq_paper_relations_active_key", table_name="paper_relations")
    op.drop_table("paper_relations")
    op.drop_index("ix_research_entity_aliases_normalized", table_name="research_entity_aliases")
    op.drop_table("research_entity_aliases")
    op.drop_index("ix_research_entities_project_type", table_name="research_entities")
    op.drop_table("research_entities")
    op.drop_index("ix_claims_project_paper_type", table_name="claims")
    op.drop_index("uq_claims_active_key", table_name="claims")
    op.drop_table("claims")
    op.drop_index("ix_profile_fields_name", table_name="paper_profile_fields")
    op.drop_table("paper_profile_fields")
    op.drop_index("ix_paper_profiles_project_paper", table_name="paper_profiles")
    op.drop_index("uq_paper_profiles_active_version", table_name="paper_profiles")
    op.drop_table("paper_profiles")
