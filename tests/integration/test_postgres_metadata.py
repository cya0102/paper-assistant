from sqlalchemy import create_mock_engine

from paper_agent.storage.postgres.models import Base, NoteRow


def test_phase1_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == {
        "projects",
        "papers",
        "paper_versions",
        "paper_files",
        "paper_file_locations",
        "parsed_documents",
        "ingestion_runs",
        "ingestion_items",
        "sections",
        "elements",
        "semantic_groups",
        "chunks",
        "derived_data_states",
        "embedding_configs",
        "indexing_states",
        "paper_embeddings",
        "section_embeddings",
        "chunk_embeddings",
        "interactions",
        "notes",
        "user_preferences",
    }


def test_file_hash_is_unique_within_project() -> None:
    constraints = Base.metadata.tables["paper_files"].constraints
    assert any(
        constraint.name == "uq_paper_files_project_hash"
        and {column.name for column in constraint.columns} == {"project_id", "file_hash"}
        for constraint in constraints
    )


def test_phase1b_metadata_columns_are_registered() -> None:
    assert {
        "normalized_title",
        "normalized_authors_json",
    }.issubset(Base.metadata.tables["papers"].columns.keys())
    assert "content_hash" in Base.metadata.tables["paper_versions"].columns.keys()
    assert {"page_count", "metadata_json"}.issubset(
        Base.metadata.tables["paper_files"].columns.keys()
    )


def test_phase1c_provenance_columns_are_registered() -> None:
    assert {"parent_section_id", "source_block_ids_json", "structure_version"}.issubset(
        Base.metadata.tables["sections"].columns.keys()
    )
    assert {"section_id", "source_block_ids_json"}.issubset(
        Base.metadata.tables["elements"].columns.keys()
    )
    assert {
        "paper_id",
        "version_id",
        "section_id",
        "page_start",
        "page_end",
        "source_group_ids_json",
        "source_block_ids_json",
        "related_element_ids_json",
        "chunking_version",
    }.issubset(Base.metadata.tables["chunks"].columns.keys())
    assert "document_hash" in Base.metadata.tables["derived_data_states"].columns
    assert any(
        constraint.name == "ck_derived_data_states_document_hash_sha256"
        for constraint in Base.metadata.tables["derived_data_states"].constraints
    )


def test_phase2a_vector_and_full_text_schema_is_registered() -> None:
    assert "search_vector" in Base.metadata.tables["papers"].columns
    assert "search_vector" in Base.metadata.tables["sections"].columns
    assert "search_vector" in Base.metadata.tables["chunks"].columns
    assert {
        "embedding_version",
        "provider",
        "model",
        "provider_version",
        "dimension",
    }.issubset(Base.metadata.tables["embedding_configs"].columns.keys())
    for table_name in ("paper_embeddings", "section_embeddings", "chunk_embeddings"):
        assert {"embedding_version", "content_hash", "embedding"}.issubset(
            Base.metadata.tables[table_name].columns.keys()
        )


def test_notes_section_foreign_key_preserves_note_on_structure_rebuild() -> None:
    section_foreign_key = next(
        foreign_key
        for foreign_key in NoteRow.__table__.foreign_keys
        if foreign_key.parent.name == "section_id"
    )
    assert section_foreign_key.ondelete == "SET NULL"


def test_postgresql_schema_emits_circular_canonical_version_fk() -> None:
    statements: list[str] = []

    def collect(sql, *args, **kwargs) -> None:
        del args, kwargs
        statements.append(str(sql.compile(dialect=engine.dialect)))

    engine = create_mock_engine("postgresql://", collect)
    Base.metadata.create_all(engine)

    assert any(
        "ALTER TABLE papers ADD CONSTRAINT fk_papers_canonical_version" in statement
        for statement in statements
    )
