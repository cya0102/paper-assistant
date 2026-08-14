import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from paper_agent.database import upgrade_database
from paper_agent.domain.chunk import Chunk, SemanticGroup
from paper_agent.domain.enums import ChunkType, PipelineStage, SemanticGroupType, SourceType
from paper_agent.domain.structure import Section, StructuredDocument
from paper_agent.storage.postgres.models import ChunkRow, PaperRow, PaperVersionRow
from paper_agent.storage.postgres.repositories import SqlAlchemyDerivedDataRepository


DATABASE_URL = os.getenv("PAPER_AGENT_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="PAPER_AGENT_TEST_DATABASE_URL is required for real PostgreSQL tests",
)


def test_phase1c_migration_and_derived_repository_round_trip() -> None:
    assert DATABASE_URL is not None
    upgrade_database(DATABASE_URL)
    engine = create_engine(DATABASE_URL)
    expected = {"sections", "elements", "semantic_groups", "chunks", "derived_data_states"}
    assert expected <= set(inspect(engine).get_table_names())

    paper_id, version_id, section_id, group_id = uuid4(), uuid4(), uuid4(), uuid4()
    structure_version = "integration-structure-v1"
    chunking_version = "integration-chunker-v1"
    document_hash = "d" * 64
    section = Section(
        section_id=section_id,
        paper_id=paper_id,
        version_id=version_id,
        parent_section_id=None,
        title="Introduction",
        normalized_title="introduction",
        level=1,
        section_order=0,
        section_path="Introduction",
        page_start=1,
        page_end=1,
        source_heading_block_id="b1",
        source_block_ids=("b1", "b2"),
        structure_version=structure_version,
    )
    structured = StructuredDocument(
        paper_id=paper_id,
        version_id=version_id,
        structure_version=structure_version,
        sections=(section,),
        elements=(),
    )
    group = SemanticGroup(
        group_id=group_id,
        paper_id=paper_id,
        version_id=version_id,
        section_id=section_id,
        group_order=0,
        group_type=SemanticGroupType.TEXT,
        text="Introduction body",
        token_count=2,
        page_start=1,
        page_end=1,
        source_block_ids=("b1", "b2"),
        related_element_ids=(),
        structure_version=structure_version,
    )
    chunk = Chunk(
        chunk_id=uuid4(),
        paper_id=paper_id,
        version_id=version_id,
        section_id=section_id,
        section_path="Introduction",
        chunk_order=0,
        chunk_type=ChunkType.TEXT,
        text="Introduction body",
        token_count=2,
        page_start=1,
        page_end=1,
        source_group_ids=(group_id,),
        source_block_ids=("b1", "b2"),
        related_element_ids=(),
        chunking_version=chunking_version,
    )

    with Session(engine) as session:
        session.add(PaperRow(paper_id=paper_id, aliases_json=[], authors_json=[], normalized_authors_json=[]))
        session.add(
            PaperVersionRow(
                version_id=version_id,
                paper_id=paper_id,
                source_type=SourceType.LOCAL.value,
                pipeline_status=PipelineStage.PARSED.value,
            )
        )
        session.flush()
        repository = SqlAlchemyDerivedDataRepository(session)
        repository.replace_structure(
            structured,
            (group,),
            document_hash=document_hash,
        )
        session.flush()
        assert repository.load_structure(version_id) == structured
        assert repository.load_groups(version_id) == (group,)
        assert repository.get_state(version_id).chunking_version is None  # type: ignore[union-attr]
        assert repository.get_state(version_id).document_hash == document_hash  # type: ignore[union-attr]

        repository.replace_chunks(version_id, (chunk,), chunking_version)
        session.flush()
        state = repository.get_state(version_id)
        assert state is not None
        assert state.structure_version == structure_version
        assert state.chunking_version == chunking_version
        assert state.document_hash == document_hash
        stored = session.scalar(select(ChunkRow).where(ChunkRow.version_id == version_id))
        assert stored is not None
        assert stored.source_group_ids_json == [str(group_id)]
        assert stored.source_block_ids_json == ["b1", "b2"]
        session.rollback()
    engine.dispose()
