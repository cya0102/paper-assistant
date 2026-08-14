import os
from uuid import uuid4

import pytest
from sqlalchemy import delete, inspect, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine

from paper_agent.database import upgrade_database
from paper_agent.domain.enums import FileStatus, PipelineStage, SearchStatus, SourceType
from paper_agent.domain.retrieval import MetadataFilter, SearchRequest, SearchScope
from paper_agent.indexing import HashingEmbeddingProvider, HierarchicalIndexingService
from paper_agent.retrieval import LexicalHybridReranker, SearchKnowledgeService
from paper_agent.storage.postgres.index_repository import SqlAlchemyIndexRepository
from paper_agent.storage.postgres.models import (
    ChunkEmbeddingRow,
    ChunkRow,
    PaperFileRow,
    PaperRow,
    PaperVersionRow,
    ProjectRow,
    SectionEmbeddingRow,
    SectionRow,
)
from paper_agent.storage.postgres.search_repository import SqlAlchemySearchRepository


DATABASE_URL = os.getenv("PAPER_AGENT_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="PAPER_AGENT_TEST_DATABASE_URL is required for real PostgreSQL tests",
)


def test_pgvector_hierarchical_index_and_hybrid_search_round_trip() -> None:
    assert DATABASE_URL is not None
    upgrade_database(DATABASE_URL)
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    project_id, paper_id, version_id, section_id = uuid4(), uuid4(), uuid4(), uuid4()
    chunk_one, chunk_two, file_id = uuid4(), uuid4(), uuid4()
    try:
        with factory.begin() as session:
            session.add(ProjectRow(project_id=project_id, name="phase2-test", root_path=f"/tmp/{project_id}"))
            session.add(
                PaperRow(
                    paper_id=paper_id,
                    canonical_title="Scene Codebook Network",
                    normalized_title="scene codebook network",
                    acronym="SCANet",
                    aliases_json=["SCANet"],
                    authors_json=["Ada Researcher"],
                    normalized_authors_json=["ada researcher"],
                    year=2025,
                    venue="CVPR",
                    abstract="A codebook models scene complexity for localization.",
                )
            )
            session.add(
                PaperVersionRow(
                    version_id=version_id,
                    paper_id=paper_id,
                    source_type=SourceType.LOCAL.value,
                    pipeline_status=PipelineStage.CHUNKED.value,
                )
            )
            session.flush()
            session.add(
                PaperFileRow(
                    file_id=file_id,
                    project_id=project_id,
                    paper_id=paper_id,
                    version_id=version_id,
                    file_size=10,
                    file_hash=(paper_id.hex + version_id.hex)[:64],
                    page_count=8,
                    metadata_json={},
                    is_canonical=True,
                    status=FileStatus.CHUNKED.value,
                )
            )
            session.add(
                SectionRow(
                    section_id=section_id,
                    paper_id=paper_id,
                    version_id=version_id,
                    title="3.2 Codebook Learning",
                    normalized_title="codebook learning",
                    level=2,
                    section_order=0,
                    section_path="3 Method > 3.2 Codebook Learning",
                    page_start=5,
                    page_end=6,
                    source_heading_block_id="b1",
                    source_block_ids_json=["b1", "b2", "b3"],
                    structure_version="integration-v1",
                )
            )
            session.flush()
            session.add_all(
                [
                    ChunkRow(
                        chunk_id=chunk_one,
                        paper_id=paper_id,
                        version_id=version_id,
                        section_id=section_id,
                        section_path="3 Method > 3.2 Codebook Learning",
                        chunk_order=0,
                        chunk_type="text",
                        text="The codebook is constructed by clustering scene features into prototypes.",
                        token_count=11,
                        page_start=5,
                        page_end=5,
                        source_group_ids_json=[str(uuid4())],
                        source_block_ids_json=["b2"],
                        related_element_ids_json=[],
                        chunking_version="integration-chunker-v1",
                    ),
                    ChunkRow(
                        chunk_id=chunk_two,
                        paper_id=paper_id,
                        version_id=version_id,
                        section_id=section_id,
                        section_path="3 Method > 3.2 Codebook Learning",
                        chunk_order=1,
                        chunk_type="equation",
                        text="Each feature is assigned to its nearest codebook prototype.",
                        token_count=9,
                        page_start=6,
                        page_end=6,
                        source_group_ids_json=[str(uuid4())],
                        source_block_ids_json=["b3"],
                        related_element_ids_json=[],
                        chunking_version="integration-chunker-v1",
                    ),
                ]
            )

        provider = HashingEmbeddingProvider()
        indexing = HierarchicalIndexingService(
            SqlAlchemyIndexRepository(factory), provider, batch_size=2
        )
        first = indexing.index_version(project_id, version_id)
        second = indexing.index_version(project_id, version_id)
        assert (first.papers, first.sections, first.chunks) == (1, 1, 2)
        assert first.generated == 4
        assert second.unchanged and second.reused == 4

        with factory() as session:
            assert session.scalar(
                select(SectionEmbeddingRow).where(SectionEmbeddingRow.section_id == section_id)
            ) is not None
            assert session.scalar(
                select(ChunkEmbeddingRow).where(ChunkEmbeddingRow.chunk_id == chunk_one)
            ) is not None

        search = SearchKnowledgeService(
            SqlAlchemySearchRepository(factory), provider, LexicalHybridReranker()
        )
        result = search.search_knowledge(
            SearchRequest(
                query="How does SCANet construct the codebook from scene features?",
                scope=SearchScope(project_id=project_id),
                filters=MetadataFilter(year_from=2024, venues=("CVPR",)),
                max_evidence=3,
            )
        )
        assert result.status == SearchStatus.OK
        assert result.resolved_papers[0].paper_id == paper_id
        assert result.evidence
        evidence = result.evidence[0]
        assert evidence.paper_id == paper_id
        assert evidence.version_id == version_id
        assert evidence.section_id == section_id
        assert evidence.chunk_id in (chunk_one, chunk_two)
        assert evidence.page_start in (5, 6)
        assert evidence.dense_score is not None
        assert evidence.bm25_score is not None
        assert evidence.rerank_score is not None

        filtered = search.search_knowledge(
            SearchRequest(
                query="codebook scene features",
                scope=SearchScope(project_id=project_id),
                filters=MetadataFilter(year_to=2020),
            )
        )
        assert filtered.status == SearchStatus.NO_EVIDENCE

        indexes = {item["name"] for item in inspect(engine).get_indexes("chunk_embeddings")}
        assert "ix_chunk_embeddings_vector_hnsw" in indexes
    finally:
        with factory.begin() as session:
            session.execute(delete(ProjectRow).where(ProjectRow.project_id == project_id))
            paper = session.get(PaperRow, paper_id)
            if paper is not None:
                session.delete(paper)
        engine.dispose()
