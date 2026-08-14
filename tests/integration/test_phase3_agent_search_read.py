import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete, inspect
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.agent.context_builder import ToolEvidenceCitationFormatter
from paper_agent.agent.runtime import AgentRuntime
from paper_agent.agent.tool_adapters import ReadPaperToolAdapter, SearchKnowledgeToolAdapter
from paper_agent.agent.tools import ToolRegistry
from paper_agent.database import upgrade_database
from paper_agent.domain.agent import ModelTurn, ToolCall
from paper_agent.domain.enums import FileStatus, PipelineStage, SourceType
from paper_agent.domain.memory import Interaction, Note, UserPreference
from paper_agent.indexing import HashingEmbeddingProvider, HierarchicalIndexingService
from paper_agent.memory import InMemoryCheckpointStore
from paper_agent.reading import ReadPaperService
from paper_agent.retrieval import LexicalHybridReranker, SearchKnowledgeService
from paper_agent.retrieval.advanced import AdvancedSearchKnowledgeService, ConservativeQueryRewriter, LexicalEvidenceJudge
from paper_agent.storage.postgres.index_repository import SqlAlchemyIndexRepository
from paper_agent.storage.postgres.memory_repository import SqlAlchemyMemoryRepository
from paper_agent.storage.postgres.models import ChunkRow, NoteRow, PaperFileRow, PaperRow, PaperVersionRow, ProjectRow, SectionRow
from paper_agent.storage.postgres.neighbor_repository import SqlAlchemyNeighborRepository
from paper_agent.storage.postgres.read_repository import SqlAlchemyPaperReadRepository
from paper_agent.storage.postgres.search_repository import SqlAlchemySearchRepository


DATABASE_URL = os.getenv("PAPER_AGENT_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="PAPER_AGENT_TEST_DATABASE_URL is required")


class ToolCallingModel:
    def __init__(self, paper_id, version_id):
        self.paper_id = paper_id
        self.version_id = version_id

    def start(self, checkpoint, tools):
        assert {item["name"] for item in tools} == {"search_knowledge", "read_paper"}
        return ModelTurn(
            "response-search-read",
            tool_calls=(
                ToolCall("search", "search_knowledge", {"query": "SCANet codebook scene features"}),
                ToolCall("read", "read_paper", {"paper_id": str(self.paper_id), "version_id": str(self.version_id), "page_range": [5, 5]}),
            ),
        )

    def continue_with_tools(self, checkpoint, results, tools):
        assert {item.name for item in results} == {"search_knowledge", "read_paper"}
        search = next(item for item in results if item.name == "search_knowledge")
        citation = search.payload["evidence"][0]["citation"]
        return ModelTurn("response-final", output_text=f"Codebook 由场景特征聚类得到。[{citation}]")


def test_phase3_agent_search_read_and_memory_round_trip():
    assert DATABASE_URL is not None
    upgrade_database(DATABASE_URL)
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    project_id, paper_id, version_id, section_id = uuid4(), uuid4(), uuid4(), uuid4()
    user_id, session_id, chunk_one, chunk_two = uuid4(), uuid4(), uuid4(), uuid4()
    try:
        assert {"interactions", "notes", "user_preferences"} <= set(inspect(engine).get_table_names())
        with factory.begin() as session:
            session.add(ProjectRow(project_id=project_id, name="phase3", root_path=f"/tmp/{project_id}"))
            session.add(PaperRow(paper_id=paper_id, canonical_title="Scene Codebook Network", normalized_title="scene codebook network", acronym="SCANet", aliases_json=["SCANet"], authors_json=[], normalized_authors_json=[]))
            session.add(PaperVersionRow(version_id=version_id, paper_id=paper_id, source_type=SourceType.LOCAL.value, pipeline_status=PipelineStage.CHUNKED.value))
            session.flush()
            session.add(PaperFileRow(file_id=uuid4(), project_id=project_id, paper_id=paper_id, version_id=version_id, file_size=10, file_hash=(paper_id.hex + version_id.hex)[:64], page_count=6, metadata_json={}, is_canonical=True, status=FileStatus.CHUNKED.value))
            session.add(SectionRow(section_id=section_id, paper_id=paper_id, version_id=version_id, title="3.2 Codebook", normalized_title="codebook", level=2, section_order=0, section_path="3 Method > 3.2 Codebook", page_start=5, page_end=5, source_heading_block_id="b1", source_block_ids_json=["b1", "b2", "b3"], structure_version="phase3-v1"))
            session.flush()
            for order, chunk_id, text in ((0, chunk_one, "The codebook clusters scene features into prototypes."), (1, chunk_two, "Each feature uses its nearest prototype.")):
                session.add(ChunkRow(chunk_id=chunk_id, paper_id=paper_id, version_id=version_id, section_id=section_id, section_path="3 Method > 3.2 Codebook", chunk_order=order, chunk_type="text", text=text, token_count=8, page_start=5, page_end=5, source_group_ids_json=[str(uuid4())], source_block_ids_json=[f"b{order + 2}"], related_element_ids_json=[], chunking_version="phase3-v1"))

        provider = HashingEmbeddingProvider()
        HierarchicalIndexingService(SqlAlchemyIndexRepository(factory), provider).index_version(project_id, version_id)
        search = AdvancedSearchKnowledgeService(
            SearchKnowledgeService(SqlAlchemySearchRepository(factory), provider, LexicalHybridReranker()),
            ConservativeQueryRewriter(),
            LexicalEvidenceJudge(),
            SqlAlchemyNeighborRepository(factory),
            judge_threshold=0.1,
        )
        registry = ToolRegistry()
        registry.register(SearchKnowledgeToolAdapter(search, project_id).contract())
        registry.register(ReadPaperToolAdapter(ReadPaperService(SqlAlchemyPaperReadRepository(factory))).contract())
        answer = AgentRuntime(ToolCallingModel(paper_id, version_id), registry, InMemoryCheckpointStore(), answer_finalizer=ToolEvidenceCitationFormatter()).run(session_id=session_id, user_id=user_id, project_id=project_id, query="How is the codebook built?")
        assert "[E" in answer.text and "来源：" in answer.text
        assert len(answer.tool_results[0].payload["evidence"]) >= 1
        assert len(answer.tool_results[1].payload["passages"]) == 2

        memory = SqlAlchemyMemoryRepository(factory)
        memory.save_interaction(Interaction(user_id=user_id, session_id=session_id, query="codebook", paper_ids=(paper_id,), retrieved_chunk_ids=(chunk_one,), answer_summary="scene features"))
        note = Note(
            user_id=user_id,
            project_id=project_id,
            paper_id=paper_id,
            section_id=section_id,
            content="important",
            tags=("method",),
        )
        memory.save_note(note)
        memory.set_preference(UserPreference(user_id=user_id, key="answer_language", value="zh-CN"))
        assert memory.search_interactions(user_id, "codebook")[0].paper_ids == (paper_id,)
        assert memory.list_notes(user_id, project_id)[0].tags == ("method",)
        assert memory.get_preferences(user_id)[0].value == "zh-CN"

        with factory.begin() as session:
            session.execute(delete(SectionRow).where(SectionRow.section_id == section_id))
            stored_note = session.get(NoteRow, note.note_id)
            assert stored_note is not None
            assert stored_note.section_id is None
    finally:
        with factory.begin() as session:
            from paper_agent.storage.postgres.models import InteractionRow, UserPreferenceRow
            session.execute(delete(InteractionRow).where(InteractionRow.user_id == user_id))
            session.execute(delete(UserPreferenceRow).where(UserPreferenceRow.user_id == user_id))
            session.execute(delete(ProjectRow).where(ProjectRow.project_id == project_id))
            paper = session.get(PaperRow, paper_id)
            if paper is not None:
                session.delete(paper)
        engine.dispose()
