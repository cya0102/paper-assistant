"""Real PostgreSQL vertical slice for standard Retrieve-Offload-Delegate RAG."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.artifacts.materializer import ToolResultMaterializer
from paper_agent.artifacts.policies import OffloadPolicy
from paper_agent.artifacts.service import ArtifactService
from paper_agent.delegation.runner import WorkerRunner
from paper_agent.delegation.scheduler import Scheduler
from paper_agent.domain.agent import ModelTurn, ToolCall
from paper_agent.domain.artifact import ArtifactType
from paper_agent.domain.enums import SearchStatus
from paper_agent.domain.retrieval import Evidence, SearchKnowledgeResult
from paper_agent.memory import InMemoryCheckpointStore
from paper_agent.rag import (
    EvidenceArtifactMaterializer,
    RagConfig,
    RagResultStatus,
    RagWorkUnitPlanner,
    RetrieveOffloadDelegateService,
    RodResultCollector,
)
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore
from paper_agent.storage.postgres.artifact_repository import (
    SqlAlchemyArtifactRepository,
)
from paper_agent.storage.postgres.models import (
    ProjectRow,
    ResearchArtifactRow,
    ResearchTaskRow,
    WorkUnitRow,
)
from paper_agent.storage.postgres.research_task_repository import (
    SqlAlchemyResearchTaskRepository,
)
from paper_agent.workers import build_worker_registry


DATABASE_URL = os.getenv("PAPER_AGENT_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="PAPER_AGENT_TEST_DATABASE_URL is required for real PostgreSQL tests",
)


class FakeSearch:
    def __init__(self, evidence: tuple[Evidence, ...]) -> None:
        self.evidence = evidence

    def search_knowledge(self, request):
        return SearchKnowledgeResult(
            query=request.query,
            status=SearchStatus.OK,
            resolved_papers=(),
            evidence=self.evidence,
            has_sufficient_evidence=True,
        )


class ChunkModel:
    def start(self, checkpoint, tools):
        brief = json.loads(checkpoint.messages[-1].content)
        return ModelTurn(
            response_id=f"start-{checkpoint.session_id}",
            tool_calls=(
                ToolCall(
                    f"read-{checkpoint.session_id}",
                    "read_artifact",
                    {"artifact_id": brief["input_artifact_ids"][0]},
                ),
            ),
        )

    def continue_with_tools(self, checkpoint, results, tools):
        del tools
        citation = results[0].citation_manifest[0].citation_label
        return ModelTurn(
            response_id=f"final-{checkpoint.session_id}",
            output_text=json.dumps(
                {
                    "relevance": "relevant",
                    "summary": "evidence supports the question",
                    "claims": [
                        {"text": "supported fact", "citations": [citation]}
                    ],
                    "unresolved_questions": [],
                }
            ),
        )


def test_postgres_rod_provenance_and_idempotent_replay(tmp_path: Path) -> None:
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    with factory.begin() as session:
        project = ProjectRow(
            name="rod-e2e", root_path=f"/tmp/rod-e2e-{uuid4()}"
        )
        session.add(project)
        session.flush()
        project_id = project.project_id
    artifacts = ArtifactService(
        LocalArtifactBlobStore(tmp_path),
        SqlAlchemyArtifactRepository(factory),
    )
    config = RagConfig(max_evidence=2, max_workers=2)
    paper_id = uuid4()
    evidence = tuple(
        Evidence(
            evidence_id=uuid4(),
            chunk_id=uuid4(),
            paper_id=paper_id,
            version_id=uuid4(),
            paper_title="Paper",
            section_id=uuid4(),
            section_path="Method",
            page_start=index + 1,
            page_end=index + 1,
            element_ids=(),
            text=f"private chunk {index}",
            relevance=0.9 - index * 0.1,
            dense_score=0.8,
            bm25_score=0.7,
            rerank_score=0.9,
        )
        for index in range(2)
    )
    runner = WorkerRunner(
        registry=build_worker_registry(),
        model=ChunkModel(),
        checkpoints=InMemoryCheckpointStore(),
        artifacts=artifacts,
        materializer=ToolResultMaterializer(artifacts, OffloadPolicy()),
    )
    repository = SqlAlchemyResearchTaskRepository(factory)
    service = RetrieveOffloadDelegateService(
        search=FakeSearch(evidence),
        repository=repository,
        scheduler=Scheduler(runner, max_workers=2),
        evidence_materializer=EvidenceArtifactMaterializer(artifacts),
        planner=RagWorkUnitPlanner(config),
        collector=RodResultCollector(artifacts),
        config=config,
    )
    user_id, session_id = uuid4(), uuid4()
    try:
        first = service.run(
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
            query="what is the method",
        )
        second = service.run(
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
            query="what is the method",
        )

        assert first.status == RagResultStatus.SUPPORTED
        assert second.task_id == first.task_id
        assert second.status == RagResultStatus.SUPPORTED
        with factory() as session:
            assert session.scalar(
                select(func.count())
                .select_from(ResearchTaskRow)
                .where(
                    ResearchTaskRow.project_id == project_id,
                    ResearchTaskRow.task_type == "rag_evidence_analysis",
                )
            ) == 1
            assert session.scalar(
                select(func.count())
                .select_from(WorkUnitRow)
                .where(WorkUnitRow.project_id == project_id)
            ) == 2
            assert session.scalar(
                select(func.count())
                .select_from(ResearchArtifactRow)
                .where(
                    ResearchArtifactRow.project_id == project_id,
                    ResearchArtifactRow.artifact_type
                    == ArtifactType.RETRIEVED_EVIDENCE.value,
                )
            ) == 2
            assert session.scalar(
                select(func.count())
                .select_from(ResearchArtifactRow)
                .where(
                    ResearchArtifactRow.project_id == project_id,
                    ResearchArtifactRow.artifact_type
                    == ArtifactType.WORKER_RESULT.value,
                )
            ) == 2
        assert repository.get_task(uuid4(), first.task_id) is None
        assert artifacts.search(uuid4()) == ()
    finally:
        with factory.begin() as session:
            session.query(ProjectRow).filter(
                ProjectRow.project_id == project_id
            ).delete()
        engine.dispose()
