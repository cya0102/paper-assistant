"""End-to-end vertical slice: Offload + Delegate against real PostgreSQL."""

import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.agent.artifact_tool_adapters import ReadArtifactToolAdapter
from paper_agent.artifacts.materializer import ToolResultMaterializer
from paper_agent.artifacts.policies import OffloadPolicy
from paper_agent.artifacts.service import ArtifactService
from paper_agent.artifacts.tokens import count_tokens
from paper_agent.delegation.collector import ResultCollector
from paper_agent.delegation.policy import DelegationPolicy
from paper_agent.delegation.runner import WorkerRunner
from paper_agent.delegation.scheduler import Scheduler
from paper_agent.domain.agent import ModelTurn, ToolCall
from paper_agent.memory import InMemoryCheckpointStore
from paper_agent.research_tasks.planner import ResearchPlanner
from paper_agent.research_tasks.service import ResearchTaskService
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore
from paper_agent.storage.postgres.artifact_repository import SqlAlchemyArtifactRepository
from paper_agent.storage.postgres.models import ProjectRow
from paper_agent.storage.postgres.research_task_repository import (
    SqlAlchemyResearchTaskRepository,
)
from paper_agent.workers.base import build_worker_registry


DATABASE_URL = os.getenv("PAPER_AGENT_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="PAPER_AGENT_TEST_DATABASE_URL is required for real PostgreSQL tests",
)


class ScriptedModel:
    def __init__(self, answer: dict):
        self.answer = answer

    def start(self, checkpoint, tools):
        return ModelTurn(
            response_id="w1",
            tool_calls=(ToolCall("wc1", "search_knowledge", {"query": "x"}),),
        )

    def continue_with_tools(self, checkpoint, results, tools):
        import json

        return ModelTurn(
            response_id="w2",
            output_text=json.dumps(self.answer, ensure_ascii=False),
        )


class FakeSearch:
    def search_knowledge(self, request):
        from paper_agent.domain.enums import SearchStatus
        from paper_agent.domain.retrieval import SearchKnowledgeResult

        return SearchKnowledgeResult(
            query=request.query,
            status=SearchStatus.OK,
            resolved_papers=(),
            evidence=(),
            has_sufficient_evidence=False,
        )


@pytest.fixture()
def stack(tmp_path: Path):
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    with factory.begin() as session:
        project = ProjectRow(name="e2e", root_path=f"/tmp/e2e-{uuid4()}")
        session.add(project)
        session.flush()
        project_id = project.project_id
    artifacts = ArtifactService(
        LocalArtifactBlobStore(tmp_path),
        SqlAlchemyArtifactRepository(factory),
    )
    yield artifacts, project_id
    with factory.begin() as session:
        session.query(ProjectRow).filter(ProjectRow.project_id == project_id).delete()


def test_compare_papers_offload_and_artifact_hydration(stack, tmp_path: Path) -> None:
    artifacts, project_id = stack
    paper_ids = [str(uuid4()) for _ in range(8)]
    raw = {
        "status": "partial",
        "paper_ids": paper_ids,
        "paper_count": 8,
        "reason": None,
        "derivation": {"method": "deterministic"},
        "dimensions": [
            {
                "name": "method",
                "directly_comparable": True,
                "non_comparable_reason": None,
                "cells": [
                    {
                        "paper_id": paper_id,
                        "paper_title": f"Paper {i}",
                        "status": "evidence_backed",
                        "normalized_value": f"method-{i}",
                        "raw_description": "long raw description " + "word " * 40,
                        "directly_comparable": True,
                        "non_comparable_reason": None,
                        "confidence": 0.9,
                        "review_status": "unreviewed",
                        "evidence": [
                            {
                                "citation": f"E{i}",
                                "evidence_id": str(uuid4()),
                                "paper_id": paper_id,
                                "version_id": str(uuid4()),
                                "section_id": str(uuid4()),
                                "chunk_id": str(uuid4()),
                                "element_id": None,
                                "pages": [1, 2],
                                "source_block_ids": ["b1"],
                                "evidence_text": "evidence text " + "word " * 30,
                                "relation_to_target": "supports",
                                "evidence_kind": "paper_fact",
                                "confidence": 0.9,
                            }
                        ],
                    }
                    for i, paper_id in enumerate(paper_ids)
                ],
            }
        ],
        # flattened top-level evidence list (matches the real adapter output)
        "evidence": [
            {
                "citation": f"E{i}",
                "evidence_id": str(uuid4()),
                "paper_id": paper_id,
                "version_id": str(uuid4()),
                "section_id": str(uuid4()),
                "chunk_id": str(uuid4()),
                "element_id": None,
                "pages": [1, 2],
                "source_block_ids": ["b1"],
                "evidence_text": "evidence text " + "word " * 30,
                "relation_to_target": "supports",
                "evidence_kind": "paper_fact",
                "confidence": 0.9,
                "paper_title": f"Paper {i}",
                "section_path": "Research Graph > method",
                "page_start": 1,
                "page_end": 2,
                "text": "evidence text " + "word " * 30,
            }
            for i, paper_id in enumerate(paper_ids)
        ],
    }
    materializer = ToolResultMaterializer(artifacts, OffloadPolicy())
    compact = materializer.materialize(
        project_id=project_id,
        session_id=uuid4(),
        call=ToolCall("cmp-1", "compare_papers", {}),
        raw_payload=raw,
    )
    # 8-paper comparison always offloads; the model view is tiny
    assert compact.artifact_ref is not None
    assert count_tokens(str(compact.model_payload)) < 2000
    assert compact.model_payload["paper_count"] == 8
    # the artifact is fully recoverable through the read_artifact tool
    adapter = ReadArtifactToolAdapter(artifacts, project_id)
    cells = adapter.execute(
        {
            "artifact_id": str(compact.artifact_ref.artifact_id),
            "view": "all-cells",
            "max_tokens": 400,
        }
    )
    assert "error" not in cells
    assert cells["content"]["total"] == 8
    derivation = adapter.execute(
        {
            "artifact_id": str(compact.artifact_ref.artifact_id),
            "view": "derivation",
        }
    )
    assert derivation["content"]["derivation"]["method"] == "deterministic"
    # citation manifest survives in the compact ToolResult
    assert len(compact.citation_manifest) == 8


def test_delegate_workflow_persists_and_collects(stack, tmp_path: Path) -> None:
    artifacts, project_id = stack
    runner = WorkerRunner(
        registry=build_worker_registry(),
        model=ScriptedModel(
            {"workstream": "method", "findings": ["A 使用 X"], "unresolved_questions": ["Q1"]}
        ),
        checkpoints=InMemoryCheckpointStore(),
        artifacts=artifacts,
        materializer=ToolResultMaterializer(artifacts, OffloadPolicy()),
        search_service=FakeSearch(),
    )
    service = ResearchTaskService(
        repository=SqlAlchemyResearchTaskRepository(
            sessionmaker(create_engine(DATABASE_URL, pool_pre_ping=True), class_=Session, expire_on_commit=False)
        ),
        planner=ResearchPlanner(),
        policy=DelegationPolicy(),
        scheduler=Scheduler(runner),
        collector=ResultCollector(artifacts),
        artifacts=artifacts,
    )
    user_id = uuid4()
    summary = service.delegate(
        project_id=project_id,
        user_id=user_id,
        session_id=None,
        objective="比较方法",
        paper_ids=(uuid4(), uuid4()),
        requested_workstreams=("method",),
    )
    assert summary["status"] == "completed"
    assert len(summary["work_unit_ids"]) == 1
    collected = service.collect(project_id=project_id, task_id=UUID(summary["task_id"]))
    assert collected["status"] == "completed"
    assert collected["summary"]
    assert len(collected["artifact_refs"]) == 1
    assert collected["unresolved_questions"] == ["Q1"]
    # worker artifact readable through the artifact service
    from paper_agent.domain.artifact import ArtifactSelector

    ref = collected["artifact_refs"][0]
    slice_ = artifacts.read_slice(
        ArtifactSelector(
            artifact_id=UUID(ref["artifact_id"]),
            project_id=project_id,
            view="result",
            max_tokens=800,
        )
    )
    assert slice_.content["result"]["findings"] == ["A 使用 X"]
    # cross-project collect is refused
    with pytest.raises(LookupError, match="not found"):
        service.collect(project_id=uuid4(), task_id=UUID(summary["task_id"]))
