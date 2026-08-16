import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.artifacts.service import ArtifactAccessError, ArtifactService
from paper_agent.domain.artifact import (
    ArtifactSelector,
    ArtifactStatus,
    ArtifactType,
    CitationReference,
)
from paper_agent.domain.errors import ErrorCode
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore
from paper_agent.storage.postgres.artifact_repository import SqlAlchemyArtifactRepository
from paper_agent.storage.postgres.models import ProjectRow


DATABASE_URL = os.getenv("PAPER_AGENT_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="PAPER_AGENT_TEST_DATABASE_URL is required for real PostgreSQL tests",
)


@pytest.fixture()
def service():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    with factory.begin() as session:
        project = ProjectRow(name="artifact-test", root_path=f"/tmp/artifact-{uuid4()}")
        session.add(project)
        session.flush()
        project_id = project.project_id
    artifacts = ArtifactService(
        LocalArtifactBlobStore(Path("/tmp/paper-agent-artifact-test")),
        SqlAlchemyArtifactRepository(factory),
    )
    yield artifacts, project_id
    with factory.begin() as session:
        session.query(ProjectRow).filter(ProjectRow.project_id == project_id).delete()


def test_pg_artifact_round_trip_and_citations(service):
    artifacts, project_id = service
    descriptor = artifacts.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.PAPER_COMPARISON,
        schema_version="paper-comparison-v1",
        media_type="application/json",
        payload={"status": "complete", "paper_ids": [str(uuid4()), str(uuid4())]},
        summary="comparison of two papers",
        citation_manifest=(
            CitationReference(
                citation_label="E1",
                paper_id=uuid4(),
                version_id=uuid4(),
                paper_title="Paper A",
                section_path="Method",
                page_start=1,
                page_end=2,
            ),
        ),
        created_by="integration-test",
    )
    assert descriptor.status == ArtifactStatus.ACTIVE
    loaded = artifacts.search(project_id, artifact_type=ArtifactType.PAPER_COMPARISON)
    assert len(loaded) == 1
    assert loaded[0].citation_manifest[0].citation_label == "E1"
    slice_ = artifacts.read_slice(
        ArtifactSelector(
            artifact_id=descriptor.artifact_id,
            project_id=project_id,
            view="default",
            max_tokens=800,
        )
    )
    assert slice_.content["status"] == "complete"


def test_pg_content_hash_dedup(service):
    artifacts, project_id = service
    payload = {"query": "same"}
    first = artifacts.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.KNOWLEDGE_SEARCH,
        schema_version="search-knowledge-v1",
        media_type="application/json",
        payload=payload,
        summary="s",
        created_by="test",
    )
    second = artifacts.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.KNOWLEDGE_SEARCH,
        schema_version="search-knowledge-v1",
        media_type="application/json",
        payload=payload,
        summary="s",
        created_by="test",
    )
    assert first.artifact_id == second.artifact_id


def test_pg_shared_blob_keeps_distinct_work_unit_provenance(service):
    artifacts, project_id = service
    payload = {"result": {"findings": ["same bytes"]}}
    first = artifacts.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.WORKER_RESULT,
        schema_version="worker-result-v1",
        media_type="application/json",
        payload=payload,
        summary="first worker",
        created_by="paper_analyzer",
        research_task_id=uuid4(),
        work_unit_id=uuid4(),
    )
    second = artifacts.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.WORKER_RESULT,
        schema_version="worker-result-v1",
        media_type="application/json",
        payload=payload,
        summary="second worker",
        created_by="paper_analyzer",
        research_task_id=uuid4(),
        work_unit_id=uuid4(),
    )

    assert first.artifact_id != second.artifact_id
    assert first.content_hash == second.content_hash
    assert first.storage_key == second.storage_key
    assert len(
        artifacts.search(project_id, artifact_type=ArtifactType.WORKER_RESULT)
    ) == 2


def test_pg_cross_project_isolation(service):
    artifacts, project_id = service
    descriptor = artifacts.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload={"x": 1},
        summary="s",
        created_by="test",
    )
    with pytest.raises(ArtifactAccessError) as excinfo:
        artifacts.read_slice(
            ArtifactSelector(
                artifact_id=descriptor.artifact_id,
                project_id=uuid4(),
                view="default",
            )
        )
    assert excinfo.value.code == ErrorCode.ARTIFACT_NOT_FOUND


def test_pg_expiry_and_search(service):
    artifacts, project_id = service
    artifacts.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload={"x": 1},
        summary="unique summary for expiry test",
        created_by="expiry-maker",
    )
    found = artifacts.search(project_id, query="expiry", created_by="expiry-maker")
    assert len(found) == 1
