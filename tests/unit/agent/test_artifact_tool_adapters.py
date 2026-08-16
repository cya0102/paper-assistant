from pathlib import Path
from uuid import uuid4

from paper_agent.agent.artifact_tool_adapters import (
    ReadArtifactToolAdapter,
    SearchArtifactToolAdapter,
)
from paper_agent.artifacts.service import ArtifactService
from paper_agent.domain.artifact import ArtifactType
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore
from tests.unit.artifacts.test_artifact_service import MemoryArtifactRepository


def _service(tmp_path: Path) -> ArtifactService:
    return ArtifactService(
        LocalArtifactBlobStore(tmp_path), MemoryArtifactRepository()
    )


def test_read_artifact_view_cursor_and_max_tokens(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = uuid4()
    payload = {
        "query": "q",
        "status": "ok",
        "evidence": [
            {
                "citation": f"E{i}",
                "paper_id": str(uuid4()),
                "version_id": str(uuid4()),
                "paper_title": "P",
                "section_path": "M",
                "text": "x " * 30,
            }
            for i in range(10)
        ],
    }
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.KNOWLEDGE_SEARCH,
        schema_version="v1",
        media_type="application/json",
        payload=payload,
        summary="s",
        created_by="test",
    )
    adapter = ReadArtifactToolAdapter(service, project_id)
    out = adapter.execute(
        {
            "artifact_id": str(descriptor.artifact_id),
            "view": "evidence",
            "max_tokens": 40,
        }
    )
    assert "error" not in out
    assert out["view"] == "evidence"
    assert out["next_cursor"] is not None
    assert out["truncated"] is True
    second = adapter.execute(
        {
            "artifact_id": str(descriptor.artifact_id),
            "view": "evidence",
            "cursor": out["next_cursor"],
            "max_tokens": 40,
        }
    )
    assert "error" not in second
    assert second["content"]["count"] >= 0


def test_read_artifact_enforces_project_isolation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = uuid4()
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload={"x": 1},
        summary="s",
        created_by="test",
    )
    stranger = ReadArtifactToolAdapter(service, uuid4())
    out = stranger.execute({"artifact_id": str(descriptor.artifact_id)})
    assert out["error"] == "artifact_not_found"


def test_read_artifact_rejects_bad_view_and_huge_budget(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = uuid4()
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload={"x": 1},
        summary="s",
        created_by="test",
    )
    adapter = ReadArtifactToolAdapter(service, project_id)
    assert adapter.execute(
        {"artifact_id": str(descriptor.artifact_id), "view": "bogus"}
    )["error"] == "artifact_invalid_view"
    assert adapter.execute(
        {"artifact_id": str(descriptor.artifact_id), "max_tokens": 99999}
    )["error"] == "invalid_request"


def test_search_artifact_structured_filters(tmp_path: Path) -> None:
    service = _service(tmp_path)
    project_id = uuid4()
    service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.PAPER_COMPARISON,
        schema_version="v1",
        media_type="application/json",
        payload={"paper_ids": []},
        summary="comparison of transformer papers",
        created_by="agent",
    )
    service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.KNOWLEDGE_SEARCH,
        schema_version="v1",
        media_type="application/json",
        payload={"evidence": []},
        summary="search about codebooks",
        created_by="worker",
    )
    adapter = SearchArtifactToolAdapter(service, project_id)
    by_type = adapter.execute(
        {"query": "", "artifact_type": "paper_comparison"}
    )
    assert by_type["count"] == 1
    assert by_type["results"][0]["artifact_type"] == "paper_comparison"
    by_query = adapter.execute({"query": "transformer"})
    assert by_query["count"] == 1
    by_creator = adapter.execute({"query": "", "created_by": "worker"})
    assert by_creator["count"] == 1
