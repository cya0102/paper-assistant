from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from paper_agent.artifacts.service import ArtifactAccessError, ArtifactService
from paper_agent.domain.artifact import (
    ArtifactSelector,
    ArtifactStatus,
    ArtifactType,
    CitationReference,
)
from paper_agent.domain.errors import ErrorCode
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore


class MemoryArtifactRepository:
    """Project-scoped in-memory catalog mirroring the Postgres semantics."""

    def __init__(self) -> None:
        self.items: dict[tuple, object] = {}
        self.by_hash: dict[tuple, object] = {}

    def save(self, descriptor, citations):
        self.items[(descriptor.project_id, descriptor.artifact_id)] = descriptor
        self.by_hash[
            (
                descriptor.project_id,
                descriptor.artifact_type,
                descriptor.schema_version,
                descriptor.content_hash,
            )
        ] = descriptor
        return descriptor

    def get(self, project_id, artifact_id):
        return self.items.get((project_id, artifact_id))

    def find_by_hash(self, project_id, artifact_type, schema_version, content_hash):
        return self.by_hash.get(
            (project_id, artifact_type, schema_version, content_hash)
        )

    def save_citations(self, project_id, artifact_id, citations):
        pass

    def list_citations(self, project_id, artifact_id):
        descriptor = self.items.get((project_id, artifact_id))
        return descriptor.citation_manifest if descriptor else ()

    def mark_expired(self, *, now):
        expired = 0
        for (pid, _aid), descriptor in self.items.items():
            if descriptor.expires_at is not None and descriptor.expires_at < now:
                from dataclasses import replace
                self.items[(pid, _aid)] = replace(
                    descriptor, status=ArtifactStatus.EXPIRED
                )
                expired += 1
        return expired

    def update_status(self, project_id, artifact_id, status):
        from dataclasses import replace

        descriptor = self.items[(project_id, artifact_id)]
        updated = replace(descriptor, status=status)
        self.items[(project_id, artifact_id)] = updated
        self.by_hash[
            (
                updated.project_id,
                updated.artifact_type,
                updated.schema_version,
                updated.content_hash,
            )
        ] = updated

    def search(self, project_id, **kwargs):
        results = [
            descriptor
            for (pid, _aid), descriptor in self.items.items()
            if pid == project_id
        ]
        artifact_type = kwargs.get("artifact_type")
        if artifact_type is not None:
            results = [d for d in results if d.artifact_type == artifact_type]
        created_by = kwargs.get("created_by")
        if created_by is not None:
            results = [d for d in results if d.created_by == created_by]
        query = kwargs.get("query")
        if query:
            results = [d for d in results if query in d.summary]
        return tuple(results)


@pytest.fixture()
def service(tmp_path: Path) -> ArtifactService:
    return ArtifactService(
        LocalArtifactBlobStore(tmp_path),
        MemoryArtifactRepository(),
        retention_days=30,
    )


def _citation() -> CitationReference:
    return CitationReference(
        citation_label="E1",
        paper_id=uuid4(),
        version_id=uuid4(),
        paper_title="Paper",
        section_path="Method",
        page_start=1,
        page_end=2,
    )


def test_materialize_and_read_back(service: ArtifactService) -> None:
    project_id = uuid4()
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.KNOWLEDGE_SEARCH,
        schema_version="search-knowledge-v1",
        media_type="application/json",
        payload={"query": "q", "evidence": [{"citation": "E1", "text": "x"}]},
        summary="one evidence",
        citation_manifest=(_citation(),),
        created_by="test",
    )
    assert descriptor.status == ArtifactStatus.ACTIVE
    slice_ = service.read_slice(
        ArtifactSelector(
            artifact_id=descriptor.artifact_id,
            project_id=project_id,
            view="default",
            max_tokens=800,
        )
    )
    assert slice_.content["query"] == "q"
    assert slice_.citations[0].citation_label == "E1"


def test_content_hash_dedup(service: ArtifactService) -> None:
    project_id = uuid4()
    payload = {"query": "same", "evidence": []}
    first = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.KNOWLEDGE_SEARCH,
        schema_version="search-knowledge-v1",
        media_type="application/json",
        payload=payload,
        summary="s",
        created_by="test",
    )
    second = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.KNOWLEDGE_SEARCH,
        schema_version="search-knowledge-v1",
        media_type="application/json",
        payload=payload,
        summary="s",
        created_by="test",
    )
    assert second.artifact_id == first.artifact_id


def test_same_blob_keeps_distinct_work_unit_provenance(
    service: ArtifactService,
) -> None:
    project_id = uuid4()
    first_work_unit, second_work_unit = uuid4(), uuid4()
    first = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.WORKER_RESULT,
        schema_version="worker-v1",
        media_type="application/json",
        payload={"same": "bytes"},
        summary="first",
        session_id=uuid4(),
        work_unit_id=first_work_unit,
        tool_call_id="worker-1",
    )
    second = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.WORKER_RESULT,
        schema_version="worker-v1",
        media_type="application/json",
        payload={"same": "bytes"},
        summary="second",
        session_id=uuid4(),
        work_unit_id=second_work_unit,
        tool_call_id="worker-2",
    )
    assert first.storage_key == second.storage_key
    assert first.artifact_id != second.artifact_id
    assert first.work_unit_id == first_work_unit
    assert second.work_unit_id == second_work_unit
    assert second.summary == "second"


def test_cross_project_read_rejected(service: ArtifactService) -> None:
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
    with pytest.raises(ArtifactAccessError) as excinfo:
        service.read_slice(
            ArtifactSelector(
                artifact_id=descriptor.artifact_id,
                project_id=uuid4(),
                view="default",
            )
        )
    assert excinfo.value.code == ErrorCode.ARTIFACT_NOT_FOUND


def test_missing_artifact_stable_error(service: ArtifactService) -> None:
    with pytest.raises(ArtifactAccessError) as excinfo:
        service.read_slice(
            ArtifactSelector(artifact_id=uuid4(), project_id=uuid4(), view="default")
        )
    assert excinfo.value.code == ErrorCode.ARTIFACT_NOT_FOUND


def test_expired_artifact_stable_error(service: ArtifactService) -> None:
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
    service._repository.mark_expired(
        now=datetime.now(UTC) + timedelta(days=31)
    )
    with pytest.raises(ArtifactAccessError) as excinfo:
        service.read_slice(
            ArtifactSelector(
                artifact_id=descriptor.artifact_id,
                project_id=project_id,
                view="default",
            )
        )
    assert excinfo.value.code == ErrorCode.ARTIFACT_EXPIRED


def test_active_catalog_row_is_lazily_marked_expired(
    service: ArtifactService,
) -> None:
    from dataclasses import replace

    project_id = uuid4()
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload={"x": 1},
        summary="lazy expiry",
    )
    stale = replace(
        descriptor,
        created_at=datetime.now(UTC) - timedelta(days=2),
        expires_at=datetime.now(UTC) - timedelta(days=1),
        status=ArtifactStatus.ACTIVE,
    )
    service._repository.items[(project_id, descriptor.artifact_id)] = stale

    with pytest.raises(ArtifactAccessError) as excinfo:
        service.read_slice(
            ArtifactSelector(
                artifact_id=descriptor.artifact_id,
                project_id=project_id,
                view="default",
            )
        )

    assert excinfo.value.code == ErrorCode.ARTIFACT_EXPIRED
    stored = service._repository.get(project_id, descriptor.artifact_id)
    assert stored.status == ArtifactStatus.EXPIRED


def test_corrupt_blob_detected(service: ArtifactService, tmp_path: Path) -> None:
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
    blob = tmp_path / ".paper-agent" / "artifacts" / "blobs" / descriptor.storage_key
    import gzip

    blob.write_bytes(gzip.compress(b'{"tampered": true}'))
    with pytest.raises(ArtifactAccessError) as excinfo:
        service.read_slice(
            ArtifactSelector(
                artifact_id=descriptor.artifact_id,
                project_id=project_id,
                view="default",
            )
        )
    assert excinfo.value.code == ErrorCode.ARTIFACT_CORRUPT
    stored = service._repository.get(project_id, descriptor.artifact_id)
    assert stored.status == ArtifactStatus.CORRUPT


def test_idempotent_materialize_repairs_corrupt_blob(
    service: ArtifactService, tmp_path: Path
) -> None:
    project_id, session_id = uuid4(), uuid4()
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload={"x": 1},
        summary="s",
        session_id=session_id,
        tool_call_id="call-1",
    )
    blob = tmp_path / ".paper-agent" / "artifacts" / "blobs" / descriptor.storage_key
    blob.write_bytes(b"corrupt")
    repaired = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload={"x": 1},
        summary="s",
        session_id=session_id,
        tool_call_id="call-1",
    )
    assert repaired.artifact_id == descriptor.artifact_id
    restored = service.read_slice(
        ArtifactSelector(
            artifact_id=repaired.artifact_id,
            project_id=project_id,
            view="full",
            max_tokens=100,
        )
    )
    assert restored.content == {"x": 1}


def test_view_pagination(service: ArtifactService) -> None:
    project_id = uuid4()
    evidence = [
        {"citation": f"E{i}", "paper_id": str(uuid4()), "version_id": str(uuid4()),
         "paper_title": "P", "section_path": "M", "text": "x" * 40}
        for i in range(20)
    ]
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.KNOWLEDGE_SEARCH,
        schema_version="v1",
        media_type="application/json",
        payload={"evidence": evidence, "query": "q"},
        summary="s",
        created_by="test",
    )
    first = service.read_slice(
        ArtifactSelector(
            artifact_id=descriptor.artifact_id,
            project_id=project_id,
            view="evidence",
            max_tokens=50,
        )
    )
    assert first.truncated
    assert first.next_cursor is not None
    second = service.read_slice(
        ArtifactSelector(
            artifact_id=descriptor.artifact_id,
            project_id=project_id,
            view="evidence",
            cursor=first.next_cursor,
            max_tokens=50,
        )
    )
    assert second.content["count"] >= 0


def test_oversized_full_view_is_losslessly_cursor_paginated(
    service: ArtifactService,
) -> None:
    project_id = uuid4()
    payload = {"report": "中文内容" * 200}
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload=payload,
        summary="large",
    )
    fragments: list[str] = []
    cursor = None
    while True:
        slice_ = service.read_slice(
            ArtifactSelector(
                artifact_id=descriptor.artifact_id,
                project_id=project_id,
                view="full",
                cursor=cursor,
                max_tokens=80,
            )
        )
        assert slice_.token_count <= 80
        fragments.append(slice_.content["json_fragment"])
        cursor = slice_.next_cursor
        if cursor is None:
            break
    import json

    assert json.loads("".join(fragments)) == payload


def test_fragment_view_only_returns_citations_visible_on_that_page(
    service: ArtifactService,
) -> None:
    project_id = uuid4()
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload={"prefix": "x" * 300, "citation": "E1", "suffix": "y" * 300},
        summary="fragment citations",
        citation_manifest=(_citation(),),
    )
    cursor = None
    seen: list[str] = []
    while True:
        slice_ = service.read_slice(
            ArtifactSelector(
                artifact_id=descriptor.artifact_id,
                project_id=project_id,
                view="full",
                cursor=cursor,
                max_tokens=60,
            )
        )
        seen.extend(item.citation_label for item in slice_.citations)
        cursor = slice_.next_cursor
        if cursor is None:
            break
    assert seen == ["E1"]
