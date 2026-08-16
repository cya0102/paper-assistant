"""Dependency boundaries for the Artifact layer."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from paper_agent.domain.artifact import (
    ArtifactDescriptor,
    ArtifactSelector,
    ArtifactSlice,
    ArtifactType,
    ArtifactStatus,
    CitationReference,
)


class ArtifactBlobStore(Protocol):
    """Content-addressed binary storage.  Keys are always server generated."""

    def put(self, *, content_hash: str, data: bytes) -> str: ...

    def get(self, *, storage_key: str) -> bytes: ...

    def exists(self, *, storage_key: str) -> bool: ...

    def delete(self, *, storage_key: str) -> None: ...


class ArtifactRepository(Protocol):
    """Project-scoped catalog of stored artifacts."""

    def save(
        self,
        descriptor: ArtifactDescriptor,
        citations: Sequence[CitationReference],
    ) -> ArtifactDescriptor: ...

    def get(
        self, project_id: UUID, artifact_id: UUID
    ) -> ArtifactDescriptor | None: ...

    def find_by_hash(
        self,
        project_id: UUID,
        artifact_type: ArtifactType,
        schema_version: str,
        content_hash: str,
    ) -> ArtifactDescriptor | None: ...

    def save_citations(
        self,
        project_id: UUID,
        artifact_id: UUID,
        citations: Sequence[CitationReference],
    ) -> None: ...

    def list_citations(
        self, project_id: UUID, artifact_id: UUID
    ) -> tuple[CitationReference, ...]: ...

    def mark_expired(self, *, now: datetime) -> int: ...

    def update_status(
        self, project_id: UUID, artifact_id: UUID, status: ArtifactStatus
    ) -> None: ...

    def search(
        self,
        project_id: UUID,
        *,
        artifact_type: ArtifactType | None = None,
        created_by: str | None = None,
        research_task_id: UUID | None = None,
        work_unit_id: UUID | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> tuple[ArtifactDescriptor, ...]: ...


class ArtifactServicePort(Protocol):
    """Application service facade used by the materializer and adapters."""

    def materialize(
        self,
        *,
        project_id: UUID,
        artifact_type: ArtifactType,
        schema_version: str,
        media_type: str,
        payload: dict[str, Any],
        summary: str,
        citation_manifest: Sequence[CitationReference] = (),
        created_by: str = "system",
        session_id: UUID | None = None,
        research_task_id: UUID | None = None,
        work_unit_id: UUID | None = None,
        tool_call_id: str | None = None,
        token_estimate: int | None = None,
        expires_at: datetime | None = None,
    ) -> ArtifactDescriptor: ...

    def read_slice(self, selector: ArtifactSelector) -> ArtifactSlice: ...

    def search(
        self,
        project_id: UUID,
        *,
        artifact_type: ArtifactType | None = None,
        created_by: str | None = None,
        research_task_id: UUID | None = None,
        work_unit_id: UUID | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> tuple[ArtifactDescriptor, ...]: ...

    def validate_hash(self, *, content_hash: str, data: bytes) -> bool: ...
