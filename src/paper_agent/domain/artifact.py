"""Artifact domain models: descriptors, references, citations, selectors, slices.

An Artifact is a fully materialized, content-addressed, project-scoped result
stored outside the model context.  The Agent receives only a compact
ArtifactReference plus a bounded slice and may hydrate more through
read_artifact.  All ids, hashes, sizes, token counts and statuses are validated
in this module so no invalid descriptor can reach storage.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be blank")


def _validate_sha256(value: str | None, field_name: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _validate_optional_uuid(value: object, field_name: str) -> None:
    if value is not None and not isinstance(value, UUID):
        raise ValueError(f"{field_name} must be a UUID or None")


def _validate_uuid(value: object, field_name: str) -> None:
    if not isinstance(value, UUID):
        raise ValueError(f"{field_name} must be a UUID")


def _validate_page_range(page_start: int | None, page_end: int | None) -> None:
    if page_start is None and page_end is None:
        return
    if page_start is None or page_end is None:
        raise ValueError("page_start and page_end must both be set or both None")
    if page_start < 1 or page_end < page_start:
        raise ValueError("Invalid page range")


class ArtifactType(StrEnum):
    TOOL_RESULT = "tool_result"
    KNOWLEDGE_SEARCH = "knowledge_search"
    PAPER_READ = "paper_read"
    PAPER_COMPARISON = "paper_comparison"
    WORKER_RESULT = "worker_result"
    RESEARCH_TASK = "research_task"


class ArtifactStatus(StrEnum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CORRUPT = "corrupt"


class StorageBackend(StrEnum):
    LOCAL_CONTENT_ADDRESSED = "local_content_addressed"
    # Future S3/Object-Storage backends extend this enum without changing ports.


@dataclass(frozen=True, slots=True)
class CitationReference:
    """One citable evidence entry attached to an Artifact.

    The Citation Finalizer validates the answer against this manifest alone and
    never needs to read the Artifact blob.  The evidence_hash field is the
    SHA-256 of the evidence text so integrity can be checked independently.
    """

    citation_label: str
    paper_id: UUID
    version_id: UUID
    paper_title: str
    section_path: str
    page_start: int | None = None
    page_end: int | None = None
    evidence_hash: str | None = None
    section_id: UUID | None = None
    chunk_id: UUID | None = None
    element_id: UUID | None = None

    def __post_init__(self) -> None:
        _validate_uuid(self.paper_id, "paper_id")
        _validate_uuid(self.version_id, "version_id")
        _require_text(self.citation_label, "citation_label")
        _require_text(self.paper_title, "paper_title")
        _require_text(self.section_path, "section_path")
        _validate_sha256(self.evidence_hash, "evidence_hash")
        _validate_page_range(self.page_start, self.page_end)
        _validate_optional_uuid(self.section_id, "section_id")
        _validate_optional_uuid(self.chunk_id, "chunk_id")
        _validate_optional_uuid(self.element_id, "element_id")

    @property
    def namespace(self) -> str:
        return self.citation_label[0] if self.citation_label else ""


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """Catalog row describing one stored Artifact."""

    artifact_id: UUID
    project_id: UUID
    artifact_type: ArtifactType
    schema_version: str
    media_type: str
    content_hash: str
    storage_backend: str
    storage_key: str
    byte_size: int
    token_estimate: int
    summary: str
    citation_manifest: tuple[CitationReference, ...] = ()
    status: ArtifactStatus = ArtifactStatus.ACTIVE
    created_by: str = "system"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    session_id: UUID | None = None
    research_task_id: UUID | None = None
    work_unit_id: UUID | None = None
    tool_call_id: str | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        _validate_uuid(self.artifact_id, "artifact_id")
        _validate_uuid(self.project_id, "project_id")
        _require_text(self.schema_version, "schema_version")
        _require_text(self.media_type, "media_type")
        _require_text(self.storage_backend, "storage_backend")
        _require_text(self.storage_key, "storage_key")
        _require_text(self.summary, "summary")
        _require_text(self.created_by, "created_by")
        _validate_sha256(self.content_hash, "content_hash")
        if self.byte_size < 0:
            raise ValueError("byte_size cannot be negative")
        if self.token_estimate < 0:
            raise ValueError("token_estimate cannot be negative")
        if not isinstance(self.artifact_type, ArtifactType):
            raise ValueError("artifact_type must be an ArtifactType")
        if not isinstance(self.status, ArtifactStatus):
            raise ValueError("status must be an ArtifactStatus")
        _validate_optional_uuid(self.session_id, "session_id")
        _validate_optional_uuid(self.research_task_id, "research_task_id")
        _validate_optional_uuid(self.work_unit_id, "work_unit_id")
        if self.tool_call_id is not None and not self.tool_call_id.strip():
            raise ValueError("tool_call_id cannot be blank")
        if self.expires_at is not None and self.expires_at < self.created_at:
            raise ValueError("expires_at cannot precede created_at")
        # ACTIVE is the persisted catalog state; ArtifactService lazily turns it
        # into EXPIRED on read/search. Domain construction must therefore remain
        # valid when the wall clock advances past expires_at.
        labels = [item.citation_label for item in self.citation_manifest]
        if len(labels) != len(set(labels)):
            raise ValueError("citation_manifest labels must be unique")

@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Compact handle handed to the main Agent instead of the full result."""

    artifact_id: UUID
    project_id: UUID
    artifact_type: ArtifactType
    media_type: str
    byte_size: int
    token_estimate: int
    summary: str
    created_by: str
    created_at: datetime
    available_views: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_uuid(self.artifact_id, "artifact_id")
        _validate_uuid(self.project_id, "project_id")
        if self.byte_size < 0 or self.token_estimate < 0:
            raise ValueError("ArtifactReference sizes cannot be negative")
        _require_text(self.media_type, "media_type")
        _require_text(self.summary, "summary")
        _require_text(self.created_by, "created_by")
        if any(not view.strip() for view in self.available_views):
            raise ValueError("available_views cannot contain blank values")

    @classmethod
    def from_descriptor(
        cls, descriptor: ArtifactDescriptor, *, available_views: tuple[str, ...] = ()
    ) -> "ArtifactReference":
        return cls(
            artifact_id=descriptor.artifact_id,
            project_id=descriptor.project_id,
            artifact_type=descriptor.artifact_type,
            media_type=descriptor.media_type,
            byte_size=descriptor.byte_size,
            token_estimate=descriptor.token_estimate,
            summary=descriptor.summary,
            created_by=descriptor.created_by,
            created_at=descriptor.created_at,
            available_views=available_views,
        )


@dataclass(frozen=True, slots=True)
class ArtifactSelector:
    artifact_id: UUID
    project_id: UUID
    view: str = "default"
    cursor: str | None = None
    max_tokens: int = 800

    def __post_init__(self) -> None:
        _validate_uuid(self.artifact_id, "artifact_id")
        _validate_uuid(self.project_id, "project_id")
        if not self.view.strip():
            raise ValueError("view cannot be blank")
        if self.cursor is not None and not self.cursor.strip():
            raise ValueError("cursor cannot be blank")
        if not 1 <= self.max_tokens <= 4000:
            raise ValueError("max_tokens must be between 1 and 4000")


@dataclass(frozen=True, slots=True)
class ArtifactSlice:
    artifact_id: UUID
    project_id: UUID
    view: str
    content: dict[str, Any]
    citations: tuple[CitationReference, ...]
    next_cursor: str | None
    truncated: bool
    token_count: int

    def __post_init__(self) -> None:
        _validate_uuid(self.artifact_id, "artifact_id")
        _validate_uuid(self.project_id, "project_id")
        if self.token_count < 0:
            raise ValueError("token_count cannot be negative")


# ---------------------------------------------------------------------------
# JSON round-trips (used by Redis checkpoints and HTTP boundaries)
# ---------------------------------------------------------------------------

def _parse_iso(value: object) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def citation_to_dict(citation: CitationReference) -> dict[str, Any]:
    return {
        "citation_label": citation.citation_label,
        "paper_id": str(citation.paper_id),
        "version_id": str(citation.version_id),
        "paper_title": citation.paper_title,
        "section_path": citation.section_path,
        "page_start": citation.page_start,
        "page_end": citation.page_end,
        "evidence_hash": citation.evidence_hash,
        "section_id": str(citation.section_id) if citation.section_id is not None else None,
        "chunk_id": str(citation.chunk_id) if citation.chunk_id is not None else None,
        "element_id": str(citation.element_id) if citation.element_id is not None else None,
    }


def citation_from_dict(raw: dict[str, Any]) -> CitationReference:
    return CitationReference(
        citation_label=str(raw["citation_label"]),
        paper_id=UUID(str(raw["paper_id"])),
        version_id=UUID(str(raw["version_id"])),
        paper_title=str(raw["paper_title"]),
        section_path=str(raw["section_path"]),
        page_start=raw.get("page_start"),
        page_end=raw.get("page_end"),
        evidence_hash=raw.get("evidence_hash"),
        section_id=UUID(str(raw["section_id"])) if raw.get("section_id") else None,
        chunk_id=UUID(str(raw["chunk_id"])) if raw.get("chunk_id") else None,
        element_id=UUID(str(raw["element_id"])) if raw.get("element_id") else None,
    )


def artifact_ref_to_dict(ref: ArtifactReference | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    return {
        "artifact_id": str(ref.artifact_id),
        "project_id": str(ref.project_id),
        "artifact_type": ref.artifact_type.value,
        "media_type": ref.media_type,
        "byte_size": ref.byte_size,
        "token_estimate": ref.token_estimate,
        "summary": ref.summary,
        "created_by": ref.created_by,
        "created_at": ref.created_at.isoformat(),
        "available_views": list(ref.available_views),
    }


def artifact_ref_from_dict(raw: dict[str, Any] | None) -> ArtifactReference | None:
    if raw is None:
        return None
    return ArtifactReference(
        artifact_id=UUID(str(raw["artifact_id"])),
        project_id=UUID(str(raw["project_id"])),
        artifact_type=ArtifactType(str(raw["artifact_type"])),
        media_type=str(raw["media_type"]),
        byte_size=int(raw["byte_size"]),
        token_estimate=int(raw["token_estimate"]),
        summary=str(raw["summary"]),
        created_by=str(raw["created_by"]),
        created_at=_parse_iso(raw.get("created_at")) or datetime.now(UTC),
        available_views=tuple(str(value) for value in raw.get("available_views", [])),
    )


def descriptor_to_dict(descriptor: ArtifactDescriptor) -> dict[str, Any]:
    return {
        "artifact_id": str(descriptor.artifact_id),
        "project_id": str(descriptor.project_id),
        "artifact_type": descriptor.artifact_type.value,
        "schema_version": descriptor.schema_version,
        "media_type": descriptor.media_type,
        "content_hash": descriptor.content_hash,
        "storage_backend": descriptor.storage_backend,
        "storage_key": descriptor.storage_key,
        "byte_size": descriptor.byte_size,
        "token_estimate": descriptor.token_estimate,
        "summary": descriptor.summary,
        "citation_manifest": [citation_to_dict(item) for item in descriptor.citation_manifest],
        "status": descriptor.status.value,
        "created_by": descriptor.created_by,
        "created_at": descriptor.created_at.isoformat(),
        "session_id": str(descriptor.session_id) if descriptor.session_id else None,
        "research_task_id": str(descriptor.research_task_id) if descriptor.research_task_id else None,
        "work_unit_id": str(descriptor.work_unit_id) if descriptor.work_unit_id else None,
        "tool_call_id": descriptor.tool_call_id,
        "expires_at": descriptor.expires_at.isoformat() if descriptor.expires_at else None,
    }


def descriptor_from_dict(raw: dict[str, Any]) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=UUID(str(raw["artifact_id"])),
        project_id=UUID(str(raw["project_id"])),
        artifact_type=ArtifactType(str(raw["artifact_type"])),
        schema_version=str(raw["schema_version"]),
        media_type=str(raw["media_type"]),
        content_hash=str(raw["content_hash"]),
        storage_backend=str(raw["storage_backend"]),
        storage_key=str(raw["storage_key"]),
        byte_size=int(raw["byte_size"]),
        token_estimate=int(raw["token_estimate"]),
        summary=str(raw["summary"]),
        citation_manifest=tuple(
            citation_from_dict(item) for item in raw.get("citation_manifest", [])
        ),
        status=ArtifactStatus(str(raw["status"])),
        created_by=str(raw.get("created_by", "system")),
        created_at=_parse_iso(raw.get("created_at")) or datetime.now(UTC),
        session_id=UUID(str(raw["session_id"])) if raw.get("session_id") else None,
        research_task_id=UUID(str(raw["research_task_id"])) if raw.get("research_task_id") else None,
        work_unit_id=UUID(str(raw["work_unit_id"])) if raw.get("work_unit_id") else None,
        tool_call_id=raw.get("tool_call_id"),
        expires_at=_parse_iso(raw.get("expires_at")),
    )
