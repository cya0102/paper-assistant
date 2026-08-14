"""File, paper, and version identity models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from uuid import UUID

from paper_agent.domain.enums import (
    FileStatus,
    IdentityMatchType,
    LocationPresence,
    PipelineStage,
    SourceType,
)


def _validate_sha256(value: str | None, field_name: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class Paper:
    paper_id: UUID
    canonical_title: str | None = None
    normalized_title: str | None = None
    short_name: str | None = None
    acronym: str | None = None
    aliases: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    normalized_authors: tuple[str, ...] = ()
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    canonical_version_id: UUID | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.canonical_title is not None and not self.canonical_title.strip():
            raise ValueError("canonical_title cannot be blank")
        if self.year is not None and not 1000 <= self.year <= 9999:
            raise ValueError("year must be a four-digit year")


@dataclass(frozen=True, slots=True)
class PaperVersion:
    version_id: UUID
    paper_id: UUID
    version_label: str | None = None
    source_type: SourceType = SourceType.LOCAL
    source_identifier: str | None = None
    parser_version: str | None = None
    content_hash: str | None = None
    pipeline_stage: PipelineStage = PipelineStage.IDENTITY_RESOLVED
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        _validate_sha256(self.content_hash, "content_hash")


@dataclass(frozen=True, slots=True)
class PaperFile:
    file_id: UUID
    project_id: UUID
    file_size: int
    file_hash: str
    paper_id: UUID | None = None
    version_id: UUID | None = None
    content_hash: str | None = None
    page_count: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)
    is_canonical: bool = False
    status: FileStatus = FileStatus.DISCOVERED
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.file_size < 0:
            raise ValueError("file_size cannot be negative")
        _validate_sha256(self.file_hash, "file_hash")
        _validate_sha256(self.content_hash, "content_hash")
        if self.page_count < 0:
            raise ValueError("page_count cannot be negative")
        if (self.paper_id is None) != (self.version_id is None):
            raise ValueError("paper_id and version_id must either both be set or both be absent")
        if self.status != FileStatus.DISCOVERED and self.status != FileStatus.FAILED and self.paper_id is None:
            raise ValueError("resolved file states require paper_id and version_id")


@dataclass(frozen=True, slots=True)
class FileLocation:
    location_id: UUID
    project_id: UUID
    file_id: UUID
    relative_path: PurePosixPath
    file_name: str
    mtime_ns: int
    presence: LocationPresence = LocationPresence.PRESENT
    last_seen_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.relative_path.is_absolute() or ".." in self.relative_path.parts:
            raise ValueError("relative_path must remain inside the project")
        if self.relative_path == PurePosixPath("."):
            raise ValueError("relative_path must point to a file")
        if self.file_name != self.relative_path.name:
            raise ValueError("file_name must match relative_path.name")
        if self.mtime_ns < 0:
            raise ValueError("mtime_ns cannot be negative")


@dataclass(frozen=True, slots=True)
class PaperIdentityResolution:
    paper: Paper
    version: PaperVersion
    aliases: tuple[str, ...] = field(default_factory=tuple)
    match_type: IdentityMatchType = IdentityMatchType.NEW_PAPER
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.version.paper_id != self.paper.paper_id:
            raise ValueError("PaperVersion must belong to the resolved Paper")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
