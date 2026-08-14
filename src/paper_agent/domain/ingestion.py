"""Ingestion requests, observations, and results."""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from uuid import UUID

from paper_agent.domain.enums import IngestionDisposition, PipelineStage
from paper_agent.domain.errors import ErrorCode


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    project_id: UUID
    project_root: Path
    paths: tuple[Path, ...] = ()
    recursive: bool = True
    force_reindex: bool = False

    def __post_init__(self) -> None:
        if not self.project_root.is_absolute():
            raise ValueError("project_root must be absolute")


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    absolute_path: Path
    relative_path: PurePosixPath
    file_size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class ScanIssue:
    path: Path
    code: ErrorCode
    message: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    files: tuple[DiscoveredFile, ...] = ()
    issues: tuple[ScanIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    sha256: str
    file_size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class IngestionItemResult:
    relative_path: PurePosixPath
    disposition: IngestionDisposition
    stage: PipelineStage
    file_id: UUID | None = None
    paper_id: UUID | None = None
    version_id: UUID | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionReport:
    run_id: UUID
    scanned: int
    items: tuple[IngestionItemResult, ...] = ()
    scan_issues: tuple[ScanIssue, ...] = ()
    missing: int = 0
    counts: dict[IngestionDisposition, int] = field(init=False)

    def __post_init__(self) -> None:
        counts = {disposition: 0 for disposition in IngestionDisposition}
        for item in self.items:
            counts[item.disposition] += 1
        object.__setattr__(self, "counts", counts)
