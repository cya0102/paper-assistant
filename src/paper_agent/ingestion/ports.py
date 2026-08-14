"""Dependency-inversion boundaries for ingestion and persistence."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Protocol, Self
from uuid import UUID

from paper_agent.domain.document import CanonicalParsedDocument
from paper_agent.domain.chunk import Chunk, DerivedDataState, SemanticGroup
from paper_agent.domain.enums import FileStatus, RunStatus
from paper_agent.domain.ingestion import DiscoveredFile, IngestionItemResult, IngestionRequest
from paper_agent.domain.metadata import PaperMetadata
from paper_agent.domain.paper import FileLocation, PaperFile, PaperIdentityResolution
from paper_agent.domain.project import Project
from paper_agent.domain.structure import StructuredDocument


@dataclass(frozen=True, slots=True)
class ParseRequest:
    source_path: Path
    source_file: PaperFile
    identity: PaperIdentityResolution


@dataclass(frozen=True, slots=True)
class ParsedDocumentArtifacts:
    document_json_path: PurePosixPath
    document_markdown_path: PurePosixPath
    assets_path: PurePosixPath
    document_hash: str


class PdfParser(Protocol):
    name: str
    version: str

    def parse(self, request: ParseRequest) -> CanonicalParsedDocument: ...


class PdfMetadataExtractor(Protocol):
    name: str
    version: str

    def extract(self, path: Path) -> PaperMetadata: ...


class IngestionPdfParser(PdfParser, PdfMetadataExtractor, Protocol):
    """Parser capable of both the pre-identity probe and canonical parse."""


class ParsedDocumentStore(Protocol):
    def save(self, document: CanonicalParsedDocument) -> ParsedDocumentArtifacts: ...

    def load(self, paper_id: UUID, version_id: UUID) -> CanonicalParsedDocument: ...

    def exists(self, paper_id: UUID, version_id: UUID) -> bool: ...


class StructureProcessor(Protocol):
    version: str

    def build(
        self, document: CanonicalParsedDocument
    ) -> tuple[StructuredDocument, tuple[SemanticGroup, ...]]: ...


class DocumentChunker(Protocol):
    version: str

    def chunk(
        self,
        structured: StructuredDocument,
        groups: tuple[SemanticGroup, ...],
    ) -> tuple[Chunk, ...]: ...


class PaperIdentityResolver(Protocol):
    def resolve(
        self,
        discovered_file: DiscoveredFile,
        paper_file: PaperFile,
        metadata: PaperMetadata,
        lookup: "PaperIdentityLookup",
    ) -> PaperIdentityResolution: ...


class PaperIdentityLookup(Protocol):
    def find_by_content_hash(
        self, project_id: UUID, content_hash: str
    ) -> PaperIdentityResolution | None: ...

    def find_by_doi(self, doi: str) -> PaperIdentityResolution | None: ...

    def find_by_arxiv_id(self, arxiv_id: str) -> PaperIdentityResolution | None: ...

    def find_by_title_authors(
        self, normalized_title: str, normalized_authors: tuple[str, ...]
    ) -> PaperIdentityResolution | None: ...


class PaperFileRepository(PaperIdentityLookup, Protocol):
    def get_location(
        self, project_id: UUID, relative_path: PurePosixPath
    ) -> tuple[FileLocation, PaperFile] | None: ...

    def find_by_hash(self, project_id: UUID, file_hash: str) -> PaperFile | None: ...

    def get_or_add_file(self, paper_file: PaperFile) -> tuple[PaperFile, bool]: ...

    def upsert_location(self, location: FileLocation) -> None: ...

    def save_identity(self, file_id: UUID, resolution: PaperIdentityResolution) -> PaperFile: ...

    def update_status(self, file_id: UUID, status: FileStatus) -> None: ...

    def save_metadata(self, file_id: UUID, metadata: PaperMetadata) -> PaperFile: ...

    def get_identity(self, file_id: UUID) -> PaperIdentityResolution | None: ...

    def mark_missing_locations(
        self, project_id: UUID, seen_paths: set[PurePosixPath]
    ) -> tuple[PurePosixPath, ...]: ...


class ProjectRepository(Protocol):
    def ensure(self, project: Project) -> None: ...


class ParsedDocumentRepository(Protocol):
    def add(
        self,
        document: CanonicalParsedDocument,
        artifacts: ParsedDocumentArtifacts,
    ) -> None: ...

    def has_current(
        self,
        version_id: UUID,
        parser_name: str,
        parser_version: str,
        schema_version: int,
    ) -> bool: ...

    def current_document_hash(
        self,
        version_id: UUID,
        parser_name: str,
        parser_version: str,
        schema_version: int,
    ) -> str | None: ...


class DerivedDataRepository(Protocol):
    def get_state(self, version_id: UUID) -> DerivedDataState | None: ...

    def replace_structure(
        self,
        structured: StructuredDocument,
        groups: tuple[SemanticGroup, ...],
        *,
        document_hash: str | None,
    ) -> None: ...

    def load_structure(self, version_id: UUID) -> StructuredDocument: ...

    def load_groups(self, version_id: UUID) -> tuple[SemanticGroup, ...]: ...

    def replace_chunks(
        self,
        version_id: UUID,
        chunks: tuple[Chunk, ...],
        chunking_version: str,
    ) -> None: ...


class IngestionRunRepository(Protocol):
    def create(self, run_id: UUID, request: IngestionRequest) -> None: ...

    def record_item(self, run_id: UUID, item: IngestionItemResult) -> None: ...

    def complete(self, run_id: UUID, status: RunStatus, counters: dict[str, int]) -> None: ...


class IngestionUnitOfWork(AbstractContextManager["IngestionUnitOfWork"], Protocol):
    @property
    def projects(self) -> ProjectRepository: ...

    @property
    def files(self) -> PaperFileRepository: ...

    @property
    def documents(self) -> ParsedDocumentRepository: ...

    @property
    def derived(self) -> DerivedDataRepository: ...

    @property
    def runs(self) -> IngestionRunRepository: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class UnitOfWorkFactory(Protocol):
    def __call__(self) -> IngestionUnitOfWork: ...
