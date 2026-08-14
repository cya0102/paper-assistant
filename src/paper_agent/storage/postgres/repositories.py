"""SQLAlchemy repository implementations."""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import PurePosixPath
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from paper_agent.domain.document import CanonicalParsedDocument
from paper_agent.domain.chunk import Chunk, DerivedDataState, SemanticGroup
from paper_agent.domain.document import BoundingBox
from paper_agent.domain.enums import (
    ChunkType,
    ElementType,
    FileStatus,
    IdentityMatchType,
    LocationPresence,
    PipelineStage,
    RunStatus,
    SourceType,
    SemanticGroupType,
)
from paper_agent.domain.ingestion import IngestionItemResult, IngestionRequest
from paper_agent.domain.metadata import PaperMetadata
from paper_agent.domain.paper import FileLocation, Paper, PaperFile, PaperIdentityResolution, PaperVersion
from paper_agent.domain.project import Project
from paper_agent.domain.structure import Element, Section, StructuredDocument
from paper_agent.ingestion.ports import ParsedDocumentArtifacts
from paper_agent.storage.postgres.models import (
    ChunkRow,
    ChunkEmbeddingRow,
    DerivedDataStateRow,
    ElementRow,
    FileLocationRow,
    IngestionItemRow,
    IngestionRunRow,
    IndexingStateRow,
    PaperFileRow,
    PaperEmbeddingRow,
    PaperRow,
    PaperVersionRow,
    ParsedDocumentRow,
    ProjectRow,
    SectionRow,
    SectionEmbeddingRow,
    SemanticGroupRow,
)


def _to_paper_file(row: PaperFileRow) -> PaperFile:
    return PaperFile(
        file_id=row.file_id,
        project_id=row.project_id,
        paper_id=row.paper_id,
        version_id=row.version_id,
        file_size=row.file_size,
        file_hash=row.file_hash,
        content_hash=row.content_hash,
        page_count=row.page_count,
        metadata=row.metadata_json,
        is_canonical=row.is_canonical,
        status=FileStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_file_location(row: FileLocationRow) -> FileLocation:
    return FileLocation(
        location_id=row.location_id,
        project_id=row.project_id,
        file_id=row.file_id,
        relative_path=PurePosixPath(row.relative_path),
        file_name=row.file_name,
        mtime_ns=row.mtime_ns,
        presence=LocationPresence(row.presence_status),
        last_seen_at=row.last_seen_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SqlAlchemyPaperFileRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_location(
        self, project_id: UUID, relative_path: PurePosixPath
    ) -> tuple[FileLocation, PaperFile] | None:
        statement = (
            select(FileLocationRow, PaperFileRow)
            .join(PaperFileRow, FileLocationRow.file_id == PaperFileRow.file_id)
            .where(
                FileLocationRow.project_id == project_id,
                FileLocationRow.relative_path == relative_path.as_posix(),
            )
        )
        result = self._session.execute(statement).one_or_none()
        if result is None:
            return None
        location_row, file_row = result
        return _to_file_location(location_row), _to_paper_file(file_row)

    def find_by_hash(self, project_id: UUID, file_hash: str) -> PaperFile | None:
        row = self._session.scalar(
            select(PaperFileRow).where(
                PaperFileRow.project_id == project_id,
                PaperFileRow.file_hash == file_hash,
            )
        )
        return _to_paper_file(row) if row else None

    def get_or_add_file(self, paper_file: PaperFile) -> tuple[PaperFile, bool]:
        statement = (
            insert(PaperFileRow)
            .values(
                file_id=paper_file.file_id,
                project_id=paper_file.project_id,
                paper_id=paper_file.paper_id,
                version_id=paper_file.version_id,
                file_size=paper_file.file_size,
                file_hash=paper_file.file_hash,
                content_hash=paper_file.content_hash,
                page_count=paper_file.page_count,
                metadata_json=dict(paper_file.metadata),
                is_canonical=paper_file.is_canonical,
                status=paper_file.status.value,
            )
            .on_conflict_do_nothing(index_elements=["project_id", "file_hash"])
            .returning(PaperFileRow.file_id)
        )
        inserted_id = self._session.scalar(statement)
        if inserted_id is not None:
            return paper_file, True
        existing = self.find_by_hash(paper_file.project_id, paper_file.file_hash)
        if existing is None:
            raise RuntimeError("Concurrent file insertion did not return the existing hash")
        return existing, False

    def upsert_location(self, location: FileLocation) -> None:
        row = self._session.scalar(
            select(FileLocationRow).where(
                FileLocationRow.project_id == location.project_id,
                FileLocationRow.relative_path == location.relative_path.as_posix(),
            )
        )
        now = datetime.now(timezone.utc)
        if row is None:
            self._session.add(
                FileLocationRow(
                    location_id=location.location_id,
                    project_id=location.project_id,
                    file_id=location.file_id,
                    relative_path=location.relative_path.as_posix(),
                    file_name=location.file_name,
                    mtime_ns=location.mtime_ns,
                    presence_status=location.presence.value,
                    last_seen_at=now,
                )
            )
            return
        row.file_id = location.file_id
        row.file_name = location.file_name
        row.mtime_ns = location.mtime_ns
        row.presence_status = location.presence.value
        row.last_seen_at = now

    def save_identity(self, file_id: UUID, resolution: PaperIdentityResolution) -> PaperFile:
        paper = resolution.paper
        version = resolution.version
        paper_row = self._session.get(PaperRow, paper.paper_id)
        if paper_row is None:
            paper_row = PaperRow(
                paper_id=paper.paper_id,
                canonical_title=paper.canonical_title,
                normalized_title=paper.normalized_title,
                short_name=paper.short_name,
                acronym=paper.acronym,
                aliases_json=list(paper.aliases),
                authors_json=list(paper.authors),
                normalized_authors_json=list(paper.normalized_authors),
                doi=paper.doi,
                arxiv_id=paper.arxiv_id,
                year=paper.year,
                venue=paper.venue,
                abstract=paper.abstract,
            )
            self._session.add(paper_row)
        else:
            paper_row.canonical_title = paper_row.canonical_title or paper.canonical_title
            paper_row.normalized_title = paper_row.normalized_title or paper.normalized_title
            paper_row.authors_json = paper_row.authors_json or list(paper.authors)
            paper_row.normalized_authors_json = (
                paper_row.normalized_authors_json or list(paper.normalized_authors)
            )
            paper_row.doi = paper_row.doi or paper.doi
            paper_row.arxiv_id = paper_row.arxiv_id or paper.arxiv_id
            paper_row.year = paper_row.year or paper.year
            paper_row.venue = paper_row.venue or paper.venue
        version_row = self._session.get(PaperVersionRow, version.version_id)
        if version_row is None:
            version_row = PaperVersionRow(
                version_id=version.version_id,
                paper_id=version.paper_id,
                version_label=version.version_label,
                source_type=version.source_type.value,
                source_identifier=version.source_identifier,
                parser_version=version.parser_version,
                content_hash=version.content_hash,
                pipeline_status=version.pipeline_stage.value,
            )
            self._session.add(version_row)
        else:
            version_row.parser_version = version.parser_version
            version_row.content_hash = version.content_hash or version_row.content_hash
            version_row.pipeline_status = version.pipeline_stage.value
        if paper_row.canonical_version_id is None:
            paper_row.canonical_version_id = version.version_id

        file_row = self._session.get(PaperFileRow, file_id)
        if file_row is None:
            raise LookupError(f"Paper file does not exist: {file_id}")
        file_row.paper_id = paper.paper_id
        file_row.version_id = version.version_id
        file_row.status = FileStatus.IDENTITY_RESOLVED.value
        self._session.flush()
        return _to_paper_file(file_row)

    def save_metadata(self, file_id: UUID, metadata: PaperMetadata) -> PaperFile:
        row = self._session.get(PaperFileRow, file_id)
        if row is None:
            raise LookupError(f"Paper file does not exist: {file_id}")
        row.content_hash = metadata.content_hash
        row.page_count = metadata.page_count
        row.metadata_json = asdict(metadata)
        self._session.flush()
        return _to_paper_file(row)

    def get_identity(self, file_id: UUID) -> PaperIdentityResolution | None:
        file_row = self._session.get(PaperFileRow, file_id)
        if file_row is None or file_row.paper_id is None or file_row.version_id is None:
            return None
        paper_row = self._session.get(PaperRow, file_row.paper_id)
        version_row = self._session.get(PaperVersionRow, file_row.version_id)
        if paper_row is None or version_row is None:
            return None
        return self._to_identity(paper_row, version_row)

    def find_by_content_hash(
        self, project_id: UUID, content_hash: str
    ) -> PaperIdentityResolution | None:
        row = self._session.scalar(
            select(PaperFileRow).where(
                PaperFileRow.project_id == project_id,
                PaperFileRow.content_hash == content_hash,
                PaperFileRow.paper_id.is_not(None),
                PaperFileRow.version_id.is_not(None),
            )
        )
        return self.get_identity(row.file_id) if row else None

    def find_by_doi(self, doi: str) -> PaperIdentityResolution | None:
        paper = self._session.scalar(select(PaperRow).where(PaperRow.doi == doi))
        return self._canonical_identity(paper)

    def find_by_arxiv_id(self, arxiv_id: str) -> PaperIdentityResolution | None:
        paper = self._session.scalar(select(PaperRow).where(PaperRow.arxiv_id == arxiv_id))
        return self._canonical_identity(paper)

    def find_by_title_authors(
        self, normalized_title: str, normalized_authors: tuple[str, ...]
    ) -> PaperIdentityResolution | None:
        papers = self._session.scalars(
            select(PaperRow).where(PaperRow.normalized_title == normalized_title)
        )
        paper = next(
            (
                candidate
                for candidate in papers
                if tuple(candidate.normalized_authors_json) == normalized_authors
            ),
            None,
        )
        return self._canonical_identity(paper)

    def mark_missing_locations(
        self, project_id: UUID, seen_paths: set[PurePosixPath]
    ) -> tuple[PurePosixPath, ...]:
        rows = self._session.scalars(
            select(FileLocationRow).where(FileLocationRow.project_id == project_id)
        )
        missing: list[PurePosixPath] = []
        for row in rows:
            path = PurePosixPath(row.relative_path)
            if path not in seen_paths and row.presence_status != LocationPresence.MISSING.value:
                row.presence_status = LocationPresence.MISSING.value
                missing.append(path)
        return tuple(sorted(missing, key=lambda path: str(path).casefold()))

    def _canonical_identity(self, paper: PaperRow | None) -> PaperIdentityResolution | None:
        if paper is None:
            return None
        version = None
        if paper.canonical_version_id:
            version = self._session.get(PaperVersionRow, paper.canonical_version_id)
        if version is None:
            version = self._session.scalar(
                select(PaperVersionRow)
                .where(PaperVersionRow.paper_id == paper.paper_id)
                .order_by(PaperVersionRow.created_at.desc())
            )
        return self._to_identity(paper, version) if version else None

    @staticmethod
    def _to_identity(
        paper: PaperRow, version: PaperVersionRow
    ) -> PaperIdentityResolution:
        return PaperIdentityResolution(
            paper=Paper(
                paper_id=paper.paper_id,
                canonical_title=paper.canonical_title,
                normalized_title=paper.normalized_title,
                short_name=paper.short_name,
                acronym=paper.acronym,
                aliases=tuple(paper.aliases_json),
                authors=tuple(paper.authors_json),
                normalized_authors=tuple(paper.normalized_authors_json),
                doi=paper.doi,
                arxiv_id=paper.arxiv_id,
                year=paper.year,
                venue=paper.venue,
                abstract=paper.abstract,
                canonical_version_id=paper.canonical_version_id,
                created_at=paper.created_at,
                updated_at=paper.updated_at,
            ),
            version=PaperVersion(
                version_id=version.version_id,
                paper_id=version.paper_id,
                version_label=version.version_label,
                source_type=SourceType(version.source_type),
                source_identifier=version.source_identifier,
                parser_version=version.parser_version,
                content_hash=version.content_hash,
                pipeline_stage=PipelineStage(version.pipeline_status),
                created_at=version.created_at,
                updated_at=version.updated_at,
            ),
            match_type=IdentityMatchType.EXISTING_FILE,
        )

    def update_status(self, file_id: UUID, status: FileStatus) -> None:
        row = self._session.get(PaperFileRow, file_id)
        if row is None:
            raise LookupError(f"Paper file does not exist: {file_id}")
        row.status = status.value
        if row.version_id is not None:
            version = self._session.get(PaperVersionRow, row.version_id)
            if version is not None:
                version.pipeline_status = status.value


class SqlAlchemyProjectRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def ensure(self, project: Project) -> None:
        row = self._session.get(ProjectRow, project.project_id)
        if row is None:
            self._session.add(
                ProjectRow(
                    project_id=project.project_id,
                    name=project.name,
                    root_path=str(project.root_path),
                )
            )
            return
        if row.root_path != str(project.root_path):
            raise ValueError(
                f"Project {project.project_id} is already bound to a different root path"
            )
        row.name = project.name


class SqlAlchemyParsedDocumentRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        document: CanonicalParsedDocument,
        artifacts: ParsedDocumentArtifacts,
    ) -> None:
        row = self._session.scalar(
            select(ParsedDocumentRow).where(
                ParsedDocumentRow.version_id == document.version_id,
                ParsedDocumentRow.parser_name == document.parser.name,
                ParsedDocumentRow.parser_version == document.parser.version,
                ParsedDocumentRow.schema_version == document.schema_version,
            )
        )
        if row is None:
            row = ParsedDocumentRow(
                parsed_document_id=uuid4(),
                paper_id=document.paper_id,
                version_id=document.version_id,
                source_file_id=document.source_file_id,
                schema_version=document.schema_version,
                parser_name=document.parser.name,
                parser_version=document.parser.version,
                document_json_path=artifacts.document_json_path.as_posix(),
                document_markdown_path=artifacts.document_markdown_path.as_posix(),
                assets_path=artifacts.assets_path.as_posix(),
                document_hash=artifacts.document_hash,
            )
            self._session.add(row)
            return
        row.source_file_id = document.source_file_id
        row.document_json_path = artifacts.document_json_path.as_posix()
        row.document_markdown_path = artifacts.document_markdown_path.as_posix()
        row.assets_path = artifacts.assets_path.as_posix()
        row.document_hash = artifacts.document_hash

    def has_current(
        self,
        version_id: UUID,
        parser_name: str,
        parser_version: str,
        schema_version: int,
    ) -> bool:
        return self.current_document_hash(
            version_id,
            parser_name,
            parser_version,
            schema_version,
        ) is not None

    def current_document_hash(
        self,
        version_id: UUID,
        parser_name: str,
        parser_version: str,
        schema_version: int,
    ) -> str | None:
        return self._session.scalar(
            select(ParsedDocumentRow.document_hash).where(
                ParsedDocumentRow.version_id == version_id,
                ParsedDocumentRow.parser_name == parser_name,
                ParsedDocumentRow.parser_version == parser_version,
                ParsedDocumentRow.schema_version == schema_version,
            )
        )


class SqlAlchemyDerivedDataRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_state(self, version_id: UUID) -> DerivedDataState | None:
        row = self._session.get(DerivedDataStateRow, version_id)
        if row is None:
            return None
        return DerivedDataState(
            version_id=row.version_id,
            structure_version=row.structure_version,
            chunking_version=row.chunking_version,
            document_hash=row.document_hash,
        )

    def replace_structure(
        self,
        structured: StructuredDocument,
        groups: tuple[SemanticGroup, ...],
        *,
        document_hash: str | None,
    ) -> None:
        version_id = structured.version_id
        self._invalidate_index(version_id)
        self._session.execute(delete(ChunkRow).where(ChunkRow.version_id == version_id))
        self._session.execute(
            delete(SemanticGroupRow).where(SemanticGroupRow.version_id == version_id)
        )
        self._session.execute(delete(ElementRow).where(ElementRow.version_id == version_id))
        self._session.execute(delete(SectionRow).where(SectionRow.version_id == version_id))
        self._session.flush()

        for section in sorted(structured.sections, key=lambda item: item.section_order):
            self._session.add(
                SectionRow(
                    section_id=section.section_id,
                    paper_id=section.paper_id,
                    version_id=section.version_id,
                    parent_section_id=section.parent_section_id,
                    title=section.title,
                    normalized_title=section.normalized_title,
                    level=section.level,
                    section_order=section.section_order,
                    section_path=section.section_path,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    source_heading_block_id=section.source_heading_block_id,
                    source_block_ids_json=list(section.source_block_ids),
                    structure_version=section.structure_version,
                )
            )
            self._session.flush()
        for element in structured.elements:
            self._session.add(
                ElementRow(
                    element_id=element.element_id,
                    paper_id=element.paper_id,
                    version_id=element.version_id,
                    section_id=element.section_id,
                    element_type=element.element_type.value,
                    label=element.label,
                    caption=element.caption,
                    content=element.content,
                    page=element.page,
                    bbox_json=element.bbox.model_dump() if element.bbox else None,
                    source_block_ids_json=list(element.source_block_ids),
                    structure_version=element.structure_version,
                )
            )
        for group in groups:
            self._session.add(
                SemanticGroupRow(
                    group_id=group.group_id,
                    paper_id=group.paper_id,
                    version_id=group.version_id,
                    section_id=group.section_id,
                    group_order=group.group_order,
                    group_type=group.group_type.value,
                    text=group.text,
                    token_count=group.token_count,
                    page_start=group.page_start,
                    page_end=group.page_end,
                    source_block_ids_json=list(group.source_block_ids),
                    related_element_ids_json=[str(value) for value in group.related_element_ids],
                    structure_version=group.structure_version,
                )
            )
        state = self._session.get(DerivedDataStateRow, version_id)
        if state is None:
            self._session.add(
                DerivedDataStateRow(
                    version_id=version_id,
                    structure_version=structured.structure_version,
                    chunking_version=None,
                    document_hash=document_hash,
                )
            )
        else:
            state.structure_version = structured.structure_version
            state.chunking_version = None
            state.document_hash = document_hash

    def load_structure(self, version_id: UUID) -> StructuredDocument:
        section_rows = tuple(
            self._session.scalars(
                select(SectionRow)
                .where(SectionRow.version_id == version_id)
                .order_by(SectionRow.section_order)
            )
        )
        if not section_rows:
            raise LookupError(f"No derived structure exists for version {version_id}")
        element_rows = tuple(
            self._session.scalars(
                select(ElementRow)
                .where(ElementRow.version_id == version_id)
                .order_by(ElementRow.created_at, ElementRow.element_id)
            )
        )
        sections = tuple(
            Section(
                section_id=row.section_id,
                paper_id=row.paper_id,
                version_id=row.version_id,
                parent_section_id=row.parent_section_id,
                title=row.title,
                normalized_title=row.normalized_title,
                level=row.level,
                section_order=row.section_order,
                section_path=row.section_path,
                page_start=row.page_start,
                page_end=row.page_end,
                source_heading_block_id=row.source_heading_block_id,
                source_block_ids=tuple(row.source_block_ids_json),
                structure_version=row.structure_version,
            )
            for row in section_rows
        )
        elements = tuple(
            Element(
                element_id=row.element_id,
                paper_id=row.paper_id,
                version_id=row.version_id,
                section_id=row.section_id,
                element_type=ElementType(row.element_type),
                label=row.label,
                caption=row.caption,
                content=row.content,
                page=row.page,
                bbox=BoundingBox.model_validate(row.bbox_json) if row.bbox_json else None,
                source_block_ids=tuple(row.source_block_ids_json),
                structure_version=row.structure_version,
            )
            for row in element_rows
        )
        return StructuredDocument(
            paper_id=section_rows[0].paper_id,
            version_id=version_id,
            structure_version=section_rows[0].structure_version,
            sections=sections,
            elements=elements,
        )

    def load_groups(self, version_id: UUID) -> tuple[SemanticGroup, ...]:
        rows = self._session.scalars(
            select(SemanticGroupRow)
            .where(SemanticGroupRow.version_id == version_id)
            .order_by(SemanticGroupRow.group_order)
        )
        return tuple(
            SemanticGroup(
                group_id=row.group_id,
                paper_id=row.paper_id,
                version_id=row.version_id,
                section_id=row.section_id,
                group_order=row.group_order,
                group_type=SemanticGroupType(row.group_type),
                text=row.text,
                token_count=row.token_count,
                page_start=row.page_start,
                page_end=row.page_end,
                source_block_ids=tuple(row.source_block_ids_json),
                related_element_ids=tuple(UUID(value) for value in row.related_element_ids_json),
                structure_version=row.structure_version,
            )
            for row in rows
        )

    def replace_chunks(
        self,
        version_id: UUID,
        chunks: tuple[Chunk, ...],
        chunking_version: str,
    ) -> None:
        self._invalidate_index(version_id)
        self._session.execute(delete(ChunkRow).where(ChunkRow.version_id == version_id))
        for chunk in chunks:
            if chunk.version_id != version_id or chunk.chunking_version != chunking_version:
                raise ValueError("Chunk identity/version does not match replacement target")
            self._session.add(
                ChunkRow(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    version_id=chunk.version_id,
                    section_id=chunk.section_id,
                    section_path=chunk.section_path,
                    chunk_order=chunk.chunk_order,
                    chunk_type=chunk.chunk_type.value,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    source_group_ids_json=[str(value) for value in chunk.source_group_ids],
                    source_block_ids_json=list(chunk.source_block_ids),
                    related_element_ids_json=[str(value) for value in chunk.related_element_ids],
                    chunking_version=chunk.chunking_version,
                )
            )
        state = self._session.get(DerivedDataStateRow, version_id)
        if state is None or state.structure_version is None:
            raise LookupError("Structure must exist before chunks can be replaced")
        state.chunking_version = chunking_version

    def _invalidate_index(self, version_id: UUID) -> None:
        for model in (
            ChunkEmbeddingRow,
            SectionEmbeddingRow,
            PaperEmbeddingRow,
            IndexingStateRow,
        ):
            self._session.execute(delete(model).where(model.version_id == version_id))


class SqlAlchemyIngestionRunRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, run_id: UUID, request: IngestionRequest) -> None:
        self._session.add(
            IngestionRunRow(
                run_id=run_id,
                project_id=request.project_id,
                requested_paths_json=[str(path) for path in request.paths],
                recursive=request.recursive,
                force_reindex=request.force_reindex,
                status=RunStatus.RUNNING.value,
                counters_json={},
            )
        )

    def record_item(self, run_id: UUID, item: IngestionItemResult) -> None:
        row = self._session.scalar(
            select(IngestionItemRow).where(
                IngestionItemRow.run_id == run_id,
                IngestionItemRow.relative_path == item.relative_path.as_posix(),
            )
        )
        if row is None:
            self._session.add(
                IngestionItemRow(
                    item_id=uuid4(),
                    run_id=run_id,
                    relative_path=item.relative_path.as_posix(),
                    file_id=item.file_id,
                    paper_id=item.paper_id,
                    version_id=item.version_id,
                    stage=item.stage.value,
                    disposition=item.disposition.value,
                    error_code=item.error_code.value if item.error_code else None,
                    error_message=item.error_message,
                )
            )
            return
        row.file_id = item.file_id
        row.paper_id = item.paper_id
        row.version_id = item.version_id
        row.stage = item.stage.value
        row.disposition = item.disposition.value
        row.error_code = item.error_code.value if item.error_code else None
        row.error_message = item.error_message
        row.attempt_count += 1

    def complete(self, run_id: UUID, status: RunStatus, counters: dict[str, int]) -> None:
        row = self._session.get(IngestionRunRow, run_id)
        if row is None:
            raise LookupError(f"Ingestion run does not exist: {run_id}")
        row.status = status.value
        row.counters_json = counters
        row.completed_at = datetime.now(timezone.utc)
