"""PostgreSQL implementation of section/page/element paper reading."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.domain.enums import ElementType
from paper_agent.domain.reading import (
    ReadElement,
    ReadPaperRequest,
    ReadPaperResult,
    ReadPassage,
)
from paper_agent.storage.postgres.models import ChunkRow, ElementRow, PaperFileRow, PaperRow, SectionRow


class SqlAlchemyPaperReadRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def read(self, request: ReadPaperRequest) -> ReadPaperResult:
        with self._session_factory() as session:
            # Project membership is checked first so errors do not reveal whether
            # a paper_id exists globally (uniform "not found in project").
            project_file = session.scalar(
                select(PaperFileRow)
                .where(
                    PaperFileRow.project_id == request.project_id,
                    PaperFileRow.paper_id == request.paper_id,
                )
                .order_by(PaperFileRow.is_canonical.desc(), PaperFileRow.updated_at.desc())
                .limit(1)
            )
            if project_file is None:
                raise LookupError("Paper not found in project")
            # Resolve the version inside the project: prefer the project's
            # canonical file; never fall back to the global canonical_version_id,
            # which may belong to another project.
            version_id = request.version_id
            if version_id is None:
                version_id = project_file.version_id
            else:
                version_owned = session.scalar(
                    select(PaperFileRow.file_id).where(
                        PaperFileRow.project_id == request.project_id,
                        PaperFileRow.paper_id == request.paper_id,
                        PaperFileRow.version_id == version_id,
                    )
                )
                if version_owned is None:
                    raise LookupError("Paper version not found in project")
            if version_id is None:
                raise LookupError("Project file has no version")
            paper = session.get(PaperRow, request.paper_id)
            if paper is None:
                raise LookupError("Paper not found in project")
            chunks = list(
                session.scalars(
                    select(ChunkRow)
                    .where(ChunkRow.paper_id == request.paper_id, ChunkRow.version_id == version_id)
                    .order_by(ChunkRow.chunk_order)
                )
            )
            elements = list(
                session.scalars(
                    select(ElementRow)
                    .where(ElementRow.paper_id == request.paper_id, ElementRow.version_id == version_id)
                    .order_by(ElementRow.page, ElementRow.element_id)
                )
            )
            section_paths = {
                row.section_id: row.section_path
                for row in session.scalars(
                    select(SectionRow).where(SectionRow.version_id == version_id)
                )
            }
        matched_orders = {
            chunk.chunk_order
            for chunk in chunks
            if self._chunk_matches(chunk, request)
        }
        if request.include_neighbors and matched_orders:
            radius = request.neighbor_radius
            matched_orders = {
                chunk.chunk_order
                for chunk in chunks
                if any(abs(chunk.chunk_order - order) <= radius for order in matched_orders)
            }
        passages = tuple(
            ReadPassage(
                chunk_id=chunk.chunk_id,
                section_id=chunk.section_id,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                chunk_order=chunk.chunk_order,
                text=chunk.text,
                source_group_ids=tuple(UUID(value) for value in chunk.source_group_ids_json),
                source_block_ids=tuple(chunk.source_block_ids_json),
                element_ids=tuple(UUID(value) for value in chunk.related_element_ids_json),
            )
            for chunk in chunks
            if chunk.chunk_order in matched_orders
        )
        selected_elements = tuple(
            ReadElement(
                element_id=element.element_id,
                element_type=ElementType(element.element_type),
                section_id=element.section_id,
                section_path=section_paths.get(element.section_id, ""),
                label=element.label,
                caption=element.caption,
                content=element.content,
                page=element.page,
                source_block_ids=tuple(element.source_block_ids_json),
            )
            for element in elements
            if self._element_matches(element, request)
        )
        return ReadPaperResult(
            paper_id=request.paper_id,
            version_id=version_id,
            title=paper.canonical_title or paper.short_name or str(paper.paper_id),
            passages=passages,
            elements=selected_elements,
        )

    @staticmethod
    def _chunk_matches(chunk: ChunkRow, request: ReadPaperRequest) -> bool:
        if request.section_id is not None and chunk.section_id != request.section_id:
            return False
        if request.page_range is not None:
            start, end = request.page_range
            if chunk.page_end < start or chunk.page_start > end:
                return False
        if request.element_id is not None and str(request.element_id) not in chunk.related_element_ids_json:
            return False
        if request.element_types and chunk.chunk_type not in {item.value for item in request.element_types}:
            return False
        return True

    @staticmethod
    def _element_matches(element: ElementRow, request: ReadPaperRequest) -> bool:
        if request.element_id is not None and element.element_id != request.element_id:
            return False
        if request.section_id is not None and element.section_id != request.section_id:
            return False
        if request.page_range is not None and not request.page_range[0] <= element.page <= request.page_range[1]:
            return False
        if request.element_types and ElementType(element.element_type) not in request.element_types:
            return False
        return True