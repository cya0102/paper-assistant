"""PostgreSQL source loading and atomic hierarchical index replacement."""

from hashlib import sha256
from uuid import UUID, uuid5

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.domain.enums import FileStatus, IndexLevel, IndexingStatus, PipelineStage
from paper_agent.domain.indexing import (
    EmbeddingDescriptor,
    IndexDocument,
    IndexedVector,
    IndexingState,
)
from paper_agent.storage.postgres.models import (
    EMBEDDING_DIMENSION,
    ChunkEmbeddingRow,
    ChunkRow,
    EmbeddingConfigRow,
    IndexingStateRow,
    PaperEmbeddingRow,
    PaperFileRow,
    PaperRow,
    PaperVersionRow,
    SectionEmbeddingRow,
    SectionRow,
)


class SqlAlchemyIndexRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def load_documents(self, project_id: UUID, version_id: UUID) -> tuple[IndexDocument, ...]:
        with self._session_factory() as session:
            version = session.get(PaperVersionRow, version_id)
            if version is None or not session.scalar(
                select(PaperFileRow.file_id).where(
                    PaperFileRow.project_id == project_id,
                    PaperFileRow.version_id == version_id,
                )
            ):
                return ()
            paper = session.get(PaperRow, version.paper_id)
            if paper is None:
                return ()
            sections = tuple(
                session.scalars(
                    select(SectionRow)
                    .where(SectionRow.version_id == version_id)
                    .order_by(SectionRow.section_order)
                )
            )
            chunks = tuple(
                session.scalars(
                    select(ChunkRow)
                    .where(ChunkRow.version_id == version_id)
                    .order_by(ChunkRow.chunk_order)
                )
            )
            if not chunks:
                return ()
            chunks_by_section: dict[UUID, list[ChunkRow]] = {}
            for chunk in chunks:
                chunks_by_section.setdefault(chunk.section_id, []).append(chunk)
            paper_text = "\n".join(
                value
                for value in (
                    paper.canonical_title,
                    " ".join(paper.aliases_json),
                    " ".join(paper.authors_json),
                    paper.venue,
                    paper.abstract,
                    "\n".join(section.section_path for section in sections),
                    "\n".join(chunk.text for chunk in chunks)[:8000],
                )
                if value
            )
            documents: list[IndexDocument] = [
                self._document(
                    target_id=paper.paper_id,
                    level=IndexLevel.PAPER,
                    project_id=project_id,
                    paper_id=paper.paper_id,
                    version_id=version_id,
                    section_id=None,
                    text=paper_text,
                )
            ]
            for section in sections:
                section_text = "\n".join(
                    (
                        section.section_path,
                        *(chunk.text for chunk in chunks_by_section.get(section.section_id, [])),
                    )
                )[:8000]
                documents.append(
                    self._document(
                        target_id=section.section_id,
                        level=IndexLevel.SECTION,
                        project_id=project_id,
                        paper_id=paper.paper_id,
                        version_id=version_id,
                        section_id=section.section_id,
                        text=section_text,
                    )
                )
            for chunk in chunks:
                documents.append(
                    self._document(
                        target_id=chunk.chunk_id,
                        level=IndexLevel.CHUNK,
                        project_id=project_id,
                        paper_id=paper.paper_id,
                        version_id=version_id,
                        section_id=chunk.section_id,
                        text=f"{chunk.section_path}\n{chunk.text}",
                    )
                )
            return tuple(documents)

    def get_state(self, project_id: UUID, version_id: UUID) -> IndexingState | None:
        with self._session_factory() as session:
            row = session.get(IndexingStateRow, (project_id, version_id))
            if row is None:
                return None
            return IndexingState(
                project_id=row.project_id,
                version_id=row.version_id,
                embedding_version=row.embedding_version,
                index_version=row.index_version,
                source_digest=row.source_digest,
                status=IndexingStatus(row.status),
            )

    def load_reusable_vectors(
        self,
        project_id: UUID,
        version_id: UUID,
        embedding_version: str,
    ) -> dict[str, tuple[float, ...]]:
        with self._session_factory() as session:
            reusable: dict[str, tuple[float, ...]] = {}
            for model in (PaperEmbeddingRow, SectionEmbeddingRow, ChunkEmbeddingRow):
                rows = session.execute(
                    select(model.content_hash, model.embedding).where(
                        model.project_id == project_id,
                        model.version_id == version_id,
                        model.embedding_version == embedding_version,
                    )
                )
                reusable.update((content_hash, embedding) for content_hash, embedding in rows)
            return reusable

    def replace_index(
        self,
        project_id: UUID,
        version_id: UUID,
        descriptor: EmbeddingDescriptor,
        index_version: str,
        source_digest: str,
        vectors: tuple[IndexedVector, ...],
    ) -> None:
        if descriptor.dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"PostgreSQL vector schema expects {EMBEDDING_DIMENSION} dimensions, "
                f"got {descriptor.dimension}"
            )
        with self._session_factory.begin() as session:
            config = session.get(EmbeddingConfigRow, descriptor.identifier)
            if config is None:
                session.add(
                    EmbeddingConfigRow(
                        embedding_version=descriptor.identifier,
                        provider=descriptor.provider,
                        model=descriptor.model,
                        provider_version=descriptor.version,
                        dimension=descriptor.dimension,
                    )
                )
            for model in (ChunkEmbeddingRow, SectionEmbeddingRow, PaperEmbeddingRow):
                session.execute(
                    delete(model).where(
                        model.project_id == project_id,
                        model.version_id == version_id,
                    )
                )
            paper_id = vectors[0].document.paper_id
            for indexed in vectors:
                document = indexed.document
                common = dict(
                    embedding_id=uuid5(
                        version_id,
                        f"embedding:{project_id}:{document.level.value}:{document.target_id}",
                    ),
                    project_id=project_id,
                    paper_id=document.paper_id,
                    version_id=version_id,
                    embedding_version=descriptor.identifier,
                    content_hash=document.content_hash,
                    embedding=indexed.embedding,
                )
                if document.level == IndexLevel.PAPER:
                    session.add(PaperEmbeddingRow(**common))
                elif document.level == IndexLevel.SECTION:
                    session.add(
                        SectionEmbeddingRow(section_id=document.target_id, **common)
                    )
                else:
                    if document.section_id is None:
                        raise ValueError("Chunk index document requires section_id")
                    session.add(
                        ChunkEmbeddingRow(
                            section_id=document.section_id,
                            chunk_id=document.target_id,
                            **common,
                        )
                    )
            state = session.get(IndexingStateRow, (project_id, version_id))
            if state is None:
                session.add(
                    IndexingStateRow(
                        project_id=project_id,
                        version_id=version_id,
                        paper_id=paper_id,
                        embedding_version=descriptor.identifier,
                        index_version=index_version,
                        source_digest=source_digest,
                        status=IndexingStatus.INDEXED.value,
                    )
                )
            else:
                state.embedding_version = descriptor.identifier
                state.index_version = index_version
                state.source_digest = source_digest
                state.status = IndexingStatus.INDEXED.value
            version = session.get(PaperVersionRow, version_id)
            if version is not None:
                version.pipeline_status = PipelineStage.INDEXED.value
            for file_row in session.scalars(
                select(PaperFileRow).where(
                    PaperFileRow.project_id == project_id,
                    PaperFileRow.version_id == version_id,
                )
            ):
                file_row.status = FileStatus.INDEXED.value

    @staticmethod
    def _document(
        *,
        target_id: UUID,
        level: IndexLevel,
        project_id: UUID,
        paper_id: UUID,
        version_id: UUID,
        section_id: UUID | None,
        text: str,
    ) -> IndexDocument:
        normalized = text.strip()
        content_hash = sha256(
            f"{level.value}\n{normalized}".encode("utf-8")
        ).hexdigest()
        return IndexDocument(
            target_id=target_id,
            level=level,
            project_id=project_id,
            paper_id=paper_id,
            version_id=version_id,
            section_id=section_id,
            text=normalized,
            content_hash=content_hash,
        )
