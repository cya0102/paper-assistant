"""Project-scoped PostgreSQL Artifact catalog repository."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.domain.artifact import (
    ArtifactDescriptor,
    ArtifactStatus,
    ArtifactType,
    CitationReference,
)
from paper_agent.storage.postgres.models import (
    ArtifactCitationRow,
    ResearchArtifactRow,
)


def _citation_from_row(row: ArtifactCitationRow) -> CitationReference:
    return CitationReference(
        citation_label=row.citation_label,
        paper_id=row.paper_id,
        version_id=row.version_id,
        paper_title=row.paper_title,
        section_path=row.section_path,
        page_start=row.page_start,
        page_end=row.page_end,
        evidence_hash=row.evidence_hash,
        section_id=row.section_id,
        chunk_id=row.chunk_id,
        element_id=row.element_id,
    )


def _descriptor_from_row(row: ResearchArtifactRow) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=row.artifact_id,
        project_id=row.project_id,
        artifact_type=ArtifactType(row.artifact_type),
        schema_version=row.schema_version,
        media_type=row.media_type,
        content_hash=row.content_hash,
        storage_backend=row.storage_backend,
        storage_key=row.storage_key,
        byte_size=row.byte_size,
        token_estimate=row.token_estimate,
        summary=row.summary,
        citation_manifest=(),
        status=ArtifactStatus(row.status),
        created_by=row.created_by,
        created_at=row.created_at,
        session_id=row.session_id,
        research_task_id=row.research_task_id,
        work_unit_id=row.work_unit_id,
        tool_call_id=row.tool_call_id,
        expires_at=row.expires_at,
    )


def _row_values(descriptor: ArtifactDescriptor) -> dict[str, object]:
    return {
        "artifact_id": descriptor.artifact_id,
        "project_id": descriptor.project_id,
        "session_id": descriptor.session_id,
        "research_task_id": descriptor.research_task_id,
        "work_unit_id": descriptor.work_unit_id,
        "tool_call_id": descriptor.tool_call_id,
        "artifact_type": descriptor.artifact_type.value,
        "schema_version": descriptor.schema_version,
        "media_type": descriptor.media_type,
        "content_hash": descriptor.content_hash,
        "storage_backend": descriptor.storage_backend,
        "storage_key": descriptor.storage_key,
        "byte_size": descriptor.byte_size,
        "token_estimate": descriptor.token_estimate,
        "summary": descriptor.summary,
        "status": descriptor.status.value,
        "created_by": descriptor.created_by,
        "created_at": descriptor.created_at,
        "expires_at": descriptor.expires_at,
    }


class SqlAlchemyArtifactRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(
        self,
        descriptor: ArtifactDescriptor,
        citations: Sequence[CitationReference],
    ) -> ArtifactDescriptor:
        with self._session_factory.begin() as session:
            statement = insert(ResearchArtifactRow).values(**_row_values(descriptor))
            statement = statement.on_conflict_do_nothing(
                index_elements=[
                    "project_id",
                    "artifact_type",
                    "schema_version",
                    "content_hash",
                ]
            )
            session.execute(statement)
            row = session.scalar(
                select(ResearchArtifactRow).where(
                    ResearchArtifactRow.artifact_id == descriptor.artifact_id,
                    ResearchArtifactRow.project_id == descriptor.project_id,
                )
            )
            if row is None:
                row = session.scalar(
                    select(ResearchArtifactRow).where(
                        ResearchArtifactRow.project_id == descriptor.project_id,
                        ResearchArtifactRow.artifact_type == descriptor.artifact_type.value,
                        ResearchArtifactRow.schema_version == descriptor.schema_version,
                        ResearchArtifactRow.content_hash == descriptor.content_hash,
                    )
                )
            if row is None:
                raise LookupError("Artifact insert did not persist")
            if citations:
                self._replace_citations(session, descriptor.project_id, row, citations)
            stored = _descriptor_from_row(row)
            stored_citations = self._list_citations_in_session(session, descriptor.project_id, row.artifact_id)
            return replace(stored, citation_manifest=stored_citations)

    def get(
        self, project_id: UUID, artifact_id: UUID
    ) -> ArtifactDescriptor | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(ResearchArtifactRow).where(
                    ResearchArtifactRow.project_id == project_id,
                    ResearchArtifactRow.artifact_id == artifact_id,
                )
            )
            if row is None:
                return None
            descriptor = _descriptor_from_row(row)
            return replace(
                descriptor,
                citation_manifest=self._list_citations_in_session(
                    session, project_id, artifact_id
                ),
            )

    def find_by_hash(
        self,
        project_id: UUID,
        artifact_type: ArtifactType,
        schema_version: str,
        content_hash: str,
    ) -> ArtifactDescriptor | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(ResearchArtifactRow).where(
                    ResearchArtifactRow.project_id == project_id,
                    ResearchArtifactRow.artifact_type == artifact_type.value,
                    ResearchArtifactRow.schema_version == schema_version,
                    ResearchArtifactRow.content_hash == content_hash,
                )
            )
            if row is None:
                return None
            descriptor = _descriptor_from_row(row)
            return replace(
                descriptor,
                citation_manifest=self._list_citations_in_session(
                    session, project_id, row.artifact_id
                ),
            )

    def save_citations(
        self,
        project_id: UUID,
        artifact_id: UUID,
        citations: Sequence[CitationReference],
    ) -> None:
        with self._session_factory.begin() as session:
            row = session.scalar(
                select(ResearchArtifactRow).where(
                    ResearchArtifactRow.project_id == project_id,
                    ResearchArtifactRow.artifact_id == artifact_id,
                )
            )
            if row is None:
                raise LookupError("Artifact not found in project")
            self._replace_citations(session, project_id, row, citations)

    def list_citations(
        self, project_id: UUID, artifact_id: UUID
    ) -> tuple[CitationReference, ...]:
        with self._session_factory() as session:
            return self._list_citations_in_session(session, project_id, artifact_id)

    def mark_expired(self, *, now: datetime) -> int:
        with self._session_factory.begin() as session:
            result = session.execute(
                update(ResearchArtifactRow)
                .where(
                    ResearchArtifactRow.expires_at.is_not(None),
                    ResearchArtifactRow.expires_at < now,
                    ResearchArtifactRow.status == ArtifactStatus.ACTIVE.value,
                )
                .values(status=ArtifactStatus.EXPIRED.value)
            )
            from typing import cast

            from sqlalchemy.engine import CursorResult

            return int(cast("CursorResult[Any]", result).rowcount or 0)

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
    ) -> tuple[ArtifactDescriptor, ...]:
        with self._session_factory() as session:
            statement = select(ResearchArtifactRow).where(
                ResearchArtifactRow.project_id == project_id
            )
            if artifact_type is not None:
                statement = statement.where(
                    ResearchArtifactRow.artifact_type == artifact_type.value
                )
            if created_by is not None:
                statement = statement.where(
                    ResearchArtifactRow.created_by == created_by
                )
            if research_task_id is not None:
                statement = statement.where(
                    ResearchArtifactRow.research_task_id == research_task_id
                )
            if work_unit_id is not None:
                statement = statement.where(
                    ResearchArtifactRow.work_unit_id == work_unit_id
                )
            if query:
                statement = statement.where(
                    ResearchArtifactRow.summary.ilike(f"%{query}%")
                )
            rows = tuple(
                session.scalars(
                    statement.order_by(ResearchArtifactRow.created_at.desc()).limit(limit)
                )
            )
            descriptors: list[ArtifactDescriptor] = []
            for row in rows:
                descriptor = _descriptor_from_row(row)
                descriptors.append(
                    replace(
                        descriptor,
                        citation_manifest=self._list_citations_in_session(
                            session, project_id, row.artifact_id
                        ),
                    )
                )
            return tuple(descriptors)

    @staticmethod
    def _replace_citations(
        session: Session,
        project_id: UUID,
        row: ResearchArtifactRow,
        citations: Sequence[CitationReference],
    ) -> None:
        session.execute(
            delete(ArtifactCitationRow).where(
                ArtifactCitationRow.artifact_id == row.artifact_id,
                ArtifactCitationRow.project_id == project_id,
            )
        )
        for citation in citations:
            session.add(
                ArtifactCitationRow(
                    artifact_id=row.artifact_id,
                    project_id=project_id,
                    citation_label=citation.citation_label,
                    paper_id=citation.paper_id,
                    version_id=citation.version_id,
                    paper_title=citation.paper_title,
                    section_path=citation.section_path,
                    section_id=citation.section_id,
                    chunk_id=citation.chunk_id,
                    element_id=citation.element_id,
                    page_start=citation.page_start,
                    page_end=citation.page_end,
                    evidence_hash=citation.evidence_hash,
                )
            )

    @staticmethod
    def _list_citations_in_session(
        session: Session, project_id: UUID, artifact_id: UUID
    ) -> tuple[CitationReference, ...]:
        rows = session.scalars(
            select(ArtifactCitationRow)
            .where(
                ArtifactCitationRow.artifact_id == artifact_id,
                ArtifactCitationRow.project_id == project_id,
            )
            .order_by(ArtifactCitationRow.citation_label)
        )
        return tuple(_citation_from_row(row) for row in rows)