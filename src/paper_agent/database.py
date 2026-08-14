"""Database migration and status helpers used by the CLI."""

from pathlib import Path
from uuid import UUID

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from paper_agent.storage.postgres.models import (
    ChunkRow,
    ChunkEmbeddingRow,
    IndexingStateRow,
    ElementRow,
    PaperFileRow,
    PaperEmbeddingRow,
    SectionRow,
    SectionEmbeddingRow,
    SemanticGroupRow,
    ClaimRow,
    PaperProfileRow,
    PaperRelationRow,
    ResearchEntityRow,
)


def upgrade_database(database_url: str) -> None:
    try:
        from alembic import command
        from alembic.config import Config
    except ImportError as error:
        raise RuntimeError("Alembic is not installed; run `uv sync --extra dev`") from error
    project_root = Path(__file__).resolve().parents[2]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


def database_status(database_url: str, project_id: UUID) -> dict[str, int]:
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with Session(engine) as session:
            counts = {
                "papers": session.scalar(
                    select(func.count(func.distinct(PaperFileRow.paper_id))).where(
                        PaperFileRow.project_id == project_id,
                        PaperFileRow.paper_id.is_not(None),
                    )
                )
                or 0,
                "versions": session.scalar(
                    select(func.count(func.distinct(PaperFileRow.version_id))).where(
                        PaperFileRow.project_id == project_id,
                        PaperFileRow.version_id.is_not(None),
                    )
                )
                or 0,
                "files": session.scalar(
                    select(func.count())
                    .select_from(PaperFileRow)
                    .where(PaperFileRow.project_id == project_id)
                )
                or 0,
                "sections": session.scalar(
                    select(func.count(func.distinct(SectionRow.section_id)))
                    .join(
                        PaperFileRow,
                        PaperFileRow.version_id == SectionRow.version_id,
                    )
                    .where(PaperFileRow.project_id == project_id)
                )
                or 0,
                "elements": session.scalar(
                    select(func.count(func.distinct(ElementRow.element_id)))
                    .join(
                        PaperFileRow,
                        PaperFileRow.version_id == ElementRow.version_id,
                    )
                    .where(PaperFileRow.project_id == project_id)
                )
                or 0,
                "semantic_groups": session.scalar(
                    select(func.count(func.distinct(SemanticGroupRow.group_id)))
                    .join(
                        PaperFileRow,
                        PaperFileRow.version_id == SemanticGroupRow.version_id,
                    )
                    .where(PaperFileRow.project_id == project_id)
                )
                or 0,
                "chunks": session.scalar(
                    select(func.count(func.distinct(ChunkRow.chunk_id)))
                    .join(
                        PaperFileRow,
                        PaperFileRow.version_id == ChunkRow.version_id,
                    )
                    .where(PaperFileRow.project_id == project_id)
                )
                or 0,
                "paper_vectors": session.scalar(
                    select(func.count())
                    .select_from(PaperEmbeddingRow)
                    .where(PaperEmbeddingRow.project_id == project_id)
                )
                or 0,
                "section_vectors": session.scalar(
                    select(func.count())
                    .select_from(SectionEmbeddingRow)
                    .where(SectionEmbeddingRow.project_id == project_id)
                )
                or 0,
                "chunk_vectors": session.scalar(
                    select(func.count())
                    .select_from(ChunkEmbeddingRow)
                    .where(ChunkEmbeddingRow.project_id == project_id)
                )
                or 0,
                "indexed_versions": session.scalar(
                    select(func.count())
                    .select_from(IndexingStateRow)
                    .where(IndexingStateRow.project_id == project_id)
                )
                or 0,
                "paper_profiles": session.scalar(
                    select(func.count())
                    .select_from(PaperProfileRow)
                    .where(
                        PaperProfileRow.project_id == project_id,
                        PaperProfileRow.is_active.is_(True),
                    )
                )
                or 0,
                "claims": session.scalar(
                    select(func.count())
                    .select_from(ClaimRow)
                    .where(
                        ClaimRow.project_id == project_id,
                        ClaimRow.is_active.is_(True),
                    )
                )
                or 0,
                "research_entities": session.scalar(
                    select(func.count())
                    .select_from(ResearchEntityRow)
                    .where(ResearchEntityRow.project_id == project_id)
                )
                or 0,
                "paper_relations": session.scalar(
                    select(func.count())
                    .select_from(PaperRelationRow)
                    .where(
                        PaperRelationRow.project_id == project_id,
                        PaperRelationRow.is_active.is_(True),
                    )
                )
                or 0,
            }
            for status, count in session.execute(
                select(PaperFileRow.status, func.count())
                .where(PaperFileRow.project_id == project_id)
                .group_by(PaperFileRow.status)
            ):
                counts[f"files_{status}"] = count
            return counts
    finally:
        engine.dispose()
