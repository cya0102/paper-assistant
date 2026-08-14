"""Database migration and status helpers used by the CLI."""

from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from paper_agent.storage.postgres.models import (
    ChunkRow,
    ChunkEmbeddingRow,
    IndexingStateRow,
    ElementRow,
    PaperFileRow,
    PaperRow,
    PaperEmbeddingRow,
    PaperVersionRow,
    SectionRow,
    SectionEmbeddingRow,
    SemanticGroupRow,
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


def database_status(database_url: str) -> dict[str, int]:
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with Session(engine) as session:
            counts = {
                "papers": session.scalar(select(func.count()).select_from(PaperRow)) or 0,
                "versions": session.scalar(select(func.count()).select_from(PaperVersionRow)) or 0,
                "files": session.scalar(select(func.count()).select_from(PaperFileRow)) or 0,
                "sections": session.scalar(select(func.count()).select_from(SectionRow)) or 0,
                "elements": session.scalar(select(func.count()).select_from(ElementRow)) or 0,
                "semantic_groups": session.scalar(
                    select(func.count()).select_from(SemanticGroupRow)
                )
                or 0,
                "chunks": session.scalar(select(func.count()).select_from(ChunkRow)) or 0,
                "paper_vectors": session.scalar(
                    select(func.count()).select_from(PaperEmbeddingRow)
                )
                or 0,
                "section_vectors": session.scalar(
                    select(func.count()).select_from(SectionEmbeddingRow)
                )
                or 0,
                "chunk_vectors": session.scalar(
                    select(func.count()).select_from(ChunkEmbeddingRow)
                )
                or 0,
                "indexed_versions": session.scalar(
                    select(func.count()).select_from(IndexingStateRow)
                )
                or 0,
            }
            for status, count in session.execute(
                select(PaperFileRow.status, func.count()).group_by(PaperFileRow.status)
            ):
                counts[f"files_{status}"] = count
            return counts
    finally:
        engine.dispose()
