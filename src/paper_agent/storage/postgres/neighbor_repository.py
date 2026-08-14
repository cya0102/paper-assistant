"""Chunk-order neighbor expansion with full Evidence provenance."""

from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.domain.retrieval import Evidence
from paper_agent.storage.postgres.models import ChunkRow, PaperRow


class SqlAlchemyNeighborRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def expand(self, evidence: tuple[Evidence, ...], radius: int) -> tuple[Evidence, ...]:
        by_id = {item.chunk_id: item for item in evidence}
        with self._session_factory() as session:
            for seed in evidence:
                rows = tuple(
                    session.scalars(
                        select(ChunkRow)
                        .where(
                            ChunkRow.version_id == seed.version_id,
                            ChunkRow.section_id == seed.section_id,
                            ChunkRow.chunk_order.between(
                                self._chunk_order(session, seed.chunk_id) - radius,
                                self._chunk_order(session, seed.chunk_id) + radius,
                            ),
                        )
                        .order_by(ChunkRow.chunk_order)
                    )
                )
                paper = session.get(PaperRow, seed.paper_id)
                for row in rows:
                    if row.chunk_id in by_id:
                        continue
                    by_id[row.chunk_id] = Evidence(
                        evidence_id=uuid5(NAMESPACE_URL, f"neighbor:{seed.evidence_id}:{row.chunk_id}"),
                        chunk_id=row.chunk_id,
                        paper_id=row.paper_id,
                        version_id=row.version_id,
                        paper_title=(paper.canonical_title if paper else None) or seed.paper_title,
                        section_id=row.section_id,
                        section_path=row.section_path,
                        page_start=row.page_start,
                        page_end=row.page_end,
                        element_ids=tuple(UUID(value) for value in row.related_element_ids_json),
                        text=row.text,
                        relevance=seed.relevance * 0.8,
                        dense_score=None,
                        bm25_score=None,
                        rerank_score=None,
                    )
        return tuple(by_id.values())

    @staticmethod
    def _chunk_order(session: Session, chunk_id: UUID) -> int:
        row = session.get(ChunkRow, chunk_id)
        if row is None:
            raise LookupError(f"Chunk not found: {chunk_id}")
        return row.chunk_order
