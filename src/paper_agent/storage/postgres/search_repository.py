"""Scoped pgvector, PostgreSQL FTS, and BM25 retrieval implementation."""

from collections import Counter
from math import log
from typing import Any
from uuid import UUID

from sqlalchemy import Float, Text, cast, exists, func, select
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.domain.indexing import EmbeddingDescriptor
from paper_agent.domain.retrieval import (
    MetadataFilter,
    RankedTarget,
    ResolvedPaper,
    RetrievalCandidate,
    SearchScope,
)
from paper_agent.retrieval.reranker import search_tokens
from paper_agent.storage.postgres.models import (
    ChunkEmbeddingRow,
    ChunkRow,
    PaperEmbeddingRow,
    PaperFileRow,
    PaperRow,
    SectionEmbeddingRow,
    SectionRow,
)


class SqlAlchemySearchRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def resolve_papers(
        self,
        query: str,
        scope: SearchScope,
        filters: MetadataFilter,
        limit: int,
    ) -> tuple[ResolvedPaper, ...]:
        with self._session_factory() as session:
            statement = (
                select(PaperRow, PaperEmbeddingRow.version_id)
                .join(PaperEmbeddingRow, PaperEmbeddingRow.paper_id == PaperRow.paper_id)
                .where(PaperEmbeddingRow.project_id == scope.project_id)
            )
            statement = statement.where(*self._paper_clauses(PaperRow, PaperEmbeddingRow, scope, filters))
            rows = session.execute(statement).all()
        normalized_query = query.casefold()
        resolved: list[ResolvedPaper] = []
        for paper, version_id in rows:
            title = paper.canonical_title or paper.short_name or str(paper.paper_id)
            names = tuple(
                value
                for value in (
                    paper.canonical_title,
                    paper.short_name,
                    paper.acronym,
                    *paper.aliases_json,
                )
                if value
            )
            best = 0.0
            matched_by = "metadata"
            for name in names:
                normalized_name = name.casefold().strip()
                if normalized_name == normalized_query.strip():
                    best, matched_by = 1.0, "exact_metadata"
                elif normalized_name and normalized_name in normalized_query:
                    score = 0.98 if name == paper.acronym else 0.92
                    if score > best:
                        best, matched_by = score, "query_metadata"
            if scope.paper_ids and paper.paper_id in scope.paper_ids:
                best, matched_by = 1.0, "explicit_scope"
            if best:
                resolved.append(
                    ResolvedPaper(
                        paper_id=paper.paper_id,
                        version_id=version_id,
                        title=title,
                        score=best,
                        matched_by=matched_by,
                    )
                )
        return tuple(sorted(resolved, key=lambda item: item.score, reverse=True)[:limit])

    def dense_papers(
        self,
        vector: tuple[float, ...],
        descriptor: EmbeddingDescriptor,
        scope: SearchScope,
        filters: MetadataFilter,
        limit: int,
    ) -> tuple[ResolvedPaper, ...]:
        with self._session_factory() as session:
            distance = PaperEmbeddingRow.embedding.op("<=>", return_type=Float)(vector)
            statement = (
                select(PaperRow, PaperEmbeddingRow.version_id, distance.label("distance"))
                .join(PaperEmbeddingRow, PaperEmbeddingRow.paper_id == PaperRow.paper_id)
                .where(
                    PaperEmbeddingRow.project_id == scope.project_id,
                    PaperEmbeddingRow.embedding_version == descriptor.identifier,
                    *self._paper_clauses(PaperRow, PaperEmbeddingRow, scope, filters),
                )
                .order_by(distance)
                .limit(limit)
            )
            rows = session.execute(statement)
            return tuple(
                ResolvedPaper(
                    paper_id=paper.paper_id,
                    version_id=version_id,
                    title=paper.canonical_title or paper.short_name or str(paper.paper_id),
                    score=max(0.0, 1.0 - float(distance_value)),
                    matched_by="dense",
                )
                for paper, version_id, distance_value in rows
            )

    def dense_sections(
        self,
        vector: tuple[float, ...],
        descriptor: EmbeddingDescriptor,
        scope: SearchScope,
        filters: MetadataFilter,
        paper_ids: tuple[UUID, ...],
        limit: int,
    ) -> tuple[RankedTarget, ...]:
        with self._session_factory() as session:
            distance = SectionEmbeddingRow.embedding.op("<=>", return_type=Float)(vector)
            statement = (
                select(SectionEmbeddingRow, distance.label("distance"))
                .join(PaperRow, PaperRow.paper_id == SectionEmbeddingRow.paper_id)
                .where(
                    SectionEmbeddingRow.project_id == scope.project_id,
                    SectionEmbeddingRow.embedding_version == descriptor.identifier,
                    *self._paper_clauses(PaperRow, SectionEmbeddingRow, scope, filters),
                )
            )
            if paper_ids:
                statement = statement.where(SectionEmbeddingRow.paper_id.in_(paper_ids))
            if scope.section_ids:
                statement = statement.where(SectionEmbeddingRow.section_id.in_(scope.section_ids))
            rows = session.execute(statement.order_by(distance).limit(limit))
            return tuple(
                RankedTarget(
                    target_id=row.section_id,
                    paper_id=row.paper_id,
                    version_id=row.version_id,
                    score=max(0.0, 1.0 - float(distance_value)),
                )
                for row, distance_value in rows
            )

    def dense_chunks(
        self,
        vector: tuple[float, ...],
        descriptor: EmbeddingDescriptor,
        scope: SearchScope,
        filters: MetadataFilter,
        paper_ids: tuple[UUID, ...],
        section_ids: tuple[UUID, ...],
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]:
        with self._session_factory() as session:
            distance = ChunkEmbeddingRow.embedding.op("<=>", return_type=Float)(vector)
            statement = (
                select(ChunkRow, PaperRow, distance.label("distance"))
                .join(ChunkEmbeddingRow, ChunkEmbeddingRow.chunk_id == ChunkRow.chunk_id)
                .join(PaperRow, PaperRow.paper_id == ChunkRow.paper_id)
                .where(
                    ChunkEmbeddingRow.project_id == scope.project_id,
                    ChunkEmbeddingRow.embedding_version == descriptor.identifier,
                    *self._paper_clauses(PaperRow, ChunkEmbeddingRow, scope, filters),
                    *self._chunk_clauses(ChunkRow, scope, filters),
                )
            )
            statement = self._hierarchy_scope(statement, ChunkRow, paper_ids, section_ids)
            rows = session.execute(statement.order_by(distance).limit(limit))
            return tuple(
                self._candidate(
                    chunk,
                    paper,
                    dense_score=max(0.0, 1.0 - float(distance_value)),
                )
                for chunk, paper, distance_value in rows
            )

    def sparse_chunks(
        self,
        query: str,
        scope: SearchScope,
        filters: MetadataFilter,
        paper_ids: tuple[UUID, ...],
        section_ids: tuple[UUID, ...],
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]:
        tokens = tuple(dict.fromkeys(token for token in search_tokens(query) if token.isalnum()))
        if not tokens:
            return ()
        web_query = " OR ".join(tokens)
        with self._session_factory() as session:
            tsquery = func.websearch_to_tsquery("simple", web_query)
            rank = func.ts_rank_cd(ChunkRow.search_vector, tsquery)
            statement = (
                select(ChunkRow, PaperRow, rank.label("fts_rank"))
                .join(PaperRow, PaperRow.paper_id == ChunkRow.paper_id)
                .where(
                    ChunkRow.search_vector.op("@@")(tsquery),
                    *self._paper_clauses(PaperRow, ChunkRow, scope, filters),
                    *self._chunk_clauses(ChunkRow, scope, filters),
                )
            )
            statement = self._hierarchy_scope(statement, ChunkRow, paper_ids, section_ids)
            rows = session.execute(statement.order_by(rank.desc()).limit(limit * 4)).all()
        candidates = [self._candidate(chunk, paper) for chunk, paper, _ in rows]
        scores = self._bm25(query, tuple(candidate.text for candidate in candidates))
        ranked = [
            candidate.with_scores(bm25_score=score)
            for candidate, score in zip(candidates, scores, strict=True)
            if score > 0
        ]
        return tuple(
            sorted(ranked, key=lambda item: item.bm25_score or 0.0, reverse=True)[:limit]
        )

    @staticmethod
    def _paper_clauses(
        paper: type[PaperRow], owner: Any, scope: SearchScope, filters: MetadataFilter
    ) -> list[Any]:
        clauses: list[Any] = []
        clauses.append(
            exists(
                select(PaperFileRow.file_id).where(
                    PaperFileRow.project_id == scope.project_id,
                    PaperFileRow.paper_id == paper.paper_id,
                    PaperFileRow.version_id == owner.version_id,
                )
            )
        )
        if scope.paper_ids:
            clauses.append(paper.paper_id.in_(scope.paper_ids))
        if scope.version_ids:
            clauses.append(owner.version_id.in_(scope.version_ids))
        if filters.year_from is not None:
            clauses.append(paper.year >= filters.year_from)
        if filters.year_to is not None:
            clauses.append(paper.year <= filters.year_to)
        if filters.venues:
            clauses.append(func.lower(paper.venue).in_(tuple(value.casefold() for value in filters.venues)))
        for author in filters.authors:
            clauses.append(cast(paper.authors_json, Text).ilike(f"%{author}%"))
        return clauses

    @staticmethod
    def _chunk_clauses(
        chunk: type[ChunkRow], scope: SearchScope, filters: MetadataFilter
    ) -> list[Any]:
        clauses: list[Any] = []
        if scope.section_ids:
            clauses.append(chunk.section_id.in_(scope.section_ids))
        if filters.chunk_types:
            clauses.append(chunk.chunk_type.in_(tuple(value.value for value in filters.chunk_types)))
        return clauses

    @staticmethod
    def _hierarchy_scope(statement: Any, chunk: type[ChunkRow], paper_ids: tuple[UUID, ...], section_ids: tuple[UUID, ...]) -> Any:
        if paper_ids:
            statement = statement.where(chunk.paper_id.in_(paper_ids))
        if section_ids:
            statement = statement.where(chunk.section_id.in_(section_ids))
        return statement

    @staticmethod
    def _candidate(
        chunk: ChunkRow,
        paper: PaperRow,
        *,
        dense_score: float | None = None,
    ) -> RetrievalCandidate:
        return RetrievalCandidate(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            version_id=chunk.version_id,
            paper_title=paper.canonical_title or paper.short_name or str(paper.paper_id),
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            element_ids=tuple(UUID(value) for value in chunk.related_element_ids_json),
            text=chunk.text,
            chunk_order=chunk.chunk_order,
            dense_score=dense_score,
        )

    @staticmethod
    def _bm25(query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        if not documents:
            return ()
        query_terms = tuple(dict.fromkeys(search_tokens(query)))
        tokenized = [search_tokens(document) for document in documents]
        average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
        document_frequency = {
            term: sum(term in set(tokens) for tokens in tokenized) for term in query_terms
        }
        k1, b = 1.2, 0.75
        scores: list[float] = []
        for tokens in tokenized:
            frequencies = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1.0 - b + b * length / average_length)
                score += idf * frequency * (k1 + 1.0) / denominator
            scores.append(score)
        return tuple(scores)
