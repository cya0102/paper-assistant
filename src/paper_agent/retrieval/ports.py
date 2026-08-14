"""Search repository and reranker boundaries."""

from typing import Protocol
from uuid import UUID

from paper_agent.domain.indexing import EmbeddingDescriptor
from paper_agent.domain.retrieval import (
    MetadataFilter,
    RankedTarget,
    ResolvedPaper,
    RetrievalCandidate,
    SearchScope,
)


class SearchRepository(Protocol):
    def resolve_papers(
        self,
        query: str,
        scope: SearchScope,
        filters: MetadataFilter,
        limit: int,
    ) -> tuple[ResolvedPaper, ...]: ...

    def dense_papers(
        self,
        vector: tuple[float, ...],
        descriptor: EmbeddingDescriptor,
        scope: SearchScope,
        filters: MetadataFilter,
        limit: int,
    ) -> tuple[ResolvedPaper, ...]: ...

    def dense_sections(
        self,
        vector: tuple[float, ...],
        descriptor: EmbeddingDescriptor,
        scope: SearchScope,
        filters: MetadataFilter,
        paper_ids: tuple[UUID, ...],
        limit: int,
    ) -> tuple[RankedTarget, ...]: ...

    def dense_chunks(
        self,
        vector: tuple[float, ...],
        descriptor: EmbeddingDescriptor,
        scope: SearchScope,
        filters: MetadataFilter,
        paper_ids: tuple[UUID, ...],
        section_ids: tuple[UUID, ...],
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]: ...

    def sparse_chunks(
        self,
        query: str,
        scope: SearchScope,
        filters: MetadataFilter,
        paper_ids: tuple[UUID, ...],
        section_ids: tuple[UUID, ...],
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]: ...


class Reranker(Protocol):
    version: str

    def rerank(
        self, query: str, candidates: tuple[RetrievalCandidate, ...]
    ) -> tuple[RetrievalCandidate, ...]: ...
