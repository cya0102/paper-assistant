"""Paper → Section → Chunk hybrid retrieval with explicit no_evidence."""

from dataclasses import dataclass
from hashlib import sha256
from math import ceil
from uuid import NAMESPACE_URL, UUID, uuid5

from paper_agent.domain.enums import SearchStatus
from paper_agent.domain.retrieval import (
    Evidence,
    RetrievalCandidate,
    SearchKnowledgeResult,
    SearchRequest,
)
from paper_agent.indexing.ports import EmbeddingProvider
from paper_agent.retrieval.ports import Reranker, SearchRepository
from paper_agent.retrieval.reranker import search_tokens


@dataclass(frozen=True, slots=True)
class SearchConfig:
    paper_candidates: int = 5
    section_candidates: int = 8
    dense_candidates: int = 30
    sparse_candidates: int = 30
    min_relevance: float = 0.22
    duplicate_similarity: float = 0.85


class SearchKnowledgeService:
    def __init__(
        self,
        repository: SearchRepository,
        provider: EmbeddingProvider,
        reranker: Reranker,
        config: SearchConfig | None = None,
    ) -> None:
        self._repository = repository
        self._provider = provider
        self._reranker = reranker
        self._config = config or SearchConfig()

    def search_knowledge(self, request: SearchRequest) -> SearchKnowledgeResult:
        query_vector = self._provider.embed_batch((request.query,))[0]
        resolved = self._repository.resolve_papers(
            request.query,
            request.scope,
            request.filters,
            self._config.paper_candidates,
        )
        if not resolved:
            resolved = self._repository.dense_papers(
                query_vector,
                self._provider.descriptor,
                request.scope,
                request.filters,
                self._config.paper_candidates,
            )
        paper_ids = tuple(item.paper_id for item in resolved)
        sections = self._repository.dense_sections(
            query_vector,
            self._provider.descriptor,
            request.scope,
            request.filters,
            paper_ids,
            self._config.section_candidates,
        )
        section_ids = tuple(item.target_id for item in sections)
        dense = self._repository.dense_chunks(
            query_vector,
            self._provider.descriptor,
            request.scope,
            request.filters,
            paper_ids,
            section_ids,
            self._config.dense_candidates,
        )
        sparse = self._repository.sparse_chunks(
            request.query,
            request.scope,
            request.filters,
            paper_ids,
            section_ids,
            self._config.sparse_candidates,
        )
        fused = self._fuse(dense, sparse)
        reranked = self._reranker.rerank(request.query, fused)
        threshold = (
            self._config.min_relevance
            if request.min_relevance is None
            else request.min_relevance
        )
        relevant = tuple(item for item in reranked if item.relevance >= threshold)
        selected = self._select_diverse(relevant, request.max_evidence)
        if not selected:
            return SearchKnowledgeResult(
                query=request.query,
                status=SearchStatus.NO_EVIDENCE,
                resolved_papers=resolved,
                evidence=(),
                has_sufficient_evidence=False,
                reason="No candidate passed the calibrated relevance threshold.",
            )
        evidence = tuple(self._to_evidence(request.query, item) for item in selected)
        return SearchKnowledgeResult(
            query=request.query,
            status=SearchStatus.OK,
            resolved_papers=resolved,
            evidence=evidence,
            has_sufficient_evidence=True,
        )

    @staticmethod
    def _fuse(
        dense: tuple[RetrievalCandidate, ...],
        sparse: tuple[RetrievalCandidate, ...],
    ) -> tuple[RetrievalCandidate, ...]:
        by_id: dict[UUID, RetrievalCandidate] = {}
        for candidate in dense:
            by_id[candidate.chunk_id] = candidate
        for candidate in sparse:
            existing = by_id.get(candidate.chunk_id)
            if existing is None:
                by_id[candidate.chunk_id] = candidate
            else:
                by_id[candidate.chunk_id] = existing.with_scores(
                    bm25_score=candidate.bm25_score
                )
        return tuple(by_id.values())

    def _select_diverse(
        self,
        candidates: tuple[RetrievalCandidate, ...],
        limit: int,
    ) -> tuple[RetrievalCandidate, ...]:
        if not candidates:
            return ()
        paper_count = len({candidate.paper_id for candidate in candidates})
        per_paper = max(2, ceil(limit / max(1, paper_count)))
        counts: dict[UUID, int] = {}
        selected: list[RetrievalCandidate] = []
        for candidate in candidates:
            if counts.get(candidate.paper_id, 0) >= per_paper:
                continue
            if any(self._near_duplicate(candidate.text, item.text) for item in selected):
                continue
            selected.append(candidate)
            counts[candidate.paper_id] = counts.get(candidate.paper_id, 0) + 1
            if len(selected) >= limit:
                break
        return tuple(selected)

    def _near_duplicate(self, left: str, right: str) -> bool:
        left_tokens, right_tokens = set(search_tokens(left)), set(search_tokens(right))
        union = left_tokens | right_tokens
        if not union:
            return True
        return len(left_tokens & right_tokens) / len(union) >= self._config.duplicate_similarity

    @staticmethod
    def _to_evidence(query: str, candidate: RetrievalCandidate) -> Evidence:
        query_hash = sha256(query.casefold().strip().encode("utf-8")).hexdigest()
        return Evidence(
            evidence_id=uuid5(NAMESPACE_URL, f"evidence:{query_hash}:{candidate.chunk_id}"),
            chunk_id=candidate.chunk_id,
            paper_id=candidate.paper_id,
            version_id=candidate.version_id,
            paper_title=candidate.paper_title,
            section_id=candidate.section_id,
            section_path=candidate.section_path,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            element_ids=candidate.element_ids,
            text=candidate.text,
            relevance=candidate.relevance,
            dense_score=candidate.dense_score,
            bm25_score=candidate.bm25_score,
            rerank_score=candidate.rerank_score,
        )
