"""Query rewrite, multi-query fusion, neighbor expansion, and evidence judging."""

from dataclasses import replace
import re
from typing import Protocol
from uuid import UUID

from paper_agent.domain.enums import SearchStatus
from paper_agent.domain.retrieval import Evidence, ResolvedPaper, SearchKnowledgeResult, SearchRequest
from paper_agent.retrieval.reranker import search_tokens


class KnowledgeSearch(Protocol):
    def search_knowledge(self, request: SearchRequest) -> SearchKnowledgeResult: ...


class QueryRewriter(Protocol):
    version: str

    def rewrite(self, query: str, *, max_queries: int) -> tuple[str, ...]: ...


class EvidenceJudge(Protocol):
    version: str

    def score(self, query: str, evidence: Evidence) -> float: ...


class NeighborRepository(Protocol):
    def expand(self, evidence: tuple[Evidence, ...], radius: int) -> tuple[Evidence, ...]: ...


class ConservativeQueryRewriter:
    version = "conservative-query-rewriter-v1"

    def rewrite(self, query: str, *, max_queries: int) -> tuple[str, ...]:
        normalized = " ".join(query.split())
        alternatives = [normalized]
        without_question = normalized.rstrip("?？。")
        if without_question != normalized:
            alternatives.append(without_question)
        focused = without_question
        for phrase in ("请问", "请解释", "为什么", "如何", "怎么", "是什么", "是否", "的", "是"):
            focused = focused.replace(phrase, " ")
        focused = re.sub(r"[^\w\u4e00-\u9fff-]+", " ", focused)
        focused = " ".join(focused.split())
        if focused and focused != without_question:
            alternatives.append(focused)
        return tuple(dict.fromkeys(alternatives))[:max_queries]


class LexicalEvidenceJudge:
    version = "lexical-evidence-judge-v1"

    def score(self, query: str, evidence: Evidence) -> float:
        query_tokens = set(search_tokens(query))
        if not query_tokens:
            return evidence.relevance
        overlap = len(query_tokens & set(search_tokens(evidence.text))) / len(query_tokens)
        return min(1.0, evidence.relevance * 0.7 + overlap * 0.3)


class AdvancedSearchKnowledgeService:
    def __init__(
        self,
        search: KnowledgeSearch,
        rewriter: QueryRewriter,
        judge: EvidenceJudge,
        neighbors: NeighborRepository | None = None,
        *,
        max_queries: int = 4,
        neighbor_radius: int = 1,
        judge_threshold: float = 0.2,
    ) -> None:
        self._search = search
        self._rewriter = rewriter
        self._judge = judge
        self._neighbors = neighbors
        self._max_queries = max_queries
        self._neighbor_radius = neighbor_radius
        self._judge_threshold = judge_threshold

    def search_knowledge(self, request: SearchRequest) -> SearchKnowledgeResult:
        queries = self._rewriter.rewrite(request.query, max_queries=self._max_queries)
        results = tuple(
            self._search.search_knowledge(replace(request, query=query)) for query in queries
        )
        resolved = self._merge_resolved(results)
        by_chunk: dict[UUID, Evidence] = {}
        for result in results:
            for item in result.evidence:
                previous = by_chunk.get(item.chunk_id)
                if previous is None or item.relevance > previous.relevance:
                    by_chunk[item.chunk_id] = item
        candidates = tuple(by_chunk.values())
        if self._neighbors is not None and candidates:
            candidates = self._neighbors.expand(candidates, self._neighbor_radius)
        judged = tuple(
            replace(item, relevance=score)
            for item in candidates
            if (score := self._judge.score(request.query, item)) >= self._judge_threshold
        )
        selected = tuple(sorted(judged, key=lambda item: item.relevance, reverse=True)[: request.max_evidence])
        if not selected:
            return SearchKnowledgeResult(
                query=request.query,
                status=SearchStatus.NO_EVIDENCE,
                resolved_papers=resolved,
                evidence=(),
                has_sufficient_evidence=False,
                reason="No rewritten-query evidence passed the evidence judge.",
            )
        return SearchKnowledgeResult(
            query=request.query,
            status=SearchStatus.OK,
            resolved_papers=resolved,
            evidence=selected,
            has_sufficient_evidence=True,
        )

    @staticmethod
    def _merge_resolved(results: tuple[SearchKnowledgeResult, ...]) -> tuple[ResolvedPaper, ...]:
        papers: dict[UUID, ResolvedPaper] = {}
        for result in results:
            for item in result.resolved_papers:
                previous = papers.get(item.paper_id)
                if previous is None or item.score > previous.score:
                    papers[item.paper_id] = item
        return tuple(sorted(papers.values(), key=lambda item: item.score, reverse=True))
