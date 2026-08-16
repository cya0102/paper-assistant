"""Replaceable boundaries used by the ROD application service."""

from typing import Protocol

from paper_agent.domain.retrieval import SearchKnowledgeResult, SearchRequest
from paper_agent.rag.domain import AnalystReport, RagTraceEvent


class RagKnowledgeSearch(Protocol):
    def search_knowledge(self, request: SearchRequest) -> SearchKnowledgeResult: ...


class RagQueryRewriter(Protocol):
    version: str

    def rewrite(self, query: str, reports: tuple[AnalystReport, ...]) -> str: ...


class RagTracer(Protocol):
    def emit(self, event: RagTraceEvent) -> None: ...


class NullRagTracer:
    def emit(self, event: RagTraceEvent) -> None:
        del event
