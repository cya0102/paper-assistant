"""Search scope, ranked candidates, and traceable Evidence."""

from dataclasses import dataclass, replace
from uuid import UUID

from paper_agent.domain.enums import ChunkType, SearchStatus


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    year_from: int | None = None
    year_to: int | None = None
    venues: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    chunk_types: tuple[ChunkType, ...] = ()

    def __post_init__(self) -> None:
        if self.year_from is not None and not 1000 <= self.year_from <= 9999:
            raise ValueError("year_from must be a four-digit year")
        if self.year_to is not None and not 1000 <= self.year_to <= 9999:
            raise ValueError("year_to must be a four-digit year")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from cannot exceed year_to")


@dataclass(frozen=True, slots=True)
class SearchScope:
    project_id: UUID
    paper_ids: tuple[UUID, ...] = ()
    version_ids: tuple[UUID, ...] = ()
    section_ids: tuple[UUID, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    scope: SearchScope
    filters: MetadataFilter = MetadataFilter()
    max_evidence: int = 5
    min_relevance: float | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Search query cannot be blank")
        if not 1 <= self.max_evidence <= 50:
            raise ValueError("max_evidence must be between 1 and 50")
        if self.min_relevance is not None and not 0 <= self.min_relevance <= 1:
            raise ValueError("min_relevance must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ResolvedPaper:
    paper_id: UUID
    version_id: UUID
    title: str
    score: float
    matched_by: str


@dataclass(frozen=True, slots=True)
class RankedTarget:
    target_id: UUID
    paper_id: UUID
    version_id: UUID
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    chunk_id: UUID
    paper_id: UUID
    version_id: UUID
    paper_title: str
    section_id: UUID
    section_path: str
    page_start: int
    page_end: int
    element_ids: tuple[UUID, ...]
    text: str
    chunk_order: int
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    relevance: float = 0.0

    def with_scores(
        self,
        *,
        dense_score: float | None = None,
        bm25_score: float | None = None,
        rerank_score: float | None = None,
        relevance: float | None = None,
    ) -> "RetrievalCandidate":
        return replace(
            self,
            dense_score=self.dense_score if dense_score is None else dense_score,
            bm25_score=self.bm25_score if bm25_score is None else bm25_score,
            rerank_score=self.rerank_score if rerank_score is None else rerank_score,
            relevance=self.relevance if relevance is None else relevance,
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: UUID
    chunk_id: UUID
    paper_id: UUID
    version_id: UUID
    paper_title: str
    section_id: UUID
    section_path: str
    page_start: int
    page_end: int
    element_ids: tuple[UUID, ...]
    text: str
    relevance: float
    dense_score: float | None
    bm25_score: float | None
    rerank_score: float | None


@dataclass(frozen=True, slots=True)
class SearchKnowledgeResult:
    query: str
    status: SearchStatus
    resolved_papers: tuple[ResolvedPaper, ...]
    evidence: tuple[Evidence, ...]
    has_sufficient_evidence: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.has_sufficient_evidence != bool(self.evidence):
            raise ValueError("Evidence flag must match evidence presence")
        if self.status == SearchStatus.NO_EVIDENCE and self.evidence:
            raise ValueError("no_evidence result cannot contain Evidence")
