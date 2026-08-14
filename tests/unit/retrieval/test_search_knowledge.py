from uuid import uuid4

from paper_agent.domain.enums import SearchStatus
from paper_agent.domain.retrieval import (
    RankedTarget,
    ResolvedPaper,
    RetrievalCandidate,
    SearchRequest,
    SearchScope,
)
from paper_agent.indexing.hashing_provider import HashingEmbeddingProvider
from paper_agent.retrieval.reranker import LexicalHybridReranker
from paper_agent.retrieval.service import SearchKnowledgeService


class MemorySearchRepository:
    def __init__(self, candidates, *, resolve=True):
        self.candidates = candidates
        self.should_resolve = resolve
        self.paper_id = candidates[0].paper_id
        self.version_id = candidates[0].version_id
        self.section_id = candidates[0].section_id

    def resolve_papers(self, query, scope, filters, limit):
        del query, scope, filters, limit
        if not self.should_resolve:
            return ()
        return (ResolvedPaper(self.paper_id, self.version_id, "Codebook Paper", 1.0, "metadata"),)

    def dense_papers(self, vector, descriptor, scope, filters, limit):
        del vector, descriptor, scope, filters, limit
        return (ResolvedPaper(self.paper_id, self.version_id, "Codebook Paper", 0.8, "dense"),)

    def dense_sections(self, vector, descriptor, scope, filters, paper_ids, limit):
        del vector, descriptor, scope, filters, paper_ids, limit
        return (RankedTarget(self.section_id, self.paper_id, self.version_id, 0.9),)

    def dense_chunks(self, vector, descriptor, scope, filters, paper_ids, section_ids, limit):
        del vector, descriptor, scope, filters, paper_ids, section_ids, limit
        return tuple(self.candidates)

    def sparse_chunks(self, query, scope, filters, paper_ids, section_ids, limit):
        del scope, filters, paper_ids, section_ids, limit
        if "codebook" not in query.casefold():
            return ()
        first = self.candidates[0]
        return (first.with_scores(bm25_score=5.0),)


def _candidate(text, dense_score, *, chunk_id=None, chunk_order=0):
    return RetrievalCandidate(
        chunk_id=chunk_id or uuid4(),
        paper_id=PAPER_ID,
        version_id=VERSION_ID,
        paper_title="Codebook Paper",
        section_id=SECTION_ID,
        section_path="3 Method > 3.2 Codebook",
        page_start=5,
        page_end=6,
        element_ids=(ELEMENT_ID,),
        text=text,
        chunk_order=chunk_order,
        dense_score=dense_score,
    )


PAPER_ID, VERSION_ID, SECTION_ID, ELEMENT_ID = uuid4(), uuid4(), uuid4(), uuid4()


def test_search_knowledge_hybrid_reranks_deduplicates_and_preserves_provenance() -> None:
    first_id = uuid4()
    candidates = (
        _candidate("The codebook is constructed by clustering scene features.", 0.85, chunk_id=first_id),
        _candidate("The codebook is constructed by clustering scene features.", 0.80, chunk_order=1),
        _candidate("Training uses a reconstruction objective.", 0.30, chunk_order=2),
    )
    service = SearchKnowledgeService(
        MemorySearchRepository(candidates),
        HashingEmbeddingProvider(),
        LexicalHybridReranker(),
    )
    result = service.search_knowledge(
        SearchRequest(
            query="How is the codebook constructed?",
            scope=SearchScope(project_id=uuid4(), paper_ids=(PAPER_ID,)),
            max_evidence=5,
        )
    )

    assert result.status == SearchStatus.OK
    assert result.has_sufficient_evidence
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.chunk_id == first_id
    assert evidence.paper_id == PAPER_ID
    assert evidence.version_id == VERSION_ID
    assert evidence.section_id == SECTION_ID
    assert evidence.page_start == 5 and evidence.page_end == 6
    assert evidence.element_ids == (ELEMENT_ID,)
    assert evidence.dense_score == 0.85
    assert evidence.bm25_score == 5.0
    assert evidence.rerank_score == evidence.relevance


def test_search_knowledge_returns_explicit_no_evidence_below_threshold() -> None:
    candidates = (_candidate("Unrelated optimization details.", 0.05),)
    service = SearchKnowledgeService(
        MemorySearchRepository(candidates, resolve=False),
        HashingEmbeddingProvider(),
        LexicalHybridReranker(),
    )
    result = service.search_knowledge(
        SearchRequest(
            query="quantum banana telescope",
            scope=SearchScope(project_id=uuid4()),
        )
    )

    assert result.status == SearchStatus.NO_EVIDENCE
    assert not result.has_sufficient_evidence
    assert result.evidence == ()
    assert result.reason
