from dataclasses import replace
from uuid import uuid4

from paper_agent.domain.enums import SearchStatus
from paper_agent.domain.retrieval import Evidence, SearchKnowledgeResult, SearchRequest, SearchScope
from paper_agent.retrieval.advanced import AdvancedSearchKnowledgeService, ConservativeQueryRewriter, LexicalEvidenceJudge


class Rewriter:
    version = "test"

    def rewrite(self, query, *, max_queries):
        return (query, "teacher guidance cross attention")[:max_queries]


class Search:
    def __init__(self, evidence):
        self.evidence = evidence
        self.queries = []

    def search_knowledge(self, request):
        self.queries.append(request.query)
        item = replace(self.evidence, relevance=0.8 if len(self.queries) == 1 else 0.9)
        return SearchKnowledgeResult(request.query, SearchStatus.OK, (), (item,), True)


class Neighbors:
    def __init__(self, neighbor):
        self.neighbor = neighbor

    def expand(self, evidence, radius):
        assert radius == 1
        return (*evidence, self.neighbor)


def evidence(text, *, chunk_id=None):
    return Evidence(
        evidence_id=uuid4(),
        chunk_id=chunk_id or uuid4(),
        paper_id=uuid4(),
        version_id=uuid4(),
        paper_title="Paper",
        section_id=uuid4(),
        section_path="Method",
        page_start=2,
        page_end=2,
        element_ids=(),
        text=text,
        relevance=0.8,
        dense_score=0.8,
        bm25_score=None,
        rerank_score=0.8,
    )


def test_multi_query_fuses_duplicates_expands_neighbors_and_judges():
    seed = evidence("teacher guidance cross attention codebook")
    neighbor = replace(
        evidence("cross attention receives teacher guidance"),
        paper_id=seed.paper_id,
        version_id=seed.version_id,
        section_id=seed.section_id,
    )
    base = Search(seed)
    service = AdvancedSearchKnowledgeService(
        base, Rewriter(), LexicalEvidenceJudge(), Neighbors(neighbor), judge_threshold=0.1
    )
    result = service.search_knowledge(
        SearchRequest("why teacher guidance?", SearchScope(project_id=uuid4()), max_evidence=5)
    )
    assert len(base.queries) == 2
    assert {item.chunk_id for item in result.evidence} == {seed.chunk_id, neighbor.chunk_id}


def test_conservative_rewriter_generates_bounded_focused_queries():
    queries = ConservativeQueryRewriter().rewrite(
        "SCANet 的 codebook 是怎么得到的？", max_queries=4
    )
    assert 2 <= len(queries) <= 4
    assert any("SCANet" in item and "codebook" in item for item in queries)
