"""Calibrated offline reranker used by the default search assembly."""

from math import exp

from paper_agent.domain.retrieval import RetrievalCandidate
from paper_agent.ingestion.semantic_blocks import TOKEN_PATTERN


def search_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in TOKEN_PATTERN.findall(text))


class LexicalHybridReranker:
    version = "lexical-hybrid-reranker-v1"

    def rerank(
        self, query: str, candidates: tuple[RetrievalCandidate, ...]
    ) -> tuple[RetrievalCandidate, ...]:
        query_tokens = set(search_tokens(query))
        ranked: list[RetrievalCandidate] = []
        for candidate in candidates:
            candidate_tokens = set(search_tokens(candidate.text))
            overlap = (
                len(query_tokens & candidate_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            phrase_bonus = 1.0 if query.casefold().strip() in candidate.text.casefold() else 0.0
            lexical = min(1.0, overlap * 0.85 + phrase_bonus * 0.15)
            dense = max(0.0, min(1.0, candidate.dense_score or 0.0))
            sparse = 1.0 - exp(-max(0.0, candidate.bm25_score or 0.0) / 3.0)
            score = min(1.0, dense * 0.45 + sparse * 0.25 + lexical * 0.30)
            ranked.append(
                candidate.with_scores(rerank_score=score, relevance=score)
            )
        return tuple(
            sorted(
                ranked,
                key=lambda item: (item.relevance, item.bm25_score or 0.0),
                reverse=True,
            )
        )
