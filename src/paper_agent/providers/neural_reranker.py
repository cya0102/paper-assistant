"""Optional sentence-transformers CrossEncoder reranker."""

from math import exp
from typing import Any

from paper_agent.domain.retrieval import RetrievalCandidate


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("Install neural providers with: uv sync --extra neural") from error
        self._model: Any = CrossEncoder(model)
        self.version = f"sentence-transformers:{model}:v1"

    def rerank(
        self, query: str, candidates: tuple[RetrievalCandidate, ...]
    ) -> tuple[RetrievalCandidate, ...]:
        if not candidates:
            return ()
        scores = self._model.predict([(query, item.text) for item in candidates])
        ranked = [
            item.with_scores(
                rerank_score=normalized,
                relevance=normalized,
            )
            for item, raw_score in zip(candidates, scores, strict=True)
            if (normalized := 1.0 / (1.0 + exp(-float(raw_score)))) >= 0.0
        ]
        return tuple(sorted(ranked, key=lambda item: item.relevance, reverse=True))
