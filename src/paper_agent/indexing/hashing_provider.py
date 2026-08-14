"""Offline deterministic embedding provider with true batch generation."""

from hashlib import blake2b
from math import log1p, sqrt

from paper_agent.domain.indexing import EmbeddingDescriptor
from paper_agent.ingestion.semantic_blocks import TOKEN_PATTERN


class HashingEmbeddingProvider:
    """Feature-hashing baseline that requires no model download or network access.

    It is intentionally replaceable through ``EmbeddingProvider``. Token and
    adjacent-token features make it useful for deterministic local retrieval,
    while production deployments can supply a neural provider with the same port.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 32:
            raise ValueError("Hashing embedding dimension must be at least 32")
        self.descriptor = EmbeddingDescriptor(
            provider="paper-agent",
            model=f"feature-hashing-{dimension}",
            version="hashing-v1",
            dimension=dimension,
        )

    def embed_batch(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed(text) for text in texts)

    def _embed(self, text: str) -> tuple[float, ...]:
        tokens = [token.casefold() for token in TOKEN_PATTERN.findall(text)]
        features = [*tokens, *(f"{left}\u241f{right}" for left, right in zip(tokens, tokens[1:]))]
        values = [0.0] * self.descriptor.dimension
        counts: dict[str, int] = {}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1
        for feature, count in counts.items():
            digest = blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.descriptor.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            values[index] += sign * log1p(count)
        norm = sqrt(sum(value * value for value in values))
        if norm == 0:
            return tuple(values)
        return tuple(value / norm for value in values)
