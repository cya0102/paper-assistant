"""Dependency-inversion ports for embedding and index persistence."""

from typing import Protocol
from uuid import UUID

from paper_agent.domain.indexing import (
    EmbeddingDescriptor,
    IndexDocument,
    IndexedVector,
    IndexingState,
)


class EmbeddingProvider(Protocol):
    descriptor: EmbeddingDescriptor

    def embed_batch(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...


class IndexRepository(Protocol):
    def load_documents(self, project_id: UUID, version_id: UUID) -> tuple[IndexDocument, ...]: ...

    def get_state(self, project_id: UUID, version_id: UUID) -> IndexingState | None: ...

    def load_reusable_vectors(
        self,
        project_id: UUID,
        version_id: UUID,
        embedding_version: str,
    ) -> dict[str, tuple[float, ...]]: ...

    def replace_index(
        self,
        project_id: UUID,
        version_id: UUID,
        descriptor: EmbeddingDescriptor,
        index_version: str,
        source_digest: str,
        vectors: tuple[IndexedVector, ...],
    ) -> None: ...


class VersionIndexer(Protocol):
    version: str

    def is_current(self, project_id: UUID, version_id: UUID) -> bool: ...

    def index_version(
        self, project_id: UUID, version_id: UUID, *, force: bool = False
    ) -> object: ...
