"""Versioned hierarchical indexing and embedding domain models."""

from dataclasses import dataclass
from uuid import UUID

from paper_agent.domain.enums import IndexLevel, IndexingStatus


@dataclass(frozen=True, slots=True)
class EmbeddingDescriptor:
    provider: str
    model: str
    version: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip() or not self.version.strip():
            raise ValueError("Embedding descriptor values cannot be blank")
        if self.dimension < 1:
            raise ValueError("Embedding dimension must be positive")

    @property
    def identifier(self) -> str:
        return f"{self.provider}:{self.model}:{self.version}:{self.dimension}"


@dataclass(frozen=True, slots=True)
class IndexDocument:
    target_id: UUID
    level: IndexLevel
    project_id: UUID
    paper_id: UUID
    version_id: UUID
    section_id: UUID | None
    text: str
    content_hash: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Index document text cannot be blank")
        if len(self.content_hash) != 64:
            raise ValueError("Index document content_hash must be SHA-256")
        if self.level == IndexLevel.PAPER and self.target_id != self.paper_id:
            raise ValueError("Paper index target must use paper_id")
        if self.level == IndexLevel.SECTION and self.section_id != self.target_id:
            raise ValueError("Section index target must use section_id")


@dataclass(frozen=True, slots=True)
class IndexedVector:
    document: IndexDocument
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.embedding:
            raise ValueError("IndexedVector embedding cannot be empty")


@dataclass(frozen=True, slots=True)
class IndexingState:
    project_id: UUID
    version_id: UUID
    embedding_version: str
    index_version: str
    source_digest: str
    status: IndexingStatus


@dataclass(frozen=True, slots=True)
class IndexingReport:
    project_id: UUID
    version_id: UUID
    embedding_version: str
    index_version: str
    papers: int
    sections: int
    chunks: int
    generated: int
    reused: int
    unchanged: bool = False
