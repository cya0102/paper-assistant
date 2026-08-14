"""Semantic dependency groups and Section-aware retrieval chunks."""

from dataclasses import dataclass
from uuid import UUID

from paper_agent.domain.enums import ChunkType, SemanticGroupType


@dataclass(frozen=True, slots=True)
class SemanticGroup:
    group_id: UUID
    paper_id: UUID
    version_id: UUID
    section_id: UUID
    group_order: int
    group_type: SemanticGroupType
    text: str
    token_count: int
    page_start: int
    page_end: int
    source_block_ids: tuple[str, ...]
    related_element_ids: tuple[UUID, ...]
    structure_version: str

    def __post_init__(self) -> None:
        if self.group_order < 0 or self.token_count < 0:
            raise ValueError("Group order and token count cannot be negative")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("Invalid SemanticGroup page range")
        if not self.source_block_ids:
            raise ValueError("SemanticGroup must retain source blocks")


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: UUID
    paper_id: UUID
    version_id: UUID
    section_id: UUID
    section_path: str
    chunk_order: int
    chunk_type: ChunkType
    text: str
    token_count: int
    page_start: int
    page_end: int
    source_group_ids: tuple[UUID, ...]
    source_block_ids: tuple[str, ...]
    related_element_ids: tuple[UUID, ...]
    chunking_version: str

    def __post_init__(self) -> None:
        if self.chunk_order < 0 or self.token_count < 0:
            raise ValueError("Chunk order and token count cannot be negative")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("Invalid Chunk page range")
        if not self.source_group_ids or not self.source_block_ids:
            raise ValueError("Chunk must retain group and block provenance")


@dataclass(frozen=True, slots=True)
class DerivedDataState:
    version_id: UUID
    structure_version: str | None = None
    chunking_version: str | None = None
    document_hash: str | None = None

    def __post_init__(self) -> None:
        if self.document_hash is not None and (
            len(self.document_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.document_hash)
        ):
            raise ValueError("document_hash must be a lowercase SHA-256 digest")
