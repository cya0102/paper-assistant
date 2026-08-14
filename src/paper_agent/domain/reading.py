"""Traceable paper reading contracts."""

from dataclasses import dataclass
from uuid import UUID

from paper_agent.domain.enums import ElementType


@dataclass(frozen=True, slots=True)
class ReadPaperRequest:
    paper_id: UUID
    version_id: UUID | None = None
    section_id: UUID | None = None
    page_range: tuple[int, int] | None = None
    element_id: UUID | None = None
    element_types: tuple[ElementType, ...] = ()
    include_neighbors: bool = True
    neighbor_radius: int = 1

    def __post_init__(self) -> None:
        if self.page_range and (self.page_range[0] < 1 or self.page_range[1] < self.page_range[0]):
            raise ValueError("Invalid page range")
        if not 0 <= self.neighbor_radius <= 5:
            raise ValueError("neighbor_radius must be between 0 and 5")


@dataclass(frozen=True, slots=True)
class ReadPassage:
    chunk_id: UUID
    section_id: UUID
    section_path: str
    page_start: int
    page_end: int
    chunk_order: int
    text: str
    source_group_ids: tuple[UUID, ...]
    source_block_ids: tuple[str, ...]
    element_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class ReadElement:
    element_id: UUID
    element_type: ElementType
    section_id: UUID
    label: str | None
    caption: str | None
    content: str | None
    page: int
    source_block_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReadPaperResult:
    paper_id: UUID
    version_id: UUID
    title: str
    passages: tuple[ReadPassage, ...]
    elements: tuple[ReadElement, ...]
