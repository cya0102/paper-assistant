"""Versioned Section Tree and element-level document structure."""

from dataclasses import dataclass
from uuid import UUID

from paper_agent.domain.document import BoundingBox
from paper_agent.domain.enums import ElementType


@dataclass(frozen=True, slots=True)
class Section:
    section_id: UUID
    paper_id: UUID
    version_id: UUID
    parent_section_id: UUID | None
    title: str
    normalized_title: str
    level: int
    section_order: int
    section_path: str
    page_start: int
    page_end: int
    source_heading_block_id: str | None
    source_block_ids: tuple[str, ...]
    structure_version: str

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.normalized_title.strip():
            raise ValueError("Section title cannot be empty")
        if self.level < 1:
            raise ValueError("Section level must be positive")
        if self.section_order < 0:
            raise ValueError("section_order cannot be negative")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("Invalid Section page range")
        if len(self.source_block_ids) != len(set(self.source_block_ids)):
            raise ValueError("Section source_block_ids must be unique")


@dataclass(frozen=True, slots=True)
class Element:
    element_id: UUID
    paper_id: UUID
    version_id: UUID
    section_id: UUID
    element_type: ElementType
    label: str | None
    caption: str | None
    content: str | None
    page: int
    bbox: BoundingBox | None
    source_block_ids: tuple[str, ...]
    structure_version: str

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError("Element page must be positive")
        if not self.source_block_ids:
            raise ValueError("Element must retain at least one source block")


@dataclass(frozen=True, slots=True)
class StructuredDocument:
    paper_id: UUID
    version_id: UUID
    structure_version: str
    sections: tuple[Section, ...]
    elements: tuple[Element, ...]

    def __post_init__(self) -> None:
        section_ids = {section.section_id for section in self.sections}
        if len(section_ids) != len(self.sections):
            raise ValueError("Section IDs must be unique")
        for section in self.sections:
            if section.paper_id != self.paper_id or section.version_id != self.version_id:
                raise ValueError("Section identity must match StructuredDocument")
            if section.parent_section_id is not None and section.parent_section_id not in section_ids:
                raise ValueError("Section parent must exist in the same StructuredDocument")
        element_ids = {element.element_id for element in self.elements}
        if len(element_ids) != len(self.elements):
            raise ValueError("Element IDs must be unique")
        for element in self.elements:
            if element.paper_id != self.paper_id or element.version_id != self.version_id:
                raise ValueError("Element identity must match StructuredDocument")
            if element.section_id not in section_ids:
                raise ValueError("Element section must exist in StructuredDocument")

