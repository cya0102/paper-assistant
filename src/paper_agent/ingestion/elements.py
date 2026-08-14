"""Basic Figure, Table, Equation, and Algorithm extraction."""

import re
from dataclasses import replace
from uuid import uuid5

from paper_agent.domain.document import CanonicalParsedDocument
from paper_agent.domain.enums import BlockType, ElementType
from paper_agent.domain.structure import Element, StructuredDocument
from paper_agent.ingestion.document_blocks import ordered_blocks

CAPTION_PATTERN = re.compile(
    r"^\s*(fig(?:ure)?\.?|table|algorithm|alg\.?|equation|eq\.?)\s*([A-Z]?\d+|[IVXLC]+)?\s*[:.)-]?\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
BLOCK_ELEMENT_TYPES = {
    BlockType.FIGURE: ElementType.FIGURE,
    BlockType.TABLE: ElementType.TABLE,
    BlockType.EQUATION: ElementType.EQUATION,
    BlockType.ALGORITHM: ElementType.ALGORITHM,
}


class ElementExtractor:
    version = "element-v1"

    def extract(
        self,
        document: CanonicalParsedDocument,
        structured: StructuredDocument,
    ) -> StructuredDocument:
        block_to_section = {
            block_id: section.section_id
            for section in structured.sections
            for block_id in section.source_block_ids
        }
        elements: list[Element] = []
        for item in ordered_blocks(document):
            block = item.block
            text = " ".join((block.text or "").split()).strip()
            element_type = BLOCK_ELEMENT_TYPES.get(block.block_type)
            label: str | None = None
            caption: str | None = None
            match = CAPTION_PATTERN.match(text)
            if element_type is None and match:
                element_type = self._caption_type(match.group(1))
            if element_type is None:
                continue
            if match:
                number = match.group(2)
                label = f"{element_type.value.title()} {number}" if number else None
                caption = text
            section_id = block_to_section.get(block.block_id)
            if section_id is None:
                continue
            elements.append(
                Element(
                    element_id=uuid5(
                        document.version_id,
                        f"element:{element_type.value}:{block.block_id}",
                    ),
                    paper_id=document.paper_id,
                    version_id=document.version_id,
                    section_id=section_id,
                    element_type=element_type,
                    label=label,
                    caption=caption if element_type != ElementType.EQUATION else None,
                    content=text if element_type == ElementType.EQUATION else None,
                    page=item.page_number,
                    bbox=block.bbox,
                    source_block_ids=(block.block_id,),
                    structure_version=structured.structure_version,
                )
            )
        return replace(structured, elements=tuple(elements))

    @staticmethod
    def _caption_type(value: str) -> ElementType:
        normalized = value.casefold()
        if normalized.startswith("fig"):
            return ElementType.FIGURE
        if normalized.startswith("table"):
            return ElementType.TABLE
        if normalized.startswith(("algorithm", "alg")):
            return ElementType.ALGORITHM
        return ElementType.EQUATION

