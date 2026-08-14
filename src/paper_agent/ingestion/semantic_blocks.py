"""Section-local semantic blocks and element dependency groups."""

import re
from collections import defaultdict
from uuid import UUID, uuid5

from paper_agent.domain.chunk import SemanticGroup
from paper_agent.domain.document import CanonicalParsedDocument
from paper_agent.domain.enums import BlockType, ElementType, SemanticGroupType
from paper_agent.domain.structure import Element, StructuredDocument
from paper_agent.ingestion.document_blocks import LocatedBlock, ordered_blocks

TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")


def count_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


class SemanticBlockBuilder:
    version = "semantic-block-v1"

    def build(
        self,
        document: CanonicalParsedDocument,
        structured: StructuredDocument,
    ) -> tuple[SemanticGroup, ...]:
        located_by_id = {item.block.block_id: item for item in ordered_blocks(document)}
        elements_by_section: dict[UUID, list[Element]] = defaultdict(list)
        for element in structured.elements:
            elements_by_section[element.section_id].append(element)

        groups: list[SemanticGroup] = []
        for section in sorted(structured.sections, key=lambda item: item.section_order):
            blocks = [
                located_by_id[block_id]
                for block_id in section.source_block_ids
                if block_id in located_by_id
            ]
            spans = self._dependency_spans(blocks, elements_by_section[section.section_id])
            consumed: set[int] = set()
            section_groups: list[tuple[list[LocatedBlock], tuple[UUID, ...]]] = []
            for start, end, element_ids in spans:
                indexes = set(range(start, end + 1))
                if indexes & consumed:
                    continue
                section_groups.append((blocks[start : end + 1], element_ids))
                consumed.update(indexes)
            for index, block in enumerate(blocks):
                if index not in consumed:
                    section_groups.append(([block], ()))
            section_groups.sort(key=lambda item: blocks.index(item[0][0]))

            for group_blocks, element_ids in section_groups:
                text = "\n\n".join(
                    value
                    for item in group_blocks
                    if (value := (item.block.text or "").strip())
                )
                if not text:
                    continue
                source_ids = tuple(item.block.block_id for item in group_blocks)
                group_order = len(groups)
                groups.append(
                    SemanticGroup(
                        group_id=uuid5(
                            document.version_id,
                            f"group:{section.section_id}:{':'.join(source_ids)}",
                        ),
                        paper_id=document.paper_id,
                        version_id=document.version_id,
                        section_id=section.section_id,
                        group_order=group_order,
                        group_type=(
                            SemanticGroupType.ELEMENT_DEPENDENCY
                            if element_ids
                            else SemanticGroupType.TEXT
                        ),
                        text=text,
                        token_count=count_tokens(text),
                        page_start=min(item.page_number for item in group_blocks),
                        page_end=max(item.page_number for item in group_blocks),
                        source_block_ids=source_ids,
                        related_element_ids=element_ids,
                        structure_version=structured.structure_version,
                    )
                )
        return tuple(groups)

    def _dependency_spans(
        self,
        blocks: list[LocatedBlock],
        elements: list[Element],
    ) -> list[tuple[int, int, tuple[UUID, ...]]]:
        index_by_block = {item.block.block_id: index for index, item in enumerate(blocks)}
        spans: list[tuple[int, int, tuple[UUID, ...]]] = []
        for element in elements:
            indexes = [
                index_by_block[block_id]
                for block_id in element.source_block_ids
                if block_id in index_by_block
            ]
            if not indexes:
                continue
            start = min(indexes)
            end = max(indexes)
            if start > 0 and self._should_bind_previous(blocks[start - 1], element):
                start -= 1
            if end + 1 < len(blocks) and self._should_bind_next(blocks[end + 1], element):
                end += 1
            spans.append((start, end, (element.element_id,)))
        return self._merge_spans(spans)

    @staticmethod
    def _should_bind_previous(block: LocatedBlock, element: Element) -> bool:
        if block.block.block_type == BlockType.HEADING:
            return False
        text = (block.block.text or "").casefold()
        if element.label and element.label.casefold() in text:
            return True
        return element.element_type == ElementType.EQUATION and len(text.split()) <= 80

    @staticmethod
    def _should_bind_next(block: LocatedBlock, element: Element) -> bool:
        text = (block.block.text or "").casefold()
        if element.label and element.label.casefold() in text:
            return True
        cues = ("where ", "whereas ", "in which ", "as shown", "denotes ", "represents ")
        return element.element_type == ElementType.EQUATION and text.startswith(cues)

    @staticmethod
    def _merge_spans(
        spans: list[tuple[int, int, tuple[UUID, ...]]],
    ) -> list[tuple[int, int, tuple[UUID, ...]]]:
        merged: list[tuple[int, int, tuple[UUID, ...]]] = []
        for start, end, element_ids in sorted(spans):
            if merged and start <= merged[-1][1]:
                old_start, old_end, old_ids = merged[-1]
                merged[-1] = (
                    old_start,
                    max(old_end, end),
                    tuple(dict.fromkeys((*old_ids, *element_ids))),
                )
            else:
                merged.append((start, end, element_ids))
        return merged
