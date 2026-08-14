"""Rule-first heading hierarchy recovery and Block-to-Section assignment."""

import re
from dataclasses import dataclass, replace
from uuid import UUID, uuid5

from paper_agent.domain.document import CanonicalParsedDocument, DocumentBlock
from paper_agent.domain.enums import BlockType
from paper_agent.domain.structure import Section, StructuredDocument
from paper_agent.ingestion.document_blocks import LocatedBlock, ordered_blocks
from paper_agent.ingestion.normalization import normalize_title

NUMBERED_HEADING = re.compile(r"^\s*((?:\d+\.)*\d+)\s*[.)]?\s+\S")
ROMAN_HEADING = re.compile(r"^\s*[IVXLC]+[.)]\s+\S", re.IGNORECASE)
LETTER_HEADING = re.compile(r"^\s*[A-Z][.)]\s+\S")
TOP_LEVEL_TITLES = frozenset(
    {
        "abstract",
        "introduction",
        "related work",
        "background",
        "method",
        "methodology",
        "approach",
        "experiments",
        "experimental results",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "appendix",
    }
)


@dataclass(frozen=True, slots=True)
class HeadingCandidate:
    located_block: LocatedBlock
    title: str
    level: int


class SectionTreeBuilder:
    version = "section-tree-v1"

    def build(self, document: CanonicalParsedDocument) -> StructuredDocument:
        located = ordered_blocks(document)
        if not located:
            raise ValueError("Cannot build a Section Tree for a document without blocks")
        candidates = {
            candidate.located_block.block.block_id: candidate
            for item in located
            if (candidate := self._heading_candidate(item)) is not None
        }
        sections = self._build_sections(document, located, candidates)
        return StructuredDocument(
            paper_id=document.paper_id,
            version_id=document.version_id,
            structure_version=self.version,
            sections=sections,
            elements=(),
        )

    def _build_sections(
        self,
        document: CanonicalParsedDocument,
        located: tuple[LocatedBlock, ...],
        candidates: dict[str, HeadingCandidate],
    ) -> tuple[Section, ...]:
        sections: list[Section] = []
        section_blocks: list[list[LocatedBlock]] = []
        stack: list[int] = []
        current_index: int | None = None

        for item in located:
            candidate = candidates.get(item.block.block_id)
            if candidate is not None:
                level = candidate.level
                if stack:
                    level = min(level, len(stack) + 1)
                else:
                    level = 1
                while len(stack) >= level:
                    stack.pop()
                parent_index = stack[-1] if stack else None
                section_index = len(sections)
                section_id = uuid5(
                    document.version_id,
                    f"section:{section_index}:{item.block.block_id}:{candidate.title}",
                )
                parent_id = sections[parent_index].section_id if parent_index is not None else None
                path_titles = [sections[index].title for index in stack] + [candidate.title]
                normalized = normalize_title(candidate.title) or candidate.title.casefold()
                sections.append(
                    Section(
                        section_id=section_id,
                        paper_id=document.paper_id,
                        version_id=document.version_id,
                        parent_section_id=parent_id,
                        title=candidate.title,
                        normalized_title=normalized,
                        level=level,
                        section_order=section_index,
                        section_path=" > ".join(path_titles),
                        page_start=item.page_number,
                        page_end=item.page_number,
                        source_heading_block_id=item.block.block_id,
                        source_block_ids=(),
                        structure_version=self.version,
                    )
                )
                section_blocks.append([])
                stack.append(section_index)
                current_index = section_index

            if current_index is None:
                current_index = self._create_front_matter(document, sections, section_blocks, item)
                stack = [current_index]
            section_blocks[current_index].append(item)

        finalized: list[Section] = []
        for section, blocks in zip(sections, section_blocks, strict=True):
            finalized.append(
                replace(
                    section,
                    page_start=min(item.page_number for item in blocks),
                    page_end=max(item.page_number for item in blocks),
                    source_block_ids=tuple(item.block.block_id for item in blocks),
                )
            )
        return tuple(finalized)

    def _create_front_matter(
        self,
        document: CanonicalParsedDocument,
        sections: list[Section],
        section_blocks: list[list[LocatedBlock]],
        first_block: LocatedBlock,
    ) -> int:
        index = len(sections)
        sections.append(
            Section(
                section_id=uuid5(document.version_id, "section:front-matter"),
                paper_id=document.paper_id,
                version_id=document.version_id,
                parent_section_id=None,
                title="Front Matter",
                normalized_title="front matter",
                level=1,
                section_order=index,
                section_path="Front Matter",
                page_start=first_block.page_number,
                page_end=first_block.page_number,
                source_heading_block_id=None,
                source_block_ids=(),
                structure_version=self.version,
            )
        )
        section_blocks.append([])
        return index

    @staticmethod
    def _heading_candidate(item: LocatedBlock) -> HeadingCandidate | None:
        block = item.block
        text = " ".join((block.text or "").split()).strip()
        if not text or len(text) > 240 or text.count(" ") > 30:
            return None
        explicit_level = block.attributes.get("level")
        if isinstance(explicit_level, int) and block.block_type == BlockType.HEADING:
            return HeadingCandidate(item, text, max(1, min(explicit_level, 6)))
        numbered = NUMBERED_HEADING.match(text)
        if numbered:
            return HeadingCandidate(item, text, numbered.group(1).count(".") + 1)
        normalized = normalize_title(text)
        if normalized in TOP_LEVEL_TITLES:
            return HeadingCandidate(item, text, 1)
        if ROMAN_HEADING.match(text):
            return HeadingCandidate(item, text, 1)
        if LETTER_HEADING.match(text):
            return HeadingCandidate(item, text, 2)
        if block.block_type == BlockType.HEADING:
            return HeadingCandidate(item, text, 1)
        return None

