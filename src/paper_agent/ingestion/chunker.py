"""Section-aware semantic chunking with complete provenance."""

import re
from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID, uuid5

from paper_agent.domain.chunk import Chunk, SemanticGroup
from paper_agent.domain.enums import ChunkType, ElementType, SemanticGroupType
from paper_agent.domain.structure import StructuredDocument
from paper_agent.ingestion.semantic_blocks import TOKEN_PATTERN, count_tokens

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 600
    hard_max_tokens: int = 800
    min_fill_ratio: float = 0.6

    def __post_init__(self) -> None:
        if self.target_tokens <= 0 or self.hard_max_tokens < self.target_tokens:
            raise ValueError("Chunk token limits are invalid")
        if not 0 < self.min_fill_ratio <= 1:
            raise ValueError("min_fill_ratio must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class _ChunkPart:
    text: str
    token_count: int
    page_start: int
    page_end: int
    group_ids: tuple[UUID, ...]
    block_ids: tuple[str, ...]
    element_ids: tuple[UUID, ...]
    dependency: bool


class SemanticChunker:
    version = "semantic-chunker-v1"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(
        self,
        structured: StructuredDocument,
        groups: tuple[SemanticGroup, ...],
    ) -> tuple[Chunk, ...]:
        groups_by_section: dict[UUID, list[SemanticGroup]] = defaultdict(list)
        for group in groups:
            groups_by_section[group.section_id].append(group)
        element_types = {
            element.element_id: element.element_type for element in structured.elements
        }
        chunks: list[Chunk] = []
        for section in sorted(structured.sections, key=lambda item: item.section_order):
            parts = [
                part
                for group in sorted(
                    groups_by_section[section.section_id], key=lambda item: item.group_order
                )
                for part in self._parts(group)
            ]
            current: list[_ChunkPart] = []
            for part in parts:
                proposed_tokens = sum(item.token_count for item in current) + part.token_count
                if not current or proposed_tokens <= self.config.target_tokens:
                    current.append(part)
                    continue
                current_tokens = sum(item.token_count for item in current)
                if (
                    proposed_tokens <= self.config.hard_max_tokens
                    and current_tokens < self.config.target_tokens * self.config.min_fill_ratio
                ):
                    current.append(part)
                    continue
                chunks.append(
                    self._make_chunk(structured, section.section_path, current, element_types, len(chunks))
                )
                current = [part]
            if current:
                chunks.append(
                    self._make_chunk(structured, section.section_path, current, element_types, len(chunks))
                )
        return tuple(chunks)

    def _parts(self, group: SemanticGroup) -> tuple[_ChunkPart, ...]:
        base = _ChunkPart(
            text=group.text,
            token_count=group.token_count,
            page_start=group.page_start,
            page_end=group.page_end,
            group_ids=(group.group_id,),
            block_ids=group.source_block_ids,
            element_ids=group.related_element_ids,
            dependency=group.group_type == SemanticGroupType.ELEMENT_DEPENDENCY,
        )
        if base.dependency or base.token_count <= self.config.hard_max_tokens:
            return (base,)
        sentences = SENTENCE_BOUNDARY.split(group.text)
        if len(sentences) == 1:
            return self._word_parts(base)
        parts: list[_ChunkPart] = []
        current: list[str] = []
        for sentence in sentences:
            proposed = " ".join((*current, sentence)).strip()
            if current and count_tokens(proposed) > self.config.hard_max_tokens:
                parts.append(self._text_part(base, " ".join(current)))
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            parts.append(self._text_part(base, " ".join(current)))
        bounded: list[_ChunkPart] = []
        for part in parts:
            if part.token_count > self.config.hard_max_tokens:
                bounded.extend(self._word_parts(part))
            else:
                bounded.append(part)
        return tuple(bounded)

    def _word_parts(self, base: _ChunkPart) -> tuple[_ChunkPart, ...]:
        words = TOKEN_PATTERN.findall(base.text)
        parts: list[_ChunkPart] = []
        current: list[str] = []
        for word in words:
            proposed = " ".join((*current, word))
            if current and count_tokens(proposed) > self.config.hard_max_tokens:
                parts.append(self._text_part(base, " ".join(current)))
                current = [word]
            else:
                current.append(word)
        if current:
            parts.append(self._text_part(base, " ".join(current)))
        return tuple(parts) or (base,)

    @staticmethod
    def _text_part(base: _ChunkPart, text: str) -> _ChunkPart:
        return _ChunkPart(
            text=text,
            token_count=count_tokens(text),
            page_start=base.page_start,
            page_end=base.page_end,
            group_ids=base.group_ids,
            block_ids=base.block_ids,
            element_ids=base.element_ids,
            dependency=False,
        )

    def _make_chunk(
        self,
        structured: StructuredDocument,
        section_path: str,
        parts: list[_ChunkPart],
        element_types: dict[UUID, ElementType],
        chunk_order: int,
    ) -> Chunk:
        group_ids = tuple(dict.fromkeys(group_id for part in parts for group_id in part.group_ids))
        block_ids = tuple(dict.fromkeys(block_id for part in parts for block_id in part.block_ids))
        element_ids = tuple(
            dict.fromkeys(element_id for part in parts for element_id in part.element_ids)
        )
        section_id = next(
            section.section_id
            for section in structured.sections
            if section.section_path == section_path
        )
        text = "\n\n".join(part.text for part in parts)
        return Chunk(
            chunk_id=uuid5(
                structured.version_id,
                f"chunk:{self.version}:{section_id}:{':'.join(map(str, group_ids))}:{chunk_order}",
            ),
            paper_id=structured.paper_id,
            version_id=structured.version_id,
            section_id=section_id,
            section_path=section_path,
            chunk_order=chunk_order,
            chunk_type=self._chunk_type(element_ids, element_types),
            text=text,
            token_count=count_tokens(text),
            page_start=min(part.page_start for part in parts),
            page_end=max(part.page_end for part in parts),
            source_group_ids=group_ids,
            source_block_ids=block_ids,
            related_element_ids=element_ids,
            chunking_version=self.version,
        )

    @staticmethod
    def _chunk_type(
        element_ids: tuple[UUID, ...], element_types: dict[UUID, ElementType]
    ) -> ChunkType:
        types = {element_types[element_id] for element_id in element_ids if element_id in element_types}
        if not types:
            return ChunkType.TEXT
        if len(types) > 1:
            return ChunkType.MIXED
        element_type = next(iter(types))
        return ChunkType(element_type.value)
