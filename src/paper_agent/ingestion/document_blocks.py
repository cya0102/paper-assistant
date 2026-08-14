"""Ordered Canonical Document block references shared by structure builders."""

from dataclasses import dataclass

from paper_agent.domain.document import CanonicalParsedDocument, DocumentBlock


@dataclass(frozen=True, slots=True)
class LocatedBlock:
    page_number: int
    block: DocumentBlock


def ordered_blocks(document: CanonicalParsedDocument) -> tuple[LocatedBlock, ...]:
    return tuple(
        LocatedBlock(page.page_number, block)
        for page in document.pages
        for block in sorted(page.blocks, key=lambda item: item.reading_order)
    )

