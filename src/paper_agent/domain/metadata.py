"""Deterministic PDF metadata used for identity resolution."""

from dataclasses import dataclass, field

from paper_agent.domain.enums import MetadataSource


@dataclass(frozen=True, slots=True)
class MetadataEvidence:
    field_name: str
    source: MetadataSource
    confidence: float

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    title: str | None = None
    normalized_title: str | None = None
    authors: tuple[str, ...] = ()
    normalized_authors: tuple[str, ...] = ()
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    venue: str | None = None
    subject: str | None = None
    keywords: tuple[str, ...] = ()
    page_count: int = 0
    content_hash: str | None = None
    evidence: tuple[MetadataEvidence, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.page_count < 0:
            raise ValueError("page_count cannot be negative")
        if self.content_hash is not None:
            if len(self.content_hash) != 64 or any(
                character not in "0123456789abcdef" for character in self.content_hash
            ):
                raise ValueError("content_hash must be a lowercase SHA-256 hex digest")

