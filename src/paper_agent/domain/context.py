"""Budgeted evidence context and citations."""

from dataclasses import dataclass

from paper_agent.domain.retrieval import Evidence


@dataclass(frozen=True, slots=True)
class Citation:
    label: str
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class BuiltContext:
    text: str
    citations: tuple[Citation, ...]
    token_count: int
    omitted_evidence: int
