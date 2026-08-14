"""Structured, evidence-backed multi-paper comparison results."""

from dataclasses import dataclass
from uuid import UUID

from paper_agent.domain.enums import (
    ComparisonCellStatus,
    ComparisonDimensionName,
    ComparisonStatus,
    ReviewStatus,
)
from paper_agent.domain.research_graph import EvidenceLink


@dataclass(frozen=True, slots=True)
class ComparisonCell:
    paper_id: UUID
    paper_title: str
    dimension: ComparisonDimensionName
    status: ComparisonCellStatus
    normalized_value: str | None
    raw_description: str | None
    directly_comparable: bool
    non_comparable_reason: str | None
    evidence_links: tuple[EvidenceLink, ...]
    confidence: float
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED

    def __post_init__(self) -> None:
        if not self.paper_title.strip():
            raise ValueError("paper_title cannot be blank")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        has_content = bool(self.raw_description and self.raw_description.strip())
        if self.status == ComparisonCellStatus.EVIDENCE_BACKED:
            if not has_content or not self.normalized_value or not self.evidence_links:
                raise ValueError("Evidence-backed ComparisonCell requires value and evidence")
            if not self.directly_comparable or self.non_comparable_reason is not None:
                raise ValueError("Evidence-backed ComparisonCell must be directly comparable")
        else:
            if has_content or self.normalized_value is not None or self.evidence_links:
                raise ValueError("Insufficient ComparisonCell cannot contain unsupported content")
            if self.directly_comparable or not self.non_comparable_reason:
                raise ValueError("Insufficient ComparisonCell requires a refusal reason")


@dataclass(frozen=True, slots=True)
class ComparisonDimension:
    name: ComparisonDimensionName
    cells: tuple[ComparisonCell, ...]
    directly_comparable: bool
    non_comparable_reason: str | None = None

    def __post_init__(self) -> None:
        if any(cell.dimension != self.name for cell in self.cells):
            raise ValueError("ComparisonDimension cells must use the same dimension")
        if self.directly_comparable:
            if sum(cell.status == ComparisonCellStatus.EVIDENCE_BACKED for cell in self.cells) < 2:
                raise ValueError("A comparable dimension requires evidence from at least two papers")
            if self.non_comparable_reason is not None:
                raise ValueError("Comparable dimension cannot contain a refusal reason")
        elif not self.non_comparable_reason:
            raise ValueError("Non-comparable dimension requires a reason")


@dataclass(frozen=True, slots=True)
class PaperComparisonResult:
    project_id: UUID
    paper_ids: tuple[UUID, ...]
    status: ComparisonStatus
    dimensions: tuple[ComparisonDimension, ...]
    derivation_method: str
    generator_version: str
    schema_version: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.derivation_method,
                self.generator_version,
                self.schema_version,
            )
        ):
            raise ValueError("Comparison derivation versions cannot be blank")
        if len(self.paper_ids) < 2:
            raise ValueError("Paper comparison requires at least two papers")
        if len(self.paper_ids) != len(set(self.paper_ids)):
            raise ValueError("Paper comparison paper_ids must be unique")
        expected = set(self.paper_ids)
        for dimension in self.dimensions:
            if {cell.paper_id for cell in dimension.cells} != expected:
                raise ValueError("Every comparison dimension must include every paper")
        comparable = any(dimension.directly_comparable for dimension in self.dimensions)
        if self.status == ComparisonStatus.INSUFFICIENT_EVIDENCE:
            if comparable or not self.reason:
                raise ValueError("Insufficient comparison requires a refusal reason")
        elif not comparable:
            raise ValueError("Complete or partial comparison requires comparable evidence")
