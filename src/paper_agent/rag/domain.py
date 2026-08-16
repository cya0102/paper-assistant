"""Domain contracts for Retrieve-Offload-Delegate RAG.

The main Agent receives only these compact objects.  Retrieved chunk text lives
inside a ``retrieved_evidence`` Artifact and is never carried by a result or
trace event in this module.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from paper_agent.domain.artifact import (
    ArtifactReference,
    CitationReference,
    artifact_ref_to_dict,
    citation_to_dict,
)


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be blank")


class AnalystRelevance(StrEnum):
    RELEVANT = "relevant"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"


class RagResultStatus(StrEnum):
    SUPPORTED = "supported"
    NO_EVIDENCE = "no_evidence"
    INSUFFICIENT = "insufficient"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RagConfig:
    max_evidence: int = 6
    max_per_paper: int = 2
    max_workers: int = 3
    max_rounds: int = 2
    worker_token_budget: int = 1200
    worker_tool_call_budget: int = 2
    worker_timeout_seconds: int = 90

    def __post_init__(self) -> None:
        if not 1 <= self.max_evidence <= 20:
            raise ValueError("max_evidence must be between 1 and 20")
        if not 1 <= self.max_per_paper <= self.max_evidence:
            raise ValueError("max_per_paper must be between 1 and max_evidence")
        if not 1 <= self.max_workers <= 5:
            raise ValueError("max_workers must be between 1 and 5")
        if not 1 <= self.max_rounds <= 2:
            raise ValueError("max_rounds must be between 1 and 2")
        if self.worker_token_budget < 1:
            raise ValueError("worker_token_budget must be positive")
        if self.worker_tool_call_budget < 1:
            raise ValueError("worker_tool_call_budget must be positive")
        if self.worker_timeout_seconds < 1:
            raise ValueError("worker_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class RetrievedEvidenceArtifact:
    """Compact handle for one offloaded Evidence chunk."""

    artifact_ref: ArtifactReference
    citation: CitationReference
    paper_id: UUID
    chunk_id: UUID
    relevance: float
    round_index: int

    def __post_init__(self) -> None:
        if self.citation.paper_id != self.paper_id:
            raise ValueError("citation paper must match retrieved evidence")
        if self.citation.chunk_id != self.chunk_id:
            raise ValueError("citation chunk must match retrieved evidence")
        if not 0 <= self.relevance <= 1:
            raise ValueError("relevance must be between 0 and 1")
        if self.round_index < 1:
            raise ValueError("round_index must be positive")


@dataclass(frozen=True, slots=True)
class AnalystClaim:
    text: str
    citations: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.text, "claim text")
        if not self.citations:
            raise ValueError("analyst claim requires at least one citation")
        if any(not value.strip() for value in self.citations):
            raise ValueError("claim citations cannot contain blank values")
        if len(self.citations) != len(set(self.citations)):
            raise ValueError("claim citations must be unique")


@dataclass(frozen=True, slots=True)
class AnalystReport:
    work_unit_id: UUID
    evidence_artifact_id: UUID
    relevance: AnalystRelevance
    summary: str
    claims: tuple[AnalystClaim, ...]
    unresolved_questions: tuple[str, ...] = ()
    worker_artifact_ref: ArtifactReference | None = None

    def __post_init__(self) -> None:
        _require_text(self.summary, "analyst summary")
        if self.relevance == AnalystRelevance.RELEVANT and not self.claims:
            raise ValueError("relevant report requires evidence-backed claims")
        if any(not value.strip() for value in self.unresolved_questions):
            raise ValueError("unresolved_questions cannot contain blanks")


@dataclass(frozen=True, slots=True)
class RagFailure:
    work_unit_id: UUID
    error: str

    def __post_init__(self) -> None:
        _require_text(self.error, "failure error")


@dataclass(frozen=True, slots=True)
class RagCollection:
    reports: tuple[AnalystReport, ...]
    citation_manifest: tuple[CitationReference, ...]
    failures: tuple[RagFailure, ...] = ()

    @property
    def sufficient(self) -> bool:
        available = {item.citation_label for item in self.citation_manifest}
        return any(
            report.relevance == AnalystRelevance.RELEVANT
            and any(set(claim.citations) <= available for claim in report.claims)
            for report in self.reports
        )

    @property
    def all_workers_failed(self) -> bool:
        return bool(self.failures) and not self.reports


@dataclass(frozen=True, slots=True)
class RetrieveOffloadDelegateResult:
    task_id: UUID
    status: RagResultStatus
    original_query: str
    final_query: str
    rounds_executed: int
    evidence_artifacts: tuple[RetrievedEvidenceArtifact, ...]
    reports: tuple[AnalystReport, ...]
    citation_manifest: tuple[CitationReference, ...]
    failures: tuple[RagFailure, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.original_query, "original_query")
        _require_text(self.final_query, "final_query")
        if self.rounds_executed < 1:
            raise ValueError("rounds_executed must be positive")
        if self.status == RagResultStatus.SUPPORTED and not self.citation_manifest:
            raise ValueError("supported RAG result requires citations")
        if self.status != RagResultStatus.SUPPORTED and self.citation_manifest:
            raise ValueError("non-supported RAG result cannot expose citations")

    @property
    def has_sufficient_evidence(self) -> bool:
        return self.status == RagResultStatus.SUPPORTED

    def to_model_payload(self) -> dict[str, Any]:
        """Serialize the strict main-Agent view without retrieved chunk text."""
        claims: list[dict[str, Any]] = []
        for report in self.reports:
            if report.relevance != AnalystRelevance.RELEVANT:
                continue
            claims.extend(
                {"text": claim.text, "citations": list(claim.citations)}
                for claim in report.claims
            )
        return {
            "task_id": str(self.task_id),
            "status": self.status.value,
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "query": self.original_query,
            "final_query": self.final_query,
            "rounds_executed": self.rounds_executed,
            "summary": (
                "；".join(
                    report.summary
                    for report in self.reports
                    if report.relevance == AnalystRelevance.RELEVANT
                )
                or self.reason
                or "no_evidence"
            ),
            "reason": self.reason,
            "claims": claims,
            "reports": [
                {
                    "work_unit_id": str(report.work_unit_id),
                    "relevance": report.relevance.value,
                    "summary": report.summary,
                    "claims": [
                        {"text": claim.text, "citations": list(claim.citations)}
                        for claim in report.claims
                    ],
                    "unresolved_questions": list(report.unresolved_questions),
                    "worker_artifact_ref": artifact_ref_to_dict(
                        report.worker_artifact_ref
                    ),
                }
                for report in self.reports
            ],
            "evidence_artifacts": [
                {
                    "artifact_id": str(item.artifact_ref.artifact_id),
                    "paper_title": item.citation.paper_title,
                    "section_path": item.citation.section_path,
                    "pages": [item.citation.page_start, item.citation.page_end],
                    "citation": item.citation.citation_label,
                    "round": item.round_index,
                }
                for item in self.evidence_artifacts
            ],
            "citation_manifest": [
                citation_to_dict(item) for item in self.citation_manifest
            ],
            "unresolved_questions": list(
                dict.fromkeys(
                    question
                    for report in self.reports
                    for question in report.unresolved_questions
                )
            ),
            "failed_work_units": [
                {
                    "work_unit_id": str(item.work_unit_id),
                    "error": item.error,
                }
                for item in self.failures
            ],
        }


@dataclass(frozen=True, slots=True)
class RagTraceEvent:
    event: str
    task_id: UUID | None = None
    round_index: int | None = None
    work_unit_id: UUID | None = None
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.event, "trace event")
        if self.round_index is not None and self.round_index < 1:
            raise ValueError("trace round_index must be positive")
        # Trace details must never become an accidental evidence side channel.
        forbidden = {"text", "content", "payload", "evidence_text"}
        if forbidden & set(self.details):
            raise ValueError("trace details cannot contain evidence content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "task_id": str(self.task_id) if self.task_id else None,
            "round": self.round_index,
            "work_unit_id": str(self.work_unit_id) if self.work_unit_id else None,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
        }
