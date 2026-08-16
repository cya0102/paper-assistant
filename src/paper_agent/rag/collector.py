"""Collect compact chunk analyst reports and their authoritative citations."""

from typing import Any
from uuid import UUID

from paper_agent.artifacts.ports import ArtifactServicePort
from paper_agent.domain.artifact import ArtifactReference, ArtifactSelector
from paper_agent.domain.errors import PaperAgentError
from paper_agent.rag.domain import (
    AnalystClaim,
    AnalystRelevance,
    AnalystReport,
    RagCollection,
    RagFailure,
)
from paper_agent.research_tasks.domain import WorkUnit, WorkUnitStatus


class RodResultCollector:
    def __init__(self, artifacts: ArtifactServicePort) -> None:
        self._artifacts = artifacts

    def collect(
        self,
        *,
        project_id: UUID,
        units: tuple[WorkUnit, ...],
    ) -> RagCollection:
        reports: list[AnalystReport] = []
        failures: list[RagFailure] = []
        manifests: dict[str, Any] = {}
        used_labels: set[str] = set()
        for unit in units:
            if unit.status != WorkUnitStatus.COMPLETED:
                if unit.status in {WorkUnitStatus.FAILED, WorkUnitStatus.SKIPPED}:
                    failures.append(
                        RagFailure(
                            work_unit_id=unit.work_unit_id,
                            error=unit.error or unit.status.value,
                        )
                    )
                continue
            if unit.output_artifact_id is None or len(unit.input_artifact_ids) != 1:
                failures.append(
                    RagFailure(
                        work_unit_id=unit.work_unit_id,
                        error="completed chunk analysis has invalid Artifact provenance",
                    )
                )
                continue
            descriptors = self._artifacts.search(
                project_id, work_unit_id=unit.work_unit_id, limit=2
            )
            descriptor = next(
                (
                    item
                    for item in descriptors
                    if item.artifact_id == unit.output_artifact_id
                ),
                None,
            )
            if descriptor is None:
                failures.append(
                    RagFailure(
                        work_unit_id=unit.work_unit_id,
                        error="worker_result Artifact is missing",
                    )
                )
                continue
            try:
                result_slice = self._artifacts.read_slice(
                    ArtifactSelector(
                        artifact_id=descriptor.artifact_id,
                        project_id=project_id,
                        view="result",
                        max_tokens=4000,
                    )
                )
                raw = result_slice.content.get("result")
                if not isinstance(raw, dict):
                    raise ValueError("worker result is not an object")
                report = self._parse_report(
                    raw=raw,
                    unit=unit,
                    worker_ref=ArtifactReference.from_descriptor(
                        descriptor,
                        available_views=(
                            "default",
                            "result",
                            "evidence",
                            "report",
                            "full",
                        ),
                    ),
                )
            except (PaperAgentError, ValueError, KeyError) as error:
                failures.append(
                    RagFailure(work_unit_id=unit.work_unit_id, error=str(error))
                )
                continue
            reports.append(report)
            if report.relevance == AnalystRelevance.RELEVANT:
                for claim in report.claims:
                    used_labels.update(claim.citations)
            for citation in descriptor.citation_manifest:
                manifests.setdefault(citation.citation_label, citation)
        manifest = tuple(
            citation
            for label, citation in manifests.items()
            if label in used_labels
        )
        return RagCollection(
            reports=tuple(reports),
            citation_manifest=manifest,
            failures=tuple(failures),
        )

    @staticmethod
    def _parse_report(
        *, raw: dict[str, Any], unit: WorkUnit, worker_ref: ArtifactReference
    ) -> AnalystReport:
        claims: list[AnalystClaim] = []
        for claim in raw.get("claims", []):
            if not isinstance(claim, dict):
                raise ValueError("analyst claim must be an object")
            citations = claim.get("citations", [])
            if not isinstance(citations, list):
                raise ValueError("analyst claim citations must be an array")
            claims.append(
                AnalystClaim(
                    text=str(claim.get("text") or ""),
                    citations=tuple(str(item).strip("[]") for item in citations),
                )
            )
        unresolved = raw.get("unresolved_questions", [])
        if not isinstance(unresolved, list):
            raise ValueError("unresolved_questions must be an array")
        return AnalystReport(
            work_unit_id=unit.work_unit_id,
            evidence_artifact_id=unit.input_artifact_ids[0],
            relevance=AnalystRelevance(str(raw["relevance"])),
            summary=str(raw.get("summary") or ""),
            claims=tuple(claims),
            unresolved_questions=tuple(str(item) for item in unresolved),
            worker_artifact_ref=worker_ref,
        )
