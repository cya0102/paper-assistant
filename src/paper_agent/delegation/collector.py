"""ResultCollector: merge Worker results into a compact main-Agent view.

The main Agent never receives full worker trajectories: only merged summaries,
artifact references, a deduplicated Citation Manifest, unresolved questions,
and failed work units.  Summaries and manifests are read from the stored Worker
Artifacts so nothing extra has to travel through the scheduler.
"""

from dataclasses import dataclass
from uuid import UUID

from paper_agent.artifacts.ports import ArtifactServicePort
from paper_agent.domain.artifact import (
    ArtifactDescriptor,
    ArtifactReference,
    ArtifactSelector,
    CitationReference,
)
from paper_agent.domain.errors import PaperAgentError
from paper_agent.research_tasks.domain import (
    ResearchTask,
    WorkUnit,
    WorkUnitStatus,
)


@dataclass(frozen=True, slots=True)
class CollectedResearchTask:
    task_id: UUID
    project_id: UUID
    status: str
    summary: str
    artifact_refs: tuple[ArtifactReference, ...]
    citation_manifest: tuple[CitationReference, ...]
    unresolved_questions: tuple[str, ...]
    failed_work_units: tuple[dict[str, str], ...]


class ResultCollector:
    def __init__(self, artifacts: ArtifactServicePort) -> None:
        self._artifacts = artifacts

    def collect(
        self, *, task: ResearchTask, units: tuple[WorkUnit, ...]
    ) -> CollectedResearchTask:
        summaries: list[str] = []
        refs: list[ArtifactReference] = []
        manifest: dict[str, CitationReference] = {}
        unresolved: list[str] = []
        failed: list[dict[str, str]] = []
        for unit in units:
            if unit.status == WorkUnitStatus.COMPLETED:
                descriptor = self._find_descriptor(task.project_id, unit)
                if descriptor is not None:
                    summaries.append(f"{unit.work_type}: {descriptor.summary}")
                    refs.append(
                        ArtifactReference.from_descriptor(
                            descriptor,
                            available_views=("default", "result", "evidence", "report", "full"),
                        )
                    )
                    for citation in descriptor.citation_manifest:
                        manifest.setdefault(citation.citation_label, citation)
                    unresolved.extend(
                        self._read_unresolved(task.project_id, descriptor)
                    )
                else:
                    summaries.append(f"{unit.work_type}: 完成")
                continue
            if unit.status in {WorkUnitStatus.FAILED, WorkUnitStatus.SKIPPED}:
                failed.append(
                    {
                        "work_unit_id": str(unit.work_unit_id),
                        "work_type": unit.work_type,
                        "status": unit.status.value,
                        "error": unit.error or "unknown error",
                    }
                )
        return CollectedResearchTask(
            task_id=task.task_id,
            project_id=task.project_id,
            status=task.status.value,
            summary="；".join(summaries) or "没有完成的 WorkUnit",
            artifact_refs=tuple(refs),
            citation_manifest=tuple(manifest.values()),
            unresolved_questions=tuple(dict.fromkeys(unresolved)),
            failed_work_units=tuple(failed),
        )

    def _find_descriptor(
        self, project_id: UUID, unit: WorkUnit
    ) -> ArtifactDescriptor | None:
        if unit.output_artifact_id is None:
            return None
        descriptors = self._artifacts.search(
            project_id, work_unit_id=unit.work_unit_id, limit=1
        )
        for descriptor in descriptors:
            if descriptor.artifact_id == unit.output_artifact_id:
                return descriptor
        return None

    def _read_unresolved(
        self, project_id: UUID, descriptor: ArtifactDescriptor
    ) -> tuple[str, ...]:
        try:
            slice_ = self._artifacts.read_slice(
                ArtifactSelector(
                    artifact_id=descriptor.artifact_id,
                    project_id=project_id,
                    view="default",
                    max_tokens=4000,
                )
            )
        except PaperAgentError:
            return ()
        raw = slice_.content.get("unresolved_questions", [])
        if not isinstance(raw, list):
            return ()
        return tuple(str(item) for item in raw if str(item).strip())
