"""Dependency boundaries for ResearchTask persistence."""

from typing import Protocol
from uuid import UUID

from paper_agent.research_tasks.domain import ResearchTask, WorkUnit


class ResearchTaskRepository(Protocol):
    def save_task(self, task: ResearchTask) -> ResearchTask: ...

    def get_task(
        self, project_id: UUID, task_id: UUID
    ) -> ResearchTask | None: ...

    def find_task_by_generation_key(
        self, project_id: UUID, generation_key: str
    ) -> ResearchTask | None: ...

    def save_work_unit(self, unit: WorkUnit) -> WorkUnit: ...

    def get_work_unit(
        self, project_id: UUID, work_unit_id: UUID
    ) -> WorkUnit | None: ...

    def list_work_units(
        self, project_id: UUID, task_id: UUID
    ) -> tuple[WorkUnit, ...]: ...

    def update_work_unit(
        self,
        project_id: UUID,
        work_unit_id: UUID,
        *,
        status: str | None = None,
        attempt_count: int | None = None,
        output_artifact_id: UUID | None = None,
        error: str | None = None,
    ) -> WorkUnit: ...
