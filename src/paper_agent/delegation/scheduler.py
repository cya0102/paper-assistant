"""Scheduler: synchronous, single-layer, dependency-aware WorkUnit execution.

v1 constraints: max one retry per WorkUnit, no worker can create workers,
units run in topological order, and already-completed units (identified by
their stable generation_key) are never re-executed.
"""

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from paper_agent.delegation.runner import WorkerRunner
from paper_agent.research_tasks.domain import (
    ResearchTask,
    ResearchTaskStatus,
    WorkUnit,
    WorkUnitStatus,
)
from paper_agent.research_tasks.ports import ResearchTaskRepository


class Scheduler:
    def __init__(self, runner: WorkerRunner, *, max_attempts: int = 2) -> None:
        self._runner = runner
        self._max_attempts = max_attempts

    def run(
        self,
        *,
        task: ResearchTask,
        units: tuple[WorkUnit, ...],
        repository: ResearchTaskRepository,
        user_id: UUID,
    ) -> tuple[ResearchTask, tuple[WorkUnit, ...]]:
        """Execute all runnable units and return (updated_task, updated_units)."""
        running = task.status == ResearchTaskStatus.CREATED or task.status == ResearchTaskStatus.PLANNED
        if running:
            task = replace(task, status=ResearchTaskStatus.RUNNING, updated_at=datetime.now(UTC))
            repository.save_task(task)
        completed: dict[UUID, WorkUnit] = {}
        updated: list[WorkUnit] = []
        for unit in units:
            if unit.status == WorkUnitStatus.COMPLETED:
                completed[unit.work_unit_id] = unit
                updated.append(unit)
                continue
            if not all(dep in completed for dep in unit.dependency_ids):
                skipped = replace(unit, status=WorkUnitStatus.SKIPPED, updated_at=datetime.now(UTC))
                repository.update_work_unit(
                    task.project_id, unit.work_unit_id, status=WorkUnitStatus.SKIPPED.value
                )
                updated.append(skipped)
                continue
            finished = self._run_with_retry(unit, repository, user_id)
            if finished.status == WorkUnitStatus.COMPLETED:
                completed[finished.work_unit_id] = finished
            updated.append(finished)
        result_task = self._finalize_task(task, tuple(updated), repository)
        return result_task, tuple(updated)

    def _run_with_retry(
        self,
        unit: WorkUnit,
        repository: ResearchTaskRepository,
        user_id: UUID,
    ) -> WorkUnit:
        attempts = 0
        while attempts < self._max_attempts:
            attempts += 1
            running = replace(
                unit,
                status=WorkUnitStatus.RUNNING,
                attempt_count=unit.attempt_count + attempts,
                updated_at=datetime.now(UTC),
            )
            repository.update_work_unit(
                running.project_id,
                running.work_unit_id,
                status=WorkUnitStatus.RUNNING.value,
                attempt_count=running.attempt_count,
            )
            result = self._runner.run(running, user_id=user_id)
            if result.succeeded:
                finished = replace(
                    running,
                    status=WorkUnitStatus.COMPLETED,
                    output_artifact_id=(
                        result.artifact_ref.artifact_id
                        if result.artifact_ref is not None
                        else None
                    ),
                    error=None,
                    updated_at=datetime.now(UTC),
                )
                repository.update_work_unit(
                    finished.project_id,
                    finished.work_unit_id,
                    status=WorkUnitStatus.COMPLETED.value,
                    output_artifact_id=finished.output_artifact_id,
                    error=None,
                )
                return finished
            repository.update_work_unit(
                running.project_id,
                running.work_unit_id,
                error=result.error,
            )
        failed = replace(
            unit,
            status=WorkUnitStatus.FAILED,
            attempt_count=unit.attempt_count + self._max_attempts,
            error=result.error,
            updated_at=datetime.now(UTC),
        )
        repository.update_work_unit(
            failed.project_id,
            failed.work_unit_id,
            status=WorkUnitStatus.FAILED.value,
            attempt_count=failed.attempt_count,
            error=failed.error,
        )
        return failed

    @staticmethod
    def _finalize_task(
        task: ResearchTask,
        units: tuple[WorkUnit, ...],
        repository: ResearchTaskRepository,
    ) -> ResearchTask:
        counts: dict[WorkUnitStatus, int] = {}
        for unit in units:
            counts[unit.status] = counts.get(unit.status, 0) + 1
        if counts.get(WorkUnitStatus.FAILED, 0) == len(units):
            status = ResearchTaskStatus.FAILED
        elif counts.get(WorkUnitStatus.COMPLETED, 0) == len(units):
            status = ResearchTaskStatus.COMPLETED
        else:
            status = ResearchTaskStatus.PARTIALLY_COMPLETED
        updated = replace(
            task,
            status=status,
            updated_at=datetime.now(UTC),
        )
        repository.save_task(updated)
        return updated
