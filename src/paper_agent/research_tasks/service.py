"""ResearchTaskService: create, run, and collect bounded research delegations.

The service is the single entry point used by the delegate_research and
collect_research_task tools.  It enforces project scoping, applies the
DelegationPolicy, persists the task and its WorkUnits, runs the synchronous
scheduler, and returns only compact summaries to the main Agent.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from paper_agent.artifacts.ports import ArtifactServicePort
from paper_agent.delegation.collector import ResultCollector
from paper_agent.delegation.policy import DelegationDecision, DelegationPolicy
from paper_agent.delegation.scheduler import Scheduler
from paper_agent.research_tasks.domain import (
    ResearchTask,
    ResearchTaskStatus,
    ResearchTaskType,
    TaskBudget,
    WorkUnit,
    WorkUnitStatus,
    task_generation_key,
)
from paper_agent.research_tasks.planner import ResearchPlanner, default_plan_for
from paper_agent.research_tasks.ports import ResearchTaskRepository


class DelegationRefusedError(ValueError):
    """Raised when the policy routes a request back to the single Agent."""


def infer_task_type(workstreams: tuple[str, ...]) -> ResearchTaskType:
    joined = set(workstreams)
    if joined & {"source_domain_search", "mechanism_abstraction", "assumption_compatibility", "novelty_check"}:
        return ResearchTaskType.CROSS_DOMAIN_EXPLORATION
    if joined & {"retrieval", "screening", "classification", "relation_analysis", "synthesis"}:
        return ResearchTaskType.LITERATURE_SURVEY
    return ResearchTaskType.MULTI_PAPER_COMPARISON


class ResearchTaskService:
    def __init__(
        self,
        *,
        repository: ResearchTaskRepository,
        planner: ResearchPlanner,
        policy: DelegationPolicy,
        scheduler: Scheduler,
        collector: ResultCollector,
        artifacts: ArtifactServicePort,
    ) -> None:
        self._repository = repository
        self._planner = planner
        self._policy = policy
        self._scheduler = scheduler
        self._collector = collector
        self._artifacts = artifacts

    def delegate(
        self,
        *,
        project_id: UUID,
        user_id: UUID,
        session_id: UUID | None,
        objective: str,
        paper_ids: tuple[UUID, ...],
        requested_workstreams: tuple[str, ...] = (),
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        decision = self._policy.decide(
            paper_ids=paper_ids,
            requested_workstreams=requested_workstreams,
            max_workers=max_workers,
        )
        if not decision.delegate:
            raise DelegationRefusedError(decision.reason)
        task = self._create_task(
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
            objective=objective,
            decision=decision,
        )
        units = self._planner.plan(
            task=task,
            paper_ids=paper_ids,
            workstreams=decision.workstreams,
        )
        for unit in units:
            self._repository.save_work_unit(unit)
        updated_task, updated_units = self._scheduler.run(
            task=task,
            units=units,
            repository=self._repository,
            user_id=user_id,
        )
        return self._task_summary(updated_task, updated_units)

    def collect(self, *, project_id: UUID, task_id: UUID) -> dict[str, Any]:
        task = self._repository.get_task(project_id, task_id)
        if task is None:
            raise LookupError("ResearchTask not found in project")
        units = self._repository.list_work_units(project_id, task_id)
        collected = self._collector.collect(task=task, units=units)
        from paper_agent.domain.artifact import (
            artifact_ref_to_dict,
            citation_to_dict,
        )

        return {
            "task_id": str(collected.task_id),
            "status": collected.status,
            "summary": collected.summary,
            "artifact_refs": [
                artifact_ref_to_dict(ref) for ref in collected.artifact_refs
            ],
            "citation_manifest": [
                citation_to_dict(item) for item in collected.citation_manifest
            ],
            "unresolved_questions": list(collected.unresolved_questions),
            "failed_work_units": list(collected.failed_work_units),
        }

    def _create_task(
        self,
        *,
        project_id: UUID,
        user_id: UUID,
        session_id: UUID | None,
        objective: str,
        decision: DelegationDecision,
    ) -> ResearchTask:
        task_type = infer_task_type(decision.workstreams)
        plan = decision.workstreams or default_plan_for(task_type)
        generation_key = task_generation_key(
            project_id=project_id,
            user_id=user_id,
            research_question=objective,
            task_type=task_type,
            plan=plan,
        )
        existing = self._repository.find_task_by_generation_key(
            project_id, generation_key
        )
        if existing is not None:
            if existing.status == ResearchTaskStatus.COMPLETED:
                return existing
            # A previously created task with the same generation key is reused
            # so retries never duplicate work units.
            return existing
        task = ResearchTask(
            task_id=uuid4(),
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
            research_question=objective,
            task_type=task_type,
            status=ResearchTaskStatus.CREATED,
            plan=plan,
            budget=TaskBudget(max_workers=decision.max_workers),
            generation_key=generation_key,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        return self._repository.save_task(task)

    @staticmethod
    @staticmethod
    def _task_summary(
        task: ResearchTask, units: tuple[WorkUnit, ...]
    ) -> dict[str, Any]:
        completed = sum(
            unit.status == WorkUnitStatus.COMPLETED for unit in units
        )
        return {
            "task_id": str(task.task_id),
            "status": task.status.value,
            "work_unit_ids": [str(unit.work_unit_id) for unit in units],
            "assigned_workers": [unit.requested_worker for unit in units],
            "progress": f"{completed}/{len(units)}",
        }
