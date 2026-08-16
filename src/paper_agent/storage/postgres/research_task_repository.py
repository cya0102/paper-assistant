"""Project-scoped PostgreSQL ResearchTask/WorkUnit repository."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.research_tasks.domain import (
    ResearchTask,
    ResearchTaskStatus,
    ResearchTaskType,
    TaskBudget,
    WorkUnit,
    WorkUnitStatus,
)
from paper_agent.storage.postgres.models import ResearchTaskRow, WorkUnitRow


def _task_from_row(row: ResearchTaskRow) -> ResearchTask:
    return ResearchTask(
        task_id=row.task_id,
        project_id=row.project_id,
        user_id=row.user_id,
        session_id=row.session_id,
        research_question=row.research_question,
        task_type=ResearchTaskType(row.task_type),
        status=ResearchTaskStatus(row.status),
        plan=tuple(str(item) for item in row.plan_json),
        budget=TaskBudget(max_workers=row.max_workers),
        generation_key=row.generation_key,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _unit_from_row(row: WorkUnitRow) -> WorkUnit:
    return WorkUnit(
        work_unit_id=row.work_unit_id,
        task_id=row.task_id,
        project_id=row.project_id,
        work_type=row.work_type,
        objective=row.objective,
        paper_ids=tuple(UUID(str(item)) for item in row.paper_ids_json),
        input_artifact_ids=tuple(
            UUID(str(item)) for item in row.input_artifact_ids_json
        ),
        dependency_ids=tuple(UUID(str(item)) for item in row.dependency_ids_json),
        requested_worker=row.requested_worker,
        allowed_tools=tuple(str(item) for item in row.allowed_tools_json),
        output_schema=dict(row.output_schema_json) if row.output_schema_json else None,
        token_budget=row.token_budget,
        tool_call_budget=row.tool_call_budget,
        timeout_seconds=row.timeout_seconds,
        status=WorkUnitStatus(row.status),
        attempt_count=row.attempt_count,
        output_artifact_id=row.output_artifact_id,
        error=row.error,
        generation_key=row.generation_key,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SqlAlchemyResearchTaskRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save_task(self, task: ResearchTask) -> ResearchTask:
        with self._session_factory.begin() as session:
            row = session.scalar(
                select(ResearchTaskRow).where(
                    ResearchTaskRow.project_id == task.project_id,
                    ResearchTaskRow.task_id == task.task_id,
                )
            )
            if row is None:
                statement = insert(ResearchTaskRow).values(
                    task_id=task.task_id,
                    project_id=task.project_id,
                    user_id=task.user_id,
                    session_id=task.session_id,
                    research_question=task.research_question,
                    task_type=task.task_type.value,
                    status=task.status.value,
                    plan_json=list(task.plan),
                    max_workers=task.budget.max_workers,
                    generation_key=task.generation_key,
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                )
                statement = statement.on_conflict_do_nothing(
                    index_elements=["task_id"]
                )
                session.execute(statement)
                row = session.get(ResearchTaskRow, task.task_id)
            else:
                row.status = task.status.value
                row.updated_at = datetime.now(UTC)
                session.flush()
            if row is None:
                raise LookupError("ResearchTask insert did not persist")
            return _task_from_row(row)

    def get_task(
        self, project_id: UUID, task_id: UUID
    ) -> ResearchTask | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(ResearchTaskRow).where(
                    ResearchTaskRow.project_id == project_id,
                    ResearchTaskRow.task_id == task_id,
                )
            )
            return _task_from_row(row) if row is not None else None

    def find_task_by_generation_key(
        self, project_id: UUID, generation_key: str
    ) -> ResearchTask | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(ResearchTaskRow).where(
                    ResearchTaskRow.project_id == project_id,
                    ResearchTaskRow.generation_key == generation_key,
                )
            )
            return _task_from_row(row) if row is not None else None

    def save_work_unit(self, unit: WorkUnit) -> WorkUnit:
        with self._session_factory.begin() as session:
            existing = session.get(WorkUnitRow, unit.work_unit_id)
            if existing is not None:
                return _unit_from_row(existing)
            statement = insert(WorkUnitRow).values(
                work_unit_id=unit.work_unit_id,
                task_id=unit.task_id,
                project_id=unit.project_id,
                work_type=unit.work_type,
                objective=unit.objective,
                paper_ids_json=[str(item) for item in unit.paper_ids],
                input_artifact_ids_json=[str(item) for item in unit.input_artifact_ids],
                dependency_ids_json=[str(item) for item in unit.dependency_ids],
                requested_worker=unit.requested_worker,
                allowed_tools_json=list(unit.allowed_tools),
                output_schema_json=unit.output_schema,
                token_budget=unit.token_budget,
                tool_call_budget=unit.tool_call_budget,
                timeout_seconds=unit.timeout_seconds,
                status=unit.status.value,
                attempt_count=unit.attempt_count,
                output_artifact_id=unit.output_artifact_id,
                error=unit.error,
                generation_key=unit.generation_key,
                created_at=unit.created_at,
                updated_at=unit.updated_at,
            )
            statement = statement.on_conflict_do_nothing(
                index_elements=["task_id", "generation_key"]
            )
            session.execute(statement)
            row = session.get(WorkUnitRow, unit.work_unit_id)
            if row is None:
                row = session.scalar(
                    select(WorkUnitRow).where(
                        WorkUnitRow.task_id == unit.task_id,
                        WorkUnitRow.generation_key == unit.generation_key,
                    )
                )
            if row is None:
                raise LookupError("WorkUnit insert did not persist")
            return _unit_from_row(row)

    def get_work_unit(
        self, project_id: UUID, work_unit_id: UUID
    ) -> WorkUnit | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(WorkUnitRow).where(
                    WorkUnitRow.project_id == project_id,
                    WorkUnitRow.work_unit_id == work_unit_id,
                )
            )
            return _unit_from_row(row) if row is not None else None

    def list_work_units(
        self, project_id: UUID, task_id: UUID
    ) -> tuple[WorkUnit, ...]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(WorkUnitRow)
                .where(
                    WorkUnitRow.project_id == project_id,
                    WorkUnitRow.task_id == task_id,
                )
                .order_by(WorkUnitRow.created_at)
            )
            return tuple(_unit_from_row(row) for row in rows)

    def update_work_unit(
        self,
        project_id: UUID,
        work_unit_id: UUID,
        *,
        status: str | None = None,
        attempt_count: int | None = None,
        output_artifact_id: UUID | None = None,
        error: str | None = None,
    ) -> WorkUnit:
        with self._session_factory.begin() as session:
            row = session.scalar(
                select(WorkUnitRow).where(
                    WorkUnitRow.project_id == project_id,
                    WorkUnitRow.work_unit_id == work_unit_id,
                )
            )
            if row is None:
                raise LookupError("WorkUnit not found in project")
            if status is not None:
                row.status = status
            if attempt_count is not None:
                row.attempt_count = attempt_count
            if output_artifact_id is not None:
                row.output_artifact_id = output_artifact_id
            if error is not None:
                row.error = error
            row.updated_at = datetime.now(UTC)
            session.flush()
            return _unit_from_row(row)
