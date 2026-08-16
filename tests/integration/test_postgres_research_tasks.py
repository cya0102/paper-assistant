import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.research_tasks.domain import (
    ResearchTask,
    ResearchTaskStatus,
    ResearchTaskType,
    TaskBudget,
    WorkUnit,
    WorkUnitStatus,
    task_generation_key,
    work_unit_generation_key,
)
from paper_agent.storage.postgres.models import ProjectRow
from paper_agent.storage.postgres.research_task_repository import (
    SqlAlchemyResearchTaskRepository,
)


DATABASE_URL = os.getenv("PAPER_AGENT_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="PAPER_AGENT_TEST_DATABASE_URL is required for real PostgreSQL tests",
)


@pytest.fixture()
def repo():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    with factory.begin() as session:
        project = ProjectRow(name="task-test", root_path=f"/tmp/task-{uuid4()}")
        session.add(project)
        session.flush()
        project_id = project.project_id
    repository = SqlAlchemyResearchTaskRepository(factory)
    yield repository, project_id
    with factory.begin() as session:
        session.query(ProjectRow).filter(ProjectRow.project_id == project_id).delete()


def _task(project_id, user_id) -> ResearchTask:
    return ResearchTask(
        task_id=uuid4(),
        project_id=project_id,
        user_id=user_id,
        research_question="compare methods",
        task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
        status=ResearchTaskStatus.CREATED,
        plan=("method", "verification"),
        budget=TaskBudget(max_workers=3),
        generation_key=task_generation_key(
            project_id=project_id,
            user_id=user_id,
            research_question="compare methods",
            task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
            plan=("method", "verification"),
        ),
    )


def _unit(task: ResearchTask) -> WorkUnit:
    objective = "extract method"
    schema = {"type": "object", "properties": {}, "required": []}
    return WorkUnit(
        work_unit_id=uuid4(),
        task_id=task.task_id,
        project_id=task.project_id,
        work_type="method",
        objective=objective,
        requested_worker="paper_analyzer",
        status=WorkUnitStatus.PENDING,
        generation_key=work_unit_generation_key(
            task_id=task.task_id,
            work_type="method",
            objective=objective,
            paper_ids=(uuid4(),),
            input_artifact_ids=(),
            requested_worker="paper_analyzer",
            output_schema=schema,
        ),
        token_budget=4000,
        tool_call_budget=6,
        timeout_seconds=180,
        paper_ids=(uuid4(),),
        allowed_tools=("search_knowledge",),
        output_schema=schema,
    )


def test_task_and_unit_persistence(repo):
    repository, project_id = repo
    user_id = uuid4()
    task = repository.save_task(_task(project_id, user_id))
    loaded = repository.get_task(project_id, task.task_id)
    assert loaded is not None
    assert loaded.plan == ("method", "verification")
    assert loaded.status == ResearchTaskStatus.CREATED

    unit = repository.save_work_unit(_unit(task))
    assert unit.status == WorkUnitStatus.PENDING
    units = repository.list_work_units(project_id, task.task_id)
    assert len(units) == 1

    updated = repository.update_work_unit(
        project_id,
        unit.work_unit_id,
        status=WorkUnitStatus.COMPLETED.value,
        output_artifact_id=uuid4(),
    )
    assert updated.status == WorkUnitStatus.COMPLETED
    assert updated.output_artifact_id is not None


def test_work_unit_generation_key_dedup(repo):
    repository, project_id = repo
    task = repository.save_task(_task(project_id, uuid4()))
    unit = _unit(task)
    # saving the identical WorkUnit twice must not create a duplicate row
    first = repository.save_work_unit(unit)
    second = repository.save_work_unit(unit)
    assert first.work_unit_id == second.work_unit_id
    units = repository.list_work_units(project_id, task.task_id)
    assert len(units) == 1
    assert units[0].work_unit_id == unit.work_unit_id


def test_task_generation_key_dedup_survives_distinct_task_ids(repo):
    repository, project_id = repo
    user_id = uuid4()
    first_input = _task(project_id, user_id)
    second_input = _task(project_id, user_id)
    assert first_input.task_id != second_input.task_id
    assert first_input.generation_key == second_input.generation_key

    first = repository.save_task(first_input)
    second = repository.save_task(second_input)

    assert second.task_id == first.task_id
    assert repository.find_task_by_generation_key(
        project_id, first.generation_key
    ) == first


def test_cross_project_task_not_found(repo):
    repository, project_id = repo
    task = repository.save_task(_task(project_id, uuid4()))
    assert repository.get_task(uuid4(), task.task_id) is None
