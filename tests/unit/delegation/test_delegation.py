import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from paper_agent.artifacts.materializer import ToolResultMaterializer
from paper_agent.artifacts.policies import OffloadPolicy
from paper_agent.artifacts.service import ArtifactService
from paper_agent.delegation.collector import ResultCollector
from paper_agent.delegation.policy import DelegationPolicy
from paper_agent.delegation.registry import WorkerDescriptor, WorkerRegistry
from paper_agent.delegation.runner import WorkerOutputValidator, WorkerRunner
from paper_agent.delegation.scheduler import Scheduler
from paper_agent.domain.agent import ModelTurn, ToolCall, ToolResult
from paper_agent.domain.artifact import CitationReference
from paper_agent.memory import InMemoryCheckpointStore
from paper_agent.research_tasks.domain import (
    ResearchTask,
    ResearchTaskStatus,
    ResearchTaskType,
    TaskBudget,
    WorkUnit,
    WorkUnitStatus,
    task_generation_key,
)
from paper_agent.research_tasks.planner import ResearchPlanner
from paper_agent.research_tasks.service import (
    DelegationRefusedError,
    ResearchTaskService,
    infer_task_type,
)
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore
from paper_agent.workers.base import build_worker_registry
from tests.unit.artifacts.test_artifact_service import MemoryArtifactRepository


class FakeSearchService:
    def search_knowledge(self, request):
        from paper_agent.domain.enums import SearchStatus
        from paper_agent.domain.retrieval import SearchKnowledgeResult

        return SearchKnowledgeResult(
            query=request.query,
            status=SearchStatus.OK,
            resolved_papers=(),
            evidence=(),
            has_sufficient_evidence=False,
        )


class ScriptedWorkerModel:
    """Emits one tool call, then a scripted JSON answer."""

    def __init__(self, answer: dict):
        self.answer = answer
        self.received_brief = None

    def start(self, checkpoint, tools):
        self.received_brief = checkpoint.messages[-1].content
        return ModelTurn(
            response_id="w1",
            tool_calls=(ToolCall("wc1", "search_knowledge", {"query": "method"}),),
        )

    def continue_with_tools(self, checkpoint, results, tools):
        return ModelTurn(
            response_id="w2",
            output_text=json.dumps(self.answer, ensure_ascii=False),
        )


def _service(tmp_path: Path) -> ArtifactService:
    return ArtifactService(
        LocalArtifactBlobStore(tmp_path), MemoryArtifactRepository()
    )


def _runner(tmp_path: Path, answer: dict) -> tuple[WorkerRunner, ArtifactService]:
    artifacts = _service(tmp_path)
    runner = WorkerRunner(
        registry=build_worker_registry(),
        model=ScriptedWorkerModel(answer),
        checkpoints=InMemoryCheckpointStore(),
        artifacts=artifacts,
        materializer=ToolResultMaterializer(artifacts, OffloadPolicy()),
        search_service=FakeSearchService(),
    )
    return runner, artifacts


def _unit(*, worker: str = "paper_analyzer", schema: dict | None = None) -> WorkUnit:
    task_id, project_id = uuid4(), uuid4()
    from paper_agent.research_tasks.domain import work_unit_generation_key

    schema = schema or {
        "type": "object",
        "properties": {"workstream": {"type": "string"}, "findings": {"type": "array"}},
        "required": ["workstream", "findings"],
        "additionalProperties": False,
    }
    objective = "提取 method 维度事实"
    return WorkUnit(
        work_unit_id=uuid4(),
        task_id=task_id,
        project_id=project_id,
        work_type="method",
        objective=objective,
        requested_worker=worker,
        status=WorkUnitStatus.PENDING,
        generation_key=work_unit_generation_key(
            task_id=task_id,
            work_type="method",
            objective=objective,
            paper_ids=(uuid4(),),
            input_artifact_ids=(),
            requested_worker=worker,
            output_schema=schema,
        ),
        token_budget=4000,
        tool_call_budget=6,
        timeout_seconds=180,
        paper_ids=(uuid4(),),
        allowed_tools=("search_knowledge", "read_paper", "read_artifact"),
        output_schema=schema,
    )


def test_policy_routes_simple_requests_back() -> None:
    policy = DelegationPolicy()
    single = policy.decide(paper_ids=(uuid4(),))
    assert not single.delegate
    small = policy.decide(paper_ids=(uuid4(), uuid4()))
    assert not small.delegate
    assert "Offload" in small.reason


def test_policy_delegates_large_batch_and_caps_workers() -> None:
    policy = DelegationPolicy(max_workers_cap=5)
    decision = policy.decide(
        paper_ids=tuple(uuid4() for _ in range(6)), max_workers=9
    )
    assert decision.delegate
    assert decision.max_workers == 5
    explicit = policy.decide(
        paper_ids=(uuid4(), uuid4()), requested_workstreams=("method", "verification")
    )
    assert explicit.delegate


def test_registry_lists_workers_and_marks_placeholders() -> None:
    registry = build_worker_registry()
    names = registry.names()
    assert "paper_analyzer" in names
    assert "evidence_verifier" in names
    assert not registry.require("landscape_scout").implemented


def test_planner_creates_deterministic_work_units() -> None:
    task = ResearchTask(
        task_id=uuid4(),
        project_id=uuid4(),
        user_id=uuid4(),
        research_question="比较方法",
        task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
        status=ResearchTaskStatus.CREATED,
        plan=("method", "verification"),
        budget=TaskBudget(),
        generation_key=task_generation_key(
            project_id=uuid4(),
            user_id=uuid4(),
            research_question="比较方法",
            task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
            plan=("method", "verification"),
        ),
    )
    units = ResearchPlanner().plan(task=task, paper_ids=(uuid4(), uuid4()))
    assert len(units) == 2
    assert {unit.work_type for unit in units} == {"method", "verification"}
    by_type = {unit.work_type: unit for unit in units}
    assert by_type["verification"].requested_worker == "evidence_verifier"
    assert by_type["method"].requested_worker == "paper_analyzer"
    assert by_type["verification"].dependency_ids == (
        by_type["method"].work_unit_id,
    )
    repeated = ResearchPlanner().plan(
        task=task, paper_ids=units[0].paper_ids
    )
    assert [unit.work_unit_id for unit in repeated] == [
        unit.work_unit_id for unit in units
    ]


def test_worker_runner_saves_artifact_and_returns_compact_result(tmp_path: Path) -> None:
    runner, artifacts = _runner(tmp_path, {"workstream": "method", "findings": ["A 使用 X"]})
    unit = _unit()
    result = runner.run(unit, user_id=uuid4())
    assert result.succeeded
    assert result.artifact_ref is not None
    assert result.summary
    descriptors = artifacts.search(
        unit.project_id,
        artifact_type=result.artifact_ref.artifact_type,
    )
    assert descriptors[0].research_task_id == unit.task_id
    assert descriptors[0].work_unit_id == unit.work_unit_id
    assert descriptors[0].created_by == "paper_analyzer"
    # the artifact is readable back from the same service
    from paper_agent.domain.artifact import ArtifactSelector

    slice_ = artifacts.read_slice(
        ArtifactSelector(
            artifact_id=result.artifact_ref.artifact_id,
            project_id=unit.project_id,
            view="result",
            max_tokens=800,
        )
    )
    assert slice_.content["result"]["findings"] == ["A 使用 X"]


def test_worker_output_schema_enforced(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path, {"wrong_key": "x"})
    unit = _unit()
    result = runner.run(unit, user_id=uuid4())
    assert not result.succeeded
    assert "missing required fields" in (result.error or "")


def test_evidence_verifier_rejects_verified_verdict() -> None:
    registry = build_worker_registry()
    descriptor = registry.require("evidence_verifier")
    validator = WorkerOutputValidator(descriptor.output_schema)
    with pytest.raises(ValueError, match="verdict"):
        validator(
            json.dumps(
                {"workstream": "verification", "verdict": "verified", "findings": []}
            ),
            (),
        )
    # valid verdicts pass
    validator(
        json.dumps(
            {"workstream": "verification", "verdict": "insufficient", "findings": []}
        ),
        (),
    )


def test_planned_evidence_verifier_rejects_verified_verdict() -> None:
    project_id = uuid4()
    user_id = uuid4()
    task = ResearchTask(
        task_id=uuid4(),
        project_id=project_id,
        user_id=user_id,
        research_question="比较方法",
        task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
        status=ResearchTaskStatus.CREATED,
        plan=("method", "verification"),
        budget=TaskBudget(),
        generation_key=task_generation_key(
            project_id=project_id,
            user_id=user_id,
            research_question="比较方法",
            task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
            plan=("method", "verification"),
        ),
    )
    verifier = ResearchPlanner().plan(
        task=task,
        paper_ids=(uuid4(), uuid4()),
        workstreams=("method", "verification"),
    )[-1]
    validator = WorkerOutputValidator(verifier.output_schema)

    with pytest.raises(ValueError, match="must be one of"):
        validator(
            json.dumps(
                {
                    "workstream": "verification",
                    "verdict": "verified",
                    "findings": [],
                }
            ),
            (),
        )


def test_worker_output_validator_enforces_types_extras_and_citations() -> None:
    validator = WorkerOutputValidator(
        {
            "type": "object",
            "properties": {
                "findings": {"type": "array", "items": {"type": "string"}},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["findings"],
            "additionalProperties": False,
        }
    )
    with pytest.raises(ValueError, match="must be array"):
        validator(json.dumps({"findings": "wrong"}), ())
    with pytest.raises(ValueError, match="unexpected"):
        validator(json.dumps({"findings": [], "extra": 1}), ())

    citation = CitationReference(
        citation_label="E1",
        paper_id=uuid4(),
        version_id=uuid4(),
        paper_title="Paper",
        section_path="Method",
        page_start=1,
        page_end=1,
    )
    result = ToolResult(
        call_id="c1",
        name="search_knowledge",
        model_payload={},
        citation_manifest=(citation,),
    )
    with pytest.raises(ValueError, match="must cite"):
        validator(json.dumps({"findings": ["claim"]}), (result,))
    with pytest.raises(ValueError, match="unknown citations"):
        validator(
            json.dumps({"findings": ["claim"], "citations": ["E2"]}),
            (result,),
        )
    normalized = validator(
        json.dumps({"findings": ["claim"], "citations": ["[E1]"]}),
        (result,),
    )
    assert json.loads(normalized)["citations"] == ["E1"]
    conflicting = ToolResult(
        call_id="c2",
        name="search_knowledge",
        model_payload={},
        citation_manifest=(
            CitationReference(
                citation_label="E1",
                paper_id=uuid4(),
                version_id=uuid4(),
                paper_title="Another paper",
                section_path="Results",
            ),
        ),
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        validator(
            json.dumps({"findings": ["claim"], "citations": ["E1"]}),
            (result, conflicting),
        )
    tiny = WorkerOutputValidator(
        {"type": "object", "properties": {}, "additionalProperties": True},
        token_budget=2,
    )
    with pytest.raises(ValueError, match="token_budget"):
        tiny(json.dumps({"long": "word " * 20}), ())


def test_scheduler_isolates_failures_and_retries_once(tmp_path: Path) -> None:
    registry = build_worker_registry()
    model = ScriptedWorkerModel({"workstream": "method", "findings": ["ok"]})

    class FlakyRunner(WorkerRunner):
        def __init__(self):
            super().__init__(
                registry=registry,
                model=model,
                checkpoints=InMemoryCheckpointStore(),
                artifacts=_service(tmp_path),
                materializer=ToolResultMaterializer(_service(tmp_path), OffloadPolicy()),
                search_service=FakeSearchService(),
            )
            self.fail_first = True

        def run(self, work_unit, *, user_id):
            if self.fail_first:
                self.fail_first = False
                from paper_agent.research_tasks.domain import WorkerResult

                return WorkerResult(
                    work_unit_id=work_unit.work_unit_id,
                    status="failed",
                    summary="",
                    error="transient",
                )
            return super().run(work_unit, user_id=user_id)

    scheduler = Scheduler(FlakyRunner(), max_attempts=2)
    unit = _unit()

    class Repo:
        def __init__(self):
            self.units = {}
            self.task = None

        def save_task(self, task):
            self.task = task
            return task

        def get_task(self, project_id, task_id):
            return self.task

        def find_task_by_generation_key(self, project_id, generation_key):
            return None

        def save_work_unit(self, unit):
            return self.units.setdefault(unit.work_unit_id, unit)

        def get_work_unit(self, project_id, work_unit_id):
            return self.units.get(work_unit_id)

        def list_work_units(self, project_id, task_id):
            return tuple(self.units.values())

        def update_work_unit(self, project_id, work_unit_id, **kwargs):
            current = self.units[work_unit_id]
            from dataclasses import replace

            current = replace(
                current, **{k: v for k, v in kwargs.items() if v is not None}
            )
            self.units[work_unit_id] = current
            return current

    repo = Repo()
    repo.units[unit.work_unit_id] = unit
    task = ResearchTask(
        task_id=unit.task_id,
        project_id=unit.project_id,
        user_id=uuid4(),
        research_question="q",
        task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
        status=ResearchTaskStatus.CREATED,
        plan=("method",),
        budget=TaskBudget(),
        generation_key=task_generation_key(
            project_id=unit.project_id,
            user_id=uuid4(),
            research_question="q",
            task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
            plan=("method",),
        ),
    )
    updated_task, updated_units = scheduler.run(
        task=task, units=(unit,), repository=repo, user_id=uuid4()
    )
    assert updated_units[0].status == WorkUnitStatus.COMPLETED
    assert updated_units[0].attempt_count == 2
    assert updated_task.status == ResearchTaskStatus.COMPLETED


def test_scheduler_topologically_injects_dependency_artifacts(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from paper_agent.domain.artifact import ArtifactReference, ArtifactType
    from paper_agent.research_tasks.domain import WorkerResult

    project_id, user_id = uuid4(), uuid4()
    task = ResearchTask(
        task_id=uuid4(),
        project_id=project_id,
        user_id=user_id,
        research_question="比较并核验",
        task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
        status=ResearchTaskStatus.CREATED,
        plan=("method", "verification"),
        budget=TaskBudget(),
        generation_key=task_generation_key(
            project_id=project_id,
            user_id=user_id,
            research_question="比较并核验",
            task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
            plan=("method", "verification"),
        ),
    )
    analyzer, verifier = ResearchPlanner().plan(
        task=task, paper_ids=(uuid4(), uuid4())
    )
    artifacts = _service(tmp_path)

    class RecordingRunner:
        def __init__(self):
            self.calls = []

        def run(self, unit, *, user_id):
            del user_id
            self.calls.append(unit)
            descriptor = artifacts.materialize(
                project_id=unit.project_id,
                artifact_type=ArtifactType.WORKER_RESULT,
                schema_version="worker-result-v1",
                media_type="application/json",
                payload={"work_unit_id": str(unit.work_unit_id)},
                summary=unit.work_type,
                research_task_id=unit.task_id,
                work_unit_id=unit.work_unit_id,
            )
            return WorkerResult(
                work_unit_id=unit.work_unit_id,
                status="succeeded",
                summary=unit.work_type,
                artifact_ref=ArtifactReference.from_descriptor(descriptor),
            )

    class Repo:
        def __init__(self):
            self.task = task
            self.units = {
                analyzer.work_unit_id: analyzer,
                verifier.work_unit_id: verifier,
            }

        def save_task(self, value):
            self.task = value
            return value

        def update_work_unit(self, project_id, work_unit_id, **kwargs):
            del project_id
            current = self.units[work_unit_id]
            current = replace(
                current, **{key: value for key, value in kwargs.items() if value is not None}
            )
            self.units[work_unit_id] = current
            return current

    runner = RecordingRunner()
    updated_task, updated_units = Scheduler(runner).run(
        task=task,
        # Deliberately reverse the input to prove order is driven by the DAG.
        units=(verifier, analyzer),
        repository=Repo(),
        user_id=user_id,
    )

    assert [unit.work_type for unit in runner.calls] == ["method", "verification"]
    analyzer_artifact = runner.calls[1].input_artifact_ids
    assert analyzer_artifact == (updated_units[1].output_artifact_id,)
    assert updated_task.status == ResearchTaskStatus.COMPLETED


def test_unimplemented_worker_fails_cleanly(tmp_path: Path) -> None:
    registry = build_worker_registry()
    runner = WorkerRunner(
        registry=registry,
        model=ScriptedWorkerModel({"findings": []}),
        checkpoints=InMemoryCheckpointStore(),
        artifacts=_service(tmp_path),
        materializer=ToolResultMaterializer(_service(tmp_path), OffloadPolicy()),
    )
    unit = _unit(worker="landscape_scout")
    result = runner.run(unit, user_id=uuid4())
    assert not result.succeeded
    assert "not implemented" in (result.error or "")


def test_worker_cannot_escalate_its_tool_allowlist(tmp_path: Path) -> None:
    from dataclasses import replace

    runner, _ = _runner(tmp_path, {"workstream": "method", "findings": []})
    unit = replace(
        _unit(),
        allowed_tools=("search_knowledge", "delegate_research"),
    )
    result = runner.run(unit, user_id=uuid4())

    assert not result.succeeded
    assert "cannot use tools: delegate_research" in (result.error or "")


def test_collector_reports_skipped_dependency_units(tmp_path: Path) -> None:
    from dataclasses import replace

    unit = replace(
        _unit(),
        status=WorkUnitStatus.SKIPPED,
        error="dependency_failed_or_missing",
    )
    user_id = uuid4()
    task = ResearchTask(
        task_id=unit.task_id,
        project_id=unit.project_id,
        user_id=user_id,
        research_question="q",
        task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
        status=ResearchTaskStatus.PARTIALLY_COMPLETED,
        plan=(unit.work_type,),
        budget=TaskBudget(),
        generation_key=task_generation_key(
            project_id=unit.project_id,
            user_id=user_id,
            research_question="q",
            task_type=ResearchTaskType.MULTI_PAPER_COMPARISON,
            plan=(unit.work_type,),
        ),
    )

    collected = ResultCollector(_service(tmp_path)).collect(
        task=task, units=(unit,)
    )

    assert collected.failed_work_units == (
        {
            "work_unit_id": str(unit.work_unit_id),
            "work_type": unit.work_type,
            "status": "skipped",
            "error": "dependency_failed_or_missing",
        },
    )


def test_service_delegates_and_collects(tmp_path: Path) -> None:
    artifacts = _service(tmp_path)
    runner = WorkerRunner(
        registry=build_worker_registry(),
        model=ScriptedWorkerModel(
            {
                "workstream": "method",
                "findings": ["A 使用 X"],
                "unresolved_questions": ["Q1"],
            }
        ),
        checkpoints=InMemoryCheckpointStore(),
        artifacts=artifacts,
        materializer=ToolResultMaterializer(artifacts, OffloadPolicy()),
        search_service=FakeSearchService(),
    )

    class Repo:
        def __init__(self):
            self.tasks = {}
            self.units = {}

        def save_task(self, task):
            self.tasks[task.task_id] = task
            return task

        def get_task(self, project_id, task_id):
            return self.tasks.get(task_id)

        def find_task_by_generation_key(self, project_id, generation_key):
            return next(
                (t for t in self.tasks.values() if t.generation_key == generation_key),
                None,
            )

        def save_work_unit(self, unit):
            self.units[unit.work_unit_id] = unit
            return unit

        def get_work_unit(self, project_id, work_unit_id):
            return self.units.get(work_unit_id)

        def list_work_units(self, project_id, task_id):
            return tuple(
                u for u in self.units.values() if u.task_id == task_id
            )

        def update_work_unit(self, project_id, work_unit_id, **kwargs):
            current = self.units[work_unit_id]
            from dataclasses import replace

            current = replace(
                current, **{k: v for k, v in kwargs.items() if v is not None}
            )
            self.units[work_unit_id] = current
            return current

    repo = Repo()
    service = ResearchTaskService(
        repository=repo,
        planner=ResearchPlanner(),
        policy=DelegationPolicy(),
        scheduler=Scheduler(runner),
        collector=ResultCollector(artifacts),
        artifacts=artifacts,
    )
    with pytest.raises(ValueError, match="blank entries"):
        service.delegate(
            project_id=uuid4(),
            user_id=uuid4(),
            session_id=None,
            objective="compare",
            paper_ids=(uuid4(), uuid4()),
            requested_workstreams=("   ",),
        )
    # simple two-paper request without workstreams is refused
    with pytest.raises(DelegationRefusedError):
        service.delegate(
            project_id=uuid4(),
            user_id=uuid4(),
            session_id=None,
            objective="compare",
            paper_ids=(uuid4(), uuid4()),
        )
    summary = service.delegate(
        project_id=uuid4(),
        user_id=uuid4(),
        session_id=None,
        objective="compare methods",
        paper_ids=(uuid4(), uuid4()),
        requested_workstreams=("method",),
    )
    assert summary["status"] == "completed"
    assert len(summary["work_unit_ids"]) == 1
    task = repo.tasks[UUID(summary["task_id"])]
    collected = service.collect(project_id=task.project_id, task_id=task.task_id)
    assert collected["status"] == "completed"
    assert collected["summary"]
    assert len(collected["artifact_refs"]) == 1
    assert collected["unresolved_questions"] == ["Q1"]

    repeated = service.delegate(
        project_id=task.project_id,
        user_id=task.user_id,
        session_id=None,
        objective="compare methods",
        paper_ids=tuple(repo.units[next(iter(repo.units))].paper_ids),
        requested_workstreams=("method",),
    )
    assert repeated["task_id"] == summary["task_id"]
    assert len(repo.units) == 1


def test_task_generation_key_includes_paper_scope() -> None:
    project_id, user_id = uuid4(), uuid4()
    common = {
        "project_id": project_id,
        "user_id": user_id,
        "research_question": "compare",
        "task_type": ResearchTaskType.MULTI_PAPER_COMPARISON,
        "plan": ("method",),
    }
    first = task_generation_key(**common, paper_ids=(uuid4(),))
    second = task_generation_key(**common, paper_ids=(uuid4(),))
    assert first != second
