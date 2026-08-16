import json
from dataclasses import replace
from pathlib import Path
from threading import Lock
import time
from uuid import UUID, uuid4

from paper_agent.agent.rod_tool_adapter import (
    RetrieveAndAnalyzeKnowledgeToolAdapter,
)
from paper_agent.artifacts.materializer import ToolResultMaterializer
from paper_agent.artifacts.policies import OffloadPolicy
from paper_agent.artifacts.service import ArtifactService
from paper_agent.delegation.runner import WorkerRunner
from paper_agent.delegation.scheduler import Scheduler
from paper_agent.domain.agent import ModelTurn, ToolCall, ToolResult
from paper_agent.domain.artifact import ArtifactSelector, ArtifactType
from paper_agent.domain.enums import SearchStatus
from paper_agent.domain.retrieval import Evidence, SearchKnowledgeResult
from paper_agent.memory import InMemoryCheckpointStore
from paper_agent.rag import (
    EvidenceArtifactMaterializer,
    RagConfig,
    RagResultStatus,
    RagWorkUnitPlanner,
    RecordingRagTracer,
    RetrieveOffloadDelegateAnswerFinalizer,
    RetrieveOffloadDelegateService,
    RodResultCollector,
)
from paper_agent.research_tasks.domain import (
    ResearchTask,
    ResearchTaskStatus,
    ResearchTaskType,
    TaskBudget,
    WorkUnit,
    WorkUnitStatus,
    WorkerResult,
    task_generation_key,
)
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore
from paper_agent.workers import build_worker_registry
from tests.unit.artifacts.test_artifact_service import MemoryArtifactRepository


class MemoryTaskRepository:
    def __init__(self) -> None:
        self.tasks: dict[UUID, ResearchTask] = {}
        self.units: dict[UUID, WorkUnit] = {}
        self._lock = Lock()

    def save_task(self, task: ResearchTask) -> ResearchTask:
        with self._lock:
            self.tasks[task.task_id] = task
            return task

    def get_task(self, project_id: UUID, task_id: UUID) -> ResearchTask | None:
        task = self.tasks.get(task_id)
        return task if task is not None and task.project_id == project_id else None

    def find_task_by_generation_key(
        self, project_id: UUID, generation_key: str
    ) -> ResearchTask | None:
        return next(
            (
                task
                for task in self.tasks.values()
                if task.project_id == project_id
                and task.generation_key == generation_key
            ),
            None,
        )

    def save_work_unit(self, unit: WorkUnit) -> WorkUnit:
        with self._lock:
            return self.units.setdefault(unit.work_unit_id, unit)

    def get_work_unit(
        self, project_id: UUID, work_unit_id: UUID
    ) -> WorkUnit | None:
        unit = self.units.get(work_unit_id)
        return unit if unit is not None and unit.project_id == project_id else None

    def list_work_units(
        self, project_id: UUID, task_id: UUID
    ) -> tuple[WorkUnit, ...]:
        return tuple(
            unit
            for unit in self.units.values()
            if unit.project_id == project_id and unit.task_id == task_id
        )

    def update_work_unit(
        self,
        project_id: UUID,
        work_unit_id: UUID,
        *,
        status: str | None = None,
        attempt_count: int | None = None,
        output_artifact_id: UUID | None = None,
        error: str | None = None,
        input_artifact_ids: tuple[UUID, ...] | None = None,
    ) -> WorkUnit:
        with self._lock:
            current = self.units[work_unit_id]
            assert current.project_id == project_id
            changes = {}
            if status is not None:
                changes["status"] = WorkUnitStatus(status)
            if attempt_count is not None:
                changes["attempt_count"] = attempt_count
            if output_artifact_id is not None:
                changes["output_artifact_id"] = output_artifact_id
            if input_artifact_ids is not None:
                changes["input_artifact_ids"] = input_artifact_ids
            if error is not None or status == WorkUnitStatus.COMPLETED.value:
                changes["error"] = error
            updated = replace(current, **changes)
            self.units[work_unit_id] = updated
            return updated


class FakeSearch:
    def __init__(self, by_query: dict[str, tuple[Evidence, ...]]) -> None:
        self.by_query = by_query
        self.calls: list[str] = []

    def search_knowledge(self, request):
        self.calls.append(request.query)
        evidence = self.by_query.get(request.query, ())
        return SearchKnowledgeResult(
            query=request.query,
            status=SearchStatus.OK if evidence else SearchStatus.NO_EVIDENCE,
            resolved_papers=(),
            evidence=evidence,
            has_sufficient_evidence=bool(evidence),
            reason=None if evidence else "no candidates",
        )


class ScriptedChunkModel:
    """Reads its one Artifact and derives relevance from the test payload."""

    def start(self, checkpoint, tools):
        assert {item["name"] for item in tools} == {"read_artifact"}
        brief = json.loads(checkpoint.messages[-1].content)
        assert len(brief["input_artifact_ids"]) == 1
        return ModelTurn(
            response_id=f"start-{checkpoint.session_id}",
            tool_calls=(
                ToolCall(
                    call_id=f"read-{checkpoint.session_id}",
                    name="read_artifact",
                    arguments={
                        "artifact_id": brief["input_artifact_ids"][0],
                        "view": "default",
                        "max_tokens": 800,
                    },
                ),
            ),
        )

    def continue_with_tools(self, checkpoint, results, tools):
        del tools
        content = results[0].model_payload["content"]
        marker = str(content["text"])
        citation = results[0].citation_manifest[0].citation_label
        relevance = "partial" if "partial" in marker else "relevant"
        claims = (
            []
            if relevance == "partial"
            else [{"text": "The chunk supports the requested fact.", "citations": [citation]}]
        )
        return ModelTurn(
            response_id=f"final-{checkpoint.session_id}",
            output_text=json.dumps(
                {
                    "relevance": relevance,
                    "summary": f"{relevance} analyst summary",
                    "claims": claims,
                    "unresolved_questions": (
                        ["second round focus"] if relevance == "partial" else []
                    ),
                }
            ),
        )


class FixedRewriter:
    version = "test-rewriter-v1"

    def __init__(self, rewritten: str) -> None:
        self.rewritten = rewritten
        self.calls = 0

    def rewrite(self, query, reports):
        del query, reports
        self.calls += 1
        return self.rewritten


def _evidence(*, text: str, paper_id: UUID | None = None) -> Evidence:
    return Evidence(
        evidence_id=uuid4(),
        chunk_id=uuid4(),
        paper_id=paper_id or uuid4(),
        version_id=uuid4(),
        paper_title="Paper",
        section_id=uuid4(),
        section_path="3 Method",
        page_start=3,
        page_end=4,
        element_ids=(),
        text=text,
        relevance=0.9,
        dense_score=0.8,
        bm25_score=0.7,
        rerank_score=0.95,
    )


def _stack(
    tmp_path: Path,
    search: FakeSearch,
    *,
    config: RagConfig | None = None,
    rewriter=None,
):
    config = config or RagConfig(max_evidence=4, max_workers=3)
    artifacts = ArtifactService(
        LocalArtifactBlobStore(tmp_path), MemoryArtifactRepository()
    )
    tasks = MemoryTaskRepository()
    runner = WorkerRunner(
        registry=build_worker_registry(),
        model=ScriptedChunkModel(),
        checkpoints=InMemoryCheckpointStore(),
        artifacts=artifacts,
        materializer=ToolResultMaterializer(artifacts, OffloadPolicy()),
    )
    tracer = RecordingRagTracer()
    service = RetrieveOffloadDelegateService(
        search=search,
        repository=tasks,
        scheduler=Scheduler(runner, max_workers=config.max_workers),
        evidence_materializer=EvidenceArtifactMaterializer(artifacts),
        planner=RagWorkUnitPlanner(config),
        collector=RodResultCollector(artifacts),
        config=config,
        rewriter=rewriter,
        tracer=tracer,
    )
    return service, artifacts, tasks, tracer


def test_one_chunk_one_artifact_and_main_payload_has_no_chunk_text(
    tmp_path: Path,
) -> None:
    first = _evidence(text="SECRET CHUNK ONE")
    second = _evidence(text="SECRET CHUNK TWO")
    search = FakeSearch({"question": (first, second)})
    service, artifacts, tasks, tracer = _stack(tmp_path, search)

    result = service.run(
        project_id=uuid4(),
        user_id=uuid4(),
        session_id=uuid4(),
        query="question",
    )

    assert result.status == RagResultStatus.SUPPORTED
    assert len(result.evidence_artifacts) == 2
    assert len(tasks.units) == 2
    assert all(unit.requested_worker == "chunk_analyst" for unit in tasks.units.values())
    assert all(unit.allowed_tools == ("read_artifact",) for unit in tasks.units.values())
    descriptors = artifacts.search(
        next(iter(tasks.tasks.values())).project_id,
        artifact_type=ArtifactType.RETRIEVED_EVIDENCE,
    )
    assert len(descriptors) == 2
    stored = artifacts.read_slice(
        ArtifactSelector(
            artifact_id=descriptors[0].artifact_id,
            project_id=descriptors[0].project_id,
            view="default",
            max_tokens=800,
        )
    )
    assert str(stored.content["text"]).startswith("SECRET CHUNK")
    model_json = json.dumps(result.to_model_payload(), ensure_ascii=False)
    assert "SECRET CHUNK" not in model_json
    assert result.citation_manifest
    names = [event.event for event in tracer.events]
    assert "rag.retrieve.started" in names
    assert names.count("rag.artifact.created") == 2
    assert names.count("rag.worker.started") == 2
    assert "rag.synthesis.started" in names


def test_no_retrieval_returns_no_evidence_without_workers(tmp_path: Path) -> None:
    service, artifacts, tasks, _ = _stack(tmp_path, FakeSearch({}))

    result = service.run(
        project_id=uuid4(),
        user_id=uuid4(),
        session_id=uuid4(),
        query="missing",
    )

    assert result.status == RagResultStatus.NO_EVIDENCE
    assert result.reports == ()
    assert tasks.units == {}
    assert artifacts.search(next(iter(tasks.tasks.values())).project_id) == ()


def test_insufficient_first_round_rewrites_once_then_succeeds(
    tmp_path: Path,
) -> None:
    search = FakeSearch(
        {
            "question": (_evidence(text="partial evidence"),),
            "question second": (_evidence(text="relevant evidence"),),
        }
    )
    rewriter = FixedRewriter("question second")
    service, _, tasks, _ = _stack(tmp_path, search, rewriter=rewriter)

    result = service.run(
        project_id=uuid4(),
        user_id=uuid4(),
        session_id=uuid4(),
        query="question",
    )

    assert result.status == RagResultStatus.SUPPORTED
    assert result.rounds_executed == 2
    assert search.calls == ["question", "question second"]
    assert rewriter.calls == 1
    assert len(tasks.units) == 2


def test_second_insufficient_round_returns_no_citations(tmp_path: Path) -> None:
    search = FakeSearch(
        {
            "question": (_evidence(text="partial first"),),
            "question second": (_evidence(text="partial second"),),
        }
    )
    service, _, _, _ = _stack(
        tmp_path, search, rewriter=FixedRewriter("question second")
    )

    result = service.run(
        project_id=uuid4(),
        user_id=uuid4(),
        session_id=uuid4(),
        query="question",
    )

    assert result.status == RagResultStatus.INSUFFICIENT
    assert result.rounds_executed == 2
    assert result.citation_manifest == ()
    assert result.to_model_payload()["has_sufficient_evidence"] is False


def test_rod_tool_contract_returns_compact_serialization(tmp_path: Path) -> None:
    evidence = _evidence(text="OFFLOADED ONLY")
    service, artifacts, tasks, _ = _stack(tmp_path, FakeSearch({"q": (evidence,)}))
    task = next(iter(tasks.tasks.values()), None)
    assert task is None
    project_id, session_id = uuid4(), uuid4()
    adapter = RetrieveAndAnalyzeKnowledgeToolAdapter(
        service,
        project_id=project_id,
        user_id=uuid4(),
        session_id=session_id,
    )

    contract = adapter.contract()
    payload = contract.handler({"query": "q"})

    assert contract.name == "retrieve_and_analyze_knowledge"
    assert contract.parameters["required"] == ["query"]
    assert payload["status"] == "supported"
    assert "OFFLOADED ONLY" not in json.dumps(payload)
    assert payload["citation_manifest"]
    compact = ToolResultMaterializer(artifacts, OffloadPolicy()).materialize(
        project_id=project_id,
        session_id=session_id,
        call=ToolCall("rod-call", contract.name, {"query": "q"}),
        raw_payload=payload,
    )
    assert compact.citation_manifest
    assert compact.citation_manifest[0].chunk_id == evidence.chunk_id


def test_rod_finalizer_rejects_model_memory_when_evidence_is_insufficient() -> None:
    finalizer = RetrieveOffloadDelegateAnswerFinalizer()
    result = ToolResult(
        call_id="rod-call",
        name="retrieve_and_analyze_knowledge",
        model_payload={
            "status": "insufficient",
            "reason": "no supported claim",
        },
    )

    answer = finalizer("A fabricated paper fact.", (result,))

    assert answer == "no_evidence：no supported claim"


def test_scheduler_runs_independent_units_with_bounded_parallelism() -> None:
    project_id, user_id, task_id = uuid4(), uuid4(), uuid4()
    task = ResearchTask(
        task_id=task_id,
        project_id=project_id,
        user_id=user_id,
        research_question="q",
        task_type=ResearchTaskType.RAG_EVIDENCE_ANALYSIS,
        status=ResearchTaskStatus.CREATED,
        plan=("chunk_analysis",),
        budget=TaskBudget(max_workers=2),
        generation_key=task_generation_key(
            project_id=project_id,
            user_id=user_id,
            research_question="q",
            task_type=ResearchTaskType.RAG_EVIDENCE_ANALYSIS,
            plan=("chunk_analysis",),
        ),
    )
    units = tuple(
        WorkUnit(
            work_unit_id=uuid4(),
            task_id=task_id,
            project_id=project_id,
            work_type="chunk_analysis",
            objective=f"unit {index}",
            requested_worker="chunk_analyst",
            status=WorkUnitStatus.PENDING,
            generation_key=(f"{index:064x}"),
            token_budget=100,
            tool_call_budget=1,
            timeout_seconds=5,
        )
        for index in range(3)
    )
    repository = MemoryTaskRepository()
    repository.save_task(task)
    for unit in units:
        repository.save_work_unit(unit)

    class SlowRunner:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = Lock()

        def run(self, work_unit, *, user_id):
            del user_id
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            with self.lock:
                self.active -= 1
            return WorkerResult(
                work_unit_id=work_unit.work_unit_id,
                status="succeeded",
                summary="ok",
            )

    runner = SlowRunner()
    _, updated = Scheduler(runner, max_workers=3).run(
        task=task,
        units=units,
        repository=repository,
        user_id=user_id,
    )

    assert runner.max_active == 2
    assert all(unit.status == WorkUnitStatus.COMPLETED for unit in updated)
