import json
from uuid import uuid4

import pytest

from paper_agent.agent.context_builder import ToolEvidenceCitationFormatter
from paper_agent.agent.artifact_tool_adapters import ReadArtifactToolAdapter
from paper_agent.agent.runtime import AgentRuntime
from paper_agent.agent.tools import ToolContract, ToolRegistry
from paper_agent.artifacts.materializer import ToolResultMaterializer
from paper_agent.artifacts.policies import OffloadPolicy, OffloadPolicyConfig
from paper_agent.artifacts.service import ArtifactService
from paper_agent.artifacts.tokens import count_tokens
from paper_agent.domain.agent import AgentRunStatus, ModelTurn, ToolCall
from paper_agent.domain.artifact import ArtifactType
from paper_agent.memory import InMemoryCheckpointStore, InMemorySessionStore
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore
from tests.unit.artifacts.test_artifact_service import MemoryArtifactRepository


class RecordingModel:
    def __init__(self, big_payload: dict):
        self.big_payload = big_payload
        self.received_results = []

    def start(self, checkpoint, tools):
        return ModelTurn(
            response_id="response-1",
            tool_calls=(ToolCall("call-1", "search_knowledge", {"query": "x"}),),
        )

    def continue_with_tools(self, checkpoint, results, tools):
        self.received_results = list(results)
        return ModelTurn(response_id="response-2", output_text="答案来自证据。[E0]")


def _tool(big_payload: dict):
    return ToolContract(
        name="search_knowledge",
        description="search",
        parameters={"type": "object"},
        handler=lambda arguments: big_payload,
    )


def _big_payload() -> dict:
    paper_id = uuid4()
    return {
        "query": "q",
        "status": "ok",
        "has_sufficient_evidence": True,
        "summary": "big",
        "resolved_papers": [
            {"paper_id": str(paper_id), "version_id": str(uuid4()), "title": "P", "score": 1.0}
        ],
        "evidence": [
            {
                "citation": f"E{i}",
                "evidence_id": str(uuid4()),
                "chunk_id": str(uuid4()),
                "paper_id": str(paper_id),
                "version_id": str(uuid4()),
                "paper_title": "Paper",
                "section_id": str(uuid4()),
                "section_path": "Method",
                "page_start": 1,
                "page_end": 2,
                "text": "evidence " + "word " * 40,
                "relevance": 0.9,
            }
            for i in range(8)
        ],
    }


def _build(tmp_path, big_payload: dict) -> tuple[AgentRuntime, RecordingModel]:
    service = ArtifactService(
        LocalArtifactBlobStore(tmp_path),
        MemoryArtifactRepository(),
    )
    policy = OffloadPolicy(
        OffloadPolicyConfig(max_inline_tokens_per_result=200, preview_tokens=60)
    )
    materializer = ToolResultMaterializer(service, policy)
    registry = ToolRegistry()
    registry.register(_tool(big_payload))
    model = RecordingModel(big_payload)
    runtime = AgentRuntime(
        model,
        registry,
        InMemoryCheckpointStore(),
        answer_finalizer=ToolEvidenceCitationFormatter(),
        sessions=InMemorySessionStore(),
        materializer=materializer,
    )
    return runtime, model


def test_runtime_offloads_large_result_and_sends_compact_payload(
    tmp_path,
) -> None:
    runtime, model = _build(tmp_path, _big_payload())
    session_id = uuid4()
    answer = runtime.run(
        session_id=session_id,
        user_id=uuid4(),
        project_id=uuid4(),
        query="question",
    )
    # The provider only ever receives the compact model payload
    assert len(model.received_results) == 1
    compact = model.received_results[0]
    assert compact.artifact_ref is not None
    assert compact.model_payload["omitted_evidence"] > 0
    # No full raw payload in the checkpoint
    assert all(
        count_tokens(json.dumps(item.model_payload, ensure_ascii=False)) < 2000
        for item in answer.tool_results
    )
    # The answer still carries the citation sources
    assert "[E0]" in answer.text and "来源：" in answer.text


def test_runtime_resume_does_not_repeat_offloaded_tool_call(tmp_path) -> None:
    runtime, model = _build(tmp_path, _big_payload())
    session_id, user_id, project_id = uuid4(), uuid4(), uuid4()
    store = runtime._checkpoints

    # Simulate a checkpoint that already completed call-1 and is waiting for call-2
    from paper_agent.domain.agent import AgentCheckpoint, ToolResult

    store.save(
        AgentCheckpoint(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            messages=[],
            status=AgentRunStatus.WAITING_FOR_TOOLS,
            response_id="response-1",
            pending_calls=[ToolCall("call-2", "search_knowledge", {"query": "more"})],
            pending_response_results=[],
            tool_results=[
                ToolResult(
                    call_id="call-1",
                    name="search_knowledge",
                    model_payload={"status": "ok"},
                )
            ],
        )
    )
    # the tool handler is only called for call-2
    calls = []

    def handler(arguments):
        calls.append(arguments)
        return {"query": "more", "status": "ok", "summary": "s", "evidence": []}

    registry = ToolRegistry()
    registry.register(
        ToolContract(
            name="search_knowledge",
            description="search",
            parameters={"type": "object"},
            handler=handler,
        )
    )
    service = ArtifactService(
        LocalArtifactBlobStore(tmp_path),
        MemoryArtifactRepository(),
    )
    runtime2 = AgentRuntime(
        model,
        registry,
        store,
        materializer=ToolResultMaterializer(
            service,
            OffloadPolicy(OffloadPolicyConfig(max_inline_tokens_per_result=200)),
        ),
    )
    answer = runtime2.run(
        session_id=session_id, user_id=user_id, project_id=project_id, query="ignored"
    )
    assert calls == [{"query": "more"}]
    assert {item.call_id for item in answer.tool_results} == {"call-1", "call-2"}


def test_runtime_delivers_hydrated_artifact_slice_to_model(tmp_path) -> None:
    service = ArtifactService(
        LocalArtifactBlobStore(tmp_path), MemoryArtifactRepository()
    )
    project_id = uuid4()
    descriptor = service.materialize(
        project_id=project_id,
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        payload={"secret": "hydrated"},
        summary="demo",
    )

    class HydrationModel:
        received = None

        def start(self, checkpoint, tools):
            return ModelTurn(
                response_id="h1",
                tool_calls=(
                    ToolCall(
                        "read-1",
                        "read_artifact",
                        {"artifact_id": str(descriptor.artifact_id)},
                    ),
                ),
            )

        def continue_with_tools(self, checkpoint, results, tools):
            self.received = results[0].model_payload
            return ModelTurn(response_id="h2", output_text="hydrated")

    model = HydrationModel()
    registry = ToolRegistry()
    registry.register(ReadArtifactToolAdapter(service, project_id).contract())
    runtime = AgentRuntime(
        model,
        registry,
        InMemoryCheckpointStore(),
        materializer=ToolResultMaterializer(service, OffloadPolicy()),
    )
    runtime.run(
        session_id=uuid4(),
        user_id=uuid4(),
        project_id=project_id,
        query="hydrate",
    )
    assert model.received["content"] == {"secret": "hydrated"}


def test_runtime_preserves_delegate_and_collect_control_fields(tmp_path) -> None:
    service = ArtifactService(
        LocalArtifactBlobStore(tmp_path), MemoryArtifactRepository()
    )
    project_id = uuid4()
    task_id = uuid4()

    class ControlModel:
        received = []

        def start(self, checkpoint, tools):
            return ModelTurn(
                response_id="d1",
                tool_calls=(ToolCall("delegate-1", "delegate_research", {}),),
            )

        def continue_with_tools(self, checkpoint, results, tools):
            self.received.append(results[0].model_payload)
            if results[0].name == "delegate_research":
                assert results[0].model_payload["task_id"] == str(task_id)
                return ModelTurn(
                    response_id="d2",
                    tool_calls=(
                        ToolCall(
                            "collect-1",
                            "collect_research_task",
                            {"task_id": str(task_id)},
                        ),
                    ),
                )
            return ModelTurn(response_id="d3", output_text="done")

    registry = ToolRegistry()
    registry.register(
        ToolContract(
            name="delegate_research",
            description="delegate",
            parameters={"type": "object"},
            handler=lambda arguments: {
                "task_id": str(task_id),
                "status": "completed",
                "progress": "1/1",
                "work_unit_ids": [str(uuid4())],
                "assigned_workers": ["paper_analyzer"],
            },
        )
    )
    registry.register(
        ToolContract(
            name="collect_research_task",
            description="collect",
            parameters={"type": "object"},
            handler=lambda arguments: {
                "task_id": str(task_id),
                "status": "completed",
                "summary": "worker summary",
                "artifact_refs": [{"artifact_id": str(uuid4())}],
                "citation_manifest": [],
                "unresolved_questions": ["q"],
                "failed_work_units": [],
            },
        )
    )
    model = ControlModel()
    runtime = AgentRuntime(
        model,
        registry,
        InMemoryCheckpointStore(),
        materializer=ToolResultMaterializer(service, OffloadPolicy()),
    )
    runtime.run(
        session_id=uuid4(),
        user_id=uuid4(),
        project_id=project_id,
        query="delegate",
    )
    assert model.received[1]["summary"] == "worker summary"
    assert model.received[1]["unresolved_questions"] == ["q"]


def test_runtime_enforces_actual_tool_call_budget() -> None:
    calls: list[str] = []

    class TwoCallModel:
        def start(self, checkpoint, tools):
            return ModelTurn(
                response_id="b1",
                tool_calls=(
                    ToolCall("c1", "noop", {}),
                    ToolCall("c2", "noop", {}),
                ),
            )

        def continue_with_tools(self, checkpoint, results, tools):
            return ModelTurn(response_id="b2", output_text="done")

    registry = ToolRegistry()
    registry.register(
        ToolContract(
            name="noop",
            description="noop",
            parameters={"type": "object"},
            handler=lambda arguments: calls.append("called") or {"ok": True},
        )
    )
    runtime = AgentRuntime(
        TwoCallModel(),
        registry,
        InMemoryCheckpointStore(),
        max_tool_calls=1,
    )
    with pytest.raises(RuntimeError, match="tool_call_budget=1"):
        runtime.run(
            session_id=uuid4(),
            user_id=uuid4(),
            project_id=uuid4(),
            query="budget",
        )
    assert calls == ["called"]
