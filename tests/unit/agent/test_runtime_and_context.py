from uuid import uuid4

import pytest

from paper_agent.agent.context_builder import ContextBuilder, ContextConfig, ToolEvidenceCitationFormatter
from paper_agent.agent.runtime import AgentRuntime
from paper_agent.agent.tools import ToolContract, ToolRegistry
from paper_agent.domain.agent import AgentCheckpoint, AgentRunStatus, ModelTurn, ToolCall, ToolResult
from paper_agent.domain.enums import SearchStatus
from paper_agent.domain.retrieval import Evidence
from paper_agent.memory import InMemoryCheckpointStore, InMemorySessionStore


class FakeModel:
    def __init__(self):
        self.continued_with = ()

    def start(self, checkpoint, tools):
        assert tools[0]["name"] == "search_knowledge"
        return ModelTurn(
            response_id="response-1",
            tool_calls=(ToolCall("call-1", "search_knowledge", {"query": "codebook"}),),
        )

    def continue_with_tools(self, checkpoint, results, tools):
        self.continued_with = results
        return ModelTurn(response_id="response-2", output_text="结论来自证据。[E1]")


class Memory:
    def __init__(self):
        self.interactions = []

    def save_interaction(self, interaction):
        self.interactions.append(interaction)

    def search_interactions(self, user_id, query, limit=10):
        return ()

    def save_note(self, note):
        pass

    def list_notes(self, user_id, project_id, limit=50):
        return ()

    def set_preference(self, preference):
        pass

    def get_preferences(self, user_id):
        return ()


def _tool():
    return ToolContract(
        name="search_knowledge",
        description="search",
        parameters={"type": "object"},
        handler=lambda arguments: {
            "evidence": [
                {
                    "citation": "E1",
                    "paper_title": "Paper",
                    "section_path": "3 Method",
                    "page_start": 5,
                    "page_end": 6,
                }
            ]
        },
    )


def test_agent_loop_executes_tools_checkpoints_and_formats_citations():
    registry = ToolRegistry()
    registry.register(_tool())
    store = InMemoryCheckpointStore()
    sessions = InMemorySessionStore()
    memory = Memory()
    model = FakeModel()
    answer = AgentRuntime(
        model,
        registry,
        store,
        answer_finalizer=ToolEvidenceCitationFormatter(),
        sessions=sessions,
        memory=memory,
    ).run(
        session_id=(session_id := uuid4()),
        user_id=uuid4(),
        project_id=uuid4(),
        query="How is the codebook built?",
    )
    assert "[E1]" in answer.text and "来源：" in answer.text
    checkpoint = store.load(session_id)
    assert checkpoint is not None and checkpoint.status == AgentRunStatus.COMPLETED
    assert checkpoint.pending_calls == []
    assert len(checkpoint.tool_results) == 1
    assert memory.interactions[0].query == "How is the codebook built?"
    assert sessions.load(session_id).recent_messages[-1]["role"] == "assistant"


def test_agent_resume_does_not_repeat_completed_tool_call():
    calls = []
    registry = ToolRegistry()
    registry.register(
        ToolContract(
            name="search_knowledge",
            description="search",
            parameters={"type": "object"},
            handler=lambda arguments: calls.append(arguments) or {"evidence": []},
        )
    )
    session_id, user_id, project_id = uuid4(), uuid4(), uuid4()
    store = InMemoryCheckpointStore()
    store.save(
        AgentCheckpoint(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            messages=[],
            status=AgentRunStatus.WAITING_FOR_TOOLS,
            response_id="response-1",
            pending_calls=[ToolCall("call-2", "search_knowledge", {"query": "remaining"})],
            pending_response_results=[ToolResult("call-1", "search_knowledge", {"evidence": []})],
            tool_results=[ToolResult("call-1", "search_knowledge", {"evidence": []})],
        )
    )
    answer = AgentRuntime(FakeModel(), registry, store).run(
        session_id=session_id,
        user_id=user_id,
        project_id=project_id,
        query="ignored during recovery",
    )
    assert answer.text
    assert calls == [{"query": "remaining"}]
    assert {item.call_id for item in answer.tool_results} == {"call-1", "call-2"}


def _evidence(paper_id, text, score):
    return Evidence(
        evidence_id=uuid4(),
        chunk_id=uuid4(),
        paper_id=paper_id,
        version_id=uuid4(),
        paper_title=f"Paper {paper_id}",
        section_id=uuid4(),
        section_path="3 Method",
        page_start=1,
        page_end=2,
        element_ids=(),
        text=text,
        relevance=score,
        dense_score=score,
        bm25_score=None,
        rerank_score=score,
    )


def test_context_builder_balances_papers_and_enforces_budget():
    first, second = uuid4(), uuid4()
    context = ContextBuilder(ContextConfig(token_budget=150, max_per_paper=2)).build(
        "compare",
        (
            _evidence(first, "first evidence " * 8, 0.9),
            _evidence(first, "second evidence " * 8, 0.8),
            _evidence(second, "other paper " * 8, 0.7),
        ),
    )
    assert context.token_count <= 150
    assert context.citations[0].evidence.paper_id == first
    assert context.citations[1].evidence.paper_id == second


def test_tool_citation_formatter_rejects_hallucinated_reference():
    formatter = ToolEvidenceCitationFormatter()
    with pytest.raises(ValueError, match="unknown citations"):
        formatter("unsupported [E9]", ())
