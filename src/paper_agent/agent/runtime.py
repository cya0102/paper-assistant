"""Checkpointed, idempotent model/tool loop."""

from collections.abc import Callable
from uuid import UUID

from paper_agent.agent.ports import AgentCheckpointStore, LanguageModel, MemoryRepository, SessionStore
from paper_agent.agent.tools import ToolRegistry
from paper_agent.domain.agent import (
    AgentAnswer,
    AgentCheckpoint,
    AgentRunStatus,
    ConversationMessage,
    ModelTurn,
    ToolResult,
)
from paper_agent.domain.memory import Interaction, SessionState


class AgentRuntime:
    def __init__(
        self,
        model: LanguageModel,
        tools: ToolRegistry,
        checkpoints: AgentCheckpointStore,
        *,
        max_steps: int = 8,
        answer_finalizer: Callable[[str, tuple[ToolResult, ...]], str] | None = None,
        sessions: SessionStore | None = None,
        memory: MemoryRepository | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._checkpoints = checkpoints
        self._max_steps = max_steps
        self._answer_finalizer = answer_finalizer
        self._sessions = sessions
        self._memory = memory

    def run(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        project_id: UUID,
        query: str,
    ) -> AgentAnswer:
        checkpoint = self._checkpoints.load(session_id)
        if checkpoint is None or checkpoint.status in {AgentRunStatus.COMPLETED, AgentRunStatus.FAILED}:
            messages: list[ConversationMessage] = []
            session = self._sessions.load(session_id) if self._sessions is not None else None
            if session is not None:
                if session.user_id != user_id or session.project_id != project_id:
                    raise ValueError("Session state identity mismatch")
                messages.extend(
                    ConversationMessage(role=item["role"], content=item["content"])
                    for item in session.recent_messages[-10:]
                )
            messages.append(ConversationMessage(role="user", content=query))
            checkpoint = AgentCheckpoint(
                session_id=session_id,
                user_id=user_id,
                project_id=project_id,
                messages=messages,
            )
            self._checkpoints.save(checkpoint)
            turn = self._model.start(checkpoint, self._tools.model_specs())
        else:
            if checkpoint.user_id != user_id or checkpoint.project_id != project_id:
                raise ValueError("Session checkpoint identity mismatch")
            turn = self._resume_turn(checkpoint)
        return self._drive(checkpoint, turn)

    def _resume_turn(self, checkpoint: AgentCheckpoint) -> ModelTurn:
        if checkpoint.pending_calls:
            return ModelTurn(response_id=checkpoint.response_id or "recovered", tool_calls=tuple(checkpoint.pending_calls))
        completed = tuple(checkpoint.pending_response_results)
        if not completed or not checkpoint.response_id:
            return self._model.start(checkpoint, self._tools.model_specs())
        return self._model.continue_with_tools(checkpoint, completed, self._tools.model_specs())

    def _drive(self, checkpoint: AgentCheckpoint, turn: ModelTurn) -> AgentAnswer:
        while True:
            if checkpoint.step >= self._max_steps:
                checkpoint.status = AgentRunStatus.FAILED
                checkpoint.error = "Agent loop exceeded max_steps"
                self._checkpoints.save(checkpoint)
                raise RuntimeError(checkpoint.error)
            checkpoint.response_id = turn.response_id
            checkpoint.step += 1
            if turn.output_text is not None:
                try:
                    answer_text = (
                        self._answer_finalizer(turn.output_text, tuple(checkpoint.tool_results))
                        if self._answer_finalizer is not None
                        else turn.output_text
                    )
                except Exception as error:
                    checkpoint.status = AgentRunStatus.FAILED
                    checkpoint.error = str(error)
                    self._checkpoints.save(checkpoint)
                    raise
                checkpoint.status = AgentRunStatus.COMPLETED
                checkpoint.messages.append(ConversationMessage(role="assistant", content=answer_text))
                self._checkpoints.save(checkpoint)
                self._persist_memory(checkpoint, answer_text)
                return AgentAnswer(
                    session_id=checkpoint.session_id,
                    text=answer_text,
                    response_id=turn.response_id,
                    tool_results=tuple(checkpoint.tool_results),
                )
            checkpoint.status = AgentRunStatus.WAITING_FOR_TOOLS
            checkpoint.pending_calls = list(turn.tool_calls)
            if not checkpoint.pending_response_results:
                checkpoint.pending_response_results = []
            self._checkpoints.save(checkpoint)
            completed_call_ids = {item.call_id for item in checkpoint.tool_results}
            for call in tuple(checkpoint.pending_calls):
                if call.call_id in completed_call_ids:
                    continue
                try:
                    payload = self._tools.execute(call.name, call.arguments)
                    result = ToolResult(call_id=call.call_id, name=call.name, payload=payload)
                except Exception as error:
                    result = ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        payload={"error": str(error)},
                        is_error=True,
                    )
                checkpoint.tool_results.append(result)
                checkpoint.pending_response_results.append(result)
                checkpoint.pending_calls = [item for item in checkpoint.pending_calls if item.call_id != call.call_id]
                self._checkpoints.save(checkpoint)
            checkpoint.status = AgentRunStatus.RUNNING
            self._checkpoints.save(checkpoint)
            turn = self._model.continue_with_tools(
                checkpoint,
                tuple(checkpoint.pending_response_results),
                self._tools.model_specs(),
            )
            checkpoint.pending_response_results = []

    def _persist_memory(self, checkpoint: AgentCheckpoint, answer: str) -> None:
        paper_ids: list[UUID] = []
        chunk_ids: list[UUID] = []
        for result in checkpoint.tool_results:
            for raw in result.payload.get("evidence", []):
                if not isinstance(raw, dict):
                    continue
                if raw.get("paper_id"):
                    paper_ids.append(UUID(str(raw["paper_id"])))
                if raw.get("chunk_id"):
                    chunk_ids.append(UUID(str(raw["chunk_id"])))
        unique_papers = tuple(dict.fromkeys(paper_ids))
        unique_chunks = tuple(dict.fromkeys(chunk_ids))
        query = next(
            (item.content for item in reversed(checkpoint.messages) if item.role == "user"),
            "",
        )
        if self._memory is not None:
            self._memory.save_interaction(
                Interaction(
                    user_id=checkpoint.user_id,
                    session_id=checkpoint.session_id,
                    query=query,
                    paper_ids=unique_papers,
                    retrieved_chunk_ids=unique_chunks,
                    answer_summary=answer[:1000],
                )
            )
        if self._sessions is not None:
            state = self._sessions.load(checkpoint.session_id) or SessionState(
                session_id=checkpoint.session_id,
                user_id=checkpoint.user_id,
                project_id=checkpoint.project_id,
            )
            state.recent_messages.extend(
                [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
            )
            state.active_chunk_ids = list(unique_chunks)
            state.last_tool_results = [item.payload for item in checkpoint.tool_results[-5:]]
            if unique_papers:
                state.current_paper_id = unique_papers[0]
            self._sessions.save(state)
