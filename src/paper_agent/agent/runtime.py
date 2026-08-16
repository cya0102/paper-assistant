"""Checkpointed, idempotent model/tool loop."""

import json
from collections.abc import Callable
from time import monotonic
from uuid import UUID

from paper_agent.agent.ports import AgentCheckpointStore, LanguageModel, MemoryRepository, SessionStore
from paper_agent.agent.tools import ToolRegistry
from paper_agent.artifacts.materializer import ToolResultMaterializer
from paper_agent.artifacts.tokens import count_tokens
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
        materializer: ToolResultMaterializer | None = None,
        max_tool_calls: int | None = None,
        timeout_seconds: int | None = None,
        required_tool_name: str | None = None,
        answer_observer: Callable[[str, tuple[ToolResult, ...]], None] | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._checkpoints = checkpoints
        self._max_steps = max_steps
        self._answer_finalizer = answer_finalizer
        self._sessions = sessions
        self._memory = memory
        self._materializer = materializer
        self._max_tool_calls = max_tool_calls
        self._timeout_seconds = timeout_seconds
        self._required_tool_name = required_tool_name
        self._answer_observer = answer_observer

    def run(
        self,
        *,
        session_id: UUID,
        user_id: UUID,
        project_id: UUID,
        query: str,
    ) -> AgentAnswer:
        started_at = monotonic()
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
        try:
            return self._drive(checkpoint, turn, started_at=started_at)
        except TimeoutError as error:
            checkpoint.status = AgentRunStatus.FAILED
            checkpoint.error = str(error)
            self._checkpoints.save(checkpoint)
            raise

    def _resume_turn(self, checkpoint: AgentCheckpoint) -> ModelTurn:
        if checkpoint.pending_calls:
            return ModelTurn(response_id=checkpoint.response_id or "recovered", tool_calls=tuple(checkpoint.pending_calls))
        completed = tuple(checkpoint.pending_response_results)
        if not completed or not checkpoint.response_id:
            return self._model.start(checkpoint, self._tools.model_specs())
        return self._model.continue_with_tools(checkpoint, completed, self._tools.model_specs())

    def _drive(
        self,
        checkpoint: AgentCheckpoint,
        turn: ModelTurn,
        *,
        started_at: float,
    ) -> AgentAnswer:
        while True:
            self._check_deadline(started_at)
            if checkpoint.step >= self._max_steps:
                checkpoint.status = AgentRunStatus.FAILED
                checkpoint.error = "Agent loop exceeded max_steps"
                self._checkpoints.save(checkpoint)
                raise RuntimeError(checkpoint.error)
            checkpoint.response_id = turn.response_id
            checkpoint.step += 1
            if turn.output_text is not None:
                try:
                    if self._required_tool_name is not None and not any(
                        item.name == self._required_tool_name and not item.is_error
                        for item in checkpoint.tool_results
                    ):
                        raise ValueError(
                            "Agent must execute required tool: "
                            f"{self._required_tool_name}"
                        )
                    answer_text = (
                        self._answer_finalizer(turn.output_text, tuple(checkpoint.tool_results))
                        if self._answer_finalizer is not None
                        else turn.output_text
                    )
                    if self._answer_observer is not None:
                        self._answer_observer(
                            answer_text, tuple(checkpoint.tool_results)
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
                if (
                    self._max_tool_calls is not None
                    and len(checkpoint.tool_results) >= self._max_tool_calls
                ):
                    checkpoint.status = AgentRunStatus.FAILED
                    checkpoint.error = (
                        f"Agent exceeded tool_call_budget={self._max_tool_calls}"
                    )
                    self._checkpoints.save(checkpoint)
                    raise RuntimeError(checkpoint.error)
                self._check_deadline(started_at)
                try:
                    raw_payload = self._tools.execute(call.name, call.arguments)
                    if self._materializer is not None:
                        accumulated = sum(
                            count_tokens(json.dumps(item.model_payload, ensure_ascii=False))
                            for item in checkpoint.tool_results
                        )
                        result = self._materializer.materialize(
                            project_id=checkpoint.project_id,
                            session_id=checkpoint.session_id,
                            call=call,
                            raw_payload=raw_payload,
                            accumulated_tokens=accumulated,
                        )
                    else:
                        if isinstance(raw_payload, bytes):
                            raise ValueError(
                                "Binary tool results require a ToolResultMaterializer"
                            )
                        result = ToolResult(
                            call_id=call.call_id,
                            name=call.name,
                            model_payload=raw_payload,
                        )
                except Exception as error:
                    result = ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        model_payload={"error": str(error)},
                        is_error=True,
                    )
                checkpoint.tool_results.append(result)
                checkpoint.pending_response_results.append(result)
                checkpoint.pending_calls = [item for item in checkpoint.pending_calls if item.call_id != call.call_id]
                self._checkpoints.save(checkpoint)
                self._check_deadline(started_at)
            checkpoint.status = AgentRunStatus.RUNNING
            self._checkpoints.save(checkpoint)
            turn = self._model.continue_with_tools(
                checkpoint,
                tuple(checkpoint.pending_response_results),
                self._tools.model_specs(),
            )
            checkpoint.pending_response_results = []

    def _check_deadline(self, started_at: float) -> None:
        if (
            self._timeout_seconds is not None
            and monotonic() - started_at > self._timeout_seconds
        ):
            raise TimeoutError(
                f"Agent exceeded timeout_seconds={self._timeout_seconds}"
            )

    def _persist_memory(self, checkpoint: AgentCheckpoint, answer: str) -> None:
        paper_ids: list[UUID] = []
        chunk_ids: list[UUID] = []
        for result in checkpoint.tool_results:
            if result.citation_manifest:
                for citation in result.citation_manifest:
                    paper_ids.append(citation.paper_id)
                    if citation.chunk_id is not None:
                        chunk_ids.append(citation.chunk_id)
                continue
            for raw in result.model_payload.get("evidence", []):
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
            state.last_tool_results = [item.model_payload for item in checkpoint.tool_results[-5:]]
            if unique_papers:
                state.current_paper_id = unique_papers[0]
            self._sessions.save(state)
