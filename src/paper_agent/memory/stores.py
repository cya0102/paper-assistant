"""In-memory and Redis stores for recoverable short-term state."""

import json
from typing import Any, cast
from uuid import UUID

from redis import Redis

from paper_agent.domain.agent import (
    AgentCheckpoint,
    AgentRunStatus,
    ConversationMessage,
    ToolCall,
    ToolResult,
)
from paper_agent.domain.memory import SessionState


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self.items: dict[UUID, AgentCheckpoint] = {}

    def load(self, session_id: UUID) -> AgentCheckpoint | None:
        return self.items.get(session_id)

    def save(self, checkpoint: AgentCheckpoint) -> None:
        self.items[checkpoint.session_id] = checkpoint

    def delete(self, session_id: UUID) -> None:
        self.items.pop(session_id, None)


class InMemorySessionStore:
    def __init__(self) -> None:
        self.items: dict[UUID, SessionState] = {}

    def load(self, session_id: UUID) -> SessionState | None:
        return self.items.get(session_id)

    def save(self, state: SessionState) -> None:
        self.items[state.session_id] = state


class RedisCheckpointStore:
    def __init__(self, client: Redis, *, ttl_seconds: int = 86_400, prefix: str = "paper-agent") -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._prefix = prefix

    def load(self, session_id: UUID) -> AgentCheckpoint | None:
        payload = self._client.get(self._key(session_id))
        if payload is None:
            return None
        raw = json.loads(cast(str, payload))
        return AgentCheckpoint(
            session_id=UUID(raw["session_id"]),
            user_id=UUID(raw["user_id"]),
            project_id=UUID(raw["project_id"]),
            messages=[ConversationMessage(**item) for item in raw["messages"]],
            status=AgentRunStatus(raw["status"]),
            response_id=raw["response_id"],
            pending_calls=[ToolCall(**item) for item in raw["pending_calls"]],
            pending_response_results=[
                ToolResult(**item) for item in raw.get("pending_response_results", [])
            ],
            tool_results=[ToolResult(**item) for item in raw["tool_results"]],
            model_history=list(raw.get("model_history", [])),
            step=int(raw["step"]),
            error=raw["error"],
        )

    def save(self, checkpoint: AgentCheckpoint) -> None:
        payload = {
            "session_id": str(checkpoint.session_id),
            "user_id": str(checkpoint.user_id),
            "project_id": str(checkpoint.project_id),
            "messages": [{"role": item.role, "content": item.content} for item in checkpoint.messages],
            "status": checkpoint.status.value,
            "response_id": checkpoint.response_id,
            "pending_calls": [
                {"call_id": item.call_id, "name": item.name, "arguments": item.arguments}
                for item in checkpoint.pending_calls
            ],
            "pending_response_results": [
                {"call_id": item.call_id, "name": item.name, "payload": item.payload, "is_error": item.is_error}
                for item in checkpoint.pending_response_results
            ],
            "tool_results": [
                {"call_id": item.call_id, "name": item.name, "payload": item.payload, "is_error": item.is_error}
                for item in checkpoint.tool_results
            ],
            "model_history": checkpoint.model_history,
            "step": checkpoint.step,
            "error": checkpoint.error,
        }
        self._client.set(self._key(checkpoint.session_id), json.dumps(payload, ensure_ascii=False), ex=self._ttl)

    def delete(self, session_id: UUID) -> None:
        self._client.delete(self._key(session_id))

    def _key(self, session_id: UUID) -> str:
        return f"{self._prefix}:agent:{session_id}"


class RedisSessionStore:
    def __init__(self, client: Redis, *, ttl_seconds: int = 86_400, prefix: str = "paper-agent") -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._prefix = prefix

    def load(self, session_id: UUID) -> SessionState | None:
        payload = self._client.get(self._key(session_id))
        if payload is None:
            return None
        raw: dict[str, Any] = json.loads(cast(str, payload))
        return SessionState(
            session_id=UUID(raw["session_id"]),
            user_id=UUID(raw["user_id"]),
            project_id=UUID(raw["project_id"]),
            current_paper_id=UUID(raw["current_paper_id"]) if raw["current_paper_id"] else None,
            current_section_id=UUID(raw["current_section_id"]) if raw["current_section_id"] else None,
            current_topic=raw["current_topic"],
            recent_messages=list(raw["recent_messages"]),
            active_chunk_ids=[UUID(value) for value in raw["active_chunk_ids"]],
            last_tool_results=list(raw["last_tool_results"]),
        )

    def save(self, state: SessionState) -> None:
        payload = {
            "session_id": str(state.session_id),
            "user_id": str(state.user_id),
            "project_id": str(state.project_id),
            "current_paper_id": str(state.current_paper_id) if state.current_paper_id else None,
            "current_section_id": str(state.current_section_id) if state.current_section_id else None,
            "current_topic": state.current_topic,
            "recent_messages": state.recent_messages[-20:],
            "active_chunk_ids": [str(value) for value in state.active_chunk_ids[-50:]],
            "last_tool_results": state.last_tool_results[-10:],
        }
        self._client.set(self._key(state.session_id), json.dumps(payload, ensure_ascii=False), ex=self._ttl)

    def _key(self, session_id: UUID) -> str:
        return f"{self._prefix}:session:{session_id}"
