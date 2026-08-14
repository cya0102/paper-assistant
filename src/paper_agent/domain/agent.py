"""Recoverable agent, tool-call, and generated-answer domain models."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID


class AgentRunStatus(StrEnum):
    RUNNING = "running"
    WAITING_FOR_TOOLS = "waiting_for_tools"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    payload: dict[str, Any]
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ModelTurn:
    response_id: str
    output_text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if bool(self.output_text) == bool(self.tool_calls):
            raise ValueError("ModelTurn must contain either output text or tool calls")


@dataclass(slots=True)
class AgentCheckpoint:
    session_id: UUID
    user_id: UUID
    project_id: UUID
    messages: list[ConversationMessage]
    status: AgentRunStatus = AgentRunStatus.RUNNING
    response_id: str | None = None
    pending_calls: list[ToolCall] = field(default_factory=list)
    pending_response_results: list[ToolResult] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    model_history: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    session_id: UUID
    text: str
    response_id: str
    tool_results: tuple[ToolResult, ...]
