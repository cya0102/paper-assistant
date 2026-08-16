"""Recoverable agent, tool-call, and generated-answer domain models."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID

from paper_agent.domain.artifact import ArtifactReference, CitationReference


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
    """Compact result handed to checkpoints and Providers.

    The full raw payload never travels with the checkpoint: large results are
    offloaded to the Artifact store and only the bounded model_payload,
    artifact_ref and citation_manifest are kept.  The payload property remains
    as a legacy alias for the model-facing payload.
    """

    call_id: str
    name: str
    model_payload: dict[str, Any]
    artifact_ref: ArtifactReference | None = None
    citation_manifest: tuple[CitationReference, ...] = ()
    is_error: bool = False

    @property
    def payload(self) -> dict[str, Any]:
        return self.model_payload


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
