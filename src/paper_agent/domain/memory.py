"""Session and durable interaction-memory models."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4


@dataclass(slots=True)
class SessionState:
    session_id: UUID
    user_id: UUID
    project_id: UUID
    current_paper_id: UUID | None = None
    current_section_id: UUID | None = None
    current_topic: str | None = None
    recent_messages: list[dict[str, str]] = field(default_factory=list)
    active_chunk_ids: list[UUID] = field(default_factory=list)
    last_tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Interaction:
    user_id: UUID
    session_id: UUID
    query: str
    paper_ids: tuple[UUID, ...] = ()
    topics: tuple[str, ...] = ()
    interaction_type: str = "paper_qa"
    retrieved_chunk_ids: tuple[UUID, ...] = ()
    answer_summary: str | None = None
    interaction_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class Note:
    user_id: UUID
    project_id: UUID
    content: str
    paper_id: UUID | None = None
    section_id: UUID | None = None
    tags: tuple[str, ...] = ()
    note_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class UserPreference:
    user_id: UUID
    key: str
    value: Any

