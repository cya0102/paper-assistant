"""Dependency boundaries for the Agent Runtime."""

from typing import Protocol
from uuid import UUID

from paper_agent.domain.agent import AgentCheckpoint, ModelTurn, ToolResult
from paper_agent.domain.memory import Interaction, Note, SessionState, UserPreference


class LanguageModel(Protocol):
    def start(self, checkpoint: AgentCheckpoint, tools: tuple[dict[str, object], ...]) -> ModelTurn: ...

    def continue_with_tools(
        self,
        checkpoint: AgentCheckpoint,
        results: tuple[ToolResult, ...],
        tools: tuple[dict[str, object], ...],
    ) -> ModelTurn: ...


class AgentCheckpointStore(Protocol):
    def load(self, session_id: UUID) -> AgentCheckpoint | None: ...

    def save(self, checkpoint: AgentCheckpoint) -> None: ...

    def delete(self, session_id: UUID) -> None: ...


class SessionStore(Protocol):
    def load(self, session_id: UUID) -> SessionState | None: ...

    def save(self, state: SessionState) -> None: ...


class MemoryRepository(Protocol):
    def save_interaction(self, interaction: Interaction) -> None: ...

    def search_interactions(self, user_id: UUID, query: str, limit: int = 10) -> tuple[Interaction, ...]: ...

    def save_note(self, note: Note) -> None: ...

    def list_notes(self, user_id: UUID, project_id: UUID, limit: int = 50) -> tuple[Note, ...]: ...

    def set_preference(self, preference: UserPreference) -> None: ...

    def get_preferences(self, user_id: UUID) -> tuple[UserPreference, ...]: ...

