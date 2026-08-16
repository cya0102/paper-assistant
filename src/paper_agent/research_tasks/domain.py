"""ResearchTask and WorkUnit domain models for bounded delegation.

A ResearchTask is a decomposed research request owned by one project; its
WorkUnits are the smallest executable research jobs.  WorkUnits carry their
own budgets and a stable generation_key so retries and replays are idempotent.
Workers only ever see the objective, paper ids, artifact ids, allowed tools
and budgets carried by their WorkUnit -- never the main Agent conversation.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any
from uuid import UUID, uuid4

from paper_agent.domain.artifact import ArtifactReference, CitationReference


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be blank")


class ResearchTaskType(StrEnum):
    MULTI_PAPER_COMPARISON = "multi_paper_comparison"
    LITERATURE_SURVEY = "literature_survey"
    CROSS_DOMAIN_EXPLORATION = "cross_domain_exploration"


class ResearchTaskStatus(StrEnum):
    CREATED = "created"
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkUnitStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class TaskBudget:
    max_workers: int = 3
    token_budget: int = 4000
    tool_call_budget: int = 6
    timeout_seconds: int = 180
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.max_workers <= 5:
            raise ValueError("max_workers must be between 1 and 5")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.tool_call_budget < 1:
            raise ValueError("tool_call_budget must be positive")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")


@dataclass(frozen=True, slots=True)
class ResearchTask:
    task_id: UUID
    project_id: UUID
    user_id: UUID
    research_question: str
    task_type: ResearchTaskType
    status: ResearchTaskStatus
    plan: tuple[str, ...]
    budget: TaskBudget
    generation_key: str
    session_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.research_question, "research_question")
        if len(self.generation_key) != 64 or any(c not in "0123456789abcdef" for c in self.generation_key):
            raise ValueError("generation_key must be a SHA-256 digest")
        if not self.plan:
            raise ValueError("research task requires a plan")
        if len(self.plan) != len(set(self.plan)):
            raise ValueError("research task plan workstreams must be unique")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")

    @property
    def workstreams(self) -> tuple[str, ...]:
        return self.plan


@dataclass(frozen=True, slots=True)
class WorkUnit:
    work_unit_id: UUID
    task_id: UUID
    project_id: UUID
    work_type: str
    objective: str
    requested_worker: str
    status: WorkUnitStatus
    generation_key: str
    token_budget: int
    tool_call_budget: int
    timeout_seconds: int
    paper_ids: tuple[UUID, ...] = ()
    input_artifact_ids: tuple[UUID, ...] = ()
    dependency_ids: tuple[UUID, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    output_schema: dict[str, Any] | None = None
    attempt_count: int = 0
    output_artifact_id: UUID | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.work_type, "work_type")
        _require_text(self.objective, "objective")
        _require_text(self.requested_worker, "requested_worker")
        if len(self.generation_key) != 64 or any(c not in "0123456789abcdef" for c in self.generation_key):
            raise ValueError("generation_key must be a SHA-256 digest")
        if self.token_budget < 1:
            raise ValueError("token_budget must be positive")
        if self.tool_call_budget < 1:
            raise ValueError("tool_call_budget must be positive")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        if self.attempt_count < 0:
            raise ValueError("attempt_count cannot be negative")
        if len(self.paper_ids) != len(set(self.paper_ids)):
            raise ValueError("paper_ids must be unique")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Structured, schema-validated result produced by one Worker execution."""

    work_unit_id: UUID
    status: str
    summary: str
    artifact_ref: ArtifactReference | None = None
    citation_manifest: tuple[CitationReference, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"


def task_generation_key(
    *,
    project_id: UUID,
    user_id: UUID,
    research_question: str,
    task_type: ResearchTaskType,
    plan: tuple[str, ...],
) -> str:
    payload = json.dumps(
        [
            str(project_id),
            str(user_id),
            research_question,
            task_type.value,
            list(plan),
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def work_unit_generation_key(
    *,
    task_id: UUID,
    work_type: str,
    objective: str,
    paper_ids: tuple[UUID, ...],
    input_artifact_ids: tuple[UUID, ...],
    requested_worker: str,
    output_schema: dict[str, Any] | None,
) -> str:
    payload = json.dumps(
        [
            str(task_id),
            work_type,
            objective,
            [str(value) for value in paper_ids],
            [str(value) for value in input_artifact_ids],
            requested_worker,
            output_schema,
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()
