"""Agent-facing adapters for bounded research delegation.

project_id, user_id and session_id are bound by the adapter and can never be
chosen by the model.  The model may request workstreams and a worker cap; the
DelegationPolicy validates both.
"""

from typing import Any
from uuid import UUID

from paper_agent.agent.tools import ToolContract
from paper_agent.research_tasks.service import (
    DelegationRefusedError,
    ResearchTaskService,
)


class DelegateResearchToolAdapter:
    def __init__(
        self,
        service: ResearchTaskService,
        project_id: UUID,
        user_id: UUID,
        session_id: UUID | None = None,
    ) -> None:
        self._service = service
        self._project_id = project_id
        self._user_id = user_id
        self._session_id = session_id

    def contract(self) -> ToolContract:
        return ToolContract(
            name="delegate_research",
            description=(
                "把复杂研究子问题交给隔离的 Worker 并行执行（同步返回）。"
                "返回 task_id 后可调用 collect_research_task 汇总。"
            ),
            strict=False,
            parameters={
                "type": "object",
                "properties": {
                    "objective": {"type": "string"},
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "minItems": 1,
                        "maxItems": 20,
                        "uniqueItems": True,
                    },
                    "requested_workstreams": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 12,
                        "uniqueItems": True,
                    },
                    "max_workers": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["objective", "paper_ids"],
                "additionalProperties": False,
            },
            handler=self.execute,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._service.delegate(
                project_id=self._project_id,
                user_id=self._user_id,
                session_id=self._session_id,
                objective=str(arguments["objective"]),
                paper_ids=tuple(UUID(str(value)) for value in arguments["paper_ids"]),
                requested_workstreams=tuple(
                    str(value) for value in arguments.get("requested_workstreams", [])
                ),
                max_workers=(
                    int(arguments["max_workers"])
                    if arguments.get("max_workers") is not None
                    else None
                ),
            )
        except DelegationRefusedError as error:
            return {
                "delegated": False,
                "reason": str(error),
                "suggestion": "用主 Agent + Offload 处理，或提供 requested_workstreams。",
            }
        except (ValueError, LookupError) as error:
            return {"delegated": False, "error": "invalid_request", "message": str(error)}


class CollectResearchTaskToolAdapter:
    def __init__(self, service: ResearchTaskService, project_id: UUID) -> None:
        self._service = service
        self._project_id = project_id

    def contract(self) -> ToolContract:
        return ToolContract(
            name="collect_research_task",
            description=(
                "汇总一个 ResearchTask 的 Worker 结果：紧凑摘要、Artifact 引用、"
                "Citation Manifest、未解决问题和失败 WorkUnit。"
            ),
            strict=False,
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "format": "uuid"},
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
            handler=self.execute,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._service.collect(
                project_id=self._project_id,
                task_id=UUID(str(arguments["task_id"])),
            )
        except (ValueError, LookupError) as error:
            return {"error": "invalid_request", "message": str(error)}
