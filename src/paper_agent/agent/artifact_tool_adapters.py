"""Agent-facing adapters for the Artifact Service: read_artifact and search_artifact.

project_id is bound by the adapter, never by the model.  Views, cursors and
token budgets are validated by the domain; errors map to stable codes so the
model can react without seeing filesystem paths or JSONPath.
"""

from typing import Any
from uuid import UUID

from paper_agent.agent.tools import ToolContract
from paper_agent.artifacts.ports import ArtifactServicePort
from paper_agent.domain.artifact import (
    ArtifactSelector,
    ArtifactType,
    citation_to_dict,
)
from paper_agent.domain.errors import PaperAgentError


class ReadArtifactToolAdapter:
    def __init__(
        self,
        artifacts: ArtifactServicePort,
        project_id: UUID,
        *,
        max_tokens_cap: int = 4000,
    ) -> None:
        self._artifacts = artifacts
        self._project_id = project_id
        self._max_tokens_cap = max_tokens_cap

    def contract(self) -> ToolContract:
        return ToolContract(
            name="read_artifact",
            description=(
                "按视图读取已保存 Artifact 的受限分片。视图只能是工具返回的"
                "available_views 之一；支持 cursor 分页和 max_tokens 上限。"
            ),
            strict=False,
            parameters={
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string", "format": "uuid"},
                    "view": {"type": "string"},
                    "cursor": {"type": ["string", "null"]},
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._max_tokens_cap,
                    },
                },
                "required": ["artifact_id"],
                "additionalProperties": False,
            },
            handler=self.execute,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self._artifacts.read_slice(
                ArtifactSelector(
                    artifact_id=UUID(str(arguments["artifact_id"])),
                    project_id=self._project_id,
                    view=str(arguments.get("view", "default")),
                    cursor=arguments.get("cursor"),
                    max_tokens=int(arguments.get("max_tokens", 800)),
                )
            )
        except PaperAgentError as error:
            return {"error": error.code.value, "message": str(error)}
        except (ValueError, LookupError) as error:
            return {"error": "invalid_request", "message": str(error)}
        return {
            "artifact_id": str(result.artifact_id),
            "view": result.view,
            "content": result.content,
            "citations": [citation_to_dict(item) for item in result.citations],
            "next_cursor": result.next_cursor,
            "truncated": result.truncated,
            "token_count": result.token_count,
        }


class SearchArtifactToolAdapter:
    """MVP structured/keyword search over the Artifact catalog for one project."""

    def __init__(self, artifacts: ArtifactServicePort, project_id: UUID) -> None:
        self._artifacts = artifacts
        self._project_id = project_id

    def contract(self) -> ToolContract:
        return ToolContract(
            name="search_artifact",
            description="在 Artifact 目录中检索已保存结果（按摘要关键词或类型过滤）。",
            strict=False,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "artifact_type": {
                        "type": ["string", "null"],
                        "enum": [item.value for item in ArtifactType],
                    },
                    "created_by": {"type": ["string", "null"]},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=self.execute,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        artifact_type_raw = arguments.get("artifact_type")
        try:
            descriptors = self._artifacts.search(
                self._project_id,
                artifact_type=(
                    ArtifactType(str(artifact_type_raw)) if artifact_type_raw else None
                ),
                created_by=arguments.get("created_by"),
                query=str(arguments.get("query", "")),
                limit=int(arguments.get("max_results", 10)),
            )
        except (ValueError, LookupError) as error:
            return {"error": "invalid_request", "message": str(error)}
        return {
            "count": len(descriptors),
            "results": [
                {
                    "artifact_id": str(item.artifact_id),
                    "artifact_type": item.artifact_type.value,
                    "schema_version": item.schema_version,
                    "summary": item.summary,
                    "token_estimate": item.token_estimate,
                    "byte_size": item.byte_size,
                    "created_by": item.created_by,
                    "created_at": item.created_at.isoformat(),
                }
                for item in descriptors
            ],
        }
