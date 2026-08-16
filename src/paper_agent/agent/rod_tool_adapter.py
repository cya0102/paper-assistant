"""Agent-facing composite Retrieve-Offload-Delegate RAG tool."""

from typing import Any
from uuid import UUID

from paper_agent.agent.tools import ToolContract
from paper_agent.rag.rod_service import RetrieveOffloadDelegateService


class RetrieveAndAnalyzeKnowledgeToolAdapter:
    def __init__(
        self,
        service: RetrieveOffloadDelegateService,
        *,
        project_id: UUID,
        user_id: UUID,
        session_id: UUID,
    ) -> None:
        self._service = service
        self._project_id = project_id
        self._user_id = user_id
        self._session_id = session_id

    def contract(self) -> ToolContract:
        return ToolContract(
            name="retrieve_and_analyze_knowledge",
            description=(
                "标准论文事实问答入口：检索证据，将每个 Chunk 单独 Offload，"
                "并行委派隔离 Worker 分析，再返回短报告与 Citation Manifest。"
            ),
            strict=False,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "maxItems": 20,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=self.execute,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        paper_ids = tuple(
            UUID(str(value)) for value in arguments.get("paper_ids", [])
        )
        result = self._service.run(
            project_id=self._project_id,
            user_id=self._user_id,
            session_id=self._session_id,
            query=str(arguments["query"]),
            paper_ids=paper_ids,
        )
        return result.to_model_payload()
