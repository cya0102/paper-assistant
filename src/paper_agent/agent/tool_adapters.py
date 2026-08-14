"""Agent-facing adapters for internal search and read services."""

from typing import Any
from uuid import UUID

from paper_agent.agent.tools import ToolContract
from paper_agent.agent.context_builder import ContextBuilder
from paper_agent.domain.enums import ChunkType, ElementType
from paper_agent.domain.reading import ReadPaperRequest, ReadPaperResult
from paper_agent.domain.retrieval import MetadataFilter, SearchRequest, SearchScope
from paper_agent.reading.service import ReadPaperService
from paper_agent.retrieval.advanced import KnowledgeSearch


class SearchKnowledgeToolAdapter:
    def __init__(
        self,
        service: KnowledgeSearch,
        project_id: UUID,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._service = service
        self._project_id = project_id
        self._context_builder = context_builder or ContextBuilder()

    def contract(self) -> ToolContract:
        return ToolContract(
            name="search_knowledge",
            description="在已索引论文中查找可追溯证据。",
            strict=False,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "paper_ids": {"type": "array", "items": {"type": "string", "format": "uuid"}},
                    "section_ids": {"type": "array", "items": {"type": "string", "format": "uuid"}},
                    "chunk_types": {"type": "array", "items": {"type": "string", "enum": [item.value for item in ChunkType]}},
                    "max_evidence": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=self.execute,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._service.search_knowledge(
            SearchRequest(
                query=str(arguments["query"]),
                scope=SearchScope(
                    project_id=self._project_id,
                    paper_ids=tuple(UUID(value) for value in arguments.get("paper_ids", [])),
                    section_ids=tuple(UUID(value) for value in arguments.get("section_ids", [])),
                ),
                filters=MetadataFilter(
                    chunk_types=tuple(ChunkType(value) for value in arguments.get("chunk_types", []))
                ),
                max_evidence=int(arguments.get("max_evidence", 5)),
            )
        )
        context = self._context_builder.build(result.query, result.evidence)
        selected = tuple(item.evidence for item in context.citations)
        return {
            "query": result.query,
            "status": result.status.value,
            "has_sufficient_evidence": result.has_sufficient_evidence,
            "reason": result.reason,
            "context": context.text,
            "context_token_count": context.token_count,
            "omitted_evidence": context.omitted_evidence,
            "resolved_papers": [
                {
                    "paper_id": str(item.paper_id),
                    "version_id": str(item.version_id),
                    "title": item.title,
                    "score": item.score,
                }
                for item in result.resolved_papers
            ],
            "evidence": [
                {
                    "citation": f"E{int(item.evidence_id.hex[:12], 16)}",
                    "evidence_id": str(item.evidence_id),
                    "chunk_id": str(item.chunk_id),
                    "paper_id": str(item.paper_id),
                    "version_id": str(item.version_id),
                    "paper_title": item.paper_title,
                    "section_id": str(item.section_id),
                    "section_path": item.section_path,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "element_ids": [str(value) for value in item.element_ids],
                    "text": item.text,
                    "relevance": item.relevance,
                }
                for item in selected
            ],
        }


class ReadPaperToolAdapter:
    def __init__(self, service: ReadPaperService, project_id: UUID) -> None:
        self._service = service
        self._project_id = project_id

    def contract(self) -> ToolContract:
        return ToolContract(
            name="read_paper",
            description="读取论文的完整 Section、页范围或 Figure/Table/Equation/Algorithm。",
            strict=False,
            parameters={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "format": "uuid"},
                    "version_id": {"type": ["string", "null"], "format": "uuid"},
                    "section_id": {"type": ["string", "null"], "format": "uuid"},
                    "page_range": {"type": ["array", "null"], "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "element_id": {"type": ["string", "null"], "format": "uuid"},
                    "element_types": {"type": "array", "items": {"type": "string", "enum": [item.value for item in ElementType]}},
                    "include_neighbors": {"type": "boolean"},
                },
                "required": ["paper_id"],
                "additionalProperties": False,
            },
            handler=self.execute,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        page_range = arguments.get("page_range")
        result = self._service.read_paper(
            ReadPaperRequest(
                paper_id=UUID(arguments["paper_id"]),
                project_id=self._project_id,
                version_id=self._optional_uuid(arguments.get("version_id")),
                section_id=self._optional_uuid(arguments.get("section_id")),
                page_range=(int(page_range[0]), int(page_range[1])) if page_range else None,
                element_id=self._optional_uuid(arguments.get("element_id")),
                element_types=tuple(ElementType(value) for value in arguments.get("element_types", [])),
                include_neighbors=bool(arguments.get("include_neighbors", True)),
            )
        )
        return self._serialize(result)

    @staticmethod
    def _optional_uuid(value: object) -> UUID | None:
        return UUID(str(value)) if value else None

    @staticmethod
    def _serialize(result: ReadPaperResult) -> dict[str, Any]:
        passages: list[dict[str, Any]] = [
            {
                "citation": f"P{int(item.chunk_id.hex[:12], 16)}",
                "chunk_id": str(item.chunk_id),
                "paper_title": result.title,
                "section_id": str(item.section_id),
                "section_path": item.section_path,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "chunk_order": item.chunk_order,
                "element_ids": [str(value) for value in item.element_ids],
                "source_group_ids": [str(value) for value in item.source_group_ids],
                "source_block_ids": list(item.source_block_ids),
                "text": item.text,
            }
            for item in result.passages
        ]
        elements: list[dict[str, Any]] = [
            {
                "citation": f"P{int(item.element_id.hex[:12], 16)}",
                "element_id": str(item.element_id),
                "element_type": item.element_type.value,
                "paper_title": result.title,
                "section_id": str(item.section_id),
                "section_path": item.section_path,
                "page_start": item.page,
                "page_end": item.page,
                "label": item.label,
                "caption": item.caption,
                "content": item.content,
                "source_block_ids": list(item.source_block_ids),
            }
            for item in result.elements
        ]
        # Unified evidence so the Finalizer, MiMo synthesis, Runtime Memory and
        # any other consumer can treat Read results exactly like Search results.
        evidence = [
            {
                "citation": entry["citation"],
                "paper_id": str(result.paper_id),
                "version_id": str(result.version_id),
                "paper_title": entry["paper_title"],
                "section_id": entry["section_id"],
                "section_path": entry["section_path"],
                "page_start": entry["page_start"],
                "page_end": entry["page_end"],
                "chunk_id": entry.get("chunk_id"),
                "element_id": entry.get("element_id"),
                "text": entry["text"]
                if "text" in entry
                else entry.get("content") or entry.get("caption") or entry.get("label") or "",
            }
            for entry in (*passages, *elements)
        ]
        return {
            "paper_id": str(result.paper_id),
            "version_id": str(result.version_id),
            "title": result.title,
            "passages": passages,
            "elements": elements,
            "evidence": evidence,
        }