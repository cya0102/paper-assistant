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
from paper_agent.domain.comparison import PaperComparisonResult
from paper_agent.research_graph.service import EvidenceBackedComparisonService


class SearchKnowledgeToolAdapter:
    def __init__(
        self,
        service: KnowledgeSearch,
        project_id: UUID,
        context_builder: ContextBuilder | None = None,
        *,
        paper_scope: tuple[UUID, ...] | None = None,
    ) -> None:
        self._service = service
        self._project_id = project_id
        self._context_builder = context_builder or ContextBuilder()
        self._paper_scope = paper_scope

    def _resolve_paper_ids(self, arguments: dict[str, Any]) -> tuple[UUID, ...]:
        requested = tuple(UUID(value) for value in arguments.get("paper_ids", []))
        if self._paper_scope is None:
            return requested
        allowed = set(self._paper_scope)
        return tuple(dict.fromkeys(value for value in requested if value in allowed))

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
                    paper_ids=self._resolve_paper_ids(arguments),
                    section_ids=tuple(UUID(value) for value in arguments.get("section_ids", [])),
                ),
                filters=MetadataFilter(
                    chunk_types=tuple(ChunkType(value) for value in arguments.get("chunk_types", []))
                ),
                max_evidence=int(arguments.get("max_evidence", 5)),
            )
        )
        evidence = [
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
            for item in result.evidence
        ]
        return {
            "query": result.query,
            "status": result.status.value,
            "has_sufficient_evidence": result.has_sufficient_evidence,
            "reason": result.reason,
            "summary": (
                f"找到 {len(result.evidence)} 条证据，覆盖 "
                f"{len(result.resolved_papers)} 篇论文"
            ),
            "resolved_papers": [
                {
                    "paper_id": str(item.paper_id),
                    "version_id": str(item.version_id),
                    "title": item.title,
                    "score": item.score,
                }
                for item in result.resolved_papers
            ],
            "evidence": evidence,
        }


class ReadPaperToolAdapter:
    def __init__(
        self,
        service: ReadPaperService,
        project_id: UUID,
        *,
        paper_scope: tuple[UUID, ...] | None = None,
    ) -> None:
        self._service = service
        self._project_id = project_id
        self._paper_scope = paper_scope

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
        paper_id = UUID(arguments["paper_id"])
        if self._paper_scope is not None and paper_id not in self._paper_scope:
            return {"error": "paper_not_in_worker_scope", "message": "Paper is outside this worker's assigned scope"}
        result = self._service.read_paper(
            ReadPaperRequest(
                paper_id=paper_id,
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
        # No unified "evidence" list is emitted: passages/elements carry the
        # citation labels and the ToolResultMaterializer builds the Citation
        # Manifest from them, so the same text is never duplicated in one payload.
        return {
            "paper_id": str(result.paper_id),
            "version_id": str(result.version_id),
            "title": result.title,
            "passages": passages,
            "elements": elements,
        }


class ComparePapersToolAdapter:
    def __init__(
        self, service: EvidenceBackedComparisonService, project_id: UUID
    ) -> None:
        self._service = service
        self._project_id = project_id

    def contract(self) -> ToolContract:
        return ToolContract(
            name="compare_papers",
            description=(
                "比较至少两篇论文的结构化 Profile/Claim；只返回带来源证据的比较项，"
                "证据不足的维度会明确拒绝。"
            ),
            strict=False,
            parameters={
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "minItems": 2,
                        "maxItems": 20,
                    }
                },
                "required": ["paper_ids"],
                "additionalProperties": False,
            },
            handler=self.execute,
        )

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        paper_ids = tuple(UUID(str(value)) for value in arguments["paper_ids"])
        return self._serialize(self._service.compare(self._project_id, paper_ids))

    @staticmethod
    def _serialize(result: PaperComparisonResult) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "project_id": str(result.project_id),
            "paper_ids": [str(value) for value in result.paper_ids],
            "paper_count": len(result.paper_ids),
            "status": result.status.value,
            "reason": result.reason,
            "derivation": {
                "method": result.derivation_method,
                "generator_version": result.generator_version,
                "schema_version": result.schema_version,
                "model_name": None,
                "prompt_version": None,
            },
            "dimensions": [
                {
                    "name": dimension.name.value,
                    "directly_comparable": dimension.directly_comparable,
                    "non_comparable_reason": dimension.non_comparable_reason,
                    "cells": [
                        {
                            "paper_id": str(cell.paper_id),
                            "paper_title": cell.paper_title,
                            "status": cell.status.value,
                            "normalized_value": cell.normalized_value,
                            "raw_description": cell.raw_description,
                            "directly_comparable": cell.directly_comparable,
                            "non_comparable_reason": cell.non_comparable_reason,
                            "confidence": cell.confidence,
                            "review_status": cell.review_status.value,
                            "evidence": [
                                {
                                    "citation": f"E{int(link.evidence_id.hex[:12], 16)}",
                                    "evidence_id": str(link.evidence_id),
                                    "paper_id": str(link.paper_id),
                                    "version_id": str(link.version_id),
                                    "section_id": str(link.section_id),
                                    "chunk_id": str(link.chunk_id),
                                    "element_id": (
                                        str(link.element_id)
                                        if link.element_id is not None
                                        else None
                                    ),
                                    "pages": [link.page_start, link.page_end],
                                    "source_block_ids": list(link.source_block_ids),
                                    "evidence_text": link.evidence_text,
                                    "relation_to_target": link.relation_to_target.value,
                                    "evidence_kind": link.evidence_kind.value,
                                    "confidence": link.confidence,
                                }
                                for link in cell.evidence_links
                            ],
                        }
                        for cell in dimension.cells
                    ],
                }
                for dimension in result.dimensions
            ],
        }
        flattened: dict[str, dict[str, Any]] = {}
        for dimension in payload["dimensions"]:
            for cell in dimension["cells"]:
                for evidence in cell["evidence"]:
                    evidence_id = str(evidence["evidence_id"])
                    flattened.setdefault(
                        evidence_id,
                        {
                            **evidence,
                            "paper_title": cell["paper_title"],
                            "section_path": f"Research Graph > {dimension['name']}",
                            "page_start": evidence["pages"][0],
                            "page_end": evidence["pages"][1],
                            "text": evidence["evidence_text"],
                        },
                    )
        payload["evidence"] = list(flattened.values())
        return payload
