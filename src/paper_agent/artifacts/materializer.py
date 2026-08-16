"""ToolResultMaterializer: raw payload -> compact ToolResult + Artifact.

The unified execution chain is:

    Tool.execute -> raw payload -> ToolResultMaterializer -> OffloadPolicy
    -> ArtifactService -> compact ToolResult -> AgentRuntime

No Tool Adapter touches the file system and the full raw payload never enters
Redis, checkpoints, or Provider requests.  Small results stay inline as the
model payload; large results are offloaded and only a bounded preview plus the
ArtifactReference reach the model.
"""

from collections import defaultdict, deque
from hashlib import sha256
import json
from typing import Any
from uuid import UUID

from paper_agent.artifacts.ports import ArtifactServicePort
from paper_agent.artifacts.policies import OffloadPolicy
from paper_agent.artifacts.service import canonical_json
from paper_agent.artifacts.tokens import count_tokens
from paper_agent.domain.agent import ToolCall, ToolResult
from paper_agent.domain.artifact import (
    ArtifactDescriptor,
    ArtifactReference,
    ArtifactType,
    CitationReference,
)


def _evidence_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _tool_artifact_type(name: str) -> ArtifactType:
    if name == "search_knowledge":
        return ArtifactType.KNOWLEDGE_SEARCH
    if name == "read_paper":
        return ArtifactType.PAPER_READ
    if name == "compare_papers":
        return ArtifactType.PAPER_COMPARISON
    if name == "worker_result":
        return ArtifactType.WORKER_RESULT
    return ArtifactType.TOOL_RESULT


def _schema_version(name: str) -> str:
    if name == "search_knowledge":
        return "search-knowledge-v1"
    if name == "read_paper":
        return "read-paper-v1"
    if name == "compare_papers":
        return "paper-comparison-v1"
    if name == "worker_result":
        return "worker-result-v1"
    return "tool-result-v1"


def extract_citation_manifest(
    name: str, payload: dict[str, Any]
) -> tuple[CitationReference, ...]:
    """Build the Citation Manifest from a raw tool payload.

    The manifest is extracted before any offload decision so the Citation
    Finalizer can validate answers against the manifest alone, without reading
    the Artifact blob.  Labels are deduplicated while keeping first occurrence.
    """
    if name == "read_paper":
        refs = _read_citations(payload)
    elif name == "compare_papers":
        refs = _compare_citations(payload)
    elif name == "worker_result":
        refs = _worker_citations(payload)
    else:
        refs = _search_citations(payload)
    unique: dict[str, CitationReference] = {}
    for ref in refs:
        unique.setdefault(ref.citation_label, ref)
    return tuple(unique.values())


def _search_citations(payload: dict[str, Any]) -> list[CitationReference]:
    refs: list[CitationReference] = []
    for raw in payload.get("evidence", []):
        if not isinstance(raw, dict) or not raw.get("citation"):
            continue
        text = str(raw.get("text") or "")
        refs.append(
            CitationReference(
                citation_label=str(raw["citation"]),
                paper_id=UUID(str(raw["paper_id"])),
                version_id=UUID(str(raw["version_id"])),
                paper_title=str(raw.get("paper_title") or ""),
                section_path=str(raw.get("section_path") or ""),
                page_start=raw.get("page_start"),
                page_end=raw.get("page_end"),
                evidence_hash=_evidence_hash(text),
                section_id=UUID(str(raw["section_id"])) if raw.get("section_id") else None,
                chunk_id=UUID(str(raw["chunk_id"])) if raw.get("chunk_id") else None,
            )
        )
    return refs


def _read_citations(payload: dict[str, Any]) -> list[CitationReference]:
    refs: list[CitationReference] = []
    paper_id = payload.get("paper_id")
    version_id = payload.get("version_id")
    title = str(payload.get("title") or "")
    for key, chunk_key, element_key in (
        ("passages", "chunk_id", None),
        ("elements", None, "element_id"),
    ):
        for raw in payload.get(key, []):
            if not isinstance(raw, dict) or not raw.get("citation"):
                continue
            text = (
                str(raw.get("text") or raw.get("content") or raw.get("caption") or raw.get("label") or "")
            )
            refs.append(
                CitationReference(
                    citation_label=str(raw["citation"]),
                    paper_id=UUID(str(paper_id)),
                    version_id=UUID(str(version_id)),
                    paper_title=title,
                    section_path=str(raw.get("section_path") or ""),
                    page_start=raw.get("page_start") or raw.get("page"),
                    page_end=raw.get("page_end") or raw.get("page"),
                    evidence_hash=_evidence_hash(text),
                    section_id=UUID(str(raw["section_id"])) if raw.get("section_id") else None,
                    chunk_id=UUID(str(raw[chunk_key])) if chunk_key and raw.get(chunk_key) else None,
                    element_id=UUID(str(raw[element_key])) if element_key and raw.get(element_key) else None,
                )
            )
    return refs


def _compare_citations(payload: dict[str, Any]) -> list[CitationReference]:
    refs: list[CitationReference] = []
    for raw in payload.get("evidence", []):
        if not isinstance(raw, dict) or not raw.get("citation"):
            continue
        pages = raw.get("pages") if isinstance(raw.get("pages"), (list, tuple)) else None
        text = str(raw.get("evidence_text") or raw.get("text") or "")
        refs.append(
            CitationReference(
                citation_label=str(raw["citation"]),
                paper_id=UUID(str(raw["paper_id"])),
                version_id=UUID(str(raw["version_id"])),
                paper_title=str(raw.get("paper_title") or ""),
                section_path=str(raw.get("section_path") or ""),
                page_start=pages[0] if pages else raw.get("page_start"),
                page_end=pages[1] if pages else raw.get("page_end"),
                evidence_hash=_evidence_hash(text),
                section_id=UUID(str(raw["section_id"])) if raw.get("section_id") else None,
                chunk_id=UUID(str(raw["chunk_id"])) if raw.get("chunk_id") else None,
                element_id=UUID(str(raw["element_id"])) if raw.get("element_id") else None,
            )
        )
    return refs


def _worker_citations(payload: dict[str, Any]) -> list[CitationReference]:
    refs: list[CitationReference] = []
    for raw in payload.get("citations", []):
        if not isinstance(raw, dict) or not raw.get("citation_label"):
            continue
        refs.append(
            CitationReference(
                citation_label=str(raw["citation_label"]),
                paper_id=UUID(str(raw["paper_id"])),
                version_id=UUID(str(raw["version_id"])),
                paper_title=str(raw.get("paper_title") or ""),
                section_path=str(raw.get("section_path") or ""),
                page_start=raw.get("page_start"),
                page_end=raw.get("page_end"),
                evidence_hash=raw.get("evidence_hash"),
                section_id=UUID(str(raw["section_id"])) if raw.get("section_id") else None,
                chunk_id=UUID(str(raw["chunk_id"])) if raw.get("chunk_id") else None,
                element_id=UUID(str(raw["element_id"])) if raw.get("element_id") else None,
            )
        )
    return refs




def _ref_dict(descriptor: ArtifactDescriptor | None) -> dict[str, Any] | None:
    if descriptor is None:
        return None
    return {
        "artifact_id": str(descriptor.artifact_id),
        "artifact_type": descriptor.artifact_type.value,
        "media_type": descriptor.media_type,
        "byte_size": descriptor.byte_size,
        "token_estimate": descriptor.token_estimate,
        "summary": descriptor.summary,
        "created_by": descriptor.created_by,
    }


def _views_for(name: str, payload: dict[str, Any]) -> tuple[str, ...]:
    if name == "compare_papers":
        views = ["default", "all-cells", "evidence", "derivation"]
        for dimension in payload.get("dimensions", []):
            if isinstance(dimension, dict) and dimension.get("name"):
                views.append(f"dimension:{dimension['name']}")
        for paper_id in payload.get("paper_ids", []):
            views.append(f"paper:{paper_id}")
        return tuple(dict.fromkeys(views))
    if name == "read_paper":
        return ("default", "passages", "elements", "evidence", "full")
    if name == "search_knowledge":
        return ("default", "evidence", "papers", "full")
    if name == "worker_result":
        return ("default", "result", "evidence", "report", "full")
    return ("default", "full")


def _summary_for(name: str, payload: dict[str, Any]) -> str:
    if name == "search_knowledge":
        evidence = payload.get("evidence", [])
        papers = payload.get("resolved_papers", [])
        return (
            f"search status={payload.get('status')} "
            f"evidence={len(evidence)} papers={len(papers)}"
        )
    if name == "read_paper":
        return (
            f"read paper={payload.get('paper_id')} "
            f"passages={len(payload.get('passages', []))} "
            f"elements={len(payload.get('elements', []))}"
        )
    if name == "compare_papers":
        return (
            f"comparison status={payload.get('status')} "
            f"papers={len(payload.get('paper_ids', []))}"
        )
    if name == "worker_result":
        return str(payload.get("summary") or "worker result")
    return f"tool result for {name}"


def _select_balanced(
    evidence: list[dict[str, Any]], budget: int, *, max_per_paper: int = 4
) -> tuple[list[dict[str, Any]], int]:
    """Per-paper balanced, token-budgeted evidence selection for the model view."""
    queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for item in sorted(
        evidence, key=lambda value: float(value.get("relevance", 0.0)), reverse=True
    ):
        key = str(item.get("paper_id") or "?")
        if len(queues[key]) < max_per_paper:
            queues[key].append(item)
    selected: list[dict[str, Any]] = []
    used = 0
    while any(queues.values()):
        progressed = False
        for queue in queues.values():
            if not queue:
                continue
            item = queue.popleft()
            progressed = True
            cost = count_tokens(json.dumps(item, ensure_ascii=False))
            if selected and used + cost > budget:
                continue
            selected.append(item)
            used += cost
        if not progressed:
            break
    return selected, used


def _compact_search(
    payload: dict[str, Any], descriptor: ArtifactDescriptor | None, budget: int
) -> dict[str, Any]:
    evidence = [item for item in payload.get("evidence", []) if isinstance(item, dict)]
    selected, used = _select_balanced(evidence, budget)
    return {
        "query": payload.get("query"),
        "status": payload.get("status"),
        "has_sufficient_evidence": payload.get("has_sufficient_evidence"),
        "reason": payload.get("reason"),
        "summary": payload.get("summary"),
        "resolved_papers": payload.get("resolved_papers", [])[:20],
        "selected_evidence": selected,
        "omitted_evidence": len(evidence) - len(selected),
        "artifact_ref": _ref_dict(descriptor),
        "next_cursor": None,
    }


def _compact_read(
    payload: dict[str, Any], descriptor: ArtifactDescriptor | None, budget: int
) -> dict[str, Any]:
    passages = [item for item in payload.get("passages", []) if isinstance(item, dict)]
    elements = [item for item in payload.get("elements", []) if isinstance(item, dict)]
    selected_passages: list[dict[str, Any]] = []
    selected_elements: list[dict[str, Any]] = []
    used = 0
    for item in passages:
        cost = count_tokens(json.dumps(item, ensure_ascii=False))
        if selected_passages and used + cost > budget:
            break
        selected_passages.append(item)
        used += cost
    for item in elements:
        cost = count_tokens(json.dumps(item, ensure_ascii=False))
        if selected_elements and used + cost > budget:
            break
        selected_elements.append(item)
        used += cost
    return {
        "paper_id": payload.get("paper_id"),
        "version_id": payload.get("version_id"),
        "title": payload.get("title"),
        "passages": selected_passages,
        "elements": selected_elements,
        "omitted_passages": len(passages) - len(selected_passages),
        "omitted_elements": len(elements) - len(selected_elements),
        "artifact_ref": _ref_dict(descriptor),
        "available_views": list(_views_for("read_paper", payload)),
    }


def _compact_comparison(
    payload: dict[str, Any], descriptor: ArtifactDescriptor | None, budget: int
) -> dict[str, Any]:
    dimensions = [item for item in payload.get("dimensions", []) if isinstance(item, dict)]
    comparable = [
        item.get("name") for item in dimensions if item.get("directly_comparable")
    ]
    insufficient = [
        item.get("name") for item in dimensions if not item.get("directly_comparable")
    ]
    findings: list[dict[str, Any]] = []
    used = 0
    for item in dimensions:
        if not item.get("directly_comparable"):
            continue
        summary = "; ".join(
            f"{cell.get('paper_title')}: {cell.get('normalized_value')}"
            for cell in item.get("cells", [])
            if isinstance(cell, dict)
        )
        cost = count_tokens(summary)
        if findings and used + cost > budget:
            break
        findings.append({"dimension": item.get("name"), "summary": summary})
        used += cost
    citations = [
        {
            "label": ref.citation_label,
            "paper_title": ref.paper_title,
            "section_path": ref.section_path,
            "page_start": ref.page_start,
            "page_end": ref.page_end,
        }
        for ref in extract_citation_manifest("compare_papers", payload)
    ]
    return {
        "status": payload.get("status"),
        "reason": payload.get("reason"),
        "paper_count": len(payload.get("paper_ids", [])),
        "comparable_dimensions": comparable,
        "insufficient_dimensions": insufficient,
        "high_level_findings": findings,
        "citations": citations,
        "artifact_ref": _ref_dict(descriptor),
        "available_views": list(_views_for("compare_papers", payload)),
    }


def _compact_worker(
    payload: dict[str, Any], descriptor: ArtifactDescriptor | None, budget: int
) -> dict[str, Any]:
    return {
        "work_unit_id": payload.get("work_unit_id"),
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "unresolved_questions": payload.get("unresolved_questions", []),
        "citations": [
            {
                "label": ref.citation_label,
                "paper_title": ref.paper_title,
                "section_path": ref.section_path,
                "page_start": ref.page_start,
                "page_end": ref.page_end,
            }
            for ref in extract_citation_manifest("worker_result", payload)
        ],
        "artifact_ref": _ref_dict(descriptor),
        "available_views": list(_views_for("worker_result", payload)),
    }


def _compact_generic(
    payload: dict[str, Any], descriptor: ArtifactDescriptor | None, budget: int
) -> dict[str, Any]:
    return {
        "summary": str(payload.get("summary") or "tool result"),
        "note": "full payload stored as Artifact; call read_artifact to hydrate",
        "artifact_ref": _ref_dict(descriptor),
    }


def _compact_view(
    name: str,
    payload: dict[str, Any],
    descriptor: ArtifactDescriptor | None,
    budget: int,
) -> dict[str, Any]:
    if name == "search_knowledge":
        return _compact_search(payload, descriptor, budget)
    if name == "read_paper":
        return _compact_read(payload, descriptor, budget)
    if name == "compare_papers":
        return _compact_comparison(payload, descriptor, budget)
    if name == "worker_result":
        return _compact_worker(payload, descriptor, budget)
    return _compact_generic(payload, descriptor, budget)


class ToolResultMaterializer:
    def __init__(
        self,
        artifacts: ArtifactServicePort,
        policy: OffloadPolicy,
        *,
        created_by: str = "agent_runtime",
    ) -> None:
        self._artifacts = artifacts
        self._policy = policy
        self._created_by = created_by

    def materialize(
        self,
        *,
        project_id: UUID,
        session_id: UUID,
        call: ToolCall,
        raw_payload: dict[str, Any],
        accumulated_tokens: int = 0,
        always_offload: bool = False,
        work_unit_id: UUID | None = None,
        research_task_id: UUID | None = None,
    ) -> ToolResult:
        manifest = extract_citation_manifest(call.name, raw_payload)
        token_count = count_tokens(canonical_json(raw_payload))
        should_offload = self._policy.should_offload(
            tool_name=call.name,
            payload=raw_payload,
            token_count=token_count,
            accumulated_tokens=accumulated_tokens,
            force=always_offload,
        )
        if should_offload:
            descriptor = self._artifacts.materialize(
                project_id=project_id,
                session_id=session_id,
                artifact_type=_tool_artifact_type(call.name),
                schema_version=_schema_version(call.name),
                media_type="application/json",
                payload=raw_payload,
                summary=_summary_for(call.name, raw_payload),
                citation_manifest=manifest,
                created_by=self._created_by,
                tool_call_id=call.call_id,
                work_unit_id=work_unit_id,
                research_task_id=research_task_id,
            )
            model_payload = _compact_view(
                call.name, raw_payload, descriptor, self._policy.preview_budget()
            )
            ref = ArtifactReference.from_descriptor(
                descriptor,
                available_views=_views_for(call.name, raw_payload),
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                model_payload=model_payload,
                artifact_ref=ref,
                citation_manifest=manifest,
            )
        model_payload = _compact_view(
            call.name,
            raw_payload,
            None,
            self._policy.config.max_inline_tokens_per_result,
        )
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            model_payload=model_payload,
            citation_manifest=manifest,
        )
