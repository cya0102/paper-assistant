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
from base64 import b64encode
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
    if name == "delegate_research":
        return ArtifactType.RESEARCH_TASK
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
    if name == "delegate_research":
        return "research-task-v1"
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
    elif name == "read_artifact":
        refs = _worker_citations({"citations": payload.get("citations", [])})
    elif name == "collect_research_task":
        refs = _worker_citations(
            {"citations": payload.get("citation_manifest", [])}
        )
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




def _ref_dict(
    descriptor: ArtifactDescriptor | None,
    *,
    available_views: tuple[str, ...] = (),
) -> dict[str, Any] | None:
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
        "available_views": list(available_views),
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
            if used + cost > budget:
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
        "artifact_ref": _ref_dict(
            descriptor, available_views=_views_for("search_knowledge", payload)
        ),
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
        if used + cost > budget:
            break
        selected_passages.append(item)
        used += cost
    for item in elements:
        cost = count_tokens(json.dumps(item, ensure_ascii=False))
        if used + cost > budget:
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
        "artifact_ref": _ref_dict(
            descriptor, available_views=_views_for("read_paper", payload)
        ),
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
        if used + cost > budget:
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
        "artifact_ref": _ref_dict(
            descriptor, available_views=_views_for("compare_papers", payload)
        ),
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
        "artifact_ref": _ref_dict(
            descriptor, available_views=_views_for("worker_result", payload)
        ),
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


def _bounded_dict_list(
    values: object,
    *,
    budget: int,
    used: int = 0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    if not isinstance(values, list):
        return selected
    for value in values:
        if not isinstance(value, dict):
            continue
        cost = count_tokens(json.dumps(value, ensure_ascii=False, sort_keys=True))
        if used + cost > budget:
            break
        selected.append(value)
        used += cost
    return selected


def _compact_search_artifact(
    payload: dict[str, Any], descriptor: ArtifactDescriptor | None, budget: int
) -> dict[str, Any]:
    results = _bounded_dict_list(payload.get("results"), budget=budget)
    return {
        "count": payload.get("count", len(results)),
        "results": results,
        "omitted_results": max(0, int(payload.get("count", len(results))) - len(results)),
        "artifact_ref": _ref_dict(descriptor, available_views=("default", "full")),
    }


def _compact_delegate(
    payload: dict[str, Any], descriptor: ArtifactDescriptor | None, budget: int
) -> dict[str, Any]:
    compact = {
        key: payload.get(key)
        for key in (
            "delegated",
            "task_id",
            "status",
            "progress",
            "reason",
            "suggestion",
            "error",
            "message",
        )
        if key in payload
    }
    compact["work_unit_ids"] = list(payload.get("work_unit_ids", []))[:20]
    compact["assigned_workers"] = list(payload.get("assigned_workers", []))[:20]
    compact["artifact_ref"] = _ref_dict(
        descriptor, available_views=("default", "full")
    )
    return compact


def _compact_collection(
    payload: dict[str, Any], descriptor: ArtifactDescriptor | None, budget: int
) -> dict[str, Any]:
    used = count_tokens(str(payload.get("summary") or ""))
    refs = _bounded_dict_list(
        payload.get("artifact_refs"), budget=budget, used=used
    )
    used += count_tokens(json.dumps(refs, ensure_ascii=False, sort_keys=True))
    failed = _bounded_dict_list(
        payload.get("failed_work_units"), budget=budget, used=used
    )
    return {
        "task_id": payload.get("task_id"),
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "artifact_refs": refs,
        "omitted_artifact_refs": max(
            0, len(payload.get("artifact_refs", [])) - len(refs)
        ),
        "unresolved_questions": list(payload.get("unresolved_questions", []))[:20],
        "failed_work_units": failed,
        "artifact_ref": _ref_dict(
            descriptor, available_views=("default", "full")
        ),
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
    if name == "search_artifact":
        return _compact_search_artifact(payload, descriptor, budget)
    if name == "delegate_research":
        return _compact_delegate(payload, descriptor, budget)
    if name == "collect_research_task":
        return _compact_collection(payload, descriptor, budget)
    return _compact_generic(payload, descriptor, budget)


def _enforce_model_budget(
    name: str, payload: dict[str, Any], budget: int
) -> dict[str, Any]:
    if count_tokens(canonical_json(payload)) <= budget:
        return payload
    priority = (
        # An offloaded payload is unusable unless the model can retain the
        # Artifact handle. Keep this ahead of optional preview fields and shrink
        # it field-by-field below when the configured preview budget is tiny.
        "artifact_ref",
        "delegated",
        "task_id",
        "status",
        "omitted_evidence",
        "omitted_passages",
        "omitted_elements",
        "omitted_results",
        "omitted_artifact_refs",
        "error",
        "message",
        "reason",
        "suggestion",
        "summary",
        "has_sufficient_evidence",
        "paper_id",
        "version_id",
        "title",
        "paper_count",
        "progress",
        "count",
        "available_views",
        "selected_evidence",
        "passages",
        "elements",
        "high_level_findings",
        "citations",
        "resolved_papers",
        "artifact_refs",
        "unresolved_questions",
        "failed_work_units",
        "work_unit_ids",
        "assigned_workers",
    )
    compact: dict[str, Any] = {}
    for key in priority:
        if key not in payload:
            continue
        value = payload[key]
        if key == "artifact_ref" and isinstance(value, dict):
            candidate = {**compact, key: value}
            if count_tokens(canonical_json(candidate)) <= budget:
                compact[key] = value
                continue
            # Reserve the rest of a very small preview budget for status and
            # omission counters. The UUID is the only field required to call
            # read_artifact; the full typed reference remains on ToolResult.
            bounded_ref = (
                {"artifact_id": value["artifact_id"]}
                if value.get("artifact_id")
                else {}
            )
            if bounded_ref:
                compact[key] = bounded_ref
            continue
        if isinstance(value, list):
            selected: list[Any] = []
            for item in value:
                candidate = {**compact, key: [*selected, item]}
                if count_tokens(canonical_json(candidate)) > budget:
                    break
                selected.append(item)
            if selected or not value:
                compact[key] = selected
            continue
        candidate = {**compact, key: value}
        if count_tokens(canonical_json(candidate)) <= budget:
            compact[key] = value
            continue
        if isinstance(value, str):
            low, high = 0, len(value)
            best = ""
            while low <= high:
                middle = (low + high) // 2
                shortened = value[:middle]
                if count_tokens(
                    canonical_json({**compact, key: shortened})
                ) <= budget:
                    best = shortened
                    low = middle + 1
                else:
                    high = middle - 1
            if best:
                compact[key] = best
    marker = {**compact, "model_payload_truncated": True}
    if count_tokens(canonical_json(marker)) <= budget:
        compact = marker
    return compact


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
        raw_payload: dict[str, Any] | bytes,
        accumulated_tokens: int = 0,
        always_offload: bool = False,
        work_unit_id: UUID | None = None,
        research_task_id: UUID | None = None,
        created_by: str | None = None,
    ) -> ToolResult:
        if isinstance(raw_payload, bytes):
            raw_payload = {
                "media_type": "application/octet-stream",
                "encoding": "base64",
                "byte_size": len(raw_payload),
                "data_base64": b64encode(raw_payload).decode("ascii"),
                "summary": "binary tool result",
            }
        manifest = extract_citation_manifest(call.name, raw_payload)
        # read_artifact already returns a server-bounded slice. Re-materializing
        # that slice would hide the hydrated content behind another Artifact and
        # make hydration recurse forever, so it always crosses the model boundary
        # inline exactly once.
        if call.name == "read_artifact":
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                model_payload=raw_payload,
                citation_manifest=manifest,
                is_error="error" in raw_payload,
            )
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
                media_type=str(
                    raw_payload.get("media_type") or "application/json"
                ),
                payload=raw_payload,
                summary=_summary_for(call.name, raw_payload),
                citation_manifest=manifest,
                created_by=created_by or self._created_by,
                tool_call_id=call.call_id,
                work_unit_id=work_unit_id,
                research_task_id=research_task_id,
            )
            model_payload = _compact_view(
                call.name, raw_payload, descriptor, self._policy.preview_budget()
            )
            model_payload = _enforce_model_budget(
                call.name, model_payload, self._policy.preview_budget()
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
        if call.name in {
            "search_knowledge",
            "read_paper",
            "compare_papers",
            "worker_result",
            "search_artifact",
            "delegate_research",
            "collect_research_task",
        }:
            model_payload = _compact_view(
                call.name,
                raw_payload,
                None,
                self._policy.config.max_inline_tokens_per_result,
            )
            model_payload = _enforce_model_budget(
                call.name,
                model_payload,
                self._policy.config.max_inline_tokens_per_result,
            )
        else:
            # A genuinely small generic tool result is already the bounded inline
            # payload. Compacting it to a note would silently discard its data.
            model_payload = raw_payload
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            model_payload=model_payload,
            citation_manifest=manifest,
            is_error="error" in raw_payload,
        )
