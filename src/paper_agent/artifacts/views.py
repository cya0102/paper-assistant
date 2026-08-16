"""Bounded, cursor-able view extractors over stored Artifact payloads.

Views are a fixed set of structural selectors per ArtifactType -- never arbitrary
JSONPath and never file paths.  Every extractor respects a token budget and
returns a next_cursor so callers can page through large results.
"""

from collections.abc import Callable
from typing import Any

from paper_agent.artifacts.tokens import count_tokens
from paper_agent.domain.artifact import ArtifactType
from paper_agent.domain.errors import ErrorCode, PaperAgentError


class ArtifactViewError(PaperAgentError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.ARTIFACT_INVALID_VIEW, message)


ViewResult = tuple[dict[str, Any], str | None, bool, int]
"""content, next_cursor, truncated, token_count"""


def _page_items(
    items: list[dict[str, Any]],
    cursor: str | None,
    max_tokens: int,
    *,
    key: str = "items",
) -> ViewResult:
    offset = int(cursor) if cursor is not None and cursor.isdigit() else 0
    if offset < 0 or offset > len(items):
        offset = len(items)
    taken: list[dict[str, Any]] = []
    used = 0
    for item in items[offset:]:
        cost = count_tokens(json_dumps(item))
        if taken and used + cost > max_tokens:
            break
        taken.append(item)
        used += cost
        if used > max_tokens:
            break
    end = offset + len(taken)
    next_cursor = str(end) if end < len(items) else None
    return {"count": len(taken), key: taken, "total": len(items)}, next_cursor, next_cursor is not None, used


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_items(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = payload.get(key, [])
    return [item for item in raw if isinstance(item, dict)]


def _read_view(payload: dict[str, Any], view: str, cursor: str | None, max_tokens: int) -> ViewResult:
    if view in {"default", "passages"}:
        passages = _read_items(payload, "passages")
        content, next_cursor, truncated, used = _page_items(passages, cursor, max_tokens)
        return {
            "paper_id": payload.get("paper_id"),
            "version_id": payload.get("version_id"),
            "title": payload.get("title"),
            **content,
        }, next_cursor, truncated, used
    if view == "elements":
        elements = _read_items(payload, "elements")
        content, next_cursor, truncated, used = _page_items(elements, cursor, max_tokens)
        return {
            "paper_id": payload.get("paper_id"),
            "version_id": payload.get("version_id"),
            "title": payload.get("title"),
            **content,
        }, next_cursor, truncated, used
    if view == "evidence":
        evidence = _flatten_read_evidence(payload)
        return _page_items(evidence, cursor, max_tokens, key="evidence")
    if view == "full":
        return _full(payload, cursor, max_tokens)
    raise ArtifactViewError(f"Unsupported read_paper view: {view}")


def _flatten_read_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _read_items(payload, "passages"):
        result.append(
            {
                "citation": item.get("citation"),
                "paper_id": payload.get("paper_id"),
                "version_id": payload.get("version_id"),
                "paper_title": payload.get("title"),
                "section_id": item.get("section_id"),
                "section_path": item.get("section_path"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "chunk_id": item.get("chunk_id"),
                "element_id": None,
                "text": item.get("text"),
            }
        )
    for item in _read_items(payload, "elements"):
        result.append(
            {
                "citation": item.get("citation"),
                "paper_id": payload.get("paper_id"),
                "version_id": payload.get("version_id"),
                "paper_title": payload.get("title"),
                "section_id": item.get("section_id"),
                "section_path": item.get("section_path"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "chunk_id": None,
                "element_id": item.get("element_id"),
                "text": item.get("content") or item.get("caption") or item.get("label"),
            }
        )
    return result


def _search_view(payload: dict[str, Any], view: str, cursor: str | None, max_tokens: int) -> ViewResult:
    if view in {"default", "evidence"}:
        evidence = _read_items(payload, "evidence")
        content, next_cursor, truncated, used = _page_items(evidence, cursor, max_tokens, key="evidence")
        return {
            "query": payload.get("query"),
            "status": payload.get("status"),
            "summary": payload.get("summary"),
            **content,
        }, next_cursor, truncated, used
    if view == "papers":
        papers = _read_items(payload, "resolved_papers")
        return _page_items(papers, cursor, max_tokens, key="papers")
    if view == "full":
        return _full(payload, cursor, max_tokens)
    raise ArtifactViewError(f"Unsupported search view: {view}")


def _comparison_view(payload: dict[str, Any], view: str, cursor: str | None, max_tokens: int) -> ViewResult:
    dimensions = _read_items(payload, "dimensions")
    if view == "derivation":
        return {"derivation": payload.get("derivation")}, None, False, count_tokens(json_dumps(payload.get("derivation", {})))
    if view == "all-cells":
        cells = [cell for dim in dimensions for cell in _read_items(dim, "cells")]
        return _page_items(cells, cursor, max_tokens, key="cells")
    if view == "evidence":
        evidence = _read_items(payload, "evidence")
        return _page_items(evidence, cursor, max_tokens, key="evidence")
    if view.startswith("dimension:"):
        name = view.split(":", 1)[1]
        for dim in dimensions:
            if dim.get("name") == name:
                return {"dimension": dim}, None, False, count_tokens(json_dumps(dim))
        raise ArtifactViewError(f"Unknown comparison dimension: {name}")
    if view.startswith("paper:"):
        paper_id = view.split(":", 1)[1]
        cells = []
        for dim in dimensions:
            for cell in _read_items(dim, "cells"):
                if str(cell.get("paper_id")) == paper_id:
                    cells.append(cell)
        if not cells:
            raise ArtifactViewError(f"Unknown paper in comparison: {paper_id}")
        return _page_items(cells, cursor, max_tokens, key="cells")
    if view == "full":
        return _full(payload, cursor, max_tokens)
    if view == "default":
        return {
            "status": payload.get("status"),
            "paper_ids": payload.get("paper_ids"),
            "comparable_dimensions": [d.get("name") for d in dimensions if d.get("directly_comparable")],
            "insufficient_dimensions": [d.get("name") for d in dimensions if not d.get("directly_comparable")],
            "dimension_count": len(dimensions),
        }, None, False, count_tokens(json_dumps({
            "status": payload.get("status"),
            "paper_ids": payload.get("paper_ids"),
            "comparable_dimensions": [d.get("name") for d in dimensions if d.get("directly_comparable")],
            "insufficient_dimensions": [d.get("name") for d in dimensions if not d.get("directly_comparable")],
        }))
    raise ArtifactViewError(f"Unsupported comparison view: {view}")


def _worker_view(payload: dict[str, Any], view: str, cursor: str | None, max_tokens: int) -> ViewResult:
    if view == "result":
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ArtifactViewError("Worker artifact has no result payload")
        return {"result": result}, None, False, count_tokens(json_dumps(result))
    if view == "evidence":
        evidence = _read_items(payload, "evidence")
        return _page_items(evidence, cursor, max_tokens, key="evidence")
    if view == "report":
        report = payload.get("report")
        if report is not None:
            return {"report": report}, None, False, count_tokens(str(report))
        raise ArtifactViewError("Worker artifact has no report")
    if view == "full":
        return _full(payload, cursor, max_tokens)
    if view == "default":
        summary = payload.get("summary") or payload.get("result", {}).get("summary") or ""
        return {
            "work_unit_id": payload.get("work_unit_id"),
            "status": payload.get("status"),
            "summary": summary,
            "unresolved_questions": payload.get("unresolved_questions", []),
            "citations": payload.get("citations", []),
        }, None, False, count_tokens(json_dumps({
            "work_unit_id": payload.get("work_unit_id"),
            "status": payload.get("status"),
            "summary": summary,
            "unresolved_questions": payload.get("unresolved_questions", []),
        }))
    raise ArtifactViewError(f"Unsupported worker view: {view}")


def _generic_view(payload: dict[str, Any], view: str, cursor: str | None, max_tokens: int) -> ViewResult:
    if view in {"default", "full"}:
        return _full(payload, cursor, max_tokens)
    raise ArtifactViewError(f"Unsupported view: {view}")


def _full(payload: dict[str, Any], cursor: str | None, max_tokens: int) -> ViewResult:
    import json

    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    used = count_tokens(rendered)
    truncated = used > max_tokens
    if truncated:
        preview = rendered[: max_tokens * 4]
        content: dict[str, Any] = {"preview": preview, "truncated": True}
        return content, None, True, count_tokens(preview)
    return dict(payload), None, False, used


_VIEWS: dict[
    ArtifactType, Callable[[dict[str, Any], str, str | None, int], ViewResult]
] = {
    ArtifactType.KNOWLEDGE_SEARCH: _search_view,
    ArtifactType.PAPER_READ: _read_view,
    ArtifactType.PAPER_COMPARISON: _comparison_view,
    ArtifactType.WORKER_RESULT: _worker_view,
    ArtifactType.TOOL_RESULT: _generic_view,
    ArtifactType.RESEARCH_TASK: _generic_view,
}


def extract_view(
    payload: dict[str, Any],
    artifact_type: ArtifactType,
    view: str,
    cursor: str | None,
    max_tokens: int,
) -> ViewResult:
    handler = _VIEWS.get(artifact_type)
    if handler is None:
        raise ArtifactViewError(f"No views defined for artifact type: {artifact_type.value}")
    result = handler(payload, view, cursor, max_tokens)
    return result
