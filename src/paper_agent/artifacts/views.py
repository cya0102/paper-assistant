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
    base: dict[str, Any] | None = None,
) -> ViewResult:
    base = dict(base or {})
    offset, fragment_offset = _parse_item_cursor(cursor, len(items))
    if fragment_offset:
        return _fragment_item(
            items,
            offset,
            fragment_offset,
            max_tokens,
            key=key,
            base=base,
        )
    taken: list[dict[str, Any]] = []
    for index, item in enumerate(items[offset:], start=offset):
        candidate = {
            **base,
            "count": len(taken) + 1,
            key: [*taken, item],
            "total": len(items),
        }
        if count_tokens(json_dumps(candidate)) > max_tokens:
            if not taken:
                return _fragment_item(
                    items,
                    index,
                    0,
                    max_tokens,
                    key=key,
                    base=base,
                )
            break
        taken.append(item)
    end = offset + len(taken)
    next_cursor = str(end) if end < len(items) else None
    content = {**base, "count": len(taken), key: taken, "total": len(items)}
    used = count_tokens(json_dumps(content))
    if used > max_tokens:
        raise ArtifactViewError("max_tokens is too small for the view envelope")
    return content, next_cursor, next_cursor is not None, used


def _parse_item_cursor(cursor: str | None, item_count: int) -> tuple[int, int]:
    if cursor is None:
        return 0, 0
    if cursor.isdigit():
        offset = int(cursor)
        if offset > item_count:
            raise ArtifactViewError("Cursor is outside the Artifact view")
        return offset, 0
    parts = cursor.split(":")
    if (
        len(parts) == 3
        and parts[0] == "f"
        and parts[1].isdigit()
        and parts[2].isdigit()
    ):
        item_offset = int(parts[1])
        fragment_offset = int(parts[2])
        if item_offset >= item_count:
            raise ArtifactViewError("Cursor is outside the Artifact view")
        return item_offset, fragment_offset
    raise ArtifactViewError("Invalid Artifact cursor")


def _fragment_item(
    items: list[dict[str, Any]],
    item_offset: int,
    fragment_offset: int,
    max_tokens: int,
    *,
    key: str,
    base: dict[str, Any],
) -> ViewResult:
    if item_offset >= len(items):
        content = {**base, "count": 0, key: [], "total": len(items)}
        used = count_tokens(json_dumps(content))
        if used > max_tokens:
            raise ArtifactViewError("max_tokens is too small for the view envelope")
        return content, None, False, used
    rendered = json_dumps(items[item_offset])
    if fragment_offset >= len(rendered):
        raise ArtifactViewError("Cursor is outside the oversized item")

    def build(end: int) -> dict[str, Any]:
        return {
            "count": 0,
            "total": len(items),
            "json_fragment": rendered[fragment_offset:end],
        }

    low, high = fragment_offset + 1, len(rendered)
    best = fragment_offset
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(json_dumps(build(middle))) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == fragment_offset:
        raise ArtifactViewError("max_tokens is too small for an item fragment")
    content = build(best)
    next_cursor: str | None
    if best < len(rendered):
        next_cursor = f"f:{item_offset}:{best}"
    else:
        next_cursor = str(item_offset + 1) if item_offset + 1 < len(items) else None
    used = count_tokens(json_dumps(content))
    return content, next_cursor, next_cursor is not None, used


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _read_items(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = payload.get(key, [])
    return [item for item in raw if isinstance(item, dict)]


def _read_view(payload: dict[str, Any], view: str, cursor: str | None, max_tokens: int) -> ViewResult:
    if view in {"default", "passages"}:
        passages = _read_items(payload, "passages")
        return _page_items(
            passages,
            cursor,
            max_tokens,
            base={
                "paper_id": payload.get("paper_id"),
                "version_id": payload.get("version_id"),
                "title": payload.get("title"),
            },
        )
    if view == "elements":
        elements = _read_items(payload, "elements")
        return _page_items(
            elements,
            cursor,
            max_tokens,
            base={
                "paper_id": payload.get("paper_id"),
                "version_id": payload.get("version_id"),
                "title": payload.get("title"),
            },
        )
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
        return _page_items(
            evidence,
            cursor,
            max_tokens,
            key="evidence",
            base={
                "query": payload.get("query"),
                "status": payload.get("status"),
                "summary": payload.get("summary"),
            },
        )
    if view == "papers":
        papers = _read_items(payload, "resolved_papers")
        return _page_items(papers, cursor, max_tokens, key="papers")
    if view == "full":
        return _full(payload, cursor, max_tokens)
    raise ArtifactViewError(f"Unsupported search view: {view}")


def _comparison_view(payload: dict[str, Any], view: str, cursor: str | None, max_tokens: int) -> ViewResult:
    dimensions = _read_items(payload, "dimensions")
    if view == "derivation":
        return _bounded_value("derivation", payload.get("derivation"), cursor, max_tokens)
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
                return _bounded_value("dimension", dim, cursor, max_tokens)
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
        content = {
            "status": payload.get("status"),
            "paper_ids": payload.get("paper_ids"),
            "comparable_dimensions": [d.get("name") for d in dimensions if d.get("directly_comparable")],
            "insufficient_dimensions": [d.get("name") for d in dimensions if not d.get("directly_comparable")],
            "dimension_count": len(dimensions),
        }
        return _bounded_content(content, content, cursor, max_tokens)
    raise ArtifactViewError(f"Unsupported comparison view: {view}")


def _worker_view(payload: dict[str, Any], view: str, cursor: str | None, max_tokens: int) -> ViewResult:
    if view == "result":
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ArtifactViewError("Worker artifact has no result payload")
        return _bounded_value("result", result, cursor, max_tokens)
    if view == "evidence":
        evidence = _read_items(payload, "evidence")
        return _page_items(evidence, cursor, max_tokens, key="evidence")
    if view == "report":
        report = payload.get("report")
        if report is not None:
            return _bounded_value("report", report, cursor, max_tokens)
        raise ArtifactViewError("Worker artifact has no report")
    if view == "full":
        return _full(payload, cursor, max_tokens)
    if view == "default":
        summary = payload.get("summary") or payload.get("result", {}).get("summary") or ""
        content = {
            "work_unit_id": payload.get("work_unit_id"),
            "status": payload.get("status"),
            "summary": summary,
            "unresolved_questions": payload.get("unresolved_questions", []),
            "citations": payload.get("citations", []),
        }
        return _bounded_content(content, content, cursor, max_tokens)
    raise ArtifactViewError(f"Unsupported worker view: {view}")


def _generic_view(payload: dict[str, Any], view: str, cursor: str | None, max_tokens: int) -> ViewResult:
    if view in {"default", "full"}:
        return _full(payload, cursor, max_tokens)
    raise ArtifactViewError(f"Unsupported view: {view}")


def _full(payload: dict[str, Any], cursor: str | None, max_tokens: int) -> ViewResult:
    return _bounded_content(dict(payload), payload, cursor, max_tokens)


def _bounded_value(
    key: str,
    value: Any,
    cursor: str | None,
    max_tokens: int,
) -> ViewResult:
    return _bounded_content(
        {key: value}, value, cursor, max_tokens, fragment_key=f"{key}_fragment"
    )


def _bounded_content(
    direct_content: dict[str, Any],
    source: Any,
    cursor: str | None,
    max_tokens: int,
    *,
    fragment_key: str = "json_fragment",
) -> ViewResult:
    direct_tokens = count_tokens(json_dumps(direct_content))
    if cursor is None and direct_tokens <= max_tokens:
        return direct_content, None, False, direct_tokens
    if cursor is not None and not cursor.isdigit():
        raise ArtifactViewError("Invalid Artifact cursor")
    rendered = json_dumps(source)
    offset = int(cursor or "0")
    if offset >= len(rendered):
        raise ArtifactViewError("Cursor is outside the Artifact view")

    def build(end: int) -> dict[str, Any]:
        return {
            fragment_key: rendered[offset:end],
            "char_start": offset,
            "char_end": end,
            "total_chars": len(rendered),
        }

    low, high = offset + 1, len(rendered)
    best = offset
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(json_dumps(build(middle))) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == offset:
        raise ArtifactViewError("max_tokens is too small for a JSON fragment")
    content = build(best)
    next_cursor = str(best) if best < len(rendered) else None
    used = count_tokens(json_dumps(content))
    return content, next_cursor, next_cursor is not None, used


_VIEWS: dict[
    ArtifactType, Callable[[dict[str, Any], str, str | None, int], ViewResult]
] = {
    ArtifactType.KNOWLEDGE_SEARCH: _search_view,
    ArtifactType.PAPER_READ: _read_view,
    ArtifactType.PAPER_COMPARISON: _comparison_view,
    ArtifactType.WORKER_RESULT: _worker_view,
    ArtifactType.TOOL_RESULT: _generic_view,
    ArtifactType.RESEARCH_TASK: _generic_view,
    ArtifactType.RETRIEVED_EVIDENCE: _generic_view,
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
