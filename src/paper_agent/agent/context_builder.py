"""Token-budgeted and per-paper balanced evidence context."""

from collections import defaultdict, deque
from dataclasses import dataclass
import re
from uuid import UUID

from paper_agent.domain.context import BuiltContext, Citation
from paper_agent.domain.retrieval import Evidence
from paper_agent.domain.agent import ToolResult


TOKEN_PATTERN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ContextConfig:
    token_budget: int = 6000
    max_per_paper: int = 4


class ContextBuilder:
    def __init__(self, config: ContextConfig | None = None) -> None:
        self._config = config or ContextConfig()

    def build(self, query: str, evidence: tuple[Evidence, ...], session_context: str = "") -> BuiltContext:
        queues: dict[UUID, deque[Evidence]] = defaultdict(deque)
        for item in sorted(evidence, key=lambda value: value.relevance, reverse=True):
            if len(queues[item.paper_id]) < self._config.max_per_paper:
                queues[item.paper_id].append(item)
        balanced: list[Evidence] = []
        while any(queues.values()):
            for queue in queues.values():
                if queue:
                    balanced.append(queue.popleft())
        prefix = f"问题：{query}\n"
        if session_context:
            prefix += f"会话上下文：{session_context}\n"
        parts = [prefix, "证据（只能引用以下编号）："]
        citations: list[Citation] = []
        used = self._count_tokens("\n".join(parts))
        for item in balanced:
            label = f"E{int(item.evidence_id.hex[:12], 16)}"
            block = (
                f"[{label}] {item.paper_title} | {item.section_path} | "
                f"pp.{item.page_start}-{item.page_end}\n{item.text}"
            )
            cost = self._count_tokens(block)
            if used + cost > self._config.token_budget:
                continue
            parts.append(block)
            citations.append(Citation(label=label, evidence=item))
            used += cost
        return BuiltContext(
            text="\n\n".join(parts),
            citations=tuple(citations),
            token_count=used,
            omitted_evidence=len(evidence) - len(citations),
        )

    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(TOKEN_PATTERN.findall(text))


class CitationFormatter:
    _citation = re.compile(r"\[(E\d+)]")

    def format(self, answer: str, context: BuiltContext) -> str:
        allowed = {item.label: item for item in context.citations}
        used = tuple(dict.fromkeys(self._citation.findall(answer)))
        unknown = tuple(label for label in used if label not in allowed)
        if unknown:
            raise ValueError(f"Answer contains unknown citations: {', '.join(unknown)}")
        if context.citations and not used:
            raise ValueError("Evidence-backed answer must contain at least one citation")
        sources = ["", "来源："]
        for label in used:
            item = allowed[label].evidence
            sources.append(
                f"- [{label}] {item.paper_title}，{item.section_path}，pp.{item.page_start}-{item.page_end}"
            )
        return answer.rstrip() + "\n" + "\n".join(sources) if used else answer.rstrip()


class ToolEvidenceCitationFormatter:
    """Validate model citations against actual Tool payloads and append provenance."""

    _citation = re.compile(r"\[(E\d+)]")

    def __call__(self, answer: str, tool_results: tuple[ToolResult, ...]) -> str:
        allowed: dict[str, dict[str, object]] = {}
        for result in tool_results:
            for raw in result.payload.get("evidence", []):
                if isinstance(raw, dict) and isinstance(raw.get("citation"), str):
                    allowed[raw["citation"]] = raw
        used = tuple(dict.fromkeys(self._citation.findall(answer)))
        unknown = tuple(label for label in used if label not in allowed)
        if unknown:
            raise ValueError(f"Answer contains unknown citations: {', '.join(unknown)}")
        if allowed and not used:
            raise ValueError("Evidence-backed answer must cite the tool Evidence")
        if not used:
            return answer.rstrip()
        sources = ["", "来源："]
        for label in used:
            item = allowed[label]
            sources.append(
                f"- [{label}] {item.get('paper_title')}，{item.get('section_path')}，"
                f"pp.{item.get('page_start')}-{item.get('page_end')}"
            )
        return answer.rstrip() + "\n" + "\n".join(sources)
