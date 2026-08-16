"""Final answer policy for the standard ROD path."""

from paper_agent.agent.context_builder import ToolEvidenceCitationFormatter
from paper_agent.domain.agent import ToolResult
from paper_agent.rag.domain import RagResultStatus


class RetrieveOffloadDelegateAnswerFinalizer:
    def __init__(self) -> None:
        self._citations = ToolEvidenceCitationFormatter()

    def __call__(
        self, answer: str, tool_results: tuple[ToolResult, ...]
    ) -> str:
        rod = next(
            (
                item
                for item in reversed(tool_results)
                if item.name == "retrieve_and_analyze_knowledge"
                and not item.is_error
            ),
            None,
        )
        if rod is None:
            raise ValueError(
                "Standard RAG answer requires retrieve_and_analyze_knowledge"
            )
        status = str(rod.model_payload.get("status") or "")
        if status == RagResultStatus.SUPPORTED.value:
            return self._citations(answer, tool_results)
        reason = str(rod.model_payload.get("reason") or status or "no evidence")
        # Never retain model-authored paper facts after an insufficient ROD run.
        return f"no_evidence：{reason}"
