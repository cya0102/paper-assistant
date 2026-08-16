"""OpenAI-compatible Responses and Embeddings API adapters."""

import json
from typing import Any, cast

from openai import OpenAI

from paper_agent.agent.prompts import SYSTEM_PROMPT
from paper_agent.domain.agent import AgentCheckpoint, ModelTurn, ToolCall, ToolResult
from paper_agent.domain.indexing import EmbeddingDescriptor


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimension: int = 256,
        client: OpenAI | None = None,
    ) -> None:
        self._client = client or OpenAI()
        self._model = model
        self._dimension = dimension
        self.descriptor = EmbeddingDescriptor(
            provider="openai",
            model=model,
            version="openai-embeddings-v1",
            dimension=dimension,
        )

    def embed_batch(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
            dimensions=self._dimension,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return tuple(tuple(float(value) for value in item.embedding) for item in ordered)


class OpenAIResponsesModel:
    def __init__(self, *, model: str, client: OpenAI | None = None) -> None:
        self._model = model
        self._client = client or OpenAI()

    def start(self, checkpoint: AgentCheckpoint, tools: tuple[dict[str, object], ...]) -> ModelTurn:
        model_input: Any = [
            {"role": item.role, "content": item.content} for item in checkpoint.messages
        ]
        response = self._client.responses.create(
            model=self._model,
            instructions=SYSTEM_PROMPT,
            input=model_input,
            tools=list(tools),  # type: ignore[arg-type]
        )
        return self._turn(response)

    def continue_with_tools(
        self,
        checkpoint: AgentCheckpoint,
        results: tuple[ToolResult, ...],
        tools: tuple[dict[str, object], ...],
    ) -> ModelTurn:
        if not checkpoint.response_id:
            raise ValueError("Cannot continue without previous response_id")
        outputs = [
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(item.payload, ensure_ascii=False),
            }
            for item in results
        ]
        response = self._client.responses.create(
            model=self._model,
            previous_response_id=checkpoint.response_id,
            input=outputs,  # type: ignore[arg-type]
            tools=list(tools),  # type: ignore[arg-type]
        )
        return self._turn(response)

    @staticmethod
    def _turn(response: Any) -> ModelTurn:
        calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            arguments = json.loads(item.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            calls.append(
                ToolCall(
                    call_id=str(item.call_id),
                    name=str(item.name),
                    arguments=arguments,
                )
            )
        if calls:
            return ModelTurn(response_id=str(response.id), tool_calls=tuple(calls))
        output_text = str(response.output_text).strip()
        if not output_text:
            raise RuntimeError("Model returned neither tool calls nor text")
        return ModelTurn(response_id=str(response.id), output_text=output_text)


class MimoResponsesModel:
    """Stateless Xiaomi MiMo Responses adapter with recoverable tool history."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.xiaomimimo.com/v1",
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        self._client = client or OpenAI(api_key=api_key, base_url=base_url)

    def start(
        self,
        checkpoint: AgentCheckpoint,
        tools: tuple[dict[str, object], ...],
    ) -> ModelTurn:
        model_input: Any = [
            {"role": item.role, "content": item.content} for item in checkpoint.messages
        ]
        response = self._create_response(
            model=self._model,
            instructions=SYSTEM_PROMPT,
            input=model_input,
            reasoning={"effort": "none"},
            tools=list(tools),
        )
        checkpoint.model_history = self._response_history(response)
        return OpenAIResponsesModel._turn(response)

    def continue_with_tools(
        self,
        checkpoint: AgentCheckpoint,
        results: tuple[ToolResult, ...],
        tools: tuple[dict[str, object], ...],
    ) -> ModelTurn:
        if not checkpoint.model_history:
            raise RuntimeError(
                "MiMo cannot resume a legacy checkpoint without model history; "
                "start a new session_id"
            )
        outputs = [
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(item.payload, ensure_ascii=False),
            }
            for item in results
        ]
        history = [
            {"role": item.role, "content": item.content} for item in checkpoint.messages
        ] + checkpoint.model_history + outputs
        if self._should_finalize(results):
            final_input, citations = self._finalization_input(checkpoint, results)
            response = self._create_final_response(final_input, citations)
            if not self._is_valid_final_response(response, citations):
                rejected_history = self._response_history(response)
                response = self._create_final_response(
                    final_input + rejected_history,
                    citations,
                    retry=True,
                )
        else:
            response = self._create_response(
                model=self._model,
                instructions=SYSTEM_PROMPT,
                input=history,
                reasoning={"effort": "none"},
                tools=list(tools),
            )
        checkpoint.model_history = (
            checkpoint.model_history + outputs + self._response_history(response)
        )
        return OpenAIResponsesModel._turn(response)

    def _create_final_response(
        self,
        history: list[dict[str, Any]],
        citations: tuple[str, ...],
        *,
        retry: bool = False,
    ) -> Any:
        allowed = ", ".join(f"[{value}]" for value in citations)
        citation_rule = (
            f"可用引用编号只有：{allowed}。最终答案必须原样包含至少一个上述编号；"
            "每个事实应紧跟支持它的引用编号。"
            if citations
            else "当前工具结果没有 Evidence 引用编号，不要编造引用。"
        )
        retry_rule = (
            "上一次生成因缺少有效引用而不合格。请重新生成完整答案。"
            if retry
            else ""
        )
        return self._create_response(
            model=self._model,
            instructions=(
                f"{SYSTEM_PROMPT}\n工具调用已经完成。不要再请求或调用工具。"
                "只输出面向用户的自然语言最终答案；禁止输出 <tool_call>、"
                "<function=...> 或其他工具调用标记。"
                f"{citation_rule}{retry_rule}"
            ),
            input=history,
            reasoning={"effort": "none"},
        )

    def _create_response(self, **kwargs: Any) -> Any:
        """Call a compatible endpoint whose accepted fields may lag SDK typing."""
        responses = cast(Any, self._client.responses)
        return responses.create(**kwargs)

    @staticmethod
    def _should_finalize(results: tuple[ToolResult, ...]) -> bool:
        successful = tuple(item for item in results if not item.is_error)
        if not successful:
            return False
        for item in successful:
            if item.name == "search_knowledge" and item.payload.get(
                "has_sufficient_evidence"
            ) is True:
                return True
            # Read results are only citable through their unified "evidence",
            # so a passages/elements-only payload must NOT finalize without
            # any citation (it would pass the citation check vacuously).
            if item.name == "read_paper" and item.payload.get("evidence"):
                return True
        return False

    @staticmethod
    def _evidence_citations(results: tuple[ToolResult, ...]) -> tuple[str, ...]:
        citations: list[str] = []
        for result in results:
            keys = ("selected_evidence", "evidence", "passages", "elements")
            for key in keys:
                for raw in result.model_payload.get(key, []):
                    if not isinstance(raw, dict):
                        continue
                    citation = raw.get("citation")
                    if isinstance(citation, str) and citation:
                        citations.append(citation)
        return tuple(dict.fromkeys(citations))

    @classmethod
    def _finalization_input(
        cls,
        checkpoint: AgentCheckpoint,
        results: tuple[ToolResult, ...],
        *,
        character_budget: int = 32_000,
    ) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        """Build a clean evidence-only synthesis turn without tool-call history.

        MiMo can imitate another tool call as plain text when function-call history is
        included in a request that no longer exposes tools.  Final synthesis therefore
        keeps the user conversation but replaces protocol history with a bounded,
        human-readable evidence pack.
        """
        model_input: list[dict[str, Any]] = [
            {"role": item.role, "content": item.content}
            for item in checkpoint.messages
        ]
        blocks: list[str] = []
        citations: list[str] = []
        seen: set[str] = set()
        used = 0
        # Merge Search Evidence and Read Evidence (passages/elements) into one
        # budgeted pack. The seen set deduplicates entries that appear both in
        # the unified "evidence" list and in the legacy "passages"/"elements".
        for result in results:
            keys = ("selected_evidence", "evidence", "passages", "elements")
            for key in keys:
                for raw in result.model_payload.get(key, []):
                    if not isinstance(raw, dict):
                        continue
                    citation = raw.get("citation")
                    if not isinstance(citation, str) or not citation or citation in seen:
                        continue
                    paper_title = raw.get("paper_title") or result.model_payload.get("title")
                    header = (
                        f"[{citation}] {paper_title} | "
                        f"{raw.get('section_path')} | "
                        f"pp.{raw.get('page_start')}-{raw.get('page_end')}\n"
                    )
                    remaining = character_budget - used - len(header)
                    if remaining <= 0:
                        continue
                    body = str(
                        raw.get("text") or raw.get("content") or raw.get("caption") or ""
                    )[:remaining]
                    block = header + body
                    blocks.append(block)
                    citations.append(citation)
                    seen.add(citation)
                    used += len(block)
        evidence_pack = "\n\n".join(blocks) or "（工具没有返回可用证据。）"
        model_input.append(
            {
                "role": "user",
                "content": (
                    "检索和阅读阶段已经结束，禁止继续调用或模拟任何工具。"
                    "请现在直接回答最初的问题，并且只能依据下面的证据包。"
                    "引用时必须原样复制证据包中每条内容前的引用编号（检索证据为 [E编号]，阅读段落为 [P编号]）。\n\n"
                    f"证据包：\n{evidence_pack}"
                ),
            }
        )
        return model_input, tuple(citations)

    @staticmethod
    def _is_valid_final_response(response: Any, citations: tuple[str, ...]) -> bool:
        output_text = str(getattr(response, "output_text", ""))
        lowered = output_text.casefold()
        if not output_text.strip() or "<tool_call" in lowered or "<function=" in lowered:
            return False
        # A finalization request always carries citable evidence; an answer
        # without any citation is a contract error, not a legal answer.
        if not citations:
            return False
        present = {citation[0] for citation in citations}
        return all(
            any(
                f"[{citation}]" in output_text
                for citation in citations
                if citation[0] == prefix
            )
            for prefix in present
        )

    @staticmethod
    def _response_history(response: Any) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                history.append(
                    {
                        "type": "function_call",
                        "call_id": str(item.call_id),
                        "name": str(item.name),
                        "arguments": str(item.arguments),
                    }
                )
                continue
            if item_type != "message":
                continue
            text_parts = [
                str(part.text)
                for part in getattr(item, "content", ())
                if getattr(part, "type", None) == "output_text"
                and getattr(part, "text", None)
            ]
            if text_parts:
                history.append(
                    {"role": "assistant", "content": "\n".join(text_parts)}
                )
        return history