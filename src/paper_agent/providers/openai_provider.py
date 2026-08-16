"""OpenAI-compatible Responses and Embeddings API adapters."""

import json
import re
from hashlib import sha256
from html import unescape
from typing import Any, cast

from openai import OpenAI

from paper_agent.agent.prompts import SYSTEM_PROMPT
from paper_agent.domain.agent import AgentCheckpoint, ModelTurn, ToolCall, ToolResult
from paper_agent.domain.indexing import EmbeddingDescriptor


_ROD_TOOL_NAME = "retrieve_and_analyze_knowledge"


def _rod_result(results: tuple[ToolResult, ...]) -> ToolResult | None:
    return next(
        (
            item
            for item in reversed(results)
            if item.name == _ROD_TOOL_NAME and not item.is_error
        ),
        None,
    )


def _rod_finalization_instructions(result: ToolResult) -> str:
    status = str(result.model_payload.get("status") or "")
    citations = tuple(
        item.citation_label for item in result.citation_manifest
    )
    if status == "supported" and citations:
        allowed = ", ".join(f"[{value}]" for value in citations)
        return (
            f"{SYSTEM_PROMPT}\n检索、Offload 和 Worker 分析已经结束。"
            "禁止继续调用工具，只能根据刚返回的 Worker Claim 合成答案。"
            f"可用引用只有：{allowed}。每个论文事实必须紧跟引用。"
        )
    return (
        f"{SYSTEM_PROMPT}\nROD 没有得到充分证据。禁止继续调用工具，"
        "禁止补写论文事实；只输出以 no_evidence 开头的简短说明。"
    )


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
        rod = _rod_result(results)
        if rod is not None:
            response = self._client.responses.create(
                model=self._model,
                previous_response_id=checkpoint.response_id,
                instructions=_rod_finalization_instructions(rod),
                input=outputs,  # type: ignore[arg-type]
            )
        else:
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

    _tool_marker = re.compile(r"<\s*(?:tool_call|function\s*=)", re.IGNORECASE)
    _text_tool_call = re.compile(
        r"<\s*tool_call\s*>\s*"
        r"<\s*function\s*=\s*([A-Za-z_][\w.-]*)\s*>"
        r"(.*?)"
        r"<\s*/\s*function\s*>\s*"
        r"<\s*/\s*tool_call\s*>",
        re.IGNORECASE | re.DOTALL,
    )
    _text_parameter = re.compile(
        r"<\s*parameter\s*=\s*([A-Za-z_][\w.-]*)\s*>"
        r"(.*?)"
        r"<\s*/\s*parameter\s*>",
        re.IGNORECASE | re.DOTALL,
    )

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
        deterministic_rod_turn = self._deterministic_rod_turn(checkpoint, tools)
        if deterministic_rod_turn is not None:
            checkpoint.model_history = self._turn_history(
                response=None,
                turn=deterministic_rod_turn,
            )
            return deterministic_rod_turn
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
        turn = self._turn(response, tools)
        checkpoint.model_history = self._turn_history(response, turn)
        return turn

    @staticmethod
    def _deterministic_rod_turn(
        checkpoint: AgentCheckpoint,
        tools: tuple[dict[str, object], ...],
    ) -> ModelTurn | None:
        """Start the standard ROD path without asking MiMo to route one tool.

        ``retrieve-offload-delegate`` exposes exactly one mandatory composite
        tool.  Asking MiMo to select that already-determined tool adds a model
        call and makes the whole run depend on MiMo's non-native text markup.
        Build the equivalent structured call locally; query rewriting still
        happens inside the composite retrieval service.
        """
        if len(tools) != 1 or tools[0].get("name") != _ROD_TOOL_NAME:
            return None
        query = next(
            (
                item.content.strip()
                for item in reversed(checkpoint.messages)
                if item.role == "user" and item.content.strip()
            ),
            "",
        )
        if not query:
            raise ValueError("ROD requires a non-empty user query")
        digest = sha256(
            f"{checkpoint.session_id}\0{query}".encode("utf-8")
        ).hexdigest()[:24]
        call = ToolCall(
            call_id=f"call_rod_{digest}",
            name=_ROD_TOOL_NAME,
            arguments={"query": query},
        )
        return ModelTurn(
            response_id=f"response_rod_{digest}",
            tool_calls=(call,),
        )

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
        should_finalize = self._should_finalize(results)
        rod = _rod_result(results)
        allow_no_evidence = bool(
            rod is not None
            and rod.model_payload.get("status") != "supported"
        )
        if should_finalize:
            final_input, citations = self._finalization_input(checkpoint, results)
            response = self._create_final_response(
                final_input,
                citations,
                force_no_evidence=allow_no_evidence,
            )
            if not self._is_valid_final_response(
                response, citations, allow_no_evidence=allow_no_evidence
            ):
                rejected_history = self._response_history(response)
                response = self._create_final_response(
                    final_input + rejected_history,
                    citations,
                    retry=True,
                    force_no_evidence=allow_no_evidence,
                )
            if not self._is_valid_final_response(
                response, citations, allow_no_evidence=allow_no_evidence
            ):
                raise RuntimeError(
                    "MiMo failed to produce a cited natural-language answer "
                    "after the finalization retry"
                )
        else:
            response = self._create_response(
                model=self._model,
                instructions=SYSTEM_PROMPT,
                input=history,
                reasoning={"effort": "none"},
                tools=list(tools),
            )
        turn = self._turn(response, () if should_finalize else tools)
        checkpoint.model_history = (
            checkpoint.model_history + outputs + self._turn_history(response, turn)
        )
        return turn

    @classmethod
    def _turn(
        cls,
        response: Any,
        tools: tuple[dict[str, object], ...],
    ) -> ModelTurn:
        """Normalize native and MiMo text-encoded calls into one ModelTurn.

        Some MiMo responses emit the documented tool-call-shaped XML as an
        assistant message instead of a Responses API ``function_call`` item.
        Treating that message as final text prematurely completes the Agent
        loop.  Parse only registered tool names and preserve the call as a
        structured history item for the following stateless request.
        """
        native_calls = cls._native_tool_calls(response)
        if native_calls:
            return ModelTurn(response_id=str(response.id), tool_calls=native_calls)
        output_text = str(getattr(response, "output_text", "")).strip()
        if not output_text:
            raise RuntimeError("Model returned neither tool calls nor text")
        text_calls = cls._parse_text_tool_calls(
            output_text,
            response_id=str(response.id),
            tools=tools,
        )
        if text_calls:
            return ModelTurn(response_id=str(response.id), tool_calls=text_calls)
        if cls._tool_marker.search(output_text):
            preview = " ".join(output_text.split())[:400]
            raise RuntimeError(
                "MiMo returned malformed or unsupported tool-call markup; "
                f"output preview: {preview!r}"
            )
        return ModelTurn(response_id=str(response.id), output_text=output_text)

    @staticmethod
    def _native_tool_calls(response: Any) -> tuple[ToolCall, ...]:
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
        return tuple(calls)

    @classmethod
    def _parse_text_tool_calls(
        cls,
        output_text: str,
        *,
        response_id: str,
        tools: tuple[dict[str, object], ...],
    ) -> tuple[ToolCall, ...]:
        if not cls._tool_marker.search(output_text):
            return ()
        allowed = {
            str(spec.get("name"))
            for spec in tools
            if isinstance(spec.get("name"), str)
        }
        matches = tuple(cls._text_tool_call.finditer(output_text))
        if not matches:
            return ()
        calls: list[ToolCall] = []
        for index, match in enumerate(matches):
            name = match.group(1)
            if name not in allowed:
                raise RuntimeError(f"MiMo requested an unavailable tool: {name}")
            arguments: dict[str, Any] = {}
            body = match.group(2)
            parameters = tuple(cls._text_parameter.finditer(body))
            for parameter in parameters:
                key = parameter.group(1)
                if key in arguments:
                    raise RuntimeError(
                        f"MiMo repeated tool parameter {key!r} for {name}"
                    )
                arguments[key] = cls._decode_text_parameter(parameter.group(2))
            remainder = cls._text_parameter.sub("", body).strip()
            if remainder:
                raise RuntimeError(
                    f"MiMo returned malformed parameters for tool {name}"
                )
            canonical = json.dumps(
                arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            digest = sha256(
                f"{response_id}\0{index}\0{name}\0{canonical}".encode("utf-8")
            ).hexdigest()[:24]
            calls.append(
                ToolCall(
                    call_id=f"call_mimo_{digest}",
                    name=name,
                    arguments=arguments,
                )
            )
        return tuple(calls)

    @staticmethod
    def _decode_text_parameter(value: str) -> Any:
        decoded = unescape(value).strip()
        try:
            return json.loads(decoded)
        except json.JSONDecodeError:
            return decoded

    @classmethod
    def _turn_history(
        cls,
        response: Any,
        turn: ModelTurn,
    ) -> list[dict[str, Any]]:
        if turn.tool_calls:
            return [
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                }
                for call in turn.tool_calls
            ]
        return cls._response_history(response)

    def _create_final_response(
        self,
        history: list[dict[str, Any]],
        citations: tuple[str, ...],
        *,
        retry: bool = False,
        force_no_evidence: bool = False,
    ) -> Any:
        allowed = ", ".join(f"[{value}]" for value in citations)
        citation_rule = (
            f"可用引用编号只有：{allowed}。最终答案必须原样包含至少一个上述编号；"
            "每个事实应紧跟支持它的引用编号。"
            if citations
            else (
                "当前 ROD 结果证据不足。只输出以 no_evidence 开头的简短说明，"
                "不要补写任何论文事实或引用。"
                if force_no_evidence
                else "当前工具结果没有 Evidence 引用编号，不要编造引用。"
            )
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

    @classmethod
    def _should_finalize(cls, results: tuple[ToolResult, ...]) -> bool:
        successful = tuple(item for item in results if not item.is_error)
        if not successful:
            return False
        for item in successful:
            if item.name == "search_knowledge" and item.payload.get(
                "has_sufficient_evidence"
            ) is True:
                return True
            if item.name == "read_paper" and (
                item.payload.get("passages") or item.payload.get("elements")
            ):
                return True
            if item.name in {
                "compare_papers",
                "collect_research_task",
            } and cls._evidence_citations((item,)):
                return True
            if item.name == _ROD_TOOL_NAME and item.payload.get("status") in {
                "supported",
                "no_evidence",
                "insufficient",
                "failed",
            }:
                return True
        return False

    @staticmethod
    def _evidence_citations(results: tuple[ToolResult, ...]) -> tuple[str, ...]:
        citations: list[str] = []
        for result in results:
            citations.extend(
                citation.citation_label for citation in result.citation_manifest
            )
            keys = (
                "selected_evidence",
                "evidence",
                "passages",
                "elements",
                "citations",
                "citation_manifest",
            )
            for key in keys:
                for raw in result.model_payload.get(key, []):
                    if not isinstance(raw, dict):
                        continue
                    citation = (
                        raw.get("citation")
                        or raw.get("citation_label")
                        or raw.get("label")
                    )
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
        # Artifact and delegated-research tools expose their authoritative
        # citations through ToolResult.citation_manifest.  Their bounded model
        # payload contains the hydrated slice or collected summary rather than
        # legacy evidence/passages arrays, so include that payload once and
        # associate it with the remaining manifest labels.
        for result in results:
            if (
                result.name == _ROD_TOOL_NAME
                and not result.citation_manifest
            ):
                rod_content = {
                    "status": result.model_payload.get("status"),
                    "reason": result.model_payload.get("reason"),
                    "summary": result.model_payload.get("summary"),
                    "failed_work_units": result.model_payload.get(
                        "failed_work_units", []
                    ),
                }
                block = (
                    "retrieve_and_analyze_knowledge | 无可用论文证据\n"
                    + json.dumps(
                        rod_content,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
                remaining = character_budget - used
                if remaining > 0:
                    block = block[:remaining]
                    blocks.append(block)
                    used += len(block)
            remaining_refs = tuple(
                ref
                for ref in result.citation_manifest
                if ref.citation_label not in seen
            )
            if not remaining_refs:
                continue
            payload = result.model_payload
            if result.name == "read_artifact":
                content: object = payload.get("content")
            elif result.name == "collect_research_task":
                content = {
                    "summary": payload.get("summary"),
                    "unresolved_questions": payload.get("unresolved_questions", []),
                    "failed_work_units": payload.get("failed_work_units", []),
                }
            else:
                content = payload
            labels = ", ".join(f"[{ref.citation_label}]" for ref in remaining_refs)
            header = f"{result.name} | 可用引用：{labels}\n"
            remaining = character_budget - used - len(header)
            if remaining <= 0:
                continue
            body = json.dumps(
                content, ensure_ascii=False, sort_keys=True, default=str
            )[:remaining]
            blocks.append(header + body)
            used += len(header) + len(body)
            for ref in remaining_refs:
                citations.append(ref.citation_label)
                seen.add(ref.citation_label)
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
    def _is_valid_final_response(
        response: Any,
        citations: tuple[str, ...],
        *,
        allow_no_evidence: bool = False,
    ) -> bool:
        output_text = str(getattr(response, "output_text", ""))
        lowered = output_text.casefold()
        if not output_text.strip() or "<tool_call" in lowered or "<function=" in lowered:
            return False
        # A finalization request always carries citable evidence; an answer
        # without any citation is a contract error, not a legal answer.
        if not citations:
            return allow_no_evidence and "no_evidence" in lowered
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
