import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from paper_agent.application import _language_model
from paper_agent.domain.agent import AgentCheckpoint, ConversationMessage, ToolResult
from paper_agent.providers.openai_provider import MimoResponsesModel


class FakeResponses:
    def __init__(self, *responses):
        self.queued = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.queued.pop(0)


def _function_response(response_id="response-1", call_id="call-1"):
    call = SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name="search_knowledge",
        arguments=json.dumps({"query": "2D-TAN"}),
    )
    return SimpleNamespace(id=response_id, output=(call,), output_text="")


def _text_response(text="2D-TAN 使用二维时间图。[E1]"):
    part = SimpleNamespace(type="output_text", text=text)
    message = SimpleNamespace(type="message", content=(part,))
    return SimpleNamespace(id="response-2", output=(message,), output_text=text)


def _checkpoint():
    return AgentCheckpoint(
        session_id=uuid4(),
        user_id=uuid4(),
        project_id=uuid4(),
        messages=[ConversationMessage("user", "2D-TAN 的主要方法是什么？")],
    )


def _tools():
    return (
        {
            "type": "function",
            "name": "search_knowledge",
            "parameters": {"type": "object"},
        },
    )


def test_mimo_replays_history_without_previous_response_id_and_hides_tools_for_final():
    responses = FakeResponses(_function_response(), _text_response())
    model = MimoResponsesModel(
        model="mimo-v2.5-pro",
        client=SimpleNamespace(responses=responses),
    )
    checkpoint = _checkpoint()

    first = model.start(checkpoint, _tools())
    result = ToolResult(
        call_id=first.tool_calls[0].call_id,
        name="search_knowledge",
        payload={
            "has_sufficient_evidence": True,
            "evidence": [{"citation": "E1", "text": "evidence"}],
        },
    )
    final = model.continue_with_tools(checkpoint, (result,), _tools())

    assert final.output_text == "2D-TAN 使用二维时间图。[E1]"
    second_request = responses.requests[1]
    assert "previous_response_id" not in second_request
    assert "tools" not in second_request
    assert second_request["reasoning"] == {"effort": "none"}
    assert not any(
        item.get("type") in {"function_call", "function_call_output"}
        for item in second_request["input"]
    )
    assert "证据包" in second_request["input"][-1]["content"]
    assert "[E1]" in second_request["input"][-1]["content"]
    assert checkpoint.model_history[-1] == {
        "role": "assistant",
        "content": "2D-TAN 使用二维时间图。[E1]",
    }


def test_mimo_keeps_tools_when_search_has_no_evidence():
    responses = FakeResponses(_function_response(), _function_response("response-2", "call-2"))
    model = MimoResponsesModel(
        model="mimo-v2.5-pro",
        client=SimpleNamespace(responses=responses),
    )
    checkpoint = _checkpoint()
    first = model.start(checkpoint, _tools())

    next_turn = model.continue_with_tools(
        checkpoint,
        (
            ToolResult(
                first.tool_calls[0].call_id,
                "search_knowledge",
                {"has_sufficient_evidence": False, "evidence": []},
            ),
        ),
        _tools(),
    )

    assert next_turn.tool_calls[0].call_id == "call-2"
    assert responses.requests[1]["tools"] == list(_tools())


def test_mimo_retries_final_answer_when_first_answer_omits_citation():
    responses = FakeResponses(
        _function_response(),
        _text_response("2D-TAN 使用二维时间图。"),
        _text_response("2D-TAN 使用二维时间图。[E1]"),
    )
    model = MimoResponsesModel(
        model="mimo-v2.5-pro",
        client=SimpleNamespace(responses=responses),
    )
    checkpoint = _checkpoint()
    first = model.start(checkpoint, _tools())
    result = ToolResult(
        first.tool_calls[0].call_id,
        "search_knowledge",
        {
            "has_sufficient_evidence": True,
            "evidence": [{"citation": "E1", "text": "evidence"}],
        },
    )

    final = model.continue_with_tools(checkpoint, (result,), _tools())

    assert final.output_text == "2D-TAN 使用二维时间图。[E1]"
    assert len(responses.requests) == 3
    assert "[E1]" in responses.requests[1]["instructions"]
    assert "上一次生成因缺少有效引用" in responses.requests[2]["instructions"]
    assert "tools" not in responses.requests[2]
    assert responses.requests[2]["input"][-1] == {
        "role": "assistant",
        "content": "2D-TAN 使用二维时间图。",
    }


def test_mimo_retries_plain_text_tool_call_with_clean_evidence_synthesis():
    responses = FakeResponses(
        _function_response(),
        _text_response(
            "<tool_call><function=search_knowledge>继续检索</function></tool_call>"
        ),
        _text_response("2D-TAN 使用二维时间图联合建模相邻候选。[E1]"),
    )
    model = MimoResponsesModel(
        model="mimo-v2.5-pro",
        client=SimpleNamespace(responses=responses),
    )
    checkpoint = _checkpoint()
    first = model.start(checkpoint, _tools())
    result = ToolResult(
        first.tool_calls[0].call_id,
        "search_knowledge",
        {
            "has_sufficient_evidence": True,
            "evidence": [
                {
                    "citation": "E1",
                    "paper_title": "2D-TAN",
                    "section_path": "Method",
                    "page_start": 2,
                    "page_end": 3,
                    "text": "The method retrieves moments on a 2D temporal map.",
                }
            ],
        },
    )

    final = model.continue_with_tools(checkpoint, (result,), _tools())

    assert final.output_text.endswith("[E1]")
    assert len(responses.requests) == 3
    assert "<tool_call>" in responses.requests[2]["input"][-1]["content"]
    assert "tools" not in responses.requests[1]
    assert "tools" not in responses.requests[2]


def test_mimo_rejects_legacy_checkpoint_without_stateless_history():
    model = MimoResponsesModel(
        model="mimo-v2.5-pro",
        client=SimpleNamespace(responses=FakeResponses()),
    )

    with pytest.raises(RuntimeError, match="new session_id"):
        model.continue_with_tools(
            _checkpoint(),
            (ToolResult("call-1", "search_knowledge", {}),),
            _tools(),
        )


def test_mimo_factory_uses_provider_specific_environment(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    model = _language_model(provider="mimo", model="mimo-v2.5-pro")

    assert isinstance(model, MimoResponsesModel)


def test_mimo_factory_rejects_markdown_base_url(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    monkeypatch.setenv(
        "MIMO_BASE_URL",
        "[https://api.xiaomimimo.com/v1](https://api.xiaomimimo.com/v1)",
    )

    with pytest.raises(ValueError, match="plain http"):
        _language_model(provider="mimo", model="mimo-v2.5-pro")


def test_mimo_merges_search_and_read_evidence_and_requires_both_namespaces():
    responses = FakeResponses(
        _function_response(),
        _text_response("2D-TAN 使用二维时间图。[E1]"),
        _text_response("2D-TAN 使用二维时间图联合建模相邻候选。[E1][P7]"),
    )
    model = MimoResponsesModel(
        model="mimo-v2.5-pro",
        client=SimpleNamespace(responses=responses),
    )
    checkpoint = _checkpoint()
    first = model.start(checkpoint, _tools())
    search_result = ToolResult(
        first.tool_calls[0].call_id,
        "search_knowledge",
        {
            "has_sufficient_evidence": True,
            "evidence": [{"citation": "E1", "paper_title": "P", "section_path": "Method", "page_start": 1, "page_end": 2, "text": "search evidence"}],
        },
    )
    read_result = ToolResult(
        "read-call",
        "read_paper",
        {
            "title": "P",
            "evidence": [{"citation": "P7", "paper_title": "P", "section_path": "Results", "page_start": 3, "page_end": 4, "text": "read evidence"}],
        },
    )

    final = model.continue_with_tools(checkpoint, (search_result, read_result), _tools())

    assert final.output_text == "2D-TAN 使用二维时间图联合建模相邻候选。[E1][P7]"
    assert len(responses.requests) == 3
    pack = responses.requests[1]["input"][-1]["content"]
    assert "[E1]" in pack and "[P7]" in pack
    assert "search evidence" in pack and "read evidence" in pack
    assert "上一次生成因缺少有效引用" in responses.requests[2]["instructions"]
