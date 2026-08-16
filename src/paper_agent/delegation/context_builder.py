"""WorkerContextBuilder: the only context a Worker receives.

The brief is a self-contained JSON document: objective, paper ids, input
artifact ids, allowed tools, output schema, and budgets.  The main Agent's
conversation history never reaches a Worker.
"""

import json
from typing import Any
from uuid import UUID

from paper_agent.delegation.registry import WorkerDescriptor


class WorkerContextBuilder:
    def build_brief(
        self,
        *,
        descriptor: WorkerDescriptor,
        objective: str,
        paper_ids: tuple[UUID, ...],
        input_artifact_ids: tuple[UUID, ...],
        token_budget: int,
        tool_call_budget: int,
        timeout_seconds: int,
    ) -> str:
        brief = {
            "role": f"worker:{descriptor.name}",
            "description": descriptor.description,
            "objective": objective,
            "paper_ids": [str(value) for value in paper_ids],
            "input_artifact_ids": [str(value) for value in input_artifact_ids],
            "allowed_tools": list(descriptor.allowed_tools),
            "output_schema": descriptor.output_schema,
            "budgets": {
                "token_budget": token_budget,
                "tool_call_budget": tool_call_budget,
                "timeout_seconds": timeout_seconds,
            },
            "rules": [
                "只使用 allowed_tools 中的工具，禁止调用其他工具。",
                "只分析 paper_ids 指定的论文，禁止扩展到其他论文。",
                "最终回答必须是符合 output_schema 的单个 JSON 对象，禁止输出任何额外文本。",
                "每个结论必须引用工具返回的引用编号（[E编号]/[P编号]）。",
                "证据不足时在 findings 中明确说明，禁止编造。",
            ],
        }
        return json.dumps(brief, ensure_ascii=False, indent=2)
