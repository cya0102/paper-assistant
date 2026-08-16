"""Worker registry: two implemented workers and four registered placeholders.

The placeholders (landscape_scout, relation_analyst, contradiction_finder,
cross_domain_analogy_scout) declare their capabilities, allowed tools and
output schema but are explicitly marked not-implemented -- the Scheduler
refuses to run them rather than pretending they work.
"""

from typing import Any

from paper_agent.delegation.registry import WorkerDescriptor, WorkerRegistry

ANALYZER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "workstream": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "claims": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "string"}},
        "unresolved_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["workstream", "findings"],
    "additionalProperties": False,
}

VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "workstream": {"type": "string"},
        "verdict": {
            "type": "string",
            "enum": ["supported", "contradicted", "insufficient", "unreviewed"],
        },
        "findings": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "string"}},
        "unresolved_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["workstream", "verdict", "findings"],
    "additionalProperties": False,
}


def build_worker_registry() -> WorkerRegistry:
    registry = WorkerRegistry()
    registry.register(
        WorkerDescriptor(
            name="paper_analyzer",
            description=(
                "从分配到的论文中提取事实、发现、局限与证据；"
                "只能分析 brief 中 paper_ids 指定的论文。"
            ),
            capabilities=("profile_extraction", "finding_extraction", "limitation_extraction"),
            allowed_tools=("search_knowledge", "read_paper", "read_artifact"),
            output_schema=ANALYZER_SCHEMA,
            default_token_budget=4000,
            default_tool_call_budget=6,
            timeout_seconds=180,
        )
    )
    registry.register(
        WorkerDescriptor(
            name="evidence_verifier",
            description=(
                "核对声明与证据的一致性；verdict 只能是 supported、contradicted、"
                "insufficient 或 unreviewed 之一，禁止输出 verified。"
            ),
            capabilities=("entailment", "verification"),
            allowed_tools=("read_artifact", "read_paper"),
            output_schema=VERIFIER_SCHEMA,
            default_token_budget=4000,
            default_tool_call_budget=4,
            timeout_seconds=180,
        )
    )
    placeholders: tuple[tuple[str, str, tuple[str, ...], dict[str, Any]], ...] = (
        (
            "landscape_scout",
            "扫描某领域内的论文格局、趋势与空白。",
            ("search_knowledge", "read_paper"),
            ANALYZER_SCHEMA,
        ),
        (
            "relation_analyst",
            "分析论文之间/实体之间的关系。",
            ("search_knowledge", "read_paper", "read_artifact"),
            ANALYZER_SCHEMA,
        ),
        (
            "contradiction_finder",
            "寻找论文结论之间的矛盾。",
            ("read_artifact", "read_paper"),
            ANALYZER_SCHEMA,
        ),
        (
            "cross_domain_analogy_scout",
            "跨领域机制类比与可行性探索。",
            ("search_knowledge", "read_paper"),
            ANALYZER_SCHEMA,
        ),
    )
    for name, description, tools, schema in placeholders:
        registry.register(
            WorkerDescriptor(
                name=name,
                description=description,
                capabilities=(name,),
                allowed_tools=tools,
                output_schema=schema,
                default_token_budget=4000,
                default_tool_call_budget=6,
                timeout_seconds=180,
                implemented=False,
            )
        )
    return registry
