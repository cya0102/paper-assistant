"""ResearchPlanner: deterministic WorkUnit templates for known workflows.

Complex requests are decomposed with deterministic templates first; open-ended
questions may use an LLM planner later, but every plan must validate against the
WorkUnit contract.  Each WorkUnit maps to a registered Worker and carries its
own budgets plus a stable generation_key for idempotent execution.
"""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from paper_agent.research_tasks.domain import (
    ResearchTask,
    ResearchTaskType,
    TaskBudget,
    WorkUnit,
    WorkUnitStatus,
    work_unit_generation_key,
)

# Verification-style workstreams map to the evidence_verifier worker; all other
# workstreams map to the paper_analyzer worker in the first deterministic tier.
VERIFICATION_WORKSTREAMS = frozenset(
    {"verification", "evidence_verification", "novelty_check"}
)

COMPARISON_PLAN = ("method", "datasets", "metrics", "results", "limitations", "verification")
SURVEY_PLAN = (
    "retrieval",
    "screening",
    "classification",
    "relation_analysis",
    "synthesis",
    "verification",
)
EXPLORATION_PLAN = (
    "target_problem_analysis",
    "source_domain_search",
    "mechanism_abstraction",
    "assumption_compatibility",
    "novelty_check",
    "evidence_verification",
)


def default_plan_for(task_type: ResearchTaskType) -> tuple[str, ...]:
    if task_type == ResearchTaskType.MULTI_PAPER_COMPARISON:
        return COMPARISON_PLAN
    if task_type == ResearchTaskType.LITERATURE_SURVEY:
        return SURVEY_PLAN
    return EXPLORATION_PLAN


def worker_for_workstream(workstream: str) -> str:
    if workstream in VERIFICATION_WORKSTREAMS:
        return "evidence_verifier"
    return "paper_analyzer"


class ResearchPlanner:
    """Deterministic WorkUnit DAG builder for a ResearchTask."""

    def plan(
        self,
        *,
        task: ResearchTask,
        paper_ids: tuple[UUID, ...],
        input_artifact_ids: tuple[UUID, ...] = (),
        workstreams: tuple[str, ...] | None = None,
    ) -> tuple[WorkUnit, ...]:
        selected = workstreams or task.plan
        units: list[WorkUnit] = []
        for workstream in selected:
            worker = worker_for_workstream(workstream)
            dependencies = (
                tuple(unit.work_unit_id for unit in units)
                if worker == "evidence_verifier"
                else ()
            )
            objective = self._objective_for(
                task.task_type, workstream, paper_ids, task.research_question
            )
            budget = task.budget
            schema = output_schema_for(worker, workstream)
            units.append(
                WorkUnit(
                    work_unit_id=uuid5(
                        NAMESPACE_URL,
                        f"work-unit:{task.task_id}:{workstream}",
                    ),
                    task_id=task.task_id,
                    project_id=task.project_id,
                    work_type=workstream,
                    objective=objective,
                    requested_worker=worker,
                    status=WorkUnitStatus.PENDING,
                    generation_key=work_unit_generation_key(
                        task_id=task.task_id,
                        work_type=workstream,
                        objective=objective,
                        paper_ids=paper_ids,
                        input_artifact_ids=input_artifact_ids,
                        requested_worker=worker,
                        output_schema=schema,
                        dependency_ids=dependencies,
                    ),
                    token_budget=budget.token_budget,
                    tool_call_budget=budget.tool_call_budget,
                    timeout_seconds=budget.timeout_seconds,
                    paper_ids=paper_ids,
                    input_artifact_ids=input_artifact_ids,
                    dependency_ids=dependencies,
                    allowed_tools=allowed_tools_for(worker),
                    output_schema=schema,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
        return tuple(units)

    @staticmethod
    def _objective_for(
        task_type: ResearchTaskType,
        workstream: str,
        paper_ids: tuple[UUID, ...],
        question: str,
    ) -> str:
        if task_type == ResearchTaskType.MULTI_PAPER_COMPARISON:
            if workstream == "verification":
                return (
                    f"核对 {len(paper_ids)} 篇论文对比结论的证据支持情况；"
                    f"原问题：{question}"
                )
            return (
                f"从 {len(paper_ids)} 篇论文中提取 {workstream} 维度的事实与证据；"
                f"原问题：{question}"
            )
        return (
            f"针对研究问题执行 {workstream} 分析（涉及 {len(paper_ids)} 篇论文）："
            f"{question}"
        )


def allowed_tools_for(worker: str) -> tuple[str, ...]:
    if worker == "evidence_verifier":
        return ("read_artifact", "read_paper")
    return ("search_knowledge", "read_paper", "read_artifact")


def output_schema_for(worker: str, workstream: str) -> dict[str, Any]:
    if worker == "evidence_verifier":
        return {
            "type": "object",
            "properties": {
                "workstream": {"type": "string"},
                "verdict": {
                    "type": "string",
                    "enum": [
                        "supported",
                        "contradicted",
                        "insufficient",
                        "unreviewed",
                    ],
                },
                "findings": {"type": "array", "items": {"type": "string"}},
                "citations": {"type": "array", "items": {"type": "string"}},
                "unresolved_questions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["workstream", "verdict", "findings"],
            "additionalProperties": False,
        }
    return {
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
