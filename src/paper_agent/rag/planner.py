"""Deterministic one-Artifact/one-WorkUnit RAG planner."""

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from paper_agent.rag.domain import RagConfig, RetrievedEvidenceArtifact
from paper_agent.research_tasks.domain import (
    ResearchTask,
    WorkUnit,
    WorkUnitStatus,
    work_unit_generation_key,
)
from paper_agent.workers.chunk_analyst import CHUNK_ANALYST_SCHEMA


class RagWorkUnitPlanner:
    def __init__(self, config: RagConfig) -> None:
        self._config = config

    def plan(
        self,
        *,
        task: ResearchTask,
        query: str,
        round_index: int,
        evidence_artifacts: tuple[RetrievedEvidenceArtifact, ...],
    ) -> tuple[WorkUnit, ...]:
        units: list[WorkUnit] = []
        for evidence in evidence_artifacts:
            artifact_id = evidence.artifact_ref.artifact_id
            objective = (
                "读取唯一获授权的 Evidence Artifact，判断它是否直接回答问题，"
                "并只输出该 Chunk 能支持的 Claim。"
                f"问题：{query}"
            )
            unit_id = uuid5(
                NAMESPACE_URL,
                f"rag-chunk-analysis:{task.task_id}:{artifact_id}",
            )
            generation_key = work_unit_generation_key(
                task_id=task.task_id,
                work_type="chunk_analysis",
                objective=objective,
                paper_ids=(evidence.paper_id,),
                input_artifact_ids=(artifact_id,),
                requested_worker="chunk_analyst",
                output_schema=CHUNK_ANALYST_SCHEMA,
            )
            units.append(
                WorkUnit(
                    work_unit_id=unit_id,
                    task_id=task.task_id,
                    project_id=task.project_id,
                    work_type="chunk_analysis",
                    objective=objective,
                    requested_worker="chunk_analyst",
                    status=WorkUnitStatus.PENDING,
                    generation_key=generation_key,
                    token_budget=self._config.worker_token_budget,
                    tool_call_budget=self._config.worker_tool_call_budget,
                    timeout_seconds=self._config.worker_timeout_seconds,
                    paper_ids=(evidence.paper_id,),
                    input_artifact_ids=(artifact_id,),
                    allowed_tools=("read_artifact",),
                    output_schema=CHUNK_ANALYST_SCHEMA,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
        return tuple(units)
