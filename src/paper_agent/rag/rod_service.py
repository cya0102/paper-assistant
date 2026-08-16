"""Deterministic Retrieve -> Offload -> Delegate -> Collect RAG service."""

from collections import defaultdict, deque
from dataclasses import replace
from datetime import UTC, datetime
import re
from uuid import NAMESPACE_URL, UUID, uuid5

from paper_agent.delegation.scheduler import Scheduler
from paper_agent.domain.retrieval import Evidence, SearchRequest, SearchScope
from paper_agent.rag.collector import RodResultCollector
from paper_agent.rag.domain import (
    AnalystReport,
    RagConfig,
    RagResultStatus,
    RagTraceEvent,
    RetrievedEvidenceArtifact,
    RetrieveOffloadDelegateResult,
)
from paper_agent.rag.evidence_materializer import EvidenceArtifactMaterializer
from paper_agent.rag.planner import RagWorkUnitPlanner
from paper_agent.rag.ports import (
    NullRagTracer,
    RagKnowledgeSearch,
    RagQueryRewriter,
    RagTracer,
)
from paper_agent.research_tasks.domain import (
    ResearchTask,
    ResearchTaskStatus,
    ResearchTaskType,
    TaskBudget,
    WorkUnit,
    task_generation_key,
)
from paper_agent.research_tasks.ports import ResearchTaskRepository


class DeterministicRagQueryRewriter:
    """Offline second-round rewrite using unresolved analyst questions."""

    version = "deterministic-rag-query-rewriter-v1"

    def rewrite(self, query: str, reports: tuple[AnalystReport, ...]) -> str:
        base = re.sub(r"[?？。]+$", "", " ".join(query.split())).strip()
        unresolved = tuple(
            dict.fromkeys(
                question.strip()
                for report in reports
                for question in report.unresolved_questions
                if question.strip()
            )
        )
        focus = " ".join(unresolved[:2])
        if focus and focus.casefold() not in base.casefold():
            return f"{base} {focus}".strip()
        return f"{base} 方法 机制 实验 结果 证据".strip()


class RetrieveOffloadDelegateService:
    def __init__(
        self,
        *,
        search: RagKnowledgeSearch,
        repository: ResearchTaskRepository,
        scheduler: Scheduler,
        evidence_materializer: EvidenceArtifactMaterializer,
        planner: RagWorkUnitPlanner,
        collector: RodResultCollector,
        config: RagConfig | None = None,
        rewriter: RagQueryRewriter | None = None,
        tracer: RagTracer | None = None,
    ) -> None:
        self._search = search
        self._repository = repository
        self._scheduler = scheduler
        self._evidence_materializer = evidence_materializer
        self._planner = planner
        self._collector = collector
        self._config = config or RagConfig()
        self._rewriter = rewriter or DeterministicRagQueryRewriter()
        self._tracer = tracer or NullRagTracer()

    def run(
        self,
        *,
        project_id: UUID,
        user_id: UUID,
        session_id: UUID,
        query: str,
        paper_ids: tuple[UUID, ...] = (),
    ) -> RetrieveOffloadDelegateResult:
        query = " ".join(query.split())
        if not query:
            raise ValueError("query cannot be blank")
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper_ids must be unique")
        task = self._get_or_create_task(
            project_id=project_id,
            user_id=user_id,
            session_id=session_id,
            query=query,
            paper_ids=paper_ids,
        )
        current_query = query
        all_evidence: list[RetrievedEvidenceArtifact] = []
        latest_collection = None
        rounds_executed = 0
        for round_index in range(1, self._config.max_rounds + 1):
            rounds_executed = round_index
            self._emit(
                "rag.retrieve.started",
                task=task,
                round_index=round_index,
                details={"query": current_query},
            )
            search_result = self._search.search_knowledge(
                SearchRequest(
                    query=current_query,
                    scope=SearchScope(
                        project_id=project_id,
                        paper_ids=paper_ids,
                    ),
                    max_evidence=self._config.max_evidence,
                )
            )
            selected = self._select_evidence(search_result.evidence)
            self._emit(
                "rag.retrieve.completed",
                task=task,
                round_index=round_index,
                details={
                    "status": search_result.status.value,
                    "selected": len(selected),
                },
            )
            if not selected:
                task = self._finish_without_workers(task)
                status = (
                    RagResultStatus.NO_EVIDENCE
                    if round_index == 1
                    else RagResultStatus.INSUFFICIENT
                )
                result = RetrieveOffloadDelegateResult(
                    task_id=task.task_id,
                    status=status,
                    original_query=query,
                    final_query=current_query,
                    rounds_executed=round_index,
                    evidence_artifacts=tuple(all_evidence),
                    reports=(
                        latest_collection.reports
                        if latest_collection is not None
                        else ()
                    ),
                    citation_manifest=(),
                    failures=(
                        latest_collection.failures
                        if latest_collection is not None
                        else ()
                    ),
                    reason=(
                        search_result.reason
                        or "No retrieved Evidence passed selection."
                    ),
                )
                self._emit_sufficiency(task, round_index, result)
                return result
            materialized = tuple(
                self._evidence_materializer.materialize(
                    project_id=project_id,
                    session_id=session_id,
                    task_id=task.task_id,
                    query=current_query,
                    round_index=round_index,
                    evidence=evidence,
                )
                for evidence in selected
            )
            all_evidence.extend(materialized)
            for evidence in materialized:
                self._emit(
                    "rag.artifact.created",
                    task=task,
                    round_index=round_index,
                    details={
                        "artifact_id": str(evidence.artifact_ref.artifact_id),
                        "citation": evidence.citation.citation_label,
                    },
                )
            planned = self._planner.plan(
                task=task,
                query=current_query,
                round_index=round_index,
                evidence_artifacts=materialized,
            )
            for unit in planned:
                self._repository.save_work_unit(unit)
            task = replace(
                task,
                status=ResearchTaskStatus.PLANNED,
                updated_at=datetime.now(UTC),
            )
            task = self._repository.save_task(task)
            units = self._repository.list_work_units(project_id, task.task_id)
            self._emit(
                "rag.delegate.started",
                task=task,
                round_index=round_index,
                details={"work_units": len(planned)},
            )
            task, units = self._scheduler.run(
                task=task,
                units=units,
                repository=self._repository,
                user_id=user_id,
                on_event=lambda event, unit: self._scheduler_event(
                    event, task, round_index, unit
                ),
            )
            latest_collection = self._collector.collect(
                project_id=project_id,
                units=units,
            )
            self._emit(
                "rag.collect.completed",
                task=task,
                round_index=round_index,
                details={
                    "reports": len(latest_collection.reports),
                    "failures": len(latest_collection.failures),
                },
            )
            if latest_collection.sufficient:
                result = RetrieveOffloadDelegateResult(
                    task_id=task.task_id,
                    status=RagResultStatus.SUPPORTED,
                    original_query=query,
                    final_query=current_query,
                    rounds_executed=round_index,
                    evidence_artifacts=tuple(all_evidence),
                    reports=latest_collection.reports,
                    citation_manifest=latest_collection.citation_manifest,
                    failures=latest_collection.failures,
                )
                self._emit_sufficiency(task, round_index, result)
                self._emit(
                    "rag.synthesis.started",
                    task=task,
                    round_index=round_index,
                    details={"citations": len(result.citation_manifest)},
                )
                return result
            if latest_collection.all_workers_failed:
                result = RetrieveOffloadDelegateResult(
                    task_id=task.task_id,
                    status=RagResultStatus.FAILED,
                    original_query=query,
                    final_query=current_query,
                    rounds_executed=round_index,
                    evidence_artifacts=tuple(all_evidence),
                    reports=(),
                    citation_manifest=(),
                    failures=latest_collection.failures,
                    reason="All chunk analysts failed after their retry budget.",
                )
                self._emit_sufficiency(task, round_index, result)
                return result
            if round_index < self._config.max_rounds:
                rewritten = self._rewriter.rewrite(
                    current_query, latest_collection.reports
                ).strip()
                if rewritten and rewritten != current_query:
                    current_query = rewritten
                    continue
            break
        assert latest_collection is not None
        result = RetrieveOffloadDelegateResult(
            task_id=task.task_id,
            status=RagResultStatus.INSUFFICIENT,
            original_query=query,
            final_query=current_query,
            rounds_executed=rounds_executed,
            evidence_artifacts=tuple(all_evidence),
            reports=latest_collection.reports,
            citation_manifest=(),
            failures=latest_collection.failures,
            reason="Chunk analyst reports did not contain a supported Claim.",
        )
        self._emit_sufficiency(task, result.rounds_executed, result)
        self._emit(
            "rag.synthesis.started",
            task=task,
            round_index=result.rounds_executed,
            details={"citations": 0, "status": result.status.value},
        )
        return result

    def _get_or_create_task(
        self,
        *,
        project_id: UUID,
        user_id: UUID,
        session_id: UUID,
        query: str,
        paper_ids: tuple[UUID, ...],
    ) -> ResearchTask:
        plan = ("retrieve", "chunk_analysis", "collect")
        generation_key = task_generation_key(
            project_id=project_id,
            user_id=user_id,
            research_question=query,
            task_type=ResearchTaskType.RAG_EVIDENCE_ANALYSIS,
            plan=plan,
            paper_ids=paper_ids,
            session_id=session_id,
        )
        existing = self._repository.find_task_by_generation_key(
            project_id, generation_key
        )
        if existing is not None:
            return existing
        now = datetime.now(UTC)
        return self._repository.save_task(
            ResearchTask(
                task_id=uuid5(NAMESPACE_URL, f"rag-task:{generation_key}"),
                project_id=project_id,
                user_id=user_id,
                session_id=session_id,
                research_question=query,
                task_type=ResearchTaskType.RAG_EVIDENCE_ANALYSIS,
                status=ResearchTaskStatus.CREATED,
                plan=plan,
                budget=TaskBudget(
                    max_workers=self._config.max_workers,
                    token_budget=self._config.worker_token_budget,
                    tool_call_budget=self._config.worker_tool_call_budget,
                    timeout_seconds=self._config.worker_timeout_seconds,
                    max_attempts=2,
                ),
                generation_key=generation_key,
                created_at=now,
                updated_at=now,
            )
        )

    def _select_evidence(
        self, evidence: tuple[Evidence, ...]
    ) -> tuple[Evidence, ...]:
        queues: dict[UUID, deque[Evidence]] = defaultdict(deque)
        seen_chunks: set[UUID] = set()
        for item in sorted(evidence, key=lambda value: value.relevance, reverse=True):
            if item.chunk_id in seen_chunks:
                continue
            seen_chunks.add(item.chunk_id)
            if len(queues[item.paper_id]) < self._config.max_per_paper:
                queues[item.paper_id].append(item)
        selected: list[Evidence] = []
        while len(selected) < self._config.max_evidence and any(queues.values()):
            for queue in queues.values():
                if queue and len(selected) < self._config.max_evidence:
                    selected.append(queue.popleft())
        return tuple(selected)

    def _finish_without_workers(self, task: ResearchTask) -> ResearchTask:
        if task.status in {
            ResearchTaskStatus.COMPLETED,
            ResearchTaskStatus.PARTIALLY_COMPLETED,
            ResearchTaskStatus.FAILED,
        }:
            return task
        return self._repository.save_task(
            replace(
                task,
                status=ResearchTaskStatus.COMPLETED,
                updated_at=datetime.now(UTC),
            )
        )

    def _emit_sufficiency(
        self,
        task: ResearchTask,
        round_index: int,
        result: RetrieveOffloadDelegateResult,
    ) -> None:
        self._emit(
            "rag.sufficiency.checked",
            task=task,
            round_index=round_index,
            details={
                "status": result.status.value,
                "sufficient": result.has_sufficient_evidence,
            },
        )

    def _scheduler_event(
        self,
        event: str,
        task: ResearchTask,
        round_index: int,
        unit: WorkUnit,
    ) -> None:
        self._emit(
            f"rag.{event}",
            task=task,
            round_index=round_index,
            work_unit_id=unit.work_unit_id,
            details={"status": unit.status.value},
        )

    def _emit(
        self,
        event: str,
        *,
        task: ResearchTask,
        round_index: int | None = None,
        work_unit_id: UUID | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        self._tracer.emit(
            RagTraceEvent(
                event=event,
                task_id=task.task_id,
                round_index=round_index,
                work_unit_id=work_unit_id,
                details=dict(details or {}),
            )
        )
