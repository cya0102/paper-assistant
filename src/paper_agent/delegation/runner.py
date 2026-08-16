"""WorkerRunner: execute one WorkUnit as an isolated Agent loop.

The Worker never receives the main Agent conversation.  It gets a compact brief,
a locked tool registry (project + paper scope), its own checkpoint namespace,
and a schema-enforcing finalizer.  Its final output is validated, saved as a
Worker Artifact through the ArtifactService, and only a compact WorkerResult
crosses back to the main Agent.
"""

import json
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from paper_agent.agent.artifact_tool_adapters import ReadArtifactToolAdapter
from paper_agent.agent.runtime import AgentRuntime
from paper_agent.agent.tool_adapters import (
    ReadPaperToolAdapter,
    SearchKnowledgeToolAdapter,
)
from paper_agent.agent.tools import ToolRegistry
from paper_agent.artifacts.materializer import ToolResultMaterializer
from paper_agent.artifacts.ports import ArtifactServicePort
from paper_agent.agent.ports import AgentCheckpointStore, LanguageModel
from paper_agent.domain.agent import ToolCall
from paper_agent.domain.artifact import citation_to_dict
from paper_agent.delegation.context_builder import WorkerContextBuilder
from paper_agent.delegation.registry import WorkerRegistry
from paper_agent.research_tasks.domain import WorkUnit, WorkerResult


class WorkerOutputValidator:
    """Finalizer that requires a schema-valid JSON object as the Worker answer."""

    def __init__(self, schema: dict[str, Any] | None) -> None:
        self._schema = schema

    def __call__(
        self, answer_text: str, tool_results: tuple[Any, ...]
    ) -> str:
        try:
            parsed = json.loads(answer_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Worker output is not valid JSON: {error.msg}"
            ) from error
        if not isinstance(parsed, dict):
            raise ValueError("Worker output must be a JSON object")
        self._validate(parsed)
        return json.dumps(parsed, ensure_ascii=False)

    def _validate(self, parsed: dict[str, Any]) -> None:
        if self._schema is None:
            return
        required = self._schema.get("required", [])
        missing = [key for key in required if key not in parsed]
        if missing:
            raise ValueError(
                f"Worker output missing required fields: {', '.join(missing)}"
            )
        properties = self._schema.get("properties", {})
        for key, spec in properties.items():
            if key not in parsed or not isinstance(spec, dict):
                continue
            allowed = spec.get("enum")
            if allowed is not None and parsed[key] not in allowed:
                raise ValueError(
                    f"Worker output field {key!r} must be one of {allowed}; "
                    f"got {parsed[key]!r}"
                )


class WorkerRunner:
    def __init__(
        self,
        *,
        registry: WorkerRegistry,
        model: LanguageModel,
        checkpoints: AgentCheckpointStore,
        artifacts: ArtifactServicePort,
        materializer: ToolResultMaterializer,
        search_service: Any = None,
        read_service: Any = None,
        context_builder: WorkerContextBuilder | None = None,
    ) -> None:
        self._registry = registry
        self._model = model
        self._checkpoints = checkpoints
        self._artifacts = artifacts
        self._materializer = materializer
        self._search_service = search_service
        self._read_service = read_service
        self._context_builder = context_builder or WorkerContextBuilder()

    def run(self, work_unit: WorkUnit, *, user_id: UUID) -> WorkerResult:
        descriptor = self._registry.require(work_unit.requested_worker)
        if not descriptor.implemented:
            return WorkerResult(
                work_unit_id=work_unit.work_unit_id,
                status="failed",
                summary="",
                error=f"worker {descriptor.name} is not implemented",
            )
        tools = self._build_tools(work_unit)
        brief = self._context_builder.build_brief(
            descriptor=descriptor,
            objective=work_unit.objective,
            paper_ids=work_unit.paper_ids,
            input_artifact_ids=work_unit.input_artifact_ids,
            token_budget=work_unit.token_budget,
            tool_call_budget=work_unit.tool_call_budget,
            timeout_seconds=work_unit.timeout_seconds,
        )
        session_id = uuid5(NAMESPACE_URL, f"worker:{work_unit.work_unit_id}")
        runtime = AgentRuntime(
            self._model,
            tools,
            self._checkpoints,
            answer_finalizer=WorkerOutputValidator(work_unit.output_schema),
            materializer=self._materializer,
            max_steps=work_unit.tool_call_budget,
        )
        try:
            answer = runtime.run(
                session_id=session_id,
                user_id=user_id,
                project_id=work_unit.project_id,
                query=brief,
            )
        except Exception as error:  # noqa: BLE001 - worker failure is a result
            return WorkerResult(
                work_unit_id=work_unit.work_unit_id,
                status="failed",
                summary="",
                error=str(error),
            )
        try:
            parsed = json.loads(answer.text)
            if not isinstance(parsed, dict):
                raise ValueError("Worker output must be a JSON object")
        except (json.JSONDecodeError, ValueError) as error:
            return WorkerResult(
                work_unit_id=work_unit.work_unit_id,
                status="failed",
                summary="",
                error=str(error),
            )
        return self._persist(work_unit, parsed, answer)

    def _persist(
        self,
        work_unit: WorkUnit,
        parsed: dict[str, Any],
        answer: Any,
    ) -> WorkerResult:
        citations = [
            citation_to_dict(item) for item in self._collect_manifest(answer)
        ]
        payload: dict[str, Any] = {
            "work_unit_id": str(work_unit.work_unit_id),
            "work_type": work_unit.work_type,
            "status": "succeeded",
            "summary": self._summarize(parsed),
            "result": parsed,
            "evidence": [
                {"citation": item["citation_label"], **item} for item in citations
            ],
            "report": None,
            "unresolved_questions": list(parsed.get("unresolved_questions", [])),
            "citations": citations,
        }
        compact = self._materializer.materialize(
            project_id=work_unit.project_id,
            session_id=uuid5(NAMESPACE_URL, f"worker:{work_unit.work_unit_id}"),
            work_unit_id=work_unit.work_unit_id,
            call=ToolCall(
                call_id=f"worker:{work_unit.work_unit_id}",
                name="worker_result",
                arguments={},
            ),
            raw_payload=payload,
            always_offload=True,
        )
        return WorkerResult(
            work_unit_id=work_unit.work_unit_id,
            status="succeeded",
            summary=str(parsed.get("summary") or self._summarize(parsed)),
            artifact_ref=compact.artifact_ref,
            citation_manifest=compact.citation_manifest,
            unresolved_questions=tuple(parsed.get("unresolved_questions", [])),
        )

    def _collect_manifest(self, answer: Any) -> tuple[Any, ...]:
        manifest: list[Any] = []
        for result in answer.tool_results:
            manifest.extend(result.citation_manifest)
        return tuple(manifest)

    @staticmethod
    def _summarize(parsed: dict[str, Any]) -> str:
        findings = parsed.get("findings") or parsed.get("verdict") or ""
        if isinstance(findings, list):
            findings = "；".join(str(item) for item in findings[:3])
        return str(findings)[:500]

    def _build_tools(self, work_unit: WorkUnit) -> ToolRegistry:
        registry = ToolRegistry()
        for name in work_unit.allowed_tools:
            if name == "search_knowledge" and self._search_service is not None:
                registry.register(
                    SearchKnowledgeToolAdapter(
                        self._search_service,
                        work_unit.project_id,
                        paper_scope=work_unit.paper_ids or None,
                    ).contract()
                )
            elif name == "read_paper" and self._read_service is not None:
                registry.register(
                    ReadPaperToolAdapter(
                        self._read_service,
                        work_unit.project_id,
                        paper_scope=work_unit.paper_ids or None,
                    ).contract()
                )
            elif name == "read_artifact":
                registry.register(
                    ReadArtifactToolAdapter(
                        self._artifacts, work_unit.project_id
                    ).contract()
                )
        return registry
