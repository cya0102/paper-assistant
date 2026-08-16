"""WorkerRunner: execute one WorkUnit as an isolated Agent loop.

The Worker never receives the main Agent conversation.  It gets a compact brief,
a locked tool registry (project + paper scope), its own checkpoint namespace,
and a schema-enforcing finalizer.  Its final output is validated, saved as a
Worker Artifact through the ArtifactService, and only a compact WorkerResult
crosses back to the main Agent.
"""

import json
import re
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
from paper_agent.artifacts.tokens import count_tokens
from paper_agent.agent.ports import AgentCheckpointStore, LanguageModel
from paper_agent.domain.agent import ToolCall
from paper_agent.domain.artifact import citation_to_dict
from paper_agent.delegation.context_builder import WorkerContextBuilder
from paper_agent.delegation.registry import WorkerRegistry
from paper_agent.research_tasks.domain import WorkUnit, WorkerResult


class WorkerOutputValidator:
    """Finalizer that requires a schema-valid JSON object as the Worker answer."""

    _citation_pattern = re.compile(r"^\[?([EP]\d+)\]?$")

    def __init__(
        self,
        schema: dict[str, Any] | None,
        *,
        token_budget: int | None = None,
    ) -> None:
        self._schema = schema
        self._token_budget = token_budget

    def __call__(
        self, answer_text: str, tool_results: tuple[Any, ...]
    ) -> str:
        if (
            self._token_budget is not None
            and count_tokens(answer_text) > self._token_budget
        ):
            raise ValueError(
                f"Worker output exceeds token_budget={self._token_budget}"
            )
        try:
            parsed = json.loads(answer_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Worker output is not valid JSON: {error.msg}"
            ) from error
        if not isinstance(parsed, dict):
            raise ValueError("Worker output must be a JSON object")
        self._validate(parsed)
        self._validate_citations(parsed, tool_results)
        if "citations" in parsed:
            parsed["citations"] = list(self._citation_labels(parsed))
        return json.dumps(parsed, ensure_ascii=False)

    def _validate(self, parsed: dict[str, Any]) -> None:
        if self._schema is None:
            return
        self._validate_schema(parsed, self._schema, path="$")

    @classmethod
    def _validate_schema(
        cls, value: Any, schema: dict[str, Any], *, path: str
    ) -> None:
        expected = schema.get("type")
        type_checks: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "object": dict,
            "array": list,
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
        }
        if isinstance(expected, str) and expected in type_checks:
            expected_type = type_checks[expected]
            if not isinstance(value, expected_type) or (
                expected in {"integer", "number"} and isinstance(value, bool)
            ):
                raise ValueError(f"Worker output field {path} must be {expected}")
        allowed = schema.get("enum")
        if allowed is not None and value not in allowed:
            raise ValueError(
                f"Worker output field {path} must be one of {allowed}; got {value!r}"
            )
        if isinstance(value, dict):
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            missing = [key for key in required if key not in value]
            if missing:
                raise ValueError(
                    f"Worker output missing required fields at {path}: "
                    + ", ".join(missing)
                )
            if schema.get("additionalProperties") is False:
                extras = sorted(set(value) - set(properties))
                if extras:
                    raise ValueError(
                        f"Worker output has unexpected fields at {path}: "
                        + ", ".join(extras)
                    )
            for key, child in value.items():
                child_schema = properties.get(key)
                if isinstance(child_schema, dict):
                    cls._validate_schema(child, child_schema, path=f"{path}.{key}")
        if isinstance(value, list) and isinstance(schema.get("items"), dict):
            for index, child in enumerate(value):
                cls._validate_schema(
                    child, schema["items"], path=f"{path}[{index}]"
                )

    @classmethod
    def _citation_labels(cls, parsed: dict[str, Any]) -> tuple[str, ...]:
        labels: list[str] = []
        for raw in parsed.get("citations", []):
            if not isinstance(raw, str):
                raise ValueError("Worker output citations must contain strings")
            match = cls._citation_pattern.fullmatch(raw.strip())
            if match is None:
                raise ValueError(f"Invalid Worker citation label: {raw!r}")
            labels.append(match.group(1))
        return tuple(dict.fromkeys(labels))

    @classmethod
    def _validate_citations(
        cls, parsed: dict[str, Any], tool_results: tuple[Any, ...]
    ) -> None:
        labels = cls._citation_labels(parsed)
        available: dict[
            str,
            tuple[UUID, UUID, UUID | None, UUID | None, UUID | None, str | None],
        ] = {}
        for result in tool_results:
            for citation in result.citation_manifest:
                identity = (
                    citation.paper_id,
                    citation.version_id,
                    citation.section_id,
                    citation.chunk_id,
                    citation.element_id,
                    citation.evidence_hash,
                )
                previous = available.get(citation.citation_label)
                if (
                    citation.citation_label in available
                    and previous != identity
                ):
                    raise ValueError(
                        f"Ambiguous Worker citation label: {citation.citation_label}"
                    )
                available[citation.citation_label] = identity
        unknown = sorted(set(labels) - set(available))
        if unknown:
            raise ValueError(
                "Worker output contains unknown citations: " + ", ".join(unknown)
            )
        findings = parsed.get("findings")
        verdict = parsed.get("verdict")
        if verdict is not None and verdict not in {
            "supported",
            "contradicted",
            "insufficient",
            "unreviewed",
        }:
            raise ValueError(f"Invalid evidence-verification verdict: {verdict!r}")
        requires_evidence = bool(findings) and bool(available)
        requires_evidence = requires_evidence or verdict in {
            "supported",
            "contradicted",
        }
        if requires_evidence and not labels:
            raise ValueError("Evidence-backed Worker output must cite tool evidence")


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
        disallowed_tools = sorted(
            set(work_unit.allowed_tools) - set(descriptor.allowed_tools)
        )
        if disallowed_tools:
            return WorkerResult(
                work_unit_id=work_unit.work_unit_id,
                status="failed",
                summary="",
                error=(
                    f"worker {descriptor.name} cannot use tools: "
                    + ", ".join(disallowed_tools)
                ),
            )
        tools = self._build_tools(work_unit)
        brief = self._context_builder.build_brief(
            descriptor=descriptor,
            objective=work_unit.objective,
            paper_ids=work_unit.paper_ids,
            input_artifact_ids=work_unit.input_artifact_ids,
            allowed_tools=work_unit.allowed_tools,
            output_schema=work_unit.output_schema,
            token_budget=work_unit.token_budget,
            tool_call_budget=work_unit.tool_call_budget,
            timeout_seconds=work_unit.timeout_seconds,
        )
        session_id = uuid5(NAMESPACE_URL, f"worker:{work_unit.work_unit_id}")
        runtime = AgentRuntime(
            self._model,
            tools,
            self._checkpoints,
            answer_finalizer=WorkerOutputValidator(
                work_unit.output_schema, token_budget=work_unit.token_budget
            ),
            materializer=self._materializer,
            max_steps=work_unit.tool_call_budget + 2,
            max_tool_calls=work_unit.tool_call_budget,
            timeout_seconds=work_unit.timeout_seconds,
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
        requested_labels = set(WorkerOutputValidator._citation_labels(parsed))
        citations = [
            citation_to_dict(item)
            for item in self._collect_manifest(answer)
            if item.citation_label in requested_labels
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
            research_task_id=work_unit.task_id,
            created_by=work_unit.requested_worker,
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
                        self._artifacts,
                        work_unit.project_id,
                        allowed_artifact_ids=work_unit.input_artifact_ids,
                    ).contract()
                )
        return registry
