from pathlib import Path
from uuid import uuid4

import pytest

from paper_agent.artifacts.materializer import (
    ToolResultMaterializer,
    extract_citation_manifest,
)
from paper_agent.artifacts.policies import OffloadPolicy, OffloadPolicyConfig
from paper_agent.artifacts.service import ArtifactService
from paper_agent.artifacts.tokens import count_tokens
from paper_agent.domain.agent import ToolCall
from paper_agent.domain.artifact import ArtifactType
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore
from tests.unit.artifacts.test_artifact_service import MemoryArtifactRepository


@pytest.fixture()
def materializer(tmp_path: Path) -> ToolResultMaterializer:
    service = ArtifactService(
        LocalArtifactBlobStore(tmp_path),
        MemoryArtifactRepository(),
    )
    policy = OffloadPolicy(
        OffloadPolicyConfig(
            max_inline_tokens_per_result=800,
            max_total_tool_tokens=1600,
            preview_tokens=60,
        )
    )
    return ToolResultMaterializer(service, policy)


def _search_payload(n_evidence: int) -> dict:
    paper_id = uuid4()
    return {
        "query": "how",
        "status": "ok",
        "has_sufficient_evidence": True,
        "summary": f"{n_evidence} evidence",
        "resolved_papers": [
            {"paper_id": str(paper_id), "version_id": str(uuid4()), "title": "P", "score": 1.0}
        ],
        "evidence": [
            {
                "citation": f"E{i}",
                "evidence_id": str(uuid4()),
                "chunk_id": str(uuid4()),
                "paper_id": str(paper_id),
                "version_id": str(uuid4()),
                "paper_title": "Paper",
                "section_id": str(uuid4()),
                "section_path": "Method",
                "page_start": 1,
                "page_end": 2,
                "text": "evidence " + "word " * 30,
                "relevance": 0.9,
            }
            for i in range(n_evidence)
        ],
    }


def test_small_result_stays_inline(materializer: ToolResultMaterializer) -> None:
    result = materializer.materialize(
        project_id=uuid4(),
        session_id=uuid4(),
        call=ToolCall("call-1", "search_knowledge", {}),
        raw_payload=_search_payload(1),
    )
    assert result.artifact_ref is None
    assert result.model_payload["status"] == "ok"
    assert len(result.citation_manifest) == 1


def test_large_result_offloads_and_keeps_compact_payload(
    materializer: ToolResultMaterializer,
) -> None:
    result = materializer.materialize(
        project_id=uuid4(),
        session_id=uuid4(),
        call=ToolCall("call-2", "search_knowledge", {}),
        raw_payload=_search_payload(10),
    )
    assert result.artifact_ref is not None
    assert result.model_payload["artifact_ref"] is not None
    assert result.model_payload["omitted_evidence"] > 0
    # The compact payload respects the per-result budget
    rendered = str(result.model_payload)
    assert count_tokens(rendered) < 2000
    # The full manifest survives offload
    assert len(result.citation_manifest) == 10


def test_manifest_built_before_offload(materializer: ToolResultMaterializer) -> None:
    payload = _search_payload(3)
    manifest = extract_citation_manifest("search_knowledge", payload)
    assert [item.citation_label for item in manifest] == ["E0", "E1", "E2"]
    assert manifest[0].evidence_hash is not None


@pytest.mark.parametrize(
    ("tool_name", "payload", "required_fields"),
    [
        (
            "delegate_research",
            {
                "delegated": True,
                "task_id": str(uuid4()),
                "status": "completed",
                "detail": "word " * 1000,
            },
            ("delegated", "task_id", "status", "artifact_ref"),
        ),
        (
            "collect_research_task",
            {
                "task_id": str(uuid4()),
                "status": "completed",
                "summary": "word " * 1000,
                "artifact_refs": [],
                "unresolved_questions": ["Q1"],
                "failed_work_units": [],
            },
            ("task_id", "status", "artifact_ref"),
        ),
    ],
)
def test_offloaded_control_tools_keep_routing_fields(
    materializer: ToolResultMaterializer,
    tool_name: str,
    payload: dict,
    required_fields: tuple[str, ...],
) -> None:
    result = materializer.materialize(
        project_id=uuid4(),
        session_id=uuid4(),
        call=ToolCall("control", tool_name, {}),
        raw_payload=payload,
    )
    assert result.artifact_ref is not None
    assert all(field in result.model_payload for field in required_fields)


def test_read_manifest_and_no_duplicate_text(materializer: ToolResultMaterializer) -> None:
    paper_id, version_id = uuid4(), uuid4()
    payload = {
        "paper_id": str(paper_id),
        "version_id": str(version_id),
        "title": "Paper",
        "passages": [
            {
                "citation": "P1",
                "chunk_id": str(uuid4()),
                "section_id": str(uuid4()),
                "section_path": "Method",
                "page_start": 1,
                "page_end": 2,
                "text": "the method section",
            }
        ],
        "elements": [],
    }
    manifest = extract_citation_manifest("read_paper", payload)
    assert manifest[0].citation_label == "P1"
    assert manifest[0].paper_id == paper_id
    # No duplicated unified evidence key in the compact model payload
    result = materializer.materialize(
        project_id=uuid4(),
        session_id=uuid4(),
        call=ToolCall("call-3", "read_paper", {}),
        raw_payload=payload,
    )
    assert "evidence" not in result.model_payload
    # A multi-passage full-section read always offloads
    many = dict(payload)
    many["passages"] = [
        {
            "citation": f"P{i}",
            "chunk_id": str(uuid4()),
            "section_id": str(uuid4()),
            "section_path": "Method",
            "page_start": 1,
            "page_end": 2,
            "text": "the method section " + "x " * 20,
        }
        for i in range(5)
    ]
    big_result = materializer.materialize(
        project_id=uuid4(),
        session_id=uuid4(),
        call=ToolCall("call-3b", "read_paper", {}),
        raw_payload=many,
    )
    assert big_result.artifact_ref is not None
    assert big_result.model_payload["omitted_passages"] > 0


def test_compare_manifest(materializer: ToolResultMaterializer) -> None:
    paper_a, paper_b = uuid4(), uuid4()
    payload = {
        "status": "complete",
        "paper_ids": [str(paper_a), str(paper_b)],
        "dimensions": [
            {
                "name": "method",
                "directly_comparable": True,
                "cells": [
                    {
                        "paper_id": str(paper_a),
                        "paper_title": "A",
                        "normalized_value": "X",
                    },
                    {
                        "paper_id": str(paper_b),
                        "paper_title": "B",
                        "normalized_value": "Y",
                    },
                ],
            }
        ],
        "evidence": [
            {
                "citation": "E1",
                "evidence_id": str(uuid4()),
                "paper_id": str(paper_a),
                "version_id": str(uuid4()),
                "paper_title": "A",
                "section_path": "Research Graph > method",
                "pages": [1, 2],
                "evidence_text": "evidence",
            }
        ],
    }
    manifest = extract_citation_manifest("compare_papers", payload)
    assert len(manifest) == 1 and manifest[0].citation_label == "E1"
    result = materializer.materialize(
        project_id=uuid4(),
        session_id=uuid4(),
        call=ToolCall("call-4", "compare_papers", {}),
        raw_payload=payload,
    )
    assert result.model_payload["status"] == "complete"
    assert result.model_payload["paper_count"] == 2
    # six-paper comparisons always offload
    six = dict(payload)
    six["paper_ids"] = [str(uuid4()) for _ in range(6)]
    big = materializer.materialize(
        project_id=uuid4(),
        session_id=uuid4(),
        call=ToolCall("call-4b", "compare_papers", {}),
        raw_payload=six,
    )
    assert big.artifact_ref is not None


def test_accumulated_budget_forces_offload(materializer: ToolResultMaterializer) -> None:
    result = materializer.materialize(
        project_id=uuid4(),
        session_id=uuid4(),
        call=ToolCall("call-5", "search_knowledge", {}),
        raw_payload=_search_payload(1),
        accumulated_tokens=1500,
    )
    assert result.artifact_ref is not None


def test_binary_result_is_encoded_and_offloaded(
    materializer: ToolResultMaterializer,
) -> None:
    result = materializer.materialize(
        project_id=uuid4(),
        session_id=uuid4(),
        call=ToolCall("binary-1", "export_binary", {}),
        raw_payload=b"\x00\x01binary",
    )
    assert result.artifact_ref is not None
    assert result.artifact_ref.media_type == "application/octet-stream"
    assert "data_base64" not in result.model_payload
