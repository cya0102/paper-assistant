from uuid import uuid4

import pytest

from paper_agent.domain.artifact import (
    ArtifactDescriptor,
    ArtifactReference,
    ArtifactSelector,
    ArtifactStatus,
    ArtifactType,
    CitationReference,
)


def _citation(label: str = "E1") -> CitationReference:
    return CitationReference(
        citation_label=label,
        paper_id=uuid4(),
        version_id=uuid4(),
        paper_title="Paper",
        section_path="Method",
        page_start=1,
        page_end=2,
    )


def test_citation_requires_valid_hash_and_pages():
    with pytest.raises(ValueError, match="SHA-256"):
        CitationReference(
            citation_label="E1",
            paper_id=uuid4(),
            version_id=uuid4(),
            paper_title="P",
            section_path="M",
            evidence_hash="not-a-hash",
        )
    with pytest.raises(ValueError, match="page range"):
        CitationReference(
            citation_label="E1",
            paper_id=uuid4(),
            version_id=uuid4(),
            paper_title="P",
            section_path="M",
            page_start=5,
            page_end=2,
        )


def test_descriptor_validation():
    base = dict(
        artifact_id=uuid4(),
        project_id=uuid4(),
        artifact_type=ArtifactType.TOOL_RESULT,
        schema_version="v1",
        media_type="application/json",
        content_hash="a" * 64,
        storage_backend="local_content_addressed",
        storage_key="sha256/aa/" + "a" * 64 + ".json.gz",
        byte_size=10,
        token_estimate=5,
        summary="s",
    )
    ArtifactDescriptor(**base)
    with pytest.raises(ValueError, match="SHA-256"):
        ArtifactDescriptor(**{**base, "content_hash": "x" * 64})
    with pytest.raises(ValueError, match="byte_size"):
        ArtifactDescriptor(**{**base, "byte_size": -1})
    with pytest.raises(ValueError, match="labels must be unique"):
        ArtifactDescriptor(**{**base, "citation_manifest": (_citation("E1"), _citation("E1"))})


def test_selector_caps_max_tokens():
    with pytest.raises(ValueError, match="max_tokens"):
        ArtifactSelector(artifact_id=uuid4(), project_id=uuid4(), max_tokens=5000)


def test_reference_round_trip():
    from paper_agent.domain.artifact import (
        artifact_ref_from_dict,
        artifact_ref_to_dict,
    )

    ref = ArtifactReference(
        artifact_id=uuid4(),
        project_id=uuid4(),
        artifact_type=ArtifactType.PAPER_COMPARISON,
        media_type="application/json",
        byte_size=100,
        token_estimate=50,
        summary="s",
        created_by="agent",
        created_at=_now(),
        available_views=("all-cells", "evidence"),
    )
    restored = artifact_ref_from_dict(artifact_ref_to_dict(ref))
    assert restored.artifact_id == ref.artifact_id
    assert restored.available_views == ref.available_views
    assert restored.artifact_type == ArtifactType.PAPER_COMPARISON


def _now():
    from datetime import UTC, datetime

    return datetime.now(UTC)


def test_descriptor_json_round_trip():
    from paper_agent.domain.artifact import descriptor_from_dict, descriptor_to_dict

    descriptor = ArtifactDescriptor(
        artifact_id=uuid4(),
        project_id=uuid4(),
        artifact_type=ArtifactType.PAPER_READ,
        schema_version="read-paper-v1",
        media_type="application/json",
        content_hash="b" * 64,
        storage_backend="local_content_addressed",
        storage_key="sha256/bb/" + "b" * 64 + ".json.gz",
        byte_size=20,
        token_estimate=8,
        summary="s",
        citation_manifest=(_citation("P1"),),
        status=ArtifactStatus.ACTIVE,
        created_by="agent",
        created_at=_now(),
        session_id=uuid4(),
        tool_call_id="call-1",
    )
    restored = descriptor_from_dict(descriptor_to_dict(descriptor))
    assert restored.artifact_id == descriptor.artifact_id
    assert restored.citation_manifest[0].citation_label == "P1"
    assert restored.session_id == descriptor.session_id
