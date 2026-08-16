import json
import os
from uuid import uuid4

import pytest
from redis import Redis

from paper_agent.domain.agent import (
    AgentCheckpoint,
    AgentRunStatus,
    ConversationMessage,
    ToolCall,
    ToolResult,
)
from paper_agent.domain.artifact import (
    ArtifactReference,
    ArtifactType,
    CitationReference,
)
from paper_agent.memory import RedisCheckpointStore


REDIS_URL = os.getenv("PAPER_AGENT_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(
    not REDIS_URL, reason="PAPER_AGENT_TEST_REDIS_URL is required"
)


def _citation(label: str) -> CitationReference:
    return CitationReference(
        citation_label=label,
        paper_id=uuid4(),
        version_id=uuid4(),
        paper_title="Paper",
        section_path="Method",
        page_start=1,
        page_end=2,
    )


def test_redis_checkpoint_round_trips_compact_tool_result():
    assert REDIS_URL is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    session_id, user_id, project_id = uuid4(), uuid4(), uuid4()
    checkpoints = RedisCheckpointStore(client, ttl_seconds=60, prefix="paper-agent-test")
    ref = ArtifactReference(
        artifact_id=uuid4(),
        project_id=project_id,
        artifact_type=ArtifactType.PAPER_COMPARISON,
        media_type="application/json",
        byte_size=1000,
        token_estimate=500,
        summary="comparison",
        created_by="agent",
        created_at=_now(),
        available_views=("all-cells", "evidence"),
    )
    try:
        checkpoint = AgentCheckpoint(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            messages=[ConversationMessage("user", "query")],
            status=AgentRunStatus.WAITING_FOR_TOOLS,
            response_id="response-1",
            pending_calls=[ToolCall("call-2", "search_knowledge", {"query": "more"})],
            tool_results=[
                ToolResult(
                    call_id="call-1",
                    name="compare_papers",
                    model_payload={"status": "complete", "paper_count": 6},
                    artifact_ref=ref,
                    citation_manifest=(_citation("E1"), _citation("E2")),
                )
            ],
        )
        checkpoints.save(checkpoint)
        loaded = checkpoints.load(session_id)
        assert loaded is not None
        result = loaded.tool_results[0]
        assert result.name == "compare_papers"
        assert result.model_payload == {"status": "complete", "paper_count": 6}
        assert result.artifact_ref is not None
        assert result.artifact_ref.artifact_id == ref.artifact_id
        assert result.artifact_ref.available_views == ("all-cells", "evidence")
        assert [item.citation_label for item in result.citation_manifest] == ["E1", "E2"]
        # the raw payload is not stored anywhere in Redis
        raw = client.get(checkpoints._key(session_id))
        assert "full_raw" not in (raw or "")
        assert json.loads(raw)["tool_results"][0]["model_payload"]["status"] == "complete"
    finally:
        client.delete(checkpoints._key(session_id))


def test_redis_rod_checkpoint_never_contains_retrieved_chunk_text():
    assert REDIS_URL is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    session_id, user_id, project_id = uuid4(), uuid4(), uuid4()
    checkpoints = RedisCheckpointStore(
        client, ttl_seconds=60, prefix="paper-agent-test-rod"
    )
    secret = "FULL RETRIEVED CHUNK MUST STAY OFFLINE"
    try:
        checkpoint = AgentCheckpoint(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            messages=[ConversationMessage("user", "query")],
            status=AgentRunStatus.RUNNING,
            response_id="response-rod",
            tool_results=[
                ToolResult(
                    call_id="rod-call",
                    name="retrieve_and_analyze_knowledge",
                    model_payload={
                        "status": "supported",
                        "summary": "compact analyst report",
                        "claims": [
                            {"text": "supported fact", "citations": ["E1"]}
                        ],
                        "evidence_artifacts": [
                            {
                                "artifact_id": str(uuid4()),
                                "citation": "E1",
                                "paper_title": "Paper",
                            }
                        ],
                    },
                    citation_manifest=(_citation("E1"),),
                )
            ],
        )
        checkpoints.save(checkpoint)
        raw = client.get(checkpoints._key(session_id)) or ""

        assert secret not in raw
        assert "compact analyst report" in raw
        assert "retrieve_and_analyze_knowledge" in raw
    finally:
        client.delete(checkpoints._key(session_id))


def _now():
    from datetime import UTC, datetime

    return datetime.now(UTC)
