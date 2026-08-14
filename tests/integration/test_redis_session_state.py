import os
from uuid import uuid4

import pytest
from redis import Redis

from paper_agent.domain.agent import AgentCheckpoint, AgentRunStatus, ConversationMessage, ToolCall
from paper_agent.domain.memory import SessionState
from paper_agent.memory import RedisCheckpointStore, RedisSessionStore


REDIS_URL = os.getenv("PAPER_AGENT_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="PAPER_AGENT_TEST_REDIS_URL is required")


def test_redis_checkpoint_and_session_state_round_trip_with_ttl():
    assert REDIS_URL is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    session_id, user_id, project_id = uuid4(), uuid4(), uuid4()
    checkpoints = RedisCheckpointStore(client, ttl_seconds=60, prefix="paper-agent-test")
    sessions = RedisSessionStore(client, ttl_seconds=60, prefix="paper-agent-test")
    try:
        checkpoint = AgentCheckpoint(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            messages=[ConversationMessage("user", "query")],
            status=AgentRunStatus.WAITING_FOR_TOOLS,
            response_id="response-1",
            pending_calls=[ToolCall("call-1", "search_knowledge", {"query": "codebook"})],
            model_history=[
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "search_knowledge",
                    "arguments": '{"query":"codebook"}',
                }
            ],
        )
        checkpoints.save(checkpoint)
        assert checkpoints.load(session_id) == checkpoint
        state = SessionState(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            current_paper_id=uuid4(),
            current_topic="codebook",
            active_chunk_ids=[uuid4()],
        )
        sessions.save(state)
        assert sessions.load(session_id) == state
        assert client.ttl(f"paper-agent-test:agent:{session_id}") > 0
        assert client.ttl(f"paper-agent-test:session:{session_id}") > 0
    finally:
        checkpoints.delete(session_id)
        client.delete(f"paper-agent-test:session:{session_id}")
