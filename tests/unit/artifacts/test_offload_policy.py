import pytest

from paper_agent.artifacts.policies import OffloadPolicy, OffloadPolicyConfig


def _policy(**overrides) -> OffloadPolicy:
    return OffloadPolicy(OffloadPolicyConfig(**overrides))


def test_small_result_stays_inline() -> None:
    policy = _policy(max_inline_tokens_per_result=2000)
    assert not policy.should_offload(
        tool_name="search_knowledge",
        payload={},
        token_count=100,
        accumulated_tokens=0,
    )


def test_large_result_offloads() -> None:
    policy = _policy(max_inline_tokens_per_result=2000)
    assert policy.should_offload(
        tool_name="search_knowledge",
        payload={},
        token_count=5000,
        accumulated_tokens=0,
    )


def test_total_budget_forces_offload() -> None:
    policy = _policy(max_inline_tokens_per_result=2000, max_total_tool_tokens=6000)
    assert policy.should_offload(
        tool_name="search_knowledge",
        payload={},
        token_count=100,
        accumulated_tokens=5950,
    )


def test_large_comparison_always_offloads() -> None:
    policy = _policy()
    assert policy.should_offload(
        tool_name="compare_papers",
        payload={"paper_ids": list(range(6))},
        token_count=10,
        accumulated_tokens=0,
    )
    assert not policy.should_offload(
        tool_name="compare_papers",
        payload={"paper_ids": list(range(5))},
        token_count=10,
        accumulated_tokens=0,
    )


def test_full_read_offloads() -> None:
    policy = _policy()
    assert policy.should_offload(
        tool_name="read_paper",
        payload={"passages": list(range(5))},
        token_count=10,
        accumulated_tokens=0,
    )
    assert policy.should_offload(
        tool_name="read_paper",
        payload={"elements": [{}]},
        token_count=10,
        accumulated_tokens=0,
    )


def test_worker_results_always_offload() -> None:
    policy = _policy()
    assert policy.should_offload(
        tool_name="worker_result",
        payload={},
        token_count=1,
        accumulated_tokens=0,
    )


def test_binary_payload_offloads() -> None:
    policy = _policy()
    assert policy.should_offload(
        tool_name="any",
        payload={"media_type": "application/pdf"},
        token_count=1,
        accumulated_tokens=0,
    )


def test_force_flag() -> None:
    policy = _policy()
    assert policy.should_offload(
        tool_name="x", payload={}, token_count=1, accumulated_tokens=0, force=True
    )


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        OffloadPolicyConfig(max_inline_tokens_per_result=0)
    with pytest.raises(ValueError):
        OffloadPolicyConfig(max_total_tool_tokens=500, max_inline_tokens_per_result=2000)
