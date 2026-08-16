"""OffloadPolicy: where a tool result stays inline vs. becomes an Artifact."""

from dataclasses import dataclass
from typing import Any

from paper_agent.artifacts.tokens import count_tokens


@dataclass(frozen=True, slots=True)
class OffloadPolicyConfig:
    max_inline_tokens_per_result: int = 2000
    max_total_tool_tokens: int = 6000
    preview_tokens: int = 800
    artifact_retention_days: int = 30
    read_artifact_max_tokens: int = 4000

    def __post_init__(self) -> None:
        if self.max_inline_tokens_per_result < 1:
            raise ValueError("max_inline_tokens_per_result must be positive")
        if self.max_total_tool_tokens < self.max_inline_tokens_per_result:
            raise ValueError("max_total_tool_tokens must cover the per-result budget")
        if self.preview_tokens < 1:
            raise ValueError("preview_tokens must be positive")
        if self.artifact_retention_days < 1:
            raise ValueError("artifact_retention_days must be positive")
        if not 1 <= self.read_artifact_max_tokens <= 4000:
            raise ValueError("read_artifact_max_tokens must be between 1 and 4000")


class OffloadPolicy:
    """Decision rules for offloading complete tool payloads to the Artifact store.

    Thresholds come exclusively from the configuration object so no rule is
    hard-coded across the code base.  Tool-name rules encode the class of result:
    binary payloads, large comparisons, full-section reads and worker results
    always offload regardless of their token count.
    """

    def __init__(self, config: OffloadPolicyConfig | None = None) -> None:
        self._config = config or OffloadPolicyConfig()

    @property
    def config(self) -> OffloadPolicyConfig:
        return self._config

    def should_offload(
        self,
        *,
        tool_name: str,
        payload: dict[str, Any],
        token_count: int,
        accumulated_tokens: int,
        force: bool = False,
    ) -> bool:
        if force:
            return True
        if payload.get("media_type") not in (None, "application/json"):
            return True  # binary or non-JSON payloads always offload
        if tool_name == "compare_papers":
            paper_ids = payload.get("paper_ids", [])
            if len(paper_ids) > 5:
                return True
        if tool_name == "read_paper":
            passages = payload.get("passages", [])
            elements = payload.get("elements", [])
            if elements or len(passages) > 4:
                return True
        if tool_name == "worker_result":
            return True
        if token_count > self._config.max_inline_tokens_per_result:
            return True
        if accumulated_tokens + token_count > self._config.max_total_tool_tokens:
            return True
        return False

    def preview_budget(self) -> int:
        return self._config.preview_tokens
