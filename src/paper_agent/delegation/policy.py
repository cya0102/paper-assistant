"""DelegationPolicy: when to delegate, how many workers, and when to refuse."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class DelegationDecision:
    delegate: bool
    reason: str
    max_workers: int
    workstreams: tuple[str, ...] = ()


class DelegationPolicy:
    """Routing rules for the three-tier execution strategy.

    - single paper Q&A: never delegate;
    - 2-5 paper comparison: single Agent + Offload;
    - 6+ paper comparison or explicit research workstreams: bounded delegation;
    - PDF parsing, indexing, database operations: never delegate.
    """

    def __init__(self, *, max_workers_cap: int = 5, batch_threshold: int = 6) -> None:
        if not 1 <= max_workers_cap <= 5:
            raise ValueError("max_workers_cap must be between 1 and 5")
        if batch_threshold < 2:
            raise ValueError("batch_threshold must be at least 2")
        self._max_workers_cap = max_workers_cap
        self._batch_threshold = batch_threshold

    @property
    def max_workers_cap(self) -> int:
        return self._max_workers_cap

    def decide(
        self,
        *,
        paper_ids: tuple[UUID, ...],
        requested_workstreams: tuple[str, ...] = (),
        max_workers: int | None = None,
    ) -> DelegationDecision:
        requested = max_workers if max_workers is not None else 3
        requested = max(1, min(requested, self._max_workers_cap))
        if not paper_ids:
            return DelegationDecision(
                False, "delegation requires at least one paper", 0
            )
        if len(paper_ids) == 1 and not requested_workstreams:
            return DelegationDecision(
                False,
                "single-paper questions are handled by the main Agent without delegation",
                0,
            )
        if (
            2 <= len(paper_ids) <= self._batch_threshold - 1
            and not requested_workstreams
        ):
            return DelegationDecision(
                False,
                "small comparisons use the main Agent with Offload",
                0,
            )
        if requested_workstreams:
            return DelegationDecision(
                True,
                f"explicit research workstreams over {len(paper_ids)} papers",
                requested,
                requested_workstreams,
            )
        return DelegationDecision(
            True,
            f"large batch of {len(paper_ids)} papers warrants bounded delegation",
            requested,
        )
