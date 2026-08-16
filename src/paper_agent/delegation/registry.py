"""WorkerRegistry: the single source of truth for worker capabilities."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkerDescriptor:
    name: str
    description: str
    capabilities: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    output_schema: dict[str, Any]
    default_token_budget: int
    default_tool_call_budget: int
    timeout_seconds: int
    implemented: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("worker name cannot be blank")
        if self.default_token_budget < 1:
            raise ValueError("default_token_budget must be positive")
        if self.default_tool_call_budget < 1:
            raise ValueError("default_tool_call_budget must be positive")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, WorkerDescriptor] = {}

    def register(self, descriptor: WorkerDescriptor) -> None:
        if descriptor.name in self._workers:
            raise ValueError(f"Worker already registered: {descriptor.name}")
        self._workers[descriptor.name] = descriptor

    def get(self, name: str) -> WorkerDescriptor | None:
        return self._workers.get(name)

    def require(self, name: str) -> WorkerDescriptor:
        descriptor = self._workers.get(name)
        if descriptor is None:
            raise LookupError(f"Unknown worker: {name}")
        return descriptor

    def names(self) -> tuple[str, ...]:
        return tuple(self._workers)

    def descriptors(self) -> tuple[WorkerDescriptor, ...]:
        return tuple(self._workers.values())
