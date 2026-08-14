"""Minimal typed Tool Registry with JSON-Schema contracts."""

from dataclasses import dataclass
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolContract:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    strict: bool = True

    def model_spec(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolContract] = {}

    def register(self, contract: ToolContract) -> None:
        if contract.name in self._tools:
            raise ValueError(f"Tool already registered: {contract.name}")
        self._tools[contract.name] = contract

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            tool = self._tools[name]
        except KeyError as error:
            raise ValueError(f"Unknown tool: {name}") from error
        return tool.handler(arguments)

    def model_specs(self) -> tuple[dict[str, object], ...]:
        return tuple(tool.model_spec() for tool in self._tools.values())

