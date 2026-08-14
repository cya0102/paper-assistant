"""Project aggregate."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Project:
    project_id: UUID
    name: str
    root_path: Path
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Project name cannot be empty")
        if not self.root_path.is_absolute():
            raise ValueError("Project root_path must be absolute")

