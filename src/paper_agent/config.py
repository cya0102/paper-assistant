"""Runtime configuration without framework coupling."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    project_root: Path

    @classmethod
    def from_environment(cls, *, project_root: Path | None = None) -> "Settings":
        database_url = os.environ.get("PAPER_AGENT_DATABASE_URL")
        if not database_url:
            raise ValueError("PAPER_AGENT_DATABASE_URL is required")
        root = (project_root or Path.cwd()).resolve()
        return cls(database_url=database_url, project_root=root)

