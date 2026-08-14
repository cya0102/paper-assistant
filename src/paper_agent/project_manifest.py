"""Small local project identity manifest; PostgreSQL remains the source of truth."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class ProjectManifest:
    project_id: UUID
    schema_version: int = 1


class ProjectManifestStore:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        self._path = self._project_root / ".paper-agent" / "project.json"

    def load(self) -> ProjectManifest:
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return ProjectManifest(
            project_id=UUID(payload["project_id"]),
            schema_version=int(payload["schema_version"]),
        )

    def load_or_create(self) -> ProjectManifest:
        if self._path.exists():
            return self.load()
        manifest = ProjectManifest(project_id=uuid4())
        self.save(manifest)
        return manifest

    def save(self, manifest: ProjectManifest) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                {
                    "project_id": str(manifest.project_id),
                    "schema_version": manifest.schema_version,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=self._path.parent, prefix=".project.json.", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

