"""Atomic local persistence for canonical parsed documents."""

import json
import os
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from uuid import UUID

from paper_agent.domain.document import CanonicalParsedDocument, DocumentBlock
from paper_agent.domain.enums import BlockType
from paper_agent.ingestion.ports import ParsedDocumentArtifacts


class LocalParsedDocumentStore:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        self._parsed_root = self._project_root / ".paper-agent" / "parsed"

    def save(self, document: CanonicalParsedDocument) -> ParsedDocumentArtifacts:
        version_directory = self._version_directory(document.paper_id, document.version_id)
        assets_directory = version_directory / "assets"
        assets_directory.mkdir(parents=True, exist_ok=True)

        json_payload = (
            json.dumps(
                document.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        markdown_payload = self._render_markdown(document).encode("utf-8")

        document_json = version_directory / "document.json"
        document_markdown = version_directory / "document.md"
        self._atomic_write(document_json, json_payload)
        self._atomic_write(document_markdown, markdown_payload)

        return ParsedDocumentArtifacts(
            document_json_path=self._relative(document_json),
            document_markdown_path=self._relative(document_markdown),
            assets_path=self._relative(assets_directory),
            document_hash=sha256(json_payload).hexdigest(),
        )

    def load(self, paper_id: UUID, version_id: UUID) -> CanonicalParsedDocument:
        document_path = self._version_directory(paper_id, version_id) / "document.json"
        return CanonicalParsedDocument.model_validate_json(document_path.read_text(encoding="utf-8"))

    def exists(self, paper_id: UUID, version_id: UUID) -> bool:
        return (self._version_directory(paper_id, version_id) / "document.json").is_file()

    def _version_directory(self, paper_id: UUID, version_id: UUID) -> Path:
        return self._parsed_root / str(paper_id) / str(version_id)

    def _relative(self, path: Path) -> PurePosixPath:
        return PurePosixPath(path.relative_to(self._project_root).as_posix())

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def _render_markdown(cls, document: CanonicalParsedDocument) -> str:
        lines = [
            "---",
            f"paper_id: {document.paper_id}",
            f"version_id: {document.version_id}",
            f"parser: {document.parser.name}@{document.parser.version}",
            f"schema_version: {document.schema_version}",
            "---",
            "",
        ]
        for page in document.pages:
            lines.extend((f"<!-- page: {page.page_number} -->", ""))
            for block in sorted(page.blocks, key=lambda item: item.reading_order):
                rendered = cls._render_block(block)
                if rendered:
                    lines.extend((rendered, ""))
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _render_block(block: DocumentBlock) -> str:
        text = (block.text or "").strip()
        if block.block_type == BlockType.HEADING:
            level = block.attributes.get("level", 1)
            if not isinstance(level, int):
                level = 1
            level = min(max(level, 1), 6)
            return f"{'#' * level} {text}" if text else ""
        if block.block_type == BlockType.EQUATION:
            return text or f"[Equation: {block.block_id}]"
        if block.block_type == BlockType.FIGURE:
            return text or f"[Figure: {block.block_id}]"
        if block.block_type == BlockType.TABLE:
            return text or f"[Table: {block.block_id}]"
        if block.block_type == BlockType.ALGORITHM:
            return text or f"[Algorithm: {block.block_id}]"
        return text

