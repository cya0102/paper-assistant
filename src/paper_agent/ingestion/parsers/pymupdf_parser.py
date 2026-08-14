"""PyMuPDF adapter for metadata probing and canonical layout extraction."""

from importlib import import_module
from pathlib import Path
from typing import Any

from paper_agent.domain.document import (
    BoundingBox,
    CanonicalParsedDocument,
    DocumentBlock,
    DocumentPage,
    ParserDescriptor,
)
from paper_agent.domain.enums import BlockType
from paper_agent.domain.errors import ErrorCode, PaperAgentError
from paper_agent.domain.metadata import PaperMetadata
from paper_agent.ingestion.metadata import build_paper_metadata
from paper_agent.ingestion.ports import ParseRequest


class PyMuPdfParser:
    name = "pymupdf"
    version = "1"

    def __init__(self) -> None:
        self._pymupdf = self._load_module()

    @staticmethod
    def is_available() -> bool:
        try:
            PyMuPdfParser._load_module()
        except PaperAgentError:
            return False
        return True

    @staticmethod
    def _load_module() -> Any:
        for module_name in ("pymupdf", "fitz"):
            try:
                return import_module(module_name)
            except ImportError:
                continue
        raise PaperAgentError(
            ErrorCode.PARSER_UNAVAILABLE,
            "PyMuPDF is not installed; install the project dependencies or use PopplerPdfParser",
        )

    def extract(self, path: Path) -> PaperMetadata:
        document = self._open(path)
        try:
            text_pages = [page.get_text("text") for page in document]
            metadata = {str(key): str(value or "") for key, value in document.metadata.items()}
            return build_paper_metadata(
                file_path=path,
                raw_metadata=metadata,
                first_page_text=text_pages[0] if text_pages else "",
                document_text="\f".join(text_pages),
                page_count=document.page_count,
            )
        finally:
            document.close()

    def parse(self, request: ParseRequest) -> CanonicalParsedDocument:
        document = self._open(request.source_path)
        try:
            pages: list[DocumentPage] = []
            for page_index, page in enumerate(document):
                page_dict = page.get_text("dict", sort=True)
                blocks: list[DocumentBlock] = []
                for raw_block in page_dict.get("blocks", []):
                    if raw_block.get("type") != 0:
                        continue
                    text, max_font_size = self._block_text(raw_block)
                    if not text.strip():
                        continue
                    bbox = raw_block.get("bbox")
                    block_type, level = self._classify_block(text, max_font_size)
                    attributes: dict[str, object] = {"max_font_size": max_font_size}
                    if level is not None:
                        attributes["level"] = level
                    blocks.append(
                        DocumentBlock(
                            block_id=f"p{page_index + 1}-b{len(blocks) + 1}",
                            block_type=block_type,
                            text=text,
                            bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])
                            if bbox
                            else None,
                            reading_order=len(blocks),
                            attributes=attributes,
                        )
                    )
                pages.append(
                    DocumentPage(
                        page_number=page_index + 1,
                        width=float(page.rect.width),
                        height=float(page.rect.height),
                        blocks=tuple(blocks),
                    )
                )
            return CanonicalParsedDocument(
                paper_id=request.identity.paper.paper_id,
                version_id=request.identity.version.version_id,
                source_file_id=request.source_file.file_id,
                parser=ParserDescriptor(name=self.name, version=self.version),
                pages=tuple(pages),
            )
        finally:
            document.close()

    def _open(self, path: Path) -> Any:
        try:
            document = self._pymupdf.open(str(path))
        except Exception as error:
            raise PaperAgentError(ErrorCode.PARSE_FAILED, f"Cannot open PDF {path}: {error}") from error
        if getattr(document, "needs_pass", False):
            document.close()
            raise PaperAgentError(ErrorCode.ENCRYPTED_PDF, f"PDF requires a password: {path}")
        if document.page_count == 0:
            document.close()
            raise PaperAgentError(ErrorCode.EMPTY_PDF, f"PDF contains no pages: {path}")
        return document

    @staticmethod
    def _block_text(raw_block: dict[str, Any]) -> tuple[str, float]:
        lines: list[str] = []
        max_font_size = 0.0
        for raw_line in raw_block.get("lines", []):
            spans = raw_line.get("spans", [])
            line = "".join(str(span.get("text", "")) for span in spans).strip()
            if line:
                lines.append(line)
            for span in spans:
                max_font_size = max(max_font_size, float(span.get("size", 0)))
        return "\n".join(lines), max_font_size

    @staticmethod
    def _classify_block(text: str, max_font_size: float) -> tuple[BlockType, int | None]:
        line_count = text.count("\n") + 1
        word_count = len(text.split())
        if line_count <= 3 and word_count <= 25 and max_font_size >= 13:
            if max_font_size >= 20:
                return BlockType.HEADING, 1
            if max_font_size >= 16:
                return BlockType.HEADING, 2
            return BlockType.HEADING, 3
        return BlockType.PARAGRAPH, None

