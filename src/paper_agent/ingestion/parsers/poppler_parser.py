"""Poppler CLI fallback parser available on many local workstations."""

import re
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
from pathlib import Path

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


class PopplerPdfParser:
    name = "poppler"
    version = "1"

    def __init__(self, *, timeout_seconds: int = 120) -> None:
        self._pdftotext = shutil.which("pdftotext")
        self._pdfinfo = shutil.which("pdfinfo")
        if not self._pdftotext or not self._pdfinfo:
            raise PaperAgentError(
                ErrorCode.PARSER_UNAVAILABLE,
                "Poppler requires both pdftotext and pdfinfo on PATH",
            )
        self._timeout_seconds = timeout_seconds

    @staticmethod
    def is_available() -> bool:
        return shutil.which("pdftotext") is not None and shutil.which("pdfinfo") is not None

    def extract(self, path: Path) -> PaperMetadata:
        info = self._pdf_info(path)
        text = self._run([self._pdftotext, "-layout", str(path), "-"])
        first_page = self._run(
            [self._pdftotext, "-f", "1", "-l", "1", "-layout", str(path), "-"]
        )
        return build_paper_metadata(
            file_path=path,
            raw_metadata=info,
            first_page_text=first_page,
            document_text=text,
            page_count=int(info.get("Pages", "0")),
        )

    def parse(self, request: ParseRequest) -> CanonicalParsedDocument:
        xml = self._run([self._pdftotext, "-bbox-layout", str(request.source_path), "-"])
        try:
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError as error:
            raise PaperAgentError(
                ErrorCode.PARSE_FAILED,
                f"Poppler returned invalid layout XML for {request.source_path}: {error}",
            ) from error
        pages: list[DocumentPage] = []
        for page_index, page_element in enumerate(root.findall(".//{*}page")):
            blocks: list[DocumentBlock] = []
            for block_element in page_element.findall(".//{*}block"):
                words = block_element.findall(".//{*}word")
                text = self._block_text(block_element)
                if not text or not words:
                    continue
                x0 = min(float(word.attrib["xMin"]) for word in words)
                y0 = min(float(word.attrib["yMin"]) for word in words)
                x1 = max(float(word.attrib["xMax"]) for word in words)
                y1 = max(float(word.attrib["yMax"]) for word in words)
                block_type, level = self._classify_block(text, y1 - y0)
                attributes: dict[str, object] = {}
                if level is not None:
                    attributes["level"] = level
                blocks.append(
                    DocumentBlock(
                        block_id=f"p{page_index + 1}-b{len(blocks) + 1}",
                        block_type=block_type,
                        text=text,
                        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                        reading_order=len(blocks),
                        attributes=attributes,
                    )
                )
            pages.append(
                DocumentPage(
                    page_number=page_index + 1,
                    width=float(page_element.attrib["width"]),
                    height=float(page_element.attrib["height"]),
                    blocks=tuple(blocks),
                )
            )
        if not pages:
            raise PaperAgentError(ErrorCode.EMPTY_PDF, f"PDF contains no readable pages: {request.source_path}")
        return CanonicalParsedDocument(
            paper_id=request.identity.paper.paper_id,
            version_id=request.identity.version.version_id,
            source_file_id=request.source_file.file_id,
            parser=ParserDescriptor(name=self.name, version=self.version),
            pages=tuple(pages),
        )

    def _pdf_info(self, path: Path) -> dict[str, str]:
        output = self._run([self._pdfinfo, str(path)])
        info: dict[str, str] = {}
        for line in output.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                normalized_key = {
                    "Title": "title",
                    "Author": "author",
                    "Subject": "subject",
                    "Keywords": "keywords",
                    "CreationDate": "creationDate",
                }.get(key.strip(), key.strip())
                info[normalized_key] = value.strip()
        return info

    def _run(self, command: list[str | None]) -> str:
        executable_command = [part for part in command if part is not None]
        try:
            result = subprocess.run(
                executable_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PaperAgentError(ErrorCode.PARSE_FAILED, str(error)) from error
        if result.returncode != 0:
            message = result.stderr.strip() or f"Poppler exited with status {result.returncode}"
            code = ErrorCode.ENCRYPTED_PDF if re.search(r"password|encrypted", message, re.I) else ErrorCode.PARSE_FAILED
            raise PaperAgentError(code, message)
        return result.stdout

    @staticmethod
    def _block_text(block: ElementTree.Element) -> str:
        lines: list[str] = []
        for line_element in block.findall(".//{*}line"):
            words = [word.text or "" for word in line_element.findall(".//{*}word")]
            line = " ".join(word for word in words if word).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _classify_block(text: str, block_height: float) -> tuple[BlockType, int | None]:
        if text.count("\n") <= 2 and len(text.split()) <= 25 and block_height >= 16:
            return BlockType.HEADING, 1 if block_height >= 24 else 2
        return BlockType.PARAGRAPH, None
