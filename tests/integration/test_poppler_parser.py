from pathlib import Path
from uuid import uuid4

import pytest

from paper_agent.domain.paper import Paper, PaperFile, PaperIdentityResolution, PaperVersion
from paper_agent.ingestion.parsers.poppler_parser import PopplerPdfParser
from paper_agent.ingestion.parsers.pymupdf_parser import PyMuPdfParser
from paper_agent.ingestion.ports import ParseRequest


def write_minimal_pdf(path: Path) -> None:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length 58 >>\nstream\nBT /F1 18 Tf 72 720 Td (Phase 1B Test Paper) Tj ET\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode())
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(payload)


@pytest.mark.skipif(not PopplerPdfParser.is_available(), reason="Poppler is not installed")
def test_poppler_extracts_real_pdf_to_canonical_document(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    write_minimal_pdf(path)
    parser = PopplerPdfParser()
    paper = Paper(paper_id=uuid4(), canonical_title="Phase 1B Test Paper")
    version = PaperVersion(version_id=uuid4(), paper_id=paper.paper_id)
    paper_file = PaperFile(uuid4(), uuid4(), path.stat().st_size, "a" * 64, paper.paper_id, version.version_id)
    identity = PaperIdentityResolution(paper=paper, version=version)

    metadata = parser.extract(path)
    document = parser.parse(ParseRequest(path, paper_file, identity))

    assert metadata.page_count == 1
    assert metadata.content_hash is not None
    assert len(document.pages) == 1
    assert "Phase 1B Test Paper" in "\n".join(
        block.text or "" for block in document.pages[0].blocks
    )


@pytest.mark.skipif(not PyMuPdfParser.is_available(), reason="PyMuPDF is not installed")
def test_pymupdf_extracts_real_pdf_to_canonical_document(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    write_minimal_pdf(path)
    parser = PyMuPdfParser()
    paper = Paper(paper_id=uuid4(), canonical_title="Phase 1B Test Paper")
    version = PaperVersion(version_id=uuid4(), paper_id=paper.paper_id)
    paper_file = PaperFile(
        uuid4(), uuid4(), path.stat().st_size, "a" * 64, paper.paper_id, version.version_id
    )
    identity = PaperIdentityResolution(paper=paper, version=version)

    metadata = parser.extract(path)
    document = parser.parse(ParseRequest(path, paper_file, identity))

    assert metadata.page_count == 1
    assert metadata.content_hash is not None
    assert len(document.pages) == 1
    assert "Phase 1B Test Paper" in "\n".join(
        block.text or "" for block in document.pages[0].blocks
    )
