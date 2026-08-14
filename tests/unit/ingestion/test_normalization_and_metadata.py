from hashlib import sha256

from paper_agent.ingestion.metadata import build_paper_metadata
from paper_agent.ingestion.normalization import (
    content_fingerprint,
    normalize_arxiv_id,
    normalize_doi,
    normalize_document_text,
    normalize_title,
)


def test_identifier_and_title_normalization() -> None:
    assert normalize_title("  A Study—of\nPaper Agents! ") == "a study of paper agents"
    assert normalize_doi("https://doi.org/10.1234/ABC.5") == "10.1234/abc.5"
    assert normalize_arxiv_id("arXiv:2401.01234v2") == "2401.01234"


def test_content_fingerprint_removes_repeated_headers_and_page_numbers() -> None:
    text = "Conference 2026\nFirst page content\n1\fConference 2026\nSecond page content\n2"
    normalized = normalize_document_text(text)

    assert "conference 2026" not in normalized
    assert normalized == "first page content\nsecond page content"
    assert content_fingerprint(text) == sha256(normalized.encode()).hexdigest()
    assert content_fingerprint("   \n\f  ") is None


def test_build_metadata_prefers_pdf_metadata_and_extracts_identifiers(tmp_path) -> None:
    metadata = build_paper_metadata(
        file_path=tmp_path / "fallback.pdf",
        raw_metadata={"title": "Paper Agents", "author": "Alice Smith; Bob Li"},
        first_page_text="Paper Agents\nAlice Smith and Bob Li\narXiv: 2401.01234v2",
        document_text="DOI: 10.1234/PAPER.1\nBody text",
        page_count=4,
    )

    assert metadata.title == "Paper Agents"
    assert metadata.authors == ("Alice Smith", "Bob Li")
    assert metadata.doi == "10.1234/paper.1"
    assert metadata.arxiv_id == "2401.01234"
    assert metadata.page_count == 4
    assert metadata.content_hash is not None

