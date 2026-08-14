"""Metadata heuristics shared by PDF parser adapters."""

import re
from pathlib import Path

from paper_agent.domain.enums import MetadataSource
from paper_agent.domain.metadata import MetadataEvidence, PaperMetadata
from paper_agent.ingestion.normalization import (
    content_fingerprint,
    extract_arxiv_id,
    extract_doi,
    extract_year,
    normalize_authors,
    normalize_title,
)

AUTHOR_SEPARATOR = re.compile(r"\s*(?:,|;|\band\b|·)\s*", re.IGNORECASE)


def build_paper_metadata(
    *,
    file_path: Path,
    raw_metadata: dict[str, str],
    first_page_text: str,
    document_text: str,
    page_count: int,
) -> PaperMetadata:
    evidence: list[MetadataEvidence] = []
    title = _clean(raw_metadata.get("title"))
    if title:
        evidence.append(MetadataEvidence("title", MetadataSource.PDF_METADATA, 0.9))
    else:
        title = infer_title(first_page_text) or file_path.stem
        source = MetadataSource.FIRST_PAGE if title != file_path.stem else MetadataSource.FILE_NAME
        confidence = 0.65 if source == MetadataSource.FIRST_PAGE else 0.3
        evidence.append(MetadataEvidence("title", source, confidence))

    authors = split_authors(raw_metadata.get("author", ""))
    if authors:
        evidence.append(MetadataEvidence("authors", MetadataSource.PDF_METADATA, 0.85))
    else:
        authors = infer_authors(first_page_text, title)
        if authors:
            evidence.append(MetadataEvidence("authors", MetadataSource.FIRST_PAGE, 0.45))

    searchable_text = "\n".join(
        value for value in (raw_metadata.get("subject", ""), first_page_text, document_text) if value
    )
    doi = extract_doi(searchable_text)
    arxiv_id = extract_arxiv_id(searchable_text)
    year = extract_year(raw_metadata.get("creationDate", "") or first_page_text)
    if doi:
        evidence.append(MetadataEvidence("doi", MetadataSource.DOCUMENT_TEXT, 0.95))
    if arxiv_id:
        evidence.append(MetadataEvidence("arxiv_id", MetadataSource.DOCUMENT_TEXT, 0.95))
    if year:
        evidence.append(MetadataEvidence("year", MetadataSource.PDF_METADATA, 0.7))

    keywords = tuple(
        item.strip()
        for item in re.split(r"[,;]", raw_metadata.get("keywords", ""))
        if item.strip()
    )
    return PaperMetadata(
        title=title,
        normalized_title=normalize_title(title),
        authors=authors,
        normalized_authors=normalize_authors(authors),
        doi=doi,
        arxiv_id=arxiv_id,
        year=year,
        subject=_clean(raw_metadata.get("subject")),
        keywords=keywords,
        page_count=page_count,
        content_hash=content_fingerprint(document_text),
        evidence=tuple(evidence),
    )


def infer_title(first_page_text: str) -> str | None:
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    candidates: list[str] = []
    for line in lines[:15]:
        lowered = line.casefold()
        if "arxiv:" in lowered or lowered.startswith(("abstract", "preprint", "doi:")):
            continue
        if len(line) < 8 or len(line) > 250:
            continue
        if re.fullmatch(r"[\d\W_]+", line):
            continue
        candidates.append(line)
        if len(" ".join(candidates)) >= 40 or len(candidates) == 2:
            break
    return " ".join(candidates) if candidates else None


def infer_authors(first_page_text: str, title: str | None) -> tuple[str, ...]:
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    if title:
        normalized_title = normalize_title(title)
        for index, line in enumerate(lines):
            if normalize_title(line) == normalized_title:
                return split_authors(lines[index + 1] if index + 1 < len(lines) else "")
    return ()


def split_authors(value: str) -> tuple[str, ...]:
    cleaned = _clean(value)
    if not cleaned:
        return ()
    return tuple(item for item in AUTHOR_SEPARATOR.split(cleaned) if 1 < len(item.split()) < 8)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None

