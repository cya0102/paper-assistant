"""Deterministic identifier, metadata, and document-text normalization."""

import re
import unicodedata
from collections import Counter
from hashlib import sha256

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)
ARXIV_PATTERN = re.compile(
    r"(?:arxiv\s*:\s*)?((?:[a-z-]+/\d{7})|(?:\d{4}\.\d{4,5})(?:v\d+)?)",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
PAGE_NUMBER_PATTERN = re.compile(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?$", re.IGNORECASE)
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_unicode(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def normalize_title(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_unicode(value).casefold().replace("‐", "-").replace("–", "-")
    normalized = PUNCTUATION_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized or None


def normalize_author(value: str) -> str:
    normalized = normalize_unicode(value).casefold()
    normalized = PUNCTUATION_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_authors(authors: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(author for value in authors if (author := normalize_author(value)))


def normalize_doi(value: str | None) -> str | None:
    if value is None:
        return None
    match = DOI_PATTERN.search(normalize_unicode(value))
    if not match:
        return None
    return match.group(0).rstrip(".,;)").casefold()


def normalize_arxiv_id(value: str | None) -> str | None:
    if value is None:
        return None
    match = ARXIV_PATTERN.search(normalize_unicode(value))
    if not match:
        return None
    return re.sub(r"v\d+$", "", match.group(1).casefold())


def extract_doi(text: str) -> str | None:
    return normalize_doi(text)


def extract_arxiv_id(text: str) -> str | None:
    return normalize_arxiv_id(text)


def extract_year(text: str) -> int | None:
    match = YEAR_PATTERN.search(text)
    return int(match.group(0)) if match else None


def normalize_document_text(text: str) -> str:
    pages = text.split("\f")
    normalized_pages = [_normalized_page_lines(page) for page in pages]
    repeated_edges = _repeated_page_edges(normalized_pages)
    kept: list[str] = []
    for lines in normalized_pages:
        for line in lines:
            if line in repeated_edges or PAGE_NUMBER_PATTERN.fullmatch(line):
                continue
            kept.append(line)
    return "\n".join(kept).strip()


def content_fingerprint(text: str) -> str | None:
    normalized = normalize_document_text(text)
    return sha256(normalized.encode("utf-8")).hexdigest() if normalized else None


def _normalized_page_lines(page: str) -> list[str]:
    result: list[str] = []
    for raw_line in normalize_unicode(page).splitlines():
        line = WHITESPACE_PATTERN.sub(" ", raw_line).strip().casefold()
        if line:
            result.append(line)
    return result


def _repeated_page_edges(pages: list[list[str]]) -> set[str]:
    if len(pages) < 2:
        return set()
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update(set(lines[:2] + lines[-2:]))
    threshold = max(2, (len(pages) + 1) // 2)
    return {line for line, count in counts.items() if count >= threshold}
