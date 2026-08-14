"""Stable domain state values shared across adapters."""

from enum import StrEnum


class PipelineStage(StrEnum):
    DISCOVERED = "discovered"
    IDENTITY_RESOLVED = "identity_resolved"
    PARSING = "parsing"
    PARSED = "parsed"
    STRUCTURED = "structured"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


class FileStatus(StrEnum):
    DISCOVERED = "discovered"
    IDENTITY_RESOLVED = "identity_resolved"
    PARSING = "parsing"
    PARSED = "parsed"
    STRUCTURED = "structured"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


class LocationPresence(StrEnum):
    PRESENT = "present"
    MISSING = "missing"


class IngestionDisposition(StrEnum):
    NEW = "new"
    MODIFIED = "modified"
    DUPLICATE = "duplicate"
    UNCHANGED = "unchanged"
    MISSING = "missing"
    FAILED = "failed"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    ALGORITHM = "algorithm"
    REFERENCE = "reference"
    OTHER = "other"


class SourceType(StrEnum):
    LOCAL = "local"
    ARXIV = "arxiv"
    DOI = "doi"
    CONFERENCE = "conference"
    OTHER = "other"


class MetadataSource(StrEnum):
    PDF_METADATA = "pdf_metadata"
    FIRST_PAGE = "first_page"
    DOCUMENT_TEXT = "document_text"
    FILE_NAME = "file_name"


class IdentityMatchType(StrEnum):
    NEW_PAPER = "new_paper"
    DOI = "doi"
    ARXIV = "arxiv"
    CONTENT_HASH = "content_hash"
    TITLE_AUTHORS = "title_authors"
    EXISTING_FILE = "existing_file"


class ElementType(StrEnum):
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    ALGORITHM = "algorithm"


class SemanticGroupType(StrEnum):
    TEXT = "text"
    ELEMENT_DEPENDENCY = "element_dependency"


class ChunkType(StrEnum):
    TEXT = "text"
    MIXED = "mixed"
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    ALGORITHM = "algorithm"


class IndexLevel(StrEnum):
    PAPER = "paper"
    SECTION = "section"
    CHUNK = "chunk"


class IndexingStatus(StrEnum):
    RUNNING = "running"
    INDEXED = "indexed"
    FAILED = "failed"


class SearchStatus(StrEnum):
    OK = "ok"
    NO_EVIDENCE = "no_evidence"
