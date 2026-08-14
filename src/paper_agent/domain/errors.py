"""Typed failures that can cross application boundaries safely."""

from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_PATH = "invalid_path"
    PATH_OUTSIDE_PROJECT = "path_outside_project"
    FILE_NOT_FOUND = "file_not_found"
    FILE_CHANGED_DURING_READ = "file_changed_during_read"
    FINGERPRINT_FAILED = "fingerprint_failed"
    IDENTITY_FAILED = "identity_failed"
    PARSE_FAILED = "parse_failed"
    STORAGE_FAILED = "storage_failed"
    DATABASE_FAILED = "database_failed"
    INVALID_DOCUMENT = "invalid_document"
    ENCRYPTED_PDF = "encrypted_pdf"
    EMPTY_PDF = "empty_pdf"
    PARSER_UNAVAILABLE = "parser_unavailable"
    METADATA_FAILED = "metadata_failed"
    STRUCTURE_FAILED = "structure_failed"
    CHUNK_FAILED = "chunk_failed"
    EMBEDDING_FAILED = "embedding_failed"
    INDEX_FAILED = "index_failed"
    SEARCH_FAILED = "search_failed"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class PaperAgentError(Exception):
    """Expected application failure with a stable machine-readable code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class FileChangedDuringReadError(PaperAgentError):
    def __init__(self, path: str) -> None:
        super().__init__(
            ErrorCode.FILE_CHANGED_DURING_READ,
            f"File changed while computing its fingerprint: {path}",
        )
