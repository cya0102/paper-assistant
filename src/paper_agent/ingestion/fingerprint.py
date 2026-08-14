"""Streaming file fingerprints used before expensive parsing."""

from hashlib import sha256
from pathlib import Path

from paper_agent.domain.errors import FileChangedDuringReadError
from paper_agent.domain.ingestion import FileFingerprint


class Sha256Fingerprinter:
    def __init__(self, *, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._chunk_size = chunk_size

    def fingerprint(self, path: Path) -> FileFingerprint:
        before = path.stat()
        digest = sha256()
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(self._chunk_size):
                digest.update(chunk)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise FileChangedDuringReadError(str(path))
        return FileFingerprint(
            sha256=digest.hexdigest(),
            file_size=after.st_size,
            mtime_ns=after.st_mtime_ns,
        )

