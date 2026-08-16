"""Local content-addressed Artifact blob store.

Blobs live below  .paper-agent/artifacts/blobs/sha256/<ab>/<hash>.json.gz .
Paths are always derived from the content hash -- never from user input.
Writes use a temporary file plus atomic rename, reads re-verify the SHA-256,
and gzip compression is deterministic so identical payloads byte-match.
The store implements the same ArtifactBlobStore port an S3 backend would use,
so the rest of the system is storage-backend agnostic.
"""

import gzip
import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Final

from paper_agent.domain.errors import ErrorCode, PaperAgentError


class ArtifactBlobError(PaperAgentError):
    """Stable error for missing, corrupt or invalid artifact blobs."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(code, message)


STORAGE_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^sha256/[0-9a-f]{2}/[0-9a-f]{64}\.json\.gz$"
)


class LocalArtifactBlobStore:
    def __init__(self, project_root: Path) -> None:
        self._blob_root = project_root / ".paper-agent" / "artifacts" / "blobs"

    def put(self, *, content_hash: str, data: bytes) -> str:
        if len(content_hash) != 64 or any(c not in "0123456789abcdef" for c in content_hash):
            raise ValueError("content_hash must be a lowercase SHA-256 digest")
        if sha256(data).hexdigest() != content_hash:
            raise ValueError("data does not match content_hash")
        key = f"sha256/{content_hash[:2]}/{content_hash}.json.gz"
        target = self._blob_root / key
        if target.is_file():
            try:
                existing = gzip.decompress(target.read_bytes())
            except (OSError, EOFError):
                existing = b""
            if sha256(existing).hexdigest() == content_hash:
                return key  # idempotent, verified content-addressed put
            # A corrupt blob at the correct content address is repaired by the
            # same atomic replacement path used for a first write.
        compressed = gzip.compress(data, compresslevel=6, mtime=0)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".gz")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(compressed)
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return key

    def get(self, *, storage_key: str) -> bytes:
        if not STORAGE_KEY_PATTERN.fullmatch(storage_key):
            raise ArtifactBlobError(ErrorCode.INVALID_PATH, "Invalid artifact storage key")
        path = self._blob_root / storage_key
        if not path.is_file():
            raise ArtifactBlobError(ErrorCode.FILE_NOT_FOUND, "Artifact blob not found")
        try:
            data = gzip.decompress(path.read_bytes())
        except (OSError, EOFError) as error:
            raise ArtifactBlobError(ErrorCode.ARTIFACT_CORRUPT, "Artifact blob is corrupt") from error
        expected = storage_key.rsplit("/", 1)[-1][:64]
        if sha256(data).hexdigest() != expected:
            raise ArtifactBlobError(ErrorCode.ARTIFACT_CORRUPT, "Artifact blob hash mismatch")
        return data

    def exists(self, *, storage_key: str) -> bool:
        if not STORAGE_KEY_PATTERN.fullmatch(storage_key):
            return False
        return (self._blob_root / storage_key).is_file()

    def delete(self, *, storage_key: str) -> None:
        if not STORAGE_KEY_PATTERN.fullmatch(storage_key):
            raise ArtifactBlobError(ErrorCode.INVALID_PATH, "Invalid artifact storage key")
        path = self._blob_root / storage_key
        if path.is_file():
            path.unlink()
