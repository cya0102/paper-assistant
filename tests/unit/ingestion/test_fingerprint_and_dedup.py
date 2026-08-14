from hashlib import sha256
from pathlib import PurePosixPath
from uuid import uuid4

from paper_agent.domain.enums import IngestionDisposition
from paper_agent.domain.paper import FileLocation, PaperFile
from paper_agent.ingestion.dedup import classify_file
from paper_agent.ingestion.fingerprint import Sha256Fingerprinter


def make_file(file_hash: str) -> PaperFile:
    return PaperFile(
        file_id=uuid4(),
        project_id=uuid4(),
        file_size=5,
        file_hash=file_hash,
    )


def test_fingerprinter_streams_sha256(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    payload = b"paper-content" * 1024
    path.write_bytes(payload)

    result = Sha256Fingerprinter(chunk_size=17).fingerprint(path)

    assert result.sha256 == sha256(payload).hexdigest()
    assert result.file_size == len(payload)


def test_dedup_classifies_all_file_cases() -> None:
    old_file = make_file("a" * 64)
    same_file = old_file
    duplicate_file = make_file("b" * 64)
    location = FileLocation(
        location_id=uuid4(),
        project_id=old_file.project_id,
        file_id=old_file.file_id,
        relative_path=PurePosixPath("paper.pdf"),
        file_name="paper.pdf",
        mtime_ns=1,
    )

    assert classify_file(
        current_location=location,
        current_file=old_file,
        matching_hash_file=same_file,
    ).disposition == IngestionDisposition.UNCHANGED
    assert classify_file(
        current_location=None,
        current_file=None,
        matching_hash_file=duplicate_file,
    ).disposition == IngestionDisposition.DUPLICATE
    assert classify_file(
        current_location=location,
        current_file=old_file,
        matching_hash_file=None,
    ).disposition == IngestionDisposition.MODIFIED
    assert classify_file(
        current_location=None,
        current_file=None,
        matching_hash_file=None,
    ).disposition == IngestionDisposition.NEW

