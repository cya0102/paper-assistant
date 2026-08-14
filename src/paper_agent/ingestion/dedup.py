"""Pure SHA-256 deduplication decisions."""

from dataclasses import dataclass
from uuid import UUID

from paper_agent.domain.enums import IngestionDisposition
from paper_agent.domain.paper import FileLocation, PaperFile


@dataclass(frozen=True, slots=True)
class DedupDecision:
    disposition: IngestionDisposition
    reusable_file_id: UUID | None = None


def classify_file(
    *,
    current_location: FileLocation | None,
    current_file: PaperFile | None,
    matching_hash_file: PaperFile | None,
) -> DedupDecision:
    if current_location is not None and current_file is None:
        raise ValueError("current_file is required when current_location exists")
    if current_location is not None and current_file is not None and matching_hash_file is not None:
        if current_file.file_id == matching_hash_file.file_id:
            return DedupDecision(IngestionDisposition.UNCHANGED, current_file.file_id)
    if matching_hash_file is not None:
        return DedupDecision(IngestionDisposition.DUPLICATE, matching_hash_file.file_id)
    if current_location is not None:
        return DedupDecision(IngestionDisposition.MODIFIED)
    return DedupDecision(IngestionDisposition.NEW)

