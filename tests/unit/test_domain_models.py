from pathlib import PurePosixPath
from uuid import uuid4

import pytest
from pydantic import ValidationError

from paper_agent.domain.document import (
    BoundingBox,
    CanonicalParsedDocument,
    DocumentBlock,
    DocumentPage,
    ParserDescriptor,
)
from paper_agent.domain.enums import BlockType, FileStatus
from paper_agent.domain.paper import FileLocation, PaperFile


def test_paper_file_requires_lowercase_sha256() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PaperFile(
            file_id=uuid4(),
            project_id=uuid4(),
            file_size=10,
            file_hash="A" * 64,
        )


def test_resolved_file_requires_paper_and_version() -> None:
    with pytest.raises(ValueError, match="resolved file states"):
        PaperFile(
            file_id=uuid4(),
            project_id=uuid4(),
            file_size=10,
            file_hash="a" * 64,
            status=FileStatus.PARSED,
        )


def test_location_cannot_escape_project() -> None:
    with pytest.raises(ValueError, match="inside the project"):
        FileLocation(
            location_id=uuid4(),
            project_id=uuid4(),
            file_id=uuid4(),
            relative_path=PurePosixPath("../paper.pdf"),
            file_name="paper.pdf",
            mtime_ns=1,
        )


def test_canonical_document_rejects_duplicate_block_ids() -> None:
    paper_id = uuid4()
    version_id = uuid4()
    block = DocumentBlock(
        block_id="b-1",
        block_type=BlockType.PARAGRAPH,
        text="content",
        bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10),
        reading_order=0,
    )
    with pytest.raises(ValidationError, match="block_id must be unique"):
        CanonicalParsedDocument(
            paper_id=paper_id,
            version_id=version_id,
            source_file_id=uuid4(),
            parser=ParserDescriptor(name="test", version="1"),
            pages=(
                DocumentPage(page_number=1, width=100, height=100, blocks=(block,)),
                DocumentPage(page_number=2, width=100, height=100, blocks=(block,)),
            ),
        )

