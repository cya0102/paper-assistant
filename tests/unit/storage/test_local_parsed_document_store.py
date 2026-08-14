from uuid import uuid4

from paper_agent.domain.document import (
    CanonicalParsedDocument,
    DocumentBlock,
    DocumentPage,
    ParserDescriptor,
)
from paper_agent.domain.enums import BlockType
from paper_agent.storage.local.parsed_document_store import LocalParsedDocumentStore


def test_store_round_trips_json_and_renders_markdown(tmp_path) -> None:
    document = CanonicalParsedDocument(
        paper_id=uuid4(),
        version_id=uuid4(),
        source_file_id=uuid4(),
        parser=ParserDescriptor(name="fake-parser", version="1.0"),
        pages=(
            DocumentPage(
                page_number=1,
                width=612,
                height=792,
                blocks=(
                    DocumentBlock(
                        block_id="b-1",
                        block_type=BlockType.HEADING,
                        text="3 Method",
                        reading_order=0,
                        attributes={"level": 1},
                    ),
                    DocumentBlock(
                        block_id="b-2",
                        block_type=BlockType.PARAGRAPH,
                        text="We first extract features.",
                        reading_order=1,
                    ),
                ),
            ),
        ),
    )
    store = LocalParsedDocumentStore(tmp_path)

    artifacts = store.save(document)
    loaded = store.load(document.paper_id, document.version_id)

    assert loaded == document
    assert store.exists(document.paper_id, document.version_id)
    markdown = (tmp_path / artifacts.document_markdown_path).read_text(encoding="utf-8")
    assert "# 3 Method" in markdown
    assert "<!-- page: 1 -->" in markdown
    assert len(artifacts.document_hash) == 64

