from uuid import uuid4

from paper_agent.domain.document import (
    BoundingBox,
    CanonicalParsedDocument,
    DocumentBlock,
    DocumentPage,
    ParserDescriptor,
)
from paper_agent.domain.chunk import SemanticGroup
from paper_agent.domain.enums import BlockType, ElementType, SemanticGroupType
from paper_agent.ingestion.chunker import ChunkingConfig, SemanticChunker
from paper_agent.ingestion.structure_pipeline import DocumentStructureProcessor


def _document() -> CanonicalParsedDocument:
    paper_id, version_id, file_id = uuid4(), uuid4(), uuid4()
    blocks = (
        DocumentBlock(
            block_id="b0", block_type=BlockType.PARAGRAPH, text="Paper title", reading_order=0
        ),
        DocumentBlock(
            block_id="b1", block_type=BlockType.HEADING, text="1 Introduction",
            reading_order=1, attributes={"level": 1},
        ),
        DocumentBlock(
            block_id="b2", block_type=BlockType.PARAGRAPH,
            text="Figure 1 summarizes the architecture.", reading_order=2,
        ),
        DocumentBlock(
            block_id="b3", block_type=BlockType.FIGURE,
            text="Figure 1: System architecture", reading_order=3,
            bbox=BoundingBox(x0=10, y0=20, x1=90, y1=60),
        ),
        DocumentBlock(
            block_id="b4", block_type=BlockType.HEADING, text="1.1 Method",
            reading_order=4, attributes={"level": 2},
        ),
        DocumentBlock(
            block_id="b5", block_type=BlockType.EQUATION, text="E = mc^2", reading_order=5
        ),
        DocumentBlock(
            block_id="b6", block_type=BlockType.PARAGRAPH,
            text="where m denotes mass.", reading_order=6,
        ),
        DocumentBlock(
            block_id="b7", block_type=BlockType.HEADING, text="2 Results",
            reading_order=7, attributes={"level": 1},
        ),
        DocumentBlock(
            block_id="b8", block_type=BlockType.TABLE,
            text="Table 1: Main results", reading_order=8,
        ),
        DocumentBlock(
            block_id="b9", block_type=BlockType.ALGORITHM,
            text="Algorithm 1: Training procedure", reading_order=9,
        ),
    )
    return CanonicalParsedDocument(
        paper_id=paper_id,
        version_id=version_id,
        source_file_id=file_id,
        parser=ParserDescriptor(name="test", version="1"),
        pages=(DocumentPage(page_number=1, width=100, height=100, blocks=blocks),),
    )


def test_structure_processor_recovers_hierarchy_and_assigns_every_block() -> None:
    document = _document()
    structured, groups = DocumentStructureProcessor().build(document)

    assert [section.title for section in structured.sections] == [
        "Front Matter", "1 Introduction", "1.1 Method", "2 Results"
    ]
    method = structured.sections[2]
    assert method.level == 2
    assert method.parent_section_id == structured.sections[1].section_id
    assert method.section_path == "1 Introduction > 1.1 Method"
    assigned = [block_id for section in structured.sections for block_id in section.source_block_ids]
    assert assigned == [f"b{index}" for index in range(10)]
    assert len(assigned) == len(set(assigned))

    assert {element.element_type for element in structured.elements} == {
        ElementType.FIGURE,
        ElementType.EQUATION,
        ElementType.TABLE,
        ElementType.ALGORITHM,
    }
    equation_group = next(
        group for group in groups if group.group_type == SemanticGroupType.ELEMENT_DEPENDENCY
        and "E = mc^2" in group.text
    )
    assert equation_group.source_block_ids == ("b5", "b6")
    assert equation_group.related_element_ids


def test_section_aware_chunker_never_crosses_sections_and_keeps_full_provenance() -> None:
    document = _document()
    structured, groups = DocumentStructureProcessor().build(document)
    chunks = SemanticChunker(
        ChunkingConfig(target_tokens=12, hard_max_tokens=18, min_fill_ratio=0.5)
    ).chunk(structured, groups)

    section_by_id = {section.section_id: section for section in structured.sections}
    element_ids = {element.element_id for element in structured.elements}
    group_ids = {group.group_id for group in groups}
    document_block_ids = {
        block.block_id for page in document.pages for block in page.blocks
    }
    assert chunks
    for chunk in chunks:
        assert chunk.paper_id == document.paper_id
        assert chunk.version_id == document.version_id
        assert chunk.section_path == section_by_id[chunk.section_id].section_path
        assert set(chunk.source_group_ids) <= group_ids
        assert set(chunk.source_block_ids) <= document_block_ids
        assert set(chunk.related_element_ids) <= element_ids
        assert chunk.page_start == 1 == chunk.page_end
        assert all(
            group.section_id == chunk.section_id
            for group in groups
            if group.group_id in chunk.source_group_ids
        )


def test_structure_and_chunk_ids_are_deterministic() -> None:
    document = _document()
    processor = DocumentStructureProcessor()
    chunker = SemanticChunker()
    first_structure, first_groups = processor.build(document)
    second_structure, second_groups = processor.build(document)

    assert first_structure == second_structure
    assert first_groups == second_groups
    assert chunker.chunk(first_structure, first_groups) == chunker.chunk(
        second_structure, second_groups
    )


def test_oversized_plain_group_is_split_but_dependency_group_stays_atomic() -> None:
    structured, _ = DocumentStructureProcessor().build(_document())
    section = structured.sections[0]
    plain = SemanticGroup(
        group_id=uuid4(), paper_id=structured.paper_id, version_id=structured.version_id,
        section_id=section.section_id, group_order=0, group_type=SemanticGroupType.TEXT,
        text="语" * 40, token_count=40, page_start=1, page_end=1,
        source_block_ids=("b0",), related_element_ids=(),
        structure_version=structured.structure_version,
    )
    dependency = SemanticGroup(
        group_id=uuid4(), paper_id=structured.paper_id, version_id=structured.version_id,
        section_id=section.section_id, group_order=1,
        group_type=SemanticGroupType.ELEMENT_DEPENDENCY,
        text="式" * 30, token_count=30, page_start=1, page_end=1,
        source_block_ids=("b0",), related_element_ids=(),
        structure_version=structured.structure_version,
    )
    chunks = SemanticChunker(
        ChunkingConfig(target_tokens=10, hard_max_tokens=15)
    ).chunk(structured, (plain, dependency))

    plain_chunks = [chunk for chunk in chunks if plain.group_id in chunk.source_group_ids]
    dependency_chunks = [chunk for chunk in chunks if dependency.group_id in chunk.source_group_ids]
    assert all(chunk.token_count <= 15 for chunk in plain_chunks)
    assert len(dependency_chunks) == 1
    assert dependency_chunks[0].token_count == 30
