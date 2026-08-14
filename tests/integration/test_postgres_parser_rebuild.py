import os
from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.database import upgrade_database
from paper_agent.domain.document import CanonicalParsedDocument, DocumentBlock, DocumentPage, ParserDescriptor
from paper_agent.domain.enums import BlockType, PipelineStage
from paper_agent.domain.ingestion import IngestionRequest
from paper_agent.domain.metadata import PaperMetadata
from paper_agent.domain.paper import Paper, PaperIdentityResolution, PaperVersion
from paper_agent.ingestion.chunker import SemanticChunker
from paper_agent.ingestion.pipeline import IngestionPipeline
from paper_agent.ingestion.structure_pipeline import DocumentStructureProcessor
from paper_agent.storage.local import LocalParsedDocumentStore
from paper_agent.storage.postgres.models import ChunkRow, DerivedDataStateRow, NoteRow, PaperFileRow, PaperRow, ParsedDocumentRow, ProjectRow, SectionRow
from paper_agent.storage.postgres.unit_of_work import SqlAlchemyUnitOfWorkFactory


DATABASE_URL = os.getenv("PAPER_AGENT_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="PAPER_AGENT_TEST_DATABASE_URL is required for real PostgreSQL tests",
)


class StableIdentityResolver:
    def __init__(self, paper_id, version_id):
        self.paper_id = paper_id
        self.version_id = version_id

    def resolve(self, discovered_file, paper_file, metadata, lookup):
        del discovered_file, paper_file, metadata, lookup
        return PaperIdentityResolution(
            paper=Paper(paper_id=self.paper_id, canonical_title="Parser Rebuild Paper"),
            version=PaperVersion(version_id=self.version_id, paper_id=self.paper_id),
        )


class IntegrationParser:
    name = "integration-parser"

    def __init__(self, version):
        self.version = version
        self.calls = 0

    def extract(self, path):
        payload = path.read_bytes()
        return PaperMetadata(
            title="Parser Rebuild Paper",
            normalized_title="parser rebuild paper",
            page_count=1,
            content_hash=sha256(payload).hexdigest(),
        )

    def parse(self, request):
        self.calls += 1
        return CanonicalParsedDocument(
            paper_id=request.identity.paper.paper_id,
            version_id=request.identity.version.version_id,
            source_file_id=request.source_file.file_id,
            parser=ParserDescriptor(name=self.name, version=self.version),
            pages=(
                DocumentPage(
                    page_number=1,
                    width=100,
                    height=100,
                    blocks=(
                        DocumentBlock(
                            block_id="heading",
                            block_type=BlockType.HEADING,
                            text="1 Introduction",
                            reading_order=0,
                            attributes={"level": 1},
                        ),
                        DocumentBlock(
                            block_id="body",
                            block_type=BlockType.PARAGRAPH,
                            text=f"Content parsed by version {self.version}.",
                            reading_order=1,
                        ),
                    ),
                ),
            ),
        )


class CountingStructureProcessor:
    def __init__(self):
        self._delegate = DocumentStructureProcessor()
        self.version = self._delegate.version
        self.calls = 0

    def build(self, document):
        self.calls += 1
        return self._delegate.build(document)


class CountingChunker:
    def __init__(self):
        self._delegate = SemanticChunker()
        self.version = self._delegate.version
        self.calls = 0

    def chunk(self, structured, groups):
        self.calls += 1
        return self._delegate.chunk(structured, groups)


def test_parser_change_rebuilds_postgres_structure_and_preserves_section_note(tmp_path):
    assert DATABASE_URL is not None
    upgrade_database(DATABASE_URL)
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    project_id, paper_id, version_id, user_id = uuid4(), uuid4(), uuid4(), uuid4()
    note_id, force_note_id = uuid4(), uuid4()
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"synthetic-pdf")
    processor = CountingStructureProcessor()
    chunker = CountingChunker()
    identity = StableIdentityResolver(paper_id, version_id)
    try:
        with factory.begin() as session:
            stale_projects = tuple(
                session.scalars(
                    select(ProjectRow.project_id).where(ProjectRow.name == "parser-rebuild-test")
                )
            )
            if stale_projects:
                stale_papers = tuple(
                    session.scalars(
                        select(PaperFileRow.paper_id).where(
                            PaperFileRow.project_id.in_(stale_projects),
                            PaperFileRow.paper_id.is_not(None),
                        )
                    )
                )
                if stale_papers:
                    session.execute(delete(PaperRow).where(PaperRow.paper_id.in_(stale_papers)))
                    session.flush()
                session.execute(
                    delete(ProjectRow).where(ProjectRow.project_id.in_(stale_projects))
                )
            session.add(
                ProjectRow(
                    project_id=project_id,
                    name="parser-rebuild-test",
                    root_path=str(tmp_path.resolve()),
                )
            )
        first_parser = IntegrationParser("1")
        first_pipeline = IngestionPipeline(
            unit_of_work_factory=SqlAlchemyUnitOfWorkFactory(factory),
            metadata_extractor=first_parser,
            identity_resolver=identity,
            parser=first_parser,
            parsed_document_store=LocalParsedDocumentStore(tmp_path),
            structure_processor=processor,
            chunker=chunker,
        )
        first = first_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))
        assert first.items[0].stage == PipelineStage.CHUNKED

        with factory.begin() as session:
            section_id = session.scalar(
                select(SectionRow.section_id).where(SectionRow.version_id == version_id)
            )
            assert section_id is not None
            session.add(
                NoteRow(
                    note_id=note_id,
                    user_id=user_id,
                    project_id=project_id,
                    paper_id=paper_id,
                    section_id=section_id,
                    content="must survive derived Section replacement",
                    tags_json=[],
                )
            )
            original_sections = session.scalar(
                select(func.count()).select_from(SectionRow).where(SectionRow.version_id == version_id)
            )
            original_chunks = session.scalar(
                select(func.count()).select_from(ChunkRow).where(ChunkRow.version_id == version_id)
            )

        second_parser = IntegrationParser("2")
        second_pipeline = IngestionPipeline(
            unit_of_work_factory=SqlAlchemyUnitOfWorkFactory(factory),
            metadata_extractor=second_parser,
            identity_resolver=identity,
            parser=second_parser,
            parsed_document_store=LocalParsedDocumentStore(tmp_path),
            structure_processor=processor,
            chunker=chunker,
        )
        second = second_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

        assert second.items[0].stage == PipelineStage.CHUNKED
        assert first_parser.calls + second_parser.calls == 2
        assert processor.calls == 2
        assert chunker.calls == 2
        with factory() as session:
            parsed_hash = session.scalar(
                select(ParsedDocumentRow.document_hash).where(
                    ParsedDocumentRow.version_id == version_id,
                    ParsedDocumentRow.parser_name == second_parser.name,
                    ParsedDocumentRow.parser_version == second_parser.version,
                    ParsedDocumentRow.schema_version == 1,
                )
            )
            state = session.get(DerivedDataStateRow, version_id)
            note = session.get(NoteRow, note_id)
            assert parsed_hash is not None
            assert state is not None and state.document_hash == parsed_hash
            assert session.scalar(
                select(func.count()).select_from(SectionRow).where(SectionRow.version_id == version_id)
            ) == original_sections
            assert session.scalar(
                select(func.count()).select_from(ChunkRow).where(ChunkRow.version_id == version_id)
            ) == original_chunks
            assert note is not None and note.section_id is None

        with factory.begin() as session:
            current_section_id = session.scalar(
                select(SectionRow.section_id).where(SectionRow.version_id == version_id)
            )
            assert current_section_id is not None
            session.add(
                NoteRow(
                    note_id=force_note_id,
                    user_id=user_id,
                    project_id=project_id,
                    paper_id=paper_id,
                    section_id=current_section_id,
                    content="must survive force reindex",
                    tags_json=[],
                )
            )

        forced = second_pipeline.ingest(
            IngestionRequest(project_id, tmp_path.resolve(), force_reindex=True)
        )

        assert forced.items[0].stage == PipelineStage.CHUNKED
        assert first_parser.calls + second_parser.calls == 3
        assert processor.calls == 3
        assert chunker.calls == 3
        with factory() as session:
            force_note = session.get(NoteRow, force_note_id)
            assert force_note is not None and force_note.section_id is None
    finally:
        with factory.begin() as session:
            paper = session.get(PaperRow, paper_id)
            if paper is not None:
                session.delete(paper)
                session.flush()
            session.execute(delete(ProjectRow).where(ProjectRow.project_id == project_id))
        engine.dispose()
