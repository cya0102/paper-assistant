from hashlib import sha256
from dataclasses import replace
from pathlib import PurePosixPath
from uuid import UUID, uuid4

from paper_agent.domain.document import CanonicalParsedDocument, DocumentPage, ParserDescriptor
from paper_agent.domain.chunk import DerivedDataState
from paper_agent.domain.document import DocumentBlock
from paper_agent.domain.enums import BlockType, FileStatus, IngestionDisposition, PipelineStage, RunStatus
from paper_agent.domain.ingestion import IngestionRequest
from paper_agent.domain.metadata import PaperMetadata
from paper_agent.domain.paper import FileLocation, Paper, PaperFile, PaperIdentityResolution, PaperVersion
from paper_agent.ingestion.identity import DeterministicPaperIdentityResolver
from paper_agent.ingestion.pipeline import IngestionPipeline
from paper_agent.ingestion.chunker import SemanticChunker
from paper_agent.ingestion.structure_pipeline import DocumentStructureProcessor
from paper_agent.storage.local.parsed_document_store import LocalParsedDocumentStore


class MemoryDatabase:
    def __init__(self) -> None:
        self.files: dict[UUID, PaperFile] = {}
        self.locations: dict[tuple[UUID, PurePosixPath], FileLocation] = {}
        self.runs: dict[UUID, dict[str, object]] = {}
        self.items: list[object] = []
        self.documents: list[object] = []
        self.identities: dict[UUID, PaperIdentityResolution] = {}
        self.structures: dict[UUID, object] = {}
        self.groups: dict[UUID, tuple[object, ...]] = {}
        self.chunks: dict[UUID, tuple[object, ...]] = {}
        self.derived_states: dict[UUID, DerivedDataState] = {}


class MemoryFileRepository:
    def __init__(self, database: MemoryDatabase) -> None:
        self.database = database

    def get_location(self, project_id, relative_path):
        location = self.database.locations.get((project_id, relative_path))
        if location is None:
            return None
        return location, self.database.files[location.file_id]

    def find_by_hash(self, project_id, file_hash):
        return next(
            (
                item
                for item in self.database.files.values()
                if item.project_id == project_id and item.file_hash == file_hash
            ),
            None,
        )

    def get_or_add_file(self, paper_file):
        existing = self.find_by_hash(paper_file.project_id, paper_file.file_hash)
        if existing:
            return existing, False
        self.database.files[paper_file.file_id] = paper_file
        return paper_file, True

    def upsert_location(self, location):
        self.database.locations[(location.project_id, location.relative_path)] = location

    def save_identity(self, file_id, resolution):
        updated = replace(
            self.database.files[file_id],
            paper_id=resolution.paper.paper_id,
            version_id=resolution.version.version_id,
            status=FileStatus.IDENTITY_RESOLVED,
        )
        self.database.files[file_id] = updated
        self.database.identities[file_id] = resolution
        return updated

    def update_status(self, file_id, status):
        self.database.files[file_id] = replace(self.database.files[file_id], status=status)

    def save_metadata(self, file_id, metadata):
        updated = replace(
            self.database.files[file_id],
            content_hash=metadata.content_hash,
            page_count=metadata.page_count,
        )
        self.database.files[file_id] = updated
        return updated

    def get_identity(self, file_id):
        return self.database.identities.get(file_id)

    def find_by_content_hash(self, project_id, content_hash):
        for file_id, paper_file in self.database.files.items():
            if paper_file.project_id == project_id and paper_file.content_hash == content_hash:
                identity = self.database.identities.get(file_id)
                if identity:
                    return identity
        return None

    def find_by_doi(self, doi):
        return next(
            (identity for identity in self.database.identities.values() if identity.paper.doi == doi),
            None,
        )

    def find_by_arxiv_id(self, arxiv_id):
        return next(
            (
                identity
                for identity in self.database.identities.values()
                if identity.paper.arxiv_id == arxiv_id
            ),
            None,
        )

    def find_by_title_authors(self, normalized_title, normalized_authors):
        return None

    def mark_missing_locations(self, project_id, seen_paths):
        return tuple(
            path
            for owner_project_id, path in self.database.locations
            if owner_project_id == project_id and path not in seen_paths
        )


class MemoryProjectRepository:
    def ensure(self, project):
        return None


class MemoryDocumentRepository:
    def __init__(self, database: MemoryDatabase) -> None:
        self.database = database

    def add(self, document, artifacts):
        self.database.documents.append((document, artifacts))

    def has_current(self, version_id, parser_name, parser_version, schema_version):
        return self.current_document_hash(
            version_id, parser_name, parser_version, schema_version
        ) is not None

    def current_document_hash(self, version_id, parser_name, parser_version, schema_version):
        return next(
            (
                artifacts.document_hash
                for document, artifacts in self.database.documents
                if document.version_id == version_id
                and document.parser.name == parser_name
                and document.parser.version == parser_version
                and document.schema_version == schema_version
            ),
            None,
        )


class MemoryRunRepository:
    def __init__(self, database: MemoryDatabase) -> None:
        self.database = database

    def create(self, run_id, request):
        self.database.runs[run_id] = {"request": request, "status": RunStatus.RUNNING}

    def record_item(self, run_id, item):
        self.database.items.append((run_id, item))

    def complete(self, run_id, status, counters):
        self.database.runs[run_id].update(status=status, counters=counters)


class MemoryDerivedDataRepository:
    def __init__(self, database: MemoryDatabase) -> None:
        self.database = database

    def get_state(self, version_id):
        return self.database.derived_states.get(version_id)

    def replace_structure(self, structured, groups, *, document_hash):
        self.database.structures[structured.version_id] = structured
        self.database.groups[structured.version_id] = groups
        self.database.chunks.pop(structured.version_id, None)
        self.database.derived_states[structured.version_id] = DerivedDataState(
            version_id=structured.version_id,
            structure_version=structured.structure_version,
            document_hash=document_hash,
        )

    def load_structure(self, version_id):
        return self.database.structures[version_id]

    def load_groups(self, version_id):
        return self.database.groups[version_id]

    def replace_chunks(self, version_id, chunks, chunking_version):
        self.database.chunks[version_id] = chunks
        state = self.database.derived_states[version_id]
        self.database.derived_states[version_id] = replace(
            state, chunking_version=chunking_version
        )


class MemoryUnitOfWork:
    def __init__(self, database: MemoryDatabase) -> None:
        self.projects = MemoryProjectRepository()
        self.files = MemoryFileRepository(database)
        self.documents = MemoryDocumentRepository(database)
        self.derived = MemoryDerivedDataRepository(database)
        self.runs = MemoryRunRepository(database)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def commit(self):
        return None

    def rollback(self):
        return None


class IdentityResolver:
    def resolve(self, discovered_file, paper_file, metadata, lookup):
        del metadata, lookup
        paper = Paper(paper_id=uuid4(), canonical_title=discovered_file.absolute_path.stem)
        return PaperIdentityResolution(
            paper=paper,
            version=PaperVersion(version_id=uuid4(), paper_id=paper.paper_id),
        )


class CountingParser:
    name = "counting-parser"
    version = "1"

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, request):
        self.calls += 1
        return CanonicalParsedDocument(
            paper_id=request.identity.paper.paper_id,
            version_id=request.identity.version.version_id,
            source_file_id=request.source_file.file_id,
            parser=ParserDescriptor(name=self.name, version=self.version),
            pages=(DocumentPage(page_number=1, width=100, height=100),),
        )

    def extract(self, path):
        payload = path.read_bytes()
        return PaperMetadata(
            title=path.stem,
            normalized_title=path.stem.casefold(),
            page_count=1,
            content_hash=sha256(payload).hexdigest(),
        )


class SelectivelyFailingParser(CountingParser):
    def parse(self, request):
        if request.source_path.name == "bad.pdf":
            raise RuntimeError("synthetic parser failure")
        return super().parse(request)


class VersionTwoParser(CountingParser):
    version = "2"


class SchemaTwoParser(CountingParser):
    def parse(self, request):
        self.calls += 1
        return CanonicalParsedDocument(
            schema_version=2,
            paper_id=request.identity.paper.paper_id,
            version_id=request.identity.version.version_id,
            source_file_id=request.source_file.file_id,
            parser=ParserDescriptor(name=self.name, version=self.version),
            pages=(DocumentPage(page_number=1, width=100, height=100),),
        )


class FixedContentParser(CountingParser):
    def extract(self, path):
        return PaperMetadata(
            title="Same Paper",
            normalized_title="same paper",
            page_count=1,
            content_hash="c" * 64,
        )


class StructuredCountingParser(CountingParser):
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
                            text="A traceable paragraph.",
                            reading_order=1,
                        ),
                    ),
                ),
            ),
        )


class StructuredVersionTwoParser(StructuredCountingParser):
    version = "2"


class StructuredSchemaTwoParser(StructuredCountingParser):
    def parse(self, request):
        return super().parse(request).model_copy(update={"schema_version": 2})


class CountingStructureProcessor(DocumentStructureProcessor):
    def __init__(self, version_suffix=""):
        super().__init__()
        self.version = f"{self.version}{version_suffix}"
        self.calls = 0

    def build(self, document):
        self.calls += 1
        structured, groups = super().build(document)
        if structured.structure_version == self.version:
            return structured, groups
        structured = replace(
            structured,
            structure_version=self.version,
            sections=tuple(replace(value, structure_version=self.version) for value in structured.sections),
            elements=tuple(replace(value, structure_version=self.version) for value in structured.elements),
        )
        groups = tuple(replace(value, structure_version=self.version) for value in groups)
        return structured, groups


class CountingChunker(SemanticChunker):
    def __init__(self, version_suffix=""):
        super().__init__()
        self.version = f"{self.version}{version_suffix}"
        self.calls = 0

    def chunk(self, structured, groups):
        self.calls += 1
        return super().chunk(structured, groups)


class CountingIndexer:
    version = "test-index-v1"

    def __init__(self):
        self.current = False
        self.calls = 0

    def is_current(self, project_id, version_id):
        del project_id, version_id
        return self.current

    def index_version(self, project_id, version_id, *, force=False):
        del project_id, version_id, force
        self.calls += 1
        self.current = True
        return object()


def test_pipeline_does_not_parse_unchanged_or_renamed_duplicate(tmp_path) -> None:
    database = MemoryDatabase()
    parser = CountingParser()
    pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=IdentityResolver(),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
    )
    project_id = uuid4()
    first = tmp_path / "first.pdf"
    first.write_bytes(b"same-pdf-bytes")
    request = IngestionRequest(project_id=project_id, project_root=tmp_path.resolve())

    first_report = pipeline.ingest(request)
    second_report = pipeline.ingest(request)
    renamed = tmp_path / "renamed.pdf"
    renamed.write_bytes(first.read_bytes())
    third_report = pipeline.ingest(request)

    assert first_report.items[0].disposition == IngestionDisposition.NEW
    assert first_report.items[0].stage == PipelineStage.PARSED
    assert second_report.items[0].disposition == IngestionDisposition.UNCHANGED
    assert {item.disposition for item in third_report.items} == {
        IngestionDisposition.UNCHANGED,
        IngestionDisposition.DUPLICATE,
    }
    assert parser.calls == 1
    assert len(database.files) == 1
    assert len(database.locations) == 2


def test_pipeline_isolates_parser_failure_per_file(tmp_path) -> None:
    database = MemoryDatabase()
    parser = SelectivelyFailingParser()
    pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=IdentityResolver(),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
    )
    (tmp_path / "bad.pdf").write_bytes(b"bad-pdf")
    (tmp_path / "good.pdf").write_bytes(b"good-pdf")

    report = pipeline.ingest(
        IngestionRequest(project_id=uuid4(), project_root=tmp_path.resolve())
    )

    by_path = {item.relative_path: item for item in report.items}
    assert by_path[PurePosixPath("bad.pdf")].disposition == IngestionDisposition.FAILED
    assert by_path[PurePosixPath("bad.pdf")].stage == PipelineStage.FAILED
    assert by_path[PurePosixPath("good.pdf")].stage == PipelineStage.PARSED
    assert len(database.documents) == 1


def test_force_reindex_parses_unchanged_file_again(tmp_path) -> None:
    database = MemoryDatabase()
    parser = CountingParser()
    pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=IdentityResolver(),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
    )
    project_id = uuid4()
    (tmp_path / "paper.pdf").write_bytes(b"paper")
    pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    report = pipeline.ingest(
        IngestionRequest(project_id, tmp_path.resolve(), force_reindex=True)
    )

    assert report.items[0].disposition == IngestionDisposition.UNCHANGED
    assert report.items[0].stage == PipelineStage.PARSED
    assert parser.calls == 2


def test_parser_version_change_reparses_without_force(tmp_path) -> None:
    database = MemoryDatabase()
    first_parser = CountingParser()
    first_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=first_parser,
        identity_resolver=IdentityResolver(),
        parser=first_parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
    )
    project_id = uuid4()
    (tmp_path / "paper.pdf").write_bytes(b"paper")
    first_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    second_parser = VersionTwoParser()
    second_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=second_parser,
        identity_resolver=IdentityResolver(),
        parser=second_parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
    )
    report = second_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    assert report.items[0].stage == PipelineStage.PARSED
    assert second_parser.calls == 1


def test_canonical_schema_change_reparses_without_force(tmp_path) -> None:
    database = MemoryDatabase()
    first_parser = CountingParser()
    first_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=first_parser,
        identity_resolver=IdentityResolver(),
        parser=first_parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
    )
    project_id = uuid4()
    (tmp_path / "paper.pdf").write_bytes(b"paper")
    first_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    second_parser = SchemaTwoParser()
    second_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=second_parser,
        identity_resolver=IdentityResolver(),
        parser=second_parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        canonical_schema_version=2,
    )
    report = second_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    assert report.items[0].stage == PipelineStage.PARSED
    assert second_parser.calls == 1


def test_full_scan_marks_removed_location_missing(tmp_path) -> None:
    database = MemoryDatabase()
    parser = CountingParser()
    pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=IdentityResolver(),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
    )
    project_id = uuid4()
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"paper")
    pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))
    path.unlink()

    report = pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    assert report.scanned == 0
    assert report.missing == 1
    assert report.items[0].disposition == IngestionDisposition.MISSING


def test_different_pdf_bytes_with_same_content_hash_reuse_parsed_version(tmp_path) -> None:
    database = MemoryDatabase()
    parser = FixedContentParser()
    pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=DeterministicPaperIdentityResolver(parser_version=parser.version),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
    )
    project_id = uuid4()
    first = tmp_path / "first.pdf"
    first.write_bytes(b"binary-one")
    first_report = pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))
    second = tmp_path / "second.pdf"
    second.write_bytes(b"different-binary")

    second_report = pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    by_path = {item.relative_path: item for item in second_report.items}
    assert first_report.items[0].stage == PipelineStage.PARSED
    assert by_path[PurePosixPath("second.pdf")].disposition == IngestionDisposition.DUPLICATE
    assert by_path[PurePosixPath("second.pdf")].stage == PipelineStage.PARSED
    assert parser.calls == 1
    assert len(database.files) == 2
    assert len({file.version_id for file in database.files.values()}) == 1


def test_phase1c_recovers_each_derived_stage_by_its_version(tmp_path) -> None:
    database = MemoryDatabase()
    parser = StructuredCountingParser()
    processor = CountingStructureProcessor()
    chunker = CountingChunker()
    project_id = uuid4()
    (tmp_path / "paper.pdf").write_bytes(b"paper")

    first_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=IdentityResolver(),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        structure_processor=processor,
        chunker=chunker,
    )
    first = first_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))
    second = first_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    assert first.items[0].stage == second.items[0].stage == PipelineStage.CHUNKED
    assert (parser.calls, processor.calls, chunker.calls) == (1, 1, 1)

    changed_chunker = CountingChunker("-2")
    reused_processor = CountingStructureProcessor()
    chunk_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=IdentityResolver(),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        structure_processor=reused_processor,
        chunker=changed_chunker,
    )
    rechunked = chunk_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))
    assert rechunked.items[0].stage == PipelineStage.CHUNKED
    assert parser.calls == 1
    assert reused_processor.calls == 0
    assert changed_chunker.calls == 1

    changed_processor = CountingStructureProcessor("-2")
    rebuilt_chunker = CountingChunker("-2")
    structure_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=IdentityResolver(),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        structure_processor=changed_processor,
        chunker=rebuilt_chunker,
    )
    rebuilt = structure_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))
    assert rebuilt.items[0].stage == PipelineStage.CHUNKED
    assert parser.calls == 1
    assert changed_processor.calls == 1
    assert rebuilt_chunker.calls == 1


def test_parser_version_change_rebuilds_structure_and_chunks(tmp_path) -> None:
    database = MemoryDatabase()
    first_parser = StructuredCountingParser()
    first_processor = CountingStructureProcessor()
    first_chunker = CountingChunker()
    first_indexer = CountingIndexer()
    project_id = uuid4()
    (tmp_path / "paper.pdf").write_bytes(b"paper")
    first_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=first_parser,
        identity_resolver=IdentityResolver(),
        parser=first_parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        structure_processor=first_processor,
        chunker=first_chunker,
        indexer=first_indexer,
    )

    first = first_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    second_parser = StructuredVersionTwoParser()
    second_processor = CountingStructureProcessor()
    second_chunker = CountingChunker()
    second_indexer = CountingIndexer()
    second_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=second_parser,
        identity_resolver=IdentityResolver(),
        parser=second_parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        structure_processor=second_processor,
        chunker=second_chunker,
        indexer=second_indexer,
    )
    second = second_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    assert first.items[0].stage == second.items[0].stage == PipelineStage.INDEXED
    assert first_parser.calls + second_parser.calls == 2
    assert first_processor.calls + second_processor.calls == 2
    assert first_chunker.calls + second_chunker.calls == 2
    version_id = next(iter(database.derived_states))
    expected_hash = MemoryDocumentRepository(database).current_document_hash(
        version_id,
        second_parser.name,
        second_parser.version,
        1,
    )
    assert expected_hash is not None
    assert database.derived_states[version_id].document_hash == expected_hash


def test_unchanged_document_hash_does_not_rebuild_structure_or_chunks(tmp_path) -> None:
    database = MemoryDatabase()
    parser = StructuredCountingParser()
    processor = CountingStructureProcessor()
    chunker = CountingChunker()
    indexer = CountingIndexer()
    project_id = uuid4()
    (tmp_path / "paper.pdf").write_bytes(b"paper")
    pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=IdentityResolver(),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        structure_processor=processor,
        chunker=chunker,
        indexer=indexer,
    )

    first = pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))
    second = pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    assert first.items[0].stage == second.items[0].stage == PipelineStage.INDEXED
    assert (parser.calls, processor.calls, chunker.calls, indexer.calls) == (1, 1, 1, 1)


def test_schema_version_change_rebuilds_structure_and_chunks(tmp_path) -> None:
    database = MemoryDatabase()
    first_parser = StructuredCountingParser()
    first_processor = CountingStructureProcessor()
    first_chunker = CountingChunker()
    project_id = uuid4()
    (tmp_path / "paper.pdf").write_bytes(b"paper")
    first_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=first_parser,
        identity_resolver=IdentityResolver(),
        parser=first_parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        structure_processor=first_processor,
        chunker=first_chunker,
    )
    first_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    second_parser = StructuredSchemaTwoParser()
    second_processor = CountingStructureProcessor()
    second_chunker = CountingChunker()
    second_pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=second_parser,
        identity_resolver=IdentityResolver(),
        parser=second_parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        structure_processor=second_processor,
        chunker=second_chunker,
        canonical_schema_version=2,
    )

    report = second_pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    assert report.items[0].stage == PipelineStage.CHUNKED
    assert first_parser.calls + second_parser.calls == 2
    assert first_processor.calls + second_processor.calls == 2
    assert first_chunker.calls + second_chunker.calls == 2


def test_index_stage_is_incremental_and_recovers_without_reparse_or_rechunk(tmp_path) -> None:
    database = MemoryDatabase()
    parser = StructuredCountingParser()
    processor = CountingStructureProcessor()
    chunker = CountingChunker()
    indexer = CountingIndexer()
    project_id = uuid4()
    (tmp_path / "paper.pdf").write_bytes(b"paper")
    pipeline = IngestionPipeline(
        unit_of_work_factory=lambda: MemoryUnitOfWork(database),
        metadata_extractor=parser,
        identity_resolver=IdentityResolver(),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(tmp_path),
        structure_processor=processor,
        chunker=chunker,
        indexer=indexer,
    )

    first = pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))
    second = pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))
    indexer.current = False
    recovered = pipeline.ingest(IngestionRequest(project_id, tmp_path.resolve()))

    assert first.items[0].stage == second.items[0].stage == PipelineStage.INDEXED
    assert recovered.items[0].stage == PipelineStage.INDEXED
    assert (parser.calls, processor.calls, chunker.calls) == (1, 1, 1)
    assert indexer.calls == 2
    assert next(iter(database.files.values())).status == FileStatus.INDEXED
