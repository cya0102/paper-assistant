"""Idempotent, stage-aware, per-file isolated ingestion orchestration."""

from dataclasses import replace
from pathlib import PurePosixPath
from uuid import UUID, uuid4

from paper_agent.domain.document import CanonicalParsedDocument
from paper_agent.domain.chunk import Chunk, SemanticGroup
from paper_agent.domain.enums import (
    FileStatus,
    IngestionDisposition,
    PipelineStage,
    RunStatus,
)
from paper_agent.domain.errors import ErrorCode, PaperAgentError
from paper_agent.domain.ingestion import (
    DiscoveredFile,
    IngestionItemResult,
    IngestionReport,
    IngestionRequest,
)
from paper_agent.domain.paper import FileLocation, PaperFile
from paper_agent.domain.project import Project
from paper_agent.domain.structure import StructuredDocument
from paper_agent.ingestion.dedup import classify_file
from paper_agent.ingestion.fingerprint import Sha256Fingerprinter
from paper_agent.ingestion.ports import (
    DocumentChunker,
    IngestionUnitOfWork,
    PaperIdentityResolver,
    ParseRequest,
    ParsedDocumentStore,
    PdfParser,
    PdfMetadataExtractor,
    StructureProcessor,
    UnitOfWorkFactory,
)
from paper_agent.ingestion.scanner import DirectoryScanner
from paper_agent.indexing.ports import VersionIndexer


class IngestionPipeline:
    def __init__(
        self,
        *,
        unit_of_work_factory: UnitOfWorkFactory,
        scanner: DirectoryScanner | None = None,
        fingerprinter: Sha256Fingerprinter | None = None,
        metadata_extractor: PdfMetadataExtractor | None = None,
        identity_resolver: PaperIdentityResolver | None = None,
        parser: PdfParser | None = None,
        parsed_document_store: ParsedDocumentStore | None = None,
        structure_processor: StructureProcessor | None = None,
        chunker: DocumentChunker | None = None,
        indexer: VersionIndexer | None = None,
        canonical_schema_version: int = 1,
    ) -> None:
        if (parser is None) != (parsed_document_store is None):
            raise ValueError("parser and parsed_document_store must be configured together")
        if (structure_processor is None) != (chunker is None):
            raise ValueError("structure_processor and chunker must be configured together")
        if structure_processor is not None and parser is None:
            raise ValueError("structure derivation requires a configured parser")
        if indexer is not None and chunker is None:
            raise ValueError("indexing requires a configured chunker")
        self._unit_of_work_factory = unit_of_work_factory
        self._scanner = scanner or DirectoryScanner()
        self._fingerprinter = fingerprinter or Sha256Fingerprinter()
        self._metadata_extractor = metadata_extractor
        self._identity_resolver = identity_resolver
        self._parser = parser
        self._parsed_document_store = parsed_document_store
        self._structure_processor = structure_processor
        self._chunker = chunker
        self._indexer = indexer
        if canonical_schema_version < 1:
            raise ValueError("canonical_schema_version must be positive")
        self._canonical_schema_version = canonical_schema_version

    def ingest(self, request: IngestionRequest) -> IngestionReport:
        run_id = uuid4()
        with self._unit_of_work_factory() as unit_of_work:
            unit_of_work.projects.ensure(
                Project(
                    project_id=request.project_id,
                    name=request.project_root.name or "paper-agent-project",
                    root_path=request.project_root,
                )
            )
            unit_of_work.runs.create(run_id, request)
            unit_of_work.commit()

        scan = self._scanner.scan(request.project_root, request.paths, recursive=request.recursive)
        results: list[IngestionItemResult] = []
        for discovered_file in scan.files:
            result = self._ingest_file(run_id, request, discovered_file)
            results.append(result)
            with self._unit_of_work_factory() as unit_of_work:
                unit_of_work.runs.record_item(run_id, result)
                unit_of_work.commit()

        missing_paths: tuple[PurePosixPath, ...] = ()
        if not request.paths:
            with self._unit_of_work_factory() as unit_of_work:
                missing_paths = unit_of_work.files.mark_missing_locations(
                    request.project_id,
                    {file.relative_path for file in scan.files},
                )
                unit_of_work.commit()
            for missing_path in missing_paths:
                item = IngestionItemResult(
                    relative_path=missing_path,
                    disposition=IngestionDisposition.MISSING,
                    stage=PipelineStage.DISCOVERED,
                )
                results.append(item)
                with self._unit_of_work_factory() as unit_of_work:
                    unit_of_work.runs.record_item(run_id, item)
                    unit_of_work.commit()

        report = IngestionReport(
            run_id=run_id,
            scanned=len(scan.files),
            items=tuple(results),
            scan_issues=scan.issues,
            missing=len(missing_paths),
        )
        run_status = (
            RunStatus.COMPLETED_WITH_ERRORS
            if report.counts[IngestionDisposition.FAILED] or scan.issues
            else RunStatus.COMPLETED
        )
        counters = {disposition.value: count for disposition, count in report.counts.items()}
        counters["scanned"] = report.scanned
        counters["scan_issues"] = len(scan.issues)
        with self._unit_of_work_factory() as unit_of_work:
            unit_of_work.runs.complete(run_id, run_status, counters)
            unit_of_work.commit()
        return report

    def _ingest_file(
        self,
        run_id: UUID,
        request: IngestionRequest,
        discovered_file: DiscoveredFile,
    ) -> IngestionItemResult:
        del run_id  # Item persistence is performed after this isolated operation completes.
        active_stage = PipelineStage.DISCOVERED
        paper_file: PaperFile | None = None
        disposition = IngestionDisposition.NEW
        active_operation = "fingerprint"

        try:
            fingerprint = self._fingerprinter.fingerprint(discovered_file.absolute_path)
            with self._unit_of_work_factory() as unit_of_work:
                location_entry = unit_of_work.files.get_location(
                    request.project_id, discovered_file.relative_path
                )
                current_location, current_file = location_entry or (None, None)
                matching_hash_file = unit_of_work.files.find_by_hash(
                    request.project_id, fingerprint.sha256
                )
                decision = classify_file(
                    current_location=current_location,
                    current_file=current_file,
                    matching_hash_file=matching_hash_file,
                )
                disposition = decision.disposition

                if decision.reusable_file_id is not None:
                    if matching_hash_file is None:
                        raise RuntimeError("Deduplication selected a missing reusable file")
                    paper_file = matching_hash_file
                else:
                    candidate_file = PaperFile(
                        file_id=uuid4(),
                        project_id=request.project_id,
                        file_size=fingerprint.file_size,
                        file_hash=fingerprint.sha256,
                    )
                    paper_file, created = unit_of_work.files.get_or_add_file(candidate_file)
                    if not created:
                        disposition = IngestionDisposition.DUPLICATE

                location = FileLocation(
                    location_id=current_location.location_id if current_location else uuid4(),
                    project_id=request.project_id,
                    file_id=paper_file.file_id,
                    relative_path=discovered_file.relative_path,
                    file_name=discovered_file.relative_path.name,
                    mtime_ns=fingerprint.mtime_ns,
                )
                unit_of_work.files.upsert_location(location)
                unit_of_work.commit()

            resolution = None
            with self._unit_of_work_factory() as unit_of_work:
                resolution = unit_of_work.files.get_identity(paper_file.file_id)
                has_current_document = bool(
                    resolution
                    and self._parser
                    and unit_of_work.documents.has_current(
                        resolution.version.version_id,
                        self._parser.name,
                        self._parser.version,
                        self._canonical_schema_version,
                    )
                )
                target_is_current = bool(
                    resolution
                    and self._target_is_current(
                        unit_of_work,
                        request.project_id,
                        resolution.version.version_id,
                        has_current_document,
                    )
                )

            if (
                disposition in (IngestionDisposition.UNCHANGED, IngestionDisposition.DUPLICATE)
                and not request.force_reindex
                and resolution is not None
                and target_is_current
            ):
                target_status = (
                    FileStatus.INDEXED
                    if self._indexer is not None
                    else FileStatus.CHUNKED
                    if self._chunker is not None
                    else FileStatus.PARSED
                )
                with self._unit_of_work_factory() as unit_of_work:
                    unit_of_work.files.update_status(paper_file.file_id, target_status)
                    unit_of_work.commit()
                return self._success_result(
                    discovered_file.relative_path,
                    disposition,
                    replace(paper_file, status=target_status),
                    stage=self._target_stage,
                )

            should_refresh_metadata = (
                resolution is None
                or request.force_reindex
                or paper_file.content_hash is None
            )
            if should_refresh_metadata:
                if self._metadata_extractor is None:
                    return self._success_result(
                        discovered_file.relative_path, disposition, paper_file
                    )
                active_operation = "metadata"
                metadata = self._metadata_extractor.extract(discovered_file.absolute_path)
                with self._unit_of_work_factory() as unit_of_work:
                    paper_file = unit_of_work.files.save_metadata(paper_file.file_id, metadata)
                    if resolution is None:
                        if self._identity_resolver is None:
                            unit_of_work.commit()
                            return self._success_result(
                                discovered_file.relative_path, disposition, paper_file
                            )
                        resolution = self._identity_resolver.resolve(
                            discovered_file,
                            paper_file,
                            metadata,
                            unit_of_work.files,
                        )
                    unit_of_work.commit()

                if resolution.match_type.value == "content_hash":
                    disposition = IngestionDisposition.DUPLICATE

            if resolution is None:
                return self._success_result(discovered_file.relative_path, disposition, paper_file)

            if self._parser is not None and resolution.version.parser_version != self._parser.version:
                resolution = replace(
                    resolution,
                    version=replace(resolution.version, parser_version=self._parser.version),
                )

            active_stage = PipelineStage.IDENTITY_RESOLVED
            active_operation = "identity"
            with self._unit_of_work_factory() as unit_of_work:
                paper_file = unit_of_work.files.save_identity(paper_file.file_id, resolution)
                unit_of_work.files.update_status(paper_file.file_id, FileStatus.IDENTITY_RESOLVED)
                unit_of_work.commit()

            if self._parser is None or self._parsed_document_store is None:
                return self._success_result(
                    discovered_file.relative_path,
                    disposition,
                    paper_file,
                    stage=PipelineStage.IDENTITY_RESOLVED,
                )

            has_current_document = False
            with self._unit_of_work_factory() as unit_of_work:
                has_current_document = unit_of_work.documents.has_current(
                    resolution.version.version_id,
                    self._parser.name,
                    self._parser.version,
                    self._canonical_schema_version,
                )
                if has_current_document and not request.force_reindex:
                    unit_of_work.files.update_status(paper_file.file_id, FileStatus.PARSED)
                    unit_of_work.commit()
            if has_current_document and not request.force_reindex:
                active_stage = PipelineStage.PARSED
                active_operation = "load_parsed"
                document = self._parsed_document_store.load(
                    resolution.paper.paper_id, resolution.version.version_id
                )
            else:
                active_stage = PipelineStage.PARSING
                active_operation = "parse"
                with self._unit_of_work_factory() as unit_of_work:
                    unit_of_work.files.update_status(paper_file.file_id, FileStatus.PARSING)
                    unit_of_work.commit()
                document = self._parser.parse(
                    ParseRequest(
                        source_path=discovered_file.absolute_path,
                        source_file=paper_file,
                        identity=resolution,
                    )
                )
                self._validate_parser_result(document, paper_file)
                if document.schema_version != self._canonical_schema_version:
                    raise ValueError(
                        "Parser result schema_version does not match the configured canonical schema"
                    )
                artifacts = self._parsed_document_store.save(document)

                active_stage = PipelineStage.PARSED
                with self._unit_of_work_factory() as unit_of_work:
                    unit_of_work.documents.add(document, artifacts)
                    unit_of_work.files.update_status(paper_file.file_id, FileStatus.PARSED)
                    unit_of_work.commit()

            if self._structure_processor is None or self._chunker is None:
                return self._success_result(
                    discovered_file.relative_path,
                    disposition,
                    replace(paper_file, status=FileStatus.PARSED),
                    stage=PipelineStage.PARSED,
                )

            active_stage = PipelineStage.STRUCTURED
            active_operation = "structure"
            with self._unit_of_work_factory() as unit_of_work:
                current_hash = unit_of_work.documents.current_document_hash(
                    resolution.version.version_id,
                    self._parser.name,
                    self._parser.version,
                    self._canonical_schema_version,
                )
                derived_state = unit_of_work.derived.get_state(resolution.version.version_id)
                structure_is_current = bool(
                    current_hash is not None
                    and derived_state
                    and derived_state.structure_version == self._structure_processor.version
                    and derived_state.document_hash == current_hash
                )
                chunks_are_current = bool(
                    structure_is_current
                    and derived_state
                    and derived_state.chunking_version == self._chunker.version
                )
                if structure_is_current and not request.force_reindex:
                    structured = unit_of_work.derived.load_structure(
                        resolution.version.version_id
                    )
                    groups = unit_of_work.derived.load_groups(resolution.version.version_id)
                else:
                    structured, groups = self._structure_processor.build(document)
                    self._validate_structure_result(
                        document,
                        structured,
                        groups,
                        self._structure_processor.version,
                    )
                    unit_of_work.derived.replace_structure(
                        structured,
                        groups,
                        document_hash=current_hash,
                    )
                    chunks_are_current = False
                unit_of_work.files.update_status(paper_file.file_id, FileStatus.STRUCTURED)
                unit_of_work.commit()

            if chunks_are_current and not request.force_reindex:
                with self._unit_of_work_factory() as unit_of_work:
                    unit_of_work.files.update_status(paper_file.file_id, FileStatus.CHUNKED)
                    unit_of_work.commit()
                if self._indexer is None:
                    return self._success_result(
                        discovered_file.relative_path,
                        disposition,
                        replace(paper_file, status=FileStatus.CHUNKED),
                        stage=PipelineStage.CHUNKED,
                    )
            else:
                active_stage = PipelineStage.CHUNKED
                active_operation = "chunk"
                chunks = self._chunker.chunk(structured, groups)
                self._validate_chunk_results(
                    chunks,
                    resolution.paper.paper_id,
                    resolution.version.version_id,
                    self._chunker.version,
                )
                with self._unit_of_work_factory() as unit_of_work:
                    unit_of_work.derived.replace_chunks(
                        resolution.version.version_id, chunks, self._chunker.version
                    )
                    unit_of_work.files.update_status(paper_file.file_id, FileStatus.CHUNKED)
                    unit_of_work.commit()

            if self._indexer is not None:
                active_stage = PipelineStage.EMBEDDED
                active_operation = "embedding"
                self._indexer.index_version(
                    request.project_id,
                    resolution.version.version_id,
                    force=request.force_reindex,
                )
                with self._unit_of_work_factory() as unit_of_work:
                    unit_of_work.files.update_status(paper_file.file_id, FileStatus.INDEXED)
                    unit_of_work.commit()
                return self._success_result(
                    discovered_file.relative_path,
                    disposition,
                    replace(paper_file, status=FileStatus.INDEXED),
                    stage=PipelineStage.INDEXED,
                )
            return self._success_result(
                discovered_file.relative_path,
                disposition,
                replace(paper_file, status=FileStatus.CHUNKED),
                stage=PipelineStage.CHUNKED,
            )
        except Exception as error:
            error_code = self._error_code(error, active_stage, active_operation)
            if paper_file is not None:
                try:
                    with self._unit_of_work_factory() as unit_of_work:
                        unit_of_work.files.update_status(paper_file.file_id, FileStatus.FAILED)
                        unit_of_work.commit()
                except Exception:
                    pass
            return IngestionItemResult(
                relative_path=discovered_file.relative_path,
                disposition=IngestionDisposition.FAILED,
                stage=PipelineStage.FAILED,
                file_id=paper_file.file_id if paper_file else None,
                paper_id=paper_file.paper_id if paper_file else None,
                version_id=paper_file.version_id if paper_file else None,
                error_code=error_code,
                error_message=str(error),
            )

    @staticmethod
    def _success_result(
        relative_path: PurePosixPath,
        disposition: IngestionDisposition,
        paper_file: PaperFile,
        *,
        stage: PipelineStage | None = None,
    ) -> IngestionItemResult:
        if stage is None:
            if paper_file.status == FileStatus.PARSED:
                stage = PipelineStage.PARSED
            elif paper_file.status == FileStatus.IDENTITY_RESOLVED:
                stage = PipelineStage.IDENTITY_RESOLVED
            else:
                stage = PipelineStage.DISCOVERED
        return IngestionItemResult(
            relative_path=relative_path,
            disposition=disposition,
            stage=stage,
            file_id=paper_file.file_id,
            paper_id=paper_file.paper_id,
            version_id=paper_file.version_id,
        )

    @staticmethod
    def _validate_parser_result(
        document: CanonicalParsedDocument, paper_file: PaperFile
    ) -> None:
        if document.paper_id != paper_file.paper_id or document.version_id != paper_file.version_id:
            raise ValueError("Parser result identity does not match the resolved PaperVersion")
        if document.source_file_id != paper_file.file_id:
            raise ValueError("Parser result source_file_id does not match the ingested file")

    @staticmethod
    def _error_code(
        error: Exception, stage: PipelineStage, operation: str
    ) -> ErrorCode:
        if isinstance(error, PaperAgentError):
            return error.code
        if operation == "metadata":
            return ErrorCode.METADATA_FAILED
        if operation == "load_parsed":
            return ErrorCode.STORAGE_FAILED
        if operation == "structure":
            return ErrorCode.STRUCTURE_FAILED
        if operation == "chunk":
            return ErrorCode.CHUNK_FAILED
        if operation == "embedding":
            return ErrorCode.EMBEDDING_FAILED
        if stage == PipelineStage.PARSING:
            return ErrorCode.PARSE_FAILED
        if stage == PipelineStage.IDENTITY_RESOLVED:
            return ErrorCode.IDENTITY_FAILED
        return ErrorCode.FINGERPRINT_FAILED

    @property
    def _target_stage(self) -> PipelineStage:
        if self._indexer is not None:
            return PipelineStage.INDEXED
        return PipelineStage.CHUNKED if self._chunker is not None else PipelineStage.PARSED

    def _target_is_current(
        self,
        unit_of_work: IngestionUnitOfWork,
        project_id: UUID,
        version_id: UUID,
        parsed_is_current: bool,
    ) -> bool:
        if not parsed_is_current:
            return False
        if self._parser is None:
            return False
        if self._structure_processor is None or self._chunker is None:
            return True
        current_hash = unit_of_work.documents.current_document_hash(
            version_id,
            self._parser.name,
            self._parser.version,
            self._canonical_schema_version,
        )
        state = unit_of_work.derived.get_state(version_id)
        derived_is_current = bool(
            current_hash is not None
            and state
            and state.structure_version == self._structure_processor.version
            and state.chunking_version == self._chunker.version
            and state.document_hash == current_hash
        )
        if not derived_is_current:
            return False
        return self._indexer is None or self._indexer.is_current(project_id, version_id)

    @staticmethod
    def _validate_structure_result(
        document: CanonicalParsedDocument,
        structured: StructuredDocument,
        groups: tuple[SemanticGroup, ...],
        expected_version: str,
    ) -> None:
        if (
            structured.paper_id != document.paper_id
            or structured.version_id != document.version_id
            or structured.structure_version != expected_version
        ):
            raise ValueError("Structure result identity/version does not match its input")
        if any(
            group.paper_id != document.paper_id
            or group.version_id != document.version_id
            or group.structure_version != expected_version
            for group in groups
        ):
            raise ValueError("Semantic Group identity/version does not match its structure")

    @staticmethod
    def _validate_chunk_results(
        chunks: tuple[Chunk, ...],
        paper_id: UUID,
        version_id: UUID,
        expected_version: str,
    ) -> None:
        if any(
            chunk.paper_id != paper_id
            or chunk.version_id != version_id
            or chunk.chunking_version != expected_version
            for chunk in chunks
        ):
            raise ValueError("Chunk identity/version does not match its input")
