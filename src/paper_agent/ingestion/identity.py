"""Conservative deterministic Paper and Version identity resolution."""

from uuid import uuid4

from paper_agent.domain.enums import IdentityMatchType, PipelineStage, SourceType
from paper_agent.domain.ingestion import DiscoveredFile
from paper_agent.domain.metadata import PaperMetadata
from paper_agent.domain.paper import Paper, PaperFile, PaperIdentityResolution, PaperVersion
from paper_agent.ingestion.ports import PaperIdentityLookup


class DeterministicPaperIdentityResolver:
    def __init__(self, *, parser_version: str) -> None:
        self._parser_version = parser_version

    def resolve(
        self,
        discovered_file: DiscoveredFile,
        paper_file: PaperFile,
        metadata: PaperMetadata,
        lookup: PaperIdentityLookup,
    ) -> PaperIdentityResolution:
        del discovered_file
        if metadata.content_hash:
            existing = lookup.find_by_content_hash(paper_file.project_id, metadata.content_hash)
            if existing is not None:
                return PaperIdentityResolution(
                    paper=existing.paper,
                    version=existing.version,
                    aliases=existing.aliases,
                    match_type=IdentityMatchType.CONTENT_HASH,
                    confidence=1.0,
                )

        match_type: IdentityMatchType | None = None
        existing = None
        if metadata.doi:
            existing = lookup.find_by_doi(metadata.doi)
            match_type = IdentityMatchType.DOI if existing else None
        if existing is None and metadata.arxiv_id:
            existing = lookup.find_by_arxiv_id(metadata.arxiv_id)
            match_type = IdentityMatchType.ARXIV if existing else None
        if (
            existing is None
            and metadata.normalized_title
            and metadata.normalized_authors
        ):
            existing = lookup.find_by_title_authors(
                metadata.normalized_title, metadata.normalized_authors
            )
            match_type = IdentityMatchType.TITLE_AUTHORS if existing else None

        if existing is None:
            paper = Paper(
                paper_id=uuid4(),
                canonical_title=metadata.title,
                normalized_title=metadata.normalized_title,
                authors=metadata.authors,
                normalized_authors=metadata.normalized_authors,
                doi=metadata.doi,
                arxiv_id=metadata.arxiv_id,
                year=metadata.year,
                venue=metadata.venue,
            )
            match_type = IdentityMatchType.NEW_PAPER
        else:
            paper = existing.paper

        source_type, source_identifier = self._source(metadata)
        version = PaperVersion(
            version_id=uuid4(),
            paper_id=paper.paper_id,
            source_type=source_type,
            source_identifier=source_identifier,
            parser_version=self._parser_version,
            content_hash=metadata.content_hash,
            pipeline_stage=PipelineStage.IDENTITY_RESOLVED,
        )
        return PaperIdentityResolution(
            paper=paper,
            version=version,
            match_type=match_type or IdentityMatchType.NEW_PAPER,
            confidence=0.98 if match_type in (IdentityMatchType.DOI, IdentityMatchType.ARXIV) else 0.9,
        )

    @staticmethod
    def _source(metadata: PaperMetadata) -> tuple[SourceType, str | None]:
        if metadata.arxiv_id:
            return SourceType.ARXIV, metadata.arxiv_id
        if metadata.doi:
            return SourceType.DOI, metadata.doi
        return SourceType.LOCAL, None

