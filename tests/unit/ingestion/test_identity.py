from uuid import uuid4

from paper_agent.domain.enums import IdentityMatchType
from paper_agent.domain.ingestion import DiscoveredFile
from paper_agent.domain.metadata import PaperMetadata
from paper_agent.domain.paper import Paper, PaperFile, PaperIdentityResolution, PaperVersion
from paper_agent.ingestion.identity import DeterministicPaperIdentityResolver


class IdentityLookup:
    def __init__(self, existing=None) -> None:
        self.existing = existing

    def find_by_content_hash(self, project_id, content_hash):
        del project_id
        return self.existing if self.existing and self.existing.version.content_hash == content_hash else None

    def find_by_doi(self, doi):
        return self.existing if self.existing and self.existing.paper.doi == doi else None

    def find_by_arxiv_id(self, arxiv_id):
        return None

    def find_by_title_authors(self, normalized_title, normalized_authors):
        return None


def inputs(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"pdf")
    project_id = uuid4()
    return (
        DiscoveredFile(path, path.name, 3, path.stat().st_mtime_ns),
        PaperFile(uuid4(), project_id, 3, "a" * 64),
    )


def test_new_metadata_creates_paper_and_version(tmp_path) -> None:
    discovered, paper_file = inputs(tmp_path)
    metadata = PaperMetadata(
        title="Paper Agents",
        normalized_title="paper agents",
        authors=("Alice Smith",),
        normalized_authors=("alice smith",),
        content_hash="b" * 64,
    )

    result = DeterministicPaperIdentityResolver(parser_version="1").resolve(
        discovered, paper_file, metadata, IdentityLookup()
    )

    assert result.match_type == IdentityMatchType.NEW_PAPER
    assert result.version.paper_id == result.paper.paper_id
    assert result.version.content_hash == "b" * 64


def test_content_hash_reuses_existing_version(tmp_path) -> None:
    discovered, paper_file = inputs(tmp_path)
    paper = Paper(paper_id=uuid4(), canonical_title="Existing")
    version = PaperVersion(
        version_id=uuid4(), paper_id=paper.paper_id, content_hash="b" * 64
    )
    existing = PaperIdentityResolution(paper=paper, version=version)

    result = DeterministicPaperIdentityResolver(parser_version="1").resolve(
        discovered,
        paper_file,
        PaperMetadata(content_hash="b" * 64),
        IdentityLookup(existing),
    )

    assert result.match_type == IdentityMatchType.CONTENT_HASH
    assert result.version.version_id == version.version_id


def test_doi_match_creates_new_version_under_existing_paper(tmp_path) -> None:
    discovered, paper_file = inputs(tmp_path)
    paper = Paper(paper_id=uuid4(), canonical_title="Existing", doi="10.1234/paper")
    old_version = PaperVersion(version_id=uuid4(), paper_id=paper.paper_id, content_hash="b" * 64)
    existing = PaperIdentityResolution(paper=paper, version=old_version)

    result = DeterministicPaperIdentityResolver(parser_version="1").resolve(
        discovered,
        paper_file,
        PaperMetadata(doi="10.1234/paper", content_hash="c" * 64),
        IdentityLookup(existing),
    )

    assert result.match_type == IdentityMatchType.DOI
    assert result.paper.paper_id == paper.paper_id
    assert result.version.version_id != old_version.version_id

