from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

from paper_agent.domain.enums import IndexLevel, IndexingStatus
from paper_agent.domain.indexing import IndexDocument, IndexingState
from paper_agent.indexing.hashing_provider import HashingEmbeddingProvider
from paper_agent.indexing.service import HierarchicalIndexingService


def _document(level, target_id, project_id, paper_id, version_id, text, section_id=None):
    return IndexDocument(
        target_id=target_id,
        level=level,
        project_id=project_id,
        paper_id=paper_id,
        version_id=version_id,
        section_id=section_id,
        text=text,
        content_hash=sha256(f"{level.value}\n{text}".encode()).hexdigest(),
    )


class CountingProvider(HashingEmbeddingProvider):
    def __init__(self):
        super().__init__()
        self.batches = []

    def embed_batch(self, texts):
        self.batches.append(texts)
        return super().embed_batch(texts)


class MemoryIndexRepository:
    def __init__(self, documents):
        self.documents = documents
        self.state = None
        self.vectors = ()

    def load_documents(self, project_id, version_id):
        return tuple(
            item for item in self.documents
            if item.project_id == project_id and item.version_id == version_id
        )

    def get_state(self, project_id, version_id):
        del project_id, version_id
        return self.state

    def load_reusable_vectors(self, project_id, version_id, embedding_version):
        del project_id, version_id
        return {
            item.document.content_hash: item.embedding
            for item in self.vectors
            if self.state and self.state.embedding_version == embedding_version
        }

    def replace_index(self, project_id, version_id, descriptor, index_version, source_digest, vectors):
        self.vectors = vectors
        self.state = IndexingState(
            project_id=project_id,
            version_id=version_id,
            embedding_version=descriptor.identifier,
            index_version=index_version,
            source_digest=source_digest,
            status=IndexingStatus.INDEXED,
        )


def test_hashing_provider_is_deterministic_normalized_and_batched() -> None:
    provider = HashingEmbeddingProvider()
    first = provider.embed_batch(("codebook learning", "temporal localization"))
    second = provider.embed_batch(("codebook learning", "temporal localization"))

    assert first == second
    assert len(first) == 2
    assert all(len(vector) == 256 for vector in first)
    assert all(abs(sum(value * value for value in vector) - 1.0) < 1e-9 for vector in first)


def test_hierarchical_indexing_batches_reuses_and_rebuilds_only_changed_text() -> None:
    project_id, paper_id, version_id, section_id, chunk_id = (
        uuid4(), uuid4(), uuid4(), uuid4(), uuid4()
    )
    documents = (
        _document(IndexLevel.PAPER, paper_id, project_id, paper_id, version_id, "Paper codebook"),
        _document(IndexLevel.SECTION, section_id, project_id, paper_id, version_id, "Method", section_id),
        _document(IndexLevel.CHUNK, chunk_id, project_id, paper_id, version_id, "Build a codebook", section_id),
    )
    repository = MemoryIndexRepository(documents)
    provider = CountingProvider()
    service = HierarchicalIndexingService(repository, provider, batch_size=2)

    first = service.index_version(project_id, version_id)
    second = service.index_version(project_id, version_id)
    changed = replace(
        documents[-1],
        text="Build a hierarchical codebook",
        content_hash=sha256(b"changed").hexdigest(),
    )
    repository.documents = (*documents[:-1], changed)
    third = service.index_version(project_id, version_id)

    assert (first.generated, first.reused, first.unchanged) == (3, 0, False)
    assert (second.generated, second.reused, second.unchanged) == (0, 3, True)
    assert (third.generated, third.reused, third.unchanged) == (1, 2, False)
    assert [len(batch) for batch in provider.batches] == [2, 1, 1]
    assert service.is_current(project_id, version_id)
