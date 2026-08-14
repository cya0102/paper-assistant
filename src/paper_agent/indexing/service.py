"""Incremental, batch-oriented hierarchical indexing service."""

from hashlib import sha256
from uuid import UUID

from paper_agent.domain.enums import IndexLevel
from paper_agent.domain.errors import ErrorCode, PaperAgentError
from paper_agent.domain.indexing import IndexDocument, IndexedVector, IndexingReport
from paper_agent.indexing.ports import EmbeddingProvider, IndexRepository


class HierarchicalIndexingService:
    version = "hierarchical-index-v1"

    def __init__(
        self,
        repository: IndexRepository,
        provider: EmbeddingProvider,
        *,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Embedding batch_size must be positive")
        self._repository = repository
        self._provider = provider
        self._batch_size = batch_size

    @property
    def embedding_version(self) -> str:
        return self._provider.descriptor.identifier

    def is_current(self, project_id: UUID, version_id: UUID) -> bool:
        try:
            documents = self._repository.load_documents(project_id, version_id)
            state = self._repository.get_state(project_id, version_id)
        except Exception as error:
            raise PaperAgentError(ErrorCode.INDEX_FAILED, str(error)) from error
        if not documents:
            return False
        return bool(
            state
            and state.embedding_version == self.embedding_version
            and state.index_version == self.version
            and state.source_digest == self._source_digest(documents)
            and state.status.value == "indexed"
        )

    def index_version(
        self, project_id: UUID, version_id: UUID, *, force: bool = False
    ) -> IndexingReport:
        try:
            documents = self._repository.load_documents(project_id, version_id)
            state = self._repository.get_state(project_id, version_id)
        except Exception as error:
            raise PaperAgentError(ErrorCode.INDEX_FAILED, str(error)) from error
        if not documents:
            raise LookupError(f"No chunked index sources exist for version {version_id}")
        source_digest = self._source_digest(documents)
        if (
            not force
            and state
            and state.embedding_version == self.embedding_version
            and state.index_version == self.version
            and state.source_digest == source_digest
            and state.status.value == "indexed"
        ):
            return self._report(project_id, version_id, documents, 0, len(documents), True)

        reusable = (
            {}
            if force
            else self._repository.load_reusable_vectors(
                project_id, version_id, self.embedding_version
            )
        )
        embeddings: dict[str, tuple[float, ...]] = {}
        pending = [document for document in documents if document.content_hash not in reusable]
        for start in range(0, len(pending), self._batch_size):
            batch = pending[start : start + self._batch_size]
            try:
                batch_embeddings = self._provider.embed_batch(tuple(item.text for item in batch))
            except Exception as error:
                raise PaperAgentError(ErrorCode.EMBEDDING_FAILED, str(error)) from error
            if len(batch_embeddings) != len(batch):
                raise ValueError("Embedding provider returned a different batch size")
            for document, embedding in zip(batch, batch_embeddings, strict=True):
                if len(embedding) != self._provider.descriptor.dimension:
                    raise ValueError("Embedding provider returned an invalid dimension")
                embeddings[document.content_hash] = embedding

        vectors = tuple(
            IndexedVector(
                document=document,
                embedding=embeddings.get(document.content_hash)
                or reusable[document.content_hash],
            )
            for document in documents
        )
        try:
            self._repository.replace_index(
                project_id,
                version_id,
                self._provider.descriptor,
                self.version,
                source_digest,
                vectors,
            )
        except Exception as error:
            raise PaperAgentError(ErrorCode.INDEX_FAILED, str(error)) from error
        return self._report(
            project_id,
            version_id,
            documents,
            len(pending),
            len(documents) - len(pending),
            False,
        )

    def _report(
        self,
        project_id: UUID,
        version_id: UUID,
        documents: tuple[IndexDocument, ...],
        generated: int,
        reused: int,
        unchanged: bool,
    ) -> IndexingReport:
        levels = [document.level for document in documents]
        return IndexingReport(
            project_id=project_id,
            version_id=version_id,
            embedding_version=self.embedding_version,
            index_version=self.version,
            papers=levels.count(IndexLevel.PAPER),
            sections=levels.count(IndexLevel.SECTION),
            chunks=levels.count(IndexLevel.CHUNK),
            generated=generated,
            reused=reused,
            unchanged=unchanged,
        )

    @staticmethod
    def _source_digest(documents: tuple[IndexDocument, ...]) -> str:
        payload = "\n".join(
            sorted(
                f"{item.level.value}:{item.target_id}:{item.content_hash}"
                for item in documents
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()
