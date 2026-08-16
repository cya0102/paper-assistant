"""ArtifactService: materialize, hydrate slices, search, and validate.

The service owns the invariant that a Blob is written and verified before the
Catalog row commits, that reads re-verify content hashes, that every read is
project-scoped, and that expired/corrupt/cross-project artifacts fail with
stable errors.
"""

import json
from collections.abc import Sequence
from typing import Any
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid5

from paper_agent.artifacts.ports import ArtifactBlobStore, ArtifactRepository
from paper_agent.artifacts.tokens import count_tokens
from paper_agent.artifacts.views import extract_view
from paper_agent.domain.artifact import (
    ArtifactDescriptor,
    ArtifactSelector,
    ArtifactSlice,
    ArtifactStatus,
    ArtifactType,
    CitationReference,
)
from paper_agent.domain.errors import ErrorCode, PaperAgentError


class ArtifactAccessError(PaperAgentError):
    """Stable error for artifacts that cannot be read back."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(code, message)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _string_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        result: set[str] = set()
        for child in value.values():
            result.update(_string_values(child))
        return result
    if isinstance(value, (list, tuple)):
        result = set()
        for child in value:
            result.update(_string_values(child))
        return result
    return set()


def _content_contains_citation(content: Any, citation_label: str) -> bool:
    for value in _string_values(content):
        if value == citation_label:
            return True
        # Lossless JSON-fragment views carry a serialized subsection as a
        # string. Match only a complete quoted JSON value, not an arbitrary
        # substring such as E1 inside E10.
        if json.dumps(citation_label, ensure_ascii=False) in value:
            return True
    return False


class ArtifactService:
    def __init__(
        self,
        blob_store: ArtifactBlobStore,
        repository: ArtifactRepository,
        *,
        retention_days: int = 30,
    ) -> None:
        self._blob_store = blob_store
        self._repository = repository
        self._retention_days = retention_days

    def materialize(
        self,
        *,
        project_id: UUID,
        artifact_type: ArtifactType,
        schema_version: str,
        media_type: str,
        payload: dict[str, Any],
        summary: str,
        citation_manifest: Sequence[CitationReference] = (),
        created_by: str = "system",
        session_id: UUID | None = None,
        research_task_id: UUID | None = None,
        work_unit_id: UUID | None = None,
        tool_call_id: str | None = None,
        token_estimate: int | None = None,
        expires_at: datetime | None = None,
    ) -> ArtifactDescriptor:
        canonical = canonical_json(payload)
        content_hash = sha256(canonical.encode("utf-8")).hexdigest()
        byte_size = len(canonical.encode("utf-8"))
        if token_estimate is None:
            token_estimate = count_tokens(canonical)
        if expires_at is None:
            expires_at = datetime.now(UTC) + timedelta(days=self._retention_days)
        identity_context = ":".join(
            str(value) if value is not None else "-"
            for value in (
                session_id,
                research_task_id,
                work_unit_id,
                tool_call_id,
            )
        )
        artifact_id = uuid5(
            NAMESPACE_URL,
            (
                f"artifact:{project_id}:{artifact_type.value}:{schema_version}:"
                f"{content_hash}:{identity_context}"
            ),
        )
        existing = self._repository.get(project_id, artifact_id)
        if existing is not None:
            now = datetime.now(UTC)
            active = existing.status == ArtifactStatus.ACTIVE and (
                existing.expires_at is None or existing.expires_at >= now
            )
            if active:
                try:
                    stored = self._blob_store.get(storage_key=existing.storage_key)
                except PaperAgentError:
                    stored = b""
                if sha256(stored).hexdigest() == existing.content_hash:
                    return existing
        storage_key = self._blob_store.put(
            content_hash=content_hash, data=canonical.encode("utf-8")
        )
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            project_id=project_id,
            artifact_type=artifact_type,
            schema_version=schema_version,
            media_type=media_type,
            content_hash=content_hash,
            storage_backend="local_content_addressed",
            storage_key=storage_key,
            byte_size=byte_size,
            token_estimate=token_estimate,
            summary=summary,
            citation_manifest=tuple(citation_manifest),
            status=ArtifactStatus.ACTIVE,
            created_by=created_by,
            created_at=datetime.now(UTC),
            session_id=session_id,
            research_task_id=research_task_id,
            work_unit_id=work_unit_id,
            tool_call_id=tool_call_id,
            expires_at=expires_at,
        )
        return self._repository.save(descriptor, citation_manifest)

    def read_slice(self, selector: ArtifactSelector) -> ArtifactSlice:
        descriptor = self._repository.get(selector.project_id, selector.artifact_id)
        if descriptor is None:
            raise ArtifactAccessError(
                ErrorCode.ARTIFACT_NOT_FOUND, "Artifact not found in project"
            )
        if descriptor.project_id != selector.project_id:
            # Defense in depth: every read must remain inside its project.
            raise ArtifactAccessError(
                ErrorCode.ARTIFACT_CROSS_PROJECT,
                "Artifact belongs to another project",
            )
        now = datetime.now(UTC)
        if descriptor.status == ArtifactStatus.EXPIRED or (
            descriptor.expires_at is not None and descriptor.expires_at < now
        ):
            self._repository.mark_expired(now=now)
            raise ArtifactAccessError(
                ErrorCode.ARTIFACT_EXPIRED, "Artifact has expired"
            )
        try:
            data = self._blob_store.get(storage_key=descriptor.storage_key)
        except PaperAgentError as error:
            if error.code == ErrorCode.FILE_NOT_FOUND:
                raise ArtifactAccessError(
                    ErrorCode.ARTIFACT_NOT_FOUND, "Artifact blob is missing"
                ) from error
            self._repository.update_status(
                selector.project_id,
                selector.artifact_id,
                ArtifactStatus.CORRUPT,
            )
            raise ArtifactAccessError(
                ErrorCode.ARTIFACT_CORRUPT, "Artifact blob is corrupt"
            ) from error
        if sha256(data).hexdigest() != descriptor.content_hash:
            self._repository.update_status(
                selector.project_id,
                selector.artifact_id,
                ArtifactStatus.CORRUPT,
            )
            raise ArtifactAccessError(
                ErrorCode.ARTIFACT_CORRUPT, "Artifact blob hash mismatch"
            )
        payload = json.loads(data.decode("utf-8"))
        content, next_cursor, truncated, token_count = extract_view(
            payload,
            descriptor.artifact_type,
            selector.view,
            selector.cursor,
            selector.max_tokens,
        )
        citations = tuple(
            citation
            for citation in descriptor.citation_manifest
            if _content_contains_citation(content, citation.citation_label)
        )
        return ArtifactSlice(
            artifact_id=descriptor.artifact_id,
            project_id=descriptor.project_id,
            view=selector.view,
            content=content,
            citations=citations,
            next_cursor=next_cursor,
            truncated=truncated,
            token_count=token_count,
        )

    def search(
        self,
        project_id: UUID,
        *,
        artifact_type: ArtifactType | None = None,
        created_by: str | None = None,
        research_task_id: UUID | None = None,
        work_unit_id: UUID | None = None,
        query: str | None = None,
        limit: int = 20,
    ) -> tuple[ArtifactDescriptor, ...]:
        self._repository.mark_expired(now=datetime.now(UTC))
        return self._repository.search(
            project_id,
            artifact_type=artifact_type,
            created_by=created_by,
            research_task_id=research_task_id,
            work_unit_id=work_unit_id,
            query=query,
            limit=limit,
        )

    def validate_hash(self, *, content_hash: str, data: bytes) -> bool:
        return sha256(data).hexdigest() == content_hash
