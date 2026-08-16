"""Materialize exactly one retrieved Evidence chunk per Artifact."""

from hashlib import sha256
from uuid import UUID

from paper_agent.artifacts.ports import ArtifactServicePort
from paper_agent.domain.artifact import (
    ArtifactReference,
    ArtifactType,
    CitationReference,
)
from paper_agent.domain.retrieval import Evidence
from paper_agent.rag.domain import RetrievedEvidenceArtifact


def evidence_citation_label(evidence: Evidence) -> str:
    return f"E{int(evidence.evidence_id.hex[:12], 16)}"


class EvidenceArtifactMaterializer:
    schema_version = "retrieved-evidence-v1"

    def __init__(self, artifacts: ArtifactServicePort) -> None:
        self._artifacts = artifacts

    def materialize(
        self,
        *,
        project_id: UUID,
        session_id: UUID,
        task_id: UUID,
        query: str,
        round_index: int,
        evidence: Evidence,
    ) -> RetrievedEvidenceArtifact:
        label = evidence_citation_label(evidence)
        citation = CitationReference(
            citation_label=label,
            paper_id=evidence.paper_id,
            version_id=evidence.version_id,
            paper_title=evidence.paper_title,
            section_path=evidence.section_path,
            page_start=evidence.page_start,
            page_end=evidence.page_end,
            evidence_hash=sha256(evidence.text.encode("utf-8")).hexdigest(),
            section_id=evidence.section_id,
            chunk_id=evidence.chunk_id,
        )
        payload = {
            "query": query,
            "citation": label,
            "paper_id": str(evidence.paper_id),
            "version_id": str(evidence.version_id),
            "paper_title": evidence.paper_title,
            "section_id": str(evidence.section_id),
            "section_path": evidence.section_path,
            "page_start": evidence.page_start,
            "page_end": evidence.page_end,
            "chunk_id": str(evidence.chunk_id),
            "element_ids": [str(value) for value in evidence.element_ids],
            "text": evidence.text,
            "retrieval_scores": {
                "dense": evidence.dense_score,
                "bm25": evidence.bm25_score,
                "rerank": evidence.rerank_score,
                "relevance": evidence.relevance,
            },
        }
        descriptor = self._artifacts.materialize(
            project_id=project_id,
            session_id=session_id,
            research_task_id=task_id,
            tool_call_id=(
                f"rag:{task_id}:round:{round_index}:chunk:{evidence.chunk_id}"
            ),
            artifact_type=ArtifactType.RETRIEVED_EVIDENCE,
            schema_version=self.schema_version,
            media_type="application/json",
            payload=payload,
            summary=(
                f"[{label}] {evidence.paper_title} | {evidence.section_path} | "
                f"pp.{evidence.page_start}-{evidence.page_end}"
            ),
            citation_manifest=(citation,),
            created_by="rag_retriever",
        )
        return RetrievedEvidenceArtifact(
            artifact_ref=ArtifactReference.from_descriptor(
                descriptor, available_views=("default", "full")
            ),
            citation=citation,
            paper_id=evidence.paper_id,
            chunk_id=evidence.chunk_id,
            relevance=evidence.relevance,
            round_index=round_index,
        )
