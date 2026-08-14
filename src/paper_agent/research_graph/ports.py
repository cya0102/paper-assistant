"""Dependency boundaries for Research Graph extraction and persistence."""

from typing import Protocol
from uuid import UUID

from paper_agent.domain.enums import ClaimType, EntailmentStatus, RelationType, ResearchEntityType
from paper_agent.domain.research_graph import (
    Claim,
    PaperProfile,
    PaperProfileExtraction,
    PaperProfileExtractionRequest,
    PaperRelation,
    ResearchEntity,
)


class PaperProfileExtractor(Protocol):
    version: str
    schema_version: str

    def extract(self, request: PaperProfileExtractionRequest) -> PaperProfileExtraction: ...


class EntailmentJudge(Protocol):
    version: str

    def judge(
        self, statement: str, evidence_texts: tuple[str, ...]
    ) -> EntailmentStatus: ...


class ResearchGraphRepository(Protocol):
    def load_extraction_request(
        self,
        project_id: UUID,
        paper_id: UUID,
        version_id: UUID | None = None,
    ) -> PaperProfileExtractionRequest: ...

    def get_paper_titles(
        self, project_id: UUID, paper_ids: tuple[UUID, ...]
    ) -> dict[UUID, str]: ...

    def save_profile(self, profile: PaperProfile) -> PaperProfile: ...

    def get_profiles(
        self,
        project_id: UUID,
        paper_ids: tuple[UUID, ...],
        version_ids: tuple[UUID, ...] = (),
        *,
        active_only: bool = True,
    ) -> tuple[PaperProfile, ...]: ...

    def save_claims(self, claims: tuple[Claim, ...]) -> tuple[Claim, ...]: ...

    def list_claims(
        self,
        project_id: UUID,
        paper_ids: tuple[UUID, ...] = (),
        claim_types: tuple[ClaimType, ...] = (),
        *,
        active_only: bool = True,
    ) -> tuple[Claim, ...]: ...

    def save_entities(
        self, entities: tuple[ResearchEntity, ...]
    ) -> tuple[ResearchEntity, ...]: ...

    def find_entities(
        self,
        project_id: UUID,
        query: str | None = None,
        entity_types: tuple[ResearchEntityType, ...] = (),
        entity_ids: tuple[UUID, ...] = (),
    ) -> tuple[ResearchEntity, ...]: ...

    def save_relations(
        self, relations: tuple[PaperRelation, ...]
    ) -> tuple[PaperRelation, ...]: ...

    def query_relations(
        self,
        project_id: UUID,
        *,
        paper_id: UUID | None = None,
        entity_id: UUID | None = None,
        relation_types: tuple[RelationType, ...] = (),
        active_only: bool = True,
    ) -> tuple[PaperRelation, ...]: ...
