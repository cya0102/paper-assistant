"""Project-scoped PostgreSQL Research Graph repository."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import exists, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.domain.enums import (
    ClaimPolarity,
    ClaimType,
    EntailmentStatus,
    EvidenceKind,
    EvidenceRelation,
    EvidenceTargetType,
    NormalizationStatus,
    ProfileField,
    RelationEndpointType,
    RelationType,
    ResearchEntityType,
    ReviewStatus,
)
from paper_agent.domain.research_graph import (
    Claim,
    EvidenceLink,
    ExtractionSource,
    GenerationProvenance,
    PaperProfile,
    PaperProfileExtractionRequest,
    PaperProfileFieldValue,
    PaperRelation,
    RelationEndpoint,
    ResearchEntity,
)
from paper_agent.storage.postgres.models import (
    ChunkRow,
    ClaimRow,
    DerivedDataStateRow,
    EvidenceLinkRow,
    PaperFileRow,
    PaperProfileFieldRow,
    PaperProfileRow,
    PaperRelationRow,
    PaperRow,
    ResearchEntityAliasRow,
    ResearchEntityRow,
)


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _provenance(row: Any) -> GenerationProvenance:
    return GenerationProvenance(
        extraction_method=row.extraction_method,
        extractor_version=row.extractor_version,
        schema_version=row.schema_version,
        model_name=row.model_name,
        prompt_version=row.prompt_version,
        source_document_hash=row.source_document_hash,
        chunking_version=row.chunking_version,
    )


def _derivation_values(provenance: GenerationProvenance) -> dict[str, object]:
    return {
        "extraction_method": provenance.extraction_method,
        "extractor_version": provenance.extractor_version,
        "schema_version": provenance.schema_version,
        "model_name": provenance.model_name,
        "prompt_version": provenance.prompt_version,
        "source_document_hash": provenance.source_document_hash,
        "chunking_version": provenance.chunking_version,
        "generation_key": provenance.generation_key,
    }


class SqlAlchemyResearchGraphRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def load_extraction_request(
        self,
        project_id: UUID,
        paper_id: UUID,
        version_id: UUID | None = None,
    ) -> PaperProfileExtractionRequest:
        with self._session_factory() as session:
            statement = select(PaperFileRow).where(
                PaperFileRow.project_id == project_id,
                PaperFileRow.paper_id == paper_id,
            )
            if version_id is not None:
                statement = statement.where(PaperFileRow.version_id == version_id)
            owner = session.scalar(
                statement.order_by(
                    PaperFileRow.is_canonical.desc(), PaperFileRow.updated_at.desc()
                ).limit(1)
            )
            if owner is None or owner.version_id is None:
                raise LookupError("Paper version not found in project")
            resolved_version_id = owner.version_id
            paper = session.get(PaperRow, paper_id)
            if paper is None:
                raise LookupError("Paper not found in project")
            chunks = tuple(
                session.scalars(
                    select(ChunkRow)
                    .where(
                        ChunkRow.paper_id == paper_id,
                        ChunkRow.version_id == resolved_version_id,
                    )
                    .order_by(ChunkRow.chunk_order)
                )
            )
            if not chunks:
                raise LookupError("Paper has no chunks available for profile extraction")
            state = session.get(DerivedDataStateRow, resolved_version_id)
            sources = tuple(
                ExtractionSource(
                    project_id=project_id,
                    paper_id=paper_id,
                    version_id=resolved_version_id,
                    section_id=chunk.section_id,
                    chunk_id=chunk.chunk_id,
                    section_path=chunk.section_path,
                    text=chunk.text,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    source_block_ids=tuple(chunk.source_block_ids_json),
                    element_ids=tuple(
                        UUID(value) for value in chunk.related_element_ids_json
                    ),
                )
                for chunk in chunks
            )
            return PaperProfileExtractionRequest(
                project_id=project_id,
                paper_id=paper_id,
                version_id=resolved_version_id,
                paper_title=paper.canonical_title
                or paper.short_name
                or str(paper.paper_id),
                sources=sources,
                source_document_hash=state.document_hash if state is not None else None,
                chunking_version=chunks[0].chunking_version,
            )

    def get_paper_titles(
        self, project_id: UUID, paper_ids: tuple[UUID, ...]
    ) -> dict[UUID, str]:
        if not paper_ids:
            return {}
        with self._session_factory() as session:
            rows = session.scalars(
                select(PaperRow).where(
                    PaperRow.paper_id.in_(paper_ids),
                    exists(
                        select(PaperFileRow.file_id).where(
                            PaperFileRow.project_id == project_id,
                            PaperFileRow.paper_id == PaperRow.paper_id,
                        )
                    ),
                )
            )
            return {
                row.paper_id: row.canonical_title or row.short_name or str(row.paper_id)
                for row in rows
            }

    def save_profile(self, profile: PaperProfile) -> PaperProfile:
        with self._session_factory.begin() as session:
            self._ensure_owned(
                session, profile.project_id, profile.paper_id, profile.version_id
            )
            existing = session.get(PaperProfileRow, profile.profile_id)
            if existing is not None:
                return self._load_profiles(session, (existing,))[0]
            active = session.scalar(
                select(PaperProfileRow).where(
                    PaperProfileRow.project_id == profile.project_id,
                    PaperProfileRow.paper_id == profile.paper_id,
                    PaperProfileRow.version_id == profile.version_id,
                    PaperProfileRow.is_active.is_(True),
                )
            )
            if active is not None and active.generation_key == profile.provenance.generation_key:
                return self._load_profiles(session, (active,))[0]
            if active is not None:
                active.is_active = False
                session.flush()
            row = PaperProfileRow(
                profile_id=profile.profile_id,
                project_id=profile.project_id,
                paper_id=profile.paper_id,
                version_id=profile.version_id,
                additional_attributes_json=dict(profile.additional_attributes),
                is_active=True,
                superseded_by_profile_id=None,
                created_at=profile.created_at,
                updated_at=profile.updated_at,
                **_derivation_values(profile.provenance),
            )
            session.add(row)
            for value in profile.values:
                session.add(
                    PaperProfileFieldRow(
                        field_id=value.field_id,
                        profile_id=profile.profile_id,
                        field_name=value.field_name.value,
                        ordinal=value.ordinal,
                        value=value.value,
                        normalized_value=value.normalized_value,
                        confidence=value.confidence,
                        review_status=value.review_status.value,
                    )
                )
                self._insert_evidence(session, value.evidence_links)
            session.flush()
            if active is not None:
                active.superseded_by_profile_id = profile.profile_id
                session.flush()
            return self._load_profiles(session, (row,))[0]

    def get_profiles(
        self,
        project_id: UUID,
        paper_ids: tuple[UUID, ...],
        version_ids: tuple[UUID, ...] = (),
        *,
        active_only: bool = True,
    ) -> tuple[PaperProfile, ...]:
        if not paper_ids:
            return ()
        with self._session_factory() as session:
            statement = select(PaperProfileRow).where(
                PaperProfileRow.project_id == project_id,
                PaperProfileRow.paper_id.in_(paper_ids),
            )
            if version_ids:
                statement = statement.where(PaperProfileRow.version_id.in_(version_ids))
            if active_only:
                statement = statement.where(PaperProfileRow.is_active.is_(True))
            rows = tuple(
                session.scalars(statement.order_by(PaperProfileRow.updated_at.desc()))
            )
            return self._load_profiles(session, rows)

    def save_claims(self, claims: tuple[Claim, ...]) -> tuple[Claim, ...]:
        if not claims:
            return ()
        with self._session_factory.begin() as session:
            result_rows: list[ClaimRow] = []
            for claim in claims:
                self._ensure_owned(
                    session, claim.project_id, claim.paper_id, claim.version_id
                )
                existing = session.get(ClaimRow, claim.claim_id)
                if existing is not None:
                    result_rows.append(existing)
                    continue
                active = session.scalar(
                    select(ClaimRow).where(
                        ClaimRow.project_id == claim.project_id,
                        ClaimRow.claim_key == claim.claim_key,
                        ClaimRow.is_active.is_(True),
                    )
                )
                if active is not None and active.generation_key == claim.provenance.generation_key:
                    result_rows.append(active)
                    self._insert_evidence(
                        session,
                        tuple(
                            link.for_target(EvidenceTargetType.CLAIM, active.claim_id)
                            for link in claim.evidence_links
                        ),
                    )
                    continue
                if active is not None:
                    active.is_active = False
                    session.flush()
                row = ClaimRow(
                    claim_id=claim.claim_id,
                    project_id=claim.project_id,
                    paper_id=claim.paper_id,
                    version_id=claim.version_id,
                    claim_type=claim.claim_type.value,
                    statement=claim.statement,
                    normalized_statement=claim.normalized_statement,
                    polarity=claim.polarity.value,
                    confidence=claim.confidence,
                    claim_key=claim.claim_key,
                    review_status=claim.review_status.value,
                    entailment_status=claim.entailment_status.value,
                    is_active=True,
                    superseded_by_claim_id=None,
                    created_at=claim.created_at,
                    updated_at=claim.updated_at,
                    **_derivation_values(claim.provenance),
                )
                session.add(row)
                self._insert_evidence(session, claim.evidence_links)
                session.flush()
                if active is not None:
                    active.superseded_by_claim_id = claim.claim_id
                result_rows.append(row)
            session.flush()
            return self._load_claims(session, tuple(result_rows))

    def list_claims(
        self,
        project_id: UUID,
        paper_ids: tuple[UUID, ...] = (),
        claim_types: tuple[ClaimType, ...] = (),
        *,
        active_only: bool = True,
    ) -> tuple[Claim, ...]:
        with self._session_factory() as session:
            statement = select(ClaimRow).where(ClaimRow.project_id == project_id)
            if paper_ids:
                statement = statement.where(ClaimRow.paper_id.in_(paper_ids))
            if claim_types:
                statement = statement.where(
                    ClaimRow.claim_type.in_(tuple(value.value for value in claim_types))
                )
            if active_only:
                statement = statement.where(ClaimRow.is_active.is_(True))
            rows = tuple(session.scalars(statement.order_by(ClaimRow.created_at)))
            return self._load_claims(session, rows)

    def save_entities(
        self, entities: tuple[ResearchEntity, ...]
    ) -> tuple[ResearchEntity, ...]:
        if not entities:
            return ()
        with self._session_factory.begin() as session:
            result_rows: list[ResearchEntityRow] = []
            for entity in entities:
                for link in entity.evidence_links:
                    self._ensure_owned(
                        session, link.project_id, link.paper_id, link.version_id
                    )
                row = session.scalar(
                    select(ResearchEntityRow).where(
                        ResearchEntityRow.project_id == entity.project_id,
                        ResearchEntityRow.entity_type == entity.entity_type.value,
                        ResearchEntityRow.normalized_name == entity.normalized_name,
                    )
                )
                if row is None:
                    row = ResearchEntityRow(
                        entity_id=entity.entity_id,
                        project_id=entity.project_id,
                        canonical_name=entity.canonical_name,
                        normalized_name=entity.normalized_name,
                        entity_type=entity.entity_type.value,
                        description=entity.description,
                        normalization_status=entity.normalization_status.value,
                        attributes_json=dict(entity.attributes),
                        created_at=entity.created_at,
                        updated_at=entity.updated_at,
                        **_derivation_values(entity.provenance),
                    )
                    session.add(row)
                    session.flush()
                elif row.description is None and entity.description is not None:
                    row.description = entity.description
                aliases = tuple(dict.fromkeys((*entity.aliases, entity.canonical_name)))
                for alias in aliases:
                    normalized_alias = _normalized(alias)
                    session.execute(
                        insert(ResearchEntityAliasRow)
                        .values(
                            alias_id=uuid5(
                                NAMESPACE_URL,
                                f"research-entity-alias:{row.entity_id}:{normalized_alias}",
                            ),
                            entity_id=row.entity_id,
                            alias=alias,
                            normalized_alias=normalized_alias,
                        )
                        .on_conflict_do_nothing(
                            constraint="uq_research_entity_aliases_entity_alias"
                        )
                    )
                links = tuple(
                    link.for_target(EvidenceTargetType.ENTITY, row.entity_id)
                    for link in entity.evidence_links
                )
                self._insert_evidence(session, links)
                result_rows.append(row)
            session.flush()
            return self._load_entities(session, tuple(result_rows))

    def find_entities(
        self,
        project_id: UUID,
        query: str | None = None,
        entity_types: tuple[ResearchEntityType, ...] = (),
        entity_ids: tuple[UUID, ...] = (),
    ) -> tuple[ResearchEntity, ...]:
        with self._session_factory() as session:
            statement = select(ResearchEntityRow).where(
                ResearchEntityRow.project_id == project_id
            )
            if entity_types:
                statement = statement.where(
                    ResearchEntityRow.entity_type.in_(
                        tuple(value.value for value in entity_types)
                    )
                )
            if entity_ids:
                statement = statement.where(ResearchEntityRow.entity_id.in_(entity_ids))
            if query and query.strip():
                pattern = f"%{_normalized(query)}%"
                statement = statement.where(
                    or_(
                        ResearchEntityRow.normalized_name.ilike(pattern),
                        exists(
                            select(ResearchEntityAliasRow.alias_id).where(
                                ResearchEntityAliasRow.entity_id
                                == ResearchEntityRow.entity_id,
                                ResearchEntityAliasRow.normalized_alias.ilike(pattern),
                            )
                        ),
                    )
                )
            rows = tuple(
                session.scalars(statement.order_by(ResearchEntityRow.canonical_name))
            )
            return self._load_entities(session, rows)

    def save_relations(
        self, relations: tuple[PaperRelation, ...]
    ) -> tuple[PaperRelation, ...]:
        if not relations:
            return ()
        with self._session_factory.begin() as session:
            result_rows: list[PaperRelationRow] = []
            for relation in relations:
                self._ensure_endpoint(session, relation.project_id, relation.source)
                self._ensure_endpoint(session, relation.project_id, relation.target)
                for link in relation.evidence_links:
                    self._ensure_owned(
                        session, link.project_id, link.paper_id, link.version_id
                    )
                existing = session.get(PaperRelationRow, relation.relation_id)
                if existing is not None:
                    result_rows.append(existing)
                    continue
                active = session.scalar(
                    select(PaperRelationRow).where(
                        PaperRelationRow.project_id == relation.project_id,
                        PaperRelationRow.relation_key == relation.relation_key,
                        PaperRelationRow.is_active.is_(True),
                    )
                )
                if active is not None and active.generation_key == relation.provenance.generation_key:
                    self._insert_evidence(
                        session,
                        tuple(
                            link.for_target(
                                EvidenceTargetType.RELATION, active.relation_id
                            )
                            for link in relation.evidence_links
                        ),
                    )
                    result_rows.append(active)
                    continue
                if active is not None:
                    active.is_active = False
                    session.flush()
                row = PaperRelationRow(
                    relation_id=relation.relation_id,
                    project_id=relation.project_id,
                    source_type=relation.source.endpoint_type.value,
                    source_id=relation.source.endpoint_id,
                    target_type=relation.target.endpoint_type.value,
                    target_id=relation.target.endpoint_id,
                    relation_type=relation.relation_type.value,
                    relation_key=relation.relation_key,
                    description=relation.description,
                    confidence=relation.confidence,
                    review_status=relation.review_status.value,
                    is_active=True,
                    superseded_by_relation_id=None,
                    created_at=relation.created_at,
                    updated_at=relation.updated_at,
                    **_derivation_values(relation.provenance),
                )
                session.add(row)
                self._insert_evidence(session, relation.evidence_links)
                session.flush()
                if active is not None:
                    active.superseded_by_relation_id = relation.relation_id
                result_rows.append(row)
            session.flush()
            return self._load_relations(session, tuple(result_rows))

    def query_relations(
        self,
        project_id: UUID,
        *,
        paper_id: UUID | None = None,
        entity_id: UUID | None = None,
        relation_types: tuple[RelationType, ...] = (),
        active_only: bool = True,
    ) -> tuple[PaperRelation, ...]:
        with self._session_factory() as session:
            statement = select(PaperRelationRow).where(
                PaperRelationRow.project_id == project_id
            )
            if paper_id is not None:
                statement = statement.where(
                    or_(
                        (
                            PaperRelationRow.source_type
                            == RelationEndpointType.PAPER.value
                        )
                        & (PaperRelationRow.source_id == paper_id),
                        (
                            PaperRelationRow.target_type
                            == RelationEndpointType.PAPER.value
                        )
                        & (PaperRelationRow.target_id == paper_id),
                    )
                )
            if entity_id is not None:
                statement = statement.where(
                    or_(
                        (
                            PaperRelationRow.source_type
                            == RelationEndpointType.ENTITY.value
                        )
                        & (PaperRelationRow.source_id == entity_id),
                        (
                            PaperRelationRow.target_type
                            == RelationEndpointType.ENTITY.value
                        )
                        & (PaperRelationRow.target_id == entity_id),
                    )
                )
            if relation_types:
                statement = statement.where(
                    PaperRelationRow.relation_type.in_(
                        tuple(value.value for value in relation_types)
                    )
                )
            if active_only:
                statement = statement.where(PaperRelationRow.is_active.is_(True))
            rows = tuple(
                session.scalars(statement.order_by(PaperRelationRow.created_at))
            )
            return self._load_relations(session, rows)

    @staticmethod
    def _ensure_owned(
        session: Session, project_id: UUID, paper_id: UUID, version_id: UUID
    ) -> None:
        owned = session.scalar(
            select(PaperFileRow.file_id).where(
                PaperFileRow.project_id == project_id,
                PaperFileRow.paper_id == paper_id,
                PaperFileRow.version_id == version_id,
            )
        )
        if owned is None:
            raise LookupError("Paper version not found in project")

    @staticmethod
    def _ensure_endpoint(
        session: Session, project_id: UUID, endpoint: RelationEndpoint
    ) -> None:
        if endpoint.endpoint_type == RelationEndpointType.PAPER:
            found = session.scalar(
                select(PaperFileRow.file_id).where(
                    PaperFileRow.project_id == project_id,
                    PaperFileRow.paper_id == endpoint.endpoint_id,
                )
            )
        else:
            found = session.scalar(
                select(ResearchEntityRow.entity_id).where(
                    ResearchEntityRow.project_id == project_id,
                    ResearchEntityRow.entity_id == endpoint.endpoint_id,
                )
            )
        if found is None:
            raise LookupError("Relation endpoint not found in project")

    @staticmethod
    def _insert_evidence(
        session: Session, evidence_links: Iterable[EvidenceLink]
    ) -> None:
        for link in evidence_links:
            session.execute(
                insert(EvidenceLinkRow)
                .values(
                    evidence_id=link.evidence_id,
                    project_id=link.project_id,
                    target_type=link.target_type.value,
                    target_id=link.target_id,
                    paper_id=link.paper_id,
                    version_id=link.version_id,
                    section_id=link.section_id,
                    chunk_id=link.chunk_id,
                    element_id=link.element_id,
                    page_start=link.page_start,
                    page_end=link.page_end,
                    source_block_ids_json=list(link.source_block_ids),
                    evidence_text=link.evidence_text,
                    relation_to_target=link.relation_to_target.value,
                    evidence_kind=link.evidence_kind.value,
                    confidence=link.confidence,
                    evidence_key=link.evidence_key,
                )
                .on_conflict_do_nothing(
                    constraint="uq_evidence_links_target_evidence"
                )
            )

    @staticmethod
    def _load_evidence(
        session: Session,
        target_type: EvidenceTargetType,
        target_ids: tuple[UUID, ...],
    ) -> dict[UUID, tuple[EvidenceLink, ...]]:
        if not target_ids:
            return {}
        rows = tuple(
            session.scalars(
                select(EvidenceLinkRow)
                .where(
                    EvidenceLinkRow.target_type == target_type.value,
                    EvidenceLinkRow.target_id.in_(target_ids),
                )
                .order_by(EvidenceLinkRow.created_at, EvidenceLinkRow.evidence_id)
            )
        )
        grouped: dict[UUID, list[EvidenceLink]] = defaultdict(list)
        for row in rows:
            grouped[row.target_id].append(
                EvidenceLink(
                    evidence_id=row.evidence_id,
                    project_id=row.project_id,
                    target_type=EvidenceTargetType(row.target_type),
                    target_id=row.target_id,
                    paper_id=row.paper_id,
                    version_id=row.version_id,
                    section_id=row.section_id,
                    chunk_id=row.chunk_id,
                    element_id=row.element_id,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    source_block_ids=tuple(row.source_block_ids_json),
                    evidence_text=row.evidence_text,
                    relation_to_target=EvidenceRelation(row.relation_to_target),
                    evidence_kind=EvidenceKind(row.evidence_kind),
                    confidence=row.confidence,
                )
            )
        return {key: tuple(value) for key, value in grouped.items()}

    def _load_profiles(
        self, session: Session, rows: tuple[PaperProfileRow, ...]
    ) -> tuple[PaperProfile, ...]:
        if not rows:
            return ()
        profile_ids = tuple(row.profile_id for row in rows)
        field_rows = tuple(
            session.scalars(
                select(PaperProfileFieldRow)
                .where(PaperProfileFieldRow.profile_id.in_(profile_ids))
                .order_by(
                    PaperProfileFieldRow.profile_id,
                    PaperProfileFieldRow.field_name,
                    PaperProfileFieldRow.ordinal,
                )
            )
        )
        evidence = self._load_evidence(
            session,
            EvidenceTargetType.PROFILE_FIELD,
            tuple(row.field_id for row in field_rows),
        )
        fields: dict[UUID, list[PaperProfileFieldValue]] = defaultdict(list)
        for row in field_rows:
            fields[row.profile_id].append(
                PaperProfileFieldValue(
                    field_id=row.field_id,
                    field_name=ProfileField(row.field_name),
                    value=row.value,
                    normalized_value=row.normalized_value,
                    ordinal=row.ordinal,
                    confidence=row.confidence,
                    review_status=ReviewStatus(row.review_status),
                    evidence_links=evidence.get(row.field_id, ()),
                )
            )
        return tuple(
            PaperProfile(
                profile_id=row.profile_id,
                project_id=row.project_id,
                paper_id=row.paper_id,
                version_id=row.version_id,
                values=tuple(fields.get(row.profile_id, [])),
                provenance=_provenance(row),
                additional_attributes=row.additional_attributes_json,
                is_active=row.is_active,
                superseded_by_profile_id=row.superseded_by_profile_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        )

    def _load_claims(
        self, session: Session, rows: tuple[ClaimRow, ...]
    ) -> tuple[Claim, ...]:
        evidence = self._load_evidence(
            session, EvidenceTargetType.CLAIM, tuple(row.claim_id for row in rows)
        )
        return tuple(
            Claim(
                claim_id=row.claim_id,
                project_id=row.project_id,
                paper_id=row.paper_id,
                version_id=row.version_id,
                claim_type=ClaimType(row.claim_type),
                statement=row.statement,
                normalized_statement=row.normalized_statement,
                polarity=ClaimPolarity(row.polarity),
                confidence=row.confidence,
                provenance=_provenance(row),
                evidence_links=evidence.get(row.claim_id, ()),
                review_status=ReviewStatus(row.review_status),
                entailment_status=EntailmentStatus(row.entailment_status),
                is_active=row.is_active,
                superseded_by_claim_id=row.superseded_by_claim_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        )

    def _load_entities(
        self, session: Session, rows: tuple[ResearchEntityRow, ...]
    ) -> tuple[ResearchEntity, ...]:
        if not rows:
            return ()
        entity_ids = tuple(row.entity_id for row in rows)
        normalized_names = {row.entity_id: row.normalized_name for row in rows}
        aliases: dict[UUID, list[str]] = defaultdict(list)
        for alias in session.scalars(
            select(ResearchEntityAliasRow)
            .where(ResearchEntityAliasRow.entity_id.in_(entity_ids))
            .order_by(ResearchEntityAliasRow.created_at)
        ):
            if _normalized(alias.alias) != normalized_names[alias.entity_id]:
                aliases[alias.entity_id].append(alias.alias)
        evidence = self._load_evidence(
            session, EvidenceTargetType.ENTITY, entity_ids
        )
        return tuple(
            ResearchEntity(
                entity_id=row.entity_id,
                project_id=row.project_id,
                canonical_name=row.canonical_name,
                aliases=tuple(aliases.get(row.entity_id, [])),
                entity_type=ResearchEntityType(row.entity_type),
                description=row.description,
                normalization_status=NormalizationStatus(row.normalization_status),
                provenance=_provenance(row),
                evidence_links=evidence.get(row.entity_id, ()),
                attributes=row.attributes_json,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        )

    def _load_relations(
        self, session: Session, rows: tuple[PaperRelationRow, ...]
    ) -> tuple[PaperRelation, ...]:
        evidence = self._load_evidence(
            session,
            EvidenceTargetType.RELATION,
            tuple(row.relation_id for row in rows),
        )
        return tuple(
            PaperRelation(
                relation_id=row.relation_id,
                project_id=row.project_id,
                source=RelationEndpoint(
                    RelationEndpointType(row.source_type), row.source_id
                ),
                target=RelationEndpoint(
                    RelationEndpointType(row.target_type), row.target_id
                ),
                relation_type=RelationType(row.relation_type),
                description=row.description,
                confidence=row.confidence,
                provenance=_provenance(row),
                evidence_links=evidence.get(row.relation_id, ()),
                review_status=ReviewStatus(row.review_status),
                is_active=row.is_active,
                superseded_by_relation_id=row.superseded_by_relation_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        )
