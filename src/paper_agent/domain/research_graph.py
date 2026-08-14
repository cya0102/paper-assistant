"""Evidence-first, versioned Research Graph domain models."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

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


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be blank")


def _validate_confidence(value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError("confidence must be between 0 and 1")


def _validate_sha256(value: str | None, field_name: str) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class GenerationProvenance:
    extraction_method: str
    extractor_version: str
    schema_version: str
    model_name: str | None = None
    prompt_version: str | None = None
    source_document_hash: str | None = None
    chunking_version: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.extraction_method, "extraction_method")
        _require_text(self.extractor_version, "extractor_version")
        _require_text(self.schema_version, "schema_version")
        _validate_sha256(self.source_document_hash, "source_document_hash")
        if self.model_name is not None:
            _require_text(self.model_name, "model_name")
        if self.prompt_version is not None:
            _require_text(self.prompt_version, "prompt_version")

    @property
    def generation_key(self) -> str:
        payload = "\n".join(
            (
                self.extraction_method,
                self.extractor_version,
                self.schema_version,
                self.model_name or "",
                self.prompt_version or "",
                self.source_document_hash or "",
                self.chunking_version or "",
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    project_id: UUID
    target_type: EvidenceTargetType
    target_id: UUID
    paper_id: UUID
    version_id: UUID
    section_id: UUID
    chunk_id: UUID
    page_start: int
    page_end: int
    source_block_ids: tuple[str, ...]
    evidence_text: str
    relation_to_target: EvidenceRelation
    confidence: float
    evidence_kind: EvidenceKind = EvidenceKind.PAPER_FACT
    element_id: UUID | None = None
    evidence_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("Invalid EvidenceLink page range")
        if not self.source_block_ids:
            raise ValueError("EvidenceLink must retain source_block_ids")
        if len(self.source_block_ids) != len(set(self.source_block_ids)):
            raise ValueError("EvidenceLink source_block_ids must be unique")
        _require_text(self.evidence_text, "evidence_text")
        _validate_confidence(self.confidence)

    @property
    def evidence_key(self) -> str:
        payload = "\n".join(
            (
                str(self.paper_id),
                str(self.version_id),
                str(self.section_id),
                str(self.chunk_id),
                str(self.element_id or ""),
                str(self.page_start),
                str(self.page_end),
                "\x1f".join(self.source_block_ids),
                self.evidence_text,
                self.relation_to_target.value,
                self.evidence_kind.value,
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def for_target(
        self,
        target_type: EvidenceTargetType,
        target_id: UUID,
        *,
        relation_to_target: EvidenceRelation | None = None,
    ) -> "EvidenceLink":
        relation = relation_to_target or self.relation_to_target
        evidence_id = uuid5(
            NAMESPACE_URL,
            f"research-evidence:{target_type.value}:{target_id}:{self.evidence_key}:{relation.value}",
        )
        return replace(
            self,
            evidence_id=evidence_id,
            target_type=target_type,
            target_id=target_id,
            relation_to_target=relation,
        )


@dataclass(frozen=True, slots=True)
class PaperProfileFieldValue:
    field_name: ProfileField
    value: str
    normalized_value: str
    ordinal: int
    confidence: float
    evidence_links: tuple[EvidenceLink, ...]
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    field_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        _require_text(self.value, "profile field value")
        _require_text(self.normalized_value, "normalized profile field value")
        if self.ordinal < 0:
            raise ValueError("profile field ordinal cannot be negative")
        _validate_confidence(self.confidence)
        if not self.evidence_links:
            raise ValueError("PaperProfile field values require EvidenceLink")
        for link in self.evidence_links:
            if link.target_type != EvidenceTargetType.PROFILE_FIELD or link.target_id != self.field_id:
                raise ValueError("Profile evidence target must match field_id")


@dataclass(frozen=True, slots=True)
class PaperProfile:
    project_id: UUID
    paper_id: UUID
    version_id: UUID
    values: tuple[PaperProfileFieldValue, ...]
    provenance: GenerationProvenance
    additional_attributes: Mapping[str, object] = field(default_factory=dict)
    profile_id: UUID = field(default_factory=uuid4)
    is_active: bool = True
    superseded_by_profile_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        field_ids = {value.field_id for value in self.values}
        if len(field_ids) != len(self.values):
            raise ValueError("PaperProfile field IDs must be unique")
        positions = {(value.field_name, value.ordinal) for value in self.values}
        if len(positions) != len(self.values):
            raise ValueError("PaperProfile field ordinals must be unique per field")
        for value in self.values:
            for link in value.evidence_links:
                if (
                    link.project_id != self.project_id
                    or link.paper_id != self.paper_id
                    or link.version_id != self.version_id
                ):
                    raise ValueError("Profile evidence identity must match PaperProfile")
        if self.is_active and self.superseded_by_profile_id is not None:
            raise ValueError("An active PaperProfile cannot be superseded")

    def field_values(self, field_name: ProfileField) -> tuple[PaperProfileFieldValue, ...]:
        return tuple(
            sorted(
                (value for value in self.values if value.field_name == field_name),
                key=lambda value: value.ordinal,
            )
        )


@dataclass(frozen=True, slots=True)
class Claim:
    project_id: UUID
    paper_id: UUID
    version_id: UUID
    claim_type: ClaimType
    statement: str
    normalized_statement: str
    polarity: ClaimPolarity
    confidence: float
    provenance: GenerationProvenance
    evidence_links: tuple[EvidenceLink, ...]
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    entailment_status: EntailmentStatus = EntailmentStatus.UNREVIEWED
    claim_id: UUID = field(default_factory=uuid4)
    is_active: bool = True
    superseded_by_claim_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.statement, "claim statement")
        _require_text(self.normalized_statement, "normalized claim statement")
        _validate_confidence(self.confidence)
        if not self.evidence_links:
            raise ValueError("Claim requires EvidenceLink")
        for link in self.evidence_links:
            if link.target_type != EvidenceTargetType.CLAIM or link.target_id != self.claim_id:
                raise ValueError("Claim evidence target must match claim_id")
            if (
                link.project_id != self.project_id
                or link.paper_id != self.paper_id
                or link.version_id != self.version_id
            ):
                raise ValueError("Claim evidence identity must match Claim")
        if self.review_status == ReviewStatus.VERIFIED and self.entailment_status != EntailmentStatus.SUPPORTED:
            raise ValueError("Only supported claims may be verified")
        if self.is_active and self.superseded_by_claim_id is not None:
            raise ValueError("An active Claim cannot be superseded")

    @property
    def claim_key(self) -> str:
        payload = f"{self.paper_id}\n{self.version_id}\n{self.claim_type.value}\n{self.normalized_statement}"
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResearchEntity:
    project_id: UUID
    canonical_name: str
    aliases: tuple[str, ...]
    entity_type: ResearchEntityType
    description: str | None
    normalization_status: NormalizationStatus
    provenance: GenerationProvenance
    evidence_links: tuple[EvidenceLink, ...]
    attributes: Mapping[str, object] = field(default_factory=dict)
    entity_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _require_text(self.canonical_name, "canonical_name")
        if len(self.normalized_name) > 512:
            raise ValueError("ResearchEntity canonical_name is too long")
        if self.description is not None:
            _require_text(self.description, "entity description")
        normalized_aliases = [alias.casefold().strip() for alias in self.aliases]
        if any(not alias for alias in normalized_aliases):
            raise ValueError("ResearchEntity aliases cannot be blank")
        if any(len(alias) > 512 for alias in normalized_aliases):
            raise ValueError("ResearchEntity alias is too long")
        if len(normalized_aliases) != len(set(normalized_aliases)):
            raise ValueError("ResearchEntity aliases must be unique")
        if not self.evidence_links:
            raise ValueError("ResearchEntity requires EvidenceLink")
        for link in self.evidence_links:
            if link.project_id != self.project_id:
                raise ValueError("Entity evidence must remain in its project")
            if link.target_type != EvidenceTargetType.ENTITY or link.target_id != self.entity_id:
                raise ValueError("Entity evidence target must match entity_id")

    @property
    def normalized_name(self) -> str:
        return " ".join(self.canonical_name.casefold().split())


@dataclass(frozen=True, slots=True)
class RelationEndpoint:
    endpoint_type: RelationEndpointType
    endpoint_id: UUID

    @property
    def key(self) -> str:
        return f"{self.endpoint_type.value}:{self.endpoint_id}"


UNDIRECTED_RELATION_TYPES = frozenset(
    {
        RelationType.SAME_PROBLEM,
        RelationType.DIFFERENT_ASSUMPTION,
        RelationType.ANALOGOUS_TO,
    }
)


@dataclass(frozen=True, slots=True)
class PaperRelation:
    project_id: UUID
    source: RelationEndpoint
    target: RelationEndpoint
    relation_type: RelationType
    description: str
    confidence: float
    provenance: GenerationProvenance
    evidence_links: tuple[EvidenceLink, ...]
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    relation_id: UUID = field(default_factory=uuid4)
    is_active: bool = True
    superseded_by_relation_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError("Relation endpoints must be different")
        _require_text(self.description, "relation description")
        _validate_confidence(self.confidence)
        if not self.evidence_links:
            raise ValueError("PaperRelation requires EvidenceLink")
        for link in self.evidence_links:
            if link.project_id != self.project_id:
                raise ValueError("Relation evidence must remain in its project")
            if link.target_type != EvidenceTargetType.RELATION or link.target_id != self.relation_id:
                raise ValueError("Relation evidence target must match relation_id")
        if self.is_active and self.superseded_by_relation_id is not None:
            raise ValueError("An active PaperRelation cannot be superseded")

    @property
    def relation_key(self) -> str:
        endpoints = (self.source.key, self.target.key)
        if self.relation_type in UNDIRECTED_RELATION_TYPES:
            endpoints = (min(endpoints), max(endpoints))
        payload = f"{self.relation_type.value}\n{endpoints[0]}\n{endpoints[1]}"
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ExtractionSource:
    project_id: UUID
    paper_id: UUID
    version_id: UUID
    section_id: UUID
    chunk_id: UUID
    section_path: str
    text: str
    page_start: int
    page_end: int
    source_block_ids: tuple[str, ...]
    element_ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        _require_text(self.section_path, "section_path")
        _require_text(self.text, "extraction source text")
        if self.page_start < 1 or self.page_end < self.page_start:
            raise ValueError("Invalid ExtractionSource page range")
        if not self.source_block_ids:
            raise ValueError("ExtractionSource requires source_block_ids")

    def evidence_for(
        self,
        target_type: EvidenceTargetType,
        target_id: UUID,
        *,
        confidence: float,
        relation_to_target: EvidenceRelation = EvidenceRelation.SUPPORTS,
        evidence_kind: EvidenceKind = EvidenceKind.PAPER_FACT,
        element_id: UUID | None = None,
    ) -> EvidenceLink:
        seed = (
            f"research-evidence:{target_type.value}:{target_id}:{self.chunk_id}:"
            f"{relation_to_target.value}:{element_id or ''}"
        )
        return EvidenceLink(
            evidence_id=uuid5(NAMESPACE_URL, seed),
            project_id=self.project_id,
            target_type=target_type,
            target_id=target_id,
            paper_id=self.paper_id,
            version_id=self.version_id,
            section_id=self.section_id,
            chunk_id=self.chunk_id,
            element_id=element_id,
            page_start=self.page_start,
            page_end=self.page_end,
            source_block_ids=self.source_block_ids,
            evidence_text=self.text,
            relation_to_target=relation_to_target,
            confidence=confidence,
            evidence_kind=evidence_kind,
        )


@dataclass(frozen=True, slots=True)
class PaperProfileExtractionRequest:
    project_id: UUID
    paper_id: UUID
    version_id: UUID
    paper_title: str
    sources: tuple[ExtractionSource, ...]
    source_document_hash: str | None
    chunking_version: str | None

    def __post_init__(self) -> None:
        _require_text(self.paper_title, "paper_title")
        _validate_sha256(self.source_document_hash, "source_document_hash")
        for source in self.sources:
            if (
                source.project_id != self.project_id
                or source.paper_id != self.paper_id
                or source.version_id != self.version_id
            ):
                raise ValueError("Extraction sources must match request identity")


@dataclass(frozen=True, slots=True)
class PaperProfileExtraction:
    profile: PaperProfile
    claims: tuple[Claim, ...] = ()
    entities: tuple[ResearchEntity, ...] = ()
    relations: tuple[PaperRelation, ...] = ()

    def __post_init__(self) -> None:
        project_id = self.profile.project_id
        if any(item.project_id != project_id for item in self.claims):
            raise ValueError("Extracted graph records must remain in one project")
        if any(item.project_id != project_id for item in self.entities):
            raise ValueError("Extracted graph records must remain in one project")
        if any(item.project_id != project_id for item in self.relations):
            raise ValueError("Extracted graph records must remain in one project")
