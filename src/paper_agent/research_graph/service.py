"""Research Graph extraction and evidence-backed comparison services."""

from collections import defaultdict
from dataclasses import replace
from uuid import UUID

from paper_agent.domain.comparison import (
    ComparisonCell,
    ComparisonDimension,
    PaperComparisonResult,
)
from paper_agent.domain.enums import (
    ClaimType,
    ComparisonCellStatus,
    ComparisonDimensionName,
    ComparisonStatus,
    ProfileField,
    RelationEndpointType,
)
from paper_agent.domain.research_graph import (
    Claim,
    EvidenceLink,
    PaperProfile,
    PaperProfileExtraction,
    RelationEndpoint,
)
from paper_agent.research_graph.entailment import ClaimVerificationService
from paper_agent.research_graph.ports import (
    EntailmentJudge,
    PaperProfileExtractor,
    ResearchGraphRepository,
)


class ResearchGraphService:
    def __init__(
        self,
        repository: ResearchGraphRepository,
        extractor: PaperProfileExtractor,
        entailment_judge: EntailmentJudge,
    ) -> None:
        self._repository = repository
        self._extractor = extractor
        self._verifier = ClaimVerificationService(entailment_judge)

    def extract_profile(
        self,
        project_id: UUID,
        paper_id: UUID,
        version_id: UUID | None = None,
    ) -> PaperProfileExtraction:
        request = self._repository.load_extraction_request(
            project_id, paper_id, version_id
        )
        extracted = self._extractor.extract(request)
        verified_claims = tuple(self._verifier.verify(claim) for claim in extracted.claims)
        profile = self._repository.save_profile(extracted.profile)
        claims = self._repository.save_claims(verified_claims)
        entities = self._repository.save_entities(extracted.entities)
        entity_ids = {
            source.entity_id: stored.entity_id
            for source, stored in zip(extracted.entities, entities, strict=True)
        }
        remapped_relations = tuple(
            replace(
                relation,
                source=self._remap_endpoint(relation.source, entity_ids),
                target=self._remap_endpoint(relation.target, entity_ids),
            )
            for relation in extracted.relations
        )
        relations = self._repository.save_relations(remapped_relations)
        return PaperProfileExtraction(
            profile=profile,
            claims=claims,
            entities=entities,
            relations=relations,
        )

    @staticmethod
    def _remap_endpoint(
        endpoint: RelationEndpoint, entity_ids: dict[UUID, UUID]
    ) -> RelationEndpoint:
        if endpoint.endpoint_type != RelationEndpointType.ENTITY:
            return endpoint
        return RelationEndpoint(
            endpoint.endpoint_type,
            entity_ids.get(endpoint.endpoint_id, endpoint.endpoint_id),
        )


DIMENSION_FIELDS: dict[ComparisonDimensionName, tuple[ProfileField, ...]] = {
    ComparisonDimensionName.RESEARCH_PROBLEM: (ProfileField.RESEARCH_PROBLEM,),
    ComparisonDimensionName.ASSUMPTIONS: (ProfileField.ASSUMPTIONS,),
    ComparisonDimensionName.METHOD: (
        ProfileField.METHOD_NAME,
        ProfileField.METHOD_FAMILY,
        ProfileField.METHOD_COMPONENTS,
    ),
    ComparisonDimensionName.DATASETS: (ProfileField.DATASETS,),
    ComparisonDimensionName.METRICS: (ProfileField.METRICS,),
    ComparisonDimensionName.EXPERIMENTAL_SETTING: (ProfileField.EXPERIMENTAL_SETTINGS,),
    ComparisonDimensionName.RESULTS: (ProfileField.KEY_RESULTS,),
    ComparisonDimensionName.ADVANTAGES: (
        ProfileField.CONTRIBUTIONS,
        ProfileField.MOTIVATION,
    ),
    ComparisonDimensionName.LIMITATIONS: (
        ProfileField.LIMITATIONS,
        ProfileField.FAILURE_CASES,
    ),
}


DIMENSION_CLAIMS: dict[ComparisonDimensionName, tuple[ClaimType, ...]] = {
    ComparisonDimensionName.RESEARCH_PROBLEM: (ClaimType.PROBLEM,),
    ComparisonDimensionName.ASSUMPTIONS: (ClaimType.ASSUMPTION,),
    ComparisonDimensionName.METHOD: (ClaimType.METHOD,),
    ComparisonDimensionName.DATASETS: (),
    ComparisonDimensionName.METRICS: (),
    ComparisonDimensionName.EXPERIMENTAL_SETTING: (),
    ComparisonDimensionName.RESULTS: (ClaimType.RESULT,),
    ComparisonDimensionName.ADVANTAGES: (ClaimType.CONTRIBUTION,),
    ComparisonDimensionName.LIMITATIONS: (ClaimType.LIMITATION,),
}


class EvidenceBackedComparisonService:
    """Build a comparison only from persisted, project-scoped Profile/Claim evidence."""

    version = "evidence-backed-comparison-v1"
    schema_version = "paper-comparison-result-v1"

    def __init__(self, repository: ResearchGraphRepository) -> None:
        self._repository = repository

    def compare(
        self, project_id: UUID, paper_ids: tuple[UUID, ...]
    ) -> PaperComparisonResult:
        if len(paper_ids) < 2:
            raise ValueError("compare requires at least two paper_ids")
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("compare paper_ids must be unique")
        titles = self._repository.get_paper_titles(project_id, paper_ids)
        missing = tuple(paper_id for paper_id in paper_ids if paper_id not in titles)
        if missing:
            raise LookupError("One or more papers were not found in project")
        profiles = self._repository.get_profiles(project_id, paper_ids)
        claims = self._repository.list_claims(project_id, paper_ids)
        profile_by_paper = self._latest_profiles(profiles)
        claims_by_paper: dict[UUID, list[Claim]] = defaultdict(list)
        for claim in claims:
            claims_by_paper[claim.paper_id].append(claim)

        dimensions: list[ComparisonDimension] = []
        for dimension_name in ComparisonDimensionName:
            cells = tuple(
                self._cell(
                    paper_id,
                    titles[paper_id],
                    dimension_name,
                    profile_by_paper.get(paper_id),
                    tuple(claims_by_paper.get(paper_id, [])),
                )
                for paper_id in paper_ids
            )
            backed_count = sum(
                cell.status == ComparisonCellStatus.EVIDENCE_BACKED for cell in cells
            )
            comparable = backed_count >= 2
            dimensions.append(
                ComparisonDimension(
                    name=dimension_name,
                    cells=cells,
                    directly_comparable=comparable,
                    non_comparable_reason=(
                        None
                        if comparable
                        else "fewer_than_two_papers_have_evidence"
                    ),
                )
            )

        comparable_count = sum(item.directly_comparable for item in dimensions)
        if comparable_count == len(dimensions):
            status = ComparisonStatus.COMPLETE
            reason = None
        elif comparable_count:
            status = ComparisonStatus.PARTIAL
            reason = "Some dimensions lack evidence for at least two papers."
        else:
            status = ComparisonStatus.INSUFFICIENT_EVIDENCE
            reason = "No comparison dimension has evidence for at least two papers."
        return PaperComparisonResult(
            project_id=project_id,
            paper_ids=paper_ids,
            status=status,
            dimensions=tuple(dimensions),
            derivation_method="deterministic_profile_claim_projection",
            generator_version=self.version,
            schema_version=self.schema_version,
            reason=reason,
        )

    @staticmethod
    def _latest_profiles(
        profiles: tuple[PaperProfile, ...],
    ) -> dict[UUID, PaperProfile]:
        result: dict[UUID, PaperProfile] = {}
        for profile in profiles:
            previous = result.get(profile.paper_id)
            if previous is None or profile.updated_at > previous.updated_at:
                result[profile.paper_id] = profile
        return result

    @staticmethod
    def _cell(
        paper_id: UUID,
        paper_title: str,
        dimension: ComparisonDimensionName,
        profile: PaperProfile | None,
        claims: tuple[Claim, ...],
    ) -> ComparisonCell:
        raw_values: list[str] = []
        normalized_values: list[str] = []
        evidence: list[EvidenceLink] = []
        confidence: list[float] = []
        if profile is not None:
            fields = set(DIMENSION_FIELDS[dimension])
            for value in profile.values:
                if value.field_name not in fields:
                    continue
                raw_values.append(value.value)
                normalized_values.append(value.normalized_value)
                evidence.extend(value.evidence_links)
                confidence.append(value.confidence)
        allowed_claims = set(DIMENSION_CLAIMS[dimension])
        selected_claim_version = (
            profile.version_id
            if profile is not None
            else (max(claims, key=lambda item: item.updated_at).version_id if claims else None)
        )
        for claim in claims:
            if (
                claim.claim_type not in allowed_claims
                or claim.version_id != selected_claim_version
            ):
                continue
            if claim.normalized_statement in normalized_values:
                continue
            raw_values.append(claim.statement)
            normalized_values.append(claim.normalized_statement)
            evidence.extend(claim.evidence_links)
            confidence.append(claim.confidence)
        if not raw_values or not evidence:
            return ComparisonCell(
                paper_id=paper_id,
                paper_title=paper_title,
                dimension=dimension,
                status=ComparisonCellStatus.INSUFFICIENT_EVIDENCE,
                normalized_value=None,
                raw_description=None,
                directly_comparable=False,
                non_comparable_reason="no_evidence",
                evidence_links=(),
                confidence=0.0,
            )
        unique_evidence: dict[str, EvidenceLink] = {}
        for link in evidence:
            unique_evidence.setdefault(link.evidence_key, link)
        return ComparisonCell(
            paper_id=paper_id,
            paper_title=paper_title,
            dimension=dimension,
            status=ComparisonCellStatus.EVIDENCE_BACKED,
            normalized_value=" | ".join(dict.fromkeys(normalized_values)),
            raw_description="\n".join(dict.fromkeys(raw_values)),
            directly_comparable=True,
            non_comparable_reason=None,
            evidence_links=tuple(unique_evidence.values()),
            confidence=min(confidence),
        )
