from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from paper_agent.agent.tool_adapters import ComparePapersToolAdapter
from paper_agent.domain.enums import (
    ClaimPolarity,
    ClaimType,
    ComparisonDimensionName,
    ComparisonStatus,
    EntailmentStatus,
    EvidenceRelation,
    EvidenceTargetType,
    ProfileField,
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
)
from paper_agent.research_graph.entailment import (
    ClaimVerificationService,
    LexicalEntailmentJudge,
)
from paper_agent.research_graph.extractor import RuleBasedPaperProfileExtractor
from paper_agent.research_graph.service import EvidenceBackedComparisonService
from paper_agent.domain.enums import RelationEndpointType, RelationType


PROVENANCE = GenerationProvenance(
    extraction_method="rule_based",
    extractor_version="test-v1",
    schema_version="test-schema-v1",
    source_document_hash="a" * 64,
    chunking_version="chunk-v1",
)


def _evidence(project_id, paper_id, version_id, target_type, target_id, text):
    return EvidenceLink(
        project_id=project_id,
        target_type=target_type,
        target_id=target_id,
        paper_id=paper_id,
        version_id=version_id,
        section_id=uuid4(),
        chunk_id=uuid4(),
        page_start=2,
        page_end=3,
        source_block_ids=("b1",),
        evidence_text=text,
        relation_to_target=EvidenceRelation.SUPPORTS,
        confidence=0.9,
    )


def _profile(project_id, paper_id, version_id, field_name, text):
    profile_id = uuid4()
    field_id = uuid4()
    value = PaperProfileFieldValue(
        field_id=field_id,
        field_name=field_name,
        value=text,
        normalized_value=text.casefold(),
        ordinal=0,
        confidence=0.9,
        evidence_links=(
            _evidence(
                project_id,
                paper_id,
                version_id,
                EvidenceTargetType.PROFILE_FIELD,
                field_id,
                text,
            ),
        ),
    )
    return PaperProfile(
        profile_id=profile_id,
        project_id=project_id,
        paper_id=paper_id,
        version_id=version_id,
        values=(value,),
        provenance=PROVENANCE,
    )


def test_evidence_link_requires_complete_chunk_and_block_provenance():
    with pytest.raises(ValueError, match="source_block_ids"):
        EvidenceLink(
            project_id=uuid4(),
            target_type=EvidenceTargetType.CLAIM,
            target_id=uuid4(),
            paper_id=uuid4(),
            version_id=uuid4(),
            section_id=uuid4(),
            chunk_id=uuid4(),
            page_start=1,
            page_end=1,
            source_block_ids=(),
            evidence_text="supported statement",
            relation_to_target=EvidenceRelation.SUPPORTS,
            confidence=0.9,
        )


def test_claim_cannot_be_verified_without_supported_entailment():
    project_id, paper_id, version_id, claim_id = uuid4(), uuid4(), uuid4(), uuid4()
    with pytest.raises(ValueError, match="Only supported claims"):
        Claim(
            claim_id=claim_id,
            project_id=project_id,
            paper_id=paper_id,
            version_id=version_id,
            claim_type=ClaimType.RESULT,
            statement="The method improves accuracy.",
            normalized_statement="the method improves accuracy",
            polarity=ClaimPolarity.POSITIVE,
            confidence=0.8,
            provenance=PROVENANCE,
            evidence_links=(
                _evidence(
                    project_id,
                    paper_id,
                    version_id,
                    EvidenceTargetType.CLAIM,
                    claim_id,
                    "Unrelated evidence.",
                ),
            ),
            review_status=ReviewStatus.VERIFIED,
            entailment_status=EntailmentStatus.INSUFFICIENT,
        )


def test_undirected_relation_key_is_canonical_for_deduplication():
    project_id, paper_id, version_id = uuid4(), uuid4(), uuid4()
    left, right = uuid4(), uuid4()

    def relation(source, target):
        relation_id = uuid4()
        return PaperRelation(
            relation_id=relation_id,
            project_id=project_id,
            source=RelationEndpoint(RelationEndpointType.PAPER, source),
            target=RelationEndpoint(RelationEndpointType.PAPER, target),
            relation_type=RelationType.SAME_PROBLEM,
            description="The papers study the same problem.",
            confidence=0.8,
            provenance=PROVENANCE,
            evidence_links=(
                _evidence(
                    project_id,
                    paper_id,
                    version_id,
                    EvidenceTargetType.RELATION,
                    relation_id,
                    "Both papers formulate the same task.",
                ),
            ),
        )

    assert relation(left, right).relation_key == relation(right, left).relation_key


def test_rule_based_extractor_is_deterministic_offline_and_evidence_first():
    project_id, paper_id, version_id, section_id, chunk_id = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    request = PaperProfileExtractionRequest(
        project_id=project_id,
        paper_id=paper_id,
        version_id=version_id,
        paper_title="GraphNet",
        source_document_hash="b" * 64,
        chunking_version="chunk-v1",
        sources=(
            ExtractionSource(
                project_id=project_id,
                paper_id=paper_id,
                version_id=version_id,
                section_id=section_id,
                chunk_id=chunk_id,
                section_path="3 Method",
                text=(
                    "We propose a graph mechanism for the task. "
                    "Method called GraphNet. Datasets: AlphaSet. Metrics: F1."
                ),
                page_start=3,
                page_end=3,
                source_block_ids=("b3",),
                element_ids=(),
            ),
        ),
    )
    extractor = RuleBasedPaperProfileExtractor()
    first = extractor.extract(request)
    second = extractor.extract(request)

    assert first.profile.profile_id == second.profile.profile_id
    assert tuple(value.field_id for value in first.profile.values) == tuple(
        value.field_id for value in second.profile.values
    )
    assert tuple(claim.claim_id for claim in first.claims) == tuple(
        claim.claim_id for claim in second.claims
    )
    assert tuple(entity.entity_id for entity in first.entities) == tuple(
        entity.entity_id for entity in second.entities
    )
    assert tuple(relation.relation_id for relation in first.relations) == tuple(
        relation.relation_id for relation in second.relations
    )
    assert first.profile.values
    assert first.claims and first.entities and first.relations
    assert all(value.evidence_links for value in first.profile.values)
    assert all(claim.evidence_links for claim in first.claims)
    verified = ClaimVerificationService(LexicalEntailmentJudge()).verify(first.claims[0])
    assert verified.entailment_status == EntailmentStatus.SUPPORTED
    assert verified.review_status == ReviewStatus.VERIFIED


class MemoryGraphRepository:
    def __init__(self, project_id, titles, profiles=(), claims=()):
        self.project_id = project_id
        self.titles = titles
        self.profiles = profiles
        self.claims = claims

    def get_paper_titles(self, project_id, paper_ids):
        assert project_id == self.project_id
        return {paper_id: self.titles[paper_id] for paper_id in paper_ids if paper_id in self.titles}

    def get_profiles(self, project_id, paper_ids, version_ids=(), *, active_only=True):
        del version_ids, active_only
        assert project_id == self.project_id
        return tuple(profile for profile in self.profiles if profile.paper_id in paper_ids)

    def list_claims(self, project_id, paper_ids=(), claim_types=(), *, active_only=True):
        del claim_types, active_only
        assert project_id == self.project_id
        return tuple(claim for claim in self.claims if not paper_ids or claim.paper_id in paper_ids)


def test_comparison_refuses_to_invent_when_profiles_and_claims_have_no_evidence():
    project_id, first, second = uuid4(), uuid4(), uuid4()
    repository = MemoryGraphRepository(
        project_id, {first: "First Paper", second: "Second Paper"}
    )

    result = EvidenceBackedComparisonService(repository).compare(
        project_id, (first, second)
    )

    assert result.status == ComparisonStatus.INSUFFICIENT_EVIDENCE
    assert result.reason
    assert all(not dimension.directly_comparable for dimension in result.dimensions)
    assert all(
        not cell.evidence_links and cell.raw_description is None
        for dimension in result.dimensions
        for cell in dimension.cells
    )


def test_comparison_and_tool_contract_serialize_evidence_backed_cells():
    project_id, first, second = uuid4(), uuid4(), uuid4()
    profiles = (
        _profile(project_id, first, uuid4(), ProfileField.METHOD_COMPONENTS, "Uses a graph encoder."),
        _profile(project_id, second, uuid4(), ProfileField.METHOD_COMPONENTS, "Uses a transformer encoder."),
    )
    repository = MemoryGraphRepository(
        project_id,
        {first: "Graph Paper", second: "Transformer Paper"},
        profiles,
    )
    adapter = ComparePapersToolAdapter(
        EvidenceBackedComparisonService(repository), project_id
    )

    contract = adapter.contract()
    payload = contract.handler({"paper_ids": [str(first), str(second)]})

    assert contract.name == "compare_papers"
    assert contract.parameters["properties"]["paper_ids"]["minItems"] == 2
    assert payload["status"] == ComparisonStatus.PARTIAL.value
    method = next(
        item
        for item in payload["dimensions"]
        if item["name"] == ComparisonDimensionName.METHOD.value
    )
    assert method["directly_comparable"]
    assert all(cell["evidence"] for cell in method["cells"])
    assert payload["evidence"]
    assert all(item["citation"].startswith("E") for item in payload["evidence"])
