import os
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.database import upgrade_database
from paper_agent.domain.enums import (
    ClaimPolarity,
    ClaimType,
    EntailmentStatus,
    EvidenceRelation,
    EvidenceTargetType,
    FileStatus,
    NormalizationStatus,
    PipelineStage,
    ProfileField,
    RelationEndpointType,
    RelationType,
    ResearchEntityType,
    ReviewStatus,
    SourceType,
)
from paper_agent.domain.research_graph import (
    Claim,
    EvidenceLink,
    GenerationProvenance,
    PaperProfile,
    PaperProfileFieldValue,
    PaperRelation,
    RelationEndpoint,
    ResearchEntity,
)
from paper_agent.storage.postgres.models import (
    PaperFileRow,
    PaperRow,
    PaperVersionRow,
    ProjectRow,
)
from paper_agent.storage.postgres.research_graph_repository import (
    SqlAlchemyResearchGraphRepository,
)


DATABASE_URL = os.getenv("PAPER_AGENT_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="PAPER_AGENT_TEST_DATABASE_URL is required for real PostgreSQL tests",
)


def _evidence(
    project_id,
    paper_id,
    version_id,
    target_type,
    target_id,
    *,
    text="The method evaluates on AlphaSet.",
):
    return EvidenceLink(
        project_id=project_id,
        target_type=target_type,
        target_id=target_id,
        paper_id=paper_id,
        version_id=version_id,
        section_id=uuid4(),
        chunk_id=uuid4(),
        page_start=4,
        page_end=4,
        source_block_ids=("b4",),
        evidence_text=text,
        relation_to_target=EvidenceRelation.SUPPORTS,
        confidence=0.9,
    )


def test_research_graph_repository_round_trip_isolation_and_relation_dedup():
    assert DATABASE_URL is not None
    upgrade_database(DATABASE_URL)
    engine = create_engine(DATABASE_URL)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    project_id, other_project_id = uuid4(), uuid4()
    paper_id, version_id = uuid4(), uuid4()
    provenance = GenerationProvenance(
        extraction_method="rule_based",
        extractor_version="integration-v1",
        schema_version="research-graph-v1",
        source_document_hash="d" * 64,
        chunking_version="chunk-v1",
    )
    try:
        with factory.begin() as session:
            session.add_all(
                [
                    ProjectRow(
                        project_id=project_id,
                        name="research-graph",
                        root_path=f"/tmp/{project_id}",
                    ),
                    ProjectRow(
                        project_id=other_project_id,
                        name="research-graph-other",
                        root_path=f"/tmp/{other_project_id}",
                    ),
                    PaperRow(
                        paper_id=paper_id,
                        canonical_title="Evidence Graph Paper",
                        aliases_json=[],
                        authors_json=[],
                        normalized_authors_json=[],
                    ),
                ]
            )
            session.flush()
            session.add(
                PaperVersionRow(
                    version_id=version_id,
                    paper_id=paper_id,
                    source_type=SourceType.LOCAL.value,
                    pipeline_status=PipelineStage.CHUNKED.value,
                )
            )
            session.flush()
            session.add(
                PaperFileRow(
                    file_id=uuid4(),
                    project_id=project_id,
                    paper_id=paper_id,
                    version_id=version_id,
                    file_size=10,
                    file_hash=(paper_id.hex + version_id.hex)[:64],
                    page_count=8,
                    metadata_json={},
                    is_canonical=True,
                    status=FileStatus.CHUNKED.value,
                )
            )

        repository = SqlAlchemyResearchGraphRepository(factory)
        profile_id, field_id = uuid4(), uuid4()
        profile = PaperProfile(
            profile_id=profile_id,
            project_id=project_id,
            paper_id=paper_id,
            version_id=version_id,
            values=(
                PaperProfileFieldValue(
                    field_id=field_id,
                    field_name=ProfileField.DATASETS,
                    value="AlphaSet",
                    normalized_value="alphaset",
                    ordinal=0,
                    confidence=0.9,
                    evidence_links=(
                        _evidence(
                            project_id,
                            paper_id,
                            version_id,
                            EvidenceTargetType.PROFILE_FIELD,
                            field_id,
                        ),
                    ),
                ),
            ),
            provenance=provenance,
        )
        stored_profile = repository.save_profile(profile)
        assert stored_profile == repository.get_profiles(project_id, (paper_id,))[0]
        assert stored_profile.values[0].evidence_links[0].source_block_ids == ("b4",)

        claim_id = uuid4()
        claim = Claim(
            claim_id=claim_id,
            project_id=project_id,
            paper_id=paper_id,
            version_id=version_id,
            claim_type=ClaimType.RESULT,
            statement="The method evaluates on AlphaSet.",
            normalized_statement="the method evaluates on alphaset",
            polarity=ClaimPolarity.POSITIVE,
            confidence=0.9,
            provenance=provenance,
            evidence_links=(
                _evidence(
                    project_id,
                    paper_id,
                    version_id,
                    EvidenceTargetType.CLAIM,
                    claim_id,
                ),
            ),
            review_status=ReviewStatus.VERIFIED,
            entailment_status=EntailmentStatus.SUPPORTED,
        )
        assert repository.save_claims((claim,))[0] == claim
        assert repository.list_claims(project_id, (paper_id,))[0].claim_id == claim_id

        entity_id = uuid4()
        entity = ResearchEntity(
            entity_id=entity_id,
            project_id=project_id,
            canonical_name="AlphaSet",
            aliases=("Alpha Set",),
            entity_type=ResearchEntityType.DATASET,
            description="Evaluation dataset.",
            normalization_status=NormalizationStatus.NORMALIZED,
            provenance=provenance,
            evidence_links=(
                _evidence(
                    project_id,
                    paper_id,
                    version_id,
                    EvidenceTargetType.ENTITY,
                    entity_id,
                ),
            ),
        )
        stored_entity = repository.save_entities((entity,))[0]
        assert stored_entity.entity_id == entity_id
        assert repository.find_entities(project_id, "alpha set")[0].entity_id == entity_id

        relation_id = uuid4()
        relation = PaperRelation(
            relation_id=relation_id,
            project_id=project_id,
            source=RelationEndpoint(RelationEndpointType.PAPER, paper_id),
            target=RelationEndpoint(RelationEndpointType.ENTITY, entity_id),
            relation_type=RelationType.EVALUATES_ON,
            description="The paper evaluates on AlphaSet.",
            confidence=0.9,
            provenance=provenance,
            evidence_links=(
                _evidence(
                    project_id,
                    paper_id,
                    version_id,
                    EvidenceTargetType.RELATION,
                    relation_id,
                ),
            ),
        )
        first = repository.save_relations((relation,))[0]
        second = repository.save_relations((relation,))[0]
        assert first.relation_id == second.relation_id == relation_id
        by_paper = repository.query_relations(
            project_id,
            paper_id=paper_id,
            relation_types=(RelationType.EVALUATES_ON,),
        )
        by_entity = repository.query_relations(project_id, entity_id=entity_id)
        assert len(by_paper) == len(by_entity) == 1
        assert by_paper[0].evidence_links

        assert repository.get_profiles(other_project_id, (paper_id,)) == ()
        assert repository.list_claims(other_project_id, (paper_id,)) == ()
        assert repository.find_entities(other_project_id, entity_ids=(entity_id,)) == ()
        assert repository.query_relations(other_project_id, paper_id=paper_id) == ()

        foreign_profile = replace(
            profile,
            profile_id=uuid4(),
            project_id=other_project_id,
            values=tuple(
                replace(
                    value,
                    evidence_links=tuple(
                        replace(link, evidence_id=uuid4(), project_id=other_project_id)
                        for link in value.evidence_links
                    ),
                )
                for value in profile.values
            ),
        )
        with pytest.raises(LookupError, match="not found in project"):
            repository.save_profile(foreign_profile)
    finally:
        with factory.begin() as session:
            session.execute(
                delete(ProjectRow).where(
                    ProjectRow.project_id.in_((project_id, other_project_id))
                )
            )
            paper = session.get(PaperRow, paper_id)
            if paper is not None:
                session.delete(paper)
        engine.dispose()
