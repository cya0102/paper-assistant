"""Deterministic offline Research Graph extraction baseline."""

from collections import defaultdict
import re
from uuid import NAMESPACE_URL, UUID, uuid5

from paper_agent.domain.enums import (
    ClaimPolarity,
    ClaimType,
    EvidenceRelation,
    EvidenceTargetType,
    NormalizationStatus,
    ProfileField,
    RelationEndpointType,
    RelationType,
    ResearchEntityType,
)
from paper_agent.domain.research_graph import (
    Claim,
    ExtractionSource,
    GenerationProvenance,
    PaperProfile,
    PaperProfileExtraction,
    PaperProfileExtractionRequest,
    PaperProfileFieldValue,
    PaperRelation,
    RelationEndpoint,
    ResearchEntity,
)


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def normalize_statement(value: str) -> str:
    return " ".join(value.casefold().strip().rstrip(".!?。！？").split())


FIELD_CUES: dict[ProfileField, tuple[str, ...]] = {
    ProfileField.RESEARCH_PROBLEM: (
        "we address",
        "research problem",
        "challenge",
        "task of",
        "问题",
        "挑战",
        "目标",
    ),
    ProfileField.MOTIVATION: (
        "motivat",
        "existing methods",
        "however",
        "because",
        "现有方法",
        "由于",
        "然而",
    ),
    ProfileField.HYPOTHESES: ("hypoth", "we posit", "we conjecture", "假设"),
    ProfileField.CONTRIBUTIONS: (
        "contribution",
        "we propose",
        "we introduce",
        "贡献",
        "我们提出",
    ),
    ProfileField.ASSUMPTIONS: ("we assume", "assumption", "假定", "假设条件"),
    ProfileField.EXPERIMENTAL_SETTINGS: (
        "implementation detail",
        "we train",
        "learning rate",
        "batch size",
        "实验设置",
        "训练设置",
    ),
    ProfileField.KEY_RESULTS: (
        "outperform",
        "achieve",
        "improve",
        "state-of-the-art",
        "结果表明",
        "优于",
        "提升",
    ),
    ProfileField.LIMITATIONS: ("limitation", "is limited", "drawback", "局限", "限制"),
    ProfileField.FAILURE_CASES: ("failure case", "fails when", "failure mode", "失败"),
    ProfileField.FUTURE_WORK: ("future work", "in the future", "后续工作", "未来工作"),
}


SECTION_FIELDS: dict[ProfileField, tuple[str, ...]] = {
    ProfileField.METHOD_COMPONENTS: ("method", "approach", "framework", "model", "方法", "模型"),
    ProfileField.KEY_RESULTS: ("result", "experiment", "evaluation", "结果", "实验"),
    ProfileField.LIMITATIONS: ("limitation", "discussion", "局限", "讨论"),
}


NAMED_ENTITY_PATTERNS: tuple[
    tuple[ProfileField, ResearchEntityType, re.Pattern[str]], ...
] = (
    (
        ProfileField.DATASETS,
        ResearchEntityType.DATASET,
        re.compile(
            r"\b(?:datasets?|benchmarks?)\s*(?:include|such as|:|=|are|is)\s*([^.;。；]+)",
            re.IGNORECASE,
        ),
    ),
    (
        ProfileField.METRICS,
        ResearchEntityType.METRIC,
        re.compile(
            r"\b(?:metrics?|measures?)\s*(?:include|such as|:|=|are|is)\s*([^.;。；]+)",
            re.IGNORECASE,
        ),
    ),
    (
        ProfileField.BASELINES,
        ResearchEntityType.BASELINE,
        re.compile(
            r"\b(?:baselines?|compared with|compare against)\s*(?:include|such as|:|=|are|is)?\s*([^.;。；]+)",
            re.IGNORECASE,
        ),
    ),
    (
        ProfileField.METHOD_NAME,
        ResearchEntityType.METHOD,
        re.compile(
            r"\b(?:method|model|framework)\s*(?:called|named|:|=|is)\s*([A-Za-z][\w-]{1,80})",
            re.IGNORECASE,
        ),
    ),
)


CLAIM_FIELDS: dict[ProfileField, ClaimType] = {
    ProfileField.RESEARCH_PROBLEM: ClaimType.PROBLEM,
    ProfileField.HYPOTHESES: ClaimType.HYPOTHESIS,
    ProfileField.CONTRIBUTIONS: ClaimType.CONTRIBUTION,
    ProfileField.METHOD_NAME: ClaimType.METHOD,
    ProfileField.METHOD_FAMILY: ClaimType.METHOD,
    ProfileField.METHOD_COMPONENTS: ClaimType.METHOD,
    ProfileField.ASSUMPTIONS: ClaimType.ASSUMPTION,
    ProfileField.KEY_RESULTS: ClaimType.RESULT,
    ProfileField.LIMITATIONS: ClaimType.LIMITATION,
    ProfileField.FAILURE_CASES: ClaimType.LIMITATION,
    ProfileField.FUTURE_WORK: ClaimType.FUTURE_WORK,
}


ENTITY_RELATIONS: dict[ResearchEntityType, RelationType] = {
    ResearchEntityType.METHOD: RelationType.USES_METHOD,
    ResearchEntityType.METHOD_COMPONENT: RelationType.USES_METHOD,
    ResearchEntityType.DATASET: RelationType.EVALUATES_ON,
    ResearchEntityType.BASELINE: RelationType.USES_METHOD,
}


class RuleBasedPaperProfileExtractor:
    """Conservative baseline that emits only text explicitly present in a Chunk."""

    version = "rule-based-profile-extractor-v1"
    schema_version = "research-graph-profile-v1"

    def extract(self, request: PaperProfileExtractionRequest) -> PaperProfileExtraction:
        provenance = GenerationProvenance(
            extraction_method="rule_based",
            extractor_version=self.version,
            schema_version=self.schema_version,
            source_document_hash=request.source_document_hash,
            chunking_version=request.chunking_version,
        )
        profile_id = uuid5(
            NAMESPACE_URL,
            f"paper-profile:{request.project_id}:{request.version_id}:{provenance.generation_key}",
        )
        candidates, named_entities = self._candidates(request.sources)
        values = self._profile_values(profile_id, candidates)
        profile = PaperProfile(
            profile_id=profile_id,
            project_id=request.project_id,
            paper_id=request.paper_id,
            version_id=request.version_id,
            values=values,
            provenance=provenance,
        )
        claims = self._claims(profile)
        entities, relations = self._entities_and_relations(
            request.project_id,
            request.paper_id,
            provenance,
            named_entities,
        )
        return PaperProfileExtraction(
            profile=profile,
            claims=claims,
            entities=entities,
            relations=relations,
        )

    @staticmethod
    def _candidates(
        sources: tuple[ExtractionSource, ...],
    ) -> tuple[
        dict[ProfileField, list[tuple[str, ExtractionSource, float]]],
        list[tuple[str, ResearchEntityType, ExtractionSource]],
    ]:
        candidates: dict[ProfileField, list[tuple[str, ExtractionSource, float]]] = defaultdict(list)
        named_entities: list[tuple[str, ResearchEntityType, ExtractionSource]] = []
        for source in sources:
            sentences = tuple(
                sentence.strip()
                for sentence in SENTENCE_BOUNDARY.split(source.text)
                if sentence.strip()
            )
            section = source.section_path.casefold()
            for field_name, cues in SECTION_FIELDS.items():
                if any(cue in section for cue in cues) and sentences:
                    sentence = sentences[0]
                    if field_name != ProfileField.KEY_RESULTS or any(
                        cue in sentence.casefold() for cue in FIELD_CUES[ProfileField.KEY_RESULTS]
                    ):
                        candidates[field_name].append((sentence, source, 0.65))
            for sentence in sentences:
                normalized_sentence = sentence.casefold()
                for field_name, cues in FIELD_CUES.items():
                    if any(cue in normalized_sentence for cue in cues):
                        candidates[field_name].append((sentence, source, 0.72))
            for field_name, entity_type, pattern in NAMED_ENTITY_PATTERNS:
                for match in pattern.finditer(source.text):
                    for raw_name in re.split(r",|\band\b|\bor\b", match.group(1)):
                        name = raw_name.strip(" \t\n:;,.()[]")
                        if not name or len(name) > 120:
                            continue
                        candidates[field_name].append((name, source, 0.82))
                        named_entities.append((name, entity_type, source))
        return candidates, named_entities

    @staticmethod
    def _profile_values(
        profile_id: UUID,
        candidates: dict[ProfileField, list[tuple[str, ExtractionSource, float]]],
    ) -> tuple[PaperProfileFieldValue, ...]:
        values: list[PaperProfileFieldValue] = []
        for field_name in ProfileField:
            seen: set[str] = set()
            ordinal = 0
            for value, source, confidence in candidates.get(field_name, []):
                normalized = normalize_statement(value)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                field_id = uuid5(
                    profile_id,
                    f"profile-field:{field_name.value}:{ordinal}:{source.chunk_id}:{normalized}",
                )
                evidence = source.evidence_for(
                    EvidenceTargetType.PROFILE_FIELD,
                    field_id,
                    confidence=confidence,
                )
                values.append(
                    PaperProfileFieldValue(
                        field_id=field_id,
                        field_name=field_name,
                        value=value.strip(),
                        normalized_value=normalized,
                        ordinal=ordinal,
                        confidence=confidence,
                        evidence_links=(evidence,),
                    )
                )
                ordinal += 1
                if ordinal >= 5:
                    break
        return tuple(values)

    @staticmethod
    def _claims(profile: PaperProfile) -> tuple[Claim, ...]:
        claims: list[Claim] = []
        for value in profile.values:
            claim_type = CLAIM_FIELDS.get(value.field_name)
            if claim_type is None:
                continue
            claim_id = uuid5(profile.profile_id, f"claim:{value.field_id}:{claim_type.value}")
            evidence = tuple(
                link.for_target(EvidenceTargetType.CLAIM, claim_id)
                for link in value.evidence_links
            )
            claims.append(
                Claim(
                    claim_id=claim_id,
                    project_id=profile.project_id,
                    paper_id=profile.paper_id,
                    version_id=profile.version_id,
                    claim_type=claim_type,
                    statement=value.value,
                    normalized_statement=value.normalized_value,
                    polarity=ClaimPolarity.NEUTRAL,
                    confidence=value.confidence,
                    provenance=profile.provenance,
                    evidence_links=evidence,
                )
            )
        return tuple(claims)

    @staticmethod
    def _entities_and_relations(
        project_id: UUID,
        paper_id: UUID,
        provenance: GenerationProvenance,
        candidates: list[tuple[str, ResearchEntityType, ExtractionSource]],
    ) -> tuple[tuple[ResearchEntity, ...], tuple[PaperRelation, ...]]:
        entities: list[ResearchEntity] = []
        relations: list[PaperRelation] = []
        seen: set[tuple[ResearchEntityType, str]] = set()
        for name, entity_type, source in candidates:
            normalized = normalize_statement(name)
            key = (entity_type, normalized)
            if key in seen:
                continue
            seen.add(key)
            entity_id = uuid5(
                NAMESPACE_URL,
                f"research-entity:{project_id}:{entity_type.value}:{normalized}",
            )
            entity_evidence = source.evidence_for(
                EvidenceTargetType.ENTITY,
                entity_id,
                confidence=0.82,
                relation_to_target=EvidenceRelation.MENTIONS,
            )
            entities.append(
                ResearchEntity(
                    entity_id=entity_id,
                    project_id=project_id,
                    canonical_name=name,
                    aliases=(),
                    entity_type=entity_type,
                    description=None,
                    normalization_status=NormalizationStatus.PROPOSED,
                    provenance=provenance,
                    evidence_links=(entity_evidence,),
                )
            )
            relation_type = ENTITY_RELATIONS.get(entity_type)
            if relation_type is None:
                continue
            relation_id = uuid5(
                NAMESPACE_URL,
                f"research-relation:{project_id}:{paper_id}:{relation_type.value}:{entity_id}:"
                f"{provenance.generation_key}",
            )
            relation_evidence = source.evidence_for(
                EvidenceTargetType.RELATION,
                relation_id,
                confidence=0.78,
                relation_to_target=EvidenceRelation.SUPPORTS,
            )
            relations.append(
                PaperRelation(
                    relation_id=relation_id,
                    project_id=project_id,
                    source=RelationEndpoint(RelationEndpointType.PAPER, paper_id),
                    target=RelationEndpoint(RelationEndpointType.ENTITY, entity_id),
                    relation_type=relation_type,
                    description=f"Paper {relation_type.value.replace('_', ' ')} {name}.",
                    confidence=0.78,
                    provenance=provenance,
                    evidence_links=(relation_evidence,),
                )
            )
        return tuple(entities), tuple(relations)
