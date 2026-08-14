"""Stable domain state values shared across adapters."""

from enum import StrEnum


class PipelineStage(StrEnum):
    DISCOVERED = "discovered"
    IDENTITY_RESOLVED = "identity_resolved"
    PARSING = "parsing"
    PARSED = "parsed"
    STRUCTURED = "structured"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


class FileStatus(StrEnum):
    DISCOVERED = "discovered"
    IDENTITY_RESOLVED = "identity_resolved"
    PARSING = "parsing"
    PARSED = "parsed"
    STRUCTURED = "structured"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


class LocationPresence(StrEnum):
    PRESENT = "present"
    MISSING = "missing"


class IngestionDisposition(StrEnum):
    NEW = "new"
    MODIFIED = "modified"
    DUPLICATE = "duplicate"
    UNCHANGED = "unchanged"
    MISSING = "missing"
    FAILED = "failed"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    ALGORITHM = "algorithm"
    REFERENCE = "reference"
    OTHER = "other"


class SourceType(StrEnum):
    LOCAL = "local"
    ARXIV = "arxiv"
    DOI = "doi"
    CONFERENCE = "conference"
    OTHER = "other"


class MetadataSource(StrEnum):
    PDF_METADATA = "pdf_metadata"
    FIRST_PAGE = "first_page"
    DOCUMENT_TEXT = "document_text"
    FILE_NAME = "file_name"


class IdentityMatchType(StrEnum):
    NEW_PAPER = "new_paper"
    DOI = "doi"
    ARXIV = "arxiv"
    CONTENT_HASH = "content_hash"
    TITLE_AUTHORS = "title_authors"
    EXISTING_FILE = "existing_file"


class ElementType(StrEnum):
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    ALGORITHM = "algorithm"


class SemanticGroupType(StrEnum):
    TEXT = "text"
    ELEMENT_DEPENDENCY = "element_dependency"


class ChunkType(StrEnum):
    TEXT = "text"
    MIXED = "mixed"
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"
    ALGORITHM = "algorithm"


class IndexLevel(StrEnum):
    PAPER = "paper"
    SECTION = "section"
    CHUNK = "chunk"


class IndexingStatus(StrEnum):
    RUNNING = "running"
    INDEXED = "indexed"
    FAILED = "failed"


class SearchStatus(StrEnum):
    OK = "ok"
    NO_EVIDENCE = "no_evidence"


class ProfileField(StrEnum):
    RESEARCH_PROBLEM = "research_problem"
    MOTIVATION = "motivation"
    HYPOTHESES = "hypotheses"
    CONTRIBUTIONS = "contributions"
    METHOD_NAME = "method_name"
    METHOD_FAMILY = "method_family"
    METHOD_COMPONENTS = "method_components"
    ASSUMPTIONS = "assumptions"
    DATASETS = "datasets"
    METRICS = "metrics"
    BASELINES = "baselines"
    EXPERIMENTAL_SETTINGS = "experimental_settings"
    KEY_RESULTS = "key_results"
    LIMITATIONS = "limitations"
    FAILURE_CASES = "failure_cases"
    FUTURE_WORK = "future_work"


class ClaimType(StrEnum):
    PROBLEM = "problem"
    HYPOTHESIS = "hypothesis"
    CONTRIBUTION = "contribution"
    METHOD = "method"
    ASSUMPTION = "assumption"
    RESULT = "result"
    LIMITATION = "limitation"
    FUTURE_WORK = "future_work"


class ClaimPolarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    REVISED = "revised"


class EntailmentStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    UNREVIEWED = "unreviewed"


class ResearchEntityType(StrEnum):
    TASK = "task"
    RESEARCH_PROBLEM = "research_problem"
    METHOD = "method"
    METHOD_COMPONENT = "method_component"
    MECHANISM = "mechanism"
    DATASET = "dataset"
    METRIC = "metric"
    BASELINE = "baseline"
    CONCEPT = "concept"
    DOMAIN = "domain"


class NormalizationStatus(StrEnum):
    PROPOSED = "proposed"
    NORMALIZED = "normalized"
    MERGED = "merged"
    REJECTED = "rejected"


class RelationEndpointType(StrEnum):
    PAPER = "paper"
    ENTITY = "entity"


class RelationType(StrEnum):
    CITES = "cites"
    CITED_BY = "cited_by"
    EXTENDS = "extends"
    IMPROVES = "improves"
    SIMPLIFIES = "simplifies"
    USES_METHOD = "uses_method"
    USES_DATASET = "uses_dataset"
    EVALUATES_ON = "evaluates_on"
    SAME_PROBLEM = "same_problem"
    DIFFERENT_ASSUMPTION = "different_assumption"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REPRODUCES = "reproduces"
    FAILS_TO_REPRODUCE = "fails_to_reproduce"
    ANALOGOUS_TO = "analogous_to"
    INSPIRED_BY = "inspired_by"


class EvidenceTargetType(StrEnum):
    PROFILE_FIELD = "profile_field"
    CLAIM = "claim"
    ENTITY = "entity"
    RELATION = "relation"
    COMPARISON_CELL = "comparison_cell"


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    MENTIONS = "mentions"
    DERIVED_FROM = "derived_from"


class EvidenceKind(StrEnum):
    PAPER_FACT = "paper_fact"
    MODEL_SUMMARY = "model_summary"
    SYSTEM_INFERENCE = "system_inference"


class ComparisonDimensionName(StrEnum):
    RESEARCH_PROBLEM = "research_problem"
    ASSUMPTIONS = "assumptions"
    METHOD = "method"
    DATASETS = "datasets"
    METRICS = "metrics"
    EXPERIMENTAL_SETTING = "experimental_setting"
    RESULTS = "results"
    ADVANTAGES = "advantages"
    LIMITATIONS = "limitations"


class ComparisonCellStatus(StrEnum):
    EVIDENCE_BACKED = "evidence_backed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ComparisonStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
