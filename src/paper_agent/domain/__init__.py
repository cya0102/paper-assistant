"""Infrastructure-independent domain models."""

from paper_agent.domain.document import CanonicalParsedDocument
from paper_agent.domain.chunk import Chunk, DerivedDataState, SemanticGroup
from paper_agent.domain.ingestion import IngestionReport, IngestionRequest
from paper_agent.domain.metadata import PaperMetadata
from paper_agent.domain.paper import FileLocation, Paper, PaperFile, PaperVersion
from paper_agent.domain.project import Project
from paper_agent.domain.structure import Element, Section, StructuredDocument
from paper_agent.domain.indexing import (
    EmbeddingDescriptor,
    IndexDocument,
    IndexedVector,
    IndexingReport,
    IndexingState,
)
from paper_agent.domain.retrieval import (
    Evidence,
    MetadataFilter,
    RetrievalCandidate,
    SearchKnowledgeResult,
    SearchRequest,
    SearchScope,
)
from paper_agent.domain.research_graph import (
    Claim,
    EvidenceLink,
    PaperProfile,
    PaperProfileFieldValue,
    PaperRelation,
    ResearchEntity,
)
from paper_agent.domain.comparison import (
    ComparisonCell,
    ComparisonDimension,
    PaperComparisonResult,
)

__all__ = [
    "CanonicalParsedDocument",
    "Chunk",
    "DerivedDataState",
    "Element",
    "FileLocation",
    "IngestionReport",
    "IngestionRequest",
    "Paper",
    "PaperMetadata",
    "PaperFile",
    "PaperVersion",
    "Project",
    "Section",
    "SemanticGroup",
    "StructuredDocument",
    "EmbeddingDescriptor",
    "IndexDocument",
    "IndexedVector",
    "IndexingReport",
    "IndexingState",
    "Evidence",
    "MetadataFilter",
    "RetrievalCandidate",
    "SearchKnowledgeResult",
    "SearchRequest",
    "SearchScope",
    "Claim",
    "ComparisonCell",
    "ComparisonDimension",
    "EvidenceLink",
    "PaperComparisonResult",
    "PaperProfile",
    "PaperProfileFieldValue",
    "PaperRelation",
    "ResearchEntity",
]
