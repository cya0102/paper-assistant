"""Research Graph extraction, grounding, and comparison public API."""

from paper_agent.research_graph.entailment import (
    ClaimVerificationService,
    LexicalEntailmentJudge,
)
from paper_agent.research_graph.extractor import RuleBasedPaperProfileExtractor
from paper_agent.research_graph.service import (
    EvidenceBackedComparisonService,
    ResearchGraphService,
)

__all__ = [
    "ClaimVerificationService",
    "EvidenceBackedComparisonService",
    "LexicalEntailmentJudge",
    "ResearchGraphService",
    "RuleBasedPaperProfileExtractor",
]
