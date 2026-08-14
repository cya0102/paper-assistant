"""Offline Claim–Evidence entailment boundary and conservative baseline."""

from dataclasses import replace
import re

from paper_agent.domain.enums import EntailmentStatus, ReviewStatus
from paper_agent.domain.research_graph import Claim
from paper_agent.research_graph.ports import EntailmentJudge


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff-]+", re.UNICODE)
NEGATIONS = frozenset({"not", "no", "never", "without", "cannot", "不", "没有", "未", "无法"})


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(text)}


class LexicalEntailmentJudge:
    """Deterministic boundary baseline; it is intentionally conservative."""

    version = "lexical-entailment-judge-v1"

    def judge(
        self, statement: str, evidence_texts: tuple[str, ...]
    ) -> EntailmentStatus:
        if not evidence_texts:
            return EntailmentStatus.INSUFFICIENT
        statement_tokens = _tokens(statement)
        if not statement_tokens:
            return EntailmentStatus.INSUFFICIENT
        normalized_statement = " ".join(statement.casefold().split())
        best_overlap = 0.0
        statement_negated = bool(statement_tokens & NEGATIONS)
        contradictory = False
        for evidence in evidence_texts:
            evidence_tokens = _tokens(evidence)
            if not evidence_tokens:
                continue
            overlap = len(statement_tokens & evidence_tokens) / len(statement_tokens)
            best_overlap = max(best_overlap, overlap)
            evidence_negated = bool(evidence_tokens & NEGATIONS)
            if overlap >= 0.6 and statement_negated != evidence_negated:
                contradictory = True
            if normalized_statement in " ".join(evidence.casefold().split()):
                return EntailmentStatus.SUPPORTED
        if contradictory:
            return EntailmentStatus.CONTRADICTED
        if best_overlap >= 0.7:
            return EntailmentStatus.SUPPORTED
        return EntailmentStatus.INSUFFICIENT


class ClaimVerificationService:
    def __init__(self, judge: EntailmentJudge) -> None:
        self._judge = judge

    def verify(self, claim: Claim) -> Claim:
        status = self._judge.judge(
            claim.statement,
            tuple(link.evidence_text for link in claim.evidence_links),
        )
        review_status = (
            ReviewStatus.VERIFIED
            if status == EntailmentStatus.SUPPORTED
            else ReviewStatus.UNREVIEWED
        )
        return replace(
            claim,
            entailment_status=status,
            review_status=review_status,
        )
