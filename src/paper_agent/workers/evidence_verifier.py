"""evidence_verifier worker entrypoint (registry descriptor lives in workers.base)."""

from paper_agent.workers.base import VERIFIER_SCHEMA  # noqa: F401

__all__ = ["VERIFIER_SCHEMA"]
