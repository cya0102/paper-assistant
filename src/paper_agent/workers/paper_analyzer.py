"""paper_analyzer worker entrypoint (registry descriptor lives in workers.base)."""

from paper_agent.workers.base import ANALYZER_SCHEMA  # noqa: F401

__all__ = ["ANALYZER_SCHEMA"]
