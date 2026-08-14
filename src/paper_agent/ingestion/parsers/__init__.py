"""Concrete PDF parser adapters."""

from paper_agent.ingestion.parsers.poppler_parser import PopplerPdfParser
from paper_agent.ingestion.parsers.pymupdf_parser import PyMuPdfParser

__all__ = ["PopplerPdfParser", "PyMuPdfParser"]

