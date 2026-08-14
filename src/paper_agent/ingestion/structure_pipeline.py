"""Composable, versioned document-structure derivation."""

from dataclasses import replace

from paper_agent.domain.chunk import SemanticGroup
from paper_agent.domain.document import CanonicalParsedDocument
from paper_agent.domain.structure import StructuredDocument
from paper_agent.ingestion.elements import ElementExtractor
from paper_agent.ingestion.section_tree import SectionTreeBuilder
from paper_agent.ingestion.semantic_blocks import SemanticBlockBuilder


class DocumentStructureProcessor:
    """Build all structure-derived data behind one invalidation version."""

    def __init__(
        self,
        section_builder: SectionTreeBuilder | None = None,
        element_extractor: ElementExtractor | None = None,
        semantic_builder: SemanticBlockBuilder | None = None,
    ) -> None:
        self._section_builder = section_builder or SectionTreeBuilder()
        self._element_extractor = element_extractor or ElementExtractor()
        self._semantic_builder = semantic_builder or SemanticBlockBuilder()
        self.version = "+".join(
            (
                self._section_builder.version,
                self._element_extractor.version,
                self._semantic_builder.version,
            )
        )

    def build(
        self, document: CanonicalParsedDocument
    ) -> tuple[StructuredDocument, tuple[SemanticGroup, ...]]:
        structured = self._section_builder.build(document)
        structured = replace(
            structured,
            structure_version=self.version,
            sections=tuple(
                replace(section, structure_version=self.version)
                for section in structured.sections
            ),
        )
        structured = self._element_extractor.extract(document, structured)
        groups = self._semantic_builder.build(document, structured)
        return structured, groups
