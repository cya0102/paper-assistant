"""Canonical parsed-document schema and parser provenance."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_agent.domain.enums import BlockType


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Bounding-box maximums must not be smaller than minimums")
        return self


class ParserDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class DocumentBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    block_id: str = Field(min_length=1)
    block_type: BlockType
    text: str | None = None
    bbox: BoundingBox | None = None
    reading_order: int = Field(ge=0)
    references: tuple[str, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def remove_nul_characters(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.replace("\x00", "")


class DocumentPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    blocks: tuple[DocumentBlock, ...] = ()

    @model_validator(mode="after")
    def validate_blocks(self) -> "DocumentPage":
        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError(f"Page {self.page_number} contains duplicate block IDs")
        orders = [block.reading_order for block in self.blocks]
        if len(orders) != len(set(orders)):
            raise ValueError(f"Page {self.page_number} contains duplicate reading-order values")
        return self


class CanonicalParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    paper_id: UUID
    version_id: UUID
    source_file_id: UUID
    parser: ParserDescriptor
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pages: tuple[DocumentPage, ...]

    @model_validator(mode="after")
    def validate_document(self) -> "CanonicalParsedDocument":
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != sorted(page_numbers) or len(page_numbers) != len(set(page_numbers)):
            raise ValueError("Page numbers must be unique and ascending")
        all_block_ids = [block.block_id for page in self.pages for block in page.blocks]
        if len(all_block_ids) != len(set(all_block_ids)):
            raise ValueError("block_id must be unique across the document")
        return self

