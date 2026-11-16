"""Pydantic schemas for research paper endpoints and library representations."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field


class PaperFileItem(BaseModel):
    """File metadata representation for a research paper file."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="UUID of the paper file")
    original_filename: str = Field(..., description="Original filename of uploaded document")
    file_size_bytes: int | None = Field(None, description="Size in bytes")
    mime_type: str = Field(..., description="MIME type of document")
    file_url: str | None = Field(None, description="Secure document access URL if available")
    status: str = Field(..., description="Status of the file ('uploaded', 'available', 'failed')")
    created_at: datetime | None = Field(None, description="Timestamp when file was uploaded")


class PaperListItemResponse(BaseModel):
    """Summarized research paper representation for catalog and library listing."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="UUID of the research paper")
    title: str = Field(..., description="Title of the research paper")
    abstract: str | None = Field(None, description="Abstract summary if available")
    doi: str | None = Field(None, description="Digital Object Identifier")
    publication_date: date | None = Field(None, description="Publication date")
    publication_year: int | None = Field(None, description="Publication year")
    venue: str | None = Field(None, description="Publication venue / conference / journal")
    publisher: str | None = Field(None, description="Publisher organization")
    authors: list[dict[str, Any]] = Field(default_factory=list, description="List of paper authors")
    created_at: datetime | None = Field(None, description="Timestamp when paper was added")
    updated_at: datetime | None = Field(None, description="Timestamp when paper was last updated")

    # Associated file summary
    original_filename: str | None = Field(None, description="Original filename of the primary/latest file")
    file_size_bytes: int | None = Field(None, description="Size of primary file in bytes")
    file_status: str | None = Field(None, description="Status of primary file")

    # Ingestion pipeline summary
    ingestion_status: str | None = Field(None, description="Latest ingestion job status ('completed', 'processing', etc.)")
    current_stage: str | None = Field(None, description="Current processing stage if in progress")

    # Extracted content statistics
    chunk_count: int = Field(0, description="Number of active indexed chunks")
    page_count: int | None = Field(None, description="Number of document pages")


class PaperListResponse(BaseModel):
    """Paginated list of research papers."""

    model_config = ConfigDict(from_attributes=True)

    items: list[PaperListItemResponse] = Field(..., description="List of research papers")
    total: int = Field(..., description="Total count of matching papers")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")


class PaperDetailResponse(PaperListItemResponse):
    """Detailed research paper representation including all associated files and metadata."""

    files: list[PaperFileItem] = Field(default_factory=list, description="All associated files for this paper")
    metadata_: dict[str, Any] = Field(default_factory=dict, description="Additional structured metadata")
