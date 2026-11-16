"""Pydantic schemas for document ingestion endpoints."""

from __future__ import annotations

from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field


class IngestionUploadResponse(BaseModel):
    """Response returned upon successful PDF ingestion."""

    model_config = ConfigDict(from_attributes=True)

    paper_id: uuid.UUID = Field(..., description="UUID of the research paper")
    job_id: uuid.UUID = Field(..., description="UUID of the ingestion job")
    paper_file_id: uuid.UUID = Field(..., description="UUID of the associated paper file")
    title: str | None = Field(None, description="Title of the research paper")
    chunks_created: int = Field(..., description="Number of chunks created or associated with this job")
    embeddings_created: int = Field(..., description="Number of embeddings created or associated with this job")
    paper_reused: bool = Field(False, description="Whether an existing paper record was reused")
    file_reused: bool = Field(False, description="Whether an existing paper file record was reused")
    job_reused: bool = Field(False, description="Whether an existing completed job was reused")
    status: str = Field(..., description="Final status of the ingestion job ('completed')")


class IngestionJobResponse(BaseModel):
    """Response returned for ingestion job status lookup."""

    model_config = ConfigDict(from_attributes=True)

    job_id: uuid.UUID = Field(..., description="UUID of the ingestion job")
    paper_id: uuid.UUID = Field(..., description="UUID of the associated paper")
    status: str = Field(..., description="Current status of the ingestion job")
    current_stage: str | None = Field(None, description="Current pipeline execution stage")
    attempt_count: int = Field(0, description="Number of execution attempts")
    error_message: str | None = Field(None, description="Error message if the job failed")
    created_at: datetime | None = Field(None, description="Timestamp when job was created")
    started_at: datetime | None = Field(None, description="Timestamp when job started processing")
    completed_at: datetime | None = Field(None, description="Timestamp when job concluded")
