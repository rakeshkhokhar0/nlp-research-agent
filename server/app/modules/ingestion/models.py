"""SQLAlchemy 2.x models for document ingestion jobs and paper chunks."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, CreatedTimestampMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.modules.conversation.models import MessageCitation
    from app.modules.papers.models import Paper, PaperFile
    from app.modules.retrieval.models import ChunkEmbedding

ACTIVE_JOB_STATUSES = frozenset({"pending", "processing"})
TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})
SUPPORTED_JOB_TYPES = frozenset({"extraction", "chunking", "embedding", "reprocessing", "cleanup"})


class IngestionJob(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Track document ingestion and processing lifecycle."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        UniqueConstraint("id", "paper_id", name="uq_ingestion_jobs_id_paper_id"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')",
            name="status",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="nonnegative_attempts",
        ),
        CheckConstraint(
            "completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at",
            name="completed_after_started",
        ),
        Index("ix_ingestion_jobs_paper_id_created_at", "paper_id", "created_at"),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    paper_file_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("paper_files.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    current_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )

    # Relationships
    paper: Mapped[Paper] = relationship("Paper", back_populates="ingestion_jobs")
    paper_file: Mapped[PaperFile | None] = relationship(
        "PaperFile",
        back_populates="ingestion_jobs",
    )
    chunks: Mapped[list[PaperChunk]] = relationship(
        "PaperChunk",
        back_populates="ingestion_job",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="[PaperChunk.ingestion_job_id, PaperChunk.paper_id]",
        overlaps="chunks,paper",
    )

    def __init__(self, **kwargs: Any) -> None:
        if "metadata" in kwargs and "metadata_" not in kwargs:
            kwargs["metadata_"] = kwargs.pop("metadata")
        super().__init__(**kwargs)


class PaperChunk(Base, UUIDPrimaryKeyMixin, CreatedTimestampMixin):
    """Store extracted, cleaned and divided research paper content."""

    __tablename__ = "paper_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["ingestion_job_id", "paper_id"],
            ["ingestion_jobs.id", "ingestion_jobs.paper_id"],
            name="fk_paper_chunks_job_paper_composite",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "chunk_index >= 0",
            name="nonnegative_index",
        ),
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="nonnegative_tokens",
        ),
        CheckConstraint(
            "char_count IS NULL OR char_count >= 0",
            name="nonnegative_chars",
        ),
        CheckConstraint(
            "page_start IS NULL OR page_start > 0",
            name="positive_page_start",
        ),
        CheckConstraint(
            "page_end IS NULL OR page_end > 0",
            name="positive_page_end",
        ),
        CheckConstraint(
            "page_start IS NULL OR page_end IS NULL OR page_end >= page_start",
            name="valid_page_range",
        ),
        CheckConstraint(
            "length(trim(content)) > 0",
            name="nonblank_content",
        ),
        Index(
            "ix_paper_chunks_paper_id_chunk_index_unique",
            "paper_id",
            "chunk_index",
            unique=True,
            postgresql_where=text("is_current = TRUE"),
        ),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ingestion_job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    chunking_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_current: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
        nullable=False,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )

    # Relationships
    paper: Mapped[Paper] = relationship(
        "Paper",
        back_populates="chunks",
        foreign_keys=[paper_id],
        overlaps="chunks,ingestion_job",
    )
    ingestion_job: Mapped[IngestionJob] = relationship(
        "IngestionJob",
        back_populates="chunks",
        foreign_keys=[ingestion_job_id, paper_id],
        overlaps="chunks,paper",
    )
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        "ChunkEmbedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    citations: Mapped[list[MessageCitation]] = relationship(
        "MessageCitation",
        back_populates="chunk",
        passive_deletes="all",
    )

    def __init__(self, **kwargs: Any) -> None:
        if "metadata" in kwargs and "metadata_" not in kwargs:
            kwargs["metadata_"] = kwargs.pop("metadata")
        kwargs.setdefault("is_current", False)
        super().__init__(**kwargs)
