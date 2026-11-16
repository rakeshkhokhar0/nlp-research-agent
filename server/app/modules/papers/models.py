"""SQLAlchemy 2.x models for research papers, sources, and document files."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.database.base import Base, CreatedTimestampMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.modules.ingestion.models import IngestionJob, PaperChunk


def normalize_doi(doi: str | None) -> str | None:
    """Normalize DOI string by stripping surrounding whitespace, URL prefixes, and lowercasing."""
    if doi is None:
        return None
    cleaned = doi.strip()
    if not cleaned:
        return None
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    if not cleaned:
        return None
    return cleaned.lower()


class Paper(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Main research paper entity."""

    __tablename__ = "papers"
    __table_args__ = (
        CheckConstraint("length(trim(title)) > 0", name="nonblank_title"),
        Index(
            "ix_papers_doi_unique",
            "doi",
            unique=True,
            postgresql_where=text("doi IS NOT NULL"),
        ),
        Index("ix_papers_publication_year", "publication_year"),
        Index("ix_papers_publication_date", "publication_date"),
    )

    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    publication_year: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(500), nullable=True)
    publisher: Mapped[str | None] = mapped_column(String(255), nullable=True)
    authors: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        server_default=text("'[]'::jsonb"),
        nullable=False,
    )
    language: Mapped[str | None] = mapped_column(String(20), nullable=True)
    paper_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    canonical_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )

    # Relationships
    sources: Mapped[list[PaperSource]] = relationship(
        "PaperSource",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    files: Mapped[list[PaperFile]] = relationship(
        "PaperFile",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        "IngestionJob",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    chunks: Mapped[list[PaperChunk]] = relationship(
        "PaperChunk",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="[PaperChunk.paper_id]",
    )

    def __init__(self, **kwargs: Any) -> None:
        if "metadata" in kwargs and "metadata_" not in kwargs:
            kwargs["metadata_"] = kwargs.pop("metadata")
        super().__init__(**kwargs)

    @validates("doi")
    def validate_doi(self, key: str, value: str | None) -> str | None:
        return normalize_doi(value)


class PaperSource(Base, UUIDPrimaryKeyMixin, CreatedTimestampMixin):
    """Track external discovery sources for research papers."""

    __tablename__ = "paper_sources"
    __table_args__ = (
        Index(
            "ix_paper_sources_source_name_external_id_unique",
            "source_name",
            "external_id",
            unique=True,
            postgresql_where=text("external_id IS NOT NULL"),
        ),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    paper: Mapped[Paper] = relationship("Paper", back_populates="sources")


class PaperFile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Metadata and external storage references for research paper files."""

    __tablename__ = "paper_files"
    __table_args__ = (
        UniqueConstraint(
            "storage_provider",
            "storage_key",
            name="uq_paper_files_storage_provider_key",
        ),
        Index(
            "ix_paper_files_paper_id_primary_unique",
            "paper_id",
            unique=True,
            postgresql_where=text("is_primary = TRUE"),
        ),
        CheckConstraint(
            "file_size_bytes IS NULL OR file_size_bytes > 0",
            name="positive_file_size",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'available', 'failed', 'deleted')",
            name="status",
        ),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    storage_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    file_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=text("false"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False)

    # Relationships
    paper: Mapped[Paper] = relationship("Paper", back_populates="files")
    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        "IngestionJob",
        back_populates="paper_file",
    )
