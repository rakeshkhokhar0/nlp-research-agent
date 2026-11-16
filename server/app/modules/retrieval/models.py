"""SQLAlchemy 2.x models for embedding models and vector chunk representations."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, CreatedTimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.modules.ingestion.models import PaperChunk

DEFAULT_EMBEDDING_MODEL_ID = uuid.UUID("c1d48c8b-6c61-55bb-8418-2e86f8a55c27")
DEFAULT_EMBEDDING_PROVIDER = "sentence-transformers"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSIONS = 384
DEFAULT_DISTANCE_METRIC = "cosine"
DEFAULT_NORMALIZE_EMBEDDINGS = True


class EmbeddingModel(Base, UUIDPrimaryKeyMixin, CreatedTimestampMixin):
    """Maintain embedding model configurations."""

    __tablename__ = "embedding_models"
    __table_args__ = (
        CheckConstraint(
            "dimensions > 0",
            name="positive_dimensions",
        ),
        CheckConstraint(
            "dimensions = 384",
            name="dimensions_384",
        ),
        CheckConstraint(
            "distance_metric = 'cosine'",
            name="valid_distance_metric",
        ),
        Index("ix_embedding_models_provider_model_name", "provider", "model_name"),
        UniqueConstraint(
            "provider",
            "model_name",
            "model_version",
            name="uq_embedding_models_provider_model_version",
        ),
        Index(
            "ix_embedding_models_provider_model_null_version_unique",
            "provider",
            "model_name",
            unique=True,
            postgresql_where=text("model_version IS NULL"),
        ),
    )

    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    normalize_embeddings: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
    )
    distance_metric: Mapped[str] = mapped_column(String(30), nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text("true"),
        nullable=False,
    )

    # Relationships
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        "ChunkEmbedding",
        back_populates="embedding_model",
    )


class ChunkEmbedding(Base, UUIDPrimaryKeyMixin, CreatedTimestampMixin):
    """Store vector representations of paper chunks."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "embedding_model_id",
            name="uq_chunk_embeddings_chunk_model",
        ),
        Index(
            "ix_chunk_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("paper_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding_model_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("embedding_models.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(DEFAULT_EMBEDDING_DIMENSIONS),
        nullable=False,
    )

    # Relationships
    chunk: Mapped[PaperChunk] = relationship(
        "PaperChunk",
        back_populates="embeddings",
    )
    embedding_model: Mapped[EmbeddingModel] = relationship(
        "EmbeddingModel",
        back_populates="embeddings",
    )
