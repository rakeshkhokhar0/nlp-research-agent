"""Create complete initial schema for NLP Research Agent V1.

Revision ID: a7f91e11ba51
Revises: 
Create Date: 2026-11-12 05:22:34.308932

"""
from typing import Sequence, Union

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a7f91e11ba51'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema: enable pgvector, create 10 core tables in dependency order."""
    # 0. Ensure pgvector extension exists
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 1. papers
    op.create_table(
        "papers",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("doi", sa.String(length=255), nullable=True),
        sa.Column("publication_date", sa.Date(), nullable=True),
        sa.Column("publication_year", sa.SmallInteger(), nullable=True),
        sa.Column("venue", sa.String(length=500), nullable=True),
        sa.Column("publisher", sa.String(length=255), nullable=True),
        sa.Column("authors", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("language", sa.String(length=20), nullable=True),
        sa.Column("paper_type", sa.String(length=50), nullable=True),
        sa.Column("canonical_url", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("length(trim(title)) > 0", name="nonblank_title"),
        sa.PrimaryKeyConstraint("id", name="pk_papers"),
    )
    op.create_index("ix_papers_publication_year", "papers", ["publication_year"], unique=False)
    op.create_index("ix_papers_publication_date", "papers", ["publication_date"], unique=False)
    op.create_index(
        "ix_papers_doi_unique",
        "papers",
        ["doi"],
        unique=True,
        postgresql_where=sa.text("doi IS NOT NULL"),
    )

    # 2. paper_sources
    op.create_table(
        "paper_sources",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("paper_id", sa.UUID(), nullable=False),
        sa.Column("source_name", sa.String(length=50), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("raw_metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], name="fk_paper_sources_paper_id_papers", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_paper_sources"),
    )
    op.create_index("ix_paper_sources_paper_id", "paper_sources", ["paper_id"], unique=False)
    op.create_index("ix_paper_sources_source_name", "paper_sources", ["source_name"], unique=False)
    op.create_index(
        "ix_paper_sources_source_name_external_id_unique",
        "paper_sources",
        ["source_name", "external_id"],
        unique=True,
        postgresql_where=sa.text("external_id IS NOT NULL"),
    )

    # 3. paper_files
    op.create_table(
        "paper_files",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("paper_id", sa.UUID(), nullable=False),
        sa.Column("storage_provider", sa.String(length=50), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("file_url", sa.Text(), nullable=True),
        sa.Column("original_filename", sa.String(length=500), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
        sa.Column("is_primary", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("file_size_bytes IS NULL OR file_size_bytes > 0", name="positive_file_size"),
        sa.CheckConstraint("status IN ('uploaded', 'available', 'failed', 'deleted')", name="status"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], name="fk_paper_files_paper_id_papers", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_paper_files"),
        sa.UniqueConstraint("storage_provider", "storage_key", name="uq_paper_files_storage_provider_key"),
    )
    op.create_index("ix_paper_files_paper_id", "paper_files", ["paper_id"], unique=False)
    op.create_index(
        "ix_paper_files_paper_id_primary_unique",
        "paper_files",
        ["paper_id"],
        unique=True,
        postgresql_where=sa.text("is_primary = TRUE"),
    )

    # 4. ingestion_jobs
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("paper_id", sa.UUID(), nullable=False),
        sa.Column("paper_file_id", sa.UUID(), nullable=True),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("current_stage", sa.String(length=50), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("pipeline_version", sa.String(length=50), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("attempt_count >= 0", name="nonnegative_attempts"),
        sa.CheckConstraint("completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at", name="completed_after_started"),
        sa.CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')", name="status"),
        sa.ForeignKeyConstraint(["paper_file_id"], ["paper_files.id"], name="fk_ingestion_jobs_paper_file_id_paper_files", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], name="fk_ingestion_jobs_paper_id_papers", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_ingestion_jobs"),
        sa.UniqueConstraint("id", "paper_id", name="uq_ingestion_jobs_id_paper_id"),
    )
    op.create_index("ix_ingestion_jobs_paper_id", "ingestion_jobs", ["paper_id"], unique=False)
    op.create_index("ix_ingestion_jobs_paper_file_id", "ingestion_jobs", ["paper_file_id"], unique=False)
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"], unique=False)
    op.create_index("ix_ingestion_jobs_paper_id_created_at", "ingestion_jobs", ["paper_id", "created_at"], unique=False)

    # 5. paper_chunks
    op.create_table(
        "paper_chunks",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("paper_id", sa.UUID(), nullable=False),
        sa.Column("ingestion_job_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("section_title", sa.String(length=500), nullable=True),
        sa.Column("chunking_version", sa.String(length=50), nullable=True),
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("char_count IS NULL OR char_count >= 0", name="nonnegative_chars"),
        sa.CheckConstraint("chunk_index >= 0", name="nonnegative_index"),
        sa.CheckConstraint("length(trim(content)) > 0", name="nonblank_content"),
        sa.CheckConstraint("page_end IS NULL OR page_end > 0", name="positive_page_end"),
        sa.CheckConstraint("page_start IS NULL OR page_start > 0", name="positive_page_start"),
        sa.CheckConstraint("page_start IS NULL OR page_end IS NULL OR page_end >= page_start", name="valid_page_range"),
        sa.CheckConstraint("token_count IS NULL OR token_count >= 0", name="nonnegative_tokens"),
        sa.ForeignKeyConstraint(["ingestion_job_id", "paper_id"], ["ingestion_jobs.id", "ingestion_jobs.paper_id"], name="fk_paper_chunks_job_paper_composite", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], name="fk_paper_chunks_paper_id_papers", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_paper_chunks"),
    )
    op.create_index("ix_paper_chunks_content_hash", "paper_chunks", ["content_hash"], unique=False)
    op.create_index("ix_paper_chunks_ingestion_job_id", "paper_chunks", ["ingestion_job_id"], unique=False)
    op.create_index("ix_paper_chunks_paper_id", "paper_chunks", ["paper_id"], unique=False)
    op.create_index(
        "ix_paper_chunks_paper_id_chunk_index_unique",
        "paper_chunks",
        ["paper_id", "chunk_index"],
        unique=True,
        postgresql_where=sa.text("is_current = TRUE"),
    )

    # 6. embedding_models
    op.create_table(
        "embedding_models",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=True),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("normalize_embeddings", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("distance_metric", sa.String(length=30), nullable=False),
        sa.Column("configuration", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("dimensions > 0", name="positive_dimensions"),
        sa.CheckConstraint("dimensions = 384", name="dimensions_384"),
        sa.CheckConstraint("distance_metric = 'cosine'", name="valid_distance_metric"),
        sa.PrimaryKeyConstraint("id", name="pk_embedding_models"),
        sa.UniqueConstraint("provider", "model_name", "model_version", name="uq_embedding_models_provider_model_version"),
    )
    op.create_index("ix_embedding_models_provider_model_name", "embedding_models", ["provider", "model_name"], unique=False)
    op.create_index(
        "ix_embedding_models_provider_model_null_version_unique",
        "embedding_models",
        ["provider", "model_name"],
        unique=True,
        postgresql_where=sa.text("model_version IS NULL"),
    )

    # 7. chunk_embeddings
    op.create_table(
        "chunk_embeddings",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("chunk_id", sa.UUID(), nullable=False),
        sa.Column("embedding_model_id", sa.UUID(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(384), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["paper_chunks.id"], name="fk_chunk_embeddings_chunk_id_paper_chunks", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["embedding_model_id"], ["embedding_models.id"], name="fk_chunk_embeddings_embedding_model_id_embedding_models", ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name="pk_chunk_embeddings"),
        sa.UniqueConstraint("chunk_id", "embedding_model_id", name="uq_chunk_embeddings_chunk_model"),
    )
    op.create_index("ix_chunk_embeddings_chunk_id", "chunk_embeddings", ["chunk_id"], unique=False)
    op.create_index("ix_chunk_embeddings_embedding_model_id", "chunk_embeddings", ["embedding_model_id"], unique=False)
    op.create_index(
        "ix_chunk_embeddings_hnsw",
        "chunk_embeddings",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    # 8. conversations
    op.create_table(
        "conversations",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=30), server_default=sa.text("'active'"), nullable=False),
        sa.Column("context_summary", sa.Text(), nullable=True),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("status IN ('active', 'archived')", name="status"),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
    )
    op.create_index("ix_conversations_created_at", "conversations", ["created_at"], unique=False)
    op.create_index("ix_conversations_last_message_at", "conversations", ["last_message_at"], unique=False)

    # 9. messages
    op.create_table(
        "messages",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=30), server_default=sa.text("'completed'"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("role IN ('user', 'assistant', 'system')", name="role"),
        sa.CheckConstraint("sequence_number > 0", name="positive_sequence_number"),
        sa.CheckConstraint("status IN ('pending', 'streaming', 'completed', 'failed')", name="status"),
        sa.CheckConstraint("token_count IS NULL OR token_count >= 0", name="nonnegative_tokens"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], name="fk_messages_conversation_id_conversations", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_messages"),
        sa.UniqueConstraint("conversation_id", "sequence_number", name="uq_messages_conversation_sequence"),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"], unique=False)
    op.create_index("ix_messages_conversation_id_created_at", "messages", ["conversation_id", "created_at"], unique=False)

    # 10. message_citations
    op.create_table(
        "message_citations",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("message_id", sa.UUID(), nullable=False),
        sa.Column("chunk_id", sa.UUID(), nullable=False),
        sa.Column("citation_order", sa.Integer(), nullable=False),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.Column("cited_text", sa.Text(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("citation_order > 0", name="positive_citation_order"),
        sa.CheckConstraint("page_number IS NULL OR page_number > 0", name="positive_page_number"),
        sa.ForeignKeyConstraint(["chunk_id"], ["paper_chunks.id"], name="fk_message_citations_chunk_id_paper_chunks", ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], name="fk_message_citations_message_id_messages", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name="pk_message_citations"),
        sa.UniqueConstraint("message_id", "chunk_id", name="uq_message_citations_message_chunk"),
    )
    op.create_index("ix_message_citations_chunk_id", "message_citations", ["chunk_id"], unique=False)
    op.create_index("ix_message_citations_message_id", "message_citations", ["message_id"], unique=False)

    # 11. Idempotent seed data for initial embedding model
    op.execute(
        """
        INSERT INTO embedding_models (
            id, provider, model_name, model_version, dimensions,
            normalize_embeddings, distance_metric, configuration, is_active, created_at
        )
        SELECT
            'c1d48c8b-6c61-55bb-8418-2e86f8a55c27',
            'sentence-transformers',
            'sentence-transformers/all-MiniLM-L6-v2',
            NULL,
            384,
            TRUE,
            'cosine',
            '{}'::jsonb,
            TRUE,
            NOW()
        WHERE NOT EXISTS (
            SELECT 1 FROM embedding_models
            WHERE id = 'c1d48c8b-6c61-55bb-8418-2e86f8a55c27'
               OR (provider = 'sentence-transformers' AND model_name = 'sentence-transformers/all-MiniLM-L6-v2' AND model_version IS NULL)
        );
        """
    )


def downgrade() -> None:
    """Downgrade schema in reverse dependency order."""
    op.drop_table("message_citations")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("chunk_embeddings")
    op.drop_table("embedding_models")
    op.drop_table("paper_chunks")
    op.drop_table("ingestion_jobs")
    op.drop_table("paper_files")
    op.drop_table("paper_sources")
    op.drop_table("papers")
