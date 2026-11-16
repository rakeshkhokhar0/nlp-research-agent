"""Unit tests for SQLAlchemy 2.x declarative models and metadata.

Runs fully offline without needing live database or external network access.
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import configure_mappers

from app.core.exceptions import ActiveJobConflictError
from app.database.model_registry import (
    Base,
    ChunkEmbedding,
    Conversation,
    EmbeddingModel,
    IngestionJob,
    Message,
    MessageCitation,
    Paper,
    PaperChunk,
    PaperFile,
    PaperSource,
)
from app.modules.ingestion.models import (
    ACTIVE_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
)
from app.modules.ingestion.service import check_active_jobs_for_deletion
from app.modules.papers.models import normalize_doi
from app.modules.retrieval.models import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_NORMALIZE_EMBEDDINGS,
)


def test_metadata_registers_all_ten_tables():
    """Verify that all ten expected business tables are discovered by Base.metadata."""
    expected_tables = {
        "papers",
        "paper_sources",
        "paper_files",
        "ingestion_jobs",
        "paper_chunks",
        "embedding_models",
        "chunk_embeddings",
        "conversations",
        "messages",
        "message_citations",
    }
    registered_tables = set(Base.metadata.tables.keys())
    assert registered_tables == expected_tables, (
        f"Mismatch in registered tables. Extra: {registered_tables - expected_tables}, "
        f"Missing: {expected_tables - registered_tables}"
    )


def test_doi_normalization():
    """Verify DOI normalization handles URLs, whitespace, and case sensitivity."""
    assert normalize_doi(" 10.1145/3372278.3390678 ") == "10.1145/3372278.3390678"
    assert normalize_doi("https://doi.org/10.1000/182") == "10.1000/182"
    assert normalize_doi("http://dx.doi.org/10.1000/182") == "10.1000/182"
    assert normalize_doi("10.1000/ABC.DEF") == "10.1000/abc.def"
    assert normalize_doi(None) is None
    assert normalize_doi("") is None
    assert normalize_doi("   ") is None


def test_paper_model_instantiation_and_validation():
    """Verify Paper model can be instantiated with defaults and normalizes DOI."""
    paper = Paper(
        title="Attention Is All You Need",
        doi="https://doi.org/10.48550/arXiv.1706.03762",
        abstract="The dominant sequence transduction models are based on attention.",
        authors=[{"name": "Ashish Vaswani", "affiliation": "Google Brain"}],
        metadata_={"conference": "NeurIPS"},
        publication_year=2017,
        publication_date=date(2017, 6, 12),
    )
    assert paper.title == "Attention Is All You Need"
    assert paper.doi == "10.48550/arxiv.1706.03762"
    assert isinstance(paper.authors, list)
    assert paper.authors[0]["name"] == "Ashish Vaswani"
    assert isinstance(paper.metadata_, dict)
    assert paper.metadata_["conference"] == "NeurIPS"
    assert paper.publication_year == 2017


def test_uuid_primary_keys_across_all_tables():
    """Verify all ten tables have UUID primary keys with gen_random_uuid() server default."""
    for table_name, table in Base.metadata.tables.items():
        pk_cols = list(table.primary_key.columns)
        assert len(pk_cols) == 1, f"Table {table_name} does not have exactly 1 primary key column"
        pk = pk_cols[0]
        assert pk.name == "id", f"Table {table_name} PK column is not 'id'"
        assert pk.type.python_type is uuid.UUID, f"Table {table_name} PK is not UUID"


def test_timestamp_conventions_across_tables():
    """Verify timestamp conventions: created_at on all tables, updated_at on relevant tables."""
    tables_with_updated_at = {"papers", "paper_files", "ingestion_jobs", "conversations", "messages"}
    tables_created_at_only = {"paper_sources", "paper_chunks", "embedding_models", "chunk_embeddings", "message_citations"}

    for table_name in tables_with_updated_at:
        table = Base.metadata.tables[table_name]
        assert "created_at" in table.c, f"{table_name} missing created_at"
        assert "updated_at" in table.c, f"{table_name} missing updated_at"
        assert table.c.created_at.nullable is False
        assert table.c.updated_at.nullable is False
        assert table.c.updated_at.onupdate is not None, f"{table_name}.updated_at missing onupdate"
        assert callable(table.c.updated_at.default.arg)

    for table_name in tables_created_at_only:
        table = Base.metadata.tables[table_name]
        assert "created_at" in table.c, f"{table_name} missing created_at"
        assert table.c.created_at.nullable is False
        assert "updated_at" not in table.c, f"{table_name} unexpectedly has updated_at"


def test_partial_unique_indexes_and_predicates():
    """Verify partial unique indexes have the exact intended WHERE predicates."""
    papers_table = Base.metadata.tables["papers"]
    doi_idx = next(i for i in papers_table.indexes if i.name == "ix_papers_doi_unique")
    assert doi_idx.unique is True
    assert "doi IS NOT NULL" in str(doi_idx.dialect_options["postgresql"]["where"])

    sources_table = Base.metadata.tables["paper_sources"]
    src_idx = next(i for i in sources_table.indexes if i.name == "ix_paper_sources_source_name_external_id_unique")
    assert src_idx.unique is True
    assert "external_id IS NOT NULL" in str(src_idx.dialect_options["postgresql"]["where"])

    files_table = Base.metadata.tables["paper_files"]
    primary_file_idx = next(i for i in files_table.indexes if i.name == "ix_paper_files_paper_id_primary_unique")
    assert primary_file_idx.unique is True
    assert "is_primary = TRUE" in str(primary_file_idx.dialect_options["postgresql"]["where"])

    chunks_table = Base.metadata.tables["paper_chunks"]
    current_chunk_idx = next(i for i in chunks_table.indexes if i.name == "ix_paper_chunks_paper_id_chunk_index_unique")
    assert current_chunk_idx.unique is True
    assert "is_current = TRUE" in str(current_chunk_idx.dialect_options["postgresql"]["where"])

    # Correction 1: EmbeddingModel partial unique index for NULL versions
    models_table = Base.metadata.tables["embedding_models"]
    null_version_idx = next(
        (i for i in models_table.indexes if i.name == "ix_embedding_models_provider_model_null_version_unique"),
        None,
    )
    assert null_version_idx is not None, "Missing partial unique index for NULL model_version"
    assert null_version_idx.unique is True
    assert "model_version IS NULL" in str(null_version_idx.dialect_options["postgresql"]["where"])


def test_composite_foreign_key_and_absence_of_redundant_fk():
    """Verify composite FK enforces paper/job consistency and redundant single FK is absent (Correction 2)."""
    chunk_table = Base.metadata.tables["paper_chunks"]
    foreign_keys = chunk_table.foreign_key_constraints

    # Composite FK must exist
    composite_fk = next(
        (fk for fk in foreign_keys if fk.name == "fk_paper_chunks_job_paper_composite"),
        None,
    )
    assert composite_fk is not None, "Missing composite foreign key on paper_chunks"

    source_cols = [col.name for col in composite_fk.columns]
    target_cols = [elem.column.name for elem in composite_fk.elements]

    assert set(source_cols) == {"ingestion_job_id", "paper_id"}
    assert set(target_cols) == {"id", "paper_id"}
    assert composite_fk.ondelete == "CASCADE"

    # Redundant single-column FK on ingestion_job_id must be absent
    single_col_job_fk = next(
        (fk for fk in foreign_keys if [c.name for c in fk.columns] == ["ingestion_job_id"]),
        None,
    )
    assert single_col_job_fk is None, "Redundant single-column FK on ingestion_job_id must be absent"


def test_foreign_key_deletion_policies():
    """Verify all foreign key delete policies strictly adhere to architectural specification."""
    sources_table = Base.metadata.tables["paper_sources"]
    paper_fk = next(fk for fk in sources_table.foreign_key_constraints if "paper_id" in [c.name for c in fk.columns])
    assert paper_fk.ondelete == "CASCADE"

    files_table = Base.metadata.tables["paper_files"]
    paper_fk = next(fk for fk in files_table.foreign_key_constraints if "paper_id" in [c.name for c in fk.columns])
    assert paper_fk.ondelete == "CASCADE"

    jobs_table = Base.metadata.tables["ingestion_jobs"]
    file_fk = next(fk for fk in jobs_table.foreign_key_constraints if "paper_file_id" in [c.name for c in fk.columns])
    assert file_fk.ondelete == "SET NULL"
    assert jobs_table.c.paper_file_id.nullable is True

    paper_fk = next(fk for fk in jobs_table.foreign_key_constraints if "paper_id" in [c.name for c in fk.columns])
    assert paper_fk.ondelete == "CASCADE"

    chunks_table = Base.metadata.tables["paper_chunks"]
    paper_chunk_fk = next(fk for fk in chunks_table.foreign_key_constraints if [c.name for c in fk.columns] == ["paper_id"])
    assert paper_chunk_fk.ondelete == "CASCADE"

    embeddings_table = Base.metadata.tables["chunk_embeddings"]
    chunk_fk = next(fk for fk in embeddings_table.foreign_key_constraints if "chunk_id" in [c.name for c in fk.columns])
    assert chunk_fk.ondelete == "CASCADE"

    model_fk = next(fk for fk in embeddings_table.foreign_key_constraints if "embedding_model_id" in [c.name for c in fk.columns])
    assert model_fk.ondelete == "RESTRICT"

    msgs_table = Base.metadata.tables["messages"]
    conv_fk = next(fk for fk in msgs_table.foreign_key_constraints if "conversation_id" in [c.name for c in fk.columns])
    assert conv_fk.ondelete == "CASCADE"

    citations_table = Base.metadata.tables["message_citations"]
    msg_fk = next(fk for fk in citations_table.foreign_key_constraints if "message_id" in [c.name for c in fk.columns])
    assert msg_fk.ondelete == "CASCADE"

    cit_chunk_fk = next(fk for fk in citations_table.foreign_key_constraints if "chunk_id" in [c.name for c in fk.columns])
    assert cit_chunk_fk.ondelete == "RESTRICT"


def test_unique_constraints_presence():
    """Verify explicit unique constraints on required column combinations."""
    files_table = Base.metadata.tables["paper_files"]
    file_uqs = {c.name: [col.name for col in c.columns] for c in files_table.constraints if isinstance(c, UniqueConstraint)}
    assert ["storage_provider", "storage_key"] in file_uqs.values()

    jobs_table = Base.metadata.tables["ingestion_jobs"]
    job_uqs = {c.name: [col.name for col in c.columns] for c in jobs_table.constraints if isinstance(c, UniqueConstraint)}
    assert ["id", "paper_id"] in job_uqs.values()

    models_table = Base.metadata.tables["embedding_models"]
    model_uqs = {c.name: [col.name for col in c.columns] for c in models_table.constraints if isinstance(c, UniqueConstraint)}
    assert ["provider", "model_name", "model_version"] in model_uqs.values()

    embeddings_table = Base.metadata.tables["chunk_embeddings"]
    emb_uqs = {c.name: [col.name for col in c.columns] for c in embeddings_table.constraints if isinstance(c, UniqueConstraint)}
    assert ["chunk_id", "embedding_model_id"] in emb_uqs.values()

    msgs_table = Base.metadata.tables["messages"]
    msg_uqs = {c.name: [col.name for col in c.columns] for c in msgs_table.constraints if isinstance(c, UniqueConstraint)}
    assert ["conversation_id", "sequence_number"] in msg_uqs.values()

    citations_table = Base.metadata.tables["message_citations"]
    cit_uqs = {c.name: [col.name for col in c.columns] for c in citations_table.constraints if isinstance(c, UniqueConstraint)}
    assert ["message_id", "chunk_id"] in cit_uqs.values()


def test_domain_check_constraints():
    """Verify check constraints across all models."""
    papers_table = Base.metadata.tables["papers"]
    paper_cks = {c.name: str(c.sqltext) for c in papers_table.constraints if hasattr(c, "sqltext")}
    assert "length(trim(title)) > 0" in paper_cks["ck_papers_nonblank_title"]

    chunks_table = Base.metadata.tables["paper_chunks"]
    chunk_cks = {c.name: str(c.sqltext) for c in chunks_table.constraints if hasattr(c, "sqltext")}
    assert "chunk_index >= 0" in chunk_cks["ck_paper_chunks_nonnegative_index"]
    assert "page_start > 0" in chunk_cks["ck_paper_chunks_positive_page_start"]
    assert "page_end > 0" in chunk_cks["ck_paper_chunks_positive_page_end"]
    assert "page_end >= page_start" in chunk_cks["ck_paper_chunks_valid_page_range"]
    assert "token_count >= 0" in chunk_cks["ck_paper_chunks_nonnegative_tokens"]
    assert "char_count >= 0" in chunk_cks["ck_paper_chunks_nonnegative_chars"]
    assert "length(trim(content)) > 0" in chunk_cks["ck_paper_chunks_nonblank_content"]

    files_table = Base.metadata.tables["paper_files"]
    file_cks = {c.name: str(c.sqltext) for c in files_table.constraints if hasattr(c, "sqltext")}
    assert "status IN" in file_cks["ck_paper_files_status"]
    assert "file_size_bytes > 0" in file_cks["ck_paper_files_positive_file_size"]

    jobs_table = Base.metadata.tables["ingestion_jobs"]
    job_cks = {c.name: str(c.sqltext) for c in jobs_table.constraints if hasattr(c, "sqltext")}
    assert "attempt_count >= 0" in job_cks["ck_ingestion_jobs_nonnegative_attempts"]
    assert "status IN" in job_cks["ck_ingestion_jobs_status"]
    assert "completed_at >= started_at" in job_cks["ck_ingestion_jobs_completed_after_started"]

    models_table = Base.metadata.tables["embedding_models"]
    model_cks = {c.name: str(c.sqltext) for c in models_table.constraints if hasattr(c, "sqltext")}
    assert "dimensions > 0" in model_cks["ck_embedding_models_positive_dimensions"]
    assert "dimensions = 384" in model_cks["ck_embedding_models_dimensions_384"]
    assert "distance_metric = 'cosine'" in model_cks["ck_embedding_models_valid_distance_metric"]

    convs_table = Base.metadata.tables["conversations"]
    conv_cks = {c.name: str(c.sqltext) for c in convs_table.constraints if hasattr(c, "sqltext")}
    assert "status IN ('active', 'archived')" in conv_cks["ck_conversations_status"]

    msgs_table = Base.metadata.tables["messages"]
    msg_cks = {c.name: str(c.sqltext) for c in msgs_table.constraints if hasattr(c, "sqltext")}
    assert "role IN ('user', 'assistant', 'system')" in msg_cks["ck_messages_role"]
    assert "sequence_number > 0" in msg_cks["ck_messages_positive_sequence_number"]
    assert "status IN" in msg_cks["ck_messages_status"]

    cit_table = Base.metadata.tables["message_citations"]
    cit_cks = {c.name: str(c.sqltext) for c in cit_table.constraints if hasattr(c, "sqltext")}
    assert "citation_order > 0" in cit_cks["ck_message_citations_positive_citation_order"]
    assert "page_number > 0" in cit_cks["ck_message_citations_positive_page_number"]


def test_embedding_model_constants_and_vector_hnsw():
    """Verify default embedding model constants, 384 dimensions, and HNSW index configuration."""
    assert DEFAULT_EMBEDDING_MODEL_ID == uuid.UUID("c1d48c8b-6c61-55bb-8418-2e86f8a55c27")
    assert DEFAULT_EMBEDDING_PROVIDER == "sentence-transformers"
    assert DEFAULT_EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
    assert DEFAULT_EMBEDDING_DIMENSIONS == 384
    assert DEFAULT_DISTANCE_METRIC == "cosine"
    assert DEFAULT_NORMALIZE_EMBEDDINGS is True

    embeddings_table = Base.metadata.tables["chunk_embeddings"]
    vector_col = embeddings_table.c.embedding
    assert vector_col.type.dim == 384

    hnsw_index = next(
        (idx for idx in embeddings_table.indexes if idx.name == "ix_chunk_embeddings_hnsw"),
        None,
    )
    assert hnsw_index is not None
    assert hnsw_index.dialect_options["postgresql"]["using"] == "hnsw"
    assert hnsw_index.dialect_options["postgresql"]["ops"]["embedding"] == "vector_cosine_ops"


def test_orm_relationships_and_bidirectional_navigation():
    """Verify mapper configuration and relationship navigation without overlapping key warnings."""
    configure_mappers()

    paper = Paper(title="Transformer Architecture")
    paper_file = PaperFile(
        paper=paper,
        storage_provider="cloudinary",
        storage_key="papers/transformer.pdf",
        original_filename="transformer.pdf",
        mime_type="application/pdf",
        status="available",
    )
    job = IngestionJob(paper=paper, paper_file=paper_file, status="completed")
    chunk = PaperChunk(
        paper=paper,
        ingestion_job=job,
        chunk_index=0,
        content="Encoder-Decoder Attention Mechanism",
    )
    conv = Conversation(title="NLP Architecture Discussion")
    msg = Message(
        conversation=conv,
        sequence_number=1,
        role="assistant",
        content="The transformer relies entirely on an attention mechanism.",
    )
    citation = MessageCitation(
        message=msg,
        chunk=chunk,
        citation_order=1,
        cited_text="Encoder-Decoder Attention Mechanism",
    )

    assert paper_file in paper.files
    assert job in paper.ingestion_jobs
    assert chunk in paper.chunks
    assert chunk in job.chunks
    assert chunk.paper is paper
    assert chunk.ingestion_job is job
    assert msg in conv.messages
    assert citation in msg.citations
    assert citation.chunk is chunk
    assert citation.quote_text == "Encoder-Decoder Attention Mechanism"


def test_active_jobs_deletion_guard():
    """Verify check_active_jobs_for_deletion blocks deletion when jobs are active."""
    file_id = uuid.uuid4()
    active_job = IngestionJob(
        id=uuid.uuid4(),
        paper_id=uuid.uuid4(),
        paper_file_id=file_id,
        status="processing",
    )
    assert active_job.status in ACTIVE_JOB_STATUSES

    raised = False
    try:
        check_active_jobs_for_deletion(
            target_type="PaperFile",
            target_id=file_id,
            active_jobs=[active_job],
        )
    except ActiveJobConflictError as exc:
        raised = True
        assert "active job(s) in progress" in str(exc)
    assert raised

    # Terminal jobs must not block deletion
    completed_job = IngestionJob(
        id=uuid.uuid4(),
        paper_id=uuid.uuid4(),
        paper_file_id=file_id,
        status="completed",
    )
    assert completed_job.status in TERMINAL_JOB_STATUSES
    # Must not raise
    check_active_jobs_for_deletion(
        target_type="PaperFile",
        target_id=file_id,
        active_jobs=[completed_job],
    )


def test_downgrade_sql_does_not_drop_vector_extension():
    """Verify that Alembic downgrade does not drop the pgvector extension (Correction 6)."""
    migration_file = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "a7f91e11ba51_create_initial_schema.py"
    content = migration_file.read_text()
    assert "DROP EXTENSION" not in content, "Downgrade must not drop the vector extension"
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in content, "Upgrade must ensure vector extension exists"


def test_paper_chunk_default_is_current_false():
    """Verify PaperChunk defaults to is_current=False in Python and server default is FALSE."""
    chunk = PaperChunk(
        paper_id=uuid.uuid4(),
        ingestion_job_id=uuid.uuid4(),
        chunk_index=0,
        content="Test content",
    )
    assert chunk.is_current is False

    chunks_table = Base.metadata.tables["paper_chunks"]
    is_current_col = chunks_table.c.is_current
    assert is_current_col.default.arg is False
    assert "false" in str(is_current_col.server_default.arg).lower()


def test_embedding_model_v1_constraints_metadata():
    """Verify embedding_models table metadata restricts dimensions to 384 and metric to cosine."""
    models_table = Base.metadata.tables["embedding_models"]
    model_cks = {c.name: str(c.sqltext) for c in models_table.constraints if hasattr(c, "sqltext")}
    assert "ck_embedding_models_dimensions_384" in model_cks
    assert "dimensions = 384" in model_cks["ck_embedding_models_dimensions_384"]
    assert "ck_embedding_models_valid_distance_metric" in model_cks
    assert "distance_metric = 'cosine'" in model_cks["ck_embedding_models_valid_distance_metric"]


