"""Integration tests for live PostgreSQL database with pgvector.

Requires explicit opt-in via TEST_DATABASE_URL environment variable.
Never runs silently against production Neon database.
Tests purely database constraints, ORM mappings, indexes, and pgvector operations.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import pytest
import pytest_asyncio
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Load .env if present
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

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
from app.modules.retrieval.models import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_EMBEDDING_PROVIDER,
)
from app.modules.ingestion.repository import IngestionRepository

TEST_DB_URL = (
    os.getenv("test_db_url")
    or os.getenv("TEST_DB_URL")
    or os.getenv("test_database_url")
    or os.getenv("TEST_DATABASE_URL")
)

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="TEST_DB_URL or TEST_DATABASE_URL environment variable not provided. Skipping live integration tests.",
)


@pytest_asyncio.fixture
async def db_engine():
    """Create async engine for the test database and ensure schema is present."""
    assert TEST_DB_URL is not None
    engine = create_async_engine(TEST_DB_URL, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(db_engine):
    """Return async sessionmaker for database sessions."""
    return async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def db_session(session_factory):
    """Provide a transactional database session for each test."""
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.mark.asyncio
async def test_live_postgres_vector_and_schema(db_session: AsyncSession):
    """Verify live PostgreSQL supports pgvector extension and vector operations."""
    res = await db_session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
    ext = res.scalar()
    assert ext == "vector", "pgvector extension is not enabled in test database"

    vec_res = await db_session.execute(text("SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector AS cosine_dist"))
    dist = vec_res.scalar()
    assert dist is not None
    assert float(dist) >= 0.0


@pytest.mark.asyncio
async def test_paper_lifecycle_and_relationships(db_session: AsyncSession):
    """Integration test: Create paper, sources, files, jobs, chunks, embeddings, conversations, citations."""
    # Ensure default embedding model is present in DB
    model_res = await db_session.execute(
        select(EmbeddingModel).where(EmbeddingModel.id == DEFAULT_EMBEDDING_MODEL_ID)
    )
    embedding_model = model_res.scalar_one_or_none()
    if not embedding_model:
        embedding_model = EmbeddingModel(
            id=DEFAULT_EMBEDDING_MODEL_ID,
            provider=DEFAULT_EMBEDDING_PROVIDER,
            model_name=DEFAULT_EMBEDDING_MODEL_NAME,
            model_version=None,
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            distance_metric=DEFAULT_DISTANCE_METRIC,
            is_active=True,
        )
        db_session.add(embedding_model)
        await db_session.flush()

    assert embedding_model.dimensions == DEFAULT_EMBEDDING_DIMENSIONS

    paper = Paper(
        title="Attention Is All You Need",
        doi=f"10.48550/arxiv.{uuid.uuid4()}",
        abstract="The dominant sequence transduction models are based on attention.",
        authors=[{"name": "Ashish Vaswani", "affiliation": "Google Brain"}],
        metadata_={"conference": "NeurIPS"},
        publication_year=2017,
        publication_date=date(2017, 6, 12),
    )
    db_session.add(paper)
    await db_session.flush()

    source_arxiv = PaperSource(
        paper_id=paper.id,
        source_name="arxiv",
        external_id=f"1706.{uuid.uuid4()}",
        source_url="https://arxiv.org/abs/1706.03762",
        raw_metadata={"primary_category": "cs.CL"},
    )
    db_session.add(source_arxiv)

    file_primary = PaperFile(
        paper_id=paper.id,
        storage_provider="cloudinary",
        storage_key=f"papers/{paper.id}/original.pdf",
        original_filename="1706.03762.pdf",
        mime_type="application/pdf",
        file_size_bytes=2215248,
        is_primary=True,
        status="available",
    )
    db_session.add(file_primary)
    await db_session.flush()

    job = IngestionJob(
        paper_id=paper.id,
        paper_file_id=file_primary.id,
        status="completed",
        attempt_count=1,
    )
    db_session.add(job)
    await db_session.flush()

    chunk_0 = PaperChunk(
        paper_id=paper.id,
        ingestion_job_id=job.id,
        chunk_index=0,
        content="The Transformer is the first transduction model relying entirely on self-attention.",
        is_current=True,
    )
    db_session.add(chunk_0)
    await db_session.flush()

    dummy_vector = [0.0] * DEFAULT_EMBEDDING_DIMENSIONS
    dummy_vector[0] = 1.0
    embedding = ChunkEmbedding(
        chunk_id=chunk_0.id,
        embedding_model_id=embedding_model.id,
        embedding=dummy_vector,
    )
    db_session.add(embedding)
    await db_session.flush()

    conversation = Conversation(title="Transformer Q&A", status="active")
    db_session.add(conversation)
    await db_session.flush()

    msg = Message(
        conversation_id=conversation.id,
        sequence_number=1,
        role="assistant",
        content="The Transformer relies entirely on self-attention without recurrent layers.",
    )
    db_session.add(msg)
    await db_session.flush()

    citation = MessageCitation(
        message_id=msg.id,
        chunk_id=chunk_0.id,
        citation_order=1,
        cited_text=chunk_0.content,
    )
    db_session.add(citation)
    await db_session.commit()

    # Verify citation resolves to chunk
    cit_loaded = (
        await db_session.execute(
            select(MessageCitation).where(MessageCitation.id == citation.id)
        )
    ).scalar_one()
    assert cit_loaded.chunk.content == chunk_0.content


@pytest.mark.asyncio
async def test_integrity_embedding_model_null_version_uniqueness(db_session: AsyncSession):
    """Verify multiple embedding models with NULL versions cannot be duplicated."""
    p_name = f"provider-{uuid.uuid4()}"
    m_name = "test-model"

    m1 = EmbeddingModel(
        provider=p_name,
        model_name=m_name,
        model_version=None,
        dimensions=384,
        distance_metric="cosine",
    )
    db_session.add(m1)
    await db_session.commit()

    # Second model with exact same provider and name, also NULL version -> must raise IntegrityError
    m2 = EmbeddingModel(
        provider=p_name,
        model_name=m_name,
        model_version=None,
        dimensions=384,
        distance_metric="cosine",
    )
    db_session.add(m2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_embedding_models_non_null_versions_coexist(db_session: AsyncSession):
    """Verify models with distinct non-NULL versions can coexist."""
    p_name = f"provider-{uuid.uuid4()}"
    m_name = "versioned-model"

    m1 = EmbeddingModel(
        provider=p_name,
        model_name=m_name,
        model_version="v1.0",
        dimensions=384,
        distance_metric="cosine",
    )
    m2 = EmbeddingModel(
        provider=p_name,
        model_name=m_name,
        model_version="v2.0",
        dimensions=384,
        distance_metric="cosine",
    )
    db_session.add_all([m1, m2])
    await db_session.commit()

    res = await db_session.execute(
        select(EmbeddingModel).where(EmbeddingModel.provider == p_name)
    )
    models = list(res.scalars().all())
    assert len(models) == 2


@pytest.mark.asyncio
async def test_composite_foreign_key_enforces_paper_job_consistency(db_session: AsyncSession):
    """Verify composite foreign key rejects chunk if paper_id does not match job.paper_id."""
    paper_a = Paper(title="Paper A")
    paper_b = Paper(title="Paper B")
    db_session.add_all([paper_a, paper_b])
    await db_session.flush()

    job_a = IngestionJob(paper_id=paper_a.id, status="completed")
    db_session.add(job_a)
    await db_session.flush()

    # Mismatch: Chunk assigned to job_a (belongs to paper_a) but with paper_b.id
    invalid_chunk = PaperChunk(
        paper_id=paper_b.id,
        ingestion_job_id=job_a.id,
        chunk_index=0,
        content="Inconsistent chunk",
    )
    db_session.add(invalid_chunk)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_integrity_doi_uniqueness(db_session: AsyncSession):
    """Verify that Paper.doi uniqueness is enforced at database level."""
    test_doi = f"10.1000/{uuid.uuid4()}"
    p1 = Paper(title="Paper 1", doi=test_doi)
    db_session.add(p1)
    await db_session.commit()

    p2 = Paper(title="Paper 2", doi=test_doi)
    db_session.add(p2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_integrity_primary_file_uniqueness(db_session: AsyncSession):
    """Verify that partial unique index allows only one primary file per paper."""
    paper = Paper(title="Multi-file Paper")
    db_session.add(paper)
    await db_session.flush()

    f1 = PaperFile(
        paper_id=paper.id,
        storage_provider="cloudinary",
        storage_key=f"files/{uuid.uuid4()}.pdf",
        original_filename="primary1.pdf",
        mime_type="application/pdf",
        status="available",
        is_primary=True,
    )
    db_session.add(f1)
    await db_session.commit()

    f2 = PaperFile(
        paper_id=paper.id,
        storage_provider="cloudinary",
        storage_key=f"files/{uuid.uuid4()}.pdf",
        original_filename="primary2.pdf",
        mime_type="application/pdf",
        status="available",
        is_primary=True,
    )
    db_session.add(f2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_integrity_message_sequence_uniqueness(db_session: AsyncSession):
    """Verify that message (conversation_id, sequence_number) unique constraint is enforced."""
    conv = Conversation(title="Sequence Test")
    db_session.add(conv)
    await db_session.flush()

    m1 = Message(conversation_id=conv.id, sequence_number=1, role="user", content="Hello")
    db_session.add(m1)
    await db_session.commit()

    m2 = Message(conversation_id=conv.id, sequence_number=1, role="assistant", content="Duplicate seq")
    db_session.add(m2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_integrity_citation_restrict_on_chunk_deletion(db_session: AsyncSession):
    """Verify foreign key ON DELETE RESTRICT prevents deleting chunks cited by messages."""
    paper = Paper(title="Citation Protected Paper")
    db_session.add(paper)
    await db_session.flush()

    job = IngestionJob(paper_id=paper.id, status="completed")
    db_session.add(job)
    await db_session.flush()

    chunk = PaperChunk(
        paper_id=paper.id,
        ingestion_job_id=job.id,
        chunk_index=0,
        content="Protected chunk content",
        is_current=True,
    )
    db_session.add(chunk)
    await db_session.flush()

    conv = Conversation(title="Citation Conv")
    db_session.add(conv)
    await db_session.flush()

    msg = Message(conversation_id=conv.id, sequence_number=1, role="assistant", content="Response")
    db_session.add(msg)
    await db_session.flush()

    citation = MessageCitation(
        message_id=msg.id,
        chunk_id=chunk.id,
        citation_order=1,
    )
    db_session.add(citation)
    await db_session.commit()

    # Attempt deleting cited chunk -> must fail with IntegrityError (ON DELETE RESTRICT)
    await db_session.delete(chunk)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_chunk_versioning_and_reprocessing_db_level(db_session: AsyncSession):
    """Verify previous chunks can be deactivated and new chunks activated in database transactions."""
    paper = Paper(title="Reprocessing Paper")
    db_session.add(paper)
    await db_session.flush()

    job1 = IngestionJob(paper_id=paper.id, status="completed")
    db_session.add(job1)
    await db_session.flush()

    c1 = PaperChunk(
        paper_id=paper.id,
        ingestion_job_id=job1.id,
        chunk_index=0,
        content="V1 chunk",
        is_current=True,
    )
    db_session.add(c1)
    await db_session.commit()

    # Reprocess: create Job 2 and chunks
    job2 = IngestionJob(paper_id=paper.id, status="completed")
    db_session.add(job2)
    await db_session.flush()

    c2 = PaperChunk(
        paper_id=paper.id,
        ingestion_job_id=job2.id,
        chunk_index=0,
        content="V2 chunk",
        is_current=False,
    )
    db_session.add(c2)
    await db_session.flush()

    # Deactivate old chunks and activate new chunks atomically
    await db_session.execute(
        update(PaperChunk)
        .where(PaperChunk.paper_id == paper.id, PaperChunk.is_current.is_(True))
        .values(is_current=False)
    )
    await db_session.execute(
        update(PaperChunk)
        .where(PaperChunk.paper_id == paper.id, PaperChunk.ingestion_job_id == job2.id)
        .values(is_current=True)
    )
    await db_session.commit()

    # Verify V1 chunk is inactive and V2 chunk is active
    c1_reloaded = await db_session.get(PaperChunk, c1.id)
    c2_reloaded = await db_session.get(PaperChunk, c2.id)
    assert c1_reloaded is not None and c1_reloaded.is_current is False
    assert c2_reloaded is not None and c2_reloaded.is_current is True


@pytest.mark.asyncio
async def test_postgres_default_is_current_false_on_insert(db_session: AsyncSession):
    """Verify PostgreSQL applies server_default FALSE when is_current is omitted in INSERT."""
    paper = Paper(title=f"Chunk Default Test Paper {uuid.uuid4()}")
    db_session.add(paper)
    await db_session.flush()

    job = IngestionJob(paper_id=paper.id, status="completed")
    db_session.add(job)
    await db_session.flush()

    chunk_id = uuid.uuid4()
    await db_session.execute(
        text(
            """
            INSERT INTO paper_chunks (id, paper_id, ingestion_job_id, chunk_index, content)
            VALUES (:id, :paper_id, :job_id, 0, 'Chunk omitting is_current')
            """
        ),
        {"id": chunk_id, "paper_id": paper.id, "job_id": job.id},
    )
    await db_session.commit()

    res = await db_session.execute(
        select(PaperChunk.is_current).where(PaperChunk.id == chunk_id)
    )
    is_current_val = res.scalar_one()
    assert is_current_val is False


@pytest.mark.asyncio
async def test_embedding_model_dimensions_constraint(db_session: AsyncSession):
    """Verify PostgreSQL rejects dimensions != 384 and accepts dimensions == 384."""
    # 1. Accept dimensions == 384
    valid_model = EmbeddingModel(
        provider="test-provider-valid-dim",
        model_name=f"model-384-{uuid.uuid4()}",
        dimensions=384,
        distance_metric="cosine",
    )
    db_session.add(valid_model)
    await db_session.commit()
    assert valid_model.dimensions == 384

    # 2. Reject dimensions != 384 (e.g. 512)
    invalid_model_512 = EmbeddingModel(
        provider="test-provider-invalid-dim",
        model_name=f"model-512-{uuid.uuid4()}",
        dimensions=512,
        distance_metric="cosine",
    )
    db_session.add(invalid_model_512)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()

    # 3. Reject dimensions != 384 (e.g. 128)
    invalid_model_128 = EmbeddingModel(
        provider="test-provider-invalid-dim",
        model_name=f"model-128-{uuid.uuid4()}",
        dimensions=128,
        distance_metric="cosine",
    )
    db_session.add(invalid_model_128)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_embedding_model_distance_metric_constraint(db_session: AsyncSession):
    """Verify PostgreSQL rejects distance_metric != 'cosine' and accepts 'cosine'."""
    # 1. Accept 'cosine'
    valid_model = EmbeddingModel(
        provider="test-provider-metric",
        model_name=f"model-cosine-{uuid.uuid4()}",
        dimensions=384,
        distance_metric="cosine",
    )
    db_session.add(valid_model)
    await db_session.commit()
    assert valid_model.distance_metric == "cosine"

    # 2. Reject 'l2'
    invalid_model_l2 = EmbeddingModel(
        provider="test-provider-metric",
        model_name=f"model-l2-{uuid.uuid4()}",
        dimensions=384,
        distance_metric="l2",
    )
    db_session.add(invalid_model_l2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()

    # 3. Reject 'inner_product'
    invalid_model_ip = EmbeddingModel(
        provider="test-provider-metric",
        model_name=f"model-ip-{uuid.uuid4()}",
        dimensions=384,
        distance_metric="inner_product",
    )
    db_session.add(invalid_model_ip)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_cancel_job_if_active_atomic_protection(db_session: AsyncSession):
    """Verify cancel_job_if_active cancels an active job and preserves a terminal completed job atomically."""
    # 1. Setup paper
    paper = Paper(
        id=uuid.uuid4(),
        title="Test Ingestion Cancellation Paper",
        abstract="Testing atomic cancellation in PostgreSQL",
    )
    db_session.add(paper)
    await db_session.flush()

    repo = IngestionRepository(db_session)

    # 2. Create active job
    active_job = await repo.create_job(paper_id=paper.id, status="processing")
    await db_session.commit()

    # Cancel active job via atomic UPDATE statement
    now = datetime.now(timezone.utc)
    res_cancelled = await repo.cancel_job_if_active(
        job_id=active_job.id,
        current_stage="embedding_cancelled",
        error_message="Cancelled by user",
        completed_at=now,
    )
    assert res_cancelled is not None
    assert res_cancelled.status == "cancelled"
    assert res_cancelled.current_stage == "embedding_cancelled"
    assert res_cancelled.error_message == "Cancelled by user"
    assert res_cancelled.completed_at is not None
    await db_session.commit()

    # 3. Create completed job
    completed_job = await repo.create_job(paper_id=paper.id, status="completed")
    await db_session.commit()

    # Attempt to cancel completed job - atomic conditional update should match 0 rows and preserve completed status
    res_preserved = await repo.cancel_job_if_active(
        job_id=completed_job.id,
        current_stage="cancel_attempt",
        error_message="Should not overwrite",
        completed_at=now,
    )
    assert res_preserved is not None
    assert res_preserved.status == "completed"
    assert res_preserved.current_stage != "cancel_attempt"
    assert res_preserved.error_message != "Should not overwrite"

