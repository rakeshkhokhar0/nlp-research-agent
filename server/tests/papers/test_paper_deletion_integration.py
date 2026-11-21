"""Comprehensive integration tests verifying research paper deletion across PostgreSQL,
vector embeddings, historical message citations, lexical retrieval index refresh, and Cloudinary cleanup.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import socket
from unittest.mock import MagicMock
import uuid

from dotenv import load_dotenv
import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database.model_registry  # noqa: F401 - ensures all mappers registered
from app.core.exceptions import ActiveJobConflictError, StorageDeleteError
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
)
from app.modules.ingestion.storage import CloudinaryStorage
from app.modules.papers.repository import PaperNotFoundError, PaperRepository
from app.modules.papers.service import PaperService
from app.modules.retrieval import RetrievalRepository, RetrievalService
from tests.retrieval.test_retrieval_integration import verify_test_database_safety

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

TEST_DB_URL = os.getenv("test_db_url") or os.getenv("TEST_DB_URL")
APP_DB_URL = os.getenv("database_url") or os.getenv("DATABASE_URL")


@pytest_asyncio.fixture
async def paper_deletion_engine():
    """Verify safety guards and create async engine for test database."""
    if not TEST_DB_URL:
        pytest.skip("TEST_DB_URL is not configured. Skipping paper deletion integration test.")

    verify_test_database_safety(TEST_DB_URL, APP_DB_URL)
    engine = create_async_engine(TEST_DB_URL, pool_pre_ping=True)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)
    except (OSError, socket.gaierror, ConnectionRefusedError) as exc:
        await engine.dispose()
        pytest.skip(
            f"Test database is unreachable ({type(exc).__name__}). Skipping paper deletion integration test."
        )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(paper_deletion_engine):
    """Provide sessionmaker for isolated test database."""
    return async_sessionmaker(bind=paper_deletion_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.mark.asyncio
async def test_full_paper_deletion_lifecycle_with_index_and_citations(session_factory):
    """Verify complete paper deletion:
    1. Cascades chunks, files, embeddings, and jobs.
    2. Snapshots message citation details into message metadata.
    3. Preserves conversations and messages.
    4. Refreshes lexical retrieval index to purge deleted chunks while keeping other papers searchable.
    5. Triggers Cloudinary file cleanup.
    """
    unique_suffix = uuid.uuid4().hex[:8]
    cloudinary_key = f"raw/papers/test_{unique_suffix}"

    mock_storage = MagicMock(spec=CloudinaryStorage)
    mock_storage.delete_file = MagicMock(return_value=True)

    retrieval_service = RetrievalService()

    async with session_factory() as session:
        async with session.begin():
            # 1. Ensure embedding model exists
            res_model = await session.execute(
                select(EmbeddingModel).where(EmbeddingModel.is_active == True)  # noqa: E712
            )
            emb_model = res_model.scalars().first()
            if not emb_model:
                emb_model = EmbeddingModel(
                    provider="sentence-transformers",
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    dimensions=384,
                    distance_metric="cosine",
                    normalize_embeddings=True,
                    is_active=True,
                )
                session.add(emb_model)
                await session.flush()
            emb_model_id = emb_model.id

            # 2. Seed Paper 1 ("Paper to Delete")
            p1 = Paper(
                title=f"Paper To Delete {unique_suffix}",
                doi=f"10.1000/del1.{unique_suffix}",
            )
            session.add(p1)
            await session.flush()

            f1 = PaperFile(
                paper_id=p1.id,
                storage_provider="cloudinary",
                storage_key=cloudinary_key,
                original_filename=f"delete_me_{unique_suffix}.pdf",
                mime_type="application/pdf",
                status="available",
                is_primary=True,
            )
            session.add(f1)
            await session.flush()

            j1 = IngestionJob(paper_id=p1.id, paper_file_id=f1.id, status="completed")
            session.add(j1)
            await session.flush()

            c1 = PaperChunk(
                paper_id=p1.id,
                ingestion_job_id=j1.id,
                chunk_index=0,
                content="Quantum computing algorithms using superconducting qubits for optimization.",
                is_current=True,
            )
            session.add(c1)
            await session.flush()

            e1 = ChunkEmbedding(
                chunk_id=c1.id,
                embedding_model_id=emb_model_id,
                embedding=[0.05] * 384,
            )
            session.add(e1)

            # 3. Seed Paper 2 ("Paper to Keep")
            p2 = Paper(
                title=f"Paper To Keep {unique_suffix}",
                doi=f"10.1000/keep2.{unique_suffix}",
            )
            session.add(p2)
            await session.flush()

            j2 = IngestionJob(paper_id=p2.id, status="completed")
            session.add(j2)
            await session.flush()

            c2 = PaperChunk(
                paper_id=p2.id,
                ingestion_job_id=j2.id,
                chunk_index=0,
                content="Convolutional neural networks for satellite image classification and semantic segmentation.",
                is_current=True,
            )
            session.add(c2)
            await session.flush()

            e2 = ChunkEmbedding(
                chunk_id=c2.id,
                embedding_model_id=emb_model_id,
                embedding=[0.05] * 384,
            )
            session.add(e2)

            # 4. Create Conversation, Message, and Citation referencing c1
            conv = Conversation(title="Research Chat", status="active")
            session.add(conv)
            await session.flush()

            msg = Message(
                conversation_id=conv.id,
                sequence_number=1,
                role="assistant",
                content="Here is quantum evidence.",
                status="completed",
                metadata_={},
            )
            session.add(msg)
            await session.flush()

            cit = MessageCitation(
                message_id=msg.id,
                chunk_id=c1.id,
                citation_order=1,
                relevance_score=4.95,
                cited_text="Quantum computing algorithms using superconducting qubits.",
                page_number=2,
            )
            session.add(cit)

            p1_id = p1.id
            p2_id = p2.id
            c1_id = c1.id
            c2_id = c2.id
            msg_id = msg.id
            conv_id = conv.id

    try:
        # 5. Initialize retrieval service index with both papers
        async with session_factory() as session:
            retrieval_repo = RetrievalRepository(session)
            indexed_count = await retrieval_service.refresh_index(retrieval_repo)
            assert indexed_count >= 2

            # Search quantum -> should find c1
            res_quantum = retrieval_service.search("quantum computing superconducting", top_k=5)
            assert any(r.chunk_id == c1_id for r in res_quantum)

            # Search satellite -> should find c2
            res_satellite = retrieval_service.search("satellite convolutional neural networks", top_k=5)
            assert any(r.chunk_id == c2_id for r in res_satellite)

        # 6. Delete Paper 1 via PaperService
        async with session_factory() as session:
            paper_repo = PaperRepository(session)
            service = PaperService(
                repository=paper_repo,
                retrieval_service=retrieval_service,
                storage=mock_storage,
            )
            await service.delete_paper(p1_id)

        # 7. Verify Database State
        async with session_factory() as session:
            # Paper 1 is deleted
            chk_p1 = await session.execute(select(Paper).where(Paper.id == p1_id))
            assert chk_p1.scalar_one_or_none() is None

            # Paper 1 chunks are deleted
            chk_c1 = await session.execute(select(PaperChunk).where(PaperChunk.paper_id == p1_id))
            assert list(chk_c1.scalars().all()) == []

            # Paper 1 embeddings are deleted
            chk_e1 = await session.execute(select(ChunkEmbedding).where(ChunkEmbedding.chunk_id == c1_id))
            assert list(chk_e1.scalars().all()) == []

            # Paper 1 files are deleted
            chk_f1 = await session.execute(select(PaperFile).where(PaperFile.paper_id == p1_id))
            assert list(chk_f1.scalars().all()) == []

            # MessageCitation referencing c1 is deleted
            chk_cit = await session.execute(select(MessageCitation).where(MessageCitation.chunk_id == c1_id))
            assert list(chk_cit.scalars().all()) == []

            # Conversation and Message are preserved!
            chk_msg = await session.execute(select(Message).where(Message.id == msg_id))
            persisted_msg = chk_msg.scalar_one()
            assert persisted_msg.content == "Here is quantum evidence."
            assert "historical_citations" in persisted_msg.metadata_
            hist = persisted_msg.metadata_["historical_citations"]
            assert len(hist) == 1
            assert hist[0]["chunk_id"] == str(c1_id)
            assert hist[0]["page_number"] == 2
            assert hist[0]["paper_title"] == f"Paper To Delete {unique_suffix}"

            # Paper 2 remains intact
            chk_p2 = await session.execute(select(Paper).where(Paper.id == p2_id))
            assert chk_p2.scalar_one_or_none() is not None

            chk_c2 = await session.execute(select(PaperChunk).where(PaperChunk.id == c2_id))
            assert chk_c2.scalar_one_or_none() is not None

            # Shared embedding model is NOT deleted
            chk_model = await session.execute(select(EmbeddingModel).where(EmbeddingModel.id == emb_model_id))
            assert chk_model.scalar_one_or_none() is not None

        # 8. Verify Lexical Retrieval Index
        # Search for quantum keywords: MUST NOT return c1
        res_quantum_after = retrieval_service.search("quantum computing superconducting", top_k=5)
        assert not any(r.chunk_id == c1_id for r in res_quantum_after)

        # Search for satellite keywords: MUST STILL return c2
        res_satellite_after = retrieval_service.search("satellite convolutional neural networks", top_k=5)
        assert any(r.chunk_id == c2_id for r in res_satellite_after)

        # 9. Verify Cloudinary storage cleanup was called with exact storage key
        mock_storage.delete_file.assert_called_once_with(cloudinary_key)

    finally:
        # Cleanup Paper 2 and Conversation
        async with session_factory() as session:
            async with session.begin():
                await session.execute(text("DELETE FROM chunk_embeddings WHERE chunk_id = :id"), {"id": c2_id})
                await session.execute(text("DELETE FROM paper_chunks WHERE id = :id"), {"id": c2_id})
                await session.execute(text("DELETE FROM ingestion_jobs WHERE paper_id = :id"), {"id": p2_id})
                await session.execute(text("DELETE FROM papers WHERE id = :id"), {"id": p2_id})
                await session.execute(text("DELETE FROM messages WHERE conversation_id = :id"), {"id": conv_id})
                await session.execute(text("DELETE FROM conversations WHERE id = :id"), {"id": conv_id})


@pytest.mark.asyncio
async def test_paper_deletion_active_job_conflict(session_factory):
    """Verify delete_paper is rejected with ActiveJobConflictError when an ingestion job is processing."""
    unique_suffix = uuid.uuid4().hex[:8]

    async with session_factory() as session:
        async with session.begin():
            paper = Paper(
                title=f"Active Job Paper {unique_suffix}",
                doi=f"10.1000/active.{unique_suffix}",
            )
            session.add(paper)
            await session.flush()

            job = IngestionJob(paper_id=paper.id, status="processing")
            session.add(job)
            await session.flush()

            paper_id = paper.id
            job_id = job.id

    try:
        async with session_factory() as session:
            paper_repo = PaperRepository(session)
            service = PaperService(repository=paper_repo)

            with pytest.raises(ActiveJobConflictError, match="active ingestion job"):
                await service.delete_paper(paper_id)

        # Verify paper still exists in database
        async with session_factory() as session:
            chk = await session.execute(select(Paper).where(Paper.id == paper_id))
            assert chk.scalar_one_or_none() is not None

    finally:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(text("DELETE FROM ingestion_jobs WHERE id = :id"), {"id": job_id})
                await session.execute(text("DELETE FROM papers WHERE id = :id"), {"id": paper_id})


@pytest.mark.asyncio
async def test_paper_deletion_nonexistent_raises_404(session_factory):
    """Verify delete_paper raises PaperNotFoundError for nonexistent paper."""
    async with session_factory() as session:
        paper_repo = PaperRepository(session)
        service = PaperService(repository=paper_repo)

        unknown_id = uuid.uuid4()
        with pytest.raises(PaperNotFoundError, match=f"Research paper with id '{unknown_id}' not found"):
            await service.delete_paper(unknown_id)
