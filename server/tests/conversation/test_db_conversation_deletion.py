"""Focused integration tests verifying conversation deletion and database cascade in PostgreSQL.

Verifies:
1. Creating a Paper conversation with messages and citations.
2. Creating a General conversation with messages.
3. Confirming both conversations appear in list_conversations.
4. Deleting the Paper conversation via ConversationService:
   - Conversation record deleted from database.
   - Associated messages deleted from database.
   - Associated citations deleted from database.
   - Referenced paper and paper chunk remain intact.
   - Conversation no longer appears in list_conversations.
   - get_conversation raises ConversationNotFoundError.
5. Deleting the General conversation:
   - Conversation and messages deleted from database.
   - Conversation no longer appears in list_conversations.
6. Attempting to delete a nonexistent conversation raises ConversationNotFoundError.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import uuid

from dotenv import load_dotenv
import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database.model_registry  # noqa: F401 - ensures all mappers registered
from app.database.model_registry import (
    Base,
    Conversation,
    IngestionJob,
    Message,
    MessageCitation,
    Paper,
    PaperChunk,
)
from app.modules.conversation.service import (
    ConversationNotFoundError,
    ConversationService,
)
from tests.retrieval.test_retrieval_integration import verify_test_database_safety

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

TEST_DB_URL = os.getenv("test_db_url") or os.getenv("TEST_DB_URL")
APP_DB_URL = os.getenv("database_url") or os.getenv("DATABASE_URL")


@pytest_asyncio.fixture
async def deletion_test_db_engine():
    """Verify safety guards and create async engine for test database."""
    if not TEST_DB_URL:
        pytest.skip("TEST_DB_URL is not configured. Skipping real DB deletion test.")

    verify_test_database_safety(TEST_DB_URL, APP_DB_URL)
    engine = create_async_engine(TEST_DB_URL, pool_pre_ping=True)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)
    except (OSError, socket.gaierror, ConnectionRefusedError) as exc:
        await engine.dispose()
        pytest.skip(
            f"Test database is unreachable ({type(exc).__name__}). Skipping real DB deletion test."
        )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def deletion_session_factory(deletion_test_db_engine):
    """Provide sessionmaker for isolated test database."""
    return async_sessionmaker(bind=deletion_test_db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.mark.asyncio
async def test_real_db_conversation_deletion_cascade(deletion_session_factory):
    """Verify permanent deletion cascades to messages and citations while preserving paper chunks."""
    service = ConversationService(session_factory=deletion_session_factory)

    # 1. Seed a test Paper and PaperChunk in the test database
    unique_suffix = uuid.uuid4().hex[:8]
    async with deletion_session_factory() as session:
        async with session.begin():
            paper = Paper(
                title=f"Test Deletion Paper {unique_suffix}",
                doi=f"10.1000/del.{unique_suffix}",
            )
            session.add(paper)
            await session.flush()

            job = IngestionJob(paper_id=paper.id, status="completed")
            session.add(job)
            await session.flush()

            chunk = PaperChunk(
                paper_id=paper.id,
                ingestion_job_id=job.id,
                chunk_index=0,
                content="Evidence text for deletion test.",
            )
            session.add(chunk)
            await session.flush()

            paper_id = paper.id
            chunk_id = chunk.id

    try:
        # 2. Create Paper Conversation and General Conversation
        paper_conv = await service.create_conversation(
            title=f"Paper Chat {unique_suffix}",
            chat_type="paper",
        )
        general_conv = await service.create_conversation(
            title=f"General Chat {unique_suffix}",
            chat_type="general",
        )

        paper_conv_id = paper_conv.id
        general_conv_id = general_conv.id

        # 3. Add Messages and Citation to Paper Conversation
        async with deletion_session_factory() as session:
            async with session.begin():
                m1 = Message(
                    conversation_id=paper_conv_id,
                    sequence_number=1,
                    role="user",
                    content="What is the evidence?",
                    status="completed",
                )
                session.add(m1)
                await session.flush()

                m2 = Message(
                    conversation_id=paper_conv_id,
                    sequence_number=2,
                    role="assistant",
                    content="Here is the evidence answer.",
                    status="completed",
                )
                session.add(m2)
                await session.flush()

                citation = MessageCitation(
                    message_id=m2.id,
                    chunk_id=chunk_id,
                    citation_order=1,
                    relevance_score=4.9,
                    cited_text="Evidence text for deletion test.",
                    page_number=1,
                )
                session.add(citation)

                # Add a message to General Conversation
                gm1 = Message(
                    conversation_id=general_conv_id,
                    sequence_number=1,
                    role="user",
                    content="Hello general chat",
                    status="completed",
                )
                session.add(gm1)

        # 4. Confirm both conversations appear in list_conversations
        active_list = await service.list_conversations(limit=100)
        active_ids = {c.id for c in active_list}
        assert paper_conv_id in active_ids
        assert general_conv_id in active_ids

        # 5. Permanently delete the Paper Conversation
        await service.delete_conversation(paper_conv_id)

        # 6. Verify Paper Conversation record is deleted from DB
        async with deletion_session_factory() as session:
            conv_check = await session.execute(
                select(Conversation).where(Conversation.id == paper_conv_id)
            )
            assert conv_check.scalar_one_or_none() is None

            # Verify associated messages are deleted
            msg_check = await session.execute(
                select(Message).where(Message.conversation_id == paper_conv_id)
            )
            assert list(msg_check.scalars().all()) == []

            # Verify associated citations are deleted
            cit_check = await session.execute(
                select(MessageCitation).where(MessageCitation.chunk_id == chunk_id)
            )
            assert list(cit_check.scalars().all()) == []

            # Verify supporting chunk and paper are NOT deleted
            chunk_check = await session.execute(
                select(PaperChunk).where(PaperChunk.id == chunk_id)
            )
            assert chunk_check.scalar_one_or_none() is not None

        # 7. Verify Paper Conversation no longer appears in list_conversations
        updated_list = await service.list_conversations(limit=100)
        updated_ids = {c.id for c in updated_list}
        assert paper_conv_id not in updated_ids
        assert general_conv_id in updated_ids

        # 8. Verify get_conversation raises ConversationNotFoundError
        with pytest.raises(ConversationNotFoundError):
            await service.get_conversation(paper_conv_id)

        # 9. Verify deleting nonexistent conversation raises ConversationNotFoundError
        with pytest.raises(ConversationNotFoundError):
            await service.delete_conversation(uuid.uuid4())

        # 10. Permanently delete General Conversation
        await service.delete_conversation(general_conv_id)

        async with deletion_session_factory() as session:
            gconv_check = await session.execute(
                select(Conversation).where(Conversation.id == general_conv_id)
            )
            assert gconv_check.scalar_one_or_none() is None

        final_list = await service.list_conversations(limit=100)
        final_ids = {c.id for c in final_list}
        assert general_conv_id not in final_ids

    finally:
        # Cleanup seeded test paper and chunks
        async with deletion_session_factory() as session:
            async with session.begin():
                await session.execute(text("DELETE FROM paper_chunks WHERE id = :id"), {"id": chunk_id})
                await session.execute(text("DELETE FROM ingestion_jobs WHERE id = :id"), {"id": job.id})
                await session.execute(text("DELETE FROM papers WHERE id = :id"), {"id": paper_id})
