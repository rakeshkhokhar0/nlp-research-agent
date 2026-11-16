"""Focused integration test against isolated PostgreSQL test database for context summarization persistence.

Verifies:
1. Create a conversation in real PostgreSQL.
2. Send enough messages to trigger Cycle 1 summarization.
3. Confirm context_summary is persisted in the PostgreSQL database.
4. Confirm last_summarized_sequence watermark is persisted in PostgreSQL metadata_ JSONB.
5. Send another message; confirm stored summary is reused without summarizing the same messages again.
6. Accumulate new messages beyond recent-history window to trigger Cycle 2 summarization.
7. Confirm Cycle 2 receives previous summary and only newly eligible messages (not previously summarized).
8. Confirm Cycle 2 summary and advanced sequence watermark are persisted in real PostgreSQL.
9. Send follow-up message; confirm Cycle 2 summary is reused.
10. Simulate upstream summarizer failure; confirm previous summary and watermark are preserved in PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import uuid

from dotenv import load_dotenv
import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database.model_registry  # noqa: F401
from app.database.model_registry import Conversation, Message
from app.modules.conversation.context import ContextManager
from app.modules.conversation.llm_client import LLMAnswerResult, OpenAIResponseClient
from app.modules.conversation.service import ConversationService
from tests.retrieval.test_retrieval_integration import verify_test_database_safety

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

TEST_DB_URL = os.getenv("test_db_url") or os.getenv("TEST_DB_URL")
APP_DB_URL = os.getenv("database_url") or os.getenv("DATABASE_URL")


@pytest_asyncio.fixture
async def real_test_db_engine():
    """Verify safety guards and create async engine for test database."""
    if not TEST_DB_URL:
        pytest.skip("test_db_url is not configured.")

    verify_test_database_safety(TEST_DB_URL, APP_DB_URL)
    engine = create_async_engine(TEST_DB_URL, pool_pre_ping=True)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def real_test_session_factory(real_test_db_engine):
    """Provide sessionmaker for isolated test database."""
    return async_sessionmaker(bind=real_test_db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.mark.asyncio
async def test_real_db_context_summarization_lifecycle(real_test_session_factory) -> None:
    """Verify end-to-end summarization persistence and reuse in real PostgreSQL test database."""
    verify_test_database_safety(TEST_DB_URL, APP_DB_URL)

    # Mock LLM client so we test deterministic database state without OpenAI rate limits
    mock_llm = MagicMock(spec=OpenAIResponseClient)
    mock_llm.default_model = "gpt-4o-mini"
    mock_llm.summarize_messages = AsyncMock(return_value="Persisted summary of older messages.")
    mock_llm.generate_general_answer = AsyncMock(
        side_effect=lambda question, conversation_history, conversation_summary: LLMAnswerResult(
            answer=f"Answer for: {question}",
            cited_chunk_ids=[],
            model_name="gpt-4o-mini",
            token_count=30,
        )
    )

    context_manager = ContextManager(
        llm_client=mock_llm,
        recent_message_limit=8,
        summarization_threshold=10,
        min_summarize_batch=3,
    )

    service = ConversationService(
        session_factory=real_test_session_factory,
        llm_client=mock_llm,
        context_manager=context_manager,
    )

    created_conv_id: uuid.UUID | None = None

    try:
        # Step 1: Create a conversation in real PostgreSQL
        created = await service.create_conversation(
            title="Real DB Summarization Test",
            chat_type="general",
        )
        created_conv_id = created.id
        assert created.status == "active"
        assert created.chat_type == "general"
        assert created.context_summary is None

        # Verify in real DB
        async with real_test_session_factory() as session:
            stmt = select(Conversation).where(Conversation.id == created_conv_id)
            conv_row = (await session.execute(stmt)).scalar_one()
            assert conv_row.context_summary is None
            assert conv_row.metadata_.get("chat_type") == "general"

        # Step 2: Send 5 exchanges (10 messages total) - below threshold at start of each turn
        for i in range(1, 6):
            await service.ask_question(
                conversation_id=created_conv_id,
                question=f"User question {i}",
                chat_type="general",
            )

        # Confirm no summarization has triggered yet (DB had 8 messages at start of Q5)
        assert mock_llm.summarize_messages.await_count == 0

        # Send 6th exchange -> at start of Q6, DB has 10 messages.
        # Threshold reached: 10 >= 10. Older messages count = 10 - 8 = 2 (messages 1..2).
        # Cutoff sequence = 2. Last summarized = 0.
        # Since this is the initial summarization (context_summary is None), it triggers immediately.
        await service.ask_question(
            conversation_id=created_conv_id,
            question="User question 6 (trigger summarization)",
            chat_type="general",
        )

        # Confirm summarization was triggered
        assert mock_llm.summarize_messages.await_count == 1

        # Step 3 & 4: Confirm context_summary and last_summarized_sequence are persisted in real DB
        async with real_test_session_factory() as session:
            stmt = select(Conversation).where(Conversation.id == created_conv_id)
            conv_row = (await session.execute(stmt)).scalar_one()

            # Confirm context_summary is persisted in PostgreSQL
            assert conv_row.context_summary == "Persisted summary of older messages."

            # Confirm last_summarized_sequence is persisted in PostgreSQL metadata_ JSONB
            assert conv_row.metadata_["last_summarized_sequence"] == 2

        # Step 5: Send another message (exchange 7)
        # At start of Q7, DB has 12 messages. Older messages = 4 (messages 1..4). Cutoff = 4.
        # Unsummarized count = 4 - 2 = 2. Since min_summarize_batch is 3, 2 < 3.
        # Re-summarization is skipped and existing stored summary is reused.
        mock_llm.summarize_messages.reset_mock()
        await service.ask_question(
            conversation_id=created_conv_id,
            question="User question 7 (reuse stored summary)",
            chat_type="general",
        )

        # Step 6: Confirm stored summary was reused without summarizing again
        assert mock_llm.summarize_messages.await_count == 0

        # Verify that the general answer LLM received the stored summary from real PostgreSQL
        last_general_call = mock_llm.generate_general_answer.call_args
        assert last_general_call.kwargs["conversation_summary"] == "Persisted summary of older messages."

        # ---------------------------------------------------------------------
        # Cycle 2: Second Incremental Summarization
        # ---------------------------------------------------------------------
        # Step 7: Send exchange 8 -> triggers Cycle 2 summarization
        # At start of Q8, DB has 14 messages. Older messages count = 14 - 8 = 6 (messages 1..6).
        # Cutoff sequence = 6. Last summarized sequence = 2.
        # Unsummarized count = 6 - 2 = 4 >= min_summarize_batch (3).
        # Newly eligible messages to summarize: sequence 3, 4, 5, 6.
        mock_llm.summarize_messages.reset_mock()
        mock_llm.summarize_messages.return_value = "Cycle 2 summary incorporating messages up to 6."

        await service.ask_question(
            conversation_id=created_conv_id,
            question="User question 8 (trigger cycle 2 summarization)",
            chat_type="general",
        )

        # Confirm summarization was triggered for Cycle 2
        assert mock_llm.summarize_messages.await_count == 1
        cycle2_call = mock_llm.summarize_messages.call_args

        # Verify previous summary was passed in
        assert cycle2_call.kwargs["existing_summary"] == "Persisted summary of older messages."

        # Verify only newly eligible messages (sequences 3..6) were passed, not sequences 1..2
        cycle2_msgs = cycle2_call.kwargs["messages"]
        assert len(cycle2_msgs) == 4
        assert cycle2_msgs[0]["content"] == "User question 2"
        assert cycle2_msgs[1]["content"] == "Answer for: User question 2"
        assert cycle2_msgs[2]["content"] == "User question 3"
        assert cycle2_msgs[3]["content"] == "Answer for: User question 3"

        # Step 8: Confirm updated summary and advanced watermark are persisted in real PostgreSQL
        async with real_test_session_factory() as session:
            stmt = select(Conversation).where(Conversation.id == created_conv_id)
            conv_row = (await session.execute(stmt)).scalar_one()
            assert conv_row.context_summary == "Cycle 2 summary incorporating messages up to 6."
            assert conv_row.metadata_["last_summarized_sequence"] == 6

        # Step 9: Send exchange 9 -> verifies Cycle 2 summary reuse
        # At start of Q9, DB has 16 messages. Older count = 16 - 8 = 8.
        # Cutoff sequence = 8. Unsummarized count = 8 - 6 = 2 < min_summarize_batch (3).
        mock_llm.summarize_messages.reset_mock()
        await service.ask_question(
            conversation_id=created_conv_id,
            question="User question 9 (reuse cycle 2 summary)",
            chat_type="general",
        )
        assert mock_llm.summarize_messages.await_count == 0
        last_general_call = mock_llm.generate_general_answer.call_args
        assert last_general_call.kwargs["conversation_summary"] == "Cycle 2 summary incorporating messages up to 6."

        # ---------------------------------------------------------------------
        # Error Resilience: Summarization Failure Fallback
        # ---------------------------------------------------------------------
        # Step 10: Send exchange 10 where LLM summarization fails
        # At start of Q10, DB has 18 messages. Older count = 18 - 8 = 10. Cutoff = 10.
        # Unsummarized count = 10 - 6 = 4 >= 3 (would trigger summarization).
        # We simulate an upstream failure. The failure must NOT destroy previous summary.
        mock_llm.summarize_messages.reset_mock()
        mock_llm.summarize_messages.side_effect = RuntimeError("Simulated transient OpenAI 503 outage")

        await service.ask_question(
            conversation_id=created_conv_id,
            question="User question 10 (test summarization failure fallback)",
            chat_type="general",
        )

        # Confirm summarization was attempted
        assert mock_llm.summarize_messages.await_count == 1

        # Confirm PostgreSQL still holds the Cycle 2 summary and watermark without corruption
        async with real_test_session_factory() as session:
            stmt = select(Conversation).where(Conversation.id == created_conv_id)
            conv_row = (await session.execute(stmt)).scalar_one()
            assert conv_row.context_summary == "Cycle 2 summary incorporating messages up to 6."
            assert conv_row.metadata_["last_summarized_sequence"] == 6

    finally:
        # Clean up test conversation from isolated test database
        if created_conv_id is not None:
            async with real_test_session_factory() as session:
                await session.execute(
                    text("DELETE FROM messages WHERE conversation_id = :cid;"),
                    {"cid": created_conv_id},
                )
                await session.execute(
                    text("DELETE FROM conversations WHERE id = :cid;"),
                    {"cid": created_conv_id},
                )
                await session.commit()
