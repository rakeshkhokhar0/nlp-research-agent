"""Unit tests for ConversationRepository persistence methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.conversation.models import Conversation, Message, MessageCitation
from app.modules.conversation.repository import ConversationRepository


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_create_conversation(mock_session: AsyncMock) -> None:
    """Verify create_conversation creates, adds, and flushes Conversation."""
    repo = ConversationRepository(mock_session)

    conv = await repo.create_conversation(title="Test Conv", metadata_={"tag": "nlp"})

    assert conv.title == "Test Conv"
    assert conv.status == "active"
    assert conv.metadata_ == {"tag": "nlp"}
    mock_session.add.assert_called_once_with(conv)
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_conversation_by_id(mock_session: AsyncMock) -> None:
    """Verify get_conversation retrieves conversation by ID."""
    repo = ConversationRepository(mock_session)
    conv_id = uuid.uuid4()
    mock_conv = Conversation(id=conv_id, title="Test")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_conv
    mock_session.execute.return_value = mock_result

    res = await repo.get_conversation(conv_id, for_update=True)

    assert res is mock_conv
    mock_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_next_sequence_number(mock_session: AsyncMock) -> None:
    """Verify get_next_sequence_number returns scalar from COALESCE query."""
    repo = ConversationRepository(mock_session)
    conv_id = uuid.uuid4()

    mock_result = MagicMock()
    mock_result.scalar_one.return_value = 5
    mock_session.execute.return_value = mock_result

    seq = await repo.get_next_sequence_number(conv_id)

    assert seq == 5
    mock_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_message_validation(mock_session: AsyncMock) -> None:
    """Verify validation on sequence number, role, and empty content."""
    repo = ConversationRepository(mock_session)
    conv_id = uuid.uuid4()

    # Invalid sequence_number <= 0
    with pytest.raises(ValueError, match="sequence_number must be positive"):
        await repo.create_message(conv_id, sequence_number=0, role="user", content="Hi")

    # Invalid role
    with pytest.raises(ValueError, match="role must be"):
        await repo.create_message(conv_id, sequence_number=1, role="invalid_role", content="Hi")

    # Empty content
    with pytest.raises(ValueError, match="content must not be empty"):
        await repo.create_message(conv_id, sequence_number=1, role="user", content="   ")


@pytest.mark.asyncio
async def test_create_message_success(mock_session: AsyncMock) -> None:
    """Verify creating a valid message flushes correctly."""
    repo = ConversationRepository(mock_session)
    conv_id = uuid.uuid4()

    msg = await repo.create_message(
        conversation_id=conv_id,
        sequence_number=1,
        role="assistant",
        content="Answer",
        model_name="gpt-4o-mini",
        token_count=15,
    )

    assert msg.conversation_id == conv_id
    assert msg.sequence_number == 1
    assert msg.role == "assistant"
    assert msg.content == "Answer"
    assert msg.model_name == "gpt-4o-mini"
    assert msg.token_count == 15
    mock_session.add.assert_called_once_with(msg)
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_citations_success(mock_session: AsyncMock) -> None:
    """Verify creating citations flushes all records."""
    repo = ConversationRepository(mock_session)
    msg_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    citations_data = [
        {
            "chunk_id": chunk_id,
            "citation_order": 1,
            "relevance_score": 3.5,
            "cited_text": "Sample text",
            "page_number": 2,
        }
    ]

    cits = await repo.create_citations(msg_id, citations_data)

    assert len(cits) == 1
    assert cits[0].message_id == msg_id
    assert cits[0].chunk_id == chunk_id
    assert cits[0].citation_order == 1
    assert cits[0].page_number == 2
    mock_session.add.assert_called_once()
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_chunk_page_information(mock_session: AsyncMock) -> None:
    """Verify get_chunk_page_information returns chunk_id -> page_start mapping."""
    repo = ConversationRepository(mock_session)
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()

    mock_result = MagicMock()
    mock_result.all.return_value = [(c1, 4), (c2, None)]
    mock_session.execute.return_value = mock_result

    pages = await repo.get_chunk_page_information([c1, c2])

    assert pages == {c1: 4, c2: None}


@pytest.mark.asyncio
async def test_update_context_summary(mock_session: AsyncMock) -> None:
    """Verify update_context_summary sets summary and metadata on conversation."""
    repo = ConversationRepository(mock_session)
    conv_id = uuid.uuid4()
    mock_conv = Conversation(id=conv_id, title="Test", context_summary=None, metadata_={})

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_conv
    mock_session.execute.return_value = mock_result

    await repo.update_context_summary(
        conversation_id=conv_id,
        context_summary="Updated summary text",
        last_summarized_sequence=8,
    )

    assert mock_conv.context_summary == "Updated summary text"
    assert mock_conv.metadata_["last_summarized_sequence"] == 8
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_messages_for_summarization(mock_session: AsyncMock) -> None:
    """Verify get_messages_for_summarization executes query with bounds and returns list."""
    repo = ConversationRepository(mock_session)
    conv_id = uuid.uuid4()
    m1 = Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Q1")
    m2 = Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=2, role="assistant", content="A1")

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [m1, m2]
    mock_session.execute.return_value = mock_result

    msgs = await repo.get_messages_for_summarization(conv_id, min_sequence=1, max_sequence=2)

    assert msgs == [m1, m2]
    mock_session.execute.assert_awaited_once()
