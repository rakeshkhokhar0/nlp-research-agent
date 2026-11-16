"""Unit tests for ConversationService orchestrating Q&A, retrieval, and LLM calls."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.conversation.llm_client import LLMAnswerResult, LLMClientError, OpenAIResponseClient
from app.modules.conversation.models import Conversation, Message, MessageCitation
from app.modules.conversation.service import (
    GREETING_RESPONSE,
    NO_EVIDENCE_RESPONSE,
    UNSUPPORTED_SCOPE_RESPONSE,
    ConversationInactiveError,
    ConversationNotFoundError,
    ConversationService,
    LLMGenerationError,
    RetrievalUnavailableError,
)
from app.modules.retrieval.service import RetrievalService, RetrievedChunk


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)
    tx = AsyncMock()
    tx.__aenter__.return_value = session
    tx.__aexit__.return_value = None
    session.begin.return_value = tx
    return session


@pytest.fixture
def mock_session_factory(mock_session: AsyncMock) -> MagicMock:
    factory = MagicMock(spec=async_sessionmaker)
    cm = AsyncMock()
    cm.__aenter__.return_value = mock_session
    cm.__aexit__.return_value = None
    factory.return_value = cm
    return factory


@pytest.fixture
def mock_retrieval_service() -> MagicMock:
    service = MagicMock(spec=RetrievalService)
    service.is_initialized = True
    service.search = MagicMock()
    return service


@pytest.fixture
def mock_llm_client() -> MagicMock:
    client = MagicMock(spec=OpenAIResponseClient)
    client.default_model = "gpt-4o-mini"
    client.generate_grounded_answer = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        paper_id=uuid.uuid4(),
        chunk_index=0,
        content="Transformer architecture uses multi-head attention.",
        score=5.0,
        paper_title="Attention Is All You Need",
        publication_year=2017,
    )


# -----------------------------------------------------------------------------
# Conversation Management Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_conversation(mock_session_factory: MagicMock, mock_session: AsyncMock) -> None:
    """Verify creating a conversation persists Conversation record and returns response."""
    service = ConversationService(session_factory=mock_session_factory)

    mock_conv = Conversation(
        id=uuid.uuid4(),
        title="NLP Study",
        status="active",
        created_at=datetime.now(timezone.utc),
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.create_conversation = AsyncMock(return_value=mock_conv)

        res = await service.create_conversation(title="NLP Study")

        assert res.id == mock_conv.id
        assert res.title == "NLP Study"
        assert res.status == "active"
        repo.create_conversation.assert_awaited_once_with(title="NLP Study", metadata_=None)


@pytest.mark.asyncio
async def test_get_conversation_not_found_raises(
    mock_session_factory: MagicMock, mock_session: AsyncMock
) -> None:
    """Verify get_conversation raises ConversationNotFoundError when ID does not exist."""
    service = ConversationService(session_factory=mock_session_factory)
    unknown_id = uuid.uuid4()

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=None)

        with pytest.raises(ConversationNotFoundError):
            await service.get_conversation(unknown_id)


# -----------------------------------------------------------------------------
# Q&A Workflow Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_question_inactive_conversation_raises(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    """Verify ask_question raises ConversationInactiveError for archived conversations."""
    conv_id = uuid.uuid4()
    archived_conv = Conversation(id=conv_id, status="archived")

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=archived_conv)

        with pytest.raises(ConversationInactiveError) as exc_info:
            await service.ask_question(conv_id, "Explain attention.")

        assert "not active" in str(exc_info.value).lower()
        mock_retrieval_service.search.assert_not_called()
        mock_llm_client.generate_grounded_answer.assert_not_called()


@pytest.mark.asyncio
async def test_ask_question_empty_retrieval_skips_openai(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    """Verify that when no chunks match, OpenAI is NOT called and fallback is saved."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval_service.search.return_value = []

    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=1,
        role="user",
        content="What is BERT?",
    )
    asst_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=2,
        role="assistant",
        content=NO_EVIDENCE_RESPONSE,
        created_at=datetime.now(timezone.utc),
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(return_value=1)
        repo.create_message = AsyncMock(side_effect=[user_msg, asst_msg])
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "What is BERT?")

        assert response.answer == NO_EVIDENCE_RESPONSE
        assert len(response.citations) == 0
        assert response.model_name is None
        mock_llm_client.generate_grounded_answer.assert_not_called()
        assert repo.create_message.await_count == 2


@pytest.mark.asyncio
async def test_ask_question_retrieval_service_none_raises(
    mock_session_factory: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    """Verify asking a question when retrieval_service is None raises RetrievalUnavailableError."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=None,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])

        with pytest.raises(RetrievalUnavailableError) as exc_info:
            await service.ask_question(conv_id, "Question?")

        assert "unavailable or not initialized" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ask_question_retrieval_service_uninitialized_raises(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    """Verify asking a question when retrieval_service.is_initialized is False raises RetrievalUnavailableError."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")
    mock_retrieval_service.is_initialized = False

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])

        with pytest.raises(RetrievalUnavailableError) as exc_info:
            await service.ask_question(conv_id, "Question?")

        assert "unavailable or not initialized" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_ask_question_success_with_valid_citations(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
    sample_chunk: RetrievedChunk,
) -> None:
    """Verify successful Q&A flow with grounded LLM response and validated citations."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval_service.search.return_value = [sample_chunk]

    llm_result = LLMAnswerResult(
        answer="Multi-head attention maps queries and keys.",
        cited_chunk_ids=[str(sample_chunk.chunk_id)],
        model_name="gpt-4o-mini",
        token_count=120,
    )
    mock_llm_client.generate_grounded_answer.return_value = llm_result

    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=1,
        role="user",
        content="How does attention work?",
    )
    asst_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=2,
        role="assistant",
        content=llm_result.answer,
        model_name="gpt-4o-mini",
        created_at=datetime.now(timezone.utc),
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        repo.create_message = AsyncMock(side_effect=[user_msg, asst_msg])
        repo.get_chunk_page_information = AsyncMock(return_value={sample_chunk.chunk_id: 3})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "How does attention work?")

        assert response.answer == llm_result.answer
        assert response.model_name == "gpt-4o-mini"
        assert len(response.citations) == 1
        citation = response.citations[0]
        assert citation.chunk_id == sample_chunk.chunk_id
        assert citation.page_number == 3
        assert citation.paper_title == sample_chunk.paper_title
        assert citation.citation_order == 1

        # Verify citation data passed to repository
        repo.create_citations.assert_awaited_once()
        _, kwargs = repo.create_citations.call_args
        citations_saved = repo.create_citations.call_args[0][1]
        assert len(citations_saved) == 1
        assert citations_saved[0]["page_number"] == 3


@pytest.mark.asyncio
async def test_ask_question_unknown_citation_id_safely_ignored(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
    sample_chunk: RetrievedChunk,
) -> None:
    """Verify hallucinated/unknown chunk IDs from LLM are ignored, preserving valid citations."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval_service.search.return_value = [sample_chunk]

    hallucinated_id = str(uuid.uuid4())
    llm_result = LLMAnswerResult(
        answer="Answer with invalid citation.",
        cited_chunk_ids=[hallucinated_id, str(sample_chunk.chunk_id)],
        model_name="gpt-4o-mini",
    )
    mock_llm_client.generate_grounded_answer.return_value = llm_result

    user_msg = Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Q")
    asst_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=2,
        role="assistant",
        content="Answer",
        created_at=datetime.now(timezone.utc),
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        repo.create_message = AsyncMock(side_effect=[user_msg, asst_msg])
        repo.get_chunk_page_information = AsyncMock(return_value={sample_chunk.chunk_id: None})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "Q")

        # Only sample_chunk should be cited, hallucinated_id discarded
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == sample_chunk.chunk_id
        assert response.citations[0].page_number is None  # Handled null page gracefully


@pytest.mark.asyncio
async def test_ask_question_openai_failure_preserves_user_message_and_raises(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
    sample_chunk: RetrievedChunk,
) -> None:
    """Verify that if OpenAI fails, user message remains persisted and LLMGenerationError is raised."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval_service.search.return_value = [sample_chunk]
    mock_llm_client.generate_grounded_answer.side_effect = LLMClientError("Rate limit exceeded")

    user_msg = Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Q")

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(return_value=1)
        repo.create_message = AsyncMock(return_value=user_msg)
        repo.update_conversation_last_message_at = AsyncMock()

        with pytest.raises(LLMGenerationError):
            await service.ask_question(conv_id, "Q")

        # User message was saved before the network call
        repo.create_message.assert_awaited_once_with(
            conversation_id=conv_id,
            sequence_number=1,
            role="user",
            content="Q",
            status="completed",
            metadata_={"mode": "normal"},
        )


@pytest.mark.asyncio
async def test_ask_question_greeting_returns_deterministic_response_without_retrieval_or_llm(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    """Verify greetings receive a deterministic response without calling retrieval or LLM."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    user_msg = Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Hello")
    asst_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=2,
        role="assistant",
        content=GREETING_RESPONSE,
        created_at=datetime.now(timezone.utc),
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(return_value=1)
        repo.create_message = AsyncMock(side_effect=[user_msg, asst_msg])
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "Hello")

        assert response.answer == GREETING_RESPONSE
        assert len(response.citations) == 0
        assert response.model_name is None
        mock_retrieval_service.search.assert_not_called()
        mock_llm_client.generate_grounded_answer.assert_not_called()
        assert repo.create_message.await_count == 2


@pytest.mark.asyncio
async def test_ask_question_unsupported_general_question_declined(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    """Verify unsupported general questions receive polite scope restriction without retrieval or LLM."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    user_msg = Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="What is the capital of France?")
    asst_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=2,
        role="assistant",
        content=UNSUPPORTED_SCOPE_RESPONSE,
        created_at=datetime.now(timezone.utc),
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(return_value=1)
        repo.create_message = AsyncMock(side_effect=[user_msg, asst_msg])
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "What is the capital of France?")

        assert response.answer == UNSUPPORTED_SCOPE_RESPONSE
        assert len(response.citations) == 0
        assert response.model_name is None
        mock_retrieval_service.search.assert_not_called()
        mock_llm_client.generate_grounded_answer.assert_not_called()
        assert repo.create_message.await_count == 2
