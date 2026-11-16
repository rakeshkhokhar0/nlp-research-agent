"""Comprehensive tests for MVP Context Management, General Chat, and Chat with Paper."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.modules.conversation.context import BuiltContext, ContextManager
from app.modules.conversation.llm_client import (
    ADVANCED_SYSTEM_INSTRUCTIONS,
    SYSTEM_INSTRUCTIONS,
    LLMAnswerResult,
    LLMClientError,
    OpenAIResponseClient,
)
from app.modules.conversation.models import Conversation, Message, MessageCitation
from app.modules.conversation.repository import ConversationRepository
from app.modules.conversation.service import (
    NO_EVIDENCE_RESPONSE,
    ConversationService,
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
    client.generate_general_answer = AsyncMock()
    client.summarize_messages = AsyncMock()
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
# 1. Recent history is included
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_recent_history_is_included(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
    sample_chunk: RetrievedChunk,
) -> None:
    """Verify that recent conversation history (up to limit) is retrieved and passed to the LLM."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(
        id=conv_id,
        status="active",
        context_summary=None,
        metadata_={"last_summarized_sequence": 0},
    )

    history_msg1 = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=1,
        role="user",
        content="What is an attention head?",
        status="completed",
    )
    history_msg2 = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=2,
        role="assistant",
        content="An attention head computes scaled dot-product attention.",
        status="completed",
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    mock_retrieval_service.search.return_value = [sample_chunk]
    mock_llm_client.generate_grounded_answer.return_value = LLMAnswerResult(
        answer="Multi-head attention allows joint attending to information.",
        cited_chunk_ids=[str(sample_chunk.chunk_id)],
        model_name="gpt-4o-mini",
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_conversation_messages = AsyncMock(return_value=[history_msg1, history_msg2])
        repo.get_recent_messages = AsyncMock(return_value=[history_msg1, history_msg2])
        repo.get_next_sequence_number = AsyncMock(side_effect=[3, 4])
        repo.create_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4(), created_at=datetime.now(timezone.utc)))
        repo.get_chunk_page_information = AsyncMock(return_value={sample_chunk.chunk_id: 3})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(
            conversation_id=conv_id,
            question="Why are multiple heads beneficial?",
            chat_type="paper",
        )

        assert response.answer == "Multi-head attention allows joint attending to information."
        mock_llm_client.generate_grounded_answer.assert_awaited_once()
        passed_history = mock_llm_client.generate_grounded_answer.call_args.kwargs["conversation_history"]
        assert len(passed_history) == 2
        assert passed_history[0]["content"] == "What is an attention head?"
        assert passed_history[1]["content"] == "An attention head computes scaled dot-product attention."


# -----------------------------------------------------------------------------
# 2. Stored summary is included
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stored_summary_is_included(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
    sample_chunk: RetrievedChunk,
) -> None:
    """Verify that stored conversation.context_summary is passed to LLM generation."""
    conv_id = uuid.uuid4()
    stored_summary = "User previously explored self-attention basics and transformer dimensions."
    active_conv = Conversation(
        id=conv_id,
        status="active",
        context_summary=stored_summary,
        metadata_={"last_summarized_sequence": 4},
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    mock_retrieval_service.search.return_value = [sample_chunk]
    mock_llm_client.generate_grounded_answer.return_value = LLMAnswerResult(
        answer="Positional encodings provide sequence order information.",
        cited_chunk_ids=[str(sample_chunk.chunk_id)],
        model_name="gpt-4o-mini",
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_conversation_messages = AsyncMock(return_value=[])
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        repo.create_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4(), created_at=datetime.now(timezone.utc)))
        repo.get_chunk_page_information = AsyncMock(return_value={sample_chunk.chunk_id: 4})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        await service.ask_question(
            conversation_id=conv_id,
            question="How does the model know word order?",
            chat_type="paper",
        )

        mock_llm_client.generate_grounded_answer.assert_awaited_once()
        passed_summary = mock_llm_client.generate_grounded_answer.call_args.kwargs["conversation_summary"]
        assert passed_summary == stored_summary


# -----------------------------------------------------------------------------
# 3. Old history can be summarized
# 4. Summary is persisted and reused
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_old_history_summarized_persisted_and_reused(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
    sample_chunk: RetrievedChunk,
) -> None:
    """Verify that older messages are summarized, persisted, and reused without re-summarizing."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(
        id=conv_id,
        status="active",
        context_summary=None,
        metadata_={"last_summarized_sequence": 0},
    )

    # 12 completed messages in history (threshold=10, recent=8, cutoff=4)
    all_msgs = [
        Message(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            sequence_number=i,
            role="user" if i % 2 == 1 else "assistant",
            content=f"Message {i}",
            status="completed",
        )
        for i in range(1, 13)
    ]

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    mock_llm_client.summarize_messages.return_value = "Summary of messages 1-4."
    mock_retrieval_service.search.return_value = [sample_chunk]
    mock_llm_client.generate_grounded_answer.return_value = LLMAnswerResult(
        answer="Answer to question.",
        cited_chunk_ids=[str(sample_chunk.chunk_id)],
        model_name="gpt-4o-mini",
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_conversation_messages = AsyncMock(return_value=all_msgs)
        repo.get_messages_for_summarization = AsyncMock(return_value=all_msgs[:4])
        repo.get_recent_messages = AsyncMock(return_value=all_msgs[4:])
        repo.update_context_summary = AsyncMock()
        repo.get_next_sequence_number = AsyncMock(side_effect=[13, 14, 15, 16])
        repo.create_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4(), created_at=datetime.now(timezone.utc)))
        repo.get_chunk_page_information = AsyncMock(return_value={sample_chunk.chunk_id: 1})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        # Turn 1: triggers summarization
        await service.ask_question(conv_id, "Question 13", chat_type="paper")

        mock_llm_client.summarize_messages.assert_awaited_once()
        repo.update_context_summary.assert_awaited_once_with(
            conversation_id=conv_id,
            context_summary="Summary of messages 1-4.",
            last_summarized_sequence=4,
        )
        assert active_conv.context_summary == "Summary of messages 1-4."
        assert active_conv.metadata_["last_summarized_sequence"] == 4

        # Turn 2: cutoff is still 4, so summarization must NOT be re-invoked
        mock_llm_client.summarize_messages.reset_mock()
        await service.ask_question(conv_id, "Question 14", chat_type="paper")

        mock_llm_client.summarize_messages.assert_not_awaited()
        # Stored summary was reused in the prompt
        passed_summary = mock_llm_client.generate_grounded_answer.call_args.kwargs["conversation_summary"]
        assert passed_summary == "Summary of messages 1-4."


@pytest.mark.asyncio
async def test_service_incremental_summarization_two_cycles(
    mock_session_factory: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    """Verify ConversationService handles two consecutive summarization cycles across turns.

    Verifies:
    1. Turn 1 triggers Cycle 1 summarization (msgs 1..4).
    2. Turn 2 reuses stored summary without re-summarizing.
    3. Turn 3 triggers Cycle 2 summarization (previous summary passed in, msgs 5..10 summarized).
    4. Sequence watermark advances from 4 to 10.
    5. Final answer generation LLM receives updated summary and recent messages.
    """
    conv_id = uuid.uuid4()
    active_conv = Conversation(
        id=conv_id,
        status="active",
        context_summary=None,
        metadata_={"chat_type": "general", "last_summarized_sequence": 0},
    )

    # Initial 12 messages: cutoff 4
    msgs_12 = [
        Message(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            sequence_number=i,
            role="user" if i % 2 == 1 else "assistant",
            content=f"Message {i}",
            status="completed",
        )
        for i in range(1, 13)
    ]

    service = ConversationService(
        session_factory=mock_session_factory,
        llm_client=mock_llm_client,
    )

    mock_llm_client.summarize_messages.side_effect = [
        "Summary Cycle 1 (msgs 1-4).",
        "Summary Cycle 2 (msgs 1-10 merged).",
    ]
    mock_llm_client.generate_general_answer.return_value = LLMAnswerResult(
        answer="General answer response.",
        cited_chunk_ids=[],
        model_name="gpt-4o-mini",
    )

    # Messages when Turn 3 occurs: 18 messages (seq 1..18)
    msgs_18 = [
        Message(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            sequence_number=i,
            role="user" if i % 2 == 1 else "assistant",
            content=f"Message {i}",
            status="completed",
        )
        for i in range(1, 19)
    ]

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        # Turn 1: 12 msgs in DB
        repo.get_conversation_messages = AsyncMock(return_value=msgs_12)
        repo.get_messages_for_summarization = AsyncMock(return_value=msgs_12[:4])
        repo.get_recent_messages = AsyncMock(return_value=msgs_12[4:])
        repo.update_context_summary = AsyncMock()
        repo.get_next_sequence_number = AsyncMock(side_effect=[13, 14, 15, 16, 19, 20])
        repo.create_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4(), created_at=datetime.now(timezone.utc)))
        repo.update_conversation_last_message_at = AsyncMock()

        # Turn 1: Triggers Cycle 1
        await service.ask_question(conv_id, "Turn 1 question", chat_type="general")

        assert mock_llm_client.summarize_messages.await_count == 1
        cycle1_call = mock_llm_client.summarize_messages.call_args
        assert cycle1_call.kwargs["existing_summary"] is None
        assert len(cycle1_call.kwargs["messages"]) == 4

        repo.update_context_summary.assert_awaited_once_with(
            conversation_id=conv_id,
            context_summary="Summary Cycle 1 (msgs 1-4).",
            last_summarized_sequence=4,
        )
        assert active_conv.context_summary == "Summary Cycle 1 (msgs 1-4)."
        assert active_conv.metadata_["last_summarized_sequence"] == 4

        # Turn 2: Reuses summary (still 12 msgs considered, cutoff 4 <= 4)
        mock_llm_client.summarize_messages.reset_mock()
        await service.ask_question(conv_id, "Turn 2 question", chat_type="general")

        mock_llm_client.summarize_messages.assert_not_awaited()
        assert mock_llm_client.generate_general_answer.call_args.kwargs["conversation_summary"] == "Summary Cycle 1 (msgs 1-4)."

        # Turn 3: 18 messages now in history (cutoff is 10, watermark was 4, eligible is 5..10)
        repo.get_conversation_messages = AsyncMock(return_value=msgs_18)
        repo.get_messages_for_summarization = AsyncMock(return_value=msgs_18[4:10])
        repo.get_recent_messages = AsyncMock(return_value=msgs_18[10:])
        repo.update_context_summary.reset_mock()

        await service.ask_question(conv_id, "Turn 3 question", chat_type="general")

        # Cycle 2 triggered!
        assert mock_llm_client.summarize_messages.await_count == 1
        cycle2_call = mock_llm_client.summarize_messages.call_args
        assert cycle2_call.kwargs["existing_summary"] == "Summary Cycle 1 (msgs 1-4)."
        cycle2_contents = [m["content"] for m in cycle2_call.kwargs["messages"]]
        assert cycle2_contents == [f"Message {i}" for i in range(5, 11)]

        repo.update_context_summary.assert_awaited_once_with(
            conversation_id=conv_id,
            context_summary="Summary Cycle 2 (msgs 1-10 merged).",
            last_summarized_sequence=10,
        )
        assert active_conv.context_summary == "Summary Cycle 2 (msgs 1-10 merged)."
        assert active_conv.metadata_["last_summarized_sequence"] == 10

        # Verify LLM generation received the updated summary and recent messages
        last_answer_call = mock_llm_client.generate_general_answer.call_args
        assert last_answer_call.kwargs["conversation_summary"] == "Summary Cycle 2 (msgs 1-10 merged)."
        history_passed = last_answer_call.kwargs["conversation_history"]
        assert len(history_passed) == 8
        assert history_passed[0]["content"] == "Message 11"
        assert history_passed[-1]["content"] == "Message 18"


# -----------------------------------------------------------------------------
# 5. General Chat handles follow-up questions
# 6. General Chat does not call retrieval
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_general_chat_handles_follow_up_and_does_not_call_retrieval(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    """Verify General Chat answers follow-up questions using context without calling retrieval."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(
        id=conv_id,
        status="active",
        context_summary="User asked about NLP tokenization.",
        metadata_={"chat_type": "general"},
    )

    recent_history = [
        Message(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            sequence_number=1,
            role="user",
            content="What is subword tokenization?",
            status="completed",
        ),
        Message(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            sequence_number=2,
            role="assistant",
            content="Subword tokenization splits words into frequent substrings (e.g. BPE or WordPiece).",
            status="completed",
        ),
    ]

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    mock_llm_client.generate_general_answer.return_value = LLMAnswerResult(
        answer="Its main advantage is balancing vocabulary size and handling out-of-vocabulary words.",
        cited_chunk_ids=[],
        model_name="gpt-4o-mini",
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_conversation_messages = AsyncMock(return_value=recent_history)
        repo.get_recent_messages = AsyncMock(return_value=recent_history)
        repo.get_next_sequence_number = AsyncMock(side_effect=[3, 4])
        repo.create_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4(), created_at=datetime.now(timezone.utc)))
        repo.update_conversation_last_message_at = AsyncMock()

        # Follow-up question referring to "it"
        response = await service.ask_question(
            conversation_id=conv_id,
            question="What is its primary advantage?",
            chat_type="general",
        )

        # Retrieval was NEVER called
        mock_retrieval_service.search.assert_not_called()

        # General LLM answer was called with context
        mock_llm_client.generate_general_answer.assert_awaited_once()
        passed_q = mock_llm_client.generate_general_answer.call_args.kwargs["question"]
        passed_h = mock_llm_client.generate_general_answer.call_args.kwargs["conversation_history"]
        passed_s = mock_llm_client.generate_general_answer.call_args.kwargs["conversation_summary"]

        assert passed_q == "What is its primary advantage?"
        assert len(passed_h) == 2
        assert passed_s == "User asked about NLP tokenization."
        assert response.answer == "Its main advantage is balancing vocabulary size and handling out-of-vocabulary words."
        assert response.citations == []
        assert response.chat_type == "general"


# -----------------------------------------------------------------------------
# 7. Chat with Paper includes retrieved evidence
# 8. Citations remain validated and persisted
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_chat_with_paper_includes_evidence_and_validates_citations(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
    sample_chunk: RetrievedChunk,
) -> None:
    """Verify Chat with Paper invokes retrieval, passes chunks to LLM, and persists validated citations."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active", context_summary=None)

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    mock_retrieval_service.search.return_value = [sample_chunk]
    mock_llm_client.generate_grounded_answer.return_value = LLMAnswerResult(
        answer="Multi-head attention projects queries, keys, and values h times.",
        cited_chunk_ids=[str(sample_chunk.chunk_id)],
        model_name="gpt-4o-mini",
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_conversation_messages = AsyncMock(return_value=[])
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        asst_id = uuid.uuid4()
        repo.create_message = AsyncMock(side_effect=[
            MagicMock(id=uuid.uuid4(), created_at=datetime.now(timezone.utc)),
            MagicMock(id=asst_id, created_at=datetime.now(timezone.utc)),
        ])
        repo.get_chunk_page_information = AsyncMock(return_value={sample_chunk.chunk_id: 5})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(
            conversation_id=conv_id,
            question="How does multi-head attention project representations?",
            chat_type="paper",
        )

        # Retrieval was called
        mock_retrieval_service.search.assert_called_once_with(
            query="How does multi-head attention project representations?",
            top_k=5,
        )

        # Evidence was passed to LLM
        mock_llm_client.generate_grounded_answer.assert_awaited_once()
        passed_evidence = mock_llm_client.generate_grounded_answer.call_args.kwargs["evidence_chunks"]
        assert len(passed_evidence) == 1
        assert passed_evidence[0].chunk_id == sample_chunk.chunk_id

        # Citations validated and persisted
        repo.create_citations.assert_awaited_once()
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == sample_chunk.chunk_id
        assert response.citations[0].paper_title == "Attention Is All You Need"
        assert response.citations[0].page_number == 5
        assert response.chat_type == "paper"


# -----------------------------------------------------------------------------
# 9. Summary failure does not break chat
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_summary_failure_does_not_break_chat(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
    sample_chunk: RetrievedChunk,
) -> None:
    """Verify that when summarization raises an error, chat continues gracefully using recent history."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(
        id=conv_id,
        status="active",
        context_summary="Prior summary",
        metadata_={"last_summarized_sequence": 0},
    )

    all_msgs = [
        Message(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            sequence_number=i,
            role="user" if i % 2 == 1 else "assistant",
            content=f"Msg {i}",
            status="completed",
        )
        for i in range(1, 13)
    ]

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    # Summarizer throws exception
    mock_llm_client.summarize_messages.side_effect = LLMClientError("OpenAI 429 rate limit on summarizer")
    mock_retrieval_service.search.return_value = [sample_chunk]
    mock_llm_client.generate_grounded_answer.return_value = LLMAnswerResult(
        answer="Chat succeeded despite summarizer failure.",
        cited_chunk_ids=[str(sample_chunk.chunk_id)],
        model_name="gpt-4o-mini",
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_conversation_messages = AsyncMock(return_value=all_msgs)
        repo.get_messages_for_summarization = AsyncMock(return_value=all_msgs[:4])
        repo.get_recent_messages = AsyncMock(return_value=all_msgs[4:])
        repo.get_next_sequence_number = AsyncMock(side_effect=[13, 14])
        repo.create_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4(), created_at=datetime.now(timezone.utc)))
        repo.get_chunk_page_information = AsyncMock(return_value={sample_chunk.chunk_id: 1})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "Q13", chat_type="paper")

        assert response.answer == "Chat succeeded despite summarizer failure."
        # Stored summary fallback was passed
        passed_summary = mock_llm_client.generate_grounded_answer.call_args.kwargs["conversation_summary"]
        assert passed_summary == "Prior summary"


# -----------------------------------------------------------------------------
# 10. Normal and Advanced modes continue working
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_normal_and_advanced_modes_continue_working(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
    sample_chunk: RetrievedChunk,
) -> None:
    """Verify both normal and advanced modes forward their respective flags to LLM client."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active", context_summary="Test summary")

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    mock_retrieval_service.search.return_value = [sample_chunk]
    mock_llm_client.generate_grounded_answer.return_value = LLMAnswerResult(
        answer="Normal answer.",
        cited_chunk_ids=[str(sample_chunk.chunk_id)],
        model_name="gpt-4o-mini",
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_conversation_messages = AsyncMock(return_value=[])
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2, 3, 4])
        repo.create_message = AsyncMock(return_value=MagicMock(id=uuid.uuid4(), created_at=datetime.now(timezone.utc)))
        repo.get_chunk_page_information = AsyncMock(return_value={sample_chunk.chunk_id: 2})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        # Turn 1: Normal mode
        res_normal = await service.ask_question(conv_id, "Explain self-attention", mode="normal")
        assert res_normal.mode == "normal"
        assert mock_llm_client.generate_grounded_answer.call_args.kwargs["mode"] == "normal"

        # Turn 2: Advanced mode
        mock_llm_client.generate_grounded_answer.return_value = LLMAnswerResult(
            answer="Advanced detailed explanation.",
            cited_chunk_ids=[str(sample_chunk.chunk_id)],
            model_name="gpt-4o-mini",
        )
        res_advanced = await service.ask_question(conv_id, "Explain self-attention in detail", mode="advanced")
        assert res_advanced.mode == "advanced"
        assert mock_llm_client.generate_grounded_answer.call_args.kwargs["mode"] == "advanced"


# -----------------------------------------------------------------------------
# 11. Conversation Type Consistency (Task 2)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_conversation_type_consistency_rejects_conflicting_type(
    mock_session_factory: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_llm_client: MagicMock,
) -> None:
    """Verify that a conversation locked to 'general' rejects 'paper', and vice versa."""
    conv_id = uuid.uuid4()
    gen_conv = Conversation(
        id=conv_id,
        status="active",
        metadata_={"chat_type": "general"},
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval_service,
        llm_client=mock_llm_client,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=gen_conv)

        # Attempting to send paper question to a general conversation must raise ValueError
        with pytest.raises(ValueError, match="is locked to chat_type 'general'"):
            await service.ask_question(conv_id, "Question", chat_type="paper")

        # Now test paper conversation rejecting general
        paper_conv = Conversation(
            id=conv_id,
            status="active",
            metadata_={"chat_type": "paper"},
        )
        repo.get_conversation = AsyncMock(return_value=paper_conv)

        with pytest.raises(ValueError, match="is locked to chat_type 'paper'"):
            await service.ask_question(conv_id, "Question", chat_type="general")
