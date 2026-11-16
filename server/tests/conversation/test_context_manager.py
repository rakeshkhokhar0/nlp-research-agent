"""Unit tests for ContextManager (history building, token budget, deduplication, summarization)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from app.modules.conversation.context import (
    BuiltContext,
    ContextManager,
    estimate_tokens,
)
from app.modules.conversation.models import Conversation, Message
from app.modules.conversation.repository import ConversationRepository


def create_mock_message(
    conv_id: uuid.UUID,
    seq: int,
    role: str,
    content: str,
) -> Message:
    msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=seq,
        role=role,
        content=content,
        status="completed",
    )
    return msg


@pytest.mark.asyncio
async def test_build_context_includes_recent_history_and_summary() -> None:
    """Verify build_context retrieves recent messages and attaches existing summary."""
    conv_id = uuid.uuid4()
    mock_repo = MagicMock(spec=ConversationRepository)

    messages = [
        create_mock_message(conv_id, 1, "user", "What is BERT?"),
        create_mock_message(conv_id, 2, "assistant", "BERT is a bidirectional transformer model."),
    ]
    mock_repo.get_recent_messages = AsyncMock(return_value=messages)

    manager = ContextManager(recent_message_limit=8, max_context_tokens=2000)
    context = await manager.build_context(
        conversation_id=conv_id,
        repo=mock_repo,
        current_question="How does it compare to GPT?",
        summary="Previous discussions covered transformer foundations.",
        chat_type="general",
    )

    assert context.conversation_id == conv_id
    assert context.summary == "Previous discussions covered transformer foundations."
    assert len(context.history_messages) == 2
    assert context.history_messages[0]["content"] == "What is BERT?"
    assert context.history_messages[1]["content"] == "BERT is a bidirectional transformer model."
    assert context.current_question == "How does it compare to GPT?"
    assert context.chat_type == "general"
    assert context.estimated_tokens > 0


@pytest.mark.asyncio
async def test_build_context_avoids_duplicating_current_user_message() -> None:
    """Verify if the current user message was already recorded in DB, it is not duplicated in history."""
    conv_id = uuid.uuid4()
    mock_repo = MagicMock(spec=ConversationRepository)

    q = "Explain self-attention"
    messages = [
        create_mock_message(conv_id, 1, "user", "Hello"),
        create_mock_message(conv_id, 2, "assistant", "Hi! How can I help?"),
        create_mock_message(conv_id, 3, "user", q),  # Already saved
    ]
    mock_repo.get_recent_messages = AsyncMock(return_value=messages)

    manager = ContextManager()
    context = await manager.build_context(
        conversation_id=conv_id,
        repo=mock_repo,
        current_question=q,
    )

    # The duplicated current question should be popped from history_messages
    assert len(context.history_messages) == 2
    assert context.history_messages[-1]["role"] == "assistant"
    assert context.current_question == q


@pytest.mark.asyncio
async def test_build_context_enforces_token_budget() -> None:
    """Verify oldest messages are dropped when total token estimate exceeds max_context_tokens."""
    conv_id = uuid.uuid4()
    mock_repo = MagicMock(spec=ConversationRepository)

    # 4 messages, each with ~100 characters (~25 tokens)
    messages = [
        create_mock_message(conv_id, 1, "user", "Message 1 " * 10),
        create_mock_message(conv_id, 2, "assistant", "Message 2 " * 10),
        create_mock_message(conv_id, 3, "user", "Message 3 " * 10),
        create_mock_message(conv_id, 4, "assistant", "Message 4 " * 10),
    ]
    mock_repo.get_recent_messages = AsyncMock(return_value=messages)

    # Budget that can only fit 2 recent messages plus the question
    manager = ContextManager(max_context_tokens=60)
    context = await manager.build_context(
        conversation_id=conv_id,
        repo=mock_repo,
        current_question="Short Q",
    )

    assert len(context.history_messages) < 4
    # Ensure chronological order is preserved for retained messages
    assert "Message 4" in context.history_messages[-1]["content"]


@pytest.mark.asyncio
async def test_maybe_summarize_history_triggers_when_threshold_exceeded() -> None:
    """Verify older messages are passed to LLM summarizer and result is persisted."""
    conv_id = uuid.uuid4()
    conv = Conversation(
        id=conv_id,
        title="Test Conversation",
        status="active",
        context_summary=None,
        metadata_={"last_summarized_sequence": 0},
    )

    # 12 messages: older 4 (seq 1..4), recent 8 (seq 5..12)
    all_messages = [
        create_mock_message(conv_id, i, "user" if i % 2 == 1 else "assistant", f"Content {i}")
        for i in range(1, 13)
    ]

    mock_repo = MagicMock(spec=ConversationRepository)
    mock_repo.get_conversation_messages = AsyncMock(return_value=all_messages)
    mock_repo.get_messages_for_summarization = AsyncMock(return_value=all_messages[:4])
    mock_repo.update_context_summary = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.summarize_messages = AsyncMock(return_value="Summary of first 4 messages.")

    manager = ContextManager(
        llm_client=mock_llm,
        recent_message_limit=8,
        summarization_threshold=10,
    )

    summary = await manager.maybe_summarize_history(conv, mock_repo)

    assert summary == "Summary of first 4 messages."
    assert conv.context_summary == "Summary of first 4 messages."
    assert conv.metadata_["last_summarized_sequence"] == 4

    mock_llm.summarize_messages.assert_awaited_once()
    mock_repo.update_context_summary.assert_awaited_once_with(
        conversation_id=conv_id,
        context_summary="Summary of first 4 messages.",
        last_summarized_sequence=4,
    )


@pytest.mark.asyncio
async def test_maybe_summarize_history_reuses_summary_when_no_new_older_messages() -> None:
    """Verify summarization is skipped and stored summary reused if cutoff <= last_summarized_sequence."""
    conv_id = uuid.uuid4()
    conv = Conversation(
        id=conv_id,
        title="Test Conversation",
        status="active",
        context_summary="Existing high quality summary.",
        metadata_={"last_summarized_sequence": 4},
    )

    # Same 12 messages: cutoff is 4 (12 - 8 = 4). Already summarized up to 4.
    all_messages = [
        create_mock_message(conv_id, i, "user" if i % 2 == 1 else "assistant", f"Content {i}")
        for i in range(1, 13)
    ]

    mock_repo = MagicMock(spec=ConversationRepository)
    mock_repo.get_conversation_messages = AsyncMock(return_value=all_messages)

    mock_llm = MagicMock()
    mock_llm.summarize_messages = AsyncMock()

    manager = ContextManager(
        llm_client=mock_llm,
        recent_message_limit=8,
        summarization_threshold=10,
    )

    summary = await manager.maybe_summarize_history(conv, mock_repo)

    assert summary == "Existing high quality summary."
    mock_llm.summarize_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_summarize_history_gracefully_falls_back_on_llm_error() -> None:
    """Verify summarizer failure does not raise and returns existing summary or None."""
    conv_id = uuid.uuid4()
    conv = Conversation(
        id=conv_id,
        title="Test Conversation",
        status="active",
        context_summary="Fallback summary",
        metadata_={"last_summarized_sequence": 0},
    )

    all_messages = [
        create_mock_message(conv_id, i, "user" if i % 2 == 1 else "assistant", f"Content {i}")
        for i in range(1, 13)
    ]

    mock_repo = MagicMock(spec=ConversationRepository)
    mock_repo.get_conversation_messages = AsyncMock(return_value=all_messages)
    mock_repo.get_messages_for_summarization = AsyncMock(return_value=all_messages[:4])

    mock_llm = MagicMock()
    mock_llm.summarize_messages = AsyncMock(side_effect=RuntimeError("OpenAI 500 error"))

    manager = ContextManager(llm_client=mock_llm, recent_message_limit=8, summarization_threshold=10)
    summary = await manager.maybe_summarize_history(conv, mock_repo)

    assert summary == "Fallback summary"
    # Metadata should NOT be updated to cutoff if summarization failed
    assert conv.metadata_["last_summarized_sequence"] == 0


@pytest.mark.asyncio
async def test_incremental_summarization_two_cycles() -> None:
    """Verify two consecutive summarization cycles preserve entire conversation context.

    Verifies:
    1. Cycle 1: initial older messages (seq 1..4) are summarized and persisted.
    2. Sequence watermark advances to 4.
    3. New messages accumulate (seq 13..18, total 18).
    4. Cycle 2: summarizer receives previous summary and only newly eligible messages (seq 5..10).
    5. Previously summarized messages (1..4) are not summarized again.
    6. Updated summary is persisted, incorporating old and new context.
    7. Sequence watermark advances to 10.
    8. Recent messages (seq 11..18) remain available separately in build_context.
    """
    conv_id = uuid.uuid4()
    conv = Conversation(
        id=conv_id,
        title="Incremental Summarization Conversation",
        status="active",
        context_summary=None,
        metadata_={"last_summarized_sequence": 0},
    )

    # Initial 12 messages: older 4 (seq 1..4), recent 8 (seq 5..12)
    cycle1_messages = [
        create_mock_message(conv_id, i, "user" if i % 2 == 1 else "assistant", f"Message {i}")
        for i in range(1, 13)
    ]

    mock_repo = MagicMock(spec=ConversationRepository)
    mock_repo.get_conversation_messages = AsyncMock(return_value=cycle1_messages)
    mock_repo.get_messages_for_summarization = AsyncMock(return_value=cycle1_messages[:4])
    mock_repo.update_context_summary = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.summarize_messages = AsyncMock(
        side_effect=[
            "Cycle 1 Summary: Covers topics from messages 1 to 4.",
            "Cycle 2 Summary: Covers topics from messages 1 to 10 (merged).",
        ]
    )

    manager = ContextManager(
        llm_client=mock_llm,
        recent_message_limit=8,
        summarization_threshold=10,
        min_summarize_batch=2,
    )

    # --- Cycle 1 Execution ---
    summary_cycle1 = await manager.maybe_summarize_history(conv, mock_repo)

    assert summary_cycle1 == "Cycle 1 Summary: Covers topics from messages 1 to 4."
    assert conv.context_summary == "Cycle 1 Summary: Covers topics from messages 1 to 4."
    assert conv.metadata_["last_summarized_sequence"] == 4

    # Verify LLM was called with no existing summary and msgs 1..4
    mock_llm.summarize_messages.assert_awaited_once()
    cycle1_call = mock_llm.summarize_messages.call_args
    assert cycle1_call.kwargs["existing_summary"] is None
    assert len(cycle1_call.kwargs["messages"]) == 4
    assert [m["content"] for m in cycle1_call.kwargs["messages"]] == [
        "Message 1",
        "Message 2",
        "Message 3",
        "Message 4",
    ]

    mock_repo.update_context_summary.assert_awaited_once_with(
        conversation_id=conv_id,
        context_summary="Cycle 1 Summary: Covers topics from messages 1 to 4.",
        last_summarized_sequence=4,
    )

    # --- Simulate new messages accumulating: 6 new messages added (seq 13..18) ---
    all_18_messages = [
        create_mock_message(conv_id, i, "user" if i % 2 == 1 else "assistant", f"Message {i}")
        for i in range(1, 19)
    ]
    cycle2_unsummarized = all_18_messages[4:10]  # seq 5..10 (indices 4..9)
    assert len(cycle2_unsummarized) == 6
    assert cycle2_unsummarized[0].sequence_number == 5
    assert cycle2_unsummarized[-1].sequence_number == 10

    mock_repo.get_conversation_messages = AsyncMock(return_value=all_18_messages)
    mock_repo.get_messages_for_summarization = AsyncMock(return_value=cycle2_unsummarized)
    mock_repo.update_context_summary.reset_mock()

    # --- Cycle 2 Execution ---
    summary_cycle2 = await manager.maybe_summarize_history(conv, mock_repo)

    assert summary_cycle2 == "Cycle 2 Summary: Covers topics from messages 1 to 10 (merged)."
    assert conv.context_summary == "Cycle 2 Summary: Covers topics from messages 1 to 10 (merged)."
    assert conv.metadata_["last_summarized_sequence"] == 10

    # Verify get_messages_for_summarization was asked strictly for sequences 5..10
    mock_repo.get_messages_for_summarization.assert_awaited_once_with(
        conversation_id=conv_id,
        min_sequence=5,
        max_sequence=10,
    )

    # Verify Cycle 2 LLM call:
    # 1. previous summary is passed in
    # 2. previously summarized messages (1..4) are NOT passed
    # 3. newly eligible messages (5..10) ARE included
    assert mock_llm.summarize_messages.await_count == 2
    cycle2_call = mock_llm.summarize_messages.call_args
    assert cycle2_call.kwargs["existing_summary"] == "Cycle 1 Summary: Covers topics from messages 1 to 4."
    cycle2_contents = [m["content"] for m in cycle2_call.kwargs["messages"]]
    assert cycle2_contents == ["Message 5", "Message 6", "Message 7", "Message 8", "Message 9", "Message 10"]
    assert "Message 1" not in cycle2_contents
    assert "Message 4" not in cycle2_contents

    # Verify repository persistence: summary and updated sequence watermark 10
    mock_repo.update_context_summary.assert_awaited_once_with(
        conversation_id=conv_id,
        context_summary="Cycle 2 Summary: Covers topics from messages 1 to 10 (merged).",
        last_summarized_sequence=10,
    )

    # --- Verify build_context: recent messages remain available separately ---
    recent_8_messages = all_18_messages[10:]  # seq 11..18
    assert len(recent_8_messages) == 8
    assert recent_8_messages[0].sequence_number == 11
    assert recent_8_messages[-1].sequence_number == 18

    mock_repo.get_recent_messages = AsyncMock(return_value=recent_8_messages)
    built_context = await manager.build_context(
        conversation_id=conv_id,
        repo=mock_repo,
        current_question="Question 19?",
        summary=conv.context_summary,
        chat_type="general",
    )

    assert built_context.summary == "Cycle 2 Summary: Covers topics from messages 1 to 10 (merged)."
    assert len(built_context.history_messages) == 8
    assert built_context.history_messages[0]["content"] == "Message 11"
    assert built_context.history_messages[-1]["content"] == "Message 18"


@pytest.mark.asyncio
async def test_summarization_failure_does_not_destroy_previous_summary() -> None:
    """Verify that an LLM failure during a later summarization cycle does not destroy previous summary."""
    conv_id = uuid.uuid4()
    previous_summary = "High-quality summary of messages 1-4."
    conv = Conversation(
        id=conv_id,
        title="Test Conversation",
        status="active",
        context_summary=previous_summary,
        metadata_={"last_summarized_sequence": 4},
    )

    # 18 messages: older 10 (seq 1..10), cutoff 10. Unsummarized: 5..10.
    all_messages = [
        create_mock_message(conv_id, i, "user" if i % 2 == 1 else "assistant", f"Content {i}")
        for i in range(1, 19)
    ]

    mock_repo = MagicMock(spec=ConversationRepository)
    mock_repo.get_conversation_messages = AsyncMock(return_value=all_messages)
    mock_repo.get_messages_for_summarization = AsyncMock(return_value=all_messages[4:10])
    mock_repo.update_context_summary = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.summarize_messages = AsyncMock(side_effect=RuntimeError("OpenAI 503 Service Unavailable"))

    manager = ContextManager(
        llm_client=mock_llm,
        recent_message_limit=8,
        summarization_threshold=10,
        min_summarize_batch=2,
    )

    # Execute summarization when LLM throws
    result = await manager.maybe_summarize_history(conv, mock_repo)

    # Verify previous summary is retained and returned
    assert result == previous_summary
    assert conv.context_summary == previous_summary
    # Watermark remains at previous sequence (4), not corrupted
    assert conv.metadata_["last_summarized_sequence"] == 4
    # Repository update was not called
    mock_repo.update_context_summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarization_empty_output_preserves_previous_summary() -> None:
    """Verify that if summarizer produces an empty string or whitespace, previous summary is preserved."""
    conv_id = uuid.uuid4()
    previous_summary = "Valid summary of early messages."
    conv = Conversation(
        id=conv_id,
        title="Test Conversation",
        status="active",
        context_summary=previous_summary,
        metadata_={"last_summarized_sequence": 4},
    )

    all_messages = [
        create_mock_message(conv_id, i, "user" if i % 2 == 1 else "assistant", f"Content {i}")
        for i in range(1, 19)
    ]

    mock_repo = MagicMock(spec=ConversationRepository)
    mock_repo.get_conversation_messages = AsyncMock(return_value=all_messages)
    mock_repo.get_messages_for_summarization = AsyncMock(return_value=all_messages[4:10])
    mock_repo.update_context_summary = AsyncMock()

    mock_llm = MagicMock()
    mock_llm.summarize_messages = AsyncMock(return_value="   ")

    manager = ContextManager(
        llm_client=mock_llm,
        recent_message_limit=8,
        summarization_threshold=10,
        min_summarize_batch=2,
    )

    result = await manager.maybe_summarize_history(conv, mock_repo)

    assert result == previous_summary
    assert conv.context_summary == previous_summary
    assert conv.metadata_["last_summarized_sequence"] == 4
    mock_repo.update_context_summary.assert_not_awaited()
