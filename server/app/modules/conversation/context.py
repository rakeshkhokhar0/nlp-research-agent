"""Lightweight conversation context manager for General Chat and Chat with Paper."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any, Literal, Sequence
import uuid

from app.modules.conversation.models import Conversation, Message

if TYPE_CHECKING:
    from app.modules.conversation.llm_client import OpenAIResponseClient
    from app.modules.conversation.repository import ConversationRepository

logger = logging.getLogger(__name__)

DEFAULT_RECENT_MESSAGE_LIMIT = 8
DEFAULT_MAX_CONTEXT_TOKENS = 3000
DEFAULT_SUMMARIZATION_THRESHOLD = 10
MIN_SUMMARIZE_BATCH = 2


def estimate_tokens(text: str) -> int:
    """Estimate token count using ~4 characters per token heuristic."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class BuiltContext:
    """Structured context built for LLM prompt assembly."""

    conversation_id: uuid.UUID
    summary: str | None
    history_messages: list[dict[str, str]] = field(default_factory=list)
    current_question: str = ""
    estimated_tokens: int = 0
    chat_type: Literal["general", "paper"] = "paper"


class ContextManager:
    """Manages short-term recent history, stored conversation summaries,

    and token budget enforcement for both General Chat and Chat with Paper.
    """

    def __init__(
        self,
        llm_client: OpenAIResponseClient | None = None,
        recent_message_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        summarization_threshold: int = DEFAULT_SUMMARIZATION_THRESHOLD,
        min_summarize_batch: int = MIN_SUMMARIZE_BATCH,
    ) -> None:
        """Initialize the context manager.

        Args:
            llm_client: Optional OpenAIResponseClient used for summarization.
            recent_message_limit: Number of recent messages to retain unsummarized (default 8).
            max_context_tokens: Total token budget for summary + recent history + question.
            summarization_threshold: Total messages before summarization triggers (default 10).
            min_summarize_batch: Minimum unsummarized messages required to trigger a new summary pass.
        """
        self.llm_client = llm_client
        self.recent_message_limit = recent_message_limit
        self.max_context_tokens = max_context_tokens
        self.summarization_threshold = summarization_threshold
        self.min_summarize_batch = min_summarize_batch

    async def maybe_summarize_history(
        self,
        conversation: Conversation,
        repo: ConversationRepository,
    ) -> str | None:
        """Summarize older messages if history exceeds the configured threshold.

        Saves summary into conversation.context_summary and records last_summarized_sequence
        in conversation.metadata_ to prevent repeated summarization.
        Falls back gracefully if LLM summarization fails.
        """
        if self.llm_client is None:
            return conversation.context_summary

        try:
            # 1. Fetch completed messages for this conversation
            messages = await repo.get_conversation_messages(
                conversation.id,
                include_citations=False,
            )
            if not isinstance(messages, (list, tuple)):
                return conversation.context_summary

            completed_messages = [
                m for m in messages
                if getattr(m, "status", "completed") == "completed"
            ]

            total_count = len(completed_messages)
            if total_count < self.summarization_threshold:
                return conversation.context_summary

            # 2. Older messages are those prior to the recent window
            older_count = total_count - self.recent_message_limit
            if older_count <= 0:
                return conversation.context_summary

            cutoff_msg = completed_messages[older_count - 1]
            cutoff_sequence = cutoff_msg.sequence_number

            # Check what sequence has already been summarized
            metadata = dict(conversation.metadata_ or {})
            last_summarized_sequence = int(metadata.get("last_summarized_sequence", 0))

            if cutoff_sequence <= last_summarized_sequence:
                # Already summarized up to or past the cutoff
                return conversation.context_summary

            unsummarized_count = cutoff_sequence - last_summarized_sequence
            if unsummarized_count < self.min_summarize_batch and conversation.context_summary:
                # Wait for minimum batch size before re-summarizing
                return conversation.context_summary

            # 3. Retrieve eligible unsummarized messages
            unsummarized_msgs = await repo.get_messages_for_summarization(
                conversation_id=conversation.id,
                min_sequence=last_summarized_sequence + 1,
                max_sequence=cutoff_sequence,
            )
            if not unsummarized_msgs:
                return conversation.context_summary

            msgs_payload = [
                {"role": m.role, "content": m.content}
                for m in unsummarized_msgs
                if m.content and m.content.strip()
            ]
            if not msgs_payload:
                return conversation.context_summary

            # 4. Generate summary using LLM
            logger.info(
                "Triggering summarization for conversation %s (messages %d..%d)",
                conversation.id,
                last_summarized_sequence + 1,
                cutoff_sequence,
            )
            new_summary = await self.llm_client.summarize_messages(
                messages=msgs_payload,
                existing_summary=conversation.context_summary,
            )

            # Defensive check: ensure summarizer produced valid text
            if not new_summary or not str(new_summary).strip():
                logger.warning(
                    "Summarizer returned empty summary text; preserving previous summary for conversation %s",
                    conversation.id,
                )
                return conversation.context_summary

            clean_summary = str(new_summary).strip()

            # 5. Persist to database
            await repo.update_context_summary(
                conversation_id=conversation.id,
                context_summary=clean_summary,
                last_summarized_sequence=cutoff_sequence,
            )
            conversation.context_summary = clean_summary
            metadata["last_summarized_sequence"] = cutoff_sequence
            conversation.metadata_ = metadata

            return clean_summary
        except Exception as exc:
            # Graceful fallback: do not break the chat flow
            logger.warning(
                "Conversation summarization failed gracefully: %s. Using recent history.",
                type(exc).__name__,
            )
            return conversation.context_summary

    async def build_context(
        self,
        conversation_id: uuid.UUID,
        repo: ConversationRepository,
        current_question: str,
        summary: str | None = None,
        chat_type: Literal["general", "paper"] = "paper",
    ) -> BuiltContext:
        """Retrieve recent messages, include summary, enforce token budget, and prevent duplication.

        Args:
            conversation_id: UUID of the conversation.
            repo: ConversationRepository instance.
            current_question: Current user question text.
            summary: Stored conversation summary (or None).
            chat_type: 'general' or 'paper'.

        Returns:
            BuiltContext with chronological history messages and token estimation.
        """
        clean_q = current_question.strip() if current_question else ""

        # 1. Fetch recent completed messages from repository
        recent_db_messages = await repo.get_recent_messages(
            conversation_id,
            limit=self.recent_message_limit,
        )

        raw_history: list[dict[str, str]] = [
            {"role": m.role, "content": m.content}
            for m in recent_db_messages
            if m.content and m.content.strip()
        ]

        # 2. Avoid duplicating the current user message if it was already recorded
        if raw_history and clean_q:
            last_msg = raw_history[-1]
            if last_msg.get("role") == "user" and last_msg.get("content", "").strip() == clean_q:
                raw_history.pop()

        # 3. Enforce token budget
        # Budget must accommodate summary, question, and recent messages
        summary_tokens = estimate_tokens(summary or "")
        question_tokens = estimate_tokens(clean_q)
        fixed_tokens = summary_tokens + question_tokens

        available_history_tokens = max(1, self.max_context_tokens - fixed_tokens)

        # Truncate oldest messages if history exceeds token budget (working backwards)
        selected_history: list[dict[str, str]] = []
        accumulated_history_tokens = 0

        for msg in reversed(raw_history):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if accumulated_history_tokens + msg_tokens > available_history_tokens and selected_history:
                # Exceeds budget, drop older messages
                break
            selected_history.append(msg)
            accumulated_history_tokens += msg_tokens

        # Restore chronological ascending order
        final_history = list(reversed(selected_history))

        total_tokens = fixed_tokens + accumulated_history_tokens

        return BuiltContext(
            conversation_id=conversation_id,
            summary=summary if summary and summary.strip() else None,
            history_messages=final_history,
            current_question=clean_q,
            estimated_tokens=total_tokens,
            chat_type=chat_type,
        )
