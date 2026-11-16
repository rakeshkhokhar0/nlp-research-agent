"""SQLAlchemy database repository for conversations, messages, and citations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import app.database.model_registry  # noqa: F401 - ensures cross-module relationship mappers are registered
from app.modules.conversation.models import Conversation, Message, MessageCitation
from app.modules.ingestion.models import PaperChunk


class ConversationRepositoryError(Exception):
    """Base exception for conversation database repository operations."""


class ConversationRepository:
    """Repository managing persistence for conversations, messages, and citations.

    Maintains transactional purity by delegating commit and rollback boundaries
    to the service layer while executing explicit flushes when database-generated
    IDs and constraints require immediate synchronization.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with an active asynchronous database session.

        Args:
            session: Active SQLAlchemy AsyncSession.

        Raises:
            TypeError: If session is not an instance of AsyncSession.
        """
        if not isinstance(session, AsyncSession):
            raise TypeError(
                f"session must be an AsyncSession, got {type(session).__name__}"
            )
        self.session: AsyncSession = session

    async def create_conversation(
        self,
        title: str | None = None,
        metadata_: dict[str, Any] | None = None,
    ) -> Conversation:
        """Create and flush a new Conversation record.

        Args:
            title: Optional title for the research conversation.
            metadata_: Optional JSONB metadata dictionary.

        Returns:
            The newly created and flushed Conversation instance.
        """
        conversation = Conversation(
            title=title.strip() if title and title.strip() else None,
            status="active",
            metadata_=metadata_ if metadata_ is not None else {},
        )
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def get_conversation(
        self,
        conversation_id: uuid.UUID,
        for_update: bool = False,
    ) -> Conversation | None:
        """Retrieve a Conversation by primary key UUID.

        Args:
            conversation_id: UUID of the target conversation.
            for_update: If True, locks the row with SELECT FOR UPDATE to prevent
                concurrent sequence number races.

        Returns:
            The Conversation instance if found, or None.
        """
        if not isinstance(conversation_id, uuid.UUID):
            raise TypeError(
                f"conversation_id must be a uuid.UUID, got {type(conversation_id).__name__}"
            )

        stmt = select(Conversation).where(Conversation.id == conversation_id)
        if for_update:
            stmt = stmt.with_for_update()

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_conversation_messages(
        self,
        conversation_id: uuid.UUID,
        limit: int | None = None,
        include_citations: bool = True,
    ) -> list[Message]:
        """Retrieve messages for a conversation ordered chronologically by sequence_number.

        Args:
            conversation_id: UUID of the conversation.
            limit: Optional maximum number of messages to return.
            include_citations: If True, eagerly loads citations and cited chunks.

        Returns:
            List of Message instances ordered by sequence_number ascending.
        """
        if not isinstance(conversation_id, uuid.UUID):
            raise TypeError(
                f"conversation_id must be a uuid.UUID, got {type(conversation_id).__name__}"
            )

        stmt = select(Message).where(Message.conversation_id == conversation_id)
        if include_citations:
            stmt = stmt.options(
                selectinload(Message.citations).selectinload(MessageCitation.chunk).selectinload(PaperChunk.paper)
            )

        stmt = stmt.order_by(Message.sequence_number.asc())
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent_messages(
        self,
        conversation_id: uuid.UUID,
        limit: int = 6,
        status: str = "completed",
    ) -> list[Message]:
        """Fetch the most recent completed messages for conversation history context.

        Returns messages ordered chronologically (ascending) for LLM prompt assembly.
        """
        if not isinstance(conversation_id, uuid.UUID):
            raise TypeError(
                f"conversation_id must be a uuid.UUID, got {type(conversation_id).__name__}"
            )

        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.status == status,
            )
            .order_by(Message.sequence_number.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        recent_desc = list(result.scalars().all())
        return list(reversed(recent_desc))

    async def get_next_sequence_number(self, conversation_id: uuid.UUID) -> int:
        """Compute the next monotonically increasing sequence number for a conversation.

        Should be called within a transaction where the conversation row is locked
        via SELECT FOR UPDATE to avoid race conditions.

        Returns:
            Next positive integer sequence number (starts at 1).
        """
        if not isinstance(conversation_id, uuid.UUID):
            raise TypeError(
                f"conversation_id must be a uuid.UUID, got {type(conversation_id).__name__}"
            )

        stmt = select(func.coalesce(func.max(Message.sequence_number), 0) + 1).where(
            Message.conversation_id == conversation_id
        )
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def create_message(
        self,
        conversation_id: uuid.UUID,
        sequence_number: int,
        role: str,
        content: str,
        model_name: str | None = None,
        token_count: int | None = None,
        status: str = "completed",
        metadata_: dict[str, Any] | None = None,
    ) -> Message:
        """Create and flush a new Message record within the conversation.

        Args:
            conversation_id: UUID of parent conversation.
            sequence_number: Positive sequence number.
            role: One of 'user', 'assistant', 'system'.
            content: Message text content.
            model_name: Optional LLM model identifier.
            token_count: Optional token count.
            status: Message status ('completed', 'pending', 'failed').
            metadata_: Optional metadata dictionary.

        Returns:
            The created and flushed Message instance.
        """
        if sequence_number <= 0:
            raise ValueError(f"sequence_number must be positive (> 0), got {sequence_number}")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"role must be 'user', 'assistant', or 'system', got {role!r}")
        if not content or not content.strip():
            raise ValueError("content must not be empty or whitespace.")

        message = Message(
            conversation_id=conversation_id,
            sequence_number=sequence_number,
            role=role,
            content=content.strip(),
            model_name=model_name,
            token_count=token_count,
            status=status,
            metadata_=metadata_ if metadata_ is not None else {},
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def update_message(
        self,
        message_id: uuid.UUID,
        *,
        content: str | None = None,
        status: str | None = None,
        token_count: int | None = None,
        metadata_: dict[str, Any] | None = None,
    ) -> Message:
        """Update fields of an existing Message."""
        stmt = select(Message).where(Message.id == message_id)
        result = await self.session.execute(stmt)
        message = result.scalar_one_or_none()
        if message is None:
            raise ConversationRepositoryError(f"Message '{message_id}' not found.")

        if content is not None:
            message.content = content.strip()
        if status is not None:
            message.status = status
        if token_count is not None:
            message.token_count = token_count
        if metadata_ is not None:
            message.metadata_ = metadata_

        await self.session.flush()
        return message

    async def create_citations(
        self,
        message_id: uuid.UUID,
        citations_data: list[dict[str, Any]],
    ) -> list[MessageCitation]:
        """Create and flush MessageCitation records for an assistant message.

        Args:
            message_id: UUID of the assistant message.
            citations_data: List of citation dictionaries containing:
                - chunk_id: uuid.UUID
                - citation_order: int (> 0)
                - relevance_score: float | None
                - cited_text: str | None
                - page_number: int | None

        Returns:
            List of created MessageCitation instances.
        """
        citations: list[MessageCitation] = []
        for item in citations_data:
            chunk_id = item["chunk_id"]
            if not isinstance(chunk_id, uuid.UUID):
                chunk_id = uuid.UUID(str(chunk_id))

            order = item["citation_order"]
            if order <= 0:
                raise ValueError(f"citation_order must be positive (> 0), got {order}")

            page_num = item.get("page_number")
            if page_num is not None and page_num <= 0:
                page_num = None  # Check constraint: positive_page_number

            citation = MessageCitation(
                message_id=message_id,
                chunk_id=chunk_id,
                citation_order=order,
                relevance_score=item.get("relevance_score"),
                cited_text=item.get("cited_text"),
                page_number=page_num,
            )
            self.session.add(citation)
            citations.append(citation)

        if citations:
            await self.session.flush()
        return citations

    async def get_chunk_page_information(
        self,
        chunk_ids: Sequence[uuid.UUID],
    ) -> dict[uuid.UUID, int | None]:
        """Retrieve page_start for a sequence of PaperChunk UUIDs from PostgreSQL.

        Args:
            chunk_ids: Sequence of chunk UUIDs to look up.

        Returns:
            Dictionary mapping chunk_id to its page_start (or None if not recorded).
        """
        if not chunk_ids:
            return {}

        uids = [cid if isinstance(cid, uuid.UUID) else uuid.UUID(str(cid)) for cid in chunk_ids]
        stmt = select(PaperChunk.id, PaperChunk.page_start).where(PaperChunk.id.in_(uids))
        result = await self.session.execute(stmt)
        return {row[0]: row[1] for row in result.all()}

    async def update_conversation_last_message_at(
        self,
        conversation_id: uuid.UUID,
        last_message_at: datetime | None = None,
    ) -> None:
        """Update last_message_at timestamp for a conversation."""
        ts = last_message_at or datetime.now(timezone.utc)
        stmt = (
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(last_message_at=ts)
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def update_context_summary(
        self,
        conversation_id: uuid.UUID,
        context_summary: str,
        last_summarized_sequence: int | None = None,
    ) -> None:
        """Update context_summary and optional last_summarized_sequence metadata for a conversation."""
        if not isinstance(conversation_id, uuid.UUID):
            raise TypeError(f"conversation_id must be a uuid.UUID, got {type(conversation_id).__name__}")

        conv = await self.get_conversation(conversation_id)
        if conv is not None:
            conv.context_summary = context_summary
            if last_summarized_sequence is not None:
                meta = dict(conv.metadata_ or {})
                meta["last_summarized_sequence"] = last_summarized_sequence
                conv.metadata_ = meta
            await self.session.flush()

    async def get_messages_for_summarization(
        self,
        conversation_id: uuid.UUID,
        min_sequence: int = 1,
        max_sequence: int | None = None,
        status: str = "completed",
    ) -> list[Message]:
        """Fetch completed messages eligible for summarization ordered chronologically."""
        if not isinstance(conversation_id, uuid.UUID):
            raise TypeError(f"conversation_id must be a uuid.UUID, got {type(conversation_id).__name__}")

        stmt = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.status == status,
                Message.sequence_number >= min_sequence,
            )
        )
        if max_sequence is not None:
            stmt = stmt.where(Message.sequence_number <= max_sequence)
        stmt = stmt.order_by(Message.sequence_number.asc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
