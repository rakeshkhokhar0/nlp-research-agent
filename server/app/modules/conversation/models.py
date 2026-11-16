"""SQLAlchemy 2.x models for conversations, messages, and message citations."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, CreatedTimestampMixin, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.modules.ingestion.models import PaperChunk


class Conversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Maintain independent research chat sessions."""

    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'archived')",
            name="status",
        ),
        Index("ix_conversations_created_at", "created_at"),
        Index("ix_conversations_last_message_at", "last_message_at"),
    )

    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        server_default=text("'active'"),
        nullable=False,
    )
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )

    # Relationships
    messages: Mapped[list[Message]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.sequence_number",
    )

    def __init__(self, **kwargs: Any) -> None:
        if "metadata" in kwargs and "metadata_" not in kwargs:
            kwargs["metadata_"] = kwargs.pop("metadata")
        super().__init__(**kwargs)


class Message(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Store messages belonging to conversations."""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence_number",
            name="uq_messages_conversation_sequence",
        ),
        CheckConstraint(
            "sequence_number > 0",
            name="positive_sequence_number",
        ),
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name="role",
        ),
        CheckConstraint(
            "token_count IS NULL OR token_count >= 0",
            name="nonnegative_tokens",
        ),
        CheckConstraint(
            "status IN ('pending', 'streaming', 'completed', 'failed')",
            name="status",
        ),
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        String(30),
        default="completed",
        server_default=text("'completed'"),
        nullable=False,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )

    # Relationships
    conversation: Mapped[Conversation] = relationship(
        "Conversation",
        back_populates="messages",
    )
    citations: Mapped[list[MessageCitation]] = relationship(
        "MessageCitation",
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MessageCitation.citation_order",
    )

    def __init__(self, **kwargs: Any) -> None:
        if "metadata" in kwargs and "metadata_" not in kwargs:
            kwargs["metadata_"] = kwargs.pop("metadata")
        super().__init__(**kwargs)


class MessageCitation(Base, UUIDPrimaryKeyMixin, CreatedTimestampMixin):
    """Link generated assistant responses to supporting paper chunks."""

    __tablename__ = "message_citations"
    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "chunk_id",
            name="uq_message_citations_message_chunk",
        ),
        CheckConstraint(
            "citation_order > 0",
            name="positive_citation_order",
        ),
        CheckConstraint(
            "page_number IS NULL OR page_number > 0",
            name="positive_page_number",
        ),
    )

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("paper_chunks.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    citation_order: Mapped[int] = mapped_column(Integer, nullable=False)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cited_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    message: Mapped[Message] = relationship("Message", back_populates="citations")
    chunk: Mapped[PaperChunk] = relationship("PaperChunk", back_populates="citations")

    def __init__(self, **kwargs: Any) -> None:
        if "quote_text" in kwargs and "cited_text" not in kwargs:
            kwargs["cited_text"] = kwargs.pop("quote_text")
        super().__init__(**kwargs)

    @property
    def quote_text(self) -> str | None:
        return self.cited_text

    @quote_text.setter
    def quote_text(self, value: str | None) -> None:
        self.cited_text = value
