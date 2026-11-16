"""Pydantic schemas for research conversation and chat endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreateRequest(BaseModel):
    """Request payload for creating a new research conversation."""

    title: str | None = Field(None, max_length=500, description="Optional title for the conversation")
    chat_type: Literal["general", "paper"] | None = Field(None, description="Optional chat type ('general' or 'paper')")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata dictionary")


class ConversationResponse(BaseModel):
    """Response payload representing a conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique conversation UUID")
    title: str | None = Field(None, description="Conversation title")
    status: str = Field(..., description="Status ('active' or 'archived')")
    chat_type: Literal["general", "paper"] = Field("paper", description="Established conversation chat type")
    context_summary: str | None = Field(None, description="Conversation context summary if generated")
    created_at: datetime = Field(..., description="Timestamp when conversation was created")
    updated_at: datetime | None = Field(None, description="Timestamp when conversation was updated")
    last_message_at: datetime | None = Field(None, description="Timestamp of the most recent message")


class ChatRequest(BaseModel):
    """Request payload for submitting a user question to a research conversation."""

    question: str = Field(..., min_length=1, max_length=5000, description="User research question")
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of evidence chunks to retrieve")
    paper_ids: list[uuid.UUID] | None = Field(None, description="Optional filter to restrict search to specific papers")
    mode: Literal["normal", "advanced"] = Field("normal", description="Research answer generation mode ('normal' or 'advanced')")
    chat_type: Literal["general", "paper"] | None = Field(
        None,
        description="Chat type: 'general' for general conversation without retrieval, or 'paper' for grounded research paper chat",
    )


class CitationResponse(BaseModel):
    """Representation of an evidence chunk citation supporting an assistant answer."""

    model_config = ConfigDict(from_attributes=True)

    chunk_id: uuid.UUID = Field(..., description="UUID of the supporting PaperChunk")
    paper_id: uuid.UUID = Field(..., description="UUID of the parent paper")
    paper_title: str = Field(..., description="Title of the cited research paper")
    citation_order: int = Field(..., description="Order of citation within the response (1, 2, ...)")
    page_number: int | None = Field(None, description="Start page number from the paper if available")
    relevance_score: float | None = Field(None, description="Lexical relevance score of the chunk")
    quote_text: str | None = Field(None, description="Exemplary quoted evidence text snippet")


class ChatResponse(BaseModel):
    """Response payload returned from question answering."""

    model_config = ConfigDict(from_attributes=True)

    conversation_id: uuid.UUID = Field(..., description="UUID of the conversation")
    user_message_id: uuid.UUID = Field(..., description="UUID of the persisted user message")
    assistant_message_id: uuid.UUID = Field(..., description="UUID of the generated assistant message")
    answer: str = Field(..., description="Grounded answer to the question")
    citations: list[CitationResponse] = Field(default_factory=list, description="Citations supporting the answer")
    model_name: str | None = Field(None, description="Model identifier used for answer generation, or None if no model was called")
    mode: Literal["normal", "advanced"] = Field("normal", description="Selected research mode used for generation")
    chat_type: Literal["general", "paper"] = Field("paper", description="Selected chat type used for response")
    created_at: datetime = Field(..., description="Timestamp when assistant message was created")


class ConversationMessageResponse(BaseModel):
    """Representation of a conversation message and its citations for history rendering."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="UUID of the message")
    conversation_id: uuid.UUID = Field(..., description="UUID of the conversation")
    sequence_number: int = Field(..., description="Sequential position of the message in conversation")
    role: str = Field(..., description="Role of the message author ('user', 'assistant', 'system')")
    content: str = Field(..., description="Message text content")
    status: str = Field(..., description="Message status ('completed', 'failed', 'pending')")
    model_name: str | None = Field(None, description="Model identifier if generated by assistant")
    token_count: int | None = Field(None, description="Token usage count if recorded")
    mode: str | None = Field(None, description="Research mode if recorded on message metadata")
    citations: list[CitationResponse] = Field(default_factory=list, description="Associated citations")
    created_at: datetime = Field(..., description="Creation timestamp")
