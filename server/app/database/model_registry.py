"""Model registry ensuring all SQLAlchemy models are registered on Base.metadata."""

from app.database.base import Base

# Papers
from app.modules.papers.models import (
    Paper,
    PaperFile,
    PaperSource,
)

# Ingestion
from app.modules.ingestion.models import (
    IngestionJob,
    PaperChunk,
)

# Retrieval
from app.modules.retrieval.models import (
    ChunkEmbedding,
    EmbeddingModel,
)

# Conversation
from app.modules.conversation.models import (
    Conversation,
    Message,
    MessageCitation,
)

__all__ = [
    "Base",
    "Paper",
    "PaperSource",
    "PaperFile",
    "IngestionJob",
    "PaperChunk",
    "EmbeddingModel",
    "ChunkEmbedding",
    "Conversation",
    "Message",
    "MessageCitation",
]


__all__ = [
    "Base",
    "Paper",
    "PaperSource",
    "PaperFile",
    "IngestionJob",
    "PaperChunk",
    "EmbeddingModel",
    "ChunkEmbedding",
    "Conversation",
    "Message",
    "MessageCitation",
]