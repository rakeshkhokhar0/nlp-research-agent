"""Unit tests for FastAPI lifespan and retrieval index lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.database.model_registry  # Ensure models are mapped
from app.main import app, lifespan
from app.modules.conversation.llm_client import LLMConfigurationError
from app.modules.conversation.service import ConversationService
from app.modules.ingestion.models import PaperChunk
from app.modules.papers.models import Paper
from app.modules.retrieval import RetrievalService, RetrievedChunk


def create_mock_chunks() -> list[PaperChunk]:
    """Helper to generate sample chunks with parent paper."""
    paper = Paper(
        id=uuid.uuid4(),
        title="Attention Is All You Need",
        publication_year=2017,
    )
    chunk1 = PaperChunk(
        id=uuid.uuid4(),
        paper_id=paper.id,
        ingestion_job_id=uuid.uuid4(),
        chunk_index=0,
        content="Transformer architecture with multi-head self-attention.",
        is_current=True,
    )
    chunk1.paper = paper

    chunk2 = PaperChunk(
        id=uuid.uuid4(),
        paper_id=paper.id,
        ingestion_job_id=uuid.uuid4(),
        chunk_index=1,
        content="Feed-forward networks and positional encodings.",
        is_current=True,
    )
    chunk2.paper = paper
    return [chunk1, chunk2]


class TestFastAPILifespanRetrievalIntegration:
    """Tests for application lifespan startup refresh and shutdown."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_and_registers_retrieval_service(self) -> None:
        """Lifespan startup creates RetrievalService, refreshes index via fresh session, and registers it in app.state."""
        mock_chunks = create_mock_chunks()

        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_chunks
        mock_session.execute.return_value = mock_result

        # Context manager for AsyncSessionLocal()
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__.return_value = mock_session
        mock_session_ctx.__aexit__.return_value = None

        # Mock AsyncEngine
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch("app.main.AsyncSessionLocal", return_value=mock_session_ctx),
            patch("app.main.engine", mock_engine),
        ):
            async with lifespan(app):
                # 1. RetrievalService registered in app.state
                assert hasattr(app.state, "retrieval_service")
                service = app.state.retrieval_service
                assert isinstance(service, RetrievalService)
                assert service.is_initialized is True
                assert len(service.id_mapper) == 2

                # 2. Startup session was closed
                mock_session_ctx.__aexit__.assert_awaited_once()

                # 3. Service search is usable in-memory after startup session closed
                results = service.search("transformer self-attention", top_k=1)
                assert len(results) == 1
                assert isinstance(results[0], RetrievedChunk)
                assert results[0].paper_title == "Attention Is All You Need"

                # Engine dispose not called during runtime
                mock_engine.dispose.assert_not_awaited()

            # 4. Engine dispose called on shutdown
            mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_lifespan_startup_with_empty_database_succeeds(self) -> None:
        """When database is reachable but contains zero chunks, startup succeeds with an empty initialized index."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__.return_value = mock_session
        mock_session_ctx.__aexit__.return_value = None

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch("app.main.AsyncSessionLocal", return_value=mock_session_ctx),
            patch("app.main.engine", mock_engine),
        ):
            async with lifespan(app):
                assert hasattr(app.state, "retrieval_service")
                service = app.state.retrieval_service
                assert service.is_initialized is True
                assert len(service.id_mapper) == 0
                assert service.search("anything") == []

    @pytest.mark.asyncio
    async def test_lifespan_startup_database_failure_propagates_clearly(self) -> None:
        """If database connection or query fails during startup, lifespan raises the exception immediately."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute.side_effect = ConnectionError("Could not connect to PostgreSQL server")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__.return_value = mock_session
        mock_session_ctx.__aexit__.return_value = None

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch("app.main.AsyncSessionLocal", return_value=mock_session_ctx),
            patch("app.main.engine", mock_engine),
        ):
            with pytest.raises(ConnectionError, match="Could not connect to PostgreSQL server"):
                async with lifespan(app):
                    pass  # Should not reach here

    @pytest.mark.asyncio
    async def test_lifespan_starts_when_openai_key_missing_and_conversation_service_remains_unavailable(
        self,
    ) -> None:
        """When OpenAI API key is missing, lifespan starts successfully and conversation service is None."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__.return_value = mock_session
        mock_session_ctx.__aexit__.return_value = None

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch("app.main.AsyncSessionLocal", return_value=mock_session_ctx),
            patch("app.main.engine", mock_engine),
            patch(
                "app.main.OpenAIResponseClient",
                side_effect=LLMConfigurationError("Missing API key"),
            ),
        ):
            async with lifespan(app):
                assert app.state.openai_client is None
                assert app.state.conversation_service is None
                assert hasattr(app.state, "retrieval_service")
                assert app.state.retrieval_service.is_initialized is True

    @pytest.mark.asyncio
    async def test_lifespan_initializes_conversation_service_when_configured(
        self,
    ) -> None:
        """When OpenAI client is configured and sessionmaker is present, conversation service is initialized."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__.return_value = mock_session
        mock_session_ctx.__aexit__.return_value = None

        mock_factory = MagicMock(spec=async_sessionmaker)
        mock_factory.return_value = mock_session_ctx

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        mock_openai_client = MagicMock()
        mock_openai_client.close = AsyncMock()

        with (
            patch("app.main.AsyncSessionLocal", mock_factory),
            patch("app.main.engine", mock_engine),
            patch("app.main.OpenAIResponseClient", return_value=mock_openai_client),
        ):
            async with lifespan(app):
                assert app.state.openai_client is mock_openai_client
                assert isinstance(app.state.conversation_service, ConversationService)

            mock_openai_client.close.assert_awaited_once()
