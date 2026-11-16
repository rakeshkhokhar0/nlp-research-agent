"""Unit tests for RetrievalService."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import app.database.model_registry  # Ensure models are mapped
from app.modules.ingestion.models import PaperChunk
from app.modules.papers.models import Paper
from app.modules.retrieval.id_mapper import ChunkIDMapper
from app.modules.retrieval.lexical import LexicalRetrievalEngine
from app.modules.retrieval.repository import RetrievalRepository
from app.modules.retrieval.service import (
    RetrievalService,
    RetrievedChunk,
    refresh_retrieval_index,
)


@pytest.fixture
def mock_session() -> AsyncMock:
    """Provide an AsyncMock for AsyncSession."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_repository(mock_session: AsyncMock) -> RetrievalRepository:
    """Provide a real RetrievalRepository wrapping an AsyncMock session."""
    return RetrievalRepository(mock_session)


def create_sample_paper_and_chunks(count: int = 3) -> tuple[Paper, list[PaperChunk]]:
    """Helper to generate a paper and associated chunks for testing."""
    paper = Paper(
        id=uuid.uuid4(),
        title="Attention Is All You Need",
        publication_year=2017,
    )
    chunks = []
    for i in range(count):
        chunk = PaperChunk(
            id=uuid.uuid4(),
            paper_id=paper.id,
            ingestion_job_id=uuid.uuid4(),
            chunk_index=i,
            content=f"Chunk {i} describes self-attention mechanisms in transformer networks.",
            is_current=True,
        )
        chunk.paper = paper
        chunks.append(chunk)
    return paper, chunks


class TestRetrievalServiceInitialization:
    """Tests for service instantiation and dependency injection."""

    def test_init_without_repository_creates_default_engine(self) -> None:
        """Service initializes without a session-bound repository and creates default components."""
        service = RetrievalService()
        assert isinstance(service.engine, LexicalRetrievalEngine)
        assert isinstance(service.id_mapper, ChunkIDMapper)
        assert len(service.id_mapper) == 0
        assert service.is_initialized is False

    def test_init_with_injected_engine(self) -> None:
        """Service accepts an injected LexicalRetrievalEngine instance."""
        custom_engine = LexicalRetrievalEngine()
        service = RetrievalService(engine=custom_engine)
        assert service.engine is custom_engine

    @pytest.mark.parametrize("invalid_engine", [123, "not_engine", MagicMock()])
    def test_init_with_invalid_engine_raises_type_error(self, invalid_engine: object) -> None:
        """Passing non-LexicalRetrievalEngine raises TypeError."""
        with pytest.raises(TypeError, match="engine must be an instance of LexicalRetrievalEngine"):
            RetrievalService(engine=invalid_engine)  # type: ignore[arg-type]


class TestRetrievalServiceIndexRefresh:
    """Tests for asynchronous index refresh workflow."""

    @pytest.mark.asyncio
    async def test_refresh_index_awaits_repository_and_populates_engine(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """refresh_index awaits get_current_chunks(), assigns integer IDs, and builds the index."""
        _, chunks = create_sample_paper_and_chunks(count=3)
        mock_repository.get_current_chunks = AsyncMock(return_value=chunks)

        service = RetrievalService()
        count = await service.refresh_index(mock_repository)

        mock_repository.get_current_chunks.assert_awaited_once_with()
        assert count == 3
        assert service.is_initialized is True
        assert len(service.id_mapper) == 3
        assert len(service.engine) == 3

        # Verify integer IDs map back to original chunk UUIDs
        for i, chunk in enumerate(chunks, start=1):
            assert service.id_mapper.get_integer_id(chunk.id) == i
            assert service.id_mapper.get_uuid(i) == chunk.id

    @pytest.mark.asyncio
    async def test_refresh_index_with_invalid_repository_raises_type_error(self) -> None:
        """Passing non-RetrievalRepository to refresh_index raises TypeError."""
        service = RetrievalService()
        with pytest.raises(TypeError, match="repository must be an instance of RetrievalRepository"):
            await service.refresh_index("invalid_repo")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_refresh_index_empty_database_creates_initialized_empty_index(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """When database has no active chunks, refresh_index initializes an empty index."""
        mock_repository.get_current_chunks = AsyncMock(return_value=[])

        service = RetrievalService()
        count = await service.refresh_index(mock_repository)

        assert count == 0
        assert service.is_initialized is True
        assert len(service.id_mapper) == 0
        assert len(service.engine) == 0

    @pytest.mark.asyncio
    async def test_repeated_refresh_works_with_different_repositories(
        self, mock_session: AsyncMock
    ) -> None:
        """Subsequent refresh_index calls using different repository/session instances rebuild cleanly."""
        _, initial_chunks = create_sample_paper_and_chunks(count=2)
        repo1 = RetrievalRepository(mock_session)
        repo1.get_current_chunks = AsyncMock(return_value=initial_chunks)

        service = RetrievalService()
        await service.refresh_index(repo1)
        assert len(service.id_mapper) == 2

        # Second refresh using a distinct repository instance and different session
        mock_session2 = AsyncMock(spec=AsyncSession)
        repo2 = RetrievalRepository(mock_session2)

        paper2 = Paper(id=uuid.uuid4(), title="Convolutional Networks")
        new_chunk = PaperChunk(
            id=uuid.uuid4(),
            paper_id=paper2.id,
            ingestion_job_id=uuid.uuid4(),
            chunk_index=0,
            content="Convolutional feature extraction.",
            is_current=True,
        )
        new_chunk.paper = paper2
        repo2.get_current_chunks = AsyncMock(return_value=[new_chunk])

        count = await service.refresh_index(repo2)
        assert count == 1
        assert len(service.id_mapper) == 1
        assert service.id_mapper.get_uuid(1) == new_chunk.id
        # Old chunk UUIDs must no longer be present
        assert service.id_mapper.get_integer_id(initial_chunks[0].id) is None
        assert service.id_mapper.get_integer_id(initial_chunks[1].id) is None

    @pytest.mark.asyncio
    async def test_repository_failure_preserves_existing_state(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """If repository.get_current_chunks() raises an exception, service state is unchanged."""
        _, chunks = create_sample_paper_and_chunks(count=2)
        mock_repository.get_current_chunks = AsyncMock(return_value=chunks)

        service = RetrievalService()
        await service.refresh_index(mock_repository)
        assert service.is_initialized is True
        assert len(service.id_mapper) == 2

        # Simulate repository failure on next refresh
        mock_repository.get_current_chunks = AsyncMock(side_effect=RuntimeError("Database failure"))

        with pytest.raises(RuntimeError, match="Database failure"):
            await service.refresh_index(mock_repository)

        # Previous state remains intact
        assert service.is_initialized is True
        assert len(service.id_mapper) == 2
        assert service.id_mapper.get_integer_id(chunks[0].id) == 1

    @pytest.mark.asyncio
    async def test_engine_build_failure_preserves_existing_state(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """If engine.build_index() raises an error, mappings are not updated."""
        mock_engine = MagicMock(spec=LexicalRetrievalEngine)
        mock_engine.build_index.side_effect = ValueError("Preprocessing failure")

        _, chunks = create_sample_paper_and_chunks(count=1)
        mock_repository.get_current_chunks = AsyncMock(return_value=chunks)

        service = RetrievalService(engine=mock_engine)

        with pytest.raises(ValueError, match="Preprocessing failure"):
            await service.refresh_index(mock_repository)

        # Service must not be marked initialized or publish mappings
        assert service.is_initialized is False
        assert len(service.id_mapper) == 0

    @pytest.mark.asyncio
    async def test_refresh_helper_function(self, mock_session: AsyncMock) -> None:
        """refresh_retrieval_index helper constructs a repository and refreshes the service."""
        _, chunks = create_sample_paper_and_chunks(count=2)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = chunks
        mock_session.execute.return_value = mock_result

        service = RetrievalService()
        count = await refresh_retrieval_index(service, mock_session)
        assert count == 2
        assert service.is_initialized is True

    @pytest.mark.asyncio
    async def test_concurrent_refreshes_are_serialized(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """Concurrent refresh_index calls are serialized by the internal lock without corruption."""
        _, chunks = create_sample_paper_and_chunks(count=3)
        mock_repository.get_current_chunks = AsyncMock(return_value=chunks)

        service = RetrievalService()

        # Run two simultaneous refreshes
        results = await asyncio.gather(
            service.refresh_index(mock_repository),
            service.refresh_index(mock_repository),
        )
        assert results == [3, 3]
        assert service.is_initialized is True
        assert len(service.id_mapper) == 3


class TestRetrievalServiceSearch:
    """Tests for synchronous in-memory search and result translation."""

    def test_search_before_refresh_raises_runtime_error(self) -> None:
        """Calling search before refresh_index raises RuntimeError."""
        service = RetrievalService()
        assert service.is_initialized is False

        with pytest.raises(RuntimeError, match="RetrievalService index is not initialized"):
            service.search("query")

    @pytest.mark.asyncio
    async def test_search_delegates_to_engine_and_maps_uuids_and_metadata(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """Search delegates to the engine, translates integer IDs to UUIDs, and constructs RetrievedChunk."""
        paper, chunks = create_sample_paper_and_chunks(count=3)
        mock_repository.get_current_chunks = AsyncMock(return_value=chunks)

        service = RetrievalService()
        await service.refresh_index(mock_repository)

        results = service.search("transformer attention", top_k=2)

        assert len(results) > 0
        assert len(results) <= 2
        for item in results:
            assert isinstance(item, RetrievedChunk)
            assert isinstance(item.chunk_id, uuid.UUID)
            assert item.paper_id == paper.id
            assert item.paper_title == "Attention Is All You Need"
            assert item.publication_year == 2017
            assert item.score > 0.0
            assert "self-attention" in item.content

    @pytest.mark.asyncio
    async def test_search_usable_after_session_closed(
        self, mock_session: AsyncMock
    ) -> None:
        """Search continues to function purely in-memory after the refresh session has been closed."""
        paper, chunks = create_sample_paper_and_chunks(count=2)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = chunks
        mock_session.execute.return_value = mock_result

        service = RetrievalService()
        # Simulate opening and closing a short-lived session
        repo = RetrievalRepository(mock_session)
        await service.refresh_index(repo)
        await mock_session.close()

        # Search runs without the session and without errors
        results = service.search("transformer attention", top_k=5)
        assert len(results) > 0
        assert results[0].paper_title == "Attention Is All You Need"

    @pytest.mark.asyncio
    async def test_search_preserves_ranking_order_and_scores(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """RetrievedChunk items preserve the exact descending score order returned by the engine."""
        paper = Paper(id=uuid.uuid4(), title="Transformer Models", publication_year=2020)
        c1 = PaperChunk(
            id=uuid.uuid4(), paper_id=paper.id, ingestion_job_id=uuid.uuid4(),
            chunk_index=0, content="Transformer architecture with self-attention.", is_current=True,
        )
        c1.paper = paper
        c2 = PaperChunk(
            id=uuid.uuid4(), paper_id=paper.id, ingestion_job_id=uuid.uuid4(),
            chunk_index=1, content="Convolutional neural networks for vision.", is_current=True,
        )
        c2.paper = paper
        c3 = PaperChunk(
            id=uuid.uuid4(), paper_id=paper.id, ingestion_job_id=uuid.uuid4(),
            chunk_index=2, content="Transformer multi-head self-attention mechanisms and layers.", is_current=True,
        )
        c3.paper = paper

        mock_repository.get_current_chunks = AsyncMock(return_value=[c1, c2, c3])
        service = RetrievalService()
        await service.refresh_index(mock_repository)

        results = service.search("transformer self-attention", top_k=5)
        scores = [item.score for item in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_forwards_top_k(self, mock_repository: RetrievalRepository) -> None:
        """top_k is forwarded to engine.search and limits results accordingly."""
        _, chunks = create_sample_paper_and_chunks(count=5)
        mock_repository.get_current_chunks = AsyncMock(return_value=chunks)

        service = RetrievalService()
        await service.refresh_index(mock_repository)

        results_top1 = service.search("transformer", top_k=1)
        assert len(results_top1) == 1

        results_top3 = service.search("transformer", top_k=3)
        assert len(results_top3) == 3

    @pytest.mark.asyncio
    async def test_search_does_not_execute_database_queries(
        self, mock_repository: RetrievalRepository, mock_session: AsyncMock
    ) -> None:
        """Search is entirely in-memory and does not invoke repository or database session methods."""
        _, chunks = create_sample_paper_and_chunks(count=2)
        mock_repository.get_current_chunks = AsyncMock(return_value=chunks)

        service = RetrievalService()
        await service.refresh_index(mock_repository)

        # Reset mock call counts after refresh
        mock_repository.get_current_chunks.reset_mock()
        mock_session.execute.reset_mock()

        service.search("transformer")

        mock_repository.get_current_chunks.assert_not_called()
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_empty_index_or_no_match_returns_empty_list(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """Searching an empty index or a query matching nothing returns []."""
        mock_repository.get_current_chunks = AsyncMock(return_value=[])

        service = RetrievalService()
        await service.refresh_index(mock_repository)

        assert service.search("transformer") == []

    @pytest.mark.asyncio
    async def test_search_handles_nullable_paper_fields(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """Paper with null publication_year is correctly handled in RetrievedChunk."""
        paper = Paper(id=uuid.uuid4(), title="Draft Research", publication_year=None)
        chunk = PaperChunk(
            id=uuid.uuid4(), paper_id=paper.id, ingestion_job_id=uuid.uuid4(),
            chunk_index=0, content="Draft paper transformer text.", is_current=True,
        )
        chunk.paper = paper

        mock_repository.get_current_chunks = AsyncMock(return_value=[chunk])
        service = RetrievalService()
        await service.refresh_index(mock_repository)

        results = service.search("transformer")
        assert len(results) == 1
        assert results[0].paper_title == "Draft Research"
        assert results[0].publication_year is None

    @pytest.mark.asyncio
    async def test_search_inconsistent_mapping_raises_runtime_error(
        self, mock_repository: RetrievalRepository
    ) -> None:
        """If engine returns an integer ID not present in mapper/metadata, a RuntimeError is raised."""
        mock_engine = MagicMock(spec=LexicalRetrievalEngine)
        # Engine returns document ID 999 which does not exist in mapper
        mock_engine.search.return_value = [(999, 0.85)]
        mock_engine.build_index.return_value = None

        _, chunks = create_sample_paper_and_chunks(count=1)
        mock_repository.get_current_chunks = AsyncMock(return_value=chunks)

        service = RetrievalService(engine=mock_engine)
        await service.refresh_index(mock_repository)

        with pytest.raises(RuntimeError, match="Inconsistent retrieval state"):
            service.search("query")
