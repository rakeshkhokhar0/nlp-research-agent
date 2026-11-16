"""Unit tests for RetrievalRepository."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

import app.database.model_registry  # Ensure all mappers are configured
from app.modules.ingestion.models import PaperChunk
from app.modules.papers.models import Paper
from app.modules.retrieval.repository import RetrievalRepository


class TestRetrievalRepositoryInitialization:
    """Tests for repository initialization and session validation."""

    def test_init_with_valid_async_session(self) -> None:
        """Repository accepts an instance of AsyncSession."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = RetrievalRepository(mock_session)
        assert repo.session is mock_session

    @pytest.mark.parametrize("invalid_session", [None, 123, "session", [AsyncSession], MagicMock()])
    def test_init_with_invalid_session_raises_type_error(self, invalid_session: object) -> None:
        """Non-AsyncSession instances raise TypeError."""
        with pytest.raises(TypeError, match="session must be an AsyncSession"):
            RetrievalRepository(invalid_session)  # type: ignore[arg-type]


class TestRetrievalRepositoryQueryStructure:
    """Tests for the constructed SQLAlchemy select statement and query options."""

    @pytest.mark.asyncio
    async def test_query_selects_paper_chunks_with_selectinload(self) -> None:
        """Query selects PaperChunk entities and eager-loads the parent Paper relationship."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = RetrievalRepository(mock_session)
        await repo.get_current_chunks()

        assert mock_session.execute.call_count == 1
        stmt = mock_session.execute.call_args[0][0]

        # Verify target entity is PaperChunk
        assert len(stmt.column_descriptions) == 1
        assert stmt.column_descriptions[0]["entity"] is PaperChunk

        # Verify eager-loading option for PaperChunk.paper via selectinload
        assert len(stmt._with_options) > 0
        loader_path = str(stmt._with_options[0].path)
        assert "PaperChunk.paper" in loader_path

    @pytest.mark.asyncio
    async def test_query_filters_strictly_current_chunks(self) -> None:
        """Query includes WHERE paper_chunks.is_current IS true to exclude historical chunks."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = RetrievalRepository(mock_session)
        await repo.get_current_chunks()

        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE paper_chunks.is_current IS true" in sql

    @pytest.mark.asyncio
    async def test_query_orders_by_paper_id_and_chunk_index(self) -> None:
        """Query orders deterministically by paper_id ASC, then chunk_index ASC."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = RetrievalRepository(mock_session)
        await repo.get_current_chunks()

        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "ORDER BY paper_chunks.paper_id ASC, paper_chunks.chunk_index ASC" in sql

        # Verify order by criteria directly on statement
        order_clauses = stmt._order_by_clauses
        assert len(order_clauses) == 2
        assert order_clauses[0].element.key == "paper_id"
        assert order_clauses[1].element.key == "chunk_index"

    @pytest.mark.asyncio
    async def test_query_with_paper_ids_filter(self) -> None:
        """Providing paper_ids adds an IN clause to restrict chunks to specified papers."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        p1 = uuid.uuid4()
        p2 = uuid.uuid4()
        repo = RetrievalRepository(mock_session)
        await repo.get_current_chunks(paper_ids=[p1, p2])

        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE paper_chunks.is_current IS true AND paper_chunks.paper_id IN" in sql

    @pytest.mark.asyncio
    async def test_empty_paper_ids_returns_empty_list_without_query(self) -> None:
        """Supplying an empty sequence of paper_ids returns [] immediately without querying."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = RetrievalRepository(mock_session)

        result_list = await repo.get_current_chunks(paper_ids=[])
        assert result_list == []
        mock_session.execute.assert_not_called()

        result_tuple = await repo.get_current_chunks(paper_ids=())
        assert result_tuple == []
        mock_session.execute.assert_not_called()

        result_set = await repo.get_current_chunks(paper_ids=set())
        assert result_set == []
        mock_session.execute.assert_not_called()

    @pytest.mark.parametrize("invalid_container", ["uuid-str", 123, True, {"not": "uuid_set"}])
    @pytest.mark.asyncio
    async def test_invalid_paper_ids_container_type_raises_type_error(self, invalid_container: object) -> None:
        """Non-sequence paper_ids raises TypeError."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = RetrievalRepository(mock_session)

        with pytest.raises(TypeError, match="paper_ids must be a collection of UUIDs"):
            await repo.get_current_chunks(paper_ids=invalid_container)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "invalid_element",
        [["not-a-uuid"], [123], [None], [uuid.uuid4(), "string-id"]],
    )
    @pytest.mark.asyncio
    async def test_invalid_paper_ids_elements_raise_type_error(self, invalid_element: list[object]) -> None:
        """Elements of paper_ids that are not uuid.UUID instances raise TypeError."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = RetrievalRepository(mock_session)

        with pytest.raises(TypeError, match="All elements in paper_ids must be uuid.UUID"):
            await repo.get_current_chunks(paper_ids=invalid_element)  # type: ignore[arg-type]


class TestRetrievalRepositoryExecution:
    """Tests for result extraction, relationship preservation, and edge cases."""

    @pytest.mark.asyncio
    async def test_get_current_chunks_returns_populated_entities(self) -> None:
        """Repository executes the query, extracts scalars, and returns PaperChunk instances."""
        mock_session = AsyncMock(spec=AsyncSession)

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
            content="The dominant sequence transduction models are based on complex recurrent networks.",
            is_current=True,
        )
        chunk1.paper = paper

        chunk2 = PaperChunk(
            id=uuid.uuid4(),
            paper_id=paper.id,
            ingestion_job_id=uuid.uuid4(),
            chunk_index=1,
            content="We propose a new simple network architecture, the Transformer.",
            is_current=True,
        )
        chunk2.paper = paper

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [chunk1, chunk2]
        mock_session.execute.return_value = mock_result

        repo = RetrievalRepository(mock_session)
        chunks = await repo.get_current_chunks()

        assert len(chunks) == 2
        assert chunks[0] is chunk1
        assert chunks[1] is chunk2
        assert chunks[0].paper.title == "Attention Is All You Need"
        assert chunks[1].paper.publication_year == 2017

    @pytest.mark.asyncio
    async def test_get_current_chunks_empty_database_returns_empty_list(self) -> None:
        """When no matching records exist in the database, returns an empty list."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = RetrievalRepository(mock_session)
        chunks = await repo.get_current_chunks()

        assert chunks == []
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_historical_chunks_excluded_by_criteria(self) -> None:
        """Verifies that the repository query criteria strictly excludes non-current chunks."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = RetrievalRepository(mock_session)
        await repo.get_current_chunks()

        stmt = mock_session.execute.call_args[0][0]
        # Inspect where criteria to ensure is_current is True
        where_clauses = stmt._where_criteria
        assert len(where_clauses) == 1
        criterion = where_clauses[0]
        assert criterion.left.key == "is_current"
        # Verify right side is SQL true element and operator is 'is'
        assert str(criterion.right).lower() == "true"
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "paper_chunks.is_current IS true" in sql

    @pytest.mark.asyncio
    async def test_multiple_calls_do_not_mutate_query_state(self) -> None:
        """Multiple calls with differing paper_ids do not leak filter state across invocations."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        repo = RetrievalRepository(mock_session)

        # Call 1: with paper_id filter
        p1 = uuid.uuid4()
        await repo.get_current_chunks(paper_ids=[p1])
        stmt1 = mock_session.execute.call_args[0][0]
        assert len(stmt1._where_criteria) == 2

        # Call 2: without filter
        await repo.get_current_chunks()
        stmt2 = mock_session.execute.call_args[0][0]
        assert len(stmt2._where_criteria) == 1
        sql2 = str(stmt2.compile(dialect=postgresql.dialect()))
        assert "paper_id IN" not in sql2
