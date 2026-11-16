"""Unit tests for IngestionRepository."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

import app.database.model_registry  # Ensure all mappers are configured
from app.modules.ingestion.chunker import ResearchChunk
from app.modules.ingestion.embedder import (
    ChunkEmbedding as ResultChunkEmbedding,
    EmbeddingConfig,
    EmbeddingDiagnostics,
    EmbeddingResult,
)
from app.modules.ingestion.models import IngestionJob, PaperChunk
from app.modules.ingestion.repository import (
    ChunkActivationError,
    ChunkPersistenceError,
    EmbeddingModelConfigurationError,
    EmbeddingPersistenceError,
    IngestionRepository,
    IngestionRepositoryError,
    JobNotFoundError,
)
from app.modules.papers.models import Paper
from app.modules.retrieval.models import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_EMBEDDING_DIMENSIONS,
    ChunkEmbedding,
    EmbeddingModel,
)


# =========================================================================
# Constructor Tests
# =========================================================================

class TestIngestionRepositoryConstructor:
    """Tests for repository initialization and session validation."""

    def test_init_with_valid_async_session(self) -> None:
        """Repository accepts an instance of AsyncSession."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        assert repo.session is mock_session

    @pytest.mark.parametrize(
        "invalid_session",
        [None, 123, "session", [AsyncSession], MagicMock()],
    )
    def test_init_with_invalid_session_raises_type_error(self, invalid_session: object) -> None:
        """Non-AsyncSession instances raise TypeError."""
        with pytest.raises(TypeError, match="session must be an AsyncSession"):
            IngestionRepository(invalid_session)  # type: ignore[arg-type]


# =========================================================================
# Ingestion Job Operations Tests
# =========================================================================

class TestIngestionRepositoryJobs:
    """Tests for IngestionJob CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_job_success(self) -> None:
        """create_job adds IngestionJob to session and flushes."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        paper_file_id = uuid.uuid4()
        job = await repo.create_job(
            paper_id=paper_id,
            paper_file_id=paper_file_id,
            status="pending",
            current_stage="parsing",
            pipeline_version="v1.0",
            metadata_={"test": True},
        )

        assert isinstance(job, IngestionJob)
        assert job.paper_id == paper_id
        assert job.paper_file_id == paper_file_id
        assert job.status == "pending"
        assert job.current_stage == "parsing"
        assert job.pipeline_version == "v1.0"
        assert job.metadata_ == {"test": True}
        assert job.started_at is None

        mock_session.add.assert_called_once_with(job)
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_job_processing_defaults_started_at_to_utc_now(self) -> None:
        """create_job with status='processing' automatically sets started_at if not provided."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        job = await repo.create_job(
            paper_id=paper_id,
            status="processing",
            current_stage="preprocessing",
        )
        assert job.status == "processing"
        assert job.started_at is not None
        assert isinstance(job.started_at, datetime)

    @pytest.mark.asyncio
    async def test_create_job_with_explicit_started_at_preserves_it(self) -> None:
        """create_job preserves an explicitly provided started_at timestamp."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        explicit_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        job = await repo.create_job(
            paper_id=paper_id,
            status="processing",
            started_at=explicit_time,
        )
        assert job.started_at == explicit_time

    @pytest.mark.parametrize("invalid_id", ["not-a-uuid", 123, None])
    @pytest.mark.asyncio
    async def test_create_job_invalid_paper_id_raises_type_error(self, invalid_id: object) -> None:
        """Invalid paper_id raises TypeError."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        with pytest.raises(TypeError, match="paper_id must be a uuid.UUID"):
            await repo.create_job(paper_id=invalid_id)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_file_id", ["not-a-uuid", 123])
    @pytest.mark.asyncio
    async def test_create_job_invalid_paper_file_id_raises_type_error(self, invalid_file_id: object) -> None:
        """Invalid paper_file_id raises TypeError."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        with pytest.raises(TypeError, match="paper_file_id must be a uuid.UUID"):
            await repo.create_job(paper_id=uuid.uuid4(), paper_file_id=invalid_file_id)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_status", ["running", "queued", "finished", "", 123])
    @pytest.mark.asyncio
    async def test_create_job_invalid_status_raises_value_error(self, invalid_status: str) -> None:
        """Invalid status string raises ValueError."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        with pytest.raises(ValueError, match="Invalid job status"):
            await repo.create_job(paper_id=uuid.uuid4(), status=invalid_status)

    @pytest.mark.asyncio
    async def test_get_job_existing(self) -> None:
        """get_job returns matching IngestionJob when present."""
        mock_session = AsyncMock(spec=AsyncSession)
        job_id = uuid.uuid4()
        expected_job = IngestionJob(id=job_id, paper_id=uuid.uuid4(), status="pending")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_job
        mock_session.execute.return_value = mock_result

        repo = IngestionRepository(mock_session)
        job = await repo.get_job(job_id)

        assert job is expected_job
        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE ingestion_jobs.id =" in sql
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_job_missing_returns_none(self) -> None:
        """get_job returns None when job does not exist."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = IngestionRepository(mock_session)
        job = await repo.get_job(uuid.uuid4())
        assert job is None

    @pytest.mark.parametrize("invalid_id", ["str-id", 123, None])
    @pytest.mark.asyncio
    async def test_get_job_invalid_id_raises_type_error(self, invalid_id: object) -> None:
        """get_job raises TypeError for invalid UUID argument."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        with pytest.raises(TypeError, match="job_id must be a uuid.UUID"):
            await repo.get_job(invalid_id)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_update_job_status_success(self) -> None:
        """update_job_status updates specified fields and flushes."""
        mock_session = AsyncMock(spec=AsyncSession)
        job_id = uuid.uuid4()
        existing_job = IngestionJob(id=job_id, paper_id=uuid.uuid4(), status="pending")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_job
        mock_session.execute.return_value = mock_result

        repo = IngestionRepository(mock_session)
        updated = await repo.update_job_status(
            job_id,
            status="completed",
            current_stage="done",
            error_message="no error",
            attempt_count=2,
            metadata_={"run": 1},
        )

        assert updated is existing_job
        assert updated.status == "completed"
        assert updated.current_stage == "done"
        assert updated.error_message == "no error"
        assert updated.attempt_count == 2
        assert updated.metadata_ == {"run": 1}

        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_job_status_not_found_raises_job_not_found_error(self) -> None:
        """update_job_status raises JobNotFoundError if job does not exist."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        repo = IngestionRepository(mock_session)
        with pytest.raises(JobNotFoundError, match="not found"):
            await repo.update_job_status(uuid.uuid4(), status="processing")

    @pytest.mark.parametrize("invalid_status", ["bogus", "active", ""])
    @pytest.mark.asyncio
    async def test_update_job_status_invalid_status_raises_value_error(self, invalid_status: str) -> None:
        """update_job_status raises ValueError for invalid status."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        with pytest.raises(ValueError, match="Invalid job status"):
            await repo.update_job_status(uuid.uuid4(), status=invalid_status)

    @pytest.mark.asyncio
    async def test_update_job_status_prevent_terminal_overwrite(self) -> None:
        """update_job_status with prevent_terminal_overwrite=True preserves terminal jobs."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        job_id = uuid.uuid4()
        completed_job = IngestionJob(id=job_id, paper_id=uuid.uuid4(), status="completed")
        repo.get_job = AsyncMock(return_value=completed_job)

        res = await repo.update_job_status(
            job_id,
            status="cancelled",
            prevent_terminal_overwrite=True,
        )
        assert res.status == "completed"
        mock_session.flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_job_if_active_updates_active_job(self) -> None:
        """cancel_job_if_active atomically executes UPDATE and returns the updated cancelled job."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        job_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        cancelled_job = IngestionJob(
            id=job_id,
            paper_id=uuid.uuid4(),
            status="cancelled",
            current_stage="chunking_cancelled",
            error_message="User requested cancel",
            completed_at=now,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cancelled_job
        mock_session.execute.return_value = mock_result

        res = await repo.cancel_job_if_active(
            job_id,
            current_stage="chunking_cancelled",
            error_message="User requested cancel",
            completed_at=now,
        )
        assert res is cancelled_job
        assert res.status == "cancelled"
        assert res.current_stage == "chunking_cancelled"
        assert res.error_message == "User requested cancel"
        assert res.completed_at == now

        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_job_if_active_generates_conditional_update_sql(self) -> None:
        """cancel_job_if_active generates an UPDATE statement with predicate on active status and RETURNING."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        job_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        cancelled_job = IngestionJob(id=job_id, paper_id=uuid.uuid4(), status="cancelled")

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cancelled_job
        mock_session.execute.return_value = mock_result

        await repo.cancel_job_if_active(
            job_id,
            current_stage="persisting_cancelled",
            error_message="Timed out",
            completed_at=now,
        )

        mock_session.execute.assert_awaited_once()
        stmt = mock_session.execute.call_args[0][0]
        compiled = str(stmt.compile(dialect=postgresql.dialect()))

        assert "UPDATE ingestion_jobs" in compiled
        assert "ingestion_jobs.id =" in compiled
        assert "ingestion_jobs.status IN" in compiled or "ingestion_jobs.status in" in compiled.lower()
        assert "RETURNING" in compiled
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

    @pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
    @pytest.mark.asyncio
    async def test_cancel_job_if_active_preserves_terminal_jobs(self, terminal_status: str) -> None:
        """cancel_job_if_active does not overwrite completed, failed, or cancelled jobs."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        job_id = uuid.uuid4()
        terminal_job = IngestionJob(id=job_id, paper_id=uuid.uuid4(), status=terminal_status)

        # UPDATE matches 0 rows because status is not in ACTIVE_JOB_STATUSES
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # get_job fetches the existing terminal job
        repo.get_job = AsyncMock(return_value=terminal_job)

        res = await repo.cancel_job_if_active(
            job_id,
            current_stage="cancelled_stage",
            error_message="Should not apply",
        )
        assert res is not None
        assert res.status == terminal_status
        assert res.current_stage != "cancelled_stage"
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_job_if_active_returns_none_for_missing_job(self) -> None:
        """cancel_job_if_active returns None if job does not exist."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        # UPDATE matches 0 rows
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # get_job finds nothing
        repo.get_job = AsyncMock(return_value=None)

        res = await repo.cancel_job_if_active(uuid.uuid4())
        assert res is None
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_job_if_active_rejects_invalid_uuid(self) -> None:
        """cancel_job_if_active raises TypeError for non-UUID job_id."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        with pytest.raises(TypeError, match="job_id must be a uuid.UUID"):
            await repo.cancel_job_if_active("invalid-uuid")  # type: ignore[arg-type]


# =========================================================================
# Embedding Model Resolution Tests
# =========================================================================

class TestIngestionRepositoryEmbeddingModel:
    """Tests for resolving and creating EmbeddingModel entities."""

    @pytest.mark.asyncio
    async def test_reuse_existing_compatible_model(self) -> None:
        """Returns existing model when compatible."""
        mock_session = AsyncMock(spec=AsyncSession)
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            normalize_embeddings=True,
            model_revision="v1",
        )
        existing_model = EmbeddingModel(
            id=uuid.uuid4(),
            provider=config.provider,
            model_name=config.model_name,
            model_version="v1",
            dimensions=384,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_model
        mock_session.execute.return_value = mock_result

        repo = IngestionRepository(mock_session)
        model = await repo.get_or_create_embedding_model(config)

        assert model is existing_model
        mock_session.flush.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_new_model_when_missing(self) -> None:
        """Executes insert on conflict do nothing and flushes when model does not exist."""
        mock_session = AsyncMock(spec=AsyncSession)
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            normalize_embeddings=True,
            model_revision=None,
        )

        new_model = EmbeddingModel(
            id=uuid.uuid4(),
            provider=config.provider,
            model_name=config.model_name,
            model_version=None,
            dimensions=384,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )

        # Call 1 (select): returns None
        mock_res1 = MagicMock()
        mock_res1.scalar_one_or_none.return_value = None

        # Call 2 (insert on conflict): returns execution result
        mock_res2 = MagicMock()

        # Call 3 (re-select): returns new_model
        mock_res3 = MagicMock()
        mock_res3.scalar_one.return_value = new_model

        mock_session.execute.side_effect = [mock_res1, mock_res2, mock_res3]

        repo = IngestionRepository(mock_session)
        model = await repo.get_or_create_embedding_model(config)

        assert model is new_model
        assert mock_session.execute.call_count == 3
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_conflict_path_rejects_inactive_model(self) -> None:
        """Initial lookup returns None, but insert conflict re-query returns an inactive model."""
        mock_session = AsyncMock(spec=AsyncSession)
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            normalize_embeddings=True,
            model_revision=None,
        )

        inactive_model = EmbeddingModel(
            id=uuid.uuid4(),
            provider=config.provider,
            model_name=config.model_name,
            model_version=None,
            dimensions=384,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=False,
        )

        mock_res1 = MagicMock()
        mock_res1.scalar_one_or_none.return_value = None  # Not found initially
        mock_res2 = MagicMock()  # Insert on conflict executes
        mock_res3 = MagicMock()
        mock_res3.scalar_one.return_value = inactive_model  # Re-query returns inactive concurrent insert

        mock_session.execute.side_effect = [mock_res1, mock_res2, mock_res3]

        repo = IngestionRepository(mock_session)
        with pytest.raises(EmbeddingModelConfigurationError, match="exists but is inactive"):
            await repo.get_or_create_embedding_model(config)

    @pytest.mark.asyncio
    async def test_conflict_path_rejects_incompatible_normalization(self) -> None:
        """Initial lookup returns None, but insert conflict re-query returns incompatible normalization."""
        mock_session = AsyncMock(spec=AsyncSession)
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            normalize_embeddings=True,
            model_revision=None,
        )

        incompatible_model = EmbeddingModel(
            id=uuid.uuid4(),
            provider=config.provider,
            model_name=config.model_name,
            model_version=None,
            dimensions=384,
            normalize_embeddings=False,  # Incompatible!
            distance_metric="cosine",
            is_active=True,
        )

        mock_res1 = MagicMock()
        mock_res1.scalar_one_or_none.return_value = None  # Not found initially
        mock_res2 = MagicMock()  # Insert on conflict executes
        mock_res3 = MagicMock()
        mock_res3.scalar_one.return_value = incompatible_model

        mock_session.execute.side_effect = [mock_res1, mock_res2, mock_res3]

        repo = IngestionRepository(mock_session)
        with pytest.raises(EmbeddingModelConfigurationError, match="incompatible with config normalize_embeddings"):
            await repo.get_or_create_embedding_model(config)

    @pytest.mark.asyncio
    async def test_conflict_path_accepts_compatible_model(self) -> None:
        """Initial lookup returns None, and insert conflict re-query returns compatible active model."""
        mock_session = AsyncMock(spec=AsyncSession)
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            normalize_embeddings=True,
            model_revision=None,
        )

        compatible_model = EmbeddingModel(
            id=uuid.uuid4(),
            provider=config.provider,
            model_name=config.model_name,
            model_version=None,
            dimensions=384,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )

        mock_res1 = MagicMock()
        mock_res1.scalar_one_or_none.return_value = None
        mock_res2 = MagicMock()
        mock_res3 = MagicMock()
        mock_res3.scalar_one.return_value = compatible_model

        mock_session.execute.side_effect = [mock_res1, mock_res2, mock_res3]

        repo = IngestionRepository(mock_session)
        model = await repo.get_or_create_embedding_model(config)

        assert model is compatible_model
        assert model.is_active is True
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reject_incompatible_dimensions_against_schema(self) -> None:
        """Rejects config whose dimensions != 384 before database interaction."""
        mock_session = AsyncMock(spec=AsyncSession)
        # Bypassing EmbeddingConfig validation to test repository guard
        config = MagicMock(spec=EmbeddingConfig)
        config.dimensions = 768
        config.provider = "sentence-transformers"
        config.model_name = "test-768"

        repo = IngestionRepository(mock_session)
        with pytest.raises(EmbeddingModelConfigurationError, match="strictly requires 384 dimensions"):
            await repo.get_or_create_embedding_model(config)

    @pytest.mark.asyncio
    async def test_reject_existing_model_when_inactive(self) -> None:
        """Raises EmbeddingModelConfigurationError if existing matching model is inactive."""
        mock_session = AsyncMock(spec=AsyncSession)
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=384,
            normalize_embeddings=True,
        )
        inactive_model = EmbeddingModel(
            id=uuid.uuid4(),
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=384,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=False,
        )

        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = inactive_model
        mock_session.execute.return_value = mock_res

        repo = IngestionRepository(mock_session)
        with pytest.raises(EmbeddingModelConfigurationError, match="exists but is inactive"):
            await repo.get_or_create_embedding_model(config)

    @pytest.mark.asyncio
    async def test_reject_existing_model_with_incompatible_normalization(self) -> None:
        """Raises EmbeddingModelConfigurationError if existing model normalization differs."""
        mock_session = AsyncMock(spec=AsyncSession)
        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=384,
            normalize_embeddings=True,
        )
        existing_model = EmbeddingModel(
            id=uuid.uuid4(),
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=384,
            normalize_embeddings=False,  # Mismatch!
            distance_metric="cosine",
            is_active=True,
        )

        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = existing_model
        mock_session.execute.return_value = mock_res

        repo = IngestionRepository(mock_session)
        with pytest.raises(EmbeddingModelConfigurationError, match="incompatible with config normalize_embeddings"):
            await repo.get_or_create_embedding_model(config)

    @pytest.mark.asyncio
    async def test_reject_invalid_config_type(self) -> None:
        """Non-EmbeddingConfig raises TypeError."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        with pytest.raises(TypeError, match="config must be an instance of EmbeddingConfig"):
            await repo.get_or_create_embedding_model("not-a-config")  # type: ignore[arg-type]


# =========================================================================
# Bulk Chunk Persistence Tests
# =========================================================================

class TestIngestionRepositoryChunks:
    """Tests for bulk_create_chunks and chunk persistence."""

    def _sample_chunk(self, index: int, content: str = "Test chunk content.") -> ResearchChunk:
        return ResearchChunk(
            chunk_index=index,
            content=content,
            token_count=10,
            char_count=len(content),
            page_start=1,
            page_end=2,
            section_title="Introduction",
            content_hash=f"hash_{index}",
            metadata={"source_paragraph_indices": [index]},
        )

    @pytest.mark.asyncio
    async def test_bulk_create_chunks_success(self) -> None:
        """Persists ResearchChunk objects as PaperChunk entities with is_current=False."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        job_id = uuid.uuid4()
        chunks = [
            self._sample_chunk(0, "Chunk 0"),
            self._sample_chunk(1, "Chunk 1"),
        ]

        res = await repo.bulk_create_chunks(
            paper_id=paper_id,
            ingestion_job_id=job_id,
            chunks=chunks,
            chunking_version="section-aware-v1",
        )

        assert len(res) == 2
        assert res[0].chunk_index == 0
        assert res[0].content == "Chunk 0"
        assert res[0].is_current is False
        assert res[0].paper_id == paper_id
        assert res[0].ingestion_job_id == job_id
        assert res[0].chunking_version == "section-aware-v1"

        assert res[1].chunk_index == 1
        assert res[1].is_current is False

        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_bulk_create_chunks_empty_returns_empty_list(self) -> None:
        """Empty input list returns [] without flushing."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        res = await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), [])
        assert res == []
        mock_session.add_all.assert_not_called()
        mock_session.flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_bulk_create_chunks_rejects_duplicate_chunk_index(self) -> None:
        """Raises ChunkPersistenceError if duplicate chunk_index is supplied."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        chunks = [
            self._sample_chunk(0, "A"),
            self._sample_chunk(0, "B"),  # Duplicate!
        ]

        with pytest.raises(ChunkPersistenceError, match="Duplicate chunk_index 0"):
            await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), chunks)

    @pytest.mark.parametrize("blank_content", ["", "   ", "\t\n"])
    @pytest.mark.asyncio
    async def test_bulk_create_chunks_rejects_blank_content(self, blank_content: str) -> None:
        """Raises ChunkPersistenceError if content is empty or whitespace."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        chunks = [self._sample_chunk(0, blank_content)]
        with pytest.raises(ChunkPersistenceError, match="empty or blank content"):
            await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), chunks)

    @pytest.mark.parametrize("invalid_token_count", [-1, True, False, "10", 1.5])
    @pytest.mark.asyncio
    async def test_bulk_create_chunks_rejects_invalid_token_count(self, invalid_token_count: object) -> None:
        """Raises ChunkPersistenceError if token_count is negative, bool, or non-int."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        chunk = ResearchChunk(
            chunk_index=0,
            content="Valid content",
            token_count=invalid_token_count,  # type: ignore[arg-type]
            char_count=13,
            page_start=1,
            page_end=1,
            section_title="Intro",
            content_hash="abc",
            metadata={},
        )
        with pytest.raises(ChunkPersistenceError, match="invalid token_count"):
            await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), [chunk])

    @pytest.mark.parametrize("invalid_char_count", [-1, True, False, "13", 2.5])
    @pytest.mark.asyncio
    async def test_bulk_create_chunks_rejects_invalid_char_count(self, invalid_char_count: object) -> None:
        """Raises ChunkPersistenceError if char_count is negative, bool, or non-int."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        chunk = ResearchChunk(
            chunk_index=0,
            content="Valid content",
            token_count=10,
            char_count=invalid_char_count,  # type: ignore[arg-type]
            page_start=1,
            page_end=1,
            section_title="Intro",
            content_hash="abc",
            metadata={},
        )
        with pytest.raises(ChunkPersistenceError, match="invalid char_count"):
            await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), [chunk])

    @pytest.mark.parametrize("invalid_page_start", [0, -1, True, False, "1", 1.0])
    @pytest.mark.asyncio
    async def test_bulk_create_chunks_rejects_invalid_page_start(self, invalid_page_start: object) -> None:
        """Raises ChunkPersistenceError if page_start is non-positive, bool, or non-int."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        chunk = ResearchChunk(
            chunk_index=0,
            content="Valid content",
            token_count=10,
            char_count=13,
            page_start=invalid_page_start,  # type: ignore[arg-type]
            page_end=2,
            section_title="Intro",
            content_hash="abc",
            metadata={},
        )
        with pytest.raises(ChunkPersistenceError, match="invalid page_start"):
            await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), [chunk])

    @pytest.mark.parametrize("invalid_page_end", [0, -1, True, False, "2", 2.0])
    @pytest.mark.asyncio
    async def test_bulk_create_chunks_rejects_invalid_page_end(self, invalid_page_end: object) -> None:
        """Raises ChunkPersistenceError if page_end is non-positive, bool, or non-int."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        chunk = ResearchChunk(
            chunk_index=0,
            content="Valid content",
            token_count=10,
            char_count=13,
            page_start=1,
            page_end=invalid_page_end,  # type: ignore[arg-type]
            section_title="Intro",
            content_hash="abc",
            metadata={},
        )
        with pytest.raises(ChunkPersistenceError, match="invalid page_end"):
            await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), [chunk])

    @pytest.mark.parametrize("invalid_metadata", ["not-a-dict", [1, 2, 3], 123, True])
    @pytest.mark.asyncio
    async def test_bulk_create_chunks_rejects_invalid_metadata(self, invalid_metadata: object) -> None:
        """Raises ChunkPersistenceError if metadata is not a dictionary."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        chunk = ResearchChunk(
            chunk_index=0,
            content="Valid content",
            token_count=10,
            char_count=13,
            page_start=1,
            page_end=1,
            section_title="Intro",
            content_hash="abc",
            metadata=invalid_metadata,  # type: ignore[arg-type]
        )
        with pytest.raises(ChunkPersistenceError, match="invalid metadata type"):
            await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), [chunk])

    @pytest.mark.parametrize("invalid_hash", ["", "   ", None, 123, True])
    @pytest.mark.asyncio
    async def test_bulk_create_chunks_rejects_invalid_content_hash(self, invalid_hash: object) -> None:
        """Raises ChunkPersistenceError if content_hash is empty, whitespace, or non-string."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)
        chunk = ResearchChunk(
            chunk_index=0,
            content="Valid content",
            token_count=10,
            char_count=13,
            page_start=1,
            page_end=1,
            section_title="Intro",
            content_hash=invalid_hash,  # type: ignore[arg-type]
            metadata={},
        )
        with pytest.raises(ChunkPersistenceError, match="invalid content_hash"):
            await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), [chunk])

    @pytest.mark.asyncio
    async def test_bulk_create_chunks_rejects_invalid_page_range(self) -> None:
        """Raises ChunkPersistenceError if page_end < page_start."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        chunk = ResearchChunk(
            chunk_index=0,
            content="Valid content",
            token_count=10,
            char_count=13,
            page_start=5,
            page_end=2,  # Invalid!
            section_title="Intro",
            content_hash="abc",
            metadata={},
        )

        with pytest.raises(ChunkPersistenceError, match="page_end .* < page_start"):
            await repo.bulk_create_chunks(uuid.uuid4(), uuid.uuid4(), [chunk])

    @pytest.mark.asyncio
    async def test_get_job_chunks_deterministic_order(self) -> None:
        """get_job_chunks queries chunks ordered by chunk_index ASC."""
        mock_session = AsyncMock(spec=AsyncSession)
        job_id = uuid.uuid4()
        c0 = PaperChunk(chunk_index=0, content="c0")
        c1 = PaperChunk(chunk_index=1, content="c1")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [c0, c1]
        mock_session.execute.return_value = mock_result

        repo = IngestionRepository(mock_session)
        res = await repo.get_job_chunks(job_id)

        assert res == [c0, c1]
        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE paper_chunks.ingestion_job_id =" in sql
        assert "ORDER BY paper_chunks.chunk_index ASC" in sql


# =========================================================================
# Bulk Embedding Persistence Tests
# =========================================================================

class TestIngestionRepositoryEmbeddings:
    """Tests for bulk_create_embeddings."""

    def _sample_paper_chunk(self, index: int, hash_val: str) -> PaperChunk:
        return PaperChunk(
            id=uuid.uuid4(),
            paper_id=uuid.uuid4(),
            ingestion_job_id=uuid.uuid4(),
            chunk_index=index,
            content=f"Content for chunk {index}",
            content_hash=hash_val,
        )

    def _sample_result(self, items: list[tuple[int, str, list[float]]]) -> EmbeddingResult:
        embeddings = [
            ResultChunkEmbedding(chunk_index=idx, content_hash=h, vector=vec)
            for idx, h, vec in items
        ]
        return EmbeddingResult(
            embeddings=embeddings,
            model_name="test-model",
            provider="sentence-transformers",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalized=True,
            diagnostics=EmbeddingDiagnostics(
                input_chunks=len(items),
                embedded_chunks=len(items),
                batch_count=1,
                elapsed_seconds=0.1,
            ),
        )

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_success(self) -> None:
        """Persists ChunkEmbedding records matching chunks by chunk_index and content_hash."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        c0 = self._sample_paper_chunk(0, "hash_0")
        c1 = self._sample_paper_chunk(1, "hash_1")

        vec0 = [0.1] * DEFAULT_EMBEDDING_DIMENSIONS
        vec1 = [0.2] * DEFAULT_EMBEDDING_DIMENSIONS

        # Supply result in reverse order to verify matching by index, not list position
        result = self._sample_result([(1, "hash_1", vec1), (0, "hash_0", vec0)])
        model_id = uuid.uuid4()

        db_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )

        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        res = await repo.bulk_create_embeddings([c0, c1], result, model_id)

        assert len(res) == 2
        # Deterministic chunk_index order
        assert res[0].chunk_id == c0.id
        assert res[0].embedding == vec0
        assert res[0].embedding_model_id == model_id

        assert res[1].chunk_id == c1.id
        assert res[1].embedding == vec1

        mock_session.add_all.assert_called_once()
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_missing_model_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when EmbeddingModel is not found in database."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "h0")
        result = self._sample_result([(0, "h0", [0.1] * DEFAULT_EMBEDDING_DIMENSIONS)])

        with pytest.raises(EmbeddingPersistenceError, match="EmbeddingModel with id .* not found"):
            await repo.bulk_create_embeddings([c0], result, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_inactive_model_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when EmbeddingModel has is_active=False."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        inactive_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=False,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = inactive_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "h0")
        result = self._sample_result([(0, "h0", [0.1] * DEFAULT_EMBEDDING_DIMENSIONS)])

        with pytest.raises(EmbeddingPersistenceError, match="is inactive"):
            await repo.bulk_create_embeddings([c0], result, model_id)

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_provider_mismatch_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when model provider does not match result."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        db_model = EmbeddingModel(
            id=model_id,
            provider="different-provider",
            model_name="test-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "h0")
        result = self._sample_result([(0, "h0", [0.1] * DEFAULT_EMBEDDING_DIMENSIONS)])

        with pytest.raises(EmbeddingPersistenceError, match="provider mismatch"):
            await repo.bulk_create_embeddings([c0], result, model_id)

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_model_name_mismatch_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when model_name does not match result."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        db_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="different-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "h0")
        result = self._sample_result([(0, "h0", [0.1] * DEFAULT_EMBEDDING_DIMENSIONS)])

        with pytest.raises(EmbeddingPersistenceError, match="model name mismatch"):
            await repo.bulk_create_embeddings([c0], result, model_id)

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_dimensions_mismatch_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when DB dimensions do not match result."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        db_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=384,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "h0")
        # EmbeddingResult specifies 512 dimensions
        result = EmbeddingResult(
            embeddings=[ResultChunkEmbedding(0, "h0", [0.1] * 512)],
            model_name="test-model",
            provider="sentence-transformers",
            dimensions=512,
            normalized=True,
            diagnostics=EmbeddingDiagnostics(1, 1, 1, 0.1),
        )

        with pytest.raises(EmbeddingPersistenceError, match="dimensions mismatch"):
            await repo.bulk_create_embeddings([c0], result, model_id)

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_normalization_mismatch_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when DB normalization does not match result."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        db_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "h0")
        # EmbeddingResult specifies normalized=False
        result = EmbeddingResult(
            embeddings=[ResultChunkEmbedding(0, "h0", [0.1] * DEFAULT_EMBEDDING_DIMENSIONS)],
            model_name="test-model",
            provider="sentence-transformers",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalized=False,
            diagnostics=EmbeddingDiagnostics(1, 1, 1, 0.1),
        )

        with pytest.raises(EmbeddingPersistenceError, match="normalization mismatch"):
            await repo.bulk_create_embeddings([c0], result, model_id)

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_count_mismatch_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when counts do not match."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        db_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "hash_0")
        result = self._sample_result([])  # 0 embeddings vs 1 chunk

        with pytest.raises(EmbeddingPersistenceError, match="Count mismatch"):
            await repo.bulk_create_embeddings([c0], result, model_id)

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_hash_mismatch_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when content_hash does not match chunk."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        db_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "original_hash")
        vec = [0.1] * DEFAULT_EMBEDDING_DIMENSIONS
        result = self._sample_result([(0, "different_hash", vec)])

        with pytest.raises(EmbeddingPersistenceError, match="Content hash mismatch"):
            await repo.bulk_create_embeddings([c0], result, model_id)

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_missing_chunk_index_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when an embedding index has no matching chunk."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        db_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "h0")
        vec = [0.1] * DEFAULT_EMBEDDING_DIMENSIONS
        result = self._sample_result([(99, "h0", vec)])  # index 99 does not match chunk 0

        with pytest.raises(EmbeddingPersistenceError, match="has no corresponding PaperChunk"):
            await repo.bulk_create_embeddings([c0], result, model_id)

    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_invalid_dimensions_raises_error(self) -> None:
        """Raises EmbeddingPersistenceError when vector length != 384."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        db_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "h0")
        vec = [0.1] * 128  # Invalid dimensions
        result = self._sample_result([(0, "h0", vec)])

        with pytest.raises(EmbeddingPersistenceError, match="dimension mismatch"):
            await repo.bulk_create_embeddings([c0], result, model_id)

    @pytest.mark.parametrize("bad_val", [math.nan, math.inf, -math.inf, True])
    @pytest.mark.asyncio
    async def test_bulk_create_embeddings_non_finite_values_raises_error(self, bad_val: object) -> None:
        """Raises EmbeddingPersistenceError when vector contains NaN, Inf, or non-reals."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        model_id = uuid.uuid4()
        db_model = EmbeddingModel(
            id=model_id,
            provider="sentence-transformers",
            model_name="test-model",
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            normalize_embeddings=True,
            distance_metric="cosine",
            is_active=True,
        )
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = db_model
        mock_session.execute.return_value = mock_res

        c0 = self._sample_paper_chunk(0, "h0")
        vec = [0.1] * DEFAULT_EMBEDDING_DIMENSIONS
        vec[5] = bad_val  # type: ignore[call-overload]
        result = self._sample_result([(0, "h0", vec)])

        with pytest.raises(EmbeddingPersistenceError, match="invalid or non-finite"):
            await repo.bulk_create_embeddings([c0], result, model_id)


# =========================================================================
# Atomic Version Activation Tests
# =========================================================================

class TestIngestionRepositoryActivation:
    """Tests for activate_chunks_for_job."""

    @pytest.mark.asyncio
    async def test_activate_chunks_success(self) -> None:
        """Locks paper, checks completeness, demotes old chunks, promotes staged chunks."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        job_id = uuid.uuid4()
        model_id = uuid.uuid4()

        paper = Paper(id=paper_id, title="Test Paper")
        job = IngestionJob(id=job_id, paper_id=paper_id, status="processing")

        chunk0 = PaperChunk(id=uuid.uuid4(), paper_id=paper_id, ingestion_job_id=job_id, chunk_index=0, is_current=False)
        chunk1 = PaperChunk(id=uuid.uuid4(), paper_id=paper_id, ingestion_job_id=job_id, chunk_index=1, is_current=False)
        staged_chunks = [chunk0, chunk1]

        # Call 1: Paper select with for update
        res_paper = MagicMock()
        res_paper.scalar_one_or_none.return_value = paper

        # Call 2: Job select
        res_job = MagicMock()
        res_job.scalar_one_or_none.return_value = job

        # Call 3: Staged chunks select
        res_chunks = MagicMock()
        res_chunks.scalars.return_value.all.return_value = staged_chunks

        # Call 4: Embedding count select
        res_emb_count = MagicMock()
        res_emb_count.scalar_one.return_value = 2  # Matches 2 staged chunks

        # Call 5: Demote update
        res_demote = MagicMock()

        # Call 6: Promote update
        res_promote = MagicMock()

        # Call 7: Final select of activated chunks
        activated_chunk0 = PaperChunk(id=chunk0.id, paper_id=paper_id, chunk_index=0, is_current=True)
        activated_chunk1 = PaperChunk(id=chunk1.id, paper_id=paper_id, chunk_index=1, is_current=True)
        res_final = MagicMock()
        res_final.scalars.return_value.all.return_value = [activated_chunk0, activated_chunk1]

        mock_session.execute.side_effect = [
            res_paper,
            res_job,
            res_chunks,
            res_emb_count,
            res_demote,
            res_promote,
            res_final,
        ]

        activated = await repo.activate_chunks_for_job(paper_id, job_id, model_id)

        assert len(activated) == 2
        assert activated[0].is_current is True
        assert activated[1].is_current is True

        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()

        # Verify Paper row-locking SQL
        paper_call_stmt = mock_session.execute.call_args_list[0][0][0]
        sql_paper = str(paper_call_stmt.compile(dialect=postgresql.dialect()))
        assert "FOR UPDATE" in sql_paper

        # Verify Demote and Promote updates
        demote_stmt = mock_session.execute.call_args_list[4][0][0]
        sql_demote = str(demote_stmt.compile(dialect=postgresql.dialect()))
        assert "UPDATE paper_chunks SET is_current=" in sql_demote
        assert "paper_chunks.paper_id =" in sql_demote
        assert "paper_chunks.is_current IS true" in sql_demote

        promote_stmt = mock_session.execute.call_args_list[5][0][0]
        sql_promote = str(promote_stmt.compile(dialect=postgresql.dialect()))
        assert "UPDATE paper_chunks SET is_current=" in sql_promote
        assert "paper_chunks.ingestion_job_id =" in sql_promote

    @pytest.mark.asyncio
    async def test_activate_chunks_job_paper_mismatch_raises_error(self) -> None:
        """Raises ChunkActivationError if job belongs to a different paper."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        other_paper_id = uuid.uuid4()
        job_id = uuid.uuid4()

        paper = Paper(id=paper_id, title="Target Paper")
        job = IngestionJob(id=job_id, paper_id=other_paper_id, status="processing")

        res_paper = MagicMock()
        res_paper.scalar_one_or_none.return_value = paper
        res_job = MagicMock()
        res_job.scalar_one_or_none.return_value = job

        mock_session.execute.side_effect = [res_paper, res_job]

        with pytest.raises(ChunkActivationError, match="belongs to paper .* not"):
            await repo.activate_chunks_for_job(paper_id, job_id, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_activate_chunks_empty_chunks_raises_error(self) -> None:
        """Raises ChunkActivationError if job has 0 staged chunks."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        job_id = uuid.uuid4()

        paper = Paper(id=paper_id, title="Target Paper")
        job = IngestionJob(id=job_id, paper_id=paper_id, status="processing")

        res_paper = MagicMock()
        res_paper.scalar_one_or_none.return_value = paper
        res_job = MagicMock()
        res_job.scalar_one_or_none.return_value = job
        res_chunks = MagicMock()
        res_chunks.scalars.return_value.all.return_value = []

        mock_session.execute.side_effect = [res_paper, res_job, res_chunks]

        with pytest.raises(ChunkActivationError, match="No staged chunks found"):
            await repo.activate_chunks_for_job(paper_id, job_id, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_activate_chunks_incomplete_embeddings_raises_error(self) -> None:
        """Raises ChunkActivationError if staged chunks do not have complete embeddings."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        job_id = uuid.uuid4()
        model_id = uuid.uuid4()

        paper = Paper(id=paper_id, title="Target Paper")
        job = IngestionJob(id=job_id, paper_id=paper_id, status="processing")
        chunk0 = PaperChunk(id=uuid.uuid4(), paper_id=paper_id, ingestion_job_id=job_id, chunk_index=0)
        chunk1 = PaperChunk(id=uuid.uuid4(), paper_id=paper_id, ingestion_job_id=job_id, chunk_index=1)

        res_paper = MagicMock()
        res_paper.scalar_one_or_none.return_value = paper
        res_job = MagicMock()
        res_job.scalar_one_or_none.return_value = job
        res_chunks = MagicMock()
        res_chunks.scalars.return_value.all.return_value = [chunk0, chunk1]
        res_emb_count = MagicMock()
        res_emb_count.scalar_one.return_value = 1  # Incomplete: 1 embedding for 2 chunks!

        mock_session.execute.side_effect = [res_paper, res_job, res_chunks, res_emb_count]

        with pytest.raises(ChunkActivationError, match="Embedding set is incomplete"):
            await repo.activate_chunks_for_job(paper_id, job_id, model_id)


# =========================================================================
# Locking & Concurrency Repository Tests
# =========================================================================

class TestIngestionRepositoryLockingAndQueries:
    """Tests for lock_paper, get_active_job_for_paper, and get_latest_completed_job."""

    @pytest.mark.asyncio
    async def test_lock_paper_success(self) -> None:
        """lock_paper executes SELECT ... FOR UPDATE."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        expected_paper = Paper(id=paper_id, title="Test")

        res = MagicMock()
        res.scalar_one_or_none.return_value = expected_paper
        mock_session.execute.return_value = res

        paper = await repo.lock_paper(paper_id)
        assert paper is expected_paper

        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE papers.id =" in sql
        assert "FOR UPDATE" in sql

    @pytest.mark.asyncio
    async def test_get_active_job_for_paper(self) -> None:
        """get_active_job_for_paper finds in-progress job."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        paper_id = uuid.uuid4()
        active_job = IngestionJob(id=uuid.uuid4(), paper_id=paper_id, status="processing")

        res = MagicMock()
        res.scalars.return_value.first.return_value = active_job
        mock_session.execute.return_value = res

        job = await repo.get_active_job_for_paper(paper_id)
        assert job is active_job

        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE ingestion_jobs.paper_id =" in sql
        assert "ingestion_jobs.status IN" in sql

    @pytest.mark.asyncio
    async def test_get_latest_completed_job(self) -> None:
        """get_latest_completed_job filters by file id and completed status."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = IngestionRepository(mock_session)

        file_id = uuid.uuid4()
        completed_job = IngestionJob(id=uuid.uuid4(), paper_id=uuid.uuid4(), paper_file_id=file_id, status="completed")

        res = MagicMock()
        res.scalars.return_value.first.return_value = completed_job
        mock_session.execute.return_value = res

        job = await repo.get_latest_completed_job(file_id, pipeline_version="v1")
        assert job is completed_job

        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE ingestion_jobs.paper_file_id =" in sql
        assert "ingestion_jobs.status = %(status_1)s" in sql
        assert "ingestion_jobs.pipeline_version = %(pipeline_version_1)s" in sql

