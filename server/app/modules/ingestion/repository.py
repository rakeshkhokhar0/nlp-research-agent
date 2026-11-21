"""Database repository for persisting ingestion jobs, paper chunks, and embeddings."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import numbers
from typing import Any, Sequence
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

import app.database.model_registry  # noqa: F401 - ensures cross-module relationship mappers are registered
from app.modules.ingestion.chunker import ResearchChunk
from app.modules.ingestion.embedder import ChunkEmbedding as ResultChunkEmbedding, EmbeddingConfig, EmbeddingResult
from app.modules.ingestion.models import (
    ACTIVE_JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    IngestionJob,
    PaperChunk,
)
from app.modules.papers.models import Paper
from app.modules.retrieval.models import (
    DEFAULT_DISTANCE_METRIC,
    DEFAULT_EMBEDDING_DIMENSIONS,
    ChunkEmbedding,
    EmbeddingModel,
)


class IngestionRepositoryError(Exception):
    """Base exception for all ingestion repository errors."""


class JobNotFoundError(IngestionRepositoryError):
    """Raised when a requested ingestion job cannot be found."""


class EmbeddingModelConfigurationError(IngestionRepositoryError):
    """Raised when an embedding model configuration is incompatible with schema constraints."""


class ChunkPersistenceError(IngestionRepositoryError):
    """Raised when chunk validation or persistence fails."""


class EmbeddingPersistenceError(IngestionRepositoryError):
    """Raised when embedding validation or persistence fails."""


class ChunkActivationError(IngestionRepositoryError):
    """Raised when atomic chunk activation preconditions fail."""


class IngestionRepository:
    """Repository managing database persistence for ingestion jobs, chunks, and embeddings.

    Adheres strictly to transactional boundaries managed by the service layer:
    never initiates commits or rollbacks internally, relying on flush() when database
    IDs or constraints need immediate synchronization.
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

    async def create_job(
        self,
        paper_id: uuid.UUID,
        paper_file_id: uuid.UUID | None = None,
        status: str = "pending",
        current_stage: str | None = None,
        pipeline_version: str | None = None,
        metadata_: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> IngestionJob:
        """Create and flush a new IngestionJob record.

        Args:
            paper_id: UUID of parent research paper.
            paper_file_id: Optional UUID of paper file being processed.
            status: Initial job status (default 'pending').
            current_stage: Pipeline stage name (e.g. 'parsing', 'chunking').
            pipeline_version: Version identifier of the pipeline code.
            metadata_: Optional metadata dict.
            started_at: Optional timestamp when processing started. Defaults to
                current UTC timestamp if status is 'processing' and started_at is omitted.

        Returns:
            The flushed IngestionJob instance with its generated UUID and timestamps.

        Raises:
            TypeError: If paper_id or paper_file_id are not UUID instances.
            ValueError: If status is not a recognized ingestion job status.
        """
        if not isinstance(paper_id, uuid.UUID):
            raise TypeError(f"paper_id must be a uuid.UUID, got {type(paper_id).__name__}")
        if paper_file_id is not None and not isinstance(paper_file_id, uuid.UUID):
            raise TypeError(f"paper_file_id must be a uuid.UUID, got {type(paper_file_id).__name__}")

        valid_statuses = ACTIVE_JOB_STATUSES | TERMINAL_JOB_STATUSES
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid job status '{status}'. Valid statuses: {sorted(valid_statuses)}"
            )

        if started_at is None and status == "processing":
            started_at = datetime.now(timezone.utc)

        job = IngestionJob(
            paper_id=paper_id,
            paper_file_id=paper_file_id,
            status=status,
            current_stage=current_stage,
            pipeline_version=pipeline_version,
            metadata_=metadata_ if metadata_ is not None else {},
            started_at=started_at,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_job(self, job_id: uuid.UUID) -> IngestionJob | None:
        """Retrieve an IngestionJob by its primary key ID.

        Args:
            job_id: UUID of the ingestion job.

        Returns:
            The IngestionJob instance if found, or None.

        Raises:
            TypeError: If job_id is not a uuid.UUID.
        """
        if not isinstance(job_id, uuid.UUID):
            raise TypeError(f"job_id must be a uuid.UUID, got {type(job_id).__name__}")

        stmt = select(IngestionJob).where(IngestionJob.id == job_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_job_status(
        self,
        job_id: uuid.UUID,
        status: str,
        *,
        current_stage: str | None = None,
        started_at: Any = None,
        completed_at: Any = None,
        error_message: str | None = None,
        attempt_count: int | None = None,
        metadata_: dict[str, Any] | None = None,
        prevent_terminal_overwrite: bool = False,
    ) -> IngestionJob:
        """Update the status and execution fields of an existing IngestionJob.

        Args:
            job_id: UUID of target ingestion job.
            status: New job status.
            current_stage: Optional updated pipeline stage.
            started_at: Optional timestamp when processing started.
            completed_at: Optional timestamp when processing concluded.
            error_message: Optional error message string if failed.
            attempt_count: Optional explicit attempt count to set.
            metadata_: Optional metadata updates.
            prevent_terminal_overwrite: If True, skips modifying the job if it
                has already reached a terminal status ('completed', 'failed', 'cancelled').

        Returns:
            The updated and flushed IngestionJob instance (or existing instance if
            prevent_terminal_overwrite was triggered).

        Raises:
            TypeError: If job_id is not a uuid.UUID.
            ValueError: If status is not a valid status string.
            JobNotFoundError: If no job with job_id exists.
        """
        if not isinstance(job_id, uuid.UUID):
            raise TypeError(f"job_id must be a uuid.UUID, got {type(job_id).__name__}")

        valid_statuses = ACTIVE_JOB_STATUSES | TERMINAL_JOB_STATUSES
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid job status '{status}'. Valid statuses: {sorted(valid_statuses)}"
            )

        job = await self.get_job(job_id)
        if job is None:
            raise JobNotFoundError(f"IngestionJob with id {job_id} not found.")

        if prevent_terminal_overwrite and job.status in TERMINAL_JOB_STATUSES:
            return job

        job.status = status
        if current_stage is not None:
            job.current_stage = current_stage
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at
        if error_message is not None:
            job.error_message = error_message
        if attempt_count is not None:
            if attempt_count < 0:
                raise ValueError(f"attempt_count cannot be negative, got {attempt_count}")
            job.attempt_count = attempt_count
        if metadata_ is not None:
            job.metadata_ = metadata_

        await self.session.flush()
        return job

    async def cancel_job_if_active(
        self,
        job_id: uuid.UUID,
        *,
        current_stage: str | None = None,
        error_message: str | None = None,
        completed_at: Any = None,
    ) -> IngestionJob | None:
        """Atomically cancel an IngestionJob only if it is currently in an active non-terminal state.

        Executes a single conditional UPDATE statement matching `id == job_id` and
        `status in ACTIVE_JOB_STATUSES`. If the job is already in a terminal state
        ('completed', 'failed', 'cancelled'), no row is updated and the existing
        job is returned unmodified. If the job does not exist, returns None.

        Args:
            job_id: UUID of target ingestion job.
            current_stage: Optional updated pipeline stage (e.g. 'chunking_cancelled').
            error_message: Optional cancellation message.
            completed_at: Optional timestamp when cancellation concluded.

        Returns:
            The IngestionJob instance (either updated if active, or unchanged if terminal),
            or None if the job is not found.

        Raises:
            TypeError: If job_id is not a uuid.UUID.
        """
        if not isinstance(job_id, uuid.UUID):
            raise TypeError(f"job_id must be a uuid.UUID, got {type(job_id).__name__}")

        values_to_update: dict[str, Any] = {"status": "cancelled"}
        if current_stage is not None:
            values_to_update["current_stage"] = current_stage
        if error_message is not None:
            values_to_update["error_message"] = error_message
        if completed_at is not None:
            values_to_update["completed_at"] = completed_at

        stmt = (
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status.in_(ACTIVE_JOB_STATUSES),
            )
            .values(**values_to_update)
            .returning(IngestionJob)
            .execution_options(synchronize_session="auto")
        )

        result = await self.session.execute(stmt)
        updated_job = result.scalar_one_or_none()
        if updated_job is not None:
            return updated_job

        # No row was updated: either the job does not exist or it is already in a terminal status.
        # Fetch and return the existing job without changing it.
        return await self.get_job(job_id)

    def _validate_embedding_model_compatibility(
        self,
        model: EmbeddingModel,
        config: EmbeddingConfig,
        distance_metric: str,
    ) -> None:
        """Validate that an existing or re-queried EmbeddingModel is active and compatible.

        Args:
            model: EmbeddingModel entity from database.
            config: Desired EmbeddingConfig.
            distance_metric: Required distance metric string (e.g. 'cosine').

        Raises:
            EmbeddingModelConfigurationError: If the model is inactive or incompatible.
        """
        if not model.is_active:
            raise EmbeddingModelConfigurationError(
                f"Matching EmbeddingModel {model.id} exists but is inactive (is_active=False)."
            )
        if model.dimensions != config.dimensions:
            raise EmbeddingModelConfigurationError(
                f"Existing EmbeddingModel {model.id} has dimensions={model.dimensions}, "
                f"incompatible with config dimensions={config.dimensions}."
            )
        if model.normalize_embeddings != config.normalize_embeddings:
            raise EmbeddingModelConfigurationError(
                f"Existing EmbeddingModel {model.id} has normalize_embeddings={model.normalize_embeddings}, "
                f"incompatible with config normalize_embeddings={config.normalize_embeddings}."
            )
        if model.distance_metric != distance_metric:
            raise EmbeddingModelConfigurationError(
                f"Existing EmbeddingModel {model.id} has distance_metric={model.distance_metric}, "
                f"incompatible with required {distance_metric}."
            )

    async def get_or_create_embedding_model(
        self,
        config: EmbeddingConfig,
    ) -> EmbeddingModel:
        """Resolve an existing compatible EmbeddingModel or idempotently create one.

        Args:
            config: EmbeddingConfig specifying provider, model_name, dimensions, etc.

        Returns:
            Active EmbeddingModel entity from database.

        Raises:
            TypeError: If config is not an instance of EmbeddingConfig.
            EmbeddingModelConfigurationError: If config values violate schema constraints
                (e.g. dimensions != 384) or conflict with existing model record.
        """
        if not isinstance(config, EmbeddingConfig):
            raise TypeError(
                f"config must be an instance of EmbeddingConfig, got {type(config).__name__}"
            )

        # Validate schema-level invariants enforced by DB check constraints
        if config.dimensions != DEFAULT_EMBEDDING_DIMENSIONS:
            raise EmbeddingModelConfigurationError(
                f"Database schema strictly requires {DEFAULT_EMBEDDING_DIMENSIONS} dimensions, "
                f"got {config.dimensions}."
            )

        distance_metric = DEFAULT_DISTANCE_METRIC

        # 1. Query for existing model
        stmt = select(EmbeddingModel).where(
            EmbeddingModel.provider == config.provider,
            EmbeddingModel.model_name == config.model_name,
        )
        if config.model_revision is None:
            stmt = stmt.where(EmbeddingModel.model_version.is_(None))
        else:
            stmt = stmt.where(EmbeddingModel.model_version == config.model_revision)

        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            self._validate_embedding_model_compatibility(existing, config, distance_metric)
            return existing

        # 2. Insert safely using ON CONFLICT DO NOTHING to handle concurrent creation
        insert_stmt = (
            insert(EmbeddingModel)
            .values(
                provider=config.provider,
                model_name=config.model_name,
                model_version=config.model_revision,
                dimensions=config.dimensions,
                normalize_embeddings=config.normalize_embeddings,
                distance_metric=distance_metric,
                configuration={
                    "device": config.device,
                    "batch_size": config.batch_size,
                },
                is_active=True,
            )
            .on_conflict_do_nothing()
        )

        await self.session.execute(insert_stmt)
        await self.session.flush()

        # Re-query to retrieve the existing or newly inserted record
        result = await self.session.execute(stmt)
        model = result.scalar_one()
        self._validate_embedding_model_compatibility(model, config, distance_metric)
        return model

    async def bulk_create_chunks(
        self,
        paper_id: uuid.UUID,
        ingestion_job_id: uuid.UUID,
        chunks: list[ResearchChunk],
        chunking_version: str | None = None,
    ) -> list[PaperChunk]:
        """Bulk persist a list of ResearchChunk objects into the paper_chunks table.

        Always sets is_current=False for staged records.

        Args:
            paper_id: Parent paper UUID.
            ingestion_job_id: Originating ingestion job UUID.
            chunks: List of ResearchChunk objects from the chunker.
            chunking_version: Strategy / algorithm version string.

        Returns:
            Persisted PaperChunk records ordered deterministically by chunk_index.

        Raises:
            TypeError: If paper_id or ingestion_job_id are not UUIDs, or chunks is not a list.
            ChunkPersistenceError: If validation fails (e.g. duplicate indexes, blank content).
        """
        if not isinstance(paper_id, uuid.UUID):
            raise TypeError(f"paper_id must be a uuid.UUID, got {type(paper_id).__name__}")
        if not isinstance(ingestion_job_id, uuid.UUID):
            raise TypeError(f"ingestion_job_id must be a uuid.UUID, got {type(ingestion_job_id).__name__}")
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")

        if not chunks:
            return []

        seen_indexes: set[int] = set()
        paper_chunks: list[PaperChunk] = []

        for i, c in enumerate(chunks):
            if not isinstance(c, ResearchChunk):
                raise ChunkPersistenceError(
                    f"Element at index {i} is not a ResearchChunk instance, got {type(c).__name__}"
                )

            # Validate index: nonnegative integer, reject bool
            if isinstance(c.chunk_index, bool) or not isinstance(c.chunk_index, int) or c.chunk_index < 0:
                raise ChunkPersistenceError(
                    f"Chunk at index {i} has invalid chunk_index: {c.chunk_index!r}"
                )
            if c.chunk_index in seen_indexes:
                raise ChunkPersistenceError(
                    f"Duplicate chunk_index {c.chunk_index} found in input chunks at element {i}."
                )
            seen_indexes.add(c.chunk_index)

            # Validate content: nonblank string
            if not isinstance(c.content, str) or not c.content or not c.content.strip():
                raise ChunkPersistenceError(
                    f"Chunk with index {c.chunk_index} has empty or blank content."
                )

            # Validate token_count: None or nonnegative integer, reject bool
            if c.token_count is not None:
                if isinstance(c.token_count, bool) or not isinstance(c.token_count, int) or c.token_count < 0:
                    raise ChunkPersistenceError(
                        f"Chunk with index {c.chunk_index} has invalid token_count: {c.token_count!r}"
                    )

            # Validate char_count: None or nonnegative integer, reject bool
            if c.char_count is not None:
                if isinstance(c.char_count, bool) or not isinstance(c.char_count, int) or c.char_count < 0:
                    raise ChunkPersistenceError(
                        f"Chunk with index {c.chunk_index} has invalid char_count: {c.char_count!r}"
                    )

            # Validate page_start: None or positive integer, reject bool
            if c.page_start is not None:
                if isinstance(c.page_start, bool) or not isinstance(c.page_start, int) or c.page_start <= 0:
                    raise ChunkPersistenceError(
                        f"Chunk with index {c.chunk_index} has invalid page_start: {c.page_start!r}"
                    )

            # Validate page_end: None or positive integer, reject bool
            if c.page_end is not None:
                if isinstance(c.page_end, bool) or not isinstance(c.page_end, int) or c.page_end <= 0:
                    raise ChunkPersistenceError(
                        f"Chunk with index {c.chunk_index} has invalid page_end: {c.page_end!r}"
                    )

            # Validate page range ordering
            if (
                c.page_start is not None
                and c.page_end is not None
                and c.page_end < c.page_start
            ):
                raise ChunkPersistenceError(
                    f"Chunk with index {c.chunk_index} has page_end ({c.page_end}) < page_start ({c.page_start})"
                )

            # Validate metadata: must be a dictionary
            if c.metadata is not None and not isinstance(c.metadata, dict):
                raise ChunkPersistenceError(
                    f"Chunk with index {c.chunk_index} has invalid metadata type: {type(c.metadata).__name__}, expected dict."
                )

            # Validate content_hash: valid string according to ResearchChunk contract
            if not isinstance(c.content_hash, str) or not c.content_hash.strip():
                raise ChunkPersistenceError(
                    f"Chunk with index {c.chunk_index} has invalid content_hash: {c.content_hash!r}"
                )

            pc = PaperChunk(
                id=uuid.uuid4(),
                paper_id=paper_id,
                ingestion_job_id=ingestion_job_id,
                chunk_index=c.chunk_index,
                content=c.content,
                content_hash=c.content_hash,
                token_count=c.token_count,
                char_count=c.char_count,
                page_start=c.page_start,
                page_end=c.page_end,
                section_title=c.section_title,
                chunking_version=chunking_version,
                is_current=False,
                metadata_=c.metadata if c.metadata is not None else {},
            )
            paper_chunks.append(pc)

        self.session.add_all(paper_chunks)
        await self.session.flush()

        # Deterministic ordering by chunk_index
        paper_chunks.sort(key=lambda x: x.chunk_index)
        return paper_chunks

    async def bulk_create_embeddings(
        self,
        chunks: list[PaperChunk],
        embedding_result: EmbeddingResult,
        embedding_model_id: uuid.UUID,
    ) -> list[ChunkEmbedding]:
        """Persist generated embeddings for already-persisted PaperChunk records.

        Validates association strictly by matching chunk_index and content_hash.

        Args:
            chunks: List of persisted PaperChunk entities (with generated IDs).
            embedding_result: Result object from ResearchEmbeddingGenerator.
            embedding_model_id: Target EmbeddingModel UUID in the database.

        Returns:
            Persisted ChunkEmbedding records ordered deterministically by chunk_index.

        Raises:
            TypeError: If arguments are of incorrect types.
            EmbeddingPersistenceError: If validation fails (counts mismatch, hash mismatch,
                invalid dimensions, non-finite values).
        """
        if not isinstance(embedding_model_id, uuid.UUID):
            raise TypeError(
                f"embedding_model_id must be a uuid.UUID, got {type(embedding_model_id).__name__}"
            )
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list, got {type(chunks).__name__}")
        if not isinstance(embedding_result, EmbeddingResult):
            raise TypeError(
                f"embedding_result must be an EmbeddingResult, got {type(embedding_result).__name__}"
            )

        # 1. Retrieve the EmbeddingModel and validate identity
        model_stmt = select(EmbeddingModel).where(EmbeddingModel.id == embedding_model_id)
        model_res = await self.session.execute(model_stmt)
        model = model_res.scalar_one_or_none()

        if model is None:
            raise EmbeddingPersistenceError(
                f"EmbeddingModel with id {embedding_model_id} not found."
            )

        if not model.is_active:
            raise EmbeddingPersistenceError(
                f"EmbeddingModel with id {embedding_model_id} is inactive (is_active=False)."
            )

        if model.provider != embedding_result.provider:
            raise EmbeddingPersistenceError(
                f"Embedding model provider mismatch: DB model has provider={model.provider!r}, "
                f"embedding_result has provider={embedding_result.provider!r}."
            )

        if model.model_name != embedding_result.model_name:
            raise EmbeddingPersistenceError(
                f"Embedding model name mismatch: DB model has model_name={model.model_name!r}, "
                f"embedding_result has model_name={embedding_result.model_name!r}."
            )

        if model.dimensions != embedding_result.dimensions:
            raise EmbeddingPersistenceError(
                f"Embedding model dimensions mismatch: DB model has dimensions={model.dimensions}, "
                f"embedding_result has dimensions={embedding_result.dimensions}."
            )

        if model.normalize_embeddings != embedding_result.normalized:
            raise EmbeddingPersistenceError(
                f"Embedding model normalization mismatch: DB model has normalize_embeddings={model.normalize_embeddings}, "
                f"embedding_result has normalized={embedding_result.normalized}."
            )

        if len(chunks) != len(embedding_result.embeddings):
            raise EmbeddingPersistenceError(
                f"Count mismatch: chunks has {len(chunks)} items, "
                f"embedding_result has {len(embedding_result.embeddings)} embeddings."
            )

        if not chunks:
            return []

        # Map chunks by chunk_index for reliable O(1) lookup
        chunks_by_index: dict[int, PaperChunk] = {}
        for c in chunks:
            if not isinstance(c, PaperChunk):
                raise EmbeddingPersistenceError(
                    f"Element in chunks is not a PaperChunk: {type(c).__name__}"
                )
            if c.id is None:
                raise EmbeddingPersistenceError(
                    f"Chunk with index {c.chunk_index} has no database ID (not flushed)."
                )
            if c.chunk_index in chunks_by_index:
                raise EmbeddingPersistenceError(
                    f"Duplicate chunk_index {c.chunk_index} in input chunks list."
                )
            chunks_by_index[c.chunk_index] = c

        seen_embedding_indices: set[int] = set()
        chunk_embeddings: list[tuple[int, ChunkEmbedding]] = []

        for i, emb in enumerate(embedding_result.embeddings):
            if not isinstance(emb, ResultChunkEmbedding):
                raise EmbeddingPersistenceError(
                    f"Embedding at index {i} is not a ChunkEmbedding dataclass: {type(emb).__name__}"
                )

            idx = emb.chunk_index
            if idx in seen_embedding_indices:
                raise EmbeddingPersistenceError(
                    f"Duplicate chunk_index {idx} in embedding results."
                )
            seen_embedding_indices.add(idx)

            target_chunk = chunks_by_index.get(idx)
            if target_chunk is None:
                raise EmbeddingPersistenceError(
                    f"Embedding with chunk_index {idx} has no corresponding PaperChunk."
                )

            # Validate content_hash match
            if target_chunk.content_hash != emb.content_hash:
                raise EmbeddingPersistenceError(
                    f"Content hash mismatch for chunk_index {idx}: "
                    f"chunk hash={target_chunk.content_hash!r}, embedding hash={emb.content_hash!r}"
                )

            # Validate vector dimensions against database requirement
            if len(emb.vector) != DEFAULT_EMBEDDING_DIMENSIONS:
                raise EmbeddingPersistenceError(
                    f"Embedding for chunk_index {idx} dimension mismatch: "
                    f"expected {DEFAULT_EMBEDDING_DIMENSIONS}, got {len(emb.vector)}"
                )

            # Strict numerical check for non-finite values
            for val_idx, val in enumerate(emb.vector):
                if isinstance(val, bool) or not isinstance(val, numbers.Real) or not math.isfinite(val):
                    raise EmbeddingPersistenceError(
                        f"Embedding for chunk_index {idx}, value {val_idx} is invalid or non-finite: {val!r}"
                    )

            db_emb = ChunkEmbedding(
                id=uuid.uuid4(),
                chunk_id=target_chunk.id,
                embedding_model_id=embedding_model_id,
                embedding=list(emb.vector),
            )
            chunk_embeddings.append((idx, db_emb))

        entities_to_add = [ce[1] for ce in chunk_embeddings]
        self.session.add_all(entities_to_add)
        await self.session.flush()

        # Sort deterministically by chunk_index
        chunk_embeddings.sort(key=lambda x: x[0])
        return [ce[1] for ce in chunk_embeddings]

    async def get_job_chunks(
        self,
        ingestion_job_id: uuid.UUID,
    ) -> list[PaperChunk]:
        """Retrieve all PaperChunk records associated with a specific ingestion job.

        Args:
            ingestion_job_id: Target ingestion job UUID.

        Returns:
            List of PaperChunk records ordered deterministically by chunk_index ASC.

        Raises:
            TypeError: If ingestion_job_id is not a uuid.UUID.
        """
        if not isinstance(ingestion_job_id, uuid.UUID):
            raise TypeError(
                f"ingestion_job_id must be a uuid.UUID, got {type(ingestion_job_id).__name__}"
            )

        stmt = (
            select(PaperChunk)
            .where(PaperChunk.ingestion_job_id == ingestion_job_id)
            .order_by(PaperChunk.chunk_index.asc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def activate_chunks_for_job(
        self,
        paper_id: uuid.UUID,
        ingestion_job_id: uuid.UUID,
        embedding_model_id: uuid.UUID,
    ) -> list[PaperChunk]:
        """Atomically activate staged chunks for an ingestion job and demote existing chunks.

        Serializes concurrent activations for the paper using `SELECT ... FOR UPDATE`
        on the parent Paper row. Verifies that staged chunks and their embeddings
        are complete before performing the atomic flip.

        Args:
            paper_id: Target paper UUID.
            ingestion_job_id: Ingestion job UUID whose chunks should become active.
            embedding_model_id: Expected embedding model UUID that must be present for all chunks.

        Returns:
            List of newly activated PaperChunk records ordered by chunk_index ASC.

        Raises:
            TypeError: If arguments are not UUID instances.
            ChunkActivationError: If job does not belong to paper, chunks are empty,
                embeddings are incomplete, or parent paper does not exist.
        """
        if not isinstance(paper_id, uuid.UUID):
            raise TypeError(f"paper_id must be a uuid.UUID, got {type(paper_id).__name__}")
        if not isinstance(ingestion_job_id, uuid.UUID):
            raise TypeError(f"ingestion_job_id must be a uuid.UUID, got {type(ingestion_job_id).__name__}")
        if not isinstance(embedding_model_id, uuid.UUID):
            raise TypeError(f"embedding_model_id must be a uuid.UUID, got {type(embedding_model_id).__name__}")

        # 1. Row-level lock on Paper to serialize activations for this paper
        paper_stmt = (
            select(Paper)
            .where(Paper.id == paper_id)
            .with_for_update()
        )
        paper_result = await self.session.execute(paper_stmt)
        paper = paper_result.scalar_one_or_none()
        if paper is None:
            raise ChunkActivationError(f"Paper with id {paper_id} not found.")

        # 2. Verify job exists and belongs to the specified paper
        job_stmt = select(IngestionJob).where(IngestionJob.id == ingestion_job_id)
        job_result = await self.session.execute(job_stmt)
        job = job_result.scalar_one_or_none()
        if job is None:
            raise ChunkActivationError(f"IngestionJob with id {ingestion_job_id} not found.")
        if job.paper_id != paper_id:
            raise ChunkActivationError(
                f"IngestionJob {ingestion_job_id} belongs to paper {job.paper_id}, not {paper_id}."
            )

        # 3. Retrieve staged chunks for this job
        chunks_stmt = (
            select(PaperChunk)
            .where(
                PaperChunk.ingestion_job_id == ingestion_job_id,
                PaperChunk.paper_id == paper_id,
            )
            .order_by(PaperChunk.chunk_index.asc())
        )
        chunks_res = await self.session.execute(chunks_stmt)
        staged_chunks = list(chunks_res.scalars().all())

        if not staged_chunks:
            raise ChunkActivationError(
                f"No staged chunks found for job {ingestion_job_id} to activate."
            )

        # 4. Verify embedding completeness: every staged chunk must have exactly one embedding
        # for embedding_model_id
        chunk_ids = [c.id for c in staged_chunks]
        emb_count_stmt = (
            select(func.count(ChunkEmbedding.id))
            .where(
                ChunkEmbedding.chunk_id.in_(chunk_ids),
                ChunkEmbedding.embedding_model_id == embedding_model_id,
            )
        )
        emb_count_res = await self.session.execute(emb_count_stmt)
        total_embeddings = emb_count_res.scalar_one()

        if total_embeddings != len(staged_chunks):
            raise ChunkActivationError(
                f"Embedding set is incomplete for job {ingestion_job_id}: "
                f"expected {len(staged_chunks)} embeddings for model {embedding_model_id}, "
                f"found {total_embeddings}."
            )

        # 5. Atomic flip:
        # Step A: Demote existing active chunks for this paper
        demote_stmt = (
            update(PaperChunk)
            .where(
                PaperChunk.paper_id == paper_id,
                PaperChunk.is_current.is_(True),
            )
            .values(is_current=False)
        )
        await self.session.execute(demote_stmt)

        # Step B: Promote staged chunks for this job
        promote_stmt = (
            update(PaperChunk)
            .where(
                PaperChunk.ingestion_job_id == ingestion_job_id,
                PaperChunk.paper_id == paper_id,
            )
            .values(is_current=True)
        )
        await self.session.execute(promote_stmt)

        await self.session.flush()

        # Refresh objects / re-query to return activated chunks
        activated_res = await self.session.execute(chunks_stmt)
        activated_chunks = list(activated_res.scalars().all())
        return activated_chunks

    async def get_latest_completed_job(
        self,
        paper_file_id: uuid.UUID,
        pipeline_version: str | None = None,
    ) -> IngestionJob | None:
        """Retrieve the most recently completed IngestionJob for a given paper file.

        Args:
            paper_file_id: Target PaperFile UUID.
            pipeline_version: Optional pipeline version filter.

        Returns:
            The latest completed IngestionJob instance if found, or None.

        Raises:
            TypeError: If paper_file_id is not a uuid.UUID.
        """
        if not isinstance(paper_file_id, uuid.UUID):
            raise TypeError(f"paper_file_id must be a uuid.UUID, got {type(paper_file_id).__name__}")

        stmt = (
            select(IngestionJob)
            .where(
                IngestionJob.paper_file_id == paper_file_id,
                IngestionJob.status == "completed",
            )
        )
        if pipeline_version is not None:
            stmt = stmt.where(IngestionJob.pipeline_version == pipeline_version)

        stmt = stmt.order_by(
            IngestionJob.completed_at.desc().nulls_last(),
            IngestionJob.created_at.desc(),
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_active_job_for_paper(
        self,
        paper_id: uuid.UUID,
    ) -> IngestionJob | None:
        """Retrieve an in-progress active IngestionJob for a paper if one exists.

        Args:
            paper_id: Target Paper UUID.

        Returns:
            An active IngestionJob instance in pending or processing state, or None.

        Raises:
            TypeError: If paper_id is not a uuid.UUID.
        """
        if not isinstance(paper_id, uuid.UUID):
            raise TypeError(f"paper_id must be a uuid.UUID, got {type(paper_id).__name__}")

        stmt = (
            select(IngestionJob)
            .where(
                IngestionJob.paper_id == paper_id,
                IngestionJob.status.in_(ACTIVE_JOB_STATUSES),
            )
            .order_by(IngestionJob.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def lock_paper(
        self,
        paper_id: uuid.UUID,
    ) -> Paper | None:
        """Acquire a row-level lock (FOR UPDATE) on a Paper to serialize concurrent job operations.

        Args:
            paper_id: Target Paper UUID.

        Returns:
            The locked Paper instance if found, or None.

        Raises:
            TypeError: If paper_id is not a uuid.UUID.
        """
        if not isinstance(paper_id, uuid.UUID):
            raise TypeError(f"paper_id must be a uuid.UUID, got {type(paper_id).__name__}")

        stmt = (
            select(Paper)
            .where(Paper.id == paper_id)
            .with_for_update()
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

