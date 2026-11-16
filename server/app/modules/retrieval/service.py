"""Retrieval service coordinating database fetching, ID mapping, and lexical search."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING, Optional
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import logging_settings
from app.core.logging_config import log_event
from app.modules.ingestion.models import PaperChunk
from app.modules.retrieval.id_mapper import ChunkIDMapper
from app.modules.retrieval.lexical.engine import LexicalRetrievalEngine
from app.modules.retrieval.repository import RetrievalRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """Immutable representation of a retrieved paper chunk with similarity scoring and paper metadata."""

    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    chunk_index: int
    content: str
    score: float
    paper_title: str
    publication_year: int | None


class RetrievalService:
    """Orchestrates in-memory lexical retrieval against active paper chunks fetched from PostgreSQL."""

    def __init__(
        self,
        engine: LexicalRetrievalEngine | None = None,
    ) -> None:
        """Initialize the retrieval service.

        The service manages its own in-memory state and does not retain a long-lived
        database session. Database repositories are supplied on each refresh operation.

        Args:
            engine: Optional LexicalRetrievalEngine instance. If None, a default
                engine is instantiated.

        Raises:
            TypeError: If engine is not a LexicalRetrievalEngine.
        """
        if engine is not None and not isinstance(engine, LexicalRetrievalEngine):
            raise TypeError(
                f"engine must be an instance of LexicalRetrievalEngine, got {type(engine).__name__}"
            )

        self.engine: LexicalRetrievalEngine = (
            engine if engine is not None else LexicalRetrievalEngine()
        )
        self.id_mapper: ChunkIDMapper = ChunkIDMapper()
        self._chunk_metadata: dict[int, PaperChunk] = {}
        self._is_initialized: bool = False
        self._refresh_lock: asyncio.Lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        """Return whether the index has been initialized via refresh_index()."""
        return self._is_initialized

    async def refresh_index(
        self,
        repository: RetrievalRepository,
    ) -> int:
        """Fetch all currently active paper chunks from PostgreSQL and rebuild the lexical index.

        Uses the supplied repository and session for the duration of the refresh,
        guaranteeing that no database session is held open across the service's lifetime.
        Concurrent refreshes on the same service instance are serialized via an internal lock.

        Args:
            repository: Active RetrievalRepository instance bound to a current database session.

        Returns:
            The number of successfully indexed chunks.

        Raises:
            TypeError: If repository is not an instance of RetrievalRepository.
        """
        if not isinstance(repository, RetrievalRepository):
            raise TypeError(
                f"repository must be an instance of RetrievalRepository, got {type(repository).__name__}"
            )

        async with self._refresh_lock:
            refresh_start_t = time.perf_counter()
            log_event(logger, logging.INFO, "retrieval_index_refresh_started")

            try:
                chunks = await repository.get_current_chunks()

                temp_mapper = ChunkIDMapper()
                documents: dict[int, str] = {}
                temp_metadata: dict[int, PaperChunk] = {}

                for chunk in chunks:
                    int_id = temp_mapper.get_or_create(chunk.id)
                    documents[int_id] = chunk.content
                    temp_metadata[int_id] = chunk

                # Build index on the engine; raises on invalid data or preprocessing failure
                self.engine.build_index(documents)

                # Atomic commit of mappings only after engine indexing succeeds
                self.id_mapper = temp_mapper
                self._chunk_metadata = temp_metadata
                self._is_initialized = True

                refresh_duration_ms = (time.perf_counter() - refresh_start_t) * 1000.0
                log_event(
                    logger,
                    logging.INFO,
                    "retrieval_index_refresh_completed",
                    indexed_chunks_count=len(documents),
                    duration_ms=round(refresh_duration_ms, 2),
                )
                return len(documents)
            except Exception as exc:
                refresh_duration_ms = (time.perf_counter() - refresh_start_t) * 1000.0
                log_event(
                    logger,
                    logging.ERROR,
                    "retrieval_index_refresh_failed",
                    error=type(exc).__name__,
                    duration_ms=round(refresh_duration_ms, 2),
                )
                raise

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Perform synchronous in-memory lexical search across the indexed paper chunks.

        Args:
            query: Raw search query text.
            top_k: Maximum number of ranked results to return (default: 5).

        Returns:
            List of RetrievedChunk objects ordered by descending similarity score.

        Raises:
            RuntimeError: If the index has not been initialized via refresh_index(),
                or if an engine integer ID cannot be resolved back to its UUID or metadata.
            TypeError: If query is not a string, or top_k is a boolean or not an int.
            ValueError: If top_k is not positive (> 0).
        """
        if not self._is_initialized:
            raise RuntimeError(
                "RetrievalService index is not initialized. Run 'await refresh_index()' first."
            )

        ranked_results = self.engine.search(query=query, top_k=top_k)
        if not ranked_results:
            log_event(
                logger,
                logging.INFO,
                "retrieval_ranking_completed",
                ranked_candidates_count=0,
                returned_count=0,
                requested_top_k=top_k,
            )
            return []

        retrieved_chunks: list[RetrievedChunk] = []
        for int_id, score in ranked_results:
            chunk_uuid = self.id_mapper.get_uuid(int_id)
            chunk = self._chunk_metadata.get(int_id)

            if chunk_uuid is None or chunk is None:
                raise RuntimeError(
                    f"Inconsistent retrieval state: engine document ID {int_id} could not "
                    f"be resolved to registered chunk metadata or UUID."
                )

            paper_title = chunk.paper.title if chunk.paper is not None else ""
            pub_year = (
                chunk.paper.publication_year if chunk.paper is not None else None
            )

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_uuid,
                    paper_id=chunk.paper_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=float(score),
                    paper_title=paper_title,
                    publication_year=pub_year,
                )
            )

        log_event(
            logger,
            logging.INFO,
            "retrieval_ranking_completed",
            ranked_candidates_count=len(ranked_results),
            returned_count=len(retrieved_chunks),
            requested_top_k=top_k,
        )

        # When NLP_TRACE_ENABLED is True, log detailed chunk ranking information
        if getattr(logging_settings, "nlp_trace_enabled", False):
            for rank, r_chunk in enumerate(retrieved_chunks, start=1):
                int_id = self.id_mapper.get_integer_id(r_chunk.chunk_id)
                meta = self._chunk_metadata.get(int_id) if int_id is not None else None
                page_start = getattr(meta, "page_start", None)
                page_end = getattr(meta, "page_end", None)
                page_range = f"{page_start}-{page_end}" if page_start is not None else "N/A"
                log_event(
                    logger,
                    logging.DEBUG,
                    "retrieval_top_k_chunk_selected",
                    rank=rank,
                    chunk_id=str(r_chunk.chunk_id),
                    paper_id=str(r_chunk.paper_id),
                    paper_title=r_chunk.paper_title,
                    page_range=page_range,
                    similarity_score=round(r_chunk.score, 6),
                )

        return retrieved_chunks


async def refresh_retrieval_index(
    service: RetrievalService,
    session: AsyncSession,
) -> int:
    """Convenience helper to refresh the application-scoped retrieval service

    using an active short-lived AsyncSession and fresh RetrievalRepository.

    Args:
        service: Application-scoped RetrievalService instance.
        session: Active SQLAlchemy AsyncSession.

    Returns:
        The number of successfully indexed chunks.
    """
    repository = RetrievalRepository(session)
    return await service.refresh_index(repository)
