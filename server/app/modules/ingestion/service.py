"""Service layer for coordinating end-to-end research paper ingestion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, Sequence
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import logging_settings
from app.core.exceptions import ActiveJobConflictError, AppError
from app.core.logging_config import correlation_context, log_event
from app.database.session import AsyncSessionLocal
from app.modules.ingestion.chunker import ResearchChunkingEngine
from app.modules.ingestion.embedder import ResearchEmbeddingGenerator
from app.modules.ingestion.models import ACTIVE_JOB_STATUSES, TERMINAL_JOB_STATUSES, IngestionJob
from app.modules.ingestion.parser import PDFParser
from app.modules.ingestion.preprocessor import ResearchTextPreprocessor
from app.modules.ingestion.repository import IngestionRepository, IngestionRepositoryError
from app.modules.ingestion.section_detector import ResearchSectionDetector
from app.modules.ingestion.storage import CloudinaryStorage, CloudinaryUploadResult
from app.modules.papers.models import Paper, PaperFile
from app.modules.papers.repository import PaperRepository
from app.modules.papers.service import PaperService
from app.modules.retrieval.repository import RetrievalRepository
from app.modules.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


def check_active_jobs_for_deletion(
    target_type: str,
    target_id: uuid.UUID,
    active_jobs: Sequence[Any],
) -> None:
    """Ensure no active ingestion jobs are in progress before target deletion (in-memory guard)."""
    in_progress = [j for j in active_jobs if getattr(j, "status", None) in ACTIVE_JOB_STATUSES]
    if in_progress:
        raise ActiveJobConflictError(
            f"Cannot delete {target_type} {target_id}: {len(in_progress)} active job(s) in progress"
        )


@dataclass(frozen=True)
class IngestionResult:
    """Immutable result representation of an executed ingestion operation.

    Contains stable identifiers and serializable metadata so callers do not
    rely on lazy-loaded ORM relationships after database sessions close.

    Attributes:
        paper_id: UUID of parent research paper.
        paper_file_id: UUID of paper file associated with this ingestion.
        ingestion_job_id: UUID of the IngestionJob (either newly executed or existing reused job).
        chunks_created: Number of chunks associated with this job. When job_reused is False,
            this represents the count of newly persisted PaperChunk records created during
            this invocation. When job_reused is True, this represents the count of existing
            chunks from the previously completed, compatible job.
        embeddings_created: Number of embeddings associated with this job. When job_reused is False,
            this represents the count of newly persisted ChunkEmbedding records generated and
            persisted during this invocation. When job_reused is True, this represents the count
            of existing embeddings associated with the reused job.
        paper_reused: True if the Paper record already existed and was reused.
        file_reused: True if the PaperFile record already existed and was reused.
        job_reused: True if an existing completed IngestionJob with matching pipeline fingerprint
            and active chunks was reused without re-chunking or re-embedding.
        status: Final status of the ingestion job ("completed").
    """

    paper_id: uuid.UUID
    paper_file_id: uuid.UUID
    ingestion_job_id: uuid.UUID
    chunks_created: int
    embeddings_created: int
    paper_reused: bool
    file_reused: bool
    job_reused: bool
    status: str
    title: str | None = None


class IngestionService:
    """Coordinates research paper parsing, processing, embedding, and atomic persistence.

    Owns transaction boundaries and executes long-running CPU and storage operations
    outside database transactions and off the main event loop.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        storage: CloudinaryStorage | None = None,
        parser: PDFParser | None = None,
        preprocessor: ResearchTextPreprocessor | None = None,
        section_detector: ResearchSectionDetector | None = None,
        chunker: ResearchChunkingEngine | None = None,
        embedder: ResearchEmbeddingGenerator | None = None,
        retrieval_service: RetrievalService | None = None,
        pipeline_version: str = "pipeline-v1",
        max_concurrent_embeddings: int = 1,
    ) -> None:
        """Initialize IngestionService with dependencies and configuration.

        Args:
            session_factory: SQLAlchemy async_sessionmaker for short-lived sessions.
            storage: Cloudinary storage manager.
            parser: PDF parsing and academic metadata extractor.
            preprocessor: Text cleaning and paragraph reconstruction engine.
            section_detector: Academic section detection engine.
            chunker: Section-aware chunking engine.
            embedder: SentenceTransformer embedding generator.
            retrieval_service: Optional lexical retrieval service to refresh upon completion.
            pipeline_version: Identifier for the current ingestion pipeline version.
            max_concurrent_embeddings: Max concurrent embedding inferences allowed (positive integer).
        """
        if session_factory is not None and not isinstance(session_factory, async_sessionmaker):
            raise TypeError(
                f"session_factory must be an async_sessionmaker, got {type(session_factory).__name__}"
            )
        if (
            isinstance(max_concurrent_embeddings, bool)
            or not isinstance(max_concurrent_embeddings, int)
            or max_concurrent_embeddings <= 0
        ):
            raise ValueError(
                f"max_concurrent_embeddings must be a positive integer, got {max_concurrent_embeddings!r}"
            )

        self.session_factory: async_sessionmaker[AsyncSession] = (
            session_factory if session_factory is not None else AsyncSessionLocal
        )
        self.storage: CloudinaryStorage = (
            storage if storage is not None else CloudinaryStorage()
        )
        self.parser: PDFParser = (
            parser if parser is not None else PDFParser()
        )
        self.preprocessor: ResearchTextPreprocessor = (
            preprocessor if preprocessor is not None else ResearchTextPreprocessor()
        )
        self.section_detector: ResearchSectionDetector = (
            section_detector if section_detector is not None else ResearchSectionDetector()
        )
        self.chunker: ResearchChunkingEngine = (
            chunker if chunker is not None else ResearchChunkingEngine()
        )
        self.embedder: ResearchEmbeddingGenerator = (
            embedder if embedder is not None else ResearchEmbeddingGenerator()
        )
        self.retrieval_service: RetrievalService | None = retrieval_service
        self.pipeline_version: str = pipeline_version
        self._embedding_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            max_concurrent_embeddings
        )

    def _compute_pipeline_fingerprint(self) -> str:
        """Compute a deterministic hash fingerprint representing pipeline configuration."""
        if hasattr(self.chunker, "tokenizer_identity"):
            token_counter_name = self.chunker.tokenizer_identity
        else:
            token_counter = getattr(self.chunker, "token_counter", None)
            if hasattr(token_counter, "identity"):
                token_counter_name = str(token_counter.identity)
            elif token_counter is not None:
                token_counter_name = type(token_counter).__name__
            else:
                token_counter_name = "WhitespaceTokenCounter"
        components = [
            f"chunking_version={self.chunker.config.chunking_version}",
            f"target_chunk_size={self.chunker.config.target_chunk_size}",
            f"max_chunk_size={self.chunker.config.max_chunk_size}",
            f"min_chunk_size={self.chunker.config.min_chunk_size}",
            f"overlap={self.chunker.config.overlap}",
            f"token_counter={token_counter_name}",
            f"embedding_provider={self.embedder.config.provider}",
            f"embedding_model={self.embedder.config.model_name}",
            f"embedding_revision={self.embedder.config.model_revision or ''}",
            f"embedding_dimensions={self.embedder.config.dimensions}",
            f"normalize_embeddings={self.embedder.config.normalize_embeddings}",
        ]
        raw = "|".join(components)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _update_job_stage(self, job_id: uuid.UUID, stage: str) -> None:
        """Update job current_stage in a dedicated short database transaction."""
        async with self.session_factory() as session:
            async with session.begin():
                repo = IngestionRepository(session)
                await repo.update_job_status(job_id, status="processing", current_stage=stage)

    async def _record_job_failure(self, job_id: uuid.UUID, error_message: str) -> None:
        """Record failed job status in a dedicated fresh transaction without masking original errors."""
        try:
            async with self.session_factory() as session:
                async with session.begin():
                    repo = IngestionRepository(session)
                    now = datetime.now(timezone.utc)
                    await repo.update_job_status(
                        job_id,
                        status="failed",
                        error_message=error_message,
                        completed_at=now,
                    )
        except Exception as secondary_exc:
            logger.error(
                "Secondary failure: failed to mark IngestionJob %s as failed: %s",
                job_id,
                secondary_exc,
                exc_info=True,
            )

    async def _cleanup_orphan_upload(self, public_id: str, checksum: str) -> None:
        """Best-effort cleanup of an uploaded Cloudinary asset if its DB registration did not commit.

        Verifies via a fresh database session that no PaperFile record exists for this storage_key.
        If verification fails or indicates the record exists, the deletion is skipped.
        Any cleanup failure is logged without raising an exception.
        """
        async def _do_cleanup() -> None:
            try:
                async with self.session_factory() as verify_session:
                    stmt = select(PaperFile).where(PaperFile.storage_key == public_id)
                    res = await verify_session.execute(stmt)
                    existing_file = res.scalar_one_or_none()
                    if existing_file is not None:
                        logger.warning(
                            "Skipping orphan cleanup for Cloudinary asset %s: "
                            "associated PaperFile exists in DB.",
                            public_id,
                        )
                        return
            except Exception as verify_exc:
                logger.warning(
                    "Skipping orphan cleanup for Cloudinary asset %s: "
                    "failed to verify DB state: %s",
                    public_id,
                    verify_exc,
                    exc_info=True,
                )
                return

            try:
                await asyncio.to_thread(self.storage.delete_file, public_id)
                logger.info("Successfully cleaned up orphan Cloudinary asset %s.", public_id)
            except Exception as delete_exc:
                logger.warning(
                    "Failed to delete orphan Cloudinary asset %s: %s",
                    public_id,
                    delete_exc,
                    exc_info=True,
                )

        cleanup_task = asyncio.create_task(_do_cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await asyncio.wait([cleanup_task])

    async def _record_job_cancellation(
        self, job_id: uuid.UUID, stage: str, error_message: str
    ) -> None:
        """Record cancelled job status in a dedicated fresh transaction protected from task cancellation.

        Atomically transitions the job to 'cancelled' only if it is currently active via
        `repo.cancel_job_if_active()`. If the job has already reached a terminal status
        ('completed', 'failed', or 'cancelled'), such as when cancellation arrives after the
        final persistence transaction committed, the terminal job is preserved without modification.
        """
        async def _do_cancel() -> None:
            try:
                async with self.session_factory() as session:
                    async with session.begin():
                        repo = IngestionRepository(session)
                        now = datetime.now(timezone.utc)
                        res_job = await repo.cancel_job_if_active(
                            job_id=job_id,
                            current_stage=stage,
                            error_message=error_message,
                            completed_at=now,
                        )
                        if (
                            res_job is not None
                            and res_job.status in TERMINAL_JOB_STATUSES
                            and res_job.status != "cancelled"
                        ):
                            logger.info(
                                "Job %s already in terminal status '%s'; cancellation update skipped.",
                                job_id,
                                res_job.status,
                            )
            except Exception as exc:
                logger.error(
                    "Secondary failure: failed to mark IngestionJob %s as cancelled: %s",
                    job_id,
                    exc,
                    exc_info=True,
                )

        cancel_task = asyncio.create_task(_do_cancel())
        try:
            await asyncio.shield(cancel_task)
        except asyncio.CancelledError:
            await asyncio.wait([cancel_task])

    async def ingest_pdf(
        self,
        pdf_bytes: bytes,
        original_filename: str,
        *,
        title: str | None = None,
        doi: str | None = None,
        force_reprocess: bool = False,
    ) -> IngestionResult:
        """Execute the end-to-end research paper ingestion and indexing workflow.

        Args:
            pdf_bytes: Raw bytes of the PDF file.
            original_filename: Name of the uploaded file.
            title: Optional paper title (takes precedence over extracted title).
            doi: Optional DOI string (takes precedence over extracted DOI).
            force_reprocess: If True, forces re-chunking and re-embedding even if a
                compatible completed job already exists.

        Returns:
            IngestionResult containing identifiers and processing summary.

        Raises:
            InvalidPDFError: If content fails PDF signature validation.
            PaperValidationError: If title is missing when creating a new paper.
            ActiveJobConflictError: If another job is currently active for this paper.
            AppError / Exception: If parsing, storage, processing, or persistence fails.
        """
        if not isinstance(original_filename, str) or not original_filename.strip():
            raise ValueError("original_filename must be a non-empty string.")

        self.storage.validate_pdf(pdf_bytes)
        checksum = hashlib.sha256(pdf_bytes).hexdigest()
        file_size = len(pdf_bytes)

        # Parse PDF outside the event loop
        parse_start_t = time.perf_counter()
        parsed_paper = await asyncio.to_thread(self.parser.parse, pdf_bytes)
        parse_duration_ms = (time.perf_counter() - parse_start_t) * 1000.0

        extracted_chars = sum(len(p.text) for p in parsed_paper.pages)
        empty_pages = getattr(parsed_paper.diagnostics, "pages_without_text", 0)
        total_pages = getattr(parsed_paper.diagnostics, "total_pages", len(parsed_paper.pages))

        log_event(
            logger,
            logging.INFO,
            "ingestion_pdf_parsed",
            original_filename=original_filename,
            file_size=file_size,
            checksum=checksum,
            validation_result="valid",
            total_pages=total_pages,
            extracted_characters=extracted_chars,
            pages_without_text=empty_pages,
            duration_ms=round(parse_duration_ms, 2),
        )

        # Extract metadata candidates with precedence rules
        extracted_meta = parsed_paper.academic_metadata
        pdf_meta = parsed_paper.pdf_metadata

        effective_title = (
            title.strip()
            if (isinstance(title, str) and title.strip())
            else (extracted_meta.title or pdf_meta.title)
        )
        effective_doi = (
            doi.strip()
            if (isinstance(doi, str) and doi.strip())
            else extracted_meta.doi
        )
        effective_abstract = extracted_meta.abstract
        effective_venue = extracted_meta.venue
        effective_publisher = extracted_meta.publisher
        effective_year = extracted_meta.publication_year

        effective_authors = None
        if extracted_meta.authors:
            effective_authors = [
                {"name": a.name, "affiliation": a.affiliation}
                if hasattr(a, "name")
                else a
                for a in extracted_meta.authors
            ]

        effective_pub_date = None
        if extracted_meta.publication_date:
            try:
                effective_pub_date = date.fromisoformat(extracted_meta.publication_date)
            except (ValueError, TypeError):
                effective_pub_date = None

        async with self.session_factory() as session:
            async with session.begin():
                paper_repo = PaperRepository(session)
                paper_service = PaperService(paper_repo)

                resolution = await paper_service.resolve_or_create_paper(
                    title=effective_title,
                    doi=effective_doi,
                    checksum_sha256=checksum,
                    abstract=effective_abstract,
                    publication_date=effective_pub_date,
                    publication_year=effective_year,
                    venue=effective_venue,
                    publisher=effective_publisher,
                    authors=effective_authors,
                )

        paper_id: uuid.UUID = resolution.paper.id
        matching_file: PaperFile | None = resolution.matching_file
        paper_reused: bool = resolution.paper_reused
        file_reused: bool = resolution.file_reused

        new_upload_result: CloudinaryUploadResult | None = None
        new_upload_created: bool = False
        registration_committed: bool = False

        if not file_reused or matching_file is None:
            new_upload_result = await asyncio.to_thread(self.storage.upload_file, pdf_bytes)
            new_upload_created = True

        pipeline_fingerprint = self._compute_pipeline_fingerprint()
        job_id: uuid.UUID | None = None

        try:
            async with self.session_factory() as session:
                async with session.begin():
                    paper_repo = PaperRepository(session)
                    paper_service = PaperService(paper_repo)
                    ingestion_repo = IngestionRepository(session)

                    # Acquire row-level lock on the parent Paper to serialize concurrent job operations
                    locked_paper = await ingestion_repo.lock_paper(paper_id)
                    if locked_paper is None:
                        raise IngestionRepositoryError(
                            f"Paper with id {paper_id} not found during job creation."
                        )

                    # Active-job conflict check
                    active_job = await ingestion_repo.get_active_job_for_paper(paper_id)
                    if active_job is not None:
                        raise ActiveJobConflictError(
                            f"Cannot start ingestion for paper {paper_id}: active job {active_job.id} "
                            f"is already in progress (status: {active_job.status}, stage: {active_job.current_stage})."
                        )

                    # Register newly uploaded file or reuse existing
                    if new_upload_result is not None:
                        registered_file = await paper_service.register_paper_file(
                            paper_id=paper_id,
                            upload_result=new_upload_result,
                            original_filename=original_filename,
                            checksum_sha256=checksum,
                            status="available",
                        )
                        paper_file_id = registered_file.id
                    else:
                        assert matching_file is not None
                        paper_file_id = matching_file.id

                    # Stage D — Idempotency check for reused files
                    if not force_reprocess and new_upload_result is None:
                        completed_job = await ingestion_repo.get_latest_completed_job(
                            paper_file_id=paper_file_id,
                            pipeline_version=self.pipeline_version,
                        )
                        if completed_job is not None:
                            job_fingerprint = completed_job.metadata_.get("pipeline_fingerprint")
                            if job_fingerprint == pipeline_fingerprint:
                                existing_chunks = await ingestion_repo.get_job_chunks(completed_job.id)
                                if existing_chunks and all(c.is_current for c in existing_chunks):
                                    # Completed job matches file, pipeline fingerprint, and active chunks
                                    return IngestionResult(
                                        paper_id=paper_id,
                                        paper_file_id=paper_file_id,
                                        ingestion_job_id=completed_job.id,
                                        chunks_created=len(existing_chunks),
                                        embeddings_created=len(existing_chunks),
                                        paper_reused=paper_reused,
                                        file_reused=file_reused,
                                        job_reused=True,
                                        status="completed",
                                        title=resolution.paper.title,
                                    )

                    # Create new IngestionJob
                    now_utc = datetime.now(timezone.utc)
                    job = await ingestion_repo.create_job(
                        paper_id=paper_id,
                        paper_file_id=paper_file_id,
                        status="processing",
                        current_stage="preprocessing",
                        pipeline_version=self.pipeline_version,
                        started_at=now_utc,
                        metadata_={
                            "pipeline_fingerprint": pipeline_fingerprint,
                            "original_filename": original_filename,
                            "checksum_sha256": checksum,
                            "chunking_config": {
                                "target_chunk_size": self.chunker.config.target_chunk_size,
                                "max_chunk_size": self.chunker.config.max_chunk_size,
                                "min_chunk_size": self.chunker.config.min_chunk_size,
                                "overlap": self.chunker.config.overlap,
                                "token_counter": getattr(
                                    self.chunker,
                                    "tokenizer_identity",
                                    type(self.chunker.token_counter).__name__
                                    if getattr(self.chunker, "token_counter", None) is not None
                                    else "WhitespaceTokenCounter",
                                ),
                                "chunking_version": self.chunker.config.chunking_version,
                            },
                            "embedding_config": {
                                "provider": self.embedder.config.provider,
                                "model_name": self.embedder.config.model_name,
                                "dimensions": self.embedder.config.dimensions,
                                "normalize_embeddings": self.embedder.config.normalize_embeddings,
                                "model_revision": self.embedder.config.model_revision,
                            },
                        },
                    )
                    job_id = job.id
            registration_committed = True
            log_event(
                logger,
                logging.INFO,
                "ingestion_job_created",
                ingestion_job_id=str(job_id),
                paper_id=str(paper_id),
                original_filename=original_filename,
                file_size=file_size,
                checksum=checksum,
                validation_result="valid",
            )
        except asyncio.CancelledError:
            if new_upload_created and new_upload_result is not None and not registration_committed:
                await self._cleanup_orphan_upload(new_upload_result.public_id, checksum)
            raise
        except Exception:
            if new_upload_created and new_upload_result is not None and not registration_committed:
                await self._cleanup_orphan_upload(new_upload_result.public_id, checksum)
            raise

        current_stage = "preprocessing"
        with correlation_context(ingestion_job_id=str(job_id) if job_id else None):
            try:
                current_stage = "preprocessing"
                await self._update_job_stage(job_id, current_stage)
                pre_start_t = time.perf_counter()
                preprocessed_paper = await asyncio.to_thread(
                    self.preprocessor.preprocess, parsed_paper
                )
                pre_duration_ms = (time.perf_counter() - pre_start_t) * 1000.0

                input_chars = sum(len(p.original_text) for p in preprocessed_paper.pages)
                output_chars = len(preprocessed_paper.full_text)
                modified_pages = preprocessed_paper.diagnostics.modified_pages
                removed_lines = len(preprocessed_paper.diagnostics.removed_repeated_lines)

                log_event(
                    logger,
                    logging.INFO,
                    "ingestion_preprocessing_completed",
                    ingestion_job_id=str(job_id),
                    paper_id=str(paper_id),
                    input_character_count=input_chars,
                    output_character_count=output_chars,
                    pages_processed=len(preprocessed_paper.pages),
                    modified_pages=modified_pages,
                    removed_repeated_lines=removed_lines,
                    duration_ms=round(pre_duration_ms, 2),
                )

                current_stage = "section_detection"
                await self._update_job_stage(job_id, current_stage)
                sec_start_t = time.perf_counter()
                section_detection_result = await asyncio.to_thread(
                    self.section_detector.detect, preprocessed_paper
                )
                sec_duration_ms = (time.perf_counter() - sec_start_t) * 1000.0

                headings_count = len(section_detection_result.sections)
                paragraph_assignments = sum(
                    len(s.paragraphs) for s in section_detection_result.sections
                ) + len(section_detection_result.preamble)

                log_event(
                    logger,
                    logging.INFO,
                    "ingestion_section_detection_completed",
                    ingestion_job_id=str(job_id),
                    paper_id=str(paper_id),
                    headings_count=headings_count,
                    detected_sections_count=headings_count,
                    paragraph_assignments_count=paragraph_assignments,
                    duration_ms=round(sec_duration_ms, 2),
                )

                current_stage = "chunking"
                await self._update_job_stage(job_id, current_stage)
                chunk_start_t = time.perf_counter()
                chunking_result = await asyncio.to_thread(
                    self.chunker.chunk, section_detection_result
                )
                chunk_duration_ms = (time.perf_counter() - chunk_start_t) * 1000.0

                token_counts = [c.token_count for c in chunking_result.chunks]
                total_tokens = sum(token_counts)
                min_tokens = min(token_counts) if token_counts else 0
                max_tokens = max(token_counts) if token_counts else 0
                avg_tokens = round(total_tokens / len(token_counts), 1) if token_counts else 0

                tokenizer_name = getattr(
                    self.chunker,
                    "tokenizer_identity",
                    type(self.chunker.token_counter).__name__
                    if getattr(self.chunker, "token_counter", None) is not None
                    else "WhitespaceTokenCounter",
                )

                log_event(
                    logger,
                    logging.INFO,
                    "ingestion_chunking_completed",
                    ingestion_job_id=str(job_id),
                    paper_id=str(paper_id),
                    tokenizer_name=tokenizer_name,
                    total_token_count=total_tokens,
                    min_token_count=min_tokens,
                    max_token_count=max_tokens,
                    avg_token_count=avg_tokens,
                    target_chunk_size=self.chunker.config.target_chunk_size,
                    max_chunk_size=self.chunker.config.max_chunk_size,
                    overlap=self.chunker.config.overlap,
                    chunks_count=len(chunking_result.chunks),
                    duration_ms=round(chunk_duration_ms, 2),
                )

                if getattr(logging_settings, "nlp_trace_enabled", False):
                    for c in chunking_result.chunks:
                        preview_content = (
                            c.content
                            if getattr(logging_settings, "nlp_trace_content_preview", False)
                            else None
                        )
                        log_event(
                            logger,
                            logging.DEBUG,
                            "ingestion_chunk_created",
                            ingestion_job_id=str(job_id),
                            chunk_index=c.chunk_index,
                            section_title=c.section_title or "Preamble",
                            page_range=f"{c.page_start}-{c.page_end}",
                            token_count=c.token_count,
                            char_count=c.char_count,
                            preview_field="content_preview" if preview_content else None,
                            content_preview=preview_content,
                        )

                current_stage = "embedding"
                await self._update_job_stage(job_id, current_stage)
                embed_start_t = time.perf_counter()
                async with self._embedding_semaphore:
                    embedding_result = await asyncio.to_thread(
                        self.embedder.embed_chunks, chunking_result.chunks
                    )
                embed_duration_ms = (time.perf_counter() - embed_start_t) * 1000.0

                log_event(
                    logger,
                    logging.INFO,
                    "ingestion_embeddings_completed",
                    ingestion_job_id=str(job_id),
                    paper_id=str(paper_id),
                    embedding_model=self.embedder.config.model_name,
                    dimensions=self.embedder.config.dimensions,
                    chunks_count=len(embedding_result.embeddings),
                    batch_size=getattr(self.embedder.config, "batch_size", 32),
                    duration_ms=round(embed_duration_ms, 2),
                )

                current_stage = "persisting"
                await self._update_job_stage(job_id, current_stage)
                persist_start_t = time.perf_counter()
                async with self.session_factory() as session:
                    async with session.begin():
                        ingestion_repo = IngestionRepository(session)

                        embedding_model = await ingestion_repo.get_or_create_embedding_model(
                            self.embedder.config
                        )

                        persisted_chunks = await ingestion_repo.bulk_create_chunks(
                            paper_id=paper_id,
                            ingestion_job_id=job_id,
                            chunks=chunking_result.chunks,
                            chunking_version=self.chunker.config.chunking_version,
                        )

                        persisted_embeddings = await ingestion_repo.bulk_create_embeddings(
                            chunks=persisted_chunks,
                            embedding_result=embedding_result,
                            embedding_model_id=embedding_model.id,
                        )

                        await ingestion_repo.activate_chunks_for_job(
                            paper_id=paper_id,
                            ingestion_job_id=job_id,
                            embedding_model_id=embedding_model.id,
                        )

                        now = datetime.now(timezone.utc)
                        await ingestion_repo.update_job_status(
                            job_id=job_id,
                            status="completed",
                            current_stage="completed",
                            completed_at=now,
                        )

                persist_duration_ms = (time.perf_counter() - persist_start_t) * 1000.0
                chunks_count = len(persisted_chunks)
                embeddings_count = len(persisted_embeddings)

                log_event(
                    logger,
                    logging.INFO,
                    "ingestion_persistence_completed",
                    ingestion_job_id=str(job_id),
                    paper_id=str(paper_id),
                    chunks_persisted=chunks_count,
                    embeddings_persisted=embeddings_count,
                    transaction_outcome="success",
                    activation_outcome="active",
                    duration_ms=round(persist_duration_ms, 2),
                )

            except asyncio.CancelledError:
                if job_id is not None:
                    cancellation_stage = (
                        f"{current_stage}_cancelled"
                        if len(f"{current_stage}_cancelled") <= 50
                        else "cancelled"
                    )
                    cancellation_msg = (
                        f"Ingestion task was cancelled during stage '{current_stage}'."
                    )
                    await self._record_job_cancellation(
                        job_id, cancellation_stage, cancellation_msg
                    )
                raise
            except Exception as exc:
                if job_id is not None:
                    await self._record_job_failure(job_id, str(exc))
                raise

        if self.retrieval_service is not None:
            try:
                refresh_start_t = time.perf_counter()
                async with self.session_factory() as session:
                    retrieval_repo = RetrievalRepository(session)
                    indexed_count = await self.retrieval_service.refresh_index(retrieval_repo)
                ref_duration_ms = (time.perf_counter() - refresh_start_t) * 1000.0
                log_event(
                    logger,
                    logging.INFO,
                    "ingestion_retrieval_refresh_completed",
                    ingestion_job_id=str(job_id),
                    paper_id=str(paper_id),
                    indexed_chunks_count=indexed_count,
                    duration_ms=round(ref_duration_ms, 2),
                )
            except Exception as refresh_exc:
                logger.warning(
                    "Retrieval index refresh failed after ingestion of paper %s: %s",
                    paper_id,
                    refresh_exc,
                    exc_info=True,
                )

        return IngestionResult(
            paper_id=paper_id,
            paper_file_id=paper_file_id,
            ingestion_job_id=job_id,
            chunks_created=chunks_count,
            embeddings_created=embeddings_count,
            paper_reused=paper_reused,
            file_reused=file_reused,
            job_reused=False,
            status="completed",
            title=resolution.paper.title,
        )
