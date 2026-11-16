"""Unit tests for IngestionService and check_active_jobs_for_deletion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import (
    ActiveJobConflictError,
    InvalidPDFError,
    PDFParsingError,
    StorageUploadError,
)
from app.modules.ingestion.chunker import (
    CharacterTokenCounter,
    ChunkingConfig,
    ChunkingDiagnostics,
    ChunkingResult,
    ResearchChunk,
    ResearchChunkingEngine,
    WhitespaceTokenCounter,
)
from app.modules.ingestion.embedder import (
    ChunkEmbedding as ResultChunkEmbedding,
    EmbeddingConfig,
    EmbeddingDiagnostics,
    EmbeddingGenerationError,
    EmbeddingResult,
    ResearchEmbeddingGenerator,
)
from app.modules.ingestion.models import IngestionJob, PaperChunk
from app.modules.ingestion.parser import (
    AcademicMetadata,
    AuthorCandidate,
    ParsedPage,
    ParsedResearchPaper,
    ParsingDiagnostics,
    PDFMetadata,
    PDFParser,
)
from app.modules.ingestion.preprocessor import (
    PreprocessedPage,
    PreprocessedResearchPaper,
    PreprocessingDiagnostics,
    ResearchTextPreprocessor,
)
from app.modules.ingestion.repository import IngestionRepository, IngestionRepositoryError
from app.modules.ingestion.section_detector import (
    DetectedSection,
    ResearchSectionDetector,
    SectionDetectionDiagnostics,
    SectionDetectionResult,
    SectionParagraph,
    SectionType,
)
from app.modules.ingestion.service import (
    IngestionResult,
    IngestionService,
    check_active_jobs_for_deletion,
)
from app.modules.ingestion.storage import CloudinaryStorage, CloudinaryUploadResult
from app.modules.papers.models import Paper, PaperFile
from app.modules.papers.repository import PaperRepository, PaperValidationError
from app.modules.papers.service import PaperResolutionResult, PaperService
from app.modules.retrieval.models import ChunkEmbedding as DBChunkEmbedding, EmbeddingModel
from app.modules.retrieval.repository import RetrievalRepository
from app.modules.retrieval.service import RetrievalService


# -------------------------------------------------------------------------
# Test Fixtures & Helpers
# -------------------------------------------------------------------------

@pytest.fixture
def mock_session() -> AsyncMock:
    """Create a mock AsyncSession that supports `async with session.begin():`."""
    session = AsyncMock(spec=AsyncSession)
    tx_cm = AsyncMock()
    tx_cm.__aenter__.return_value = session
    tx_cm.__aexit__.return_value = None
    session.begin.return_value = tx_cm
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    session.execute.return_value = mock_result
    return session


@pytest.fixture
def mock_session_factory(mock_session: AsyncMock) -> MagicMock:
    """Create a mock async_sessionmaker yielding mock_session."""
    factory = MagicMock(spec=async_sessionmaker)
    session_cm = AsyncMock()
    session_cm.__aenter__.return_value = mock_session
    session_cm.__aexit__.return_value = None
    factory.return_value = session_cm
    return factory


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return b"%PDF-1.5 \n sample pdf content for unit test \n"


@pytest.fixture
def sample_parsed_paper() -> ParsedResearchPaper:
    return ParsedResearchPaper(
        pdf_metadata=PDFMetadata(title="PDF Title", page_count=2),
        academic_metadata=AcademicMetadata(
            title="Extracted Paper Title",
            doi="10.1000/182",
            abstract="Sample Abstract",
            authors=[AuthorCandidate(name="Alice", affiliation="Uni")],
            publication_year=2023,
            publication_date="2023-05-15",
            venue="NeurIPS",
            publisher="Curran Associates",
        ),
        pages=[ParsedPage(page_number=1, text="Page 1 text", extraction_status="SUCCESS")],
        diagnostics=ParsingDiagnostics(total_pages=1, pages_with_text=1, pages_without_text=0),
    )


@pytest.fixture
def sample_preprocessed_paper() -> PreprocessedResearchPaper:
    return PreprocessedResearchPaper(
        pages=[PreprocessedPage(page_number=1, original_text="Orig", cleaned_text="Cleaned", paragraphs=["Cleaned"])],
        full_text="Cleaned",
        diagnostics=PreprocessingDiagnostics(total_pages=1, modified_pages=0),
    )


@pytest.fixture
def sample_section_result() -> SectionDetectionResult:
    return SectionDetectionResult(
        sections=[
            DetectedSection(
                section_id="sec_1",
                section_index=0,
                heading_paragraph_index=0,
                title="1. Introduction",
                normalized_title="introduction",
                section_type=SectionType.INTRODUCTION,
                level=1,
                parent_section_id=None,
                heading_page=1,
                start_page=1,
                end_page=1,
                paragraphs=[SectionParagraph(text="Intro para", page_number=1, paragraph_index=1)],
            )
        ],
        preamble=[],
        diagnostics=SectionDetectionDiagnostics(
            total_paragraphs=1,
            detected_headings=1,
            rejected_candidates=0,
            unclassified_headings=0,
            sections_created=1,
        ),
    )


@pytest.fixture
def sample_chunking_result() -> ChunkingResult:
    chunk = ResearchChunk(
        chunk_index=0,
        content="Intro para",
        token_count=10,
        char_count=10,
        page_start=1,
        page_end=1,
        section_title="1. Introduction",
        content_hash=hashlib.sha256(b"Intro para").hexdigest(),
        metadata={"section_type": "introduction"},
    )
    return ChunkingResult(
        chunks=[chunk],
        diagnostics=ChunkingDiagnostics(
            input_paragraph_count=1,
            output_chunk_count=1,
            sections_processed=1,
            oversized_paragraphs_split=0,
            total_overlap_tokens=0,
        ),
    )


@pytest.fixture
def sample_embedding_result(sample_chunking_result: ChunkingResult) -> EmbeddingResult:
    c = sample_chunking_result.chunks[0]
    return EmbeddingResult(
        embeddings=[
            ResultChunkEmbedding(
                chunk_index=c.chunk_index,
                content_hash=c.content_hash,
                vector=[0.1] * 384,
            )
        ],
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        provider="sentence-transformers",
        dimensions=384,
        normalized=True,
        diagnostics=EmbeddingDiagnostics(
            input_chunks=1,
            embedded_chunks=1,
            batch_count=1,
            elapsed_seconds=0.01,
            warnings=[],
        ),
    )


# -------------------------------------------------------------------------
# Test Suite 1: Existing check_active_jobs_for_deletion Functionality
# -------------------------------------------------------------------------

class TestCheckActiveJobsForDeletion:
    """Verifies that check_active_jobs_for_deletion behavior is preserved."""

    def test_active_jobs_present_raises_error(self) -> None:
        target_id = uuid.uuid4()
        active_job = MagicMock(status="processing")
        with pytest.raises(ActiveJobConflictError, match="active job.*in progress"):
            check_active_jobs_for_deletion("paper", target_id, [active_job])

    def test_only_terminal_jobs_present_succeeds(self) -> None:
        target_id = uuid.uuid4()
        completed_job = MagicMock(status="completed")
        failed_job = MagicMock(status="failed")
        check_active_jobs_for_deletion("paper", target_id, [completed_job, failed_job])

    def test_empty_jobs_succeeds(self) -> None:
        check_active_jobs_for_deletion("paper", uuid.uuid4(), [])


# -------------------------------------------------------------------------
# Test Suite 2: IngestionService Input Validation
# -------------------------------------------------------------------------

class TestIngestionServiceInputValidation:
    """Tests for input parameter and PDF signature validation."""

    @pytest.mark.asyncio
    async def test_invalid_filename_raises_value_error(self, mock_session_factory: MagicMock) -> None:
        service = IngestionService(session_factory=mock_session_factory)
        with pytest.raises(ValueError, match="original_filename must be a non-empty string"):
            await service.ingest_pdf(b"%PDF-1.5 test", "")

    @pytest.mark.asyncio
    async def test_invalid_session_factory_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="session_factory must be an async_sessionmaker"):
            IngestionService(session_factory="not-a-sessionmaker")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_invalid_pdf_bytes_rejected(self, mock_session_factory: MagicMock) -> None:
        storage = MagicMock(spec=CloudinaryStorage)
        storage.validate_pdf.side_effect = InvalidPDFError("PDF content must start with b'%PDF-'.")

        service = IngestionService(session_factory=mock_session_factory, storage=storage)
        with pytest.raises(InvalidPDFError, match="PDF content must start with"):
            await service.ingest_pdf(b"Not a PDF", "paper.pdf")


# -------------------------------------------------------------------------
# Test Suite 3: Metadata Precedence & Title Fallback
# -------------------------------------------------------------------------

class TestMetadataPrecedenceAndFallback:
    """Tests for title, DOI, and metadata extraction precedence."""

    @pytest.mark.asyncio
    async def test_user_supplied_title_and_doi_take_precedence(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="User Title", doi="10.1234/custom")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job
        ingestion_repo_inst.bulk_create_chunks.return_value = [MagicMock()]
        ingestion_repo_inst.bulk_create_embeddings.return_value = [MagicMock()]
        ingestion_repo_inst.activate_chunks_for_job.return_value = [MagicMock()]

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper

            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result

            chunker = MagicMock(spec=ResearchChunkingEngine)
            chunker.config = ChunkingConfig()
            chunker.chunk.return_value = sample_chunking_result

            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            result = await service.ingest_pdf(
                sample_pdf_bytes,
                "custom.pdf",
                title="User Title",
                doi="10.1234/custom",
            )

            assert result.status == "completed"
            _, kwargs = paper_service_inst.resolve_or_create_paper.call_args
            assert kwargs["title"] == "User Title"
            assert kwargs["doi"] == "10.1234/custom"

    @pytest.mark.asyncio
    async def test_extracted_title_used_when_user_title_omitted(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Extracted Paper Title")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.return_value = paper_file

        storage.upload_file.return_value = CloudinaryUploadResult(
            public_id="p1", secure_url="https://c.com/p1.pdf", byte_size=1024
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.create_job.return_value = job
        ingestion_repo_inst.bulk_create_chunks.return_value = [MagicMock()]
        ingestion_repo_inst.bulk_create_embeddings.return_value = [MagicMock()]
        ingestion_repo_inst.activate_chunks_for_job.return_value = [MagicMock()]

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper

            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result

            chunker = MagicMock(spec=ResearchChunkingEngine)
            chunker.config = ChunkingConfig()
            chunker.chunk.return_value = sample_chunking_result

            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            await service.ingest_pdf(sample_pdf_bytes, "paper.pdf", title=None)

            _, kwargs = paper_service_inst.resolve_or_create_paper.call_args
            assert kwargs["title"] == "Extracted Paper Title"
            assert kwargs["doi"] == "10.1000/182"


# -------------------------------------------------------------------------
# Test Suite 4: File Reuse & Cloudinary Upload Logic
# -------------------------------------------------------------------------

class TestFileReuseAndStorageUpload:
    """Tests for Cloudinary uploads, file reuse, and registration failure cleanup."""

    @pytest.mark.asyncio
    async def test_reused_file_skips_cloudinary_upload(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Existing Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job
        ingestion_repo_inst.bulk_create_chunks.return_value = [MagicMock()]
        ingestion_repo_inst.bulk_create_embeddings.return_value = [MagicMock()]
        ingestion_repo_inst.activate_chunks_for_job.return_value = [MagicMock()]

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper

            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result

            chunker = MagicMock(spec=ResearchChunkingEngine)
            chunker.config = ChunkingConfig()
            chunker.chunk.return_value = sample_chunking_result

            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            result = await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            assert result.file_reused is True
            storage.upload_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_upload_fails_registration_triggers_cloudinary_cleanup(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="temp_pub_123", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res

        paper = Paper(id=uuid.uuid4(), title="New Paper")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.side_effect = RuntimeError("DB Disk Full")

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(RuntimeError, match="DB Disk Full"):
                await service.ingest_pdf(sample_pdf_bytes, "new_paper.pdf")

            storage.upload_file.assert_called_once_with(sample_pdf_bytes)
            storage.delete_file.assert_called_once_with("temp_pub_123")


# -------------------------------------------------------------------------
# Test Suite 5: Idempotency & Completed Job Reuse
# -------------------------------------------------------------------------

class TestIdempotencyAndCompletedJobReuse:
    """Tests for completed job detection, fingerprint matching, and force_reprocess."""

    @pytest.mark.asyncio
    async def test_idempotent_completed_job_reused(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Existing Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")

        chunker = MagicMock(spec=ResearchChunkingEngine)
        chunker.config = ChunkingConfig()
        embedder = MagicMock(spec=ResearchEmbeddingGenerator)
        embedder.config = EmbeddingConfig()

        service = IngestionService(
            session_factory=mock_session_factory,
            storage=storage,
            parser=parser,
            chunker=chunker,
            embedder=embedder,
        )

        expected_fingerprint = service._compute_pipeline_fingerprint()

        completed_job = IngestionJob(
            id=uuid.uuid4(),
            paper_id=paper.id,
            paper_file_id=paper_file.id,
            status="completed",
            metadata_={"pipeline_fingerprint": expected_fingerprint},
        )
        active_chunk = PaperChunk(
            id=uuid.uuid4(), paper_id=paper.id, ingestion_job_id=completed_job.id, chunk_index=0, is_current=True, content="c"
        )

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = completed_job
        ingestion_repo_inst.get_job_chunks.return_value = [active_chunk]

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            result = await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            assert result.job_reused is True
            assert result.status == "completed"
            assert result.ingestion_job_id == completed_job.id
            ingestion_repo_inst.create_job.assert_not_called()
            embedder.embed_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_idempotency_bypassed_when_min_chunk_size_changes(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        """When min_chunk_size changes, fingerprint differs and previous completed job is not reused."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Existing Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        new_job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        completed_job = IngestionJob(
            id=uuid.uuid4(),
            paper_id=paper.id,
            paper_file_id=paper_file.id,
            status="completed",
            metadata_={"pipeline_fingerprint": "old_fingerprint_with_different_min_chunk_size"},
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = completed_job
        ingestion_repo_inst.create_job.return_value = new_job
        ingestion_repo_inst.bulk_create_chunks.return_value = [MagicMock()]
        ingestion_repo_inst.bulk_create_embeddings.return_value = [MagicMock()]
        ingestion_repo_inst.activate_chunks_for_job.return_value = [MagicMock()]

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper
            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result
            chunker = ResearchChunkingEngine(
                token_counter=WhitespaceTokenCounter(),
                config=ChunkingConfig(min_chunk_size=50),
            )
            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            result = await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            assert result.job_reused is False
            assert result.ingestion_job_id == new_job.id
            ingestion_repo_inst.create_job.assert_called_once()
            embedder.embed_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_reprocess_bypasses_completed_job(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Existing Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        new_job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.create_job.return_value = new_job
        ingestion_repo_inst.bulk_create_chunks.return_value = [MagicMock()]
        ingestion_repo_inst.bulk_create_embeddings.return_value = [MagicMock()]
        ingestion_repo_inst.activate_chunks_for_job.return_value = [MagicMock()]

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper

            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result

            chunker = MagicMock(spec=ResearchChunkingEngine)
            chunker.config = ChunkingConfig()
            chunker.chunk.return_value = sample_chunking_result

            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            result = await service.ingest_pdf(sample_pdf_bytes, "paper.pdf", force_reprocess=True)

            assert result.job_reused is False
            assert result.status == "completed"
            ingestion_repo_inst.create_job.assert_called_once()
            embedder.embed_chunks.assert_called_once()


# -------------------------------------------------------------------------
# Test Suite 6: Concurrency Protection
# -------------------------------------------------------------------------

class TestConcurrencyProtection:
    """Tests for active job conflict and row-level locking."""

    @pytest.mark.asyncio
    async def test_active_job_for_paper_raises_conflict(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Existing Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        active_job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing", current_stage="chunking")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = active_job

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(ActiveJobConflictError, match="active job .* is already in progress"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            ingestion_repo_inst.lock_paper.assert_called_once_with(paper.id)


# -------------------------------------------------------------------------
# Test Suite 7: Pipeline Stage Failures & Failed Job Recording
# -------------------------------------------------------------------------

class TestPipelineStageFailures:
    """Tests for failure handling at every processing and persistence stage."""

    @pytest.mark.parametrize("failed_stage,exception_class,msg", [
        ("preprocessing", RuntimeError, "Preprocessor crashed"),
        ("section_detection", ValueError, "Malformed section hierarchy"),
        ("chunking", RuntimeError, "Chunker out of memory"),
        ("embedding", EmbeddingGenerationError, "Inference device failure"),
        ("persisting", IngestionRepositoryError, "Activation foreign key violation"),
    ])
    @pytest.mark.asyncio
    async def test_stage_failure_marks_job_failed_and_raises(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
        failed_stage: str,
        exception_class: type[Exception],
        msg: str,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job_id = uuid.uuid4()
        job = IngestionJob(id=job_id, paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job

        preprocessor = MagicMock(spec=ResearchTextPreprocessor)
        section_detector = MagicMock(spec=ResearchSectionDetector)
        chunker = MagicMock(spec=ResearchChunkingEngine)
        chunker.config = ChunkingConfig()
        embedder = MagicMock(spec=ResearchEmbeddingGenerator)
        embedder.config = EmbeddingConfig()

        if failed_stage == "preprocessing":
            preprocessor.preprocess.side_effect = exception_class(msg)
        else:
            preprocessor.preprocess.return_value = sample_preprocessed_paper

        if failed_stage == "section_detection":
            section_detector.detect.side_effect = exception_class(msg)
        else:
            section_detector.detect.return_value = sample_section_result

        if failed_stage == "chunking":
            chunker.chunk.side_effect = exception_class(msg)
        else:
            chunker.chunk.return_value = sample_chunking_result

        if failed_stage == "embedding":
            embedder.embed_chunks.side_effect = exception_class(msg)
        else:
            embedder.embed_chunks.return_value = sample_embedding_result

        if failed_stage == "persisting":
            ingestion_repo_inst.bulk_create_chunks.side_effect = exception_class(msg)

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            with pytest.raises(exception_class, match=msg):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            failed_calls = [
                call for call in ingestion_repo_inst.update_job_status.call_args_list
                if call.kwargs.get("status") == "failed"
                or (len(call.args) > 1 and call.args[1] == "failed")
            ]
            assert len(failed_calls) >= 1
            error_msg_recorded = (
                failed_calls[0].kwargs.get("error_message")
                or (failed_calls[0].args[2] if len(failed_calls[0].args) > 2 else "")
            )
            assert msg in error_msg_recorded


# -------------------------------------------------------------------------
# Test Suite 8: Retrieval Refresh Behavior
# -------------------------------------------------------------------------

class TestRetrievalRefreshBehavior:
    """Tests for retrieval index refresh after commit and failure isolation."""

    @pytest.mark.asyncio
    async def test_retrieval_refresh_called_after_commit(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        retrieval_service = AsyncMock(spec=RetrievalService)

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job
        ingestion_repo_inst.bulk_create_chunks.return_value = [MagicMock()]
        ingestion_repo_inst.bulk_create_embeddings.return_value = [MagicMock()]
        ingestion_repo_inst.activate_chunks_for_job.return_value = [MagicMock()]

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst), \
             patch("app.modules.ingestion.service.RetrievalRepository"):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper
            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result
            chunker = MagicMock(spec=ResearchChunkingEngine)
            chunker.config = ChunkingConfig()
            chunker.chunk.return_value = sample_chunking_result
            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
                retrieval_service=retrieval_service,
            )

            result = await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            assert result.status == "completed"
            retrieval_service.refresh_index.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieval_refresh_failure_does_not_fail_ingestion(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        retrieval_service = AsyncMock(spec=RetrievalService)
        retrieval_service.refresh_index.side_effect = RuntimeError("Index rebuild failed")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job
        ingestion_repo_inst.bulk_create_chunks.return_value = [MagicMock()]
        ingestion_repo_inst.bulk_create_embeddings.return_value = [MagicMock()]
        ingestion_repo_inst.activate_chunks_for_job.return_value = [MagicMock()]

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst), \
             patch("app.modules.ingestion.service.RetrievalRepository"):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper
            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result
            chunker = MagicMock(spec=ResearchChunkingEngine)
            chunker.config = ChunkingConfig()
            chunker.chunk.return_value = sample_chunking_result
            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
                retrieval_service=retrieval_service,
            )

            result = await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            assert result.status == "completed"
            retrieval_service.refresh_index.assert_awaited_once()


# =========================================================================
# Concurrency & Semaphore Validation Tests
# =========================================================================

class TestConcurrencyAndSemaphoreValidation:
    """Tests for max_concurrent_embeddings parameter validation."""

    @pytest.mark.parametrize("invalid_val", [True, False, 1.5, 0.0, 0, -1, -5, "3"])
    def test_invalid_max_concurrent_embeddings_raises_value_error(
        self, invalid_val: Any
    ) -> None:
        """max_concurrent_embeddings must strictly be a positive integer."""
        with pytest.raises(ValueError, match="max_concurrent_embeddings must be a positive integer"):
            IngestionService(max_concurrent_embeddings=invalid_val)

    @pytest.mark.parametrize("valid_val", [1, 2, 4, 16])
    def test_valid_max_concurrent_embeddings_initializes_semaphore(
        self, valid_val: int
    ) -> None:
        """Valid positive integer initializes the embedding semaphore."""
        service = IngestionService(max_concurrent_embeddings=valid_val)
        assert service._embedding_semaphore._value == valid_val


# =========================================================================
# Pipeline Fingerprint & Metadata Tests
# =========================================================================

class TestPipelineFingerprintConfiguration:
    """Tests for pipeline fingerprint and metadata persistence including min_chunk_size and token_counter."""

    def test_fingerprint_changes_with_min_chunk_size(self) -> None:
        """Different min_chunk_size values produce different fingerprint hashes."""
        engine1 = ResearchChunkingEngine(
            token_counter=WhitespaceTokenCounter(),
            config=ChunkingConfig(min_chunk_size=40),
        )
        engine2 = ResearchChunkingEngine(
            token_counter=WhitespaceTokenCounter(),
            config=ChunkingConfig(min_chunk_size=50),
        )
        s1 = IngestionService(chunker=engine1)
        s2 = IngestionService(chunker=engine2)

        fp1 = s1._compute_pipeline_fingerprint()
        fp2 = s2._compute_pipeline_fingerprint()
        assert fp1 != fp2

    def test_fingerprint_changes_with_token_counter(self) -> None:
        """Different TokenCounter implementations produce different fingerprint hashes."""
        engine1 = ResearchChunkingEngine(
            token_counter=WhitespaceTokenCounter(),
            config=ChunkingConfig(min_chunk_size=40),
        )
        engine2 = ResearchChunkingEngine(
            token_counter=CharacterTokenCounter(),
            config=ChunkingConfig(min_chunk_size=40),
        )
        s1 = IngestionService(chunker=engine1)
        s2 = IngestionService(chunker=engine2)

        fp1 = s1._compute_pipeline_fingerprint()
        fp2 = s2._compute_pipeline_fingerprint()
        assert fp1 != fp2

    @pytest.mark.asyncio
    async def test_min_chunk_size_and_token_counter_persisted_in_job_metadata(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        """Job metadata contains min_chunk_size and token_counter in chunking_config."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job
        ingestion_repo_inst.bulk_create_chunks.return_value = [MagicMock()]
        ingestion_repo_inst.bulk_create_embeddings.return_value = [MagicMock()]
        ingestion_repo_inst.activate_chunks_for_job.return_value = [MagicMock()]

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper
            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result
            chunker = ResearchChunkingEngine(
                token_counter=WhitespaceTokenCounter(),
                config=ChunkingConfig(min_chunk_size=45),
            )
            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            create_kwargs = ingestion_repo_inst.create_job.call_args[1]
            chunking_cfg = create_kwargs["metadata_"]["chunking_config"]
            assert chunking_cfg["min_chunk_size"] == 45
            assert chunking_cfg["token_counter"] == "WhitespaceTokenCounter"
            assert create_kwargs["started_at"] is not None


# =========================================================================
# Cloudinary Orphan Cleanup Tests
# =========================================================================

class TestCloudinaryOrphanCleanupScenarios:
    """Comprehensive tests for Cloudinary orphan cleanup under various failure conditions."""

    @pytest.mark.asyncio
    async def test_active_job_conflict_cleans_up_new_cloudinary_upload(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """When an active job conflict is detected after upload, the newly uploaded asset is deleted."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="conflict_pub_1", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res

        paper = Paper(id=uuid.uuid4(), title="Conflict Paper")
        existing_job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing", current_stage="chunking")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = existing_job

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(ActiveJobConflictError, match="active job"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            storage.upload_file.assert_called_once_with(sample_pdf_bytes)
            storage.delete_file.assert_called_once_with("conflict_pub_1")

    @pytest.mark.asyncio
    async def test_create_job_failure_cleans_up_new_cloudinary_upload(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """When create_job fails after file registration, the newly uploaded asset is deleted."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="create_job_fail_pub", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res

        paper = Paper(id=uuid.uuid4(), title="Paper")
        registered_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="create_job_fail_pub")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.return_value = registered_file

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.create_job.side_effect = RuntimeError("Database write error during create_job")

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(RuntimeError, match="Database write error during create_job"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            storage.delete_file.assert_called_once_with("create_job_fail_pub")

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_mask_original_exception(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """When storage.delete_file fails during orphan cleanup, the original exception is preserved."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="cleanup_err_pub", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res
        storage.delete_file.side_effect = StorageUploadError("Cloudinary network timeout on delete")

        paper = Paper(id=uuid.uuid4(), title="Paper")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.side_effect = RuntimeError("Original registration failure")

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
            )

            # Original RuntimeError must be raised, not masked by StorageUploadError
            with pytest.raises(RuntimeError, match="Original registration failure"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            storage.delete_file.assert_called_once_with("cleanup_err_pub")

    @pytest.mark.asyncio
    async def test_reused_file_never_triggers_cloudinary_cleanup(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """When an existing file is reused and a later step fails, delete_file is NEVER called."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Paper")
        existing_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="existing_pub")
        existing_job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=existing_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = existing_job

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(ActiveJobConflictError):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            storage.upload_file.assert_not_called()
            storage.delete_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_verification_finds_paper_file_skips_orphan_cleanup(
        self,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """If verification query confirms PaperFile exists in DB, orphan cleanup is skipped."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="verified_pub_in_db", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res

        paper = Paper(id=uuid.uuid4(), title="Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="verified_pub_in_db")

        # Session factory for verification that returns the paper_file
        mock_verify_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = paper_file
        mock_verify_session.execute.return_value = mock_result

        tx_cm = AsyncMock()
        tx_cm.__aenter__.return_value = mock_verify_session
        mock_verify_session.begin.return_value = tx_cm

        session_factory = MagicMock(spec=async_sessionmaker)
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = mock_verify_session
        session_factory.return_value = session_cm

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.side_effect = RuntimeError("Something failed")

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(RuntimeError, match="Something failed"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            # delete_file must NOT be called because verification found PaperFile in DB
            storage.delete_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_verification_error_skips_orphan_cleanup(
        self,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """If verification query encounters a database error, orphan cleanup is safely skipped."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="uncertain_pub", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res

        paper = Paper(id=uuid.uuid4(), title="Paper")

        # Session factory that raises an error during verification
        mock_verify_session = AsyncMock(spec=AsyncSession)
        mock_verify_session.execute.side_effect = RuntimeError("DB connection lost during verify")

        tx_cm = AsyncMock()
        tx_cm.__aenter__.return_value = mock_verify_session
        mock_verify_session.begin.return_value = tx_cm

        session_factory = MagicMock(spec=async_sessionmaker)
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = mock_verify_session
        session_factory.return_value = session_cm

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.side_effect = RuntimeError("Initial failure")

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(RuntimeError, match="Initial failure"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            # Uncertainty means we do NOT delete
            storage.delete_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_failure_cleans_up_new_upload_when_file_not_in_db(
        self,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """When transaction commit fails and DB confirms PaperFile does not exist, upload is deleted."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="commit_fail_orphan", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res

        paper = Paper(id=uuid.uuid4(), title="Paper")
        registered_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="commit_fail_orphan")

        mock_session = AsyncMock(spec=AsyncSession)

        stage_b_tx = AsyncMock()
        stage_b_tx.__aenter__.return_value = mock_session
        stage_b_tx.__aexit__.return_value = None

        stage_cd_tx = AsyncMock()
        stage_cd_tx.__aenter__.return_value = mock_session
        stage_cd_tx.__aexit__.side_effect = RuntimeError("Commit failed: database serialization failure")

        mock_session.begin.side_effect = [stage_b_tx, stage_cd_tx]

        mock_verify_result = MagicMock()
        mock_verify_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_verify_result

        session_factory = MagicMock(spec=async_sessionmaker)
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = mock_session
        session_cm.__aexit__.return_value = None
        session_factory.return_value = session_cm

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.return_value = registered_file

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.create_job.return_value = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(RuntimeError, match="Commit failed: database serialization failure"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            storage.upload_file.assert_called_once_with(sample_pdf_bytes)
            storage.delete_file.assert_called_once_with("commit_fail_orphan")

    @pytest.mark.asyncio
    async def test_commit_failure_skips_cleanup_if_paper_file_persisted(
        self,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """When transaction commit fails but verification confirms PaperFile persisted, upload is retained."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="commit_fail_persisted", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res

        paper = Paper(id=uuid.uuid4(), title="Paper")
        registered_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="commit_fail_persisted")

        mock_session = AsyncMock(spec=AsyncSession)

        stage_b_tx = AsyncMock()
        stage_b_tx.__aenter__.return_value = mock_session
        stage_b_tx.__aexit__.return_value = None

        stage_cd_tx = AsyncMock()
        stage_cd_tx.__aenter__.return_value = mock_session
        stage_cd_tx.__aexit__.side_effect = RuntimeError("Commit error at boundary")

        mock_session.begin.side_effect = [stage_b_tx, stage_cd_tx]

        mock_verify_result = MagicMock()
        mock_verify_result.scalar_one_or_none.return_value = registered_file
        mock_session.execute.return_value = mock_verify_result

        session_factory = MagicMock(spec=async_sessionmaker)
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = mock_session
        session_cm.__aexit__.return_value = None
        session_factory.return_value = session_cm

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.return_value = registered_file

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.create_job.return_value = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(RuntimeError, match="Commit error at boundary"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            storage.upload_file.assert_called_once_with(sample_pdf_bytes)
            storage.delete_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_failure_skips_cleanup_if_db_verification_fails(
        self,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """When transaction commit fails and verification query fails, upload is retained due to uncertainty."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="commit_fail_uncertain", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res

        paper = Paper(id=uuid.uuid4(), title="Paper")
        registered_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="commit_fail_uncertain")

        mock_session = AsyncMock(spec=AsyncSession)

        stage_b_tx = AsyncMock()
        stage_b_tx.__aenter__.return_value = mock_session
        stage_b_tx.__aexit__.return_value = None

        stage_cd_tx = AsyncMock()
        stage_cd_tx.__aenter__.return_value = mock_session
        stage_cd_tx.__aexit__.side_effect = RuntimeError("Primary commit failure")

        mock_session.begin.side_effect = [stage_b_tx, stage_cd_tx]

        mock_session.execute.side_effect = RuntimeError("DB verification query network error")

        session_factory = MagicMock(spec=async_sessionmaker)
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = mock_session
        session_cm.__aexit__.return_value = None
        session_factory.return_value = session_cm

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.return_value = registered_file

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.create_job.return_value = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=session_factory,
                storage=storage,
                parser=parser,
            )

            with pytest.raises(RuntimeError, match="Primary commit failure"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            storage.upload_file.assert_called_once_with(sample_pdf_bytes)
            storage.delete_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_commit_failure_preserves_original_exception_when_cleanup_fails(
        self,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """When transaction commit fails and delete_file also fails, original commit exception is preserved."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)
        upload_res = CloudinaryUploadResult(public_id="commit_fail_del_err", secure_url="https://c/1", byte_size=100)
        storage.upload_file.return_value = upload_res
        storage.delete_file.side_effect = StorageUploadError("Cloudinary delete timeout")

        paper = Paper(id=uuid.uuid4(), title="Paper")
        registered_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="commit_fail_del_err")

        mock_session = AsyncMock(spec=AsyncSession)

        stage_b_tx = AsyncMock()
        stage_b_tx.__aenter__.return_value = mock_session
        stage_b_tx.__aexit__.return_value = None

        stage_cd_tx = AsyncMock()
        stage_cd_tx.__aenter__.return_value = mock_session
        stage_cd_tx.__aexit__.side_effect = RuntimeError("Primary commit failure")

        mock_session.begin.side_effect = [stage_b_tx, stage_cd_tx]

        mock_verify_result = MagicMock()
        mock_verify_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_verify_result

        session_factory = MagicMock(spec=async_sessionmaker)
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = mock_session
        session_cm.__aexit__.return_value = None
        session_factory.return_value = session_cm

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=None, paper_reused=False, file_reused=False
        )
        paper_service_inst.register_paper_file.return_value = registered_file

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.create_job.return_value = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=session_factory,
                storage=storage,
                parser=parser,
            )

            # Original commit RuntimeError must be preserved, not masked by StorageUploadError
            with pytest.raises(RuntimeError, match="Primary commit failure"):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            storage.upload_file.assert_called_once_with(sample_pdf_bytes)
            storage.delete_file.assert_called_once_with("commit_fail_del_err")


# =========================================================================
# Task Cancellation Handling Tests
# =========================================================================

class TestTaskCancellationHandling:
    """Tests for handling asyncio.CancelledError during pipeline execution."""

    @pytest.mark.asyncio
    async def test_cancellation_during_preprocessing_marks_job_cancelled_and_reraises(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
    ) -> None:
        """When cancelled during preprocessing, job is marked cancelled with stage and CancelledError propagates."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job

        preprocessor = MagicMock(spec=ResearchTextPreprocessor)
        preprocessor.preprocess.side_effect = asyncio.CancelledError()

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
            )

            with pytest.raises(asyncio.CancelledError):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            # Check cancel_job_if_active was called with stage and error message
            ingestion_repo_inst.cancel_job_if_active.assert_awaited_once()
            call_kwargs = ingestion_repo_inst.cancel_job_if_active.call_args[1]
            assert call_kwargs["job_id"] == job.id
            assert call_kwargs["current_stage"] == "preprocessing_cancelled"
            assert "cancelled during stage 'preprocessing'" in call_kwargs["error_message"]
            assert call_kwargs["completed_at"] is not None

            # update_job_status must NOT have been called directly for cancellation
            cancelled_update_calls = [
                c for c in ingestion_repo_inst.update_job_status.call_args_list
                if c[1].get("status") == "cancelled"
            ]
            assert len(cancelled_update_calls) == 0

    @pytest.mark.asyncio
    async def test_cancellation_during_embedding_marks_job_cancelled_and_reraises(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
    ) -> None:
        """When cancelled during embedding, job is marked cancelled with embedding_cancelled stage."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job

        preprocessor = MagicMock(spec=ResearchTextPreprocessor)
        preprocessor.preprocess.return_value = sample_preprocessed_paper
        section_detector = MagicMock(spec=ResearchSectionDetector)
        section_detector.detect.return_value = sample_section_result
        chunker = MagicMock(spec=ResearchChunkingEngine)
        chunker.config = ChunkingConfig()
        chunker.chunk.return_value = sample_chunking_result
        embedder = MagicMock(spec=ResearchEmbeddingGenerator)
        embedder.config = EmbeddingConfig()
        embedder.embed_chunks.side_effect = asyncio.CancelledError()

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            with pytest.raises(asyncio.CancelledError):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            # Check cancel_job_if_active was called with stage and error message
            ingestion_repo_inst.cancel_job_if_active.assert_awaited_once()
            call_kwargs = ingestion_repo_inst.cancel_job_if_active.call_args[1]
            assert call_kwargs["job_id"] == job.id
            assert call_kwargs["current_stage"] == "embedding_cancelled"
            assert "cancelled during stage 'embedding'" in call_kwargs["error_message"]
            assert call_kwargs["completed_at"] is not None

            # update_job_status must NOT have been called directly for cancellation
            cancelled_update_calls = [
                c for c in ingestion_repo_inst.update_job_status.call_args_list
                if c[1].get("status") == "cancelled"
            ]
            assert len(cancelled_update_calls) == 0

            # Persistence must not have occurred
            ingestion_repo_inst.bulk_create_chunks.assert_not_called()
            ingestion_repo_inst.bulk_create_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancellation_during_final_persistence_transaction_rolls_back_and_marks_job_cancelled(
        self,
        mock_session_factory: MagicMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        """When cancellation occurs during Stage F and transaction rolls back, job becomes cancelled."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job
        # When get_job is called during cancellation, the persisted DB job is still processing because persistence rolled back
        ingestion_repo_inst.get_job.return_value = job

        # Cancellation occurs during bulk_create_chunks inside persistence transaction
        ingestion_repo_inst.bulk_create_chunks.side_effect = asyncio.CancelledError("Cancelled during chunk insertion")

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper
            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result
            chunker = MagicMock(spec=ResearchChunkingEngine)
            chunker.config = ChunkingConfig()
            chunker.chunk.return_value = sample_chunking_result
            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=mock_session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            with pytest.raises(asyncio.CancelledError):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            # Check cancel_job_if_active was called with stage and error message
            ingestion_repo_inst.cancel_job_if_active.assert_awaited_once()
            call_kwargs = ingestion_repo_inst.cancel_job_if_active.call_args[1]
            assert call_kwargs["job_id"] == job.id
            assert call_kwargs["current_stage"] == "persisting_cancelled"
            assert "cancelled during stage 'persisting'" in call_kwargs["error_message"]
            assert call_kwargs["completed_at"] is not None

            # update_job_status must NOT have been called directly for cancellation
            cancelled_update_calls = [
                c for c in ingestion_repo_inst.update_job_status.call_args_list
                if c[1].get("status") == "cancelled"
            ]
            assert len(cancelled_update_calls) == 0

    @pytest.mark.asyncio
    async def test_cancellation_after_persistence_transaction_committed_preserves_completed_job(
        self,
        mock_session: AsyncMock,
        sample_pdf_bytes: bytes,
        sample_parsed_paper: ParsedResearchPaper,
        sample_preprocessed_paper: PreprocessedResearchPaper,
        sample_section_result: SectionDetectionResult,
        sample_chunking_result: ChunkingResult,
        sample_embedding_result: EmbeddingResult,
    ) -> None:
        """When cancellation arrives after final persistence transaction committed, job remains completed."""
        parser = MagicMock(spec=PDFParser)
        parser.parse.return_value = sample_parsed_paper
        storage = MagicMock(spec=CloudinaryStorage)

        paper = Paper(id=uuid.uuid4(), title="Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")
        job = IngestionJob(id=uuid.uuid4(), paper_id=paper.id, status="processing")

        paper_service_inst = AsyncMock(spec=PaperService)
        paper_service_inst.resolve_or_create_paper.return_value = PaperResolutionResult(
            paper=paper, matching_file=paper_file, paper_reused=True, file_reused=True
        )

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.lock_paper.return_value = paper
        ingestion_repo_inst.get_active_job_for_paper.return_value = None
        ingestion_repo_inst.get_latest_completed_job.return_value = None
        ingestion_repo_inst.create_job.return_value = job
        ingestion_repo_inst.bulk_create_chunks.return_value = [MagicMock()]
        ingestion_repo_inst.bulk_create_embeddings.return_value = [MagicMock()]
        ingestion_repo_inst.activate_chunks_for_job.return_value = [MagicMock()]

        def update_job_status_side_effect(job_id: Any = None, status: str = "completed", **kwargs: Any) -> IngestionJob:
            job.status = status
            if "current_stage" in kwargs:
                job.current_stage = kwargs["current_stage"]
            return job

        ingestion_repo_inst.update_job_status.side_effect = update_job_status_side_effect
        ingestion_repo_inst.get_job.return_value = job
        ingestion_repo_inst.cancel_job_if_active.return_value = job

        # Simulate cancellation immediately after persistence transaction commits
        tx_counter = 0
        tx_cm = AsyncMock()
        tx_cm.__aenter__.return_value = mock_session

        async def tx_aexit(*args: Any) -> None:
            nonlocal tx_counter
            tx_counter += 1
            # If update_job_status has set status="completed", this transaction just committed successfully
            if job.status == "completed":
                raise asyncio.CancelledError("Cancelled at persistence transaction commit boundary")
            return None

        tx_cm.__aexit__.side_effect = tx_aexit
        mock_session.begin.return_value = tx_cm

        session_factory = MagicMock(spec=async_sessionmaker)
        session_cm = AsyncMock()
        session_cm.__aenter__.return_value = mock_session
        session_cm.__aexit__.return_value = None
        session_factory.return_value = session_cm

        with patch("app.modules.ingestion.service.PaperRepository"), \
             patch("app.modules.ingestion.service.PaperService", return_value=paper_service_inst), \
             patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):

            preprocessor = MagicMock(spec=ResearchTextPreprocessor)
            preprocessor.preprocess.return_value = sample_preprocessed_paper
            section_detector = MagicMock(spec=ResearchSectionDetector)
            section_detector.detect.return_value = sample_section_result
            chunker = MagicMock(spec=ResearchChunkingEngine)
            chunker.config = ChunkingConfig()
            chunker.chunk.return_value = sample_chunking_result
            embedder = MagicMock(spec=ResearchEmbeddingGenerator)
            embedder.config = EmbeddingConfig()
            embedder.embed_chunks.return_value = sample_embedding_result

            service = IngestionService(
                session_factory=session_factory,
                storage=storage,
                parser=parser,
                preprocessor=preprocessor,
                section_detector=section_detector,
                chunker=chunker,
                embedder=embedder,
            )

            # Cancellation must propagate to caller
            with pytest.raises(asyncio.CancelledError):
                await service.ingest_pdf(sample_pdf_bytes, "paper.pdf")

            # Persisted job must remain completed!
            assert job.status == "completed"

            # cancel_job_if_active was called during cancellation
            ingestion_repo_inst.cancel_job_if_active.assert_awaited_once()

            # update_job_status must NOT have been called with status="cancelled"
            cancelled_calls = [
                c for c in ingestion_repo_inst.update_job_status.call_args_list
                if c[1].get("status") == "cancelled"
            ]
            assert len(cancelled_calls) == 0

    @pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
    @pytest.mark.asyncio
    async def test_cancellation_does_not_overwrite_terminal_job(
        self,
        mock_session_factory: MagicMock,
        terminal_status: str,
    ) -> None:
        """_record_job_cancellation never overwrites an existing terminal job status."""
        service = IngestionService(session_factory=mock_session_factory)
        job_id = uuid.uuid4()
        terminal_job = IngestionJob(id=job_id, paper_id=uuid.uuid4(), status=terminal_status)

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.cancel_job_if_active.return_value = terminal_job

        with patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):
            await service._record_job_cancellation(job_id, "test_cancelled", "Task cancelled")

            ingestion_repo_inst.cancel_job_if_active.assert_awaited_once()
            call_kwargs = ingestion_repo_inst.cancel_job_if_active.call_args[1]
            assert call_kwargs["job_id"] == job_id
            assert call_kwargs["current_stage"] == "test_cancelled"
            assert call_kwargs["error_message"] == "Task cancelled"
            assert call_kwargs["completed_at"] is not None
            ingestion_repo_inst.update_job_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_job_cancellation_calls_cancel_job_if_active_and_not_update_job_status(
        self,
        mock_session_factory: MagicMock,
    ) -> None:
        """Verify _record_job_cancellation delegates directly to cancel_job_if_active without update_job_status."""
        service = IngestionService(session_factory=mock_session_factory)
        job_id = uuid.uuid4()
        active_job = IngestionJob(id=job_id, paper_id=uuid.uuid4(), status="cancelled")

        ingestion_repo_inst = AsyncMock(spec=IngestionRepository)
        ingestion_repo_inst.cancel_job_if_active.return_value = active_job

        with patch("app.modules.ingestion.service.IngestionRepository", return_value=ingestion_repo_inst):
            await service._record_job_cancellation(job_id, "parsing_cancelled", "User abort")

            ingestion_repo_inst.cancel_job_if_active.assert_awaited_once()
            call_kwargs = ingestion_repo_inst.cancel_job_if_active.call_args[1]
            assert call_kwargs["job_id"] == job_id
            assert call_kwargs["current_stage"] == "parsing_cancelled"
            assert call_kwargs["error_message"] == "User abort"
            assert call_kwargs["completed_at"] is not None
            ingestion_repo_inst.update_job_status.assert_not_called()
            ingestion_repo_inst.get_job.assert_not_called()
