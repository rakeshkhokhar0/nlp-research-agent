"""Real-PDF End-to-End (E2E) Integration Test for Document Ingestion Pipeline.

Executes the complete IngestionService workflow using a real academic research-paper PDF
against a live, isolated PostgreSQL test database, Cloudinary storage, real SentenceTransformer
embedding inference, and lexical RetrievalService search.

Strict isolation and safety requirements:
- Explicit opt-in via RUN_INGESTION_E2E=1 (or RUN_E2E=1).
- Requires TEST_DB_URL (or test_db_url) pointing to an isolated test database.
- Requires DATABASE_URL for comparison and safety validation.
- Rejects application database; fails closed if URLs match or database paths collide.
- Never falls back to AsyncSessionLocal.
- Verifies that pgvector extension and required ingestion tables exist; does not use create_all().
- Real SentenceTransformer model inference (sentence-transformers/all-MiniLM-L6-v2).
- Clean teardown strictly scoped to records created by this test via unique test DOI ownership.
- Never deletes shared EmbeddingModel records or pre-existing records.
- Deletes Cloudinary asset only if positively confirmed to have been newly uploaded by this test.
- Engine is guaranteed to be disposed in all error and exit paths.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
import socket
from typing import Any
import uuid

from dotenv import load_dotenv
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Load environment variables from server/.env if present
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

import app.database.model_registry  # noqa: F401 - ensures all relationships are registered
from app.core.config import cloudinary_settings, configure_cloudinary
from app.database.model_registry import (
    Base,
    ChunkEmbedding,
    EmbeddingModel,
    IngestionJob,
    Paper,
    PaperChunk,
    PaperFile,
)
from app.modules.ingestion.chunker import ResearchChunk, ResearchChunkingEngine
from app.modules.ingestion.embedder import ResearchEmbeddingGenerator
from app.modules.ingestion.parser import PDFParser
from app.modules.ingestion.preprocessor import ResearchTextPreprocessor
from app.modules.ingestion.section_detector import ResearchSectionDetector
from app.modules.ingestion.service import IngestionResult, IngestionService
from app.modules.ingestion.storage import CloudinaryStorage
from app.modules.retrieval import RetrievalService
from tests.retrieval.test_retrieval_integration import (
    mask_url,
    verify_test_database_safety,
)

logger = logging.getLogger(__name__)

# Path to the real research paper PDF fixture
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_paper.pdf"

# Opt-in environment variable flag
OPT_IN_ENV_VARS = ("RUN_INGESTION_E2E", "RUN_E2E", "ENABLE_INGESTION_E2E")

# Required tables for document ingestion workflow (created via migrations)
REQUIRED_INGESTION_TABLES: frozenset[str] = frozenset({
    "papers",
    "paper_files",
    "ingestion_jobs",
    "paper_chunks",
    "chunk_embeddings",
    "embedding_models",
})


def is_e2e_enabled() -> bool:
    """Return True if an explicit opt-in environment variable is set to a truthy value."""
    for var in OPT_IN_ENV_VARS:
        val = os.getenv(var, "").strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
    return False


def get_test_db_url() -> str | None:
    """Retrieve the configured test database connection URL."""
    return (
        os.getenv("TEST_DATABASE_URL")
        or os.getenv("test_database_url")
        or os.getenv("TEST_DB_URL")
        or os.getenv("test_db_url")
    )


def get_app_db_url() -> str | None:
    """Retrieve the application/production database connection URL."""
    return os.getenv("DATABASE_URL") or os.getenv("database_url")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_real_pdf_ingestion_end_to_end() -> None:
    """Execute and verify end-to-end ingestion of a real research paper PDF."""
    # -------------------------------------------------------------------------
    # 1. Opt-In & Prerequisites Preflight Checks
    # -------------------------------------------------------------------------
    if not is_e2e_enabled():
        pytest.skip(
            f"Real-PDF E2E ingestion test is skipped. Opt-in required: set one of {OPT_IN_ENV_VARS} to 1/true."
        )

    test_db_url = get_test_db_url()
    app_db_url = get_app_db_url()

    if not test_db_url:
        pytest.skip("TEST_DB_URL or test_db_url not set. Skipping live E2E test.")

    if not app_db_url:
        pytest.skip("DATABASE_URL not set for comparison. Skipping live E2E test.")

    # 1.1 Strict Database Safety Verification (isolation from production)
    try:
        verify_test_database_safety(test_db_url, app_db_url)
    except Exception as exc:
        pytest.fail(f"Database safety validation failed: {exc}")

    # 1.2 Cloudinary Credentials Verification
    cloud_name = cloudinary_settings.cloudinary_cloud_name
    api_key = cloudinary_settings.cloudinary_api_key.get_secret_value()
    api_secret = cloudinary_settings.cloudinary_api_secret.get_secret_value()

    if not cloud_name or not api_key or not api_secret:
        pytest.skip(
            "Cloudinary credentials (CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET) "
            "are incomplete. Skipping live Cloudinary E2E test."
        )

    # Configure Cloudinary SDK globally
    configure_cloudinary(cloudinary_settings)

    # 1.3 PDF Fixture Verification
    if not FIXTURE_PATH.exists():
        pytest.fail(
            f"Real PDF fixture not found at {FIXTURE_PATH}. "
            "Please ensure sample_paper.pdf exists before running E2E tests."
        )

    pdf_bytes = FIXTURE_PATH.read_bytes()
    if len(pdf_bytes) == 0 or not pdf_bytes.startswith(b"%PDF-"):
        pytest.fail(f"Fixture at {FIXTURE_PATH} is empty or not a valid PDF file.")

    expected_sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    # 1.4 Embedding Model Preflight & Real Inference Check
    embedder = ResearchEmbeddingGenerator()
    try:
        content = "Preflight test chunk for real sentence-transformers inference."
        preflight_chunk = ResearchChunk(
            chunk_index=0,
            content=content,
            token_count=len(content.split()),
            char_count=len(content),
            page_start=1,
            page_end=1,
            section_title="Preflight",
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            metadata={"test": "preflight"},
        )
        preflight_res = embedder.embed_chunks([preflight_chunk])
        assert len(preflight_res.embeddings) == 1
        assert len(preflight_res.embeddings[0].vector) == embedder.config.dimensions
        assert all(
            isinstance(v, float) and not math.isnan(v) and not math.isinf(v)
            for v in preflight_res.embeddings[0].vector
        )
    except Exception as exc:
        pytest.fail(
            f"Embedding model preflight failed (model '{embedder.config.model_name}' unavailable "
            f"or inference error): {exc}"
        )

    # -------------------------------------------------------------------------
    # 2. Engine & Session Factory Initialization
    # -------------------------------------------------------------------------
    # Initialize engine and session_factory BEFORE entering try/finally to ensure
    # test_session_factory is always defined if teardown logic is reached.
    test_engine = create_async_engine(test_db_url, pool_pre_ping=True)
    test_session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Generate unique test-run identifier for strict, verifiable ownership
    test_run_id = uuid.uuid4().hex[:12]
    test_doi = f"10.9999/e2e-{test_run_id}"

    created_paper_id: uuid.UUID | None = None
    created_job_id: uuid.UUID | None = None
    created_file_id: uuid.UUID | None = None
    asset_cleanup_eligible: bool = False

    try:
        # 2.1 Verify connection and that required ingestion tables exist
        try:
            async with test_engine.begin() as conn:
                # Verify pgvector extension
                res_vec = await conn.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                )
                if res_vec.scalar_one_or_none() != "vector":
                    pytest.fail(
                        f"The pgvector extension is not installed in the test database ({mask_url(test_db_url)}). "
                        "Please run 'CREATE EXTENSION IF NOT EXISTS vector;' or apply migrations first."
                    )

                # Verify required ingestion tables exist without schema-changing DDL or create_all()
                res_tables = await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public';"
                    )
                )
                existing_tables = {row[0].lower() for row in res_tables.fetchall()}
                missing_tables = REQUIRED_INGESTION_TABLES - existing_tables
                if missing_tables:
                    pytest.fail(
                        f"The isolated test database at {mask_url(test_db_url)} is missing required "
                        f"ingestion tables: {sorted(missing_tables)}. "
                        "Please apply database migrations to the test database first (e.g. 'alembic upgrade head')."
                    )
        except (OSError, socket.gaierror, ConnectionRefusedError) as conn_err:
            pytest.skip(
                f"Isolated test database is unreachable ({type(conn_err).__name__}: {conn_err}). "
                "Skipping live PostgreSQL E2E test."
            )

        # 2.2 Pre-run Check: Verify whether any matching Cloudinary asset pre-exists.
        # Set asset_cleanup_eligible = True ONLY when the query succeeds and confirms
        # that no matching record existed before this test.
        try:
            async with test_session_factory() as pre_session:
                stmt_chk = (
                    select(PaperFile.id)
                    .where(
                        PaperFile.checksum_sha256 == expected_sha256,
                        PaperFile.storage_provider == "cloudinary",
                    )
                    .limit(1)
                )
                res_chk = await pre_session.execute(stmt_chk)
                pre_existing_match = res_chk.scalars().first()
                if pre_existing_match is None:
                    asset_cleanup_eligible = True
                else:
                    logger.info(
                        "Pre-existing Cloudinary PaperFile found (%s); marking asset_cleanup_eligible=False.",
                        pre_existing_match,
                    )
        except Exception as chk_err:
            logger.error(
                "Pre-run checksum existence check failed: %s; leaving asset_cleanup_eligible=False.",
                chk_err,
                exc_info=True,
            )

        # ---------------------------------------------------------------------
        # 3. Construct IngestionService and Dependencies
        # ---------------------------------------------------------------------
        storage = CloudinaryStorage()
        parser = PDFParser()
        preprocessor = ResearchTextPreprocessor()
        section_detector = ResearchSectionDetector()
        chunker = ResearchChunkingEngine()
        retrieval_service = RetrievalService()

        service = IngestionService(
            session_factory=test_session_factory,
            storage=storage,
            parser=parser,
            preprocessor=preprocessor,
            section_detector=section_detector,
            chunker=chunker,
            embedder=embedder,
            retrieval_service=retrieval_service,
        )

        # ---------------------------------------------------------------------
        # 4. Execute End-to-End Ingestion
        # ---------------------------------------------------------------------
        # Pass unique test_doi for verifiable record ownership
        result: IngestionResult = await service.ingest_pdf(
            pdf_bytes,
            "sample_paper.pdf",
            doi=test_doi,
        )

        # Capture record IDs immediately upon return
        created_paper_id = result.paper_id
        created_job_id = result.ingestion_job_id
        created_file_id = result.paper_file_id

        # ---------------------------------------------------------------------
        # 5. Assert Ingestion Result Contract
        # ---------------------------------------------------------------------
        assert result.status == "completed", f"Expected completed status, got {result.status}"
        assert result.paper_id is not None
        assert isinstance(result.paper_id, uuid.UUID)
        assert result.paper_file_id is not None
        assert isinstance(result.paper_file_id, uuid.UUID)
        assert result.ingestion_job_id is not None
        assert isinstance(result.ingestion_job_id, uuid.UUID)
        assert result.chunks_created > 0, "Expected chunks_created > 0"
        assert result.embeddings_created == result.chunks_created, (
            f"Embedding count ({result.embeddings_created}) does not match chunk count ({result.chunks_created})"
        )
        assert result.job_reused is False, "Expected newly executed job, not reused job"

        # ---------------------------------------------------------------------
        # 6. Database Verification
        # ---------------------------------------------------------------------
        async with test_session_factory() as db_session:
            # 6.1 Paper record
            stmt_paper = select(Paper).where(Paper.id == created_paper_id)
            res_paper = await db_session.execute(stmt_paper)
            paper_row = res_paper.scalar_one_or_none()
            assert paper_row is not None, f"Paper row {created_paper_id} not found in database"
            assert paper_row.title is not None and len(paper_row.title.strip()) > 0
            assert paper_row.doi == test_doi, f"Expected DOI {test_doi}, got {paper_row.doi}"

            # 6.2 PaperFile record - verified against actual PaperFile model attributes
            stmt_file = select(PaperFile).where(PaperFile.id == created_file_id)
            res_file = await db_session.execute(stmt_file)
            file_row = res_file.scalar_one_or_none()
            assert file_row is not None, f"PaperFile row {created_file_id} not found"
            assert file_row.paper_id == created_paper_id
            assert file_row.storage_provider == "cloudinary"
            assert file_row.storage_key is not None and len(file_row.storage_key.strip()) > 0
            assert file_row.file_url is not None and len(file_row.file_url.strip()) > 0
            assert file_row.checksum_sha256 == expected_sha256
            assert file_row.file_size_bytes == len(pdf_bytes)

            uploaded_storage_key = file_row.storage_key

            # 6.3 IngestionJob record
            stmt_job = select(IngestionJob).where(IngestionJob.id == created_job_id)
            res_job = await db_session.execute(stmt_job)
            job_row = res_job.scalar_one_or_none()
            assert job_row is not None, f"IngestionJob row {created_job_id} not found"
            assert job_row.paper_id == created_paper_id
            assert job_row.paper_file_id == file_row.id
            assert job_row.status == "completed"
            assert job_row.current_stage == "completed"
            assert job_row.started_at is not None
            assert job_row.completed_at is not None
            assert job_row.completed_at >= job_row.started_at
            assert "pipeline_fingerprint" in job_row.metadata_

            # 6.4 EmbeddingModel record (shared; not test-owned)
            stmt_model = select(EmbeddingModel).where(
                EmbeddingModel.model_name == embedder.config.model_name,
                EmbeddingModel.provider == embedder.config.provider,
            )
            res_model = await db_session.execute(stmt_model)
            model_row = res_model.scalar_one_or_none()
            assert model_row is not None, "EmbeddingModel record not found"
            assert model_row.dimensions == 384
            assert model_row.distance_metric == "cosine"
            assert model_row.is_active is True

            # 6.5 PaperChunk records
            stmt_chunks = (
                select(PaperChunk)
                .where(PaperChunk.ingestion_job_id == created_job_id)
                .order_by(PaperChunk.chunk_index.asc())
            )
            res_chunks = await db_session.execute(stmt_chunks)
            chunk_rows = list(res_chunks.scalars().all())
            assert len(chunk_rows) == result.chunks_created
            assert all(c.is_current is True for c in chunk_rows)
            assert all(c.paper_id == created_paper_id for c in chunk_rows)
            assert all(len(c.content.strip()) > 0 for c in chunk_rows)
            assert all(c.content_hash is not None and len(c.content_hash) == 64 for c in chunk_rows)
            assert all(c.page_start is not None and c.page_start > 0 for c in chunk_rows)
            assert all(c.page_end is not None and c.page_end >= c.page_start for c in chunk_rows)

            # 6.6 ChunkEmbedding records
            stmt_embeddings = (
                select(ChunkEmbedding)
                .join(PaperChunk, ChunkEmbedding.chunk_id == PaperChunk.id)
                .where(PaperChunk.ingestion_job_id == created_job_id)
            )
            res_embeddings = await db_session.execute(stmt_embeddings)
            embedding_rows = list(res_embeddings.scalars().all())
            assert len(embedding_rows) == result.embeddings_created
            for emb in embedding_rows:
                assert emb.embedding_model_id == model_row.id
                vec = list(emb.embedding)
                assert len(vec) == 384
                assert all(
                    isinstance(val, float) and not math.isnan(val) and not math.isinf(val)
                    for val in vec
                )

        # ---------------------------------------------------------------------
        # 7. Cloudinary Asset Verification
        # ---------------------------------------------------------------------
        assert uploaded_storage_key is not None
        # Verify asset metadata via Cloudinary API
        cloudinary_info = storage.get_file_info(uploaded_storage_key)
        assert cloudinary_info is not None
        assert "public_id" in cloudinary_info or "bytes" in cloudinary_info

        # Verify download and byte integrity
        downloaded_bytes = storage.download_file(uploaded_storage_key)
        assert len(downloaded_bytes) == len(pdf_bytes)
        downloaded_sha256 = hashlib.sha256(downloaded_bytes).hexdigest()
        assert downloaded_sha256 == expected_sha256, (
            f"Cloudinary downloaded SHA-256 ({downloaded_sha256}) mismatch with expected ({expected_sha256})"
        )
        # Validate downloaded PDF header
        storage.validate_pdf(downloaded_bytes)

        # ---------------------------------------------------------------------
        # 8. Retrieval Verification
        # ---------------------------------------------------------------------
        assert retrieval_service.is_initialized, "RetrievalService should have been refreshed in Stage G"

        # Search for academic concepts specific to the Word2Vec paper
        search_query = "continuous bag of words skip-gram vector space representation"
        retrieved_chunks = retrieval_service.search(search_query, top_k=5)

        assert len(retrieved_chunks) > 0, f"Expected search results for query '{search_query}', got none"
        # Confirm returned chunks belong to the ingested paper
        assert any(c.paper_id == created_paper_id for c in retrieved_chunks), (
            f"None of the retrieved chunks belong to ingested paper {created_paper_id}"
        )
        # Confirm scores are strictly positive
        for r_chunk in retrieved_chunks:
            assert r_chunk.score > 0.0
            assert r_chunk.content is not None and len(r_chunk.content) > 0

    finally:
        # ---------------------------------------------------------------------
        # 9. Scoped Teardown & Resource Cleanup
        # Nested try/finally ensures:
        # 1. Cloudinary asset cleanup is attempted.
        # 2. Database cleanup is attempted regardless of Cloudinary cleanup outcome.
        # 3. test_engine.dispose() is GUARANTEED to execute even if cleanup fails.
        # 4. Teardown exceptions are logged and never mask the original test exception.
        # ---------------------------------------------------------------------
        try:
            # 9.1 Lookup test-owned Paper by unique test_doi if not captured
            resolved_paper_id: uuid.UUID | None = created_paper_id
            if resolved_paper_id is None:
                try:
                    async with test_session_factory() as lookup_session:
                        stmt_doi = select(Paper.id).where(Paper.doi == test_doi)
                        res_doi = await lookup_session.execute(stmt_doi)
                        resolved_paper_id = res_doi.scalar_one_or_none()
                except Exception as lookup_err:
                    logger.warning(
                        "Failed to lookup test paper by unique DOI %s during cleanup: %s",
                        test_doi,
                        lookup_err,
                    )

            # 9.2 Cloudinary Cleanup: positive ownership and non-reused confirmation
            # Requires asset_cleanup_eligible to be True, guaranteeing that:
            # - Pre-run check succeeded with no errors.
            # - No matching asset pre-existed in the database.
            # In addition, positively verify:
            # - test-owned Paper resolved (resolved_paper_id is not None)
            # - associated PaperFile belongs to test-owned Paper
            # - storage_provider == "cloudinary"
            # - checksum_sha256 matches expected SHA-256
            # - storage_key is non-empty
            storage_key_to_delete: str | None = None
            if resolved_paper_id is not None and asset_cleanup_eligible:
                try:
                    async with test_session_factory() as pf_session:
                        stmt_pf = select(PaperFile).where(PaperFile.paper_id == resolved_paper_id)
                        res_pf = await pf_session.execute(stmt_pf)
                        pf_rows = list(res_pf.scalars().all())
                        for pf in pf_rows:
                            if (
                                pf.paper_id == resolved_paper_id
                                and pf.storage_provider == "cloudinary"
                                and pf.checksum_sha256 == expected_sha256
                                and pf.storage_key
                                and pf.storage_key.strip()
                            ):
                                storage_key_to_delete = pf.storage_key.strip()
                                break
                except Exception as pf_lookup_err:
                    logger.warning("Failed to lookup PaperFile for Cloudinary cleanup: %s", pf_lookup_err)

            if storage_key_to_delete is not None:
                try:
                    storage = CloudinaryStorage()
                    storage.delete_file(storage_key_to_delete)
                    logger.info("Deleted verified test Cloudinary asset: %s", storage_key_to_delete)
                except Exception as cld_cleanup_err:
                    logger.error(
                        "Cloudinary cleanup failed for asset %s: %s",
                        storage_key_to_delete,
                        cld_cleanup_err,
                        exc_info=True,
                    )
            else:
                logger.info(
                    "Cloudinary cleanup skipped: no test-owned, newly uploaded asset confirmed "
                    "(asset_cleanup_eligible=%s, DOI=%s)",
                    asset_cleanup_eligible,
                    test_doi,
                )

            # 9.3 Scoped Database Cleanup: delete strictly test-owned paper and dependents
            if resolved_paper_id is not None:
                try:
                    async with test_session_factory() as cleanup_session:
                        async with cleanup_session.begin():
                            # Delete in strict dependency order:
                            # 1. ChunkEmbedding for this paper's chunks
                            await cleanup_session.execute(
                                text(
                                    "DELETE FROM chunk_embeddings WHERE chunk_id IN ("
                                    "  SELECT id FROM paper_chunks WHERE paper_id = :pid"
                                    ")"
                                ),
                                {"pid": resolved_paper_id},
                            )
                            # 2. PaperChunk
                            await cleanup_session.execute(
                                text("DELETE FROM paper_chunks WHERE paper_id = :pid"),
                                {"pid": resolved_paper_id},
                            )
                            # 3. IngestionJob
                            await cleanup_session.execute(
                                text("DELETE FROM ingestion_jobs WHERE paper_id = :pid"),
                                {"pid": resolved_paper_id},
                            )
                            # 4. PaperFile
                            await cleanup_session.execute(
                                text("DELETE FROM paper_files WHERE paper_id = :pid"),
                                {"pid": resolved_paper_id},
                            )
                            # 5. Paper (test-owned)
                            await cleanup_session.execute(
                                text("DELETE FROM papers WHERE id = :pid"),
                                {"pid": resolved_paper_id},
                            )
                            logger.info("Scoped database cleanup succeeded for test paper %s", resolved_paper_id)
                except Exception as db_cleanup_err:
                    logger.error(
                        "Database cleanup failed for test paper %s: %s",
                        resolved_paper_id,
                        db_cleanup_err,
                        exc_info=True,
                    )
            else:
                logger.info("No test-owned records found for DOI %s; skipping database cleanup.", test_doi)

        finally:
            # 9.4 Engine disposal is guaranteed to execute even if any prior teardown step failed
            try:
                await test_engine.dispose()
            except Exception as disp_err:
                logger.warning("Error disposing test engine: %s", disp_err, exc_info=True)
