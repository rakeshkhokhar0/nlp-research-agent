"""PostgreSQL integration tests for the retrieval module.

Tests the full retrieval workflow against a live, isolated PostgreSQL database:
- RetrievalRepository fetching active chunks with eager-loaded Paper metadata
- Versioning and historical chunk exclusions
- RetrievalService refresh and end-to-end search after database session is closed
- Empty database initialization and search
- Non-matching query behavior
- Dynamic re-indexing after database mutations
- Atomic failure recovery preserving previously working index

DATABASE SAFETY REQUIREMENTS:
- Requires an explicitly configured TEST_DATABASE_URL or TEST_DB_URL.
- Never falls back to DATABASE_URL.
- Skips safely if no test URL is configured or if unreachable.
- Fails closed if database URLs cannot be parsed safely.
- Rejects test URL if its database path/name matches the application database name,
  even across different hosts/endpoints.
- Never prints full database credentials or unmasked URLs in logs or error messages.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
from typing import Sequence
from urllib.parse import urlparse
import uuid

from dotenv import load_dotenv
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Ensure environment variables from server/.env are loaded
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

import app.database.model_registry  # noqa: F401 - ensures cross-module relationship mappers are registered
from app.database.model_registry import (
    Base,
    IngestionJob,
    Paper,
    PaperChunk,
)
from app.modules.retrieval import (
    ChunkIDMapper,
    RetrievalRepository,
    RetrievalService,
    RetrievedChunk,
)
from app.modules.retrieval.lexical.engine import LexicalRetrievalEngine

TEST_DB_URL = (
    os.getenv("TEST_DATABASE_URL")
    or os.getenv("test_database_url")
    or os.getenv("TEST_DB_URL")
    or os.getenv("test_db_url")
)

APP_DB_URL = os.getenv("DATABASE_URL") or os.getenv("database_url")


SUPPORTED_PG_SCHEMES = {
    "postgresql",
    "postgresql+asyncpg",
    "postgres",
    "postgresql+psycopg",
    "postgresql+psycopg2",
}


def mask_url(url: str | None) -> str:
    """Safely mask all credentials (both username and password) in a database URL.

    Preserves safe scheme, host, port, and database-path information.
    """
    if not url:
        return "<unset>"
    try:
        parsed = urlparse(url.strip())
        host = parsed.hostname or ""
        if parsed.port:
            host_port = f"{host}:{parsed.port}"
        else:
            host_port = host
        return f"{parsed.scheme}://***:***@{host_port}{parsed.path}"
    except Exception:
        return "<invalid url>"


def validate_postgres_url(url_str: str, label: str) -> tuple[str, str, int | None, str]:
    """Validate that a URL has a supported PostgreSQL scheme, usable hostname, and non-empty database path.

    Returns:
        tuple of (scheme, hostname, port, database_name)

    Raises:
        ValueError: If scheme is unsupported, netloc has no hostname, or database path is missing.
    """
    if not url_str or not url_str.strip():
        raise ValueError(f"{label} is not configured or empty.")

    try:
        p = urlparse(url_str.strip())
    except Exception as exc:
        raise ValueError(f"{label} could not be parsed: {exc}") from exc

    if not p.scheme or p.scheme.lower() not in SUPPORTED_PG_SCHEMES:
        raise ValueError(
            f"{label} has unsupported scheme '{p.scheme}'. Must be a supported PostgreSQL scheme."
        )

    if not p.netloc or not p.netloc.strip():
        raise ValueError(f"{label} missing netloc/host authority.")

    if not p.hostname or not p.hostname.strip():
        raise ValueError(f"{label} missing usable hostname.")

    dbname = p.path.strip("/")
    if not dbname:
        raise ValueError(f"{label} missing non-empty database path.")

    return p.scheme.lower(), p.hostname.lower(), p.port, dbname


def verify_test_database_safety(test_url: str | None, app_url: str | None) -> None:
    """Verify that the test database is explicitly configured and completely isolated from the application database.

    Strengthened isolation rules:
    - Never falls back to application database.
    - Requires both test database URL and application database URL to be explicitly available.
    - Validates that both URLs have a supported PostgreSQL scheme, usable hostname, and non-empty database path.
    - Rejects identical URLs.
    - Fails closed if either URL cannot be parsed safely or fails validation.
    - Rejects a test URL when its database path/name matches the application database
      path/name, even if the hostnames differ (preventing accidental targeting of production
      clusters via separate endpoints).
    - Masks all credentials in all error messages and never prints full unmasked URLs.

    Raises:
        ValueError: If test_url or app_url is None or empty.
        RuntimeError: If either URL cannot be parsed safely, has an unsupported scheme,
            lacks a usable hostname, if URLs match, or if the database name in the test
            URL matches the application database name.
    """
    if not test_url or not test_url.strip():
        raise ValueError(
            "TEST_DATABASE_URL or TEST_DB_URL is not configured. "
            "Integration tests must not run without a dedicated, isolated test database."
        )

    if not app_url or not app_url.strip():
        raise ValueError(
            "DATABASE_URL (application database URL) is not configured. "
            "Safety verification requires both test and application URLs to verify isolation."
        )

    if test_url.strip() == app_url.strip():
        raise RuntimeError(
            f"DATABASE SAFETY GUARD: Configured test URL matches application DATABASE_URL "
            f"({mask_url(app_url)}). DDL operations are strictly forbidden on the application database."
        )

    try:
        test_scheme, test_host, test_port, test_dbname = validate_postgres_url(test_url, "Test database URL")
        app_scheme, app_host, app_port, app_dbname = validate_postgres_url(app_url, "Application database URL")
    except ValueError as val_err:
        raise RuntimeError(
            f"DATABASE SAFETY GUARD: Failed to safely parse database URL: {val_err}"
        ) from val_err
    except Exception as exc:
        # Fail closed
        raise RuntimeError(
            "DATABASE SAFETY GUARD: Failed to safely parse database URLs. "
            "Failing closed to prevent destructive operations."
        ) from exc

    # Reject when database path/name matches, EVEN IF hostnames differ!
    if test_dbname.lower() == app_dbname.lower():
        raise RuntimeError(
            f"DATABASE SAFETY GUARD: Configured test database name '{test_dbname}' matches "
            f"application database name '{app_dbname}' ({mask_url(app_url)}). "
            "Different hostnames alone do not guarantee database isolation on hosted providers like Neon. "
            "The test database must use a distinct, isolated database name."
        )


@pytest_asyncio.fixture
async def db_engine():
    """Create async engine for the test database, verify safety, and ensure schema is present."""
    verify_test_database_safety(TEST_DB_URL, APP_DB_URL)
    engine = create_async_engine(TEST_DB_URL, pool_pre_ping=True)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)
    except (OSError, socket.gaierror, ConnectionRefusedError) as exc:
        await engine.dispose()
        pytest.skip(
            f"Test database is unreachable ({type(exc).__name__}). Skipping live PostgreSQL tests."
        )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(db_engine):
    """Return async sessionmaker for the test database."""
    return async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def db_session(session_factory):
    """Provide a transactional database session for each test, ensuring clean state before and after."""
    async with session_factory() as session:
        await session.execute(text("TRUNCATE TABLE papers, ingestion_jobs, paper_chunks CASCADE;"))
        await session.commit()
        yield session
        try:
            if session.is_active:
                await session.execute(text("TRUNCATE TABLE papers, ingestion_jobs, paper_chunks CASCADE;"))
                await session.commit()
            else:
                async with session_factory() as cleanup_session:
                    await cleanup_session.execute(text("TRUNCATE TABLE papers, ingestion_jobs, paper_chunks CASCADE;"))
                    await cleanup_session.commit()
        except Exception:
            pass


async def create_paper(
    session: AsyncSession,
    title: str = "Attention Is All You Need",
    publication_year: int | None = 2017,
    abstract: str | None = "Transformer architecture overview.",
) -> Paper:
    """Helper to insert a valid Paper record."""
    paper = Paper(
        title=title,
        publication_year=publication_year,
        abstract=abstract,
    )
    session.add(paper)
    await session.flush()
    return paper


async def create_job(
    session: AsyncSession,
    paper_id: uuid.UUID,
    status: str = "completed",
) -> IngestionJob:
    """Helper to insert an IngestionJob satisfying composite foreign key constraints."""
    job = IngestionJob(
        paper_id=paper_id,
        status=status,
    )
    session.add(job)
    await session.flush()
    return job


async def create_chunk(
    session: AsyncSession,
    paper_id: uuid.UUID,
    job_id: uuid.UUID,
    chunk_index: int,
    content: str,
    is_current: bool = True,
) -> PaperChunk:
    """Helper to insert a PaperChunk record."""
    chunk = PaperChunk(
        paper_id=paper_id,
        ingestion_job_id=job_id,
        chunk_index=chunk_index,
        content=content,
        is_current=is_current,
    )
    session.add(chunk)
    await session.flush()
    return chunk


class TestDatabaseSafetyGuards:
    """Unit tests for the database isolation safety guards.

    These execute independently without requiring TEST_DATABASE_URL to be configured.
    """

    def test_verify_test_database_safety_rejects_empty_test_url(self) -> None:
        """Empty or None test database URL must raise ValueError."""
        with pytest.raises(ValueError, match="TEST_DATABASE_URL or TEST_DB_URL is not configured"):
            verify_test_database_safety("", "postgresql+asyncpg://user:pass@app-db:5432/nlp")

        with pytest.raises(ValueError, match="TEST_DATABASE_URL or TEST_DB_URL is not configured"):
            verify_test_database_safety(None, "postgresql+asyncpg://user:pass@app-db:5432/nlp")

    def test_verify_test_database_safety_rejects_empty_app_url(self) -> None:
        """Empty or None application database URL must raise ValueError."""
        with pytest.raises(ValueError, match="DATABASE_URL .* is not configured"):
            verify_test_database_safety("postgresql+asyncpg://user:pass@test-db:5432/test_nlp", None)

        with pytest.raises(ValueError, match="DATABASE_URL .* is not configured"):
            verify_test_database_safety("postgresql+asyncpg://user:pass@test-db:5432/test_nlp", "")

        with pytest.raises(ValueError, match="DATABASE_URL .* is not configured"):
            verify_test_database_safety("postgresql+asyncpg://user:pass@test-db:5432/test_nlp", "   ")

    def test_verify_test_database_safety_rejects_identical_urls(self) -> None:
        """Identical test and application URLs must trigger safety guard exception."""
        url = "postgresql+asyncpg://user:secret@neon-db.tech:5432/neondb"
        with pytest.raises(RuntimeError, match="matches application DATABASE_URL"):
            verify_test_database_safety(url, url)

    def test_verify_test_database_safety_rejects_identical_database_names_on_different_hosts(self) -> None:
        """Reject when database name matches application database name even if hosts differ."""
        url_app = "postgresql+asyncpg://app_user:pass1@prod-neon.tech:5432/neondb?ssl=require"
        url_test = "postgresql+asyncpg://test_user:pass2@test-neon.tech:5432/neondb?ssl=require"
        with pytest.raises(RuntimeError, match="matches application database name"):
            verify_test_database_safety(url_test, url_app)

    def test_verify_test_database_safety_accepts_different_database_names_on_same_host(self) -> None:
        """Accept distinct database names located on the same host."""
        url_app = "postgresql+asyncpg://app_user:pass1@neon-host.tech:5432/production_db?ssl=require"
        url_test = "postgresql+asyncpg://test_user:pass2@neon-host.tech:5432/isolated_test_db?ssl=require"
        # Must not raise
        verify_test_database_safety(url_test, url_app)

    def test_verify_test_database_safety_accepts_distinct_database_urls(self) -> None:
        """Different hosts and different database names must pass safety checks."""
        url_app = "postgresql+asyncpg://user:pass@app-neon.tech:5432/app_db"
        url_test = "postgresql+asyncpg://user:pass@test-neon.tech:5432/test_db"
        verify_test_database_safety(url_test, url_app)

    def test_verify_test_database_safety_fails_closed_on_malformed_url(self) -> None:
        """Unparseable or malformed URLs must fail closed."""
        url_app = "postgresql+asyncpg://user:pass@app-neon.tech:5432/app_db"
        with pytest.raises(RuntimeError, match="Failed to safely parse"):
            verify_test_database_safety("not a valid url", url_app)

    def test_verify_test_database_safety_fails_closed_on_missing_database_name(self) -> None:
        """URLs missing database name/path must fail closed."""
        url_app = "postgresql+asyncpg://user:pass@app-neon.tech:5432/app_db"
        with pytest.raises(RuntimeError, match="missing non-empty database path"):
            verify_test_database_safety("postgresql+asyncpg://user:pass@test-neon.tech:5432/", url_app)

    def test_verify_test_database_safety_rejects_netloc_without_hostname(self) -> None:
        """URLs with non-empty netloc authority but lacking a usable hostname must be rejected."""
        url_app = "postgresql+asyncpg://user:pass@app-neon.tech:5432/app_db"

        # e.g. Port only (netloc is ':5432', but hostname is None or empty)
        with pytest.raises(RuntimeError, match="missing usable hostname"):
            verify_test_database_safety("postgresql+asyncpg://:5432/test_db", url_app)

        # e.g. User and port only (netloc is 'user:pass@:5432', but hostname is empty)
        with pytest.raises(RuntimeError, match="missing usable hostname"):
            verify_test_database_safety("postgresql+asyncpg://user:pass@:5432/test_db", url_app)

    def test_verify_test_database_safety_rejects_unsupported_scheme(self) -> None:
        """URLs with non-PostgreSQL schemes must be rejected."""
        url_app = "postgresql+asyncpg://user:pass@app-neon.tech:5432/app_db"

        with pytest.raises(RuntimeError, match=r"unsupported scheme 'mysql\+aiomysql'"):
            verify_test_database_safety("mysql+aiomysql://user:pass@test-db:3306/test_db", url_app)

        with pytest.raises(RuntimeError, match=r"unsupported scheme 'http'"):
            verify_test_database_safety("http://test-db:5432/test_db", url_app)

        with pytest.raises(RuntimeError, match=r"unsupported scheme 'sqlite\+aiosqlite'"):
            verify_test_database_safety("sqlite+aiosqlite:///test.db", url_app)

    def test_mask_url_redacts_all_credentials(self) -> None:
        """Ensure both username and password are redacted, while host, port, and db path are preserved."""
        masked = mask_url("postgresql+asyncpg://myuser:supersecretpass@myhost:5432/mydb")
        assert "myuser" not in masked
        assert "supersecretpass" not in masked
        assert "myhost:5432/mydb" in masked
        assert masked == "postgresql+asyncpg://***:***@myhost:5432/mydb"


@pytest.mark.skipif(
    not TEST_DB_URL,
    reason="TEST_DATABASE_URL or TEST_DB_URL environment variable not provided. Skipping live PostgreSQL integration tests.",
)
class TestRetrievalPostgresIntegration:
    """Live PostgreSQL integration tests for the retrieval module."""

    @pytest_asyncio.fixture(autouse=True)
    async def cleanup_database(self, session_factory):
        """Ensure test tables are clean before and after each test, even if assertions fail."""
        async with session_factory() as session:
            await session.execute(text("TRUNCATE TABLE papers, ingestion_jobs, paper_chunks CASCADE;"))
            await session.commit()
        try:
            yield
        finally:
            async with session_factory() as cleanup_session:
                await cleanup_session.execute(text("TRUNCATE TABLE papers, ingestion_jobs, paper_chunks CASCADE;"))
                await cleanup_session.commit()

    @pytest.mark.asyncio
    async def test_scenario_a_fetch_current_chunks_from_postgres(
        self, db_session: AsyncSession
    ) -> None:
        """Scenario A: Fetch current chunks through RetrievalRepository.

        Verifies:
        - At least two papers with multiple chunks are retrieved.
        - Returned records contain chunk content and eager-loaded parent Paper metadata.
        - Deterministic ordering by paper_id ascending and chunk_index ascending.
        - Parent Paper relationship is accessible without initiating a new database query.
        - Optional paper_ids restriction filters chunks accurately.
        """
        # 1. Insert two papers
        paper_1 = await create_paper(
            db_session,
            title="Attention Is All You Need",
            publication_year=2017,
        )
        paper_2 = await create_paper(
            db_session,
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            publication_year=2018,
        )

        job_1 = await create_job(db_session, paper_id=paper_1.id)
        job_2 = await create_job(db_session, paper_id=paper_2.id)

        # 2. Insert chunks
        chunk_1_0 = await create_chunk(
            db_session,
            paper_id=paper_1.id,
            job_id=job_1.id,
            chunk_index=0,
            content="The Transformer is the first sequence transduction model based entirely on self-attention.",
            is_current=True,
        )
        chunk_1_1 = await create_chunk(
            db_session,
            paper_id=paper_1.id,
            job_id=job_1.id,
            chunk_index=1,
            content="Multi-head attention allows the model to attend to information from different representation subspaces.",
            is_current=True,
        )
        chunk_2_0 = await create_chunk(
            db_session,
            paper_id=paper_2.id,
            job_id=job_2.id,
            chunk_index=0,
            content="BERT stands for Bidirectional Encoder Representations from Transformers.",
            is_current=True,
        )
        await db_session.commit()

        # 3. Query via RetrievalRepository
        repository = RetrievalRepository(db_session)
        chunks = await repository.get_current_chunks()

        # 4. Verify count and content
        assert len(chunks) == 3
        chunk_ids = [c.id for c in chunks]
        assert chunk_1_0.id in chunk_ids
        assert chunk_1_1.id in chunk_ids
        assert chunk_2_0.id in chunk_ids

        # 5. Verify deterministic ordering: paper_id asc, chunk_index asc
        for i in range(len(chunks) - 1):
            curr_key = (chunks[i].paper_id, chunks[i].chunk_index)
            next_key = (chunks[i + 1].paper_id, chunks[i + 1].chunk_index)
            assert curr_key < next_key

        # 6. Verify parent Paper is eager loaded (relationship accessible)
        for chunk in chunks:
            assert chunk.paper is not None
            assert chunk.paper.title in [
                "Attention Is All You Need",
                "BERT: Pre-training of Deep Bidirectional Transformers",
            ]
            assert chunk.paper.publication_year in [2017, 2018]

        # 7. Verify filtering by specific paper_ids
        p1_chunks = await repository.get_current_chunks(paper_ids=[paper_1.id])
        assert len(p1_chunks) == 2
        assert all(c.paper_id == paper_1.id for c in p1_chunks)
        assert [c.chunk_index for c in p1_chunks] == [0, 1]

    @pytest.mark.asyncio
    async def test_scenario_b_historical_versus_current_chunks(
        self, db_session: AsyncSession
    ) -> None:
        """Scenario B: Historical versus current chunks.

        Verifies:
        - Chunks with is_current=False coexist with is_current=True (conforming to partial unique index).
        - RetrievalRepository.get_current_chunks() fetches only is_current=True records.
        - Historical chunk terms do not appear in index search results.
        """
        paper = await create_paper(db_session, title="Document Revision History Study")
        job = await create_job(db_session, paper_id=paper.id)

        # Historical chunk (is_current=False)
        historical_chunk = await create_chunk(
            db_session,
            paper_id=paper.id,
            job_id=job.id,
            chunk_index=0,
            content="Ancient obsolete draft discussing discarded recurrent network approaches.",
            is_current=False,
        )

        # Active replacement chunk with same index 0 (is_current=True)
        active_chunk = await create_chunk(
            db_session,
            paper_id=paper.id,
            job_id=job.id,
            chunk_index=0,
            content="Approved modern production revision focusing purely on feedforward transformers.",
            is_current=True,
        )
        await db_session.commit()

        # Repository retrieval check
        repository = RetrievalRepository(db_session)
        current_chunks = await repository.get_current_chunks()
        assert len(current_chunks) == 1
        assert current_chunks[0].id == active_chunk.id
        assert current_chunks[0].content == active_chunk.content

        # Service index check
        service = RetrievalService()
        await service.refresh_index(repository)

        # Search for unique historical term: must return nothing
        historical_results = service.search("discarded recurrent approaches")
        assert historical_results == []

        # Search for active term: must return the active chunk
        active_results = service.search("feedforward transformers")
        assert len(active_results) == 1
        assert active_results[0].chunk_id == active_chunk.id
        assert active_results[0].paper_title == "Document Revision History Study"

    @pytest.mark.asyncio
    async def test_scenario_c_end_to_end_retrieval_after_closing_session(
        self, session_factory
    ) -> None:
        """Scenario C: End-to-end retrieval after closing the database session.

        Crucial Architectural Verification:
        1. Insert papers and chunks into PostgreSQL.
        2. Commit changes.
        3. Instantiate RetrievalService.
        4. Instantiate RetrievalRepository with an active AsyncSession.
        5. Refresh index via service.refresh_index(repository).
        6. Explicitly close the database session.
        7. Execute service.search(query) after the session is closed.
        8. Verify all RetrievedChunk fields:
           - Original chunk UUID
           - Parent paper UUID
           - Chunk index
           - Original chunk text content
           - Parent paper title
           - Publication year
           - Valid positive relevance score
           - Correct ranking
        """
        try:
            # 1. Create records using a temporary session
            async with session_factory() as session:
                paper_dpr = await create_paper(
                    session,
                    title="Dense Passage Retrieval for Open-Domain Question Answering",
                    publication_year=2020,
                )
                paper_gpt = await create_paper(
                    session,
                    title="Language Models are Few-Shot Learners",
                    publication_year=2020,
                )
                job_dpr = await create_job(session, paper_id=paper_dpr.id)
                job_gpt = await create_job(session, paper_id=paper_gpt.id)

                chunk_dpr = await create_chunk(
                    session,
                    paper_id=paper_dpr.id,
                    job_id=job_dpr.id,
                    chunk_index=0,
                    content="Dense passage retrieval demonstrates that retrieval can be practically implemented using dual-encoders.",
                    is_current=True,
                )
                chunk_gpt = await create_chunk(
                    session,
                    paper_id=paper_gpt.id,
                    job_id=job_gpt.id,
                    chunk_index=0,
                    content="We demonstrate that scaling up language models greatly improves task-agnostic few-shot performance.",
                    is_current=True,
                )
                await session.commit()

                dpr_chunk_id = chunk_dpr.id
                dpr_paper_id = paper_dpr.id
                gpt_chunk_id = chunk_gpt.id
                gpt_paper_id = paper_gpt.id

            # 2. Open a fresh short-lived session, refresh service index, then explicitly close it
            service = RetrievalService()
            refresh_session = session_factory()
            try:
                repository = RetrievalRepository(refresh_session)
                indexed_count = await service.refresh_index(repository)
                assert indexed_count == 2
            finally:
                await refresh_session.close()

            # 3. Perform search completely in-memory with session closed
            results = service.search(query="dual-encoders dense passage retrieval", top_k=5)

            # 4. Assert top result accuracy
            assert len(results) >= 1
            top_match: RetrievedChunk = results[0]

            assert top_match.chunk_id == dpr_chunk_id
            assert top_match.paper_id == dpr_paper_id
            assert top_match.chunk_index == 0
            assert "dense passage retrieval" in top_match.content.lower()
            assert top_match.paper_title == "Dense Passage Retrieval for Open-Domain Question Answering"
            assert top_match.publication_year == 2020
            assert top_match.score > 0.0

            # 5. Search for GPT content
            gpt_results = service.search(query="few-shot performance task-agnostic", top_k=5)
            assert len(gpt_results) >= 1
            assert gpt_results[0].chunk_id == gpt_chunk_id
            assert gpt_results[0].paper_id == gpt_paper_id
            assert gpt_results[0].paper_title == "Language Models are Few-Shot Learners"
        finally:
            # Reliable asynchronous cleanup even if assertions fail
            async with session_factory() as cleanup_session:
                await cleanup_session.execute(text("TRUNCATE TABLE papers, ingestion_jobs, paper_chunks CASCADE;"))
                await cleanup_session.commit()

    @pytest.mark.asyncio
    async def test_scenario_d_empty_database_lifecycle(
        self, db_session: AsyncSession
    ) -> None:
        """Scenario D: Empty database refresh and search.

        Verifies:
        - Refreshing an empty database succeeds cleanly.
        - Service reports initialized with 0 indexed documents.
        - Subsequent searches return an empty list without error.
        """
        repository = RetrievalRepository(db_session)
        service = RetrievalService()

        indexed_count = await service.refresh_index(repository)

        assert indexed_count == 0
        assert service.is_initialized is True
        assert len(service.id_mapper) == 0

        search_results = service.search(query="transformer architecture", top_k=5)
        assert search_results == []

    @pytest.mark.asyncio
    async def test_scenario_e_query_with_no_matching_results(
        self, db_session: AsyncSession
    ) -> None:
        """Scenario E: Query with no matching terms.

        Verifies:
        - When the database has chunks, querying terms absent from the vocabulary
          returns an empty list.
        """
        paper = await create_paper(db_session, title="Quantum Computing Review")
        job = await create_job(db_session, paper_id=paper.id)
        await create_chunk(
            db_session,
            paper_id=paper.id,
            job_id=job.id,
            chunk_index=0,
            content="Superconducting qubits and quantum error correction circuits.",
            is_current=True,
        )
        await db_session.commit()

        repository = RetrievalRepository(db_session)
        service = RetrievalService()
        await service.refresh_index(repository)

        # Unrelated query
        results = service.search(query="photosynthesis chlorophyll botany plant", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_scenario_f_refresh_after_database_changes(
        self, session_factory
    ) -> None:
        """Scenario F: Re-indexing after database changes.

        Verifies:
        - Index initially reflects original chunk.
        - When database records are mutated (chunk made historical and new chunk added),
          a subsequent refresh accurately indexes the new state.
        - Old terms are no longer searchable; new terms are immediately searchable.
        """
        try:
            service = RetrievalService()

            # Step 1: Insert initial chunk
            async with session_factory() as session1:
                paper = await create_paper(session1, title="Vision Architecture Evolution")
                job1 = await create_job(session1, paper_id=paper.id)
                chunk_v1 = await create_chunk(
                    session1,
                    paper_id=paper.id,
                    job_id=job1.id,
                    chunk_index=0,
                    content="Subsampling filters slide across spatial feature maps.",
                    is_current=True,
                )
                await session1.commit()

                repo1 = RetrievalRepository(session1)
                await service.refresh_index(repo1)

            # Initial search succeeds for CNN chunk
            assert len(service.search("subsampling spatial")) == 1

            # Step 2: Mutate database state (chunk_v1 becomes historical, chunk_v2 is current)
            async with session_factory() as session2:
                # Re-fetch chunk_v1 and mark as historical
                loaded_chunk_v1 = (await session2.execute(
                    text("SELECT id FROM paper_chunks WHERE id = :cid"),
                    {"cid": chunk_v1.id},
                )).scalar_one()

                await session2.execute(
                    text("UPDATE paper_chunks SET is_current = FALSE WHERE id = :cid"),
                    {"cid": loaded_chunk_v1},
                )

                job2 = await create_job(session2, paper_id=paper.id)
                chunk_v2 = await create_chunk(
                    session2,
                    paper_id=paper.id,
                    job_id=job2.id,
                    chunk_index=0,
                    content="Vision transformers replace recurrence with patch token embeddings.",
                    is_current=True,
                )
                await session2.commit()

                repo2 = RetrievalRepository(session2)
                indexed_count = await service.refresh_index(repo2)
                assert indexed_count == 1

            # Step 3: Verify old terms gone, new terms searchable
            assert service.search("subsampling spatial") == []

            new_results = service.search("vision transformers patch token")
            assert len(new_results) == 1
            assert new_results[0].chunk_id == chunk_v2.id
            assert new_results[0].paper_title == "Vision Architecture Evolution"
        finally:
            # Reliable asynchronous cleanup even if assertions fail
            async with session_factory() as cleanup_session:
                await cleanup_session.execute(text("TRUNCATE TABLE papers, ingestion_jobs, paper_chunks CASCADE;"))
                await cleanup_session.commit()

    @pytest.mark.asyncio
    async def test_scenario_g_refresh_failure_preserves_previous_index(
        self, db_session: AsyncSession
    ) -> None:
        """Scenario G: Refresh failure preserves previous index.

        Verifies:
        - When a refresh operation fails (e.g. database error), the previous
          in-memory index, ID mappings, and chunk metadata remain completely intact.
        - Subsequent searches still return the original results.
        """
        paper = await create_paper(db_session, title="Resilience and Fault Tolerance")
        job = await create_job(db_session, paper_id=paper.id)
        chunk = await create_chunk(
            db_session,
            paper_id=paper.id,
            job_id=job.id,
            chunk_index=0,
            content="Fault tolerance maintains system consistency despite component failures.",
            is_current=True,
        )
        await db_session.commit()

        service = RetrievalService()
        initial_repo = RetrievalRepository(db_session)
        await service.refresh_index(initial_repo)

        initial_results = service.search("fault tolerance consistency")
        assert len(initial_results) == 1
        assert initial_results[0].chunk_id == chunk.id

        # Define a failing repository test double
        class FailingRepository(RetrievalRepository):
            def __init__(self) -> None:
                pass  # Bypass session check for failure test double

            async def get_current_chunks(
                self, paper_ids: Sequence[uuid.UUID] | None = None
            ) -> list[PaperChunk]:
                raise ConnectionError("Simulated database connection loss during refresh")

        # Refresh must raise the exception
        with pytest.raises(ConnectionError, match="Simulated database connection loss"):
            await service.refresh_index(FailingRepository())

        # Index and search must remain preserved and functional
        assert service.is_initialized is True
        preserved_results = service.search("fault tolerance consistency")
        assert len(preserved_results) == 1
        assert preserved_results[0].chunk_id == chunk.id
        assert preserved_results[0].paper_title == "Resilience and Fault Tolerance"
