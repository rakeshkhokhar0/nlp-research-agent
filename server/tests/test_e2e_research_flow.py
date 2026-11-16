"""End-to-End Research Workflow Integration Test.

Exercises the complete HTTP-level lifecycle against a PostgreSQL database:
1. Upload a research paper PDF via POST /api/v1/ingestion/upload.
2. Poll / verify job completion via GET /api/v1/ingestion/jobs/{job_id}.
3. Verify that Paper, PaperFile, IngestionJob, PaperChunk, and ChunkEmbedding records are persisted.
4. Verify that the shared lexical RetrievalService index was refreshed automatically.
5. Create a conversation via POST /api/v1/conversations.
6. Submit a research question in Normal Mode.
7. Submit the exact same research question in Advanced Mode.
8. Verify that both modes retrieve the exact same evidence chunk IDs in the exact same ranking order.
9. Verify that each mode makes exactly one OpenAI Responses API call.
10. Verify that the correct system instructions are passed for each mode (SYSTEM_INSTRUCTIONS vs ADVANCED_SYSTEM_INSTRUCTIONS).
11. Verify that citations are validated against retrieved chunks and persisted in the database.
12. Verify that the returned citations expose paper title and page number.
13. Verify that an out-of-scope non-research question bypasses both retrieval and LLM generation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
import uuid

from dotenv import load_dotenv
import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

try:
    from httpx import ASGITransport, AsyncClient
except ImportError:
    from httpx2 import ASGITransport, AsyncClient

# Load environment variables
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

import app.database.model_registry  # noqa: F401 - registers all mappers
from app.database.model_registry import (
    Base,
    ChunkEmbedding,
    Conversation,
    IngestionJob,
    Message,
    MessageCitation,
    Paper,
    PaperChunk,
    PaperFile,
)
from app.database.session import get_db
from app.main import app
from app.modules.conversation.llm_client import (
    ADVANCED_SYSTEM_INSTRUCTIONS,
    OpenAIResponseClient,
    SYSTEM_INSTRUCTIONS,
)
from app.modules.conversation.router import get_conversation_service
from app.modules.conversation.service import (
    ConversationService,
    UNSUPPORTED_SCOPE_RESPONSE,
)
from app.modules.ingestion.embedder import (
    EmbeddingConfig,
    EmbeddingModelManager,
    ResearchEmbeddingGenerator,
)
from app.modules.ingestion.router import get_ingestion_service
from app.modules.ingestion.service import IngestionService
from app.modules.ingestion.storage import CloudinaryStorage, CloudinaryUploadResult
from app.modules.retrieval import RetrievalRepository, RetrievalService
from tests.retrieval.test_retrieval_integration import verify_test_database_safety

TEST_DB_URL = (
    os.getenv("TEST_DATABASE_URL")
    or os.getenv("test_database_url")
    or os.getenv("TEST_DB_URL")
    or os.getenv("test_db_url")
)

APP_DB_URL = os.getenv("DATABASE_URL") or os.getenv("database_url")
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_paper.pdf"


class DeterministicEmbeddingModel:
    """Deterministic test double adhering to EmbeddingModelProtocol.

    Generates valid unit-length float vectors of specified dimension without downloading
    external HuggingFace models.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.call_count = 0

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> list[list[float]]:
        self.call_count += 1
        unit_val = 1.0 / (self.dimensions ** 0.5)
        return [[unit_val] * self.dimensions for _ in sentences]


@pytest_asyncio.fixture
async def e2e_db_engine():
    """Create async engine for the test database, verify safety, and ensure schema is present."""
    if not TEST_DB_URL:
        pytest.skip("TEST_DB_URL is not configured. Skipping E2E test.")

    verify_test_database_safety(TEST_DB_URL, APP_DB_URL)
    engine = create_async_engine(TEST_DB_URL, pool_pre_ping=True)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)
    except (OSError, socket.gaierror, ConnectionRefusedError) as exc:
        await engine.dispose()
        pytest.skip(
            f"Test database is unreachable ({type(exc).__name__}). Skipping E2E integration test."
        )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def e2e_session_factory(e2e_db_engine):
    """Return an async_sessionmaker bound to the test database."""
    return async_sessionmaker(bind=e2e_db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def clean_database(e2e_session_factory):
    """Clean all tables before and after test execution with strict database safety checks."""
    verify_test_database_safety(TEST_DB_URL, APP_DB_URL)
    async with e2e_session_factory() as session:
        await session.execute(
            text(
                "TRUNCATE TABLE message_citations, messages, conversations, "
                "chunk_embeddings, paper_chunks, ingestion_jobs, paper_files, papers CASCADE;"
            )
        )
        await session.commit()
    yield
    verify_test_database_safety(TEST_DB_URL, APP_DB_URL)
    async with e2e_session_factory() as session:
        await session.execute(
            text(
                "TRUNCATE TABLE message_citations, messages, conversations, "
                "chunk_embeddings, paper_chunks, ingestion_jobs, paper_files, papers CASCADE;"
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_end_to_end_research_workflow(
    e2e_session_factory,
    clean_database,
):
    """Test the complete research workflow from upload through grounded conversation in both modes."""
    assert FIXTURE_PATH.exists(), f"PDF fixture missing at {FIXTURE_PATH}"
    pdf_bytes = FIXTURE_PATH.read_bytes()

    # 1. Setup deterministic storage test double
    mock_storage = MagicMock(spec=CloudinaryStorage)
    mock_storage.upload_file.return_value = CloudinaryUploadResult(
        public_id=f"e2e_test_paper_{uuid.uuid4().hex[:8]}",
        secure_url="https://res.cloudinary.com/test/raw/upload/sample_paper.pdf",
        byte_size=len(pdf_bytes),
        format="pdf",
        resource_type="raw",
    )
    mock_storage.validate_pdf.side_effect = CloudinaryStorage.validate_pdf

    # 2. Setup deterministic embedder test double (no network model downloads)
    test_model = DeterministicEmbeddingModel(dimensions=384)
    model_mgr = EmbeddingModelManager(
        config=EmbeddingConfig(dimensions=384),
        model=test_model,
    )
    test_embedder = ResearchEmbeddingGenerator(
        config=EmbeddingConfig(dimensions=384),
        model_manager=model_mgr,
    )

    # 3. Setup shared lexical retrieval service with search invocation spy
    shared_retrieval_service = RetrievalService()
    async with e2e_session_factory() as init_session:
        init_repo = RetrievalRepository(init_session)
        await shared_retrieval_service.refresh_index(init_repo)

    assert len(shared_retrieval_service.id_mapper) == 0
    assert shared_retrieval_service.is_initialized

    # Wrap the real RetrievalService.search with a spy to record calls while keeping the real algorithm active
    original_search = shared_retrieval_service.search
    retrieval_calls: list[dict[str, Any]] = []

    def search_spy(query: str, top_k: int = 5) -> list[Any]:
        retrieval_calls.append({"query": query, "top_k": top_k})
        return original_search(query=query, top_k=top_k)

    shared_retrieval_service.search = search_spy

    # 4. Setup real ingestion service with test doubles
    ingestion_service = IngestionService(
        session_factory=e2e_session_factory,
        storage=mock_storage,
        embedder=test_embedder,
        retrieval_service=shared_retrieval_service,
    )

    # 5. Setup deterministic OpenAI Responses API mock and client
    recorded_llm_calls: list[dict[str, Any]] = []

    async def mock_responses_create(**kwargs: Any):
        recorded_llm_calls.append(kwargs)
        input_text = kwargs.get("input", "")
        # Extract chunk UUIDs from the grounded evidence prompt
        chunk_ids = re.findall(r"Chunk ID:\s*([0-9a-fA-F-]+)", input_text)
        # Select the first chunk ID as the cited chunk
        cited = chunk_ids[:1] if chunk_ids else []

        mock_resp = MagicMock()
        mock_resp.status = "completed"
        mock_resp.refusal = None
        mock_resp.error = None
        mock_resp.output_text = json.dumps({
            "answer": "The continuous Skip-gram and CBOW architectures learn high-quality distributed representations.",
            "cited_chunk_ids": cited,
        })
        usage = MagicMock()
        usage.input_tokens = 150
        usage.output_tokens = 40
        usage.total_tokens = 190
        mock_resp.usage = usage
        mock_resp.model = kwargs.get("model", "gpt-4o-2024-08-06")
        return mock_resp

    mock_async_openai = MagicMock()
    mock_async_openai.responses.create = AsyncMock(side_effect=mock_responses_create)
    mock_async_openai.close = AsyncMock()

    test_llm_client = OpenAIResponseClient(
        api_key="sk-mock-deterministic-e2e-key-1234567890",
        client=mock_async_openai,
    )

    # 6. Setup real conversation service
    conversation_service = ConversationService(
        session_factory=e2e_session_factory,
        retrieval_service=shared_retrieval_service,
        llm_client=test_llm_client,
    )

    # 7. Wire application dependencies and state (saving existing global state for clean restoration)
    original_overrides = dict(app.dependency_overrides)
    state_attr_names = ["retrieval_service", "ingestion_service", "conversation_service", "openai_client"]
    original_state_attrs: dict[str, Any] = {}
    state_attrs_existed: dict[str, bool] = {}

    for attr in state_attr_names:
        existed = hasattr(app.state, attr)
        state_attrs_existed[attr] = existed
        if existed:
            original_state_attrs[attr] = getattr(app.state, attr)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with e2e_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service

    app.state.retrieval_service = shared_retrieval_service
    app.state.ingestion_service = ingestion_service
    app.state.conversation_service = conversation_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:

            # -------------------------------------------------------------
            # STEP 1: Upload a PDF through POST /api/v1/ingestion/upload
            # -------------------------------------------------------------
            files = {
                "file": ("sample_paper.pdf", pdf_bytes, "application/pdf"),
            }
            upload_resp = await client.post("/api/v1/ingestion/upload", files=files)
            assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
            upload_data = upload_resp.json()

            paper_id_str = upload_data["paper_id"]
            job_id_str = upload_data["job_id"]
            paper_id = uuid.UUID(paper_id_str)
            job_id = uuid.UUID(job_id_str)

            assert upload_data["chunks_created"] > 0
            assert upload_data["embeddings_created"] == upload_data["chunks_created"]
            assert upload_data["status"] == "completed"

            # -------------------------------------------------------------
            # STEP 2: Wait/verify job completion via GET /api/v1/ingestion/jobs/{job_id}
            # -------------------------------------------------------------
            job_status_resp = await client.get(f"/api/v1/ingestion/jobs/{job_id_str}")
            assert job_status_resp.status_code == 200
            job_status_data = job_status_resp.json()
            assert job_status_data["job_id"] == job_id_str
            assert job_status_data["status"] == "completed"
            assert job_status_data["paper_id"] == paper_id_str

            # -------------------------------------------------------------
            # STEP 3: Verify paper, file, job, chunk, and embedding persistence in DB
            # -------------------------------------------------------------
            async with e2e_session_factory() as verify_session:
                # 3a. Verify Paper record
                paper_row = await verify_session.get(Paper, paper_id)
                assert paper_row is not None
                assert "Word Representations" in paper_row.title

                # 3b. Verify PaperFile record and relationship to Paper
                paper_file_id = uuid.UUID(upload_data["paper_file_id"])
                paper_file_row = await verify_session.get(PaperFile, paper_file_id)
                assert paper_file_row is not None
                assert paper_file_row.paper_id == paper_id
                assert paper_file_row.original_filename == "sample_paper.pdf"
                assert paper_file_row.file_size_bytes == len(pdf_bytes)
                assert paper_file_row.status in ("uploaded", "available")
                assert paper_file_row.storage_key == mock_storage.upload_file.return_value.public_id

                # 3c. Verify IngestionJob record and relationship to Paper & PaperFile
                job_db_row = await verify_session.get(IngestionJob, job_id)
                assert job_db_row is not None
                assert job_db_row.paper_id == paper_id
                assert job_db_row.paper_file_id == paper_file_id
                assert job_db_row.status == "completed"
                assert job_db_row.completed_at is not None

                # 3d. Verify PaperChunk records and relationship to Paper
                chunks_query = await verify_session.execute(
                    select(PaperChunk).where(PaperChunk.paper_id == paper_id)
                )
                persisted_chunks = chunks_query.scalars().all()
                assert len(persisted_chunks) == upload_data["chunks_created"]
                assert all(c.is_current for c in persisted_chunks)
                assert all(c.paper_id == paper_id for c in persisted_chunks)

                # 3e. Verify ChunkEmbedding records and 1-to-1 relationship to PaperChunk
                chunk_ids = [c.id for c in persisted_chunks]
                embeddings_query = await verify_session.execute(
                    select(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids))
                )
                persisted_embeddings = embeddings_query.scalars().all()
                assert len(persisted_embeddings) == len(persisted_chunks)
                assert len(persisted_embeddings) == upload_data["embeddings_created"]
                embedded_chunk_ids = {e.chunk_id for e in persisted_embeddings}
                assert embedded_chunk_ids == set(chunk_ids)
                assert all(len(e.embedding) == 384 for e in persisted_embeddings)
                assert all(e.embedding_model_id is not None for e in persisted_embeddings)

            # -------------------------------------------------------------
            # STEP 4: Verify shared lexical retrieval index was refreshed automatically
            # -------------------------------------------------------------
            assert shared_retrieval_service.is_initialized
            assert len(shared_retrieval_service.id_mapper) == upload_data["chunks_created"]
            assert len(shared_retrieval_service._chunk_metadata) == upload_data["chunks_created"]
            assert len(shared_retrieval_service.engine.index.index) > 50

            # -------------------------------------------------------------
            # STEP 5: Create a conversation through HTTP API
            # -------------------------------------------------------------
            conv_resp = await client.post(
                "/api/v1/conversations",
                json={"title": "Word2Vec Research Chat"},
            )
            assert conv_resp.status_code == 201
            conv_data = conv_resp.json()
            conv_id_str = conv_data["id"]
            conv_id = uuid.UUID(conv_id_str)
            assert conv_data["status"] == "active"

            # -------------------------------------------------------------
            # STEP 6: Submit a research question in Normal Mode
            # -------------------------------------------------------------
            research_question = (
                "What are the continuous skip-gram and CBOW model architectures "
                "introduced for learning word representations in vector space?"
            )

            normal_resp = await client.post(
                f"/api/v1/conversations/{conv_id_str}/messages",
                json={
                    "question": research_question,
                    "mode": "normal",
                    "top_k": 5,
                },
            )
            assert normal_resp.status_code == 200, f"Normal request failed: {normal_resp.text}"
            normal_data = normal_resp.json()
            assert normal_data["mode"] == "normal"
            assert normal_data["conversation_id"] == conv_id_str
            assert len(normal_data["citations"]) > 0

            # -------------------------------------------------------------
            # STEP 7: Submit the exact same research question in Advanced Mode
            # -------------------------------------------------------------
            advanced_resp = await client.post(
                f"/api/v1/conversations/{conv_id_str}/messages",
                json={
                    "question": research_question,
                    "mode": "advanced",
                    "top_k": 5,
                },
            )
            assert advanced_resp.status_code == 200, f"Advanced request failed: {advanced_resp.text}"
            advanced_data = advanced_resp.json()
            assert advanced_data["mode"] == "advanced"
            assert advanced_data["conversation_id"] == conv_id_str
            assert len(advanced_data["citations"]) > 0

            # -------------------------------------------------------------
            # STEP 8: Verify both modes retrieve the same chunk IDs in the same order
            # -------------------------------------------------------------
            assert len(recorded_llm_calls) == 2, f"Expected 2 LLM calls, got {len(recorded_llm_calls)}"
            assert len(retrieval_calls) == 2, f"Expected 2 retrieval calls, got {len(retrieval_calls)}"

            # Verify both Normal and Advanced invoked real retrieval with the exact research question
            assert retrieval_calls[0]["query"] == research_question
            assert retrieval_calls[0]["top_k"] == 5
            assert retrieval_calls[1]["query"] == research_question
            assert retrieval_calls[1]["top_k"] == 5

            normal_call = recorded_llm_calls[0]
            advanced_call = recorded_llm_calls[1]

            normal_chunk_ids = re.findall(r"Chunk ID:\s*([0-9a-fA-F-]+)", normal_call["input"])
            advanced_chunk_ids = re.findall(r"Chunk ID:\s*([0-9a-fA-F-]+)", advanced_call["input"])

            assert len(normal_chunk_ids) > 0, "No chunks retrieved in Normal Mode"
            assert normal_chunk_ids == advanced_chunk_ids, (
                f"Retrieved chunk IDs mismatch between modes:\n"
                f"Normal:   {normal_chunk_ids}\n"
                f"Advanced: {advanced_chunk_ids}"
            )

            # -------------------------------------------------------------
            # STEP 9: Verify exactly one OpenAI Responses API call per message
            # -------------------------------------------------------------
            # Exactly 2 messages submitted so far -> exactly 2 calls
            assert len(recorded_llm_calls) == 2

            # -------------------------------------------------------------
            # STEP 10: Verify correct system instructions passed for each mode
            # -------------------------------------------------------------
            assert normal_call["instructions"] == SYSTEM_INSTRUCTIONS
            assert advanced_call["instructions"] == ADVANCED_SYSTEM_INSTRUCTIONS
            assert normal_call["instructions"] != advanced_call["instructions"]

            # -------------------------------------------------------------
            # STEP 11: Verify citations are validated and persisted in DB
            # -------------------------------------------------------------
            async with e2e_session_factory() as verify_session:
                normal_asst_msg_id = uuid.UUID(normal_data["assistant_message_id"])
                adv_asst_msg_id = uuid.UUID(advanced_data["assistant_message_id"])

                normal_cits = (
                    await verify_session.execute(
                        select(MessageCitation).where(
                            MessageCitation.message_id == normal_asst_msg_id
                        )
                    )
                ).scalars().all()
                assert len(normal_cits) == len(normal_data["citations"])
                assert str(normal_cits[0].chunk_id) in normal_chunk_ids

                adv_cits = (
                    await verify_session.execute(
                        select(MessageCitation).where(
                            MessageCitation.message_id == adv_asst_msg_id
                        )
                    )
                ).scalars().all()
                assert len(adv_cits) == len(advanced_data["citations"])
                assert str(adv_cits[0].chunk_id) in advanced_chunk_ids

            # -------------------------------------------------------------
            # STEP 12: Verify returned citations expose paper title and page number
            # -------------------------------------------------------------
            for cit in normal_data["citations"]:
                assert cit["paper_title"] is not None
                assert "Word Representations" in cit["paper_title"]
                assert cit["page_number"] is not None
                assert isinstance(cit["page_number"], int)
                assert cit["page_number"] >= 1

            for cit in advanced_data["citations"]:
                assert cit["paper_title"] is not None
                assert "Word Representations" in cit["paper_title"]
                assert cit["page_number"] is not None
                assert isinstance(cit["page_number"], int)
                assert cit["page_number"] >= 1

            # -------------------------------------------------------------
            # STEP 13: Verify out-of-scope question bypasses retrieval & LLM
            # -------------------------------------------------------------
            retrieval_count_before = len(retrieval_calls)
            llm_call_count_before = len(recorded_llm_calls)
            out_of_scope_resp = await client.post(
                f"/api/v1/conversations/{conv_id_str}/messages",
                json={
                    "question": "What is the capital of France?",
                    "mode": "normal",
                },
            )
            assert out_of_scope_resp.status_code == 200
            out_of_scope_data = out_of_scope_resp.json()
            assert out_of_scope_data["answer"] == UNSUPPORTED_SCOPE_RESPONSE
            assert out_of_scope_data["citations"] == []

            # Verify retrieval was completely bypassed and invocation count did not increase
            assert len(retrieval_calls) == retrieval_count_before

            # Verify OpenAI was not called
            assert len(recorded_llm_calls) == llm_call_count_before

    finally:
        # Restore application dependency overrides to previous state
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

        # Restore application state attributes or remove if they did not exist prior to test
        for attr, existed in state_attrs_existed.items():
            if existed:
                setattr(app.state, attr, original_state_attrs[attr])
            elif hasattr(app.state, attr):
                delattr(app.state, attr)
