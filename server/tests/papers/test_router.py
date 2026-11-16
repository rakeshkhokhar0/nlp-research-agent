"""Unit tests for the research papers catalog router endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import uuid

from fastapi import status
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import app.database.model_registry  # noqa: F401
from app.database.session import get_db
from app.main import app
from app.modules.papers.models import Paper, PaperFile
from app.modules.papers.repository import PaperRepository


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Mock SQLAlchemy AsyncSession for repository testing."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def client(mock_db_session: AsyncMock) -> TestClient:
    """TestClient with dependency overrides for get_db."""
    app.dependency_overrides[get_db] = lambda: mock_db_session
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def test_list_papers_empty(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify empty catalog returns 200 with empty items list and total 0."""
    async def mock_list(self, **kwargs):
        return [], 0

    monkeypatch.setattr(PaperRepository, "list_papers", mock_list)

    response = client.get("/api/v1/papers")
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["limit"] == 50
    assert body["offset"] == 0


def test_list_papers_with_items(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify listing papers returns formatted metadata, files, and chunk counts."""
    paper_id = uuid.uuid4()
    file_id = uuid.uuid4()

    mock_paper = Paper(
        id=paper_id,
        title="Attention Is All You Need",
        doi="10.48550/arXiv.1706.03762",
        created_at=datetime(2026, 11, 15, 10, 0, 0, tzinfo=timezone.utc),
    )
    mock_file = PaperFile(
        id=file_id,
        paper_id=paper_id,
        storage_provider="cloudinary",
        storage_key="raw/paper1",
        original_filename="sample_paper.pdf",
        mime_type="application/pdf",
        file_size_bytes=1048576,
        is_primary=True,
        status="available",
        created_at=datetime(2026, 11, 15, 10, 0, 0, tzinfo=timezone.utc),
    )
    mock_paper.files = [mock_file]
    mock_paper.ingestion_jobs = []

    async def mock_list(self, **kwargs):
        return [(mock_paper, 42, 12)], 1

    monkeypatch.setattr(PaperRepository, "list_papers", mock_list)

    response = client.get("/api/v1/papers?limit=10&offset=0")
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1

    item = body["items"][0]
    assert item["id"] == str(paper_id)
    assert item["title"] == "Attention Is All You Need"
    assert item["original_filename"] == "sample_paper.pdf"
    assert item["file_size_bytes"] == 1048576
    assert item["chunk_count"] == 42
    assert item["page_count"] == 12
    assert item["ingestion_status"] == "completed"


def test_list_papers_search_filter(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify search parameter is forwarded to repository."""
    captured_search = None

    async def mock_list(self, *, limit, offset, search=None):
        nonlocal captured_search
        captured_search = search
        return [], 0

    monkeypatch.setattr(PaperRepository, "list_papers", mock_list)

    response = client.get("/api/v1/papers?search=Transformer")
    assert response.status_code == status.HTTP_200_OK
    assert captured_search == "Transformer"


def test_get_paper_details_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GET /api/v1/papers/{paper_id} returns 200 with complete details and files list."""
    paper_id = uuid.uuid4()
    file_id = uuid.uuid4()

    mock_paper = Paper(
        id=paper_id,
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        doi="10.18653/v1/N19-1423",
        created_at=datetime(2026, 11, 15, 10, 0, 0, tzinfo=timezone.utc),
    )
    mock_file = PaperFile(
        id=file_id,
        paper_id=paper_id,
        storage_provider="cloudinary",
        storage_key="raw/bert",
        original_filename="bert_paper.pdf",
        mime_type="application/pdf",
        file_size_bytes=2097152,
        is_primary=True,
        status="available",
        created_at=datetime(2026, 11, 15, 10, 0, 0, tzinfo=timezone.utc),
    )
    mock_paper.files = [mock_file]
    mock_paper.ingestion_jobs = []

    async def mock_get(self, pid):
        if pid == paper_id:
            return (mock_paper, 55, 16)
        return None

    monkeypatch.setattr(PaperRepository, "get_paper_details", mock_get)

    response = client.get(f"/api/v1/papers/{paper_id}")
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["id"] == str(paper_id)
    assert body["title"] == "BERT: Pre-training of Deep Bidirectional Transformers"
    assert body["chunk_count"] == 55
    assert body["page_count"] == 16
    assert len(body["files"]) == 1
    assert body["files"][0]["original_filename"] == "bert_paper.pdf"


def test_get_paper_details_not_found(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GET /api/v1/papers/{paper_id} with unknown UUID returns 404."""
    async def mock_get(self, pid):
        return None

    monkeypatch.setattr(PaperRepository, "get_paper_details", mock_get)

    unknown_id = uuid.uuid4()
    response = client.get(f"/api/v1/papers/{unknown_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_paper_details_invalid_uuid(client: TestClient) -> None:
    """Verify GET /api/v1/papers/{paper_id} with malformed UUID returns 422."""
    response = client.get("/api/v1/papers/not-a-valid-uuid")
    assert response.status_code in {status.HTTP_422_UNPROCESSABLE_ENTITY, 422}
