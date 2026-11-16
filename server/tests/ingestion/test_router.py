"""Unit tests for the ingestion FastAPI router endpoints.

Tests:
- POST /api/v1/ingestion/upload
  - Successful PDF upload with mock IngestionService
  - Validation failures (missing file, empty filename, empty content, non-PDF content)
  - Domain error mapping (InvalidPDFError, PDFParsingError, PaperValidationError,
    ActiveJobConflictError, StorageUploadError, generic internal error)
- GET /api/v1/ingestion/jobs/{job_id}
  - Successful job status retrieval
  - Unknown job ID (404)
  - Malformed job UUID (422)
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
from unittest.mock import AsyncMock, MagicMock
import uuid

from fastapi import status
from fastapi.testclient import TestClient
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import app.database.model_registry  # noqa: F401
from app.core.exceptions import (
    ActiveJobConflictError,
    AppError,
    InvalidPDFError,
    PDFParsingError,
    StorageUploadError,
)
from app.database.session import get_db
from app.main import app
from app.modules.ingestion.models import IngestionJob
from app.modules.ingestion.router import get_ingestion_service
from app.modules.ingestion.service import IngestionResult, IngestionService
from app.modules.papers.service import PaperValidationError


@pytest.fixture
def mock_ingestion_service() -> MagicMock:
    """Mock IngestionService to avoid heavy model execution or network uploads."""
    service = MagicMock(spec=IngestionService)
    service.ingest_pdf = AsyncMock()
    return service


@pytest.fixture
def mock_db_session() -> AsyncMock:
    """Mock SQLAlchemy AsyncSession for repository testing."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def client(mock_ingestion_service: MagicMock, mock_db_session: AsyncMock) -> TestClient:
    """TestClient with dependency overrides for IngestionService and get_db."""
    app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion_service
    app.dependency_overrides[get_db] = lambda: mock_db_session
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# POST /api/v1/ingestion/upload Tests
# -----------------------------------------------------------------------------


def test_upload_pdf_success(client: TestClient, mock_ingestion_service: MagicMock) -> None:
    """Verify successful PDF upload returns 200 with complete IngestionUploadResponse."""
    expected_paper_id = uuid.uuid4()
    expected_file_id = uuid.uuid4()
    expected_job_id = uuid.uuid4()

    mock_result = IngestionResult(
        paper_id=expected_paper_id,
        paper_file_id=expected_file_id,
        ingestion_job_id=expected_job_id,
        chunks_created=12,
        embeddings_created=12,
        paper_reused=False,
        file_reused=False,
        job_reused=False,
        status="completed",
    )
    mock_ingestion_service.ingest_pdf.return_value = mock_result

    valid_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    files = {"file": ("sample.pdf", io.BytesIO(valid_pdf_bytes), "application/pdf")}
    data = {"doi": "10.1234/test.doi"}

    response = client.post("/api/v1/ingestion/upload", files=files, data=data)

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["paper_id"] == str(expected_paper_id)
    assert body["job_id"] == str(expected_job_id)
    assert body["paper_file_id"] == str(expected_file_id)
    assert body["chunks_created"] == 12
    assert body["embeddings_created"] == 12
    assert body["paper_reused"] is False
    assert body["file_reused"] is False
    assert body["job_reused"] is False
    assert body["status"] == "completed"

    mock_ingestion_service.ingest_pdf.assert_awaited_once_with(
        pdf_bytes=valid_pdf_bytes,
        original_filename="sample.pdf",
        title=None,
        doi="10.1234/test.doi",
    )


def test_upload_pdf_success_without_doi(
    client: TestClient, mock_ingestion_service: MagicMock
) -> None:
    """Verify successful PDF upload without optional DOI passes None to service."""
    mock_result = IngestionResult(
        paper_id=uuid.uuid4(),
        paper_file_id=uuid.uuid4(),
        ingestion_job_id=uuid.uuid4(),
        chunks_created=5,
        embeddings_created=5,
        paper_reused=True,
        file_reused=True,
        job_reused=True,
        status="completed",
    )
    mock_ingestion_service.ingest_pdf.return_value = mock_result

    valid_pdf_bytes = b"%PDF-1.5 test document"
    files = {"file": ("paper.pdf", io.BytesIO(valid_pdf_bytes), "application/pdf")}

    response = client.post("/api/v1/ingestion/upload", files=files)

    assert response.status_code == status.HTTP_200_OK
    mock_ingestion_service.ingest_pdf.assert_awaited_once_with(
        pdf_bytes=valid_pdf_bytes,
        original_filename="paper.pdf",
        title=None,
        doi=None,
    )


def test_upload_pdf_with_manual_title(
    client: TestClient, mock_ingestion_service: MagicMock
) -> None:
    """Verify PDF upload with manually supplied title passes title to IngestionService."""
    mock_result = IngestionResult(
        paper_id=uuid.uuid4(),
        paper_file_id=uuid.uuid4(),
        ingestion_job_id=uuid.uuid4(),
        chunks_created=8,
        embeddings_created=8,
        paper_reused=False,
        file_reused=False,
        job_reused=False,
        status="completed",
        title="Attention Is All You Need (Official)",
    )
    mock_ingestion_service.ingest_pdf.return_value = mock_result

    valid_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    files = {"file": ("attention.pdf", io.BytesIO(valid_pdf_bytes), "application/pdf")}
    data = {
        "title": "Attention Is All You Need (Official)",
        "doi": "10.48550/arXiv.1706.03762",
    }

    response = client.post("/api/v1/ingestion/upload", files=files, data=data)

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["title"] == "Attention Is All You Need (Official)"
    assert body["chunks_created"] == 8

    mock_ingestion_service.ingest_pdf.assert_awaited_once_with(
        pdf_bytes=valid_pdf_bytes,
        original_filename="attention.pdf",
        title="Attention Is All You Need (Official)",
        doi="10.48550/arXiv.1706.03762",
    )


def test_upload_pdf_missing_file_parameter(client: TestClient) -> None:
    """Verify request without file parameter returns 422 Unprocessable Entity."""
    response = client.post("/api/v1/ingestion/upload", data={"doi": "10.1234/foo"})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_upload_pdf_empty_filename(client: TestClient) -> None:
    """Verify upload with empty or whitespace filename returns 400 Bad Request or 422."""
    # Whitespace filename is caught by route validation
    files = {"file": ("   ", io.BytesIO(b"%PDF-1.4 content"), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "filename" in response.json()["detail"].lower()

    # Empty string filename is rejected by multipart parser
    files_empty = {"file": ("", io.BytesIO(b"%PDF-1.4 content"), "application/pdf")}
    resp_empty = client.post("/api/v1/ingestion/upload", files=files_empty)
    assert resp_empty.status_code in (status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY)


def test_upload_pdf_empty_content(client: TestClient) -> None:
    """Verify upload of 0-byte file returns 400 Bad Request."""
    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "empty" in response.json()["detail"].lower()


def test_upload_pdf_invalid_signature(client: TestClient) -> None:
    """Verify file without %PDF- header returns 400 Bad Request."""
    files = {"file": ("fake.pdf", io.BytesIO(b"NOT A PDF DOCUMENT"), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "invalid pdf" in response.json()["detail"].lower()


def test_upload_pdf_service_invalid_pdf_error(
    client: TestClient, mock_ingestion_service: MagicMock
) -> None:
    """Verify service InvalidPDFError is mapped to 400 Bad Request."""
    mock_ingestion_service.ingest_pdf.side_effect = InvalidPDFError("PDF header corrupt")

    files = {"file": ("corrupt.pdf", io.BytesIO(b"%PDF-corrupt"), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "PDF header corrupt" in response.json()["detail"]


def test_upload_pdf_service_parsing_error(
    client: TestClient, mock_ingestion_service: MagicMock
) -> None:
    """Verify service PDFParsingError is mapped to 400 Bad Request."""
    mock_ingestion_service.ingest_pdf.side_effect = PDFParsingError("Unable to extract pages")

    files = {"file": ("unparseable.pdf", io.BytesIO(b"%PDF-1.4 unparseable"), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "parsing" in response.json()["detail"].lower()


def test_upload_pdf_paper_validation_error(
    client: TestClient, mock_ingestion_service: MagicMock
) -> None:
    """Verify PaperValidationError is mapped to 422 Unprocessable Entity."""
    mock_ingestion_service.ingest_pdf.side_effect = PaperValidationError("Title is missing")

    files = {"file": ("untitled.pdf", io.BytesIO(b"%PDF-1.4 valid"), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "Title is missing" in response.json()["detail"]


def test_upload_pdf_active_job_conflict(
    client: TestClient, mock_ingestion_service: MagicMock
) -> None:
    """Verify ActiveJobConflictError is mapped to 409 Conflict."""
    mock_ingestion_service.ingest_pdf.side_effect = ActiveJobConflictError(
        "Ingestion job already active"
    )

    files = {"file": ("duplicate.pdf", io.BytesIO(b"%PDF-1.4 valid"), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)

    assert response.status_code == status.HTTP_409_CONFLICT
    assert "already active" in response.json()["detail"]


def test_upload_pdf_storage_failure(
    client: TestClient, mock_ingestion_service: MagicMock
) -> None:
    """Verify StorageUploadError is mapped to 502 Bad Gateway without leaking internal details."""
    mock_ingestion_service.ingest_pdf.side_effect = StorageUploadError(
        "Cloudinary internal connection timeout secret=xyz"
    )

    files = {"file": ("upload_fail.pdf", io.BytesIO(b"%PDF-1.4 valid"), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert response.json()["detail"] == "Storage service error during PDF upload."
    assert "secret" not in response.json()["detail"]


def test_upload_pdf_domain_app_error(
    client: TestClient, mock_ingestion_service: MagicMock
) -> None:
    """Verify AppError is mapped to 500 Internal Server Error."""
    mock_ingestion_service.ingest_pdf.side_effect = AppError("Processing pipeline halted")

    files = {"file": ("app_error.pdf", io.BytesIO(b"%PDF-1.4 valid"), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.json()["detail"] == "Document ingestion processing error."


def test_upload_pdf_unexpected_exception(
    client: TestClient, mock_ingestion_service: MagicMock
) -> None:
    """Verify unhandled Exception is mapped to generic 500 without leaking traceback."""
    mock_ingestion_service.ingest_pdf.side_effect = RuntimeError("Low-level crash at /internal/code")

    files = {"file": ("crash.pdf", io.BytesIO(b"%PDF-1.4 valid"), "application/pdf")}
    response = client.post("/api/v1/ingestion/upload", files=files)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.json()["detail"] == "An unexpected error occurred during document ingestion."
    assert "traceback" not in response.text.lower()


# -----------------------------------------------------------------------------
# GET /api/v1/ingestion/jobs/{job_id} Tests
# -----------------------------------------------------------------------------


def test_get_job_status_success(client: TestClient, mock_db_session: AsyncMock) -> None:
    """Verify successful lookup of an existing IngestionJob returns 200 with job details."""
    target_job_id = uuid.uuid4()
    target_paper_id = uuid.uuid4()
    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)

    mock_job = IngestionJob(
        id=target_job_id,
        paper_id=target_paper_id,
        status="completed",
        current_stage="indexing",
        attempt_count=2,
        error_message=None,
        created_at=now,
        started_at=now,
        completed_at=now,
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_db_session.execute.return_value = mock_result

    response = client.get(f"/api/v1/ingestion/jobs/{target_job_id}")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["job_id"] == str(target_job_id)
    assert body["paper_id"] == str(target_paper_id)
    assert body["status"] == "completed"
    assert body["current_stage"] == "indexing"
    assert body["attempt_count"] == 2
    assert body["error_message"] is None
    assert "2026-11-14" in body["created_at"]


def test_get_job_status_not_found(client: TestClient, mock_db_session: AsyncMock) -> None:
    """Verify lookup of nonexistent job ID returns 404 Not Found."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute.return_value = mock_result

    unknown_id = uuid.uuid4()
    response = client.get(f"/api/v1/ingestion/jobs/{unknown_id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert f"Ingestion job '{unknown_id}' not found." in response.json()["detail"]


def test_get_job_status_invalid_uuid(client: TestClient) -> None:
    """Verify lookup with invalid UUID string returns 422 Unprocessable Entity."""
    response = client.get("/api/v1/ingestion/jobs/not-a-valid-uuid")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# -----------------------------------------------------------------------------
# Dependency Provider Tests
# -----------------------------------------------------------------------------


def test_get_ingestion_service_uninitialized_raises_runtime_error() -> None:
    """Verify get_ingestion_service raises RuntimeError if app.state.ingestion_service is missing."""
    mock_request = MagicMock()
    # Explicitly clear ingestion_service from app.state
    del mock_request.app.state.ingestion_service

    with pytest.raises(RuntimeError) as exc_info:
        get_ingestion_service(mock_request)

    assert "Application lifespan initialization is required" in str(exc_info.value)


def test_get_ingestion_service_returns_app_state_service() -> None:
    """Verify get_ingestion_service returns request.app.state.ingestion_service."""
    mock_request = MagicMock()
    expected_service = MagicMock(spec=IngestionService)
    mock_request.app.state.ingestion_service = expected_service

    result = get_ingestion_service(mock_request)
    assert result is expected_service
