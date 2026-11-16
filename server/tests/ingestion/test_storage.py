"""Unit tests for CloudinaryStorage component.

Tests all operations using mocked Cloudinary SDK calls and mocked network requests:
- PDF header validation and strict bytes input enforcement
- Uploading raw files and returning typed metadata
- Upload response validation (dict type, required fields, bytes range)
- Querying asset metadata
- Downloading via secure URL with fallback to signed private URL
- Downloaded content PDF validation and exception mapping
- Deleting files idempotently and delete response validation
- Proper domain exception mapping
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock, patch
import urllib.error

import cloudinary.exceptions
import pytest

from app.core.exceptions import (
    InvalidPDFError,
    StorageDeleteError,
    StorageDownloadError,
    StorageError,
    StorageFileNotFoundError,
    StorageUploadError,
)
from app.modules.ingestion.storage import (
    CloudinaryStorage,
    CloudinaryUploadResult,
)

VALID_PDF_SAMPLE = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def storage() -> CloudinaryStorage:
    """Fixture providing a fresh CloudinaryStorage instance."""
    return CloudinaryStorage()


class TestPDFValidation:
    """Tests for PDF content and header signature validation."""

    def test_valid_pdf_header_accepted(self, storage: CloudinaryStorage) -> None:
        """Valid bytes starting with %PDF- header must pass validation."""
        storage.validate_pdf(VALID_PDF_SAMPLE)
        storage.validate_pdf(b"%PDF-1.7 extra content")

    @pytest.mark.parametrize(
        "invalid_content",
        [
            b"",
            b"Hello, this is just plain text.",
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",  # PNG header
            b"\xff\xd8\xff\xe0\x00\x10JFIF",          # JPEG header
            b"pdf-without-percent",
            b" %PDF-leading-space",
        ],
    )
    def test_invalid_header_rejected(
        self,
        storage: CloudinaryStorage,
        invalid_content: bytes,
    ) -> None:
        """Content without valid %PDF- header must raise InvalidPDFError."""
        with pytest.raises(InvalidPDFError, match="Invalid PDF header|must not be empty"):
            storage.validate_pdf(invalid_content)

    @pytest.mark.parametrize(
        "non_bytes",
        [
            "string_data",
            12345,
            None,
            ["list", "of", "data"],
            {"key": "val"},
            bytearray(VALID_PDF_SAMPLE),
            bytearray(b"%PDF-1.4 header"),
        ],
    )
    def test_non_bytes_input_rejected(
        self,
        storage: CloudinaryStorage,
        non_bytes: Any,
    ) -> None:
        """Non-bytes inputs, including bytearray, must raise InvalidPDFError."""
        with pytest.raises(InvalidPDFError, match="PDF content must be a bytes instance"):
            storage.validate_pdf(non_bytes)


class TestUploadFile:
    """Tests for uploading files to Cloudinary."""

    def test_successful_upload(self, storage: CloudinaryStorage) -> None:
        """Successful upload passes resource_type='raw' and returns CloudinaryUploadResult."""
        mock_response = {
            "public_id": "nlp-research-agent/papers/attention_paper_123",
            "secure_url": "https://res.cloudinary.com/test_cloud/raw/upload/v1/attention.pdf",
            "bytes": len(VALID_PDF_SAMPLE),
            "format": "pdf",
            "resource_type": "raw",
            "created_at": "2026-11-13T10:00:00Z",
        }

        with patch("cloudinary.uploader.upload", return_value=mock_response) as mock_upload:
            result = storage.upload_file(VALID_PDF_SAMPLE, folder="nlp-research-agent/papers")

            mock_upload.assert_called_once_with(
                VALID_PDF_SAMPLE,
                resource_type="raw",
                folder="nlp-research-agent/papers",
            )

            assert isinstance(result, CloudinaryUploadResult)
            assert result.public_id == "nlp-research-agent/papers/attention_paper_123"
            assert result.secure_url == "https://res.cloudinary.com/test_cloud/raw/upload/v1/attention.pdf"
            assert result.byte_size == len(VALID_PDF_SAMPLE)
            assert result.resource_type == "raw"
            assert result.format == "pdf"
            assert result.created_at == "2026-11-13T10:00:00Z"

    def test_upload_invalid_pdf_fails_before_sdk_call(self, storage: CloudinaryStorage) -> None:
        """Invalid PDF content must fail validation before invoking the Cloudinary SDK."""
        with patch("cloudinary.uploader.upload") as mock_upload:
            with pytest.raises(InvalidPDFError):
                storage.upload_file(b"not-a-pdf-document")
            mock_upload.assert_not_called()

    def test_upload_sdk_failure_mapped_to_storage_upload_error(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """Cloudinary SDK exceptions must be wrapped in StorageUploadError."""
        with patch(
            "cloudinary.uploader.upload",
            side_effect=cloudinary.exceptions.Error("Upload rate limit exceeded"),
        ):
            with pytest.raises(StorageUploadError, match="Cloudinary PDF upload failed"):
                storage.upload_file(VALID_PDF_SAMPLE)

    @pytest.mark.parametrize(
        "non_dict_response",
        [None, "string_response", [1, 2, 3], 12345],
    )
    def test_upload_non_dict_response_rejected(
        self,
        storage: CloudinaryStorage,
        non_dict_response: Any,
    ) -> None:
        """Non-dictionary upload response must raise StorageUploadError."""
        with patch("cloudinary.uploader.upload", return_value=non_dict_response):
            with pytest.raises(StorageUploadError, match="malformed response: expected dict"):
                storage.upload_file(VALID_PDF_SAMPLE)

    @pytest.mark.parametrize("missing_or_empty_id", [None, "", "   ", "\t", 123])
    def test_upload_missing_or_empty_public_id_rejected(
        self,
        storage: CloudinaryStorage,
        missing_or_empty_id: Any,
    ) -> None:
        """Upload response missing or empty public_id must raise StorageUploadError."""
        resp: dict[str, Any] = {
            "secure_url": "https://res.cloudinary.com/test/raw/upload/paper.pdf",
            "bytes": 100,
        }
        if missing_or_empty_id is not None:
            resp["public_id"] = missing_or_empty_id

        with patch("cloudinary.uploader.upload", return_value=resp):
            with pytest.raises(StorageUploadError, match="missing or empty required 'public_id' field"):
                storage.upload_file(VALID_PDF_SAMPLE)

    @pytest.mark.parametrize("missing_or_empty_url", [None, "", "   ", "\t", 123])
    def test_upload_missing_or_empty_secure_url_rejected(
        self,
        storage: CloudinaryStorage,
        missing_or_empty_url: Any,
    ) -> None:
        """Upload response missing or empty secure_url must raise StorageUploadError."""
        resp: dict[str, Any] = {
            "public_id": "nlp-research-agent/papers/sample_paper",
            "bytes": 100,
        }
        if missing_or_empty_url is not None:
            resp["secure_url"] = missing_or_empty_url

        with patch("cloudinary.uploader.upload", return_value=resp):
            with pytest.raises(StorageUploadError, match="missing or empty required 'secure_url' field"):
                storage.upload_file(VALID_PDF_SAMPLE)

    @pytest.mark.parametrize("invalid_bytes", [-1, -500, "100", 2.5, True, False])
    def test_upload_invalid_bytes_metadata_rejected(
        self,
        storage: CloudinaryStorage,
        invalid_bytes: Any,
    ) -> None:
        """Upload response with invalid bytes field must raise StorageUploadError."""
        resp = {
            "public_id": "nlp-research-agent/papers/sample_paper",
            "secure_url": "https://res.cloudinary.com/test/raw/upload/sample.pdf",
            "bytes": invalid_bytes,
        }

        with patch("cloudinary.uploader.upload", return_value=resp):
            with pytest.raises(StorageUploadError, match="invalid 'bytes' field"):
                storage.upload_file(VALID_PDF_SAMPLE)


class TestGetFileInfo:
    """Tests for metadata retrieval."""

    def test_successful_get_file_info(self, storage: CloudinaryStorage) -> None:
        """get_file_info queries Cloudinary Admin API with resource_type='raw'."""
        mock_resource = {
            "public_id": "sample_doc",
            "secure_url": "https://res.cloudinary.com/test/raw/upload/sample_doc",
            "bytes": 1024,
            "resource_type": "raw",
        }

        with patch("cloudinary.api.resource", return_value=mock_resource) as mock_resource_call:
            info = storage.get_file_info("sample_doc")
            mock_resource_call.assert_called_once_with("sample_doc", resource_type="raw")
            assert info["public_id"] == "sample_doc"
            assert info["secure_url"] == "https://res.cloudinary.com/test/raw/upload/sample_doc"

    @pytest.mark.parametrize("empty_id", ["", "   ", "\t"])
    def test_get_file_info_empty_id_rejected(
        self,
        storage: CloudinaryStorage,
        empty_id: str,
    ) -> None:
        """Empty or whitespace public ID must raise StorageError."""
        with pytest.raises(StorageError, match="public_id must be a non-empty string"):
            storage.get_file_info(empty_id)

    def test_get_file_info_not_found_mapped_to_storage_file_not_found_error(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """Cloudinary NotFound exception must be mapped to StorageFileNotFoundError."""
        with patch(
            "cloudinary.api.resource",
            side_effect=cloudinary.exceptions.NotFound("Resource not found"),
        ):
            with pytest.raises(StorageFileNotFoundError, match="not found on Cloudinary"):
                storage.get_file_info("non_existent_doc")

    def test_get_file_info_general_api_error_mapped(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """General Cloudinary API errors must be mapped to StorageError."""
        with patch(
            "cloudinary.api.resource",
            side_effect=cloudinary.exceptions.Error("Server error"),
        ):
            with pytest.raises(StorageError, match="Failed to retrieve metadata"):
                storage.get_file_info("some_doc")


class TestDownloadFile:
    """Tests for downloading raw PDF files."""

    @pytest.mark.parametrize("empty_id", ["", "   ", "\t"])
    def test_download_empty_id_rejected(
        self,
        storage: CloudinaryStorage,
        empty_id: str,
    ) -> None:
        """Empty or whitespace public ID must raise StorageDownloadError."""
        with pytest.raises(StorageDownloadError, match="public_id must be a non-empty string"):
            storage.download_file(empty_id)

    def test_download_success_via_primary_secure_url(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """Download fetches bytes using the authoritative secure_url and validates PDF header."""
        metadata = {
            "public_id": "test_paper",
            "secure_url": "https://res.cloudinary.com/test/raw/upload/test_paper.pdf",
        }

        mock_http_response = MagicMock()
        mock_http_response.read.return_value = VALID_PDF_SAMPLE
        mock_http_response.__enter__.return_value = mock_http_response
        mock_http_response.__exit__.return_value = None

        with (
            patch.object(storage, "get_file_info", return_value=metadata),
            patch("urllib.request.urlopen", return_value=mock_http_response) as mock_urlopen,
        ):
            data = storage.download_file("test_paper", timeout=15.0)

            assert data == VALID_PDF_SAMPLE
            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "https://res.cloudinary.com/test/raw/upload/test_paper.pdf"

    def test_download_fallback_to_signed_url_on_http_401_or_403(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """When primary secure_url returns 401 or 403, fallback to signed private_download_url."""
        metadata = {
            "public_id": "private_paper",
            "secure_url": "https://res.cloudinary.com/test/raw/upload/private_paper.pdf",
        }

        auth_error = urllib.error.HTTPError(
            url=metadata["secure_url"],
            code=403,
            msg="Forbidden",
            hdrs=MagicMock(),
            fp=io.BytesIO(b"Access Denied"),
        )

        mock_success_resp = MagicMock()
        mock_success_resp.read.return_value = VALID_PDF_SAMPLE
        mock_success_resp.__enter__.return_value = mock_success_resp
        mock_success_resp.__exit__.return_value = None

        def mock_urlopen_side_effect(req: urllib.request.Request, timeout: float = 30.0):
            if req.full_url == metadata["secure_url"]:
                raise auth_error
            return mock_success_resp

        signed_url = "https://api.cloudinary.com/v1_1/test/raw/download?signature=signed123"

        with (
            patch.object(storage, "get_file_info", return_value=metadata),
            patch("cloudinary.utils.private_download_url", return_value=signed_url) as mock_private_url,
            patch("urllib.request.urlopen", side_effect=mock_urlopen_side_effect),
        ):
            data = storage.download_file("private_paper")

            assert data == VALID_PDF_SAMPLE
            mock_private_url.assert_called_once_with("private_paper", format="", resource_type="raw")

    def test_download_fallback_when_secure_url_missing(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """When metadata has no secure_url, directly use signed private_download_url."""
        metadata = {
            "public_id": "paper_without_url",
            # No secure_url present
        }

        signed_url = "https://api.cloudinary.com/v1_1/test/raw/download?signature=direct123"
        mock_resp = MagicMock()
        mock_resp.read.return_value = VALID_PDF_SAMPLE
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with (
            patch.object(storage, "get_file_info", return_value=metadata),
            patch("cloudinary.utils.private_download_url", return_value=signed_url),
            patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen,
        ):
            data = storage.download_file("paper_without_url")
            assert data == VALID_PDF_SAMPLE
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == signed_url

    def test_download_empty_content_raises_storage_download_error(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """Download returning 0 bytes must raise StorageDownloadError."""
        metadata = {
            "public_id": "empty_file",
            "secure_url": "https://res.cloudinary.com/test/raw/upload/empty_file",
        }

        mock_empty_resp = MagicMock()
        mock_empty_resp.read.return_value = b""
        mock_empty_resp.__enter__.return_value = mock_empty_resp
        mock_empty_resp.__exit__.return_value = None

        with (
            patch.object(storage, "get_file_info", return_value=metadata),
            patch("urllib.request.urlopen", return_value=mock_empty_resp),
        ):
            with pytest.raises(StorageDownloadError, match="Downloaded resource 'empty_file' is empty"):
                storage.download_file("empty_file")

    def test_download_non_pdf_content_raises_storage_download_error(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """Downloaded content lacking %PDF- header must raise StorageDownloadError, converting InvalidPDFError."""
        metadata = {
            "public_id": "corrupted_paper",
            "secure_url": "https://res.cloudinary.com/test/raw/upload/corrupted_paper",
        }

        mock_html_resp = MagicMock()
        mock_html_resp.read.return_value = b"<html><body>404 Not Found</body></html>"
        mock_html_resp.__enter__.return_value = mock_html_resp
        mock_html_resp.__exit__.return_value = None

        with (
            patch.object(storage, "get_file_info", return_value=metadata),
            patch("urllib.request.urlopen", return_value=mock_html_resp),
        ):
            with pytest.raises(StorageDownloadError, match="not a valid PDF"):
                storage.download_file("corrupted_paper")

    def test_download_http_404_mapped_to_storage_file_not_found(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """HTTP 404 during download must raise StorageFileNotFoundError."""
        metadata = {
            "public_id": "missing_download_file",
            "secure_url": "https://res.cloudinary.com/test/raw/upload/missing_download_file",
        }

        not_found_error = urllib.error.HTTPError(
            url=metadata["secure_url"],
            code=404,
            msg="Not Found",
            hdrs=MagicMock(),
            fp=io.BytesIO(b"Not Found"),
        )

        with (
            patch.object(storage, "get_file_info", return_value=metadata),
            patch("urllib.request.urlopen", side_effect=not_found_error),
        ):
            with pytest.raises(StorageFileNotFoundError, match="not found at download URL"):
                storage.download_file("missing_download_file")

    def test_download_network_error_mapped(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """URLError or socket timeout during download must raise StorageDownloadError."""
        metadata = {
            "public_id": "timeout_file",
            "secure_url": "https://res.cloudinary.com/test/raw/upload/timeout_file",
        }

        with (
            patch.object(storage, "get_file_info", return_value=metadata),
            patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")),
        ):
            with pytest.raises(StorageDownloadError, match="Network error downloading 'timeout_file'"):
                storage.download_file("timeout_file")


class TestDeleteFile:
    """Tests for file deletion."""

    @pytest.mark.parametrize("empty_id", ["", "   ", "\t"])
    def test_delete_empty_id_rejected(
        self,
        storage: CloudinaryStorage,
        empty_id: str,
    ) -> None:
        """Empty or whitespace public ID must raise StorageDeleteError."""
        with pytest.raises(StorageDeleteError, match="public_id must be a non-empty string"):
            storage.delete_file(empty_id)

    def test_successful_deletion_returns_true(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """When Cloudinary returns result='ok', delete_file returns True."""
        with patch("cloudinary.uploader.destroy", return_value={"result": "ok"}) as mock_destroy:
            deleted = storage.delete_file("nlp-research-agent/papers/paper_to_delete")

            mock_destroy.assert_called_once_with(
                "nlp-research-agent/papers/paper_to_delete",
                resource_type="raw",
            )
            assert deleted is True

    def test_already_missing_file_deletion_returns_false_safely(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """When Cloudinary returns result='not found', delete_file returns False safely (idempotent)."""
        with patch("cloudinary.uploader.destroy", return_value={"result": "not found"}):
            deleted = storage.delete_file("already_absent_doc")
            assert deleted is False

    @pytest.mark.parametrize(
        "non_dict_resp",
        [None, "ok", [1, 2, 3], 100],
    )
    def test_delete_non_dict_response_rejected(
        self,
        storage: CloudinaryStorage,
        non_dict_resp: Any,
    ) -> None:
        """Non-dictionary delete response must raise StorageDeleteError."""
        with patch("cloudinary.uploader.destroy", return_value=non_dict_resp):
            with pytest.raises(StorageDeleteError, match="malformed response.*expected dict"):
                storage.delete_file("doc_with_non_dict_response")

    def test_delete_missing_result_field_rejected(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """Delete response missing 'result' field must raise StorageDeleteError."""
        with patch("cloudinary.uploader.destroy", return_value={"status": "ok"}):
            with pytest.raises(StorageDeleteError, match="missing 'result' field"):
                storage.delete_file("doc_without_result_field")

    @pytest.mark.parametrize("invalid_result_type", [123, None, True, ["ok"], {"nested": "result"}])
    def test_delete_non_string_result_rejected(
        self,
        storage: CloudinaryStorage,
        invalid_result_type: Any,
    ) -> None:
        """Delete response with non-string 'result' field must raise StorageDeleteError."""
        with patch("cloudinary.uploader.destroy", return_value={"result": invalid_result_type}):
            with pytest.raises(StorageDeleteError, match="must be a string"):
                storage.delete_file("doc_with_non_string_result")

    def test_unexpected_deletion_result_raises_storage_delete_error(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """When Cloudinary returns an unexpected string status, raise StorageDeleteError."""
        with patch("cloudinary.uploader.destroy", return_value={"result": "error"}):
            with pytest.raises(StorageDeleteError, match="Cloudinary deletion rejected"):
                storage.delete_file("problematic_doc")

    def test_deletion_sdk_exception_mapped(
        self,
        storage: CloudinaryStorage,
    ) -> None:
        """Cloudinary SDK exceptions during destroy must be mapped to StorageDeleteError."""
        with patch(
            "cloudinary.uploader.destroy",
            side_effect=cloudinary.exceptions.Error("Cloudinary connection lost"),
        ):
            with pytest.raises(StorageDeleteError, match="Cloudinary deletion call failed"):
                storage.delete_file("sample_doc")
