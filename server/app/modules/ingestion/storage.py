"""Cloudinary storage integration for research paper PDF assets.

This module provides a centralized, synchronous storage client for uploading,
retrieving, inspecting, and deleting raw PDF files via the Cloudinary Python SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import urllib.error
import urllib.request

import cloudinary
import cloudinary.api
import cloudinary.exceptions
import cloudinary.uploader
import cloudinary.utils

from app.core.exceptions import (
    InvalidPDFError,
    StorageDeleteError,
    StorageDownloadError,
    StorageError,
    StorageFileNotFoundError,
    StorageUploadError,
)

DEFAULT_DOWNLOAD_TIMEOUT: float = 30.0


@dataclass(frozen=True)
class CloudinaryUploadResult:
    """Immutable result container representing an uploaded Cloudinary asset."""

    public_id: str
    secure_url: str
    byte_size: int
    format: str | None = None
    resource_type: str = "raw"
    created_at: str | None = None


class CloudinaryStorage:
    """Centralized synchronous storage manager for raw PDF files in Cloudinary."""

    def __init__(self) -> None:
        """Initialize the storage manager.

        Relies on the global Cloudinary SDK configuration established by app.core.config.
        Does not configure credentials inside this class.
        """

    @staticmethod
    def validate_pdf(content: bytes) -> None:
        """Validate that content is non-empty bytes with a valid %PDF- header.

        Note: This is a fast header signature check (%PDF-), not a complete structural
        or semantic validation of the entire PDF file.

        Raises:
            InvalidPDFError: If content is not a bytes instance (bytearray, str, etc. are rejected),
                             is empty, or lacks the %PDF- header.
        """
        if type(content) is not bytes:
            raise InvalidPDFError("PDF content must be a bytes instance.")
        if not content:
            raise InvalidPDFError("PDF content must not be empty.")
        if not content.startswith(b"%PDF-"):
            raise InvalidPDFError("Invalid PDF header: content must start with b'%PDF-'.")

    def upload_file(
        self,
        content: bytes,
        folder: str = "nlp-research-agent/papers",
    ) -> CloudinaryUploadResult:
        """Validate and upload raw PDF bytes to Cloudinary.

        Args:
            content: Raw bytes of the PDF file.
            folder: Cloudinary folder prefix (defaults to 'nlp-research-agent/papers').

        Returns:
            CloudinaryUploadResult containing public_id, secure_url, and file metadata.

        Raises:
            InvalidPDFError: If content fails PDF header validation.
            StorageUploadError: If the Cloudinary upload operation fails or returns
                                an incomplete/malformed response.
        """
        self.validate_pdf(content)

        try:
            response = cloudinary.uploader.upload(
                content,
                resource_type="raw",
                folder=folder,
            )
        except Exception as e:
            raise StorageUploadError(f"Cloudinary PDF upload failed: {e}") from e

        if not isinstance(response, dict):
            raise StorageUploadError(
                f"Cloudinary upload returned malformed response: expected dict, got {type(response).__name__}"
            )

        public_id = response.get("public_id")
        if not isinstance(public_id, str) or not public_id.strip():
            raise StorageUploadError(
                "Cloudinary upload response missing or empty required 'public_id' field."
            )

        secure_url = response.get("secure_url")
        if not isinstance(secure_url, str) or not secure_url.strip():
            raise StorageUploadError(
                "Cloudinary upload response missing or empty required 'secure_url' field."
            )

        byte_size = response.get("bytes")
        if byte_size is not None:
            if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0:
                raise StorageUploadError(
                    f"Cloudinary upload response contains invalid 'bytes' field: {byte_size!r}"
                )
        else:
            byte_size = len(content)

        return CloudinaryUploadResult(
            public_id=public_id.strip(),
            secure_url=secure_url.strip(),
            byte_size=byte_size,
            format=response.get("format"),
            resource_type=str(response.get("resource_type", "raw")),
            created_at=response.get("created_at"),
        )

    def get_file_info(self, public_id: str) -> dict[str, Any]:
        """Retrieve authoritative metadata for a raw Cloudinary asset.

        Args:
            public_id: The public ID of the resource.

        Returns:
            Dictionary containing Cloudinary resource metadata.

        Raises:
            StorageFileNotFoundError: If the resource does not exist on Cloudinary.
            StorageError: If the API request fails.
        """
        if not public_id or not public_id.strip():
            raise StorageError("public_id must be a non-empty string.")

        try:
            response = cloudinary.api.resource(
                public_id.strip(),
                resource_type="raw",
            )
            return dict(response)
        except cloudinary.exceptions.NotFound as e:
            raise StorageFileNotFoundError(f"Resource '{public_id}' not found on Cloudinary.") from e
        except Exception as e:
            raise StorageError(f"Failed to retrieve metadata for '{public_id}': {e}") from e

    def download_file(
        self,
        public_id: str,
        timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    ) -> bytes:
        """Download raw PDF bytes from Cloudinary and validate PDF header.

        Retrieval strategy:
        1. Queries get_file_info(public_id) for authoritative asset metadata.
        2. Attempts download using the authoritative secure_url.
        3. If secure_url access returns HTTP 401 or 403, falls back to a signed
           private download URL generated via cloudinary.utils.private_download_url.
        4. Validates that the downloaded content is non-empty and has a valid %PDF- header.

        Args:
            public_id: The Cloudinary public ID of the raw resource.
            timeout: Network request timeout in seconds (defaults to 30.0).

        Returns:
            Downloaded raw bytes of the file.

        Raises:
            StorageFileNotFoundError: If the resource is not found.
            StorageDownloadError: If download fails, times out, returns empty bytes,
                                  or content lacks a valid PDF header.
        """
        clean_public_id = public_id.strip() if public_id else ""
        if not clean_public_id:
            raise StorageDownloadError("public_id must be a non-empty string.")

        info = self.get_file_info(clean_public_id)
        primary_url = info.get("secure_url")

        def _fetch(url: str) -> bytes:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "nlp-research-agent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()

        data: bytes | None = None

        if primary_url:
            try:
                data = _fetch(primary_url)
            except urllib.error.HTTPError as http_err:
                if http_err.code in (401, 403):
                    # Fallback to signed private download URL
                    fallback_url = cloudinary.utils.private_download_url(
                        clean_public_id,
                        format="",
                        resource_type="raw",
                    )
                    try:
                        data = _fetch(fallback_url)
                    except Exception as fallback_err:
                        raise StorageDownloadError(
                            f"Signed fallback download failed for '{clean_public_id}': {fallback_err}"
                        ) from fallback_err
                elif http_err.code == 404:
                    raise StorageFileNotFoundError(
                        f"Resource '{clean_public_id}' not found at download URL: {http_err.reason}"
                    ) from http_err
                else:
                    raise StorageDownloadError(
                        f"HTTP error {http_err.code} downloading '{clean_public_id}': {http_err.reason}"
                    ) from http_err
            except (urllib.error.URLError, TimeoutError, OSError) as net_err:
                raise StorageDownloadError(
                    f"Network error downloading '{clean_public_id}': {net_err}"
                ) from net_err
        else:
            # No secure_url returned by metadata; attempt signed download URL directly
            fallback_url = cloudinary.utils.private_download_url(
                clean_public_id,
                format="",
                resource_type="raw",
            )
            try:
                data = _fetch(fallback_url)
            except Exception as e:
                raise StorageDownloadError(
                    f"Direct signed download failed for '{clean_public_id}': {e}"
                ) from e

        if not data:
            raise StorageDownloadError(f"Downloaded resource '{clean_public_id}' is empty.")

        try:
            self.validate_pdf(data)
        except InvalidPDFError as pdf_err:
            raise StorageDownloadError(
                f"Downloaded resource '{clean_public_id}' is not a valid PDF: {pdf_err}"
            ) from pdf_err

        return data

    def delete_file(self, public_id: str) -> bool:
        """Delete an uploaded raw PDF from Cloudinary by public ID.

        Handles already-missing resources safely:
        - If the asset was deleted ('result': 'ok'), returns True.
        - If the asset was already absent ('result': 'not found'), returns False
          without raising an error, ensuring safe idempotent deletion.

        Args:
            public_id: The Cloudinary public ID of the resource to delete.

        Returns:
            True if deleted, False if already absent.

        Raises:
            StorageDeleteError: If the deletion API call fails, or the response
                                is malformed or unexpected.
        """
        clean_public_id = public_id.strip() if public_id else ""
        if not clean_public_id:
            raise StorageDeleteError("public_id must be a non-empty string.")

        try:
            response = cloudinary.uploader.destroy(
                clean_public_id,
                resource_type="raw",
            )
        except Exception as e:
            raise StorageDeleteError(f"Cloudinary deletion call failed for '{clean_public_id}': {e}") from e

        if not isinstance(response, dict):
            raise StorageDeleteError(
                f"Cloudinary deletion returned malformed response for '{clean_public_id}': expected dict, got {type(response).__name__}"
            )

        if "result" not in response:
            raise StorageDeleteError(
                f"Cloudinary deletion response for '{clean_public_id}' missing 'result' field."
            )

        result_status = response.get("result")
        if not isinstance(result_status, str):
            raise StorageDeleteError(
                f"Cloudinary deletion response 'result' field for '{clean_public_id}' must be a string, got {type(result_status).__name__}."
            )

        if result_status == "ok":
            return True
        if result_status == "not found":
            return False

        raise StorageDeleteError(
            f"Cloudinary deletion rejected for '{clean_public_id}': result='{result_status}'"
        )
