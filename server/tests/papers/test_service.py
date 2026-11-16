"""Unit tests for PaperService deduplication and metadata enrichment."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest

from app.modules.ingestion.storage import CloudinaryUploadResult
from app.modules.papers.models import Paper, PaperFile
from app.modules.papers.repository import PaperRepository
from app.modules.papers.service import (
    PaperConflictError,
    PaperResolutionResult,
    PaperService,
    PaperValidationError,
)


@pytest.fixture
def mock_repository() -> AsyncMock:
    return AsyncMock(spec=PaperRepository)


class TestPaperServiceInitialization:
    """Tests for service initialization."""

    def test_init_with_valid_repository(self, mock_repository: AsyncMock) -> None:
        service = PaperService(mock_repository)
        assert service.repository is mock_repository

    @pytest.mark.parametrize("invalid_repo", [None, 123, "repo", MagicMock()])
    def test_init_with_invalid_repository_raises_type_error(self, invalid_repo: object) -> None:
        with pytest.raises(TypeError, match="repository must be a PaperRepository instance"):
            PaperService(invalid_repo)  # type: ignore[arg-type]


class TestPaperServiceDeduplication:
    """Tests for resolution and deduplication across DOI and file checksum."""

    @pytest.mark.asyncio
    async def test_same_doi_same_checksum_reuses_paper_and_file(self, mock_repository: AsyncMock) -> None:
        """Same DOI + same checksum: reuse existing paper and file; no duplicate file record."""
        paper = Paper(id=uuid.uuid4(), title="Existing Paper", doi="10.1000/182")
        paper_file = PaperFile(
            id=uuid.uuid4(),
            paper_id=paper.id,
            storage_provider="cloudinary",
            storage_key="raw/paper1",
            checksum_sha256="c" * 64,
            status="available",
        )

        mock_repository.get_by_doi.return_value = paper
        mock_repository.get_by_checksum.return_value = (paper, paper_file)

        service = PaperService(mock_repository)
        result = await service.resolve_or_create_paper(
            doi="10.1000/182",
            checksum_sha256="c" * 64,
        )

        assert isinstance(result, PaperResolutionResult)
        assert result.paper is paper
        assert result.matching_file is paper_file
        assert result.paper_reused is True
        assert result.file_reused is True

        mock_repository.create_paper.assert_not_called()

    @pytest.mark.asyncio
    async def test_same_doi_different_checksum_reuses_paper_allows_new_file(self, mock_repository: AsyncMock) -> None:
        """Same DOI + different checksum: reuse existing paper; allow a new file record."""
        paper = Paper(id=uuid.uuid4(), title="Existing Paper", doi="10.1000/182")

        mock_repository.get_by_doi.return_value = paper
        mock_repository.get_by_checksum.return_value = None  # No file with this new checksum

        service = PaperService(mock_repository)
        result = await service.resolve_or_create_paper(
            doi="10.1000/182",
            checksum_sha256="1" * 64,
        )

        assert result.paper is paper
        assert result.matching_file is None
        assert result.paper_reused is True
        assert result.file_reused is False

        mock_repository.create_paper.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_doi_same_checksum_reuses_paper_and_file(self, mock_repository: AsyncMock) -> None:
        """No DOI + same checksum: reuse the paper and file associated with that checksum."""
        paper = Paper(id=uuid.uuid4(), title="Checksum Paper", doi=None)
        paper_file = PaperFile(
            id=uuid.uuid4(),
            paper_id=paper.id,
            storage_provider="cloudinary",
            storage_key="raw/paper2",
            checksum_sha256="e" * 64,
            status="available",
        )

        mock_repository.get_by_checksum.return_value = (paper, paper_file)

        service = PaperService(mock_repository)
        result = await service.resolve_or_create_paper(
            doi=None,
            checksum_sha256="e" * 64,
        )

        assert result.paper is paper
        assert result.matching_file is paper_file
        assert result.paper_reused is True
        assert result.file_reused is True

        mock_repository.get_by_doi.assert_not_called()
        mock_repository.create_paper.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_doi_different_checksum_creates_new_paper(self, mock_repository: AsyncMock) -> None:
        """No DOI + different checksum: create a new paper."""
        mock_repository.get_by_checksum.return_value = None

        new_paper = Paper(id=uuid.uuid4(), title="Brand New Paper")
        mock_repository.create_paper.return_value = new_paper

        service = PaperService(mock_repository)
        result = await service.resolve_or_create_paper(
            title="Brand New Paper",
            doi=None,
            checksum_sha256="f" * 64,
        )

        assert result.paper is new_paper
        assert result.matching_file is None
        assert result.paper_reused is False
        assert result.file_reused is False

        mock_repository.create_paper.assert_awaited_once()

    @pytest.mark.parametrize("invalid_title", [
        None,
        "",
        "   ",
        123,
        True,
        False,
        ["Title"],
        {"title": "Title"},
        object(),
    ])
    @pytest.mark.asyncio
    async def test_create_new_paper_missing_or_invalid_title_raises_validation_error(
        self, mock_repository: AsyncMock, invalid_title: object
    ) -> None:
        """When neither DOI nor checksum matches, creating a new Paper requires a non-empty string title."""
        mock_repository.get_by_checksum.return_value = None

        service = PaperService(mock_repository)
        with pytest.raises(PaperValidationError, match="Cannot create a new Paper without a non-empty string title"):
            await service.resolve_or_create_paper(
                title=invalid_title,  # type: ignore[arg-type]
                doi=None,
                checksum_sha256="f" * 64,
            )

    @pytest.mark.asyncio
    async def test_paper_validation_error_imported_from_package_catches_service_errors(
        self, mock_repository: AsyncMock
    ) -> None:
        """Verify that PaperValidationError imported from app.modules.papers catches errors from PaperService."""
        from app.modules.papers import PaperValidationError as PackagePaperValidationError

        mock_repository.get_by_checksum.return_value = None
        service = PaperService(mock_repository)

        with pytest.raises(PackagePaperValidationError):
            await service.resolve_or_create_paper(
                title=None,
                doi=None,
                checksum_sha256="f" * 64,
            )

        with pytest.raises(PackagePaperValidationError):
            await service.resolve_or_create_paper(
                title="Valid Title",
                checksum_sha256="not-a-valid-checksum",
            )

    @pytest.mark.asyncio
    async def test_doi_and_checksum_conflict_raises_paper_conflict_error(self, mock_repository: AsyncMock) -> None:
        """If DOI and checksum point to different papers, raise PaperConflictError."""
        paper_a = Paper(id=uuid.uuid4(), title="Paper A", doi="10.1000/a")
        paper_b = Paper(id=uuid.uuid4(), title="Paper B", doi="10.1000/b")
        file_b = PaperFile(id=uuid.uuid4(), paper_id=paper_b.id, storage_provider="c", storage_key="k")

        mock_repository.get_by_doi.return_value = paper_a
        mock_repository.get_by_checksum.return_value = (paper_b, file_b)

        service = PaperService(mock_repository)
        with pytest.raises(PaperConflictError, match="Paper conflict: DOI .* matches paper .* but checksum .* matches paper"):
            await service.resolve_or_create_paper(
                doi="10.1000/a",
                checksum_sha256="b" * 64,
            )

    @pytest.mark.parametrize("file_status,is_reusable", [
        ("available", True),
        ("uploaded", True),
        ("failed", False),
        ("deleted", False),
    ])
    @pytest.mark.asyncio
    async def test_checksum_file_reuse_status_behavior(
        self, mock_repository: AsyncMock, file_status: str, is_reusable: bool
    ) -> None:
        """Deleted and failed files must not be reused, but the paper can still be reused."""
        paper = Paper(id=uuid.uuid4(), title="Checksum Paper", doi=None)
        paper_file = PaperFile(
            id=uuid.uuid4(),
            paper_id=paper.id,
            storage_provider="cloudinary",
            storage_key="raw/paper",
            checksum_sha256="d" * 64,
            status=file_status,
        )

        mock_repository.get_by_checksum.return_value = (paper, paper_file)

        service = PaperService(mock_repository)
        result = await service.resolve_or_create_paper(
            doi=None,
            checksum_sha256="d" * 64,
        )

        assert result.paper is paper
        assert result.paper_reused is True
        assert result.file_reused is is_reusable
        if is_reusable:
            assert result.matching_file is paper_file
        else:
            assert result.matching_file is None

    @pytest.mark.parametrize("invalid_hash", [
        "short",
        "b" * 63,
        "c" * 65,
        "g" * 64,
        123,
        True,
        " " + ("a" * 64),
        ("a" * 64) + " ",
    ])
    @pytest.mark.asyncio
    async def test_resolve_or_create_paper_invalid_checksum_raises_error(
        self, mock_repository: AsyncMock, invalid_hash: object
    ) -> None:
        """PaperService.resolve_or_create_paper raises PaperValidationError on invalid checksum format."""
        service = PaperService(mock_repository)
        with pytest.raises(PaperValidationError, match="checksum_sha256"):
            await service.resolve_or_create_paper(
                title="Title",
                checksum_sha256=invalid_hash,  # type: ignore[arg-type]
            )


class TestPaperServiceMetadataEnrichment:
    """Tests for metadata enrichment on resolved existing papers."""

    @pytest.mark.asyncio
    async def test_enrich_missing_metadata_fields(self, mock_repository: AsyncMock) -> None:
        """Enriches missing fields on existing paper without overwriting existing data."""
        existing_paper = Paper(
            id=uuid.uuid4(),
            title="Existing Title",
            abstract=None,  # Missing
            doi="10.1000/182",
            publication_year=None,  # Missing
            venue="Existing Venue",  # Present
            metadata_={"existing_key": 1},
        )

        mock_repository.get_by_doi.return_value = existing_paper
        mock_repository.get_by_checksum.return_value = None

        service = PaperService(mock_repository)
        await service.resolve_or_create_paper(
            doi="10.1000/182",
            title="New Incoming Title",  # Should not overwrite existing title
            abstract="New Extracted Abstract",  # Should be enriched
            publication_year=2023,  # Should be enriched
            venue="Different Venue",  # Should not overwrite existing venue
            metadata_={"existing_key": 999, "new_key": 2},  # Only new_key should be added
        )

        mock_repository.update_paper.assert_awaited_once()
        _, kwargs = mock_repository.update_paper.call_args

        assert kwargs["abstract"] == "New Extracted Abstract"
        assert kwargs["publication_year"] == 2023
        assert kwargs["metadata_"] == {"existing_key": 1, "new_key": 2}
        assert "title" not in kwargs  # Existing meaningful title preserved
        assert "venue" not in kwargs

    @pytest.mark.asyncio
    async def test_enrich_placeholder_title_with_valid_incoming_title(self, mock_repository: AsyncMock) -> None:
        """Allows placeholder titles such as 'Untitled Paper' to be enriched by a valid incoming title."""
        existing_paper = Paper(
            id=uuid.uuid4(),
            title="Untitled Paper",
            doi="10.1000/182",
        )

        mock_repository.get_by_doi.return_value = existing_paper
        mock_repository.get_by_checksum.return_value = None

        service = PaperService(mock_repository)
        await service.resolve_or_create_paper(
            doi="10.1000/182",
            title="Attention Is All You Need",
        )

        mock_repository.update_paper.assert_awaited_once()
        _, kwargs = mock_repository.update_paper.call_args
        assert kwargs["title"] == "Attention Is All You Need"

    @pytest.mark.asyncio
    async def test_does_not_call_update_if_nothing_to_enrich(self, mock_repository: AsyncMock) -> None:
        """Does not invoke update_paper if existing paper already has all incoming fields."""
        existing_paper = Paper(
            id=uuid.uuid4(),
            title="Existing Title",
            abstract="Existing Abstract",
            doi="10.1000/182",
            publication_year=2020,
            venue="Existing Venue",
            publisher="Existing Pub",
            authors=[{"name": "Vaswani"}],
            language="en",
            paper_type="research",
            canonical_url="https://example.com/paper",
            metadata_={"key": "val"},
        )

        mock_repository.get_by_doi.return_value = existing_paper
        mock_repository.get_by_checksum.return_value = None

        service = PaperService(mock_repository)
        await service.resolve_or_create_paper(
            doi="10.1000/182",
            abstract="Incoming Abstract",
            publication_year=2020,
            metadata_={"key": "val"},
        )

        mock_repository.update_paper.assert_not_called()


class TestPaperServiceFileRegistration:
    """Tests for registering uploaded paper files with Cloudinary metadata."""

    @pytest.mark.asyncio
    async def test_register_paper_file_success(self, mock_repository: AsyncMock) -> None:
        """Registers a PaperFile via the repository using CloudinaryUploadResult."""
        upload_result = CloudinaryUploadResult(
            public_id="raw/papers/paper_123",
            secure_url="https://res.cloudinary.com/test/paper_123.pdf",
            byte_size=2048,
            format="pdf",
        )
        paper_id = uuid.uuid4()
        expected_file = PaperFile(
            id=uuid.uuid4(),
            paper_id=paper_id,
            storage_provider="cloudinary",
            storage_key="raw/papers/paper_123",
        )
        mock_repository.create_paper_file.return_value = expected_file

        service = PaperService(mock_repository)
        created_file = await service.register_paper_file(
            paper_id=paper_id,
            upload_result=upload_result,
            original_filename="sample_paper.pdf",
            checksum_sha256="a" * 64,
            is_primary=True,
            status="available",
        )

        assert created_file is expected_file
        mock_repository.create_paper_file.assert_awaited_once_with(
            paper_id=paper_id,
            storage_provider="cloudinary",
            storage_key="raw/papers/paper_123",
            original_filename="sample_paper.pdf",
            mime_type="application/pdf",
            file_url="https://res.cloudinary.com/test/paper_123.pdf",
            file_size_bytes=2048,
            checksum_sha256="a" * 64,
            is_primary=True,
            status="available",
        )

    @pytest.mark.parametrize("invalid_result", [None, 123, "upload_res", MagicMock()])
    @pytest.mark.asyncio
    async def test_register_paper_file_invalid_upload_result_raises_type_error(
        self, mock_repository: AsyncMock, invalid_result: object
    ) -> None:
        """Raises TypeError if upload_result is not a CloudinaryUploadResult."""
        service = PaperService(mock_repository)
        with pytest.raises(TypeError, match="upload_result must be a CloudinaryUploadResult"):
            await service.register_paper_file(
                paper_id=uuid.uuid4(),
                upload_result=invalid_result,  # type: ignore[arg-type]
                original_filename="test.pdf",
            )
