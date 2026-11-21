"""Unit tests for PaperRepository."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

import app.database.model_registry  # Ensure mappers are configured
from app.modules.papers.models import Paper, PaperFile
from app.modules.papers.repository import (
    PaperFileNotFoundError,
    PaperNotFoundError,
    PaperRepository,
    PaperRepositoryError,
    PaperValidationError,
    validate_and_normalize_sha256,
)


class TestPaperRepositoryInitialization:
    """Tests for repository initialization and session validation."""

    def test_init_with_valid_async_session(self) -> None:
        """Accepts an active AsyncSession."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)
        assert repo.session is mock_session

    @pytest.mark.parametrize("invalid_session", [None, 123, "session", [AsyncSession], MagicMock()])
    def test_init_with_invalid_session_raises_type_error(self, invalid_session: object) -> None:
        """Raises TypeError for non-AsyncSession input."""
        with pytest.raises(TypeError, match="session must be an AsyncSession"):
            PaperRepository(invalid_session)  # type: ignore[arg-type]


class TestPaperRepositoryPaperCRUD:
    """Tests for Paper querying and creation operations."""

    @pytest.mark.asyncio
    async def test_get_by_id_found(self) -> None:
        """Retrieves existing Paper by UUID."""
        mock_session = AsyncMock(spec=AsyncSession)
        paper_id = uuid.uuid4()
        expected = Paper(id=paper_id, title="Test Paper")

        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = expected
        mock_session.execute.return_value = mock_res

        repo = PaperRepository(mock_session)
        result = await repo.get_by_id(paper_id)

        assert result is expected
        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE papers.id =" in sql
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self) -> None:
        """Returns None if Paper does not exist."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_res

        repo = PaperRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    @pytest.mark.parametrize("invalid_id", ["not-uuid", 123, None])
    @pytest.mark.asyncio
    async def test_get_by_id_invalid_id_raises_type_error(self, invalid_id: object) -> None:
        """Raises TypeError when paper_id is not a UUID."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)
        with pytest.raises(TypeError, match="paper_id must be a uuid.UUID"):
            await repo.get_by_id(invalid_id)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_get_by_doi_normalizes_and_queries(self) -> None:
        """Normalizes raw DOI URL before querying."""
        mock_session = AsyncMock(spec=AsyncSession)
        expected = Paper(id=uuid.uuid4(), title="Attention Paper", doi="10.1000/182")

        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = expected
        mock_session.execute.return_value = mock_res

        repo = PaperRepository(mock_session)
        result = await repo.get_by_doi("https://doi.org/10.1000/182")

        assert result is expected
        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE papers.doi =" in sql

    @pytest.mark.parametrize("empty_doi", [None, "", "   ", "https://doi.org/"])
    @pytest.mark.asyncio
    async def test_get_by_doi_empty_returns_none_without_query(self, empty_doi: str | None) -> None:
        """Empty or invalid DOI returns None without executing a database query."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        result = await repo.get_by_doi(empty_doi)
        assert result is None
        mock_session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_checksum_found(self) -> None:
        """Queries Paper and PaperFile by file checksum."""
        mock_session = AsyncMock(spec=AsyncSession)
        paper = Paper(id=uuid.uuid4(), title="Checksum Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="cloudinary", storage_key="key")

        mock_res = MagicMock()
        mock_res.all.return_value = [(paper, paper_file)]
        mock_session.execute.return_value = mock_res

        repo = PaperRepository(mock_session)
        res = await repo.get_by_checksum("a" * 64)

        assert res is not None
        assert res[0] is paper
        assert res[1] is paper_file
        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE paper_files.checksum_sha256 =" in sql

    @pytest.mark.asyncio
    async def test_get_by_checksum_none_returns_none(self) -> None:
        """None checksum returns None without querying."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        res = await repo.get_by_checksum(None)
        assert res is None
        mock_session.execute.assert_not_called()

    @pytest.mark.parametrize("empty_hash", ["", "   ", "short", "a" * 63, "a" * 65, "g" * 64, 123, True, " " + ("a" * 64), ("a" * 64) + " "])
    @pytest.mark.asyncio
    async def test_get_by_checksum_invalid_raises_validation_error(self, empty_hash: object) -> None:
        """Invalid checksum format raises PaperValidationError."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        with pytest.raises(PaperValidationError, match="checksum_sha256"):
            await repo.get_by_checksum(empty_hash)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_get_by_checksum_normalizes_uppercase_to_lowercase(self) -> None:
        """Valid uppercase hex checksum is normalized to lowercase in query."""
        mock_session = AsyncMock(spec=AsyncSession)
        paper = Paper(id=uuid.uuid4(), title="Uppercase Checksum Paper")
        paper_file = PaperFile(id=uuid.uuid4(), paper_id=paper.id, storage_provider="c", storage_key="k")

        mock_res = MagicMock()
        mock_res.all.return_value = [(paper, paper_file)]
        mock_session.execute.return_value = mock_res

        repo = PaperRepository(mock_session)
        hex_upper = ("A" * 64)
        res = await repo.get_by_checksum(hex_upper)

        assert res is not None
        stmt = mock_session.execute.call_args[0][0]
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        assert "WHERE paper_files.checksum_sha256 =" in sql

    @pytest.mark.asyncio
    async def test_get_by_checksum_ambiguous_papers_raises_repository_error(self) -> None:
        """Raises PaperRepositoryError if the same checksum is associated with distinct papers."""
        mock_session = AsyncMock(spec=AsyncSession)
        paper_1 = Paper(id=uuid.uuid4(), title="Paper 1")
        paper_2 = Paper(id=uuid.uuid4(), title="Paper 2")
        file_1 = PaperFile(id=uuid.uuid4(), paper_id=paper_1.id, storage_provider="c", storage_key="k1")
        file_2 = PaperFile(id=uuid.uuid4(), paper_id=paper_2.id, storage_provider="c", storage_key="k2")

        mock_res = MagicMock()
        mock_res.all.return_value = [(paper_1, file_1), (paper_2, file_2)]
        mock_session.execute.return_value = mock_res

        repo = PaperRepository(mock_session)
        with pytest.raises(PaperRepositoryError, match="Ambiguous checksum .* matches files across 2 distinct papers"):
            await repo.get_by_checksum("a" * 64)

    @pytest.mark.asyncio
    async def test_create_paper_success(self) -> None:
        """Creates and flushes Paper entity."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        paper = await repo.create_paper(
            title="Attention Is All You Need",
            abstract="Transformer architecture.",
            doi="10.48550/arxiv.1706.03762",
            publication_year=2017,
            publication_date=date(2017, 6, 12),
            venue="NeurIPS",
            authors=[{"name": "Vaswani"}],
            metadata_={"tag": "nlp"},
        )

        assert isinstance(paper, Paper)
        assert paper.title == "Attention Is All You Need"
        assert paper.abstract == "Transformer architecture."
        assert paper.doi == "10.48550/arxiv.1706.03762"
        assert paper.publication_year == 2017
        assert paper.authors == [{"name": "Vaswani"}]
        assert paper.metadata_ == {"tag": "nlp"}

        mock_session.add.assert_called_once_with(paper)
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.parametrize("invalid_title", ["", "   ", None])
    @pytest.mark.asyncio
    async def test_create_paper_invalid_title_raises_validation_error(self, invalid_title: object) -> None:
        """Blank or non-string title raises PaperValidationError."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        with pytest.raises(PaperValidationError, match="title must be a non-blank string"):
            await repo.create_paper(title=invalid_title)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_update_paper_partial_updates(self) -> None:
        """Updates only supplied fields without overwriting unsupplied fields."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        paper = Paper(
            id=uuid.uuid4(),
            title="Initial Title",
            abstract="Initial Abstract",
            venue="Old Venue",
            publication_year=2020,
        )

        updated = await repo.update_paper(
            paper,
            venue="New Venue 2021",
            doi="10.1234/new-doi",
        )

        assert updated is paper
        assert updated.title == "Initial Title"  # Preserved
        assert updated.abstract == "Initial Abstract"  # Preserved
        assert updated.publication_year == 2020  # Preserved
        assert updated.venue == "New Venue 2021"  # Updated
        assert updated.doi == "10.1234/new-doi"  # Updated

        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()


class TestPaperRepositoryFileCRUD:
    """Tests for PaperFile CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_paper_file_success(self) -> None:
        """Creates and flushes PaperFile entity."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        paper_id = uuid.uuid4()
        pf = await repo.create_paper_file(
            paper_id=paper_id,
            storage_provider="cloudinary",
            storage_key="raw/papers/123",
            original_filename="paper.pdf",
            file_url="https://res.cloudinary.com/test.pdf",
            file_size_bytes=1024,
            checksum_sha256="b" * 64,
            is_primary=True,
            status="available",
        )

        assert isinstance(pf, PaperFile)
        assert pf.paper_id == paper_id
        assert pf.storage_provider == "cloudinary"
        assert pf.storage_key == "raw/papers/123"
        assert pf.file_size_bytes == 1024
        assert pf.checksum_sha256 == "b" * 64
        assert pf.is_primary is True
        assert pf.status == "available"

        mock_session.add.assert_called_once_with(pf)
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.parametrize("invalid_primary", [None, 1, 0, "True", "False", 123, []])
    @pytest.mark.asyncio
    async def test_create_paper_file_invalid_is_primary_raises_error(self, invalid_primary: object) -> None:
        """Raises PaperValidationError when is_primary is not a strict boolean."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        with pytest.raises(PaperValidationError, match="is_primary must be a boolean"):
            await repo.create_paper_file(
                paper_id=uuid.uuid4(),
                storage_provider="cloudinary",
                storage_key="key",
                original_filename="test.pdf",
                is_primary=invalid_primary,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("invalid_checksum", ["too_short", "g" * 64, "b" * 63, "c" * 65, 123])
    @pytest.mark.asyncio
    async def test_create_paper_file_invalid_checksum_raises_error(self, invalid_checksum: object) -> None:
        """Raises PaperValidationError when checksum_sha256 is not a valid 64-char hex string."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        with pytest.raises(PaperValidationError, match="checksum_sha256"):
            await repo.create_paper_file(
                paper_id=uuid.uuid4(),
                storage_provider="cloudinary",
                storage_key="key",
                original_filename="test.pdf",
                checksum_sha256=invalid_checksum,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("invalid_status", ["invalid", "pending", ""])
    @pytest.mark.asyncio
    async def test_create_paper_file_invalid_status_raises_error(self, invalid_status: str) -> None:
        """Raises PaperValidationError for unsupported file status."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        with pytest.raises(PaperValidationError, match="Invalid file status"):
            await repo.create_paper_file(
                paper_id=uuid.uuid4(),
                storage_provider="cloudinary",
                storage_key="key",
                original_filename="test.pdf",
                status=invalid_status,
            )

    @pytest.mark.parametrize("invalid_size", [0, -1, True, "1024"])
    @pytest.mark.asyncio
    async def test_create_paper_file_invalid_size_raises_error(self, invalid_size: object) -> None:
        """Raises PaperValidationError for non-positive or non-integer file size."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(mock_session)

        with pytest.raises(PaperValidationError, match="file_size_bytes must be a positive integer"):
            await repo.create_paper_file(
                paper_id=uuid.uuid4(),
                storage_provider="cloudinary",
                storage_key="key",
                original_filename="test.pdf",
                file_size_bytes=invalid_size,  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_get_file_by_id_found(self) -> None:
        """Retrieves PaperFile by UUID."""
        mock_session = AsyncMock(spec=AsyncSession)
        file_id = uuid.uuid4()
        expected = PaperFile(id=file_id, paper_id=uuid.uuid4(), storage_provider="c", storage_key="k")

        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = expected
        mock_session.execute.return_value = mock_res

        repo = PaperRepository(mock_session)
        pf = await repo.get_file_by_id(file_id)

        assert pf is expected

    @pytest.mark.asyncio
    async def test_update_file_status_success(self) -> None:
        """Updates file status and flushes."""
        mock_session = AsyncMock(spec=AsyncSession)
        file_id = uuid.uuid4()
        pf = PaperFile(id=file_id, paper_id=uuid.uuid4(), storage_provider="c", storage_key="k", status="uploaded")

        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = pf
        mock_session.execute.return_value = mock_res

        repo = PaperRepository(mock_session)
        updated = await repo.update_file_status(file_id, "available")

        assert updated.status == "available"
        mock_session.flush.assert_awaited_once()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_file_status_not_found_raises_error(self) -> None:
        """Raises PaperFileNotFoundError if file does not exist."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_res

        repo = PaperRepository(mock_session)
        with pytest.raises(PaperFileNotFoundError, match="not found"):
            await repo.update_file_status(uuid.uuid4(), "deleted")


class TestValidateAndNormalizeSha256:
    """Direct unit tests for validate_and_normalize_sha256."""

    def test_none_returns_none(self) -> None:
        """None input safely returns None."""
        assert validate_and_normalize_sha256(None) is None

    def test_valid_lowercase_hex_returned(self) -> None:
        """64 valid lowercase hex characters are returned unchanged."""
        val = "0123456789abcdef" * 4
        assert validate_and_normalize_sha256(val) == val

    def test_valid_uppercase_hex_normalized(self) -> None:
        """64 valid uppercase hex characters are normalized to lowercase."""
        val = "0123456789ABCDEF" * 4
        expected = "0123456789abcdef" * 4
        assert validate_and_normalize_sha256(val) == expected

    def test_mixed_case_hex_normalized(self) -> None:
        """64 mixed-case hex characters are normalized to lowercase."""
        val = "0123456789aBcDeF" * 4
        expected = "0123456789abcdef" * 4
        assert validate_and_normalize_sha256(val) == expected

    @pytest.mark.parametrize("leading_whitespace", [
        " " + ("a" * 64),
        "\t" + ("a" * 64),
        "\n" + ("a" * 64),
    ])
    def test_leading_whitespace_raises_validation_error(self, leading_whitespace: str) -> None:
        """Leading whitespace is rejected and not silently stripped."""
        with pytest.raises(PaperValidationError, match="leading or trailing whitespace"):
            validate_and_normalize_sha256(leading_whitespace)

    @pytest.mark.parametrize("trailing_whitespace", [
        ("a" * 64) + " ",
        ("a" * 64) + "\t",
        ("a" * 64) + "\n",
    ])
    def test_trailing_whitespace_raises_validation_error(self, trailing_whitespace: str) -> None:
        """Trailing whitespace is rejected and not silently stripped."""
        with pytest.raises(PaperValidationError, match="leading or trailing whitespace"):
            validate_and_normalize_sha256(trailing_whitespace)

    @pytest.mark.parametrize("short_val", [
        "",
        "a",
        "a" * 63,
        "0123456789abcdef",
    ])
    def test_short_values_raise_validation_error(self, short_val: str) -> None:
        """Values under 64 characters raise PaperValidationError."""
        with pytest.raises(PaperValidationError, match="must be exactly 64 characters"):
            validate_and_normalize_sha256(short_val)

    @pytest.mark.parametrize("long_val", [
        "a" * 65,
        "a" * 128,
    ])
    def test_long_values_raise_validation_error(self, long_val: str) -> None:
        """Values over 64 characters raise PaperValidationError."""
        with pytest.raises(PaperValidationError, match="must be exactly 64 characters"):
            validate_and_normalize_sha256(long_val)

    @pytest.mark.parametrize("non_hex", [
        ("g" * 64),
        ("z" * 64),
        ("a" * 63) + "X",
        ("a" * 63) + "-",
        ("a" * 63) + "!",
    ])
    def test_non_hex_characters_raise_validation_error(self, non_hex: str) -> None:
        """Non-hexadecimal characters raise PaperValidationError."""
        with pytest.raises(PaperValidationError, match="non-hexadecimal characters"):
            validate_and_normalize_sha256(non_hex)

    @pytest.mark.parametrize("non_string", [
        123,
        12.34,
        True,
        False,
        ["a" * 64],
        {"hash": "a" * 64},
        object(),
    ])
    def test_non_string_inputs_raise_validation_error(self, non_string: object) -> None:
        """Non-string inputs raise PaperValidationError."""
        with pytest.raises(PaperValidationError, match="checksum_sha256 must be a string"):
            validate_and_normalize_sha256(non_string)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_hash,description", [
        ("_" + ("a" * 63), "underscore at beginning"),
        (("a" * 32) + "_" + ("a" * 31), "underscore in middle"),
        (("a" * 63) + "_", "underscore at end"),
    ])
    def test_underscore_in_checksum_raises_validation_error(
        self, invalid_hash: str, description: str
    ) -> None:
        """Underscores are strictly rejected (even though Python int(s, 16) allows them)."""
        with pytest.raises(PaperValidationError, match="non-hexadecimal characters"):
            validate_and_normalize_sha256(invalid_hash)

    def test_valid_uppercase_and_lowercase_hex_checksums_pass(self) -> None:
        """Confirm valid uppercase and lowercase hexadecimal checksums pass and normalize properly."""
        lowercase = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        uppercase = "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
        assert validate_and_normalize_sha256(lowercase) == lowercase
        assert validate_and_normalize_sha256(uppercase) == lowercase


class TestPaperRepositoryDelete:
    """Tests for PaperRepository.delete_paper."""

    @pytest.mark.asyncio
    async def test_delete_paper_success(self) -> None:
        """Verify delete_paper extracts storage keys, deletes citations and paper entity."""
        from app.modules.ingestion.models import PaperChunk
        from app.modules.conversation.models import Message, MessageCitation

        session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(session)
        paper_id = uuid.uuid4()

        mock_paper = Paper(id=paper_id, title="Test Paper")
        mock_file = PaperFile(
            id=uuid.uuid4(),
            paper_id=paper_id,
            storage_provider="cloudinary",
            storage_key="raw/papers/test_key",
            original_filename="test.pdf",
            mime_type="application/pdf",
            status="available",
        )
        mock_paper.files = [mock_file]

        # 1. get_by_id returns mock_paper
        # 2. active jobs check returns empty
        # 3. citations query returns empty
        mock_res_paper = MagicMock()
        mock_res_paper.scalar_one_or_none.return_value = mock_paper

        mock_res_jobs = MagicMock()
        mock_res_jobs.scalars.return_value.first.return_value = None

        mock_res_citations = MagicMock()
        mock_res_citations.all.return_value = []

        session.execute = AsyncMock(side_effect=[mock_res_paper, mock_res_jobs, mock_res_citations])

        deleted, keys = await repo.delete_paper(paper_id)

        assert deleted is True
        assert keys == ["raw/papers/test_key"]
        session.delete.assert_called_once_with(mock_paper)
        session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_paper_not_found(self) -> None:
        """Verify delete_paper returns (False, []) when paper does not exist."""
        session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(session)
        paper_id = uuid.uuid4()

        mock_res = MagicMock()
        mock_res.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=mock_res)

        deleted, keys = await repo.delete_paper(paper_id)

        assert deleted is False
        assert keys == []
        session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_paper_active_job_conflict(self) -> None:
        """Verify delete_paper raises ActiveJobConflictError when ingestion job is active."""
        from app.core.exceptions import ActiveJobConflictError
        from app.modules.ingestion.models import IngestionJob

        session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(session)
        paper_id = uuid.uuid4()

        mock_paper = Paper(id=paper_id, title="Test Paper")
        mock_paper.files = []

        mock_job = IngestionJob(id=uuid.uuid4(), paper_id=paper_id, status="processing")

        mock_res_paper = MagicMock()
        mock_res_paper.scalar_one_or_none.return_value = mock_paper

        mock_res_jobs = MagicMock()
        mock_res_jobs.scalars.return_value.first.return_value = mock_job

        session.execute = AsyncMock(side_effect=[mock_res_paper, mock_res_jobs])

        with pytest.raises(ActiveJobConflictError, match="active ingestion job"):
            await repo.delete_paper(paper_id)

        session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_paper_snapshots_citations(self) -> None:
        """Verify delete_paper snapshots citation info into Message.metadata_ before deleting."""
        from app.modules.ingestion.models import PaperChunk
        from app.modules.conversation.models import Message, MessageCitation

        session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(session)
        paper_id = uuid.uuid4()

        mock_paper = Paper(id=paper_id, title="Attention Paper")
        mock_paper.files = []

        mock_chunk = PaperChunk(id=uuid.uuid4(), paper_id=paper_id, content="Transformer chunk", chunk_index=0)
        mock_msg = Message(id=uuid.uuid4(), content="Answer text", metadata_={})
        mock_cit = MessageCitation(
            id=uuid.uuid4(),
            message_id=mock_msg.id,
            chunk_id=mock_chunk.id,
            citation_order=1,
            relevance_score=4.5,
            cited_text="Transformer chunk",
            page_number=3,
        )

        mock_res_paper = MagicMock()
        mock_res_paper.scalar_one_or_none.return_value = mock_paper

        mock_res_jobs = MagicMock()
        mock_res_jobs.scalars.return_value.first.return_value = None

        mock_res_citations = MagicMock()
        mock_res_citations.all.return_value = [(mock_cit, mock_msg, mock_chunk)]

        session.execute = AsyncMock(side_effect=[mock_res_paper, mock_res_jobs, mock_res_citations])

        deleted, keys = await repo.delete_paper(paper_id)

        assert deleted is True
        # Verify citation was deleted
        session.delete.assert_any_call(mock_cit)
        # Verify paper was deleted
        session.delete.assert_any_call(mock_paper)
        # Verify message metadata has snapshot
        assert "historical_citations" in mock_msg.metadata_
        hist = mock_msg.metadata_["historical_citations"]
        assert len(hist) == 1
        assert hist[0]["paper_title"] == "Attention Paper"
        assert hist[0]["page_number"] == 3
        assert hist[0]["citation_order"] == 1

    @pytest.mark.asyncio
    async def test_delete_paper_invalid_uuid_raises_type_error(self) -> None:
        """Verify delete_paper raises TypeError for non-UUID input."""
        session = AsyncMock(spec=AsyncSession)
        repo = PaperRepository(session)

        with pytest.raises(TypeError, match="paper_id must be a uuid.UUID"):
            await repo.delete_paper("not-a-uuid")  # type: ignore[arg-type]

