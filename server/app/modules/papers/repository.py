"""Database repository for Paper and PaperFile entities."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import app.database.model_registry  # noqa: F401 - ensures cross-module relationship mappers are registered
from app.core.exceptions import ActiveJobConflictError
from app.modules.conversation.models import Message, MessageCitation
from app.modules.ingestion.models import IngestionJob, PaperChunk
from app.modules.papers.models import Paper, PaperFile, normalize_doi


SUPPORTED_FILE_STATUSES: frozenset[str] = frozenset({"uploaded", "available", "failed", "deleted"})
REUSABLE_FILE_STATUSES: frozenset[str] = frozenset({"uploaded", "available"})
HEX_DIGITS: frozenset[str] = frozenset("0123456789abcdefABCDEF")


def validate_and_normalize_sha256(checksum: str | None) -> str | None:
    """Validate that a checksum is None or a valid 64-char hexadecimal SHA-256 digest string.

    Requires exactly 64 characters from [0-9a-fA-F]. Rejects leading/trailing whitespace,
    underscores, non-hexadecimal characters, and non-string inputs. Returns the normalized
    lowercase 64-char hex string, or None if checksum is None.

    Raises:
        PaperValidationError: If checksum is not a string, not exactly 64 characters,
            contains leading/trailing whitespace, or contains non-hex characters.
    """
    if checksum is None:
        return None

    if not isinstance(checksum, str):
        raise PaperValidationError(f"checksum_sha256 must be a string, got {type(checksum).__name__}")

    if checksum.strip() != checksum:
        raise PaperValidationError(
            f"checksum_sha256 must not contain leading or trailing whitespace: {checksum!r}"
        )

    if len(checksum) != 64:
        raise PaperValidationError(
            f"checksum_sha256 must be exactly 64 characters, got {len(checksum)} (value: {checksum!r})"
        )

    if any(c not in HEX_DIGITS for c in checksum):
        raise PaperValidationError(f"checksum_sha256 contains non-hexadecimal characters: {checksum!r}")

    return checksum.lower()


class PaperRepositoryError(Exception):
    """Base exception for paper repository errors."""


class PaperNotFoundError(PaperRepositoryError):
    """Raised when a requested Paper is not found."""


class PaperFileNotFoundError(PaperRepositoryError):
    """Raised when a requested PaperFile is not found."""


class PaperValidationError(PaperRepositoryError):
    """Raised when invalid parameters or arguments are supplied to the repository."""


class PaperRepository:
    """Asynchronous repository for Paper and PaperFile data access.

    Follows the application's transaction pattern: accepts an external AsyncSession,
    never initiates commit() or rollback(), and uses flush() when generated database
    values (such as generated UUIDs or server timestamps) are needed.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository with an active asynchronous database session.

        Args:
            session: Active SQLAlchemy AsyncSession.

        Raises:
            TypeError: If session is not an instance of AsyncSession.
        """
        if not isinstance(session, AsyncSession):
            raise TypeError(
                f"session must be an AsyncSession, got {type(session).__name__}"
            )
        self.session: AsyncSession = session

    async def get_by_id(
        self,
        paper_id: uuid.UUID,
        *,
        include_files: bool = False,
    ) -> Paper | None:
        """Retrieve a Paper by its primary key UUID.

        Args:
            paper_id: UUID of the research paper.
            include_files: Whether to eagerly load the files relationship.

        Returns:
            The Paper entity if found, or None.

        Raises:
            TypeError: If paper_id is not a uuid.UUID instance.
        """
        if not isinstance(paper_id, uuid.UUID):
            raise TypeError(f"paper_id must be a uuid.UUID, got {type(paper_id).__name__}")

        stmt = select(Paper).where(Paper.id == paper_id)
        if include_files:
            stmt = stmt.options(selectinload(Paper.files))

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_paper(self, paper_id: uuid.UUID) -> tuple[bool, list[str]]:
        """Delete a research paper, cascading all chunks, files, and jobs, while preserving citation snapshots.

        Args:
            paper_id: UUID of the research paper.

        Returns:
            Tuple of (deleted: bool, storage_keys: list[str]).
            If paper is not found, returns (False, []).

        Raises:
            TypeError: If paper_id is not a uuid.UUID instance.
            ActiveJobConflictError: If the paper is currently undergoing active ingestion.
        """
        if not isinstance(paper_id, uuid.UUID):
            raise TypeError(f"paper_id must be a uuid.UUID, got {type(paper_id).__name__}")

        paper = await self.get_by_id(paper_id, include_files=True)
        if paper is None:
            return False, []

        stmt_active_jobs = select(IngestionJob).where(
            IngestionJob.paper_id == paper_id,
            IngestionJob.status.in_(["pending", "processing"]),
        )
        active_job_res = await self.session.execute(stmt_active_jobs)
        active_job = active_job_res.scalars().first()
        if active_job is not None:
            raise ActiveJobConflictError(
                f"Cannot delete paper '{paper_id}': active ingestion job '{active_job.id}' is currently in status '{active_job.status}'."
            )

        storage_keys = [
            f.storage_key
            for f in (paper.files or [])
            if f.storage_provider.lower() == "cloudinary" and f.storage_key
        ]

        # Snapshot citations into message metadata before deleting to prevent RESTRICT fk violation
        stmt_citations = (
            select(MessageCitation, Message, PaperChunk)
            .join(PaperChunk, MessageCitation.chunk_id == PaperChunk.id)
            .join(Message, MessageCitation.message_id == Message.id)
            .where(PaperChunk.paper_id == paper_id)
        )
        res_citations = await self.session.execute(stmt_citations)
        citations_to_delete = res_citations.all()

        for cit, msg, chunk in citations_to_delete:
            meta = dict(msg.metadata_ or {})
            history = list(meta.get("historical_citations", []))
            history.append({
                "citation_order": cit.citation_order,
                "page_number": cit.page_number,
                "relevance_score": cit.relevance_score,
                "cited_text": cit.cited_text,
                "paper_title": paper.title,
                "paper_id": str(paper.id),
                "chunk_id": str(chunk.id),
                "deleted_at": datetime.now(timezone.utc).isoformat(),
            })
            meta["historical_citations"] = history
            msg.metadata_ = meta
            await self.session.delete(cit)

        await self.session.delete(paper)
        await self.session.flush()

        return True, storage_keys

    async def get_by_doi(
        self,
        doi: str | None,
        *,
        include_files: bool = False,
    ) -> Paper | None:
        """Retrieve a Paper by its Digital Object Identifier (DOI).

        Normalizes the incoming DOI string before querying. If the normalized
        DOI is None or empty, returns None immediately without executing a query.

        Args:
            doi: Raw DOI string (e.g., '10.48550/arXiv.1706.03762' or full URL).
            include_files: Whether to eagerly load the files relationship.

        Returns:
            The Paper entity if found, or None.
        """
        cleaned_doi = normalize_doi(doi)
        if cleaned_doi is None:
            return None

        stmt = select(Paper).where(Paper.doi == cleaned_doi)
        if include_files:
            stmt = stmt.options(selectinload(Paper.files))

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_checksum(
        self,
        checksum_sha256: str | None,
    ) -> tuple[Paper, PaperFile] | None:
        """Find a PaperFile by its SHA-256 checksum and return the associated Paper and PaperFile.

        Args:
            checksum_sha256: Hexadecimal SHA-256 digest string of the file.

        Returns:
            A tuple of (Paper, PaperFile) if a matching file is found, or None.

        Raises:
            PaperValidationError: If checksum_sha256 is not a valid 64-character hex string when supplied.
            PaperRepositoryError: If duplicate checksum records across different papers create an ambiguous conflict.
        """
        if checksum_sha256 is None:
            return None

        cleaned_checksum = validate_and_normalize_sha256(checksum_sha256)
        if cleaned_checksum is None:
            return None

        stmt = (
            select(Paper, PaperFile)
            .join(PaperFile, Paper.id == PaperFile.paper_id)
            .where(PaperFile.checksum_sha256 == cleaned_checksum)
            .order_by(PaperFile.created_at.asc())
        )

        result = await self.session.execute(stmt)
        rows = list(result.all())
        if not rows:
            return None

        # Verify that all files with this checksum belong to the same paper
        distinct_paper_ids = {row[0].id for row in rows}
        if len(distinct_paper_ids) > 1:
            raise PaperRepositoryError(
                f"Ambiguous checksum '{cleaned_checksum}' matches files across {len(distinct_paper_ids)} distinct papers."
            )

        # Prefer the first reusable (available/uploaded) file if present, otherwise first row
        reusable_row = next((r for r in rows if r[1].status in REUSABLE_FILE_STATUSES), rows[0])
        return reusable_row[0], reusable_row[1]

    async def create_paper(
        self,
        title: str,
        *,
        abstract: str | None = None,
        doi: str | None = None,
        publication_date: date | None = None,
        publication_year: int | None = None,
        venue: str | None = None,
        publisher: str | None = None,
        authors: list[dict[str, Any]] | None = None,
        language: str | None = None,
        paper_type: str | None = None,
        canonical_url: str | None = None,
        metadata_: dict[str, Any] | None = None,
    ) -> Paper:
        """Create and flush a new Paper entity.

        Args:
            title: Title of research paper (must be non-blank).
            abstract: Paper abstract text.
            doi: DOI identifier.
            publication_date: Publication date.
            publication_year: Publication year.
            venue: Conference / journal venue.
            publisher: Publishing organization.
            authors: List of author dictionaries.
            language: Language code.
            paper_type: Document classification.
            canonical_url: Official publication URL.
            metadata_: Additional structured metadata.

        Returns:
            The flushed Paper entity with generated UUID.

        Raises:
            PaperValidationError: If title is empty or blank, or argument types are invalid.
        """
        if not isinstance(title, str) or not title.strip():
            raise PaperValidationError("Paper title must be a non-blank string.")

        if publication_year is not None:
            if isinstance(publication_year, bool) or not isinstance(publication_year, int):
                raise PaperValidationError(f"publication_year must be an integer, got {publication_year!r}")

        if authors is not None and not isinstance(authors, list):
            raise PaperValidationError(f"authors must be a list of dictionaries, got {type(authors).__name__}")

        if metadata_ is not None and not isinstance(metadata_, dict):
            raise PaperValidationError(f"metadata_ must be a dictionary, got {type(metadata_).__name__}")

        paper = Paper(
            title=title.strip(),
            abstract=abstract,
            doi=normalize_doi(doi),
            publication_date=publication_date,
            publication_year=publication_year,
            venue=venue,
            publisher=publisher,
            authors=authors if authors is not None else [],
            language=language,
            paper_type=paper_type,
            canonical_url=canonical_url,
            metadata_=metadata_ if metadata_ is not None else {},
        )

        self.session.add(paper)
        await self.session.flush()
        return paper

    async def update_paper(
        self,
        paper: Paper,
        *,
        title: str | None = None,
        abstract: str | None = None,
        doi: str | None = None,
        publication_date: date | None = None,
        publication_year: int | None = None,
        venue: str | None = None,
        publisher: str | None = None,
        authors: list[dict[str, Any]] | None = None,
        language: str | None = None,
        paper_type: str | None = None,
        canonical_url: str | None = None,
        metadata_: dict[str, Any] | None = None,
    ) -> Paper:
        """Update supplied fields of an existing Paper without overwriting omitted fields.

        Args:
            paper: Paper entity to update.
            title: Updated title (if provided, must be non-blank).
            abstract: Updated abstract.
            doi: Updated DOI.
            publication_date: Updated publication date.
            publication_year: Updated publication year.
            venue: Updated venue.
            publisher: Updated publisher.
            authors: Updated authors list.
            language: Updated language code.
            paper_type: Updated paper type.
            canonical_url: Updated canonical URL.
            metadata_: Updated metadata dict (replaces or augments).

        Returns:
            The updated and flushed Paper entity.

        Raises:
            TypeError: If paper is not a Paper instance.
            PaperValidationError: If supplied values fail validation.
        """
        if not isinstance(paper, Paper):
            raise TypeError(f"paper must be a Paper instance, got {type(paper).__name__}")

        if title is not None:
            if not isinstance(title, str) or not title.strip():
                raise PaperValidationError("Paper title must be a non-blank string.")
            paper.title = title.strip()

        if abstract is not None:
            paper.abstract = abstract

        if doi is not None:
            paper.doi = normalize_doi(doi)

        if publication_date is not None:
            paper.publication_date = publication_date

        if publication_year is not None:
            if isinstance(publication_year, bool) or not isinstance(publication_year, int):
                raise PaperValidationError(f"publication_year must be an integer, got {publication_year!r}")
            paper.publication_year = publication_year

        if venue is not None:
            paper.venue = venue

        if publisher is not None:
            paper.publisher = publisher

        if authors is not None:
            if not isinstance(authors, list):
                raise PaperValidationError(f"authors must be a list, got {type(authors).__name__}")
            paper.authors = authors

        if language is not None:
            paper.language = language

        if paper_type is not None:
            paper.paper_type = paper_type

        if canonical_url is not None:
            paper.canonical_url = canonical_url

        if metadata_ is not None:
            if not isinstance(metadata_, dict):
                raise PaperValidationError(f"metadata_ must be a dictionary, got {type(metadata_).__name__}")
            paper.metadata_ = metadata_

        await self.session.flush()
        return paper

    async def create_paper_file(
        self,
        paper_id: uuid.UUID,
        storage_provider: str,
        storage_key: str,
        original_filename: str,
        mime_type: str = "application/pdf",
        *,
        file_url: str | None = None,
        file_size_bytes: int | None = None,
        checksum_sha256: str | None = None,
        is_primary: bool = False,
        status: str = "available",
    ) -> PaperFile:
        """Create and flush a new PaperFile entity.

        Args:
            paper_id: Parent Paper UUID.
            storage_provider: Storage provider name (e.g. 'cloudinary').
            storage_key: Provider storage key / public_id.
            original_filename: Original file name.
            mime_type: File MIME type (default 'application/pdf').
            file_url: Direct secure or accessible file URL.
            file_size_bytes: Size in bytes (must be > 0 if provided).
            checksum_sha256: SHA-256 hexadecimal hash.
            is_primary: Whether this is the primary file for the paper.
            status: Initial file status ('uploaded' | 'available' | 'failed' | 'deleted').

        Returns:
            The flushed PaperFile entity with generated UUID.

        Raises:
            TypeError: If paper_id is not a uuid.UUID.
            PaperValidationError: If status is unsupported, file_size_bytes <= 0, or required strings are empty.
        """
        if not isinstance(paper_id, uuid.UUID):
            raise TypeError(f"paper_id must be a uuid.UUID, got {type(paper_id).__name__}")

        if not isinstance(storage_provider, str) or not storage_provider.strip():
            raise PaperValidationError("storage_provider must be a non-empty string.")

        if not isinstance(storage_key, str) or not storage_key.strip():
            raise PaperValidationError("storage_key must be a non-empty string.")

        if not isinstance(original_filename, str) or not original_filename.strip():
            raise PaperValidationError("original_filename must be a non-empty string.")

        if status not in SUPPORTED_FILE_STATUSES:
            raise PaperValidationError(
                f"Invalid file status '{status}'. Supported statuses: {sorted(SUPPORTED_FILE_STATUSES)}"
            )

        if file_size_bytes is not None:
            if isinstance(file_size_bytes, bool) or not isinstance(file_size_bytes, int) or file_size_bytes <= 0:
                raise PaperValidationError(f"file_size_bytes must be a positive integer, got {file_size_bytes!r}")

        if not isinstance(is_primary, bool):
            raise PaperValidationError(f"is_primary must be a boolean, got {type(is_primary).__name__} ({is_primary!r})")

        cleaned_checksum = validate_and_normalize_sha256(checksum_sha256)

        paper_file = PaperFile(
            paper_id=paper_id,
            storage_provider=storage_provider.strip(),
            storage_key=storage_key.strip(),
            file_url=file_url,
            original_filename=original_filename.strip(),
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            checksum_sha256=cleaned_checksum,
            is_primary=is_primary,
            status=status,
        )

        self.session.add(paper_file)
        await self.session.flush()
        return paper_file

    async def get_file_by_id(self, file_id: uuid.UUID) -> PaperFile | None:
        """Retrieve a PaperFile by its primary key UUID.

        Args:
            file_id: UUID of the PaperFile.

        Returns:
            The PaperFile entity if found, or None.

        Raises:
            TypeError: If file_id is not a uuid.UUID instance.
        """
        if not isinstance(file_id, uuid.UUID):
            raise TypeError(f"file_id must be a uuid.UUID, got {type(file_id).__name__}")

        stmt = select(PaperFile).where(PaperFile.id == file_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_file_status(
        self,
        file_id: uuid.UUID,
        status: str,
    ) -> PaperFile:
        """Update the status of a PaperFile after validating supported values.

        Args:
            file_id: UUID of the PaperFile.
            status: New status ('uploaded' | 'available' | 'failed' | 'deleted').

        Returns:
            The updated and flushed PaperFile entity.

        Raises:
            TypeError: If file_id is not a uuid.UUID instance.
            PaperValidationError: If status is not a valid status string.
            PaperFileNotFoundError: If no file with file_id exists.
        """
        if not isinstance(file_id, uuid.UUID):
            raise TypeError(f"file_id must be a uuid.UUID, got {type(file_id).__name__}")

        if status not in SUPPORTED_FILE_STATUSES:
            raise PaperValidationError(
                f"Invalid file status '{status}'. Supported statuses: {sorted(SUPPORTED_FILE_STATUSES)}"
            )

        paper_file = await self.get_file_by_id(file_id)
        if paper_file is None:
            raise PaperFileNotFoundError(f"PaperFile with id {file_id} not found.")

        paper_file.status = status
        await self.session.flush()
        return paper_file

    async def list_papers(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> tuple[list[tuple[Paper, int, int | None]], int]:
        """Retrieve paginated papers with associated files, ingestion jobs, chunk count, and page count.

        Avoids N+1 queries by executing a single query for papers and correlated subqueries
        for chunk count and max page, and selectinload for files and ingestion_jobs.

        Args:
            limit: Maximum number of papers to return (clamped 1..100).
            offset: Number of items to skip.
            search: Optional substring to search across title and DOI.

        Returns:
            Tuple of (list of (Paper, chunk_count, page_count) tuples, total_count).
        """
        safe_limit = max(1, min(limit, 100))
        safe_offset = max(0, offset)

        chunk_count_subq = (
            select(func.count(PaperChunk.id))
            .where(PaperChunk.paper_id == Paper.id, PaperChunk.is_current.is_(True))
            .scalar_subquery()
        )
        max_page_subq = (
            select(func.max(PaperChunk.page_end))
            .where(PaperChunk.paper_id == Paper.id, PaperChunk.is_current.is_(True))
            .scalar_subquery()
        )

        base_filter = []
        if search and search.strip():
            term = f"%{search.strip()}%"
            base_filter.append(Paper.title.ilike(term) | Paper.doi.ilike(term))

        # Total count query
        count_stmt = select(func.count(Paper.id))
        if base_filter:
            count_stmt = count_stmt.where(*base_filter)
        total_res = await self.session.execute(count_stmt)
        total_count = total_res.scalar() or 0

        # Items query
        stmt = (
            select(Paper, chunk_count_subq.label("chunk_count"), max_page_subq.label("max_page"))
            .options(
                selectinload(Paper.files),
                selectinload(Paper.ingestion_jobs),
            )
            .order_by(Paper.created_at.desc())
            .limit(safe_limit)
            .offset(safe_offset)
        )
        if base_filter:
            stmt = stmt.where(*base_filter)

        res = await self.session.execute(stmt)
        rows = [
            (row[0], int(row[1] or 0), int(row[2]) if row[2] is not None else None)
            for row in res.all()
        ]
        return rows, total_count

    async def get_paper_details(
        self,
        paper_id: uuid.UUID,
    ) -> tuple[Paper, int, int | None] | None:
        """Retrieve full details of a single paper with files, ingestion jobs, chunk count, and page count."""
        if not isinstance(paper_id, uuid.UUID):
            raise TypeError(f"paper_id must be a uuid.UUID, got {type(paper_id).__name__}")

        chunk_count_subq = (
            select(func.count(PaperChunk.id))
            .where(PaperChunk.paper_id == Paper.id, PaperChunk.is_current.is_(True))
            .scalar_subquery()
        )
        max_page_subq = (
            select(func.max(PaperChunk.page_end))
            .where(PaperChunk.paper_id == Paper.id, PaperChunk.is_current.is_(True))
            .scalar_subquery()
        )

        stmt = (
            select(Paper, chunk_count_subq.label("chunk_count"), max_page_subq.label("max_page"))
            .options(
                selectinload(Paper.files),
                selectinload(Paper.ingestion_jobs),
            )
            .where(Paper.id == paper_id)
        )

        res = await self.session.execute(stmt)
        row = res.first()
        if row is None:
            return None
        return (row[0], int(row[1] or 0), int(row[2]) if row[2] is not None else None)
