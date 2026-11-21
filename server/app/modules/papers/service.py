"""Service layer for research paper resolution, metadata enrichment, and file registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import logging
from typing import Any
import uuid

from app.core.exceptions import ActiveJobConflictError
from app.modules.ingestion.models import IngestionJob
from app.modules.ingestion.storage import CloudinaryStorage, CloudinaryUploadResult
from app.modules.papers.models import Paper, PaperFile, normalize_doi
from app.modules.papers.repository import (
    REUSABLE_FILE_STATUSES,
    PaperNotFoundError,
    PaperRepository,
    PaperRepositoryError,
    PaperValidationError,
    validate_and_normalize_sha256,
)
from app.modules.papers.schemas import (
    PaperDetailResponse,
    PaperFileItem,
    PaperListItemResponse,
    PaperListResponse,
)
from app.modules.retrieval.repository import RetrievalRepository
from app.modules.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


class PaperServiceError(Exception):
    """Base exception for paper service errors."""


class PaperConflictError(PaperServiceError):
    """Raised when DOI lookup and checksum lookup point to two different existing papers."""


@dataclass(frozen=True)
class PaperResolutionResult:
    """Immutable result container representing the outcome of a paper resolution operation.

    Attributes:
        paper: The resolved or newly created Paper entity.
        matching_file: The matching PaperFile if a file with the exact checksum already exists.
        paper_reused: True if an existing Paper was resolved via DOI or checksum; False if newly created.
        file_reused: True if an existing PaperFile matching the checksum was found.
    """

    paper: Paper
    matching_file: PaperFile | None
    paper_reused: bool
    file_reused: bool


class PaperService:
    """Coordinates research paper business rules, resolution, deduplication, and file registration.

    Delegates all persistence operations to an injected PaperRepository.
    Does not initiate database commits, rollbacks, or storage uploads/downloads.
    """

    def __init__(
        self,
        repository: PaperRepository,
        retrieval_service: RetrievalService | None = None,
        storage: CloudinaryStorage | None = None,
    ) -> None:
        """Initialize the PaperService with an injected PaperRepository, optional RetrievalService, and Storage.

        Args:
            repository: Active PaperRepository bound to a database session.
            retrieval_service: Optional application-scoped RetrievalService for index refreshing.
            storage: Optional CloudinaryStorage client for asset cleanup.

        Raises:
            TypeError: If repository is not a PaperRepository instance.
        """
        if not isinstance(repository, PaperRepository):
            raise TypeError(
                f"repository must be a PaperRepository instance, got {type(repository).__name__}"
            )
        self.repository: PaperRepository = repository
        self.retrieval_service: RetrievalService | None = retrieval_service
        self.storage: CloudinaryStorage = storage if storage is not None else CloudinaryStorage()

    async def resolve_or_create_paper(
        self,
        *,
        title: str | None = None,
        doi: str | None = None,
        checksum_sha256: str | None = None,
        abstract: str | None = None,
        publication_date: date | None = None,
        publication_year: int | None = None,
        venue: str | None = None,
        publisher: str | None = None,
        authors: list[dict[str, Any]] | None = None,
        language: str | None = None,
        paper_type: str | None = None,
        canonical_url: str | None = None,
        metadata_: dict[str, Any] | None = None,
    ) -> PaperResolutionResult:
        """Resolve an existing research paper or create a new one based on DOI and file checksum.

        Deduplication and Resolution Rules:
        1. Normalizes the incoming DOI (if provided).
        2. Searches by normalized DOI first.
        3. Searches by file checksum (if provided).
        4. Detects cross-paper conflicts: If DOI matches Paper A and checksum matches Paper B (A != B),
           raises PaperConflictError.
        5. If DOI matches:
           - Reuses the DOI-matched paper.
           - If a file with the checksum exists and belongs to this paper, reuses the file.
           - Enriches missing metadata fields on the existing paper.
        6. If no DOI match, but checksum matches:
           - Reuses the checksum-associated paper and the matching file.
           - Enriches missing metadata fields (including updating DOI if previously missing).
        7. If neither matches:
           - Creates a new Paper entity using the supplied metadata (requires a non-blank title).
           - Returns paper_reused=False, file_reused=False.

        Args:
            title: Research paper title.
            doi: Digital Object Identifier.
            checksum_sha256: SHA-256 hexadecimal checksum of source PDF.
            abstract: Paper abstract.
            publication_date: Publication date.
            publication_year: Publication year.
            venue: Conference / journal venue.
            publisher: Publishing organization.
            authors: Authors list.
            language: Language code.
            paper_type: Document classification.
            canonical_url: Canonical URL.
            metadata_: Additional metadata dictionary.

        Returns:
            PaperResolutionResult containing the resolved Paper, optional matching PaperFile,
            paper_reused flag, and file_reused flag.

        Raises:
            PaperConflictError: If DOI and checksum identify two different existing papers.
            PaperRepositoryError: If validation or persistence in the repository fails.
        """
        cleaned_doi = normalize_doi(doi)
        cleaned_checksum = validate_and_normalize_sha256(checksum_sha256)

        paper_by_doi: Paper | None = None
        if cleaned_doi is not None:
            paper_by_doi = await self.repository.get_by_doi(cleaned_doi)

        checksum_match: tuple[Paper, PaperFile] | None = None
        if cleaned_checksum is not None:
            checksum_match = await self.repository.get_by_checksum(cleaned_checksum)

        paper_by_checksum = checksum_match[0] if checksum_match else None
        file_by_checksum = checksum_match[1] if checksum_match else None

        if (
            paper_by_doi is not None
            and paper_by_checksum is not None
            and paper_by_doi.id != paper_by_checksum.id
        ):
            raise PaperConflictError(
                f"Paper conflict: DOI '{cleaned_doi}' matches paper {paper_by_doi.id} ('{paper_by_doi.title}'), "
                f"but checksum '{cleaned_checksum}' matches paper {paper_by_checksum.id} ('{paper_by_checksum.title}')."
            )

        def get_reusable_file(pf: PaperFile | None) -> PaperFile | None:
            if pf is not None and pf.status in REUSABLE_FILE_STATUSES:
                return pf
            return None

        if paper_by_doi is not None:
            paper = paper_by_doi
            matching_file = get_reusable_file(file_by_checksum) if (file_by_checksum and file_by_checksum.paper_id == paper.id) else None
            paper_reused = True
            file_reused = matching_file is not None

            await self._enrich_metadata_if_missing(
                paper=paper,
                title=title,
                doi=cleaned_doi,
                abstract=abstract,
                publication_date=publication_date,
                publication_year=publication_year,
                venue=venue,
                publisher=publisher,
                authors=authors,
                language=language,
                paper_type=paper_type,
                canonical_url=canonical_url,
                metadata_=metadata_,
            )

            return PaperResolutionResult(
                paper=paper,
                matching_file=matching_file,
                paper_reused=paper_reused,
                file_reused=file_reused,
            )

        elif paper_by_checksum is not None:
            paper = paper_by_checksum
            matching_file = get_reusable_file(file_by_checksum)
            paper_reused = True
            file_reused = matching_file is not None

            await self._enrich_metadata_if_missing(
                paper=paper,
                title=title,
                doi=cleaned_doi,
                abstract=abstract,
                publication_date=publication_date,
                publication_year=publication_year,
                venue=venue,
                publisher=publisher,
                authors=authors,
                language=language,
                paper_type=paper_type,
                canonical_url=canonical_url,
                metadata_=metadata_,
            )

            return PaperResolutionResult(
                paper=paper,
                matching_file=matching_file,
                paper_reused=paper_reused,
                file_reused=file_reused,
            )

        else:
            if not isinstance(title, str) or not title.strip():
                raise PaperValidationError(
                    f"Cannot create a new Paper without a non-empty string title, got {title!r} ({type(title).__name__})."
                )

            paper_title = title.strip()
            new_paper = await self.repository.create_paper(
                title=paper_title,
                abstract=abstract,
                doi=cleaned_doi,
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

            return PaperResolutionResult(
                paper=new_paper,
                matching_file=None,
                paper_reused=False,
                file_reused=False,
            )

    async def register_paper_file(
        self,
        paper_id: uuid.UUID,
        upload_result: CloudinaryUploadResult,
        original_filename: str,
        *,
        checksum_sha256: str | None = None,
        is_primary: bool = False,
        status: str = "available",
        mime_type: str = "application/pdf",
    ) -> PaperFile:
        """Register a newly uploaded file record in PostgreSQL using Cloudinary metadata.

        Args:
            paper_id: Target Paper UUID.
            upload_result: CloudinaryUploadResult containing public_id, secure_url, and byte_size.
            original_filename: Name of the original uploaded file.
            checksum_sha256: SHA-256 hexadecimal hash of the file content.
            is_primary: Whether this file is the primary document representation.
            status: Status to record for this file (default 'available').
            mime_type: MIME type of the uploaded asset (default 'application/pdf').

        Returns:
            The flushed PaperFile database record.

        Raises:
            TypeError: If paper_id is not a UUID or upload_result is not a CloudinaryUploadResult.
            PaperRepositoryError: If file creation fails validation.
        """
        if not isinstance(upload_result, CloudinaryUploadResult):
            raise TypeError(
                f"upload_result must be a CloudinaryUploadResult, got {type(upload_result).__name__}"
            )

        return await self.repository.create_paper_file(
            paper_id=paper_id,
            storage_provider="cloudinary",
            storage_key=upload_result.public_id,
            original_filename=original_filename,
            mime_type=mime_type,
            file_url=upload_result.secure_url,
            file_size_bytes=upload_result.byte_size,
            checksum_sha256=checksum_sha256,
            is_primary=is_primary,
            status=status,
        )

    async def _enrich_metadata_if_missing(
        self,
        paper: Paper,
        *,
        title: str | None,
        doi: str | None,
        abstract: str | None,
        publication_date: date | None,
        publication_year: int | None,
        venue: str | None,
        publisher: str | None,
        authors: list[dict[str, Any]] | None,
        language: str | None,
        paper_type: str | None,
        canonical_url: str | None,
        metadata_: dict[str, Any] | None,
    ) -> None:
        """Enrich only missing or empty attributes on an existing paper without overwriting existing data."""
        updates: dict[str, Any] = {}

        # Title enrichment: replace recognized placeholder or empty title with valid incoming title
        if isinstance(title, str) and title.strip():
            candidate_title = title.strip()
            is_placeholder = paper.title.strip().lower() in {"untitled paper", "untitled", "placeholder"}
            if not paper.title or not paper.title.strip() or is_placeholder:
                updates["title"] = candidate_title

        if not paper.doi and doi:
            updates["doi"] = doi

        if (not paper.abstract or not paper.abstract.strip()) and (abstract and abstract.strip()):
            updates["abstract"] = abstract

        if paper.publication_date is None and publication_date is not None:
            updates["publication_date"] = publication_date

        if paper.publication_year is None and publication_year is not None:
            updates["publication_year"] = publication_year

        if (not paper.venue or not paper.venue.strip()) and (venue and venue.strip()):
            updates["venue"] = venue

        if (not paper.publisher or not paper.publisher.strip()) and (publisher and publisher.strip()):
            updates["publisher"] = publisher

        if (not paper.authors) and (authors and len(authors) > 0):
            updates["authors"] = authors

        if (not paper.language or not paper.language.strip()) and (language and language.strip()):
            updates["language"] = language

        if (not paper.paper_type or not paper.paper_type.strip()) and (paper_type and paper_type.strip()):
            updates["paper_type"] = paper_type

        if (not paper.canonical_url or not paper.canonical_url.strip()) and (canonical_url and canonical_url.strip()):
            updates["canonical_url"] = canonical_url

        # For metadata dict: merge keys if new keys are provided
        if metadata_:
            current_meta = dict(paper.metadata_) if paper.metadata_ else {}
            has_new_meta = False
            for k, v in metadata_.items():
                if k not in current_meta and v is not None:
                    current_meta[k] = v
                    has_new_meta = True
            if has_new_meta:
                updates["metadata_"] = current_meta

        if updates:
            await self.repository.update_paper(paper, **updates)

    async def list_papers(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> PaperListResponse:
        """Retrieve paginated research papers with file details and ingestion statistics."""
        rows, total = await self.repository.list_papers(limit=limit, offset=offset, search=search)
        items = [build_paper_list_item(paper, chunk_cnt, max_p) for paper, chunk_cnt, max_p in rows]
        return PaperListResponse(items=items, total=total, limit=limit, offset=offset)

    async def get_paper_details(
        self,
        paper_id: uuid.UUID,
    ) -> PaperDetailResponse | None:
        """Retrieve complete details for a single research paper."""
        result = await self.repository.get_paper_details(paper_id)
        if result is None:
            return None
        paper, chunk_cnt, max_p = result
        return build_paper_detail(paper, chunk_cnt, max_p)

    async def delete_paper(self, paper_id: uuid.UUID) -> None:
        """Permanently delete a research paper, associated data, index chunks, and Cloudinary storage file.

        Args:
            paper_id: UUID of the research paper to delete.

        Raises:
            PaperNotFoundError: If the paper does not exist.
            ActiveJobConflictError: If the paper is currently undergoing active ingestion.
        """
        deleted, storage_keys = await self.repository.delete_paper(paper_id)
        if not deleted:
            raise PaperNotFoundError(f"Research paper with id '{paper_id}' not found.")

        await self.repository.session.commit()
        logger.info("Successfully deleted paper %s and associated database records.", paper_id)

        if self.retrieval_service is not None:
            try:
                retrieval_repo = RetrievalRepository(self.repository.session)
                await self.retrieval_service.refresh_index(retrieval_repo)
                logger.info("Refreshed lexical retrieval index after deleting paper %s.", paper_id)
            except Exception as exc:
                logger.error(
                    "Failed to refresh lexical retrieval index after paper %s deletion: %s",
                    paper_id,
                    exc,
                    exc_info=True,
                )

        storage_failures = []
        for key in storage_keys:
            try:
                self.storage.delete_file(key)
                logger.info("Deleted Cloudinary storage asset '%s' for paper %s.", key, paper_id)
            except Exception as exc:
                logger.error(
                    "Failed to delete Cloudinary storage asset '%s' for paper %s: %s",
                    key,
                    paper_id,
                    exc,
                    exc_info=True,
                )
                storage_failures.append((key, str(exc)))

        if storage_failures:
            logger.warning(
                "Paper %s database records deleted, but %d Cloudinary file(s) failed cleanup: %s",
                paper_id,
                len(storage_failures),
                storage_failures,
            )


def build_paper_list_item(
    paper: Paper, chunk_count: int, max_page: int | None
) -> PaperListItemResponse:
    """Build a summarized PaperListItemResponse from a Paper entity and aggregated stats."""
    primary_file: PaperFile | None = None
    if paper.files:
        primary_file = next((f for f in paper.files if f.is_primary), None)
        if primary_file is None:
            sorted_files = sorted(
                paper.files,
                key=lambda f: f.created_at or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            primary_file = sorted_files[0]

    original_filename = primary_file.original_filename if primary_file else None
    file_size_bytes = primary_file.file_size_bytes if primary_file else None
    file_status = primary_file.status if primary_file else None

    latest_job: IngestionJob | None = None
    if paper.ingestion_jobs:
        sorted_jobs = sorted(
            paper.ingestion_jobs,
            key=lambda j: j.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        latest_job = sorted_jobs[0]

    ingestion_status = latest_job.status if latest_job else ("completed" if chunk_count > 0 else None)
    current_stage = latest_job.current_stage if latest_job else ("indexed" if chunk_count > 0 else None)

    page_count: int | None = None
    if paper.metadata_ and isinstance(paper.metadata_, dict):
        page_count = paper.metadata_.get("total_pages") or paper.metadata_.get("page_count")

    if page_count is None and latest_job and latest_job.metadata_ and isinstance(latest_job.metadata_, dict):
        diag = latest_job.metadata_.get("diagnostics")
        if isinstance(diag, dict):
            page_count = diag.get("total_pages")
        if page_count is None:
            page_count = latest_job.metadata_.get("total_pages")

    if page_count is None and max_page is not None and max_page > 0:
        page_count = max_page

    return PaperListItemResponse(
        id=paper.id,
        title=paper.title,
        abstract=paper.abstract,
        doi=paper.doi,
        publication_date=paper.publication_date,
        publication_year=paper.publication_year,
        venue=paper.venue,
        publisher=paper.publisher,
        authors=paper.authors if isinstance(paper.authors, list) else [],
        created_at=paper.created_at,
        updated_at=paper.updated_at,
        original_filename=original_filename,
        file_size_bytes=file_size_bytes,
        file_status=file_status,
        ingestion_status=ingestion_status,
        current_stage=current_stage,
        chunk_count=chunk_count,
        page_count=page_count,
    )


def build_paper_detail(
    paper: Paper, chunk_count: int, max_page: int | None
) -> PaperDetailResponse:
    """Build a comprehensive PaperDetailResponse including all associated files and metadata."""
    base = build_paper_list_item(paper, chunk_count, max_page)
    file_items = [
        PaperFileItem(
            id=f.id,
            original_filename=f.original_filename,
            file_size_bytes=f.file_size_bytes,
            mime_type=f.mime_type,
            file_url=f.file_url,
            status=f.status,
            created_at=f.created_at,
        )
        for f in (paper.files or [])
    ]
    return PaperDetailResponse(
        **base.model_dump(),
        files=file_items,
        metadata_=paper.metadata_ if isinstance(paper.metadata_, dict) else {},
    )
