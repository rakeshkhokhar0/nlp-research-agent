"""FastAPI router for document ingestion endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ActiveJobConflictError,
    AppError,
    InvalidPDFError,
    PDFEncryptedError,
    PDFNoExtractableTextError,
    PDFParsingError,
    StorageError,
    StorageUploadError,
)
from app.core.logging_config import correlation_context
from app.database.session import get_db
from app.modules.ingestion.repository import IngestionRepository
from app.modules.ingestion.schemas import IngestionJobResponse, IngestionUploadResponse
from app.modules.ingestion.service import IngestionResult, IngestionService
from app.modules.papers.service import PaperValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def get_ingestion_service(request: Request) -> IngestionService:
    """Dependency provider for IngestionService.

    Retrieves the application-scoped IngestionService initialized during application lifespan.

    Raises:
        RuntimeError: If IngestionService has not been initialized on app.state via lifespan.
    """
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        raise RuntimeError(
            "IngestionService is not initialized on app.state. "
            "Application lifespan initialization is required before serving ingestion requests."
        )
    return service


@router.post(
    "/upload",
    response_model=IngestionUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and ingest a research paper PDF",
    description="Accepts a PDF file upload, optional title, and optional DOI, parses, chunks, embeds, and stores the paper.",
)
async def upload_pdf(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    doi: str | None = Form(None),
    request: Request = None,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestionUploadResponse:
    """Upload and ingest a research paper PDF."""
    req_id = request.headers.get("X-Request-ID") if request else str(uuid.uuid4())
    with correlation_context(request_id=req_id):
        if not file or not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A PDF file must be provided.",
            )

        filename = file.filename.strip()
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid filename must be provided.",
            )

        try:
            content = await file.read()
        except Exception as read_err:
            logger.error("Failed to read uploaded file: %s", read_err)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to read uploaded file content.",
            ) from read_err

        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        # Validate PDF content signature
        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file format. Content must start with '%PDF-'.",
            )

        clean_title = title.strip() if title and title.strip() else None
        clean_doi = doi.strip() if doi and doi.strip() else None

        try:
            result: IngestionResult = await ingestion_service.ingest_pdf(
                pdf_bytes=content,
                original_filename=filename,
                title=clean_title,
                doi=clean_doi,
            )
        except InvalidPDFError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid PDF: {exc.message}",
            ) from exc
        except (PDFEncryptedError, PDFNoExtractableTextError, PDFParsingError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PDF parsing error: {exc.message}",
            ) from exc
        except PaperValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Paper validation error: {str(exc)}",
            ) from exc
        except ActiveJobConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Active job conflict: {exc.message}",
            ) from exc
        except (StorageUploadError, StorageError) as exc:
            logger.error("Storage error during PDF upload: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Storage service error during PDF upload.",
            ) from exc
        except AppError as exc:
            logger.error("Domain error during PDF ingestion: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document ingestion processing error.",
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error during PDF ingestion: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during document ingestion.",
            ) from exc

        return IngestionUploadResponse(
            paper_id=result.paper_id,
            job_id=result.ingestion_job_id,
            paper_file_id=result.paper_file_id,
            title=getattr(result, "title", None) or clean_title,
            chunks_created=result.chunks_created,
            embeddings_created=result.embeddings_created,
            paper_reused=result.paper_reused,
            file_reused=result.file_reused,
            job_reused=result.job_reused,
            status=result.status,
        )


@router.get(
    "/jobs/{job_id}",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Get ingestion job status",
    description="Retrieve the execution status and details of an ingestion job.",
)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> IngestionJobResponse:
    """Retrieve the status of an ingestion job by ID."""
    repo = IngestionRepository(db)
    job = await repo.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job '{job_id}' not found.",
        )

    return IngestionJobResponse(
        job_id=job.id,
        paper_id=job.paper_id,
        status=job.status,
        current_stage=job.current_stage,
        attempt_count=job.attempt_count,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
