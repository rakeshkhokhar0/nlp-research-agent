"""FastAPI router for research paper catalog and library endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ActiveJobConflictError
from app.database.session import get_db
from app.modules.papers.repository import PaperNotFoundError, PaperRepository
from app.modules.papers.schemas import PaperDetailResponse, PaperListResponse
from app.modules.papers.service import PaperService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])


def get_paper_service(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> PaperService:
    """Dependency provider for PaperService with injected database session and optional RetrievalService."""
    repository = PaperRepository(db)
    retrieval_service = getattr(request.app.state, "retrieval_service", None)
    return PaperService(repository=repository, retrieval_service=retrieval_service)


@router.get(
    "",
    response_model=PaperListResponse,
    status_code=status.HTTP_200_OK,
    summary="List uploaded research papers",
    description="Retrieve paginated list of all uploaded research papers with file details and ingestion status.",
)
async def list_papers(
    limit: int = Query(default=50, ge=1, le=100, description="Max number of papers to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    search: str | None = Query(default=None, description="Optional title or DOI search query"),
    service: PaperService = Depends(get_paper_service),
) -> PaperListResponse:
    """Retrieve catalog of uploaded research papers."""
    return await service.list_papers(limit=limit, offset=offset, search=search)


@router.get(
    "/{paper_id}",
    response_model=PaperDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get research paper details",
    description="Retrieve complete metadata, files, and ingestion information for a specific research paper.",
)
async def get_paper_details(
    paper_id: uuid.UUID,
    service: PaperService = Depends(get_paper_service),
) -> PaperDetailResponse:
    """Retrieve detailed research paper representation."""
    paper = await service.get_paper_details(paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research paper with id '{paper_id}' not found.",
        )
    return paper


@router.delete(
    "/{paper_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete research paper",
    description="Permanently delete a research paper, associated files, chunks, embeddings, and Cloudinary storage.",
)
async def delete_paper(
    paper_id: uuid.UUID,
    service: PaperService = Depends(get_paper_service),
) -> Response:
    """Permanently delete a research paper by UUID."""
    try:
        await service.delete_paper(paper_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except PaperNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ActiveJobConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Failed to delete paper %s: %s", paper_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting the research paper.",
        ) from exc
