"""FastAPI router for research paper catalog and library endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.modules.papers.repository import PaperRepository
from app.modules.papers.schemas import PaperDetailResponse, PaperListResponse
from app.modules.papers.service import PaperService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])


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
    db: AsyncSession = Depends(get_db),
) -> PaperListResponse:
    """Retrieve catalog of uploaded research papers."""
    repository = PaperRepository(db)
    service = PaperService(repository)
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
    db: AsyncSession = Depends(get_db),
) -> PaperDetailResponse:
    """Retrieve detailed research paper representation."""
    repository = PaperRepository(db)
    service = PaperService(repository)
    paper = await service.get_paper_details(paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research paper with id '{paper_id}' not found.",
        )
    return paper
