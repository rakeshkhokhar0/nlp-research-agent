"""Database repository for retrieving research paper chunks and associated paper metadata."""

from __future__ import annotations

from typing import Sequence
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import app.database.model_registry  # noqa: F401 - ensures cross-module relationship mappers are registered
from app.modules.ingestion.models import PaperChunk
from app.modules.papers.models import Paper


class RetrievalRepository:
    """Repository for querying active research paper chunks from PostgreSQL."""

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

    async def get_current_chunks(
        self,
        paper_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[PaperChunk]:
        """Fetch all currently active paper chunks with eagerly-loaded parent papers.

        Args:
            paper_ids: Optional sequence of paper UUIDs to restrict the search.
                If provided and non-empty, only active chunks belonging to these
                papers are retrieved. If provided as an empty sequence, returns []
                immediately without executing a database query.

        Returns:
            List of active PaperChunk instances ordered deterministically by
            paper_id ascending, then chunk_index ascending.

        Raises:
            TypeError: If paper_ids is not a sequence or contains non-UUID elements.
        """
        if paper_ids is not None:
            if not isinstance(paper_ids, (list, tuple, set, frozenset)):
                raise TypeError(
                    f"paper_ids must be a collection of UUIDs, got {type(paper_ids).__name__}"
                )
            for pid in paper_ids:
                if not isinstance(pid, uuid.UUID):
                    raise TypeError(
                        f"All elements in paper_ids must be uuid.UUID instances, got {type(pid).__name__}"
                    )
            if len(paper_ids) == 0:
                return []

        stmt = (
            select(PaperChunk)
            .options(selectinload(PaperChunk.paper))
            .where(PaperChunk.is_current.is_(True))
            .order_by(PaperChunk.paper_id.asc(), PaperChunk.chunk_index.asc())
        )

        if paper_ids is not None:
            stmt = stmt.where(PaperChunk.paper_id.in_(list(paper_ids)))

        result = await self.session.execute(stmt)
        return list(result.scalars().all())
