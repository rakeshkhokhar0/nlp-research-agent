"""FastAPI router for research conversation and chat endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from app.core.logging_config import correlation_context
from app.modules.conversation.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreateRequest,
    ConversationMessageResponse,
    ConversationResponse,
)
from app.modules.conversation.service import (
    ConversationInactiveError,
    ConversationNotFoundError,
    ConversationService,
    LLMGenerationError,
    RetrievalUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


def get_conversation_service(request: Request) -> ConversationService:
    """Dependency provider for ConversationService.

    Retrieves the application-scoped ConversationService initialized during lifespan.

    Raises:
        HTTPException: HTTP 503 if ConversationService is not initialized on app.state via lifespan.
    """
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service is unavailable. Configure OPENAI_API_KEY to enable research chat.",
        )
    return service


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new research conversation",
    description="Initializes a new independent research conversation session.",
)
async def create_conversation(
    payload: ConversationCreateRequest | None = None,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Create a new research conversation session."""
    title = payload.title if payload else None
    metadata_ = payload.metadata if payload else None
    kwargs: dict[str, Any] = {"title": title, "metadata_": metadata_}
    if payload and payload.chat_type is not None:
        kwargs["chat_type"] = payload.chat_type
    return await service.create_conversation(**kwargs)


@router.get(
    "",
    response_model=list[ConversationResponse],
    status_code=status.HTTP_200_OK,
    summary="List all conversations",
    description=(
        "Retrieve conversations ordered by most recent activity. "
        "Security Note: In this prototype, conversations are not yet user-isolated. "
        "This is acceptable only for our controlled single-user prototype and "
        "must be restricted by user ownership before public multi-user deployment."
    ),
)
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=100, description="Max number of conversations to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    status: str | None = Query(default="active", description="Filter by status ('active', 'archived')"),
    service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationResponse]:
    """Retrieve list of conversations sorted by most recent activity."""
    return await service.list_conversations(limit=limit, offset=offset, status=status)


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get conversation metadata",
    description="Retrieve metadata for a conversation by UUID.",
)
async def get_conversation(
    conversation_id: uuid.UUID,
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationResponse:
    """Retrieve metadata for a conversation."""
    try:
        return await service.get_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete conversation",
    description="Permanently delete a conversation and all associated messages and citations.",
)
async def delete_conversation(
    conversation_id: uuid.UUID,
    service: ConversationService = Depends(get_conversation_service),
) -> Response:
    """Permanently delete a conversation by UUID."""
    try:
        await service.delete_conversation(conversation_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Failed to delete conversation %s: %s", conversation_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting the conversation.",
        ) from exc

@router.post(
    "/{conversation_id}/messages",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a research question in a conversation",
    description="Retrieves evidence chunks from indexed research papers and generates a grounded answer.",
)
async def ask_question(
    conversation_id: uuid.UUID,
    payload: ChatRequest,
    request: Request,
    service: ConversationService = Depends(get_conversation_service),
) -> ChatResponse:
    """Ask a question within a conversation and receive a grounded answer with citations."""
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    with correlation_context(request_id=req_id, conversation_id=str(conversation_id)):
        try:
            kwargs = {
                "conversation_id": conversation_id,
                "question": payload.question,
                "top_k": payload.top_k,
                "paper_ids": payload.paper_ids,
                "mode": payload.mode,
            }
            if payload.chat_type is not None:
                kwargs["chat_type"] = payload.chat_type
            return await service.ask_question(**kwargs)
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except ConversationInactiveError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except RetrievalUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except LLMGenerationError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error handling conversation question: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred while processing the question.",
            ) from exc


@router.get(
    "/{conversation_id}/messages",
    response_model=list[ConversationMessageResponse],
    status_code=status.HTTP_200_OK,
    summary="Get conversation message history",
    description="Retrieve all messages and citations belonging to a conversation ordered chronologically.",
)
async def get_conversation_history(
    conversation_id: uuid.UUID,
    service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationMessageResponse]:
    """Retrieve ordered message history and citations for a conversation."""
    try:
        return await service.get_conversation_history(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc