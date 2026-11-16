"""Unit tests for conversation FastAPI router endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import uuid

from fastapi import HTTPException, status
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.modules.conversation.router import get_conversation_service
from app.modules.conversation.schemas import (
    ChatResponse,
    CitationResponse,
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


@pytest.fixture
def mock_conversation_service() -> MagicMock:
    service = MagicMock(spec=ConversationService)
    service.create_conversation = AsyncMock()
    service.get_conversation = AsyncMock()
    service.get_conversation_history = AsyncMock()
    service.ask_question = AsyncMock()
    return service


@pytest.fixture
def client(mock_conversation_service: MagicMock) -> TestClient:
    app.dependency_overrides[get_conversation_service] = lambda: mock_conversation_service
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


# -----------------------------------------------------------------------------
# POST /api/v1/conversations Tests
# -----------------------------------------------------------------------------


def test_create_conversation_endpoint(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify POST /api/v1/conversations creates a conversation and returns 201."""
    conv_id = uuid.uuid4()
    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)

    mock_conversation_service.create_conversation.return_value = ConversationResponse(
        id=conv_id,
        title="Transformer Analysis",
        status="active",
        created_at=now,
        updated_at=None,
        last_message_at=None,
    )

    response = client.post("/api/v1/conversations", json={"title": "Transformer Analysis"})

    assert response.status_code == status.HTTP_201_CREATED
    body = response.json()
    assert body["id"] == str(conv_id)
    assert body["title"] == "Transformer Analysis"
    assert body["status"] == "active"
    mock_conversation_service.create_conversation.assert_awaited_once_with(
        title="Transformer Analysis",
        metadata_=None,
    )


def test_get_conversation_endpoint(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify GET /api/v1/conversations/{id} returns conversation metadata."""
    conv_id = uuid.uuid4()
    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)

    mock_conversation_service.get_conversation.return_value = ConversationResponse(
        id=conv_id,
        title="My Chat",
        status="active",
        created_at=now,
        updated_at=None,
        last_message_at=None,
    )

    response = client.get(f"/api/v1/conversations/{conv_id}")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["id"] == str(conv_id)


def test_get_conversation_not_found(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify GET /api/v1/conversations/{id} returns 404 for missing conversation."""
    conv_id = uuid.uuid4()
    mock_conversation_service.get_conversation.side_effect = ConversationNotFoundError("Not found")

    response = client.get(f"/api/v1/conversations/{conv_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


# -----------------------------------------------------------------------------
# POST /api/v1/conversations/{id}/messages Tests
# -----------------------------------------------------------------------------


def test_ask_question_endpoint_success(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify POST message generates answer with citations and returns 200."""
    conv_id = uuid.uuid4()
    user_msg_id = uuid.uuid4()
    asst_msg_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    paper_id = uuid.uuid4()
    now = datetime(2026, 11, 14, 12, 1, 0, tzinfo=timezone.utc)

    mock_conversation_service.ask_question.return_value = ChatResponse(
        conversation_id=conv_id,
        user_message_id=user_msg_id,
        assistant_message_id=asst_msg_id,
        answer="Attention is all you need.",
        citations=[
            CitationResponse(
                chunk_id=chunk_id,
                paper_id=paper_id,
                paper_title="Attention Is All You Need",
                citation_order=1,
                page_number=1,
                relevance_score=4.9,
                quote_text="We propose the Transformer.",
            )
        ],
        model_name="gpt-4o-mini",
        created_at=now,
    )

    response = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"question": "What is the Transformer?", "top_k": 3},
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["conversation_id"] == str(conv_id)
    assert body["user_message_id"] == str(user_msg_id)
    assert body["assistant_message_id"] == str(asst_msg_id)
    assert body["answer"] == "Attention is all you need."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == str(chunk_id)
    assert body["citations"][0]["page_number"] == 1
    assert body["mode"] == "normal"


def test_ask_question_endpoint_forwards_mode_advanced(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify posting with mode='advanced' forwards mode to service and returns it in response."""
    conv_id = uuid.uuid4()
    now = datetime(2026, 11, 14, 12, 1, 0, tzinfo=timezone.utc)

    mock_conversation_service.ask_question.return_value = ChatResponse(
        conversation_id=conv_id,
        user_message_id=uuid.uuid4(),
        assistant_message_id=uuid.uuid4(),
        answer="Advanced detailed synthesis.",
        citations=[],
        model_name="gpt-4o-mini",
        mode="advanced",
        created_at=now,
    )

    response = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"question": "Compare architectures.", "top_k": 5, "mode": "advanced"},
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["mode"] == "advanced"
    assert body["answer"] == "Advanced detailed synthesis."
    mock_conversation_service.ask_question.assert_awaited_once_with(
        conversation_id=conv_id,
        question="Compare architectures.",
        top_k=5,
        paper_ids=None,
        mode="advanced",
    )


def test_ask_question_endpoint_general_chat(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify posting with chat_type='general' forwards chat_type to service and returns it in response."""
    conv_id = uuid.uuid4()
    now = datetime(2026, 11, 14, 12, 1, 0, tzinfo=timezone.utc)

    mock_conversation_service.ask_question.return_value = ChatResponse(
        conversation_id=conv_id,
        user_message_id=uuid.uuid4(),
        assistant_message_id=uuid.uuid4(),
        answer="General conversational answer.",
        citations=[],
        model_name="gpt-4o-mini",
        mode="normal",
        chat_type="general",
        created_at=now,
    )

    response = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"question": "What is Python?", "chat_type": "general"},
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["chat_type"] == "general"
    assert body["answer"] == "General conversational answer."
    assert body["citations"] == []
    mock_conversation_service.ask_question.assert_awaited_once_with(
        conversation_id=conv_id,
        question="What is Python?",
        top_k=5,
        paper_ids=None,
        mode="normal",
        chat_type="general",
    )


def test_ask_question_inactive_conversation_returns_400(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify posting to an inactive conversation returns 400 Bad Request."""
    conv_id = uuid.uuid4()
    mock_conversation_service.ask_question.side_effect = ConversationInactiveError("Conversation archived")

    response = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"question": "Explain BERT."},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "archived" in response.json()["detail"].lower()


def test_ask_question_not_found_returns_404(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify posting to nonexistent conversation returns 404."""
    conv_id = uuid.uuid4()
    mock_conversation_service.ask_question.side_effect = ConversationNotFoundError("Missing")

    response = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"question": "Explain BERT."},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_ask_question_llm_failure_returns_502(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify LLM failure returns 502 Bad Gateway without leaking sensitive secrets."""
    conv_id = uuid.uuid4()
    mock_conversation_service.ask_question.side_effect = LLMGenerationError(
        "Failed to generate answer from language model."
    )

    response = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"question": "Explain BERT."},
    )

    assert response.status_code == status.HTTP_502_BAD_GATEWAY
    assert "Failed to generate answer" in response.json()["detail"]


def test_ask_question_retrieval_unavailable_returns_503(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify retrieval service unavailable returns 503 Service Unavailable."""
    conv_id = uuid.uuid4()
    mock_conversation_service.ask_question.side_effect = RetrievalUnavailableError(
        "Retrieval service is unavailable or not initialized."
    )

    response = client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"question": "Explain BERT."},
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Retrieval service is unavailable" in response.json()["detail"]


# -----------------------------------------------------------------------------
# GET /api/v1/conversations/{id}/messages Tests
# -----------------------------------------------------------------------------


def test_get_conversation_history_endpoint(
    client: TestClient, mock_conversation_service: MagicMock
) -> None:
    """Verify GET /api/v1/conversations/{id}/messages returns chronological history."""
    conv_id = uuid.uuid4()
    now = datetime(2026, 11, 14, 12, 0, 0, tzinfo=timezone.utc)

    msg1 = ConversationMessageResponse(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=1,
        role="user",
        content="What is attention?",
        status="completed",
        model_name=None,
        token_count=None,
        citations=[],
        created_at=now,
    )
    msg2 = ConversationMessageResponse(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        sequence_number=2,
        role="assistant",
        content="Attention mechanism connects representations.",
        status="completed",
        model_name="gpt-4o-mini",
        token_count=45,
        citations=[],
        created_at=now,
    )

    mock_conversation_service.get_conversation_history.return_value = [msg1, msg2]

    response = client.get(f"/api/v1/conversations/{conv_id}/messages")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert len(body) == 2
    assert body[0]["sequence_number"] == 1
    assert body[0]["role"] == "user"
    assert body[1]["sequence_number"] == 2
    assert body[1]["role"] == "assistant"


def test_get_conversation_service_uninitialized_raises_http_503() -> None:
    """Verify get_conversation_service raises HTTPException 503 if app.state.conversation_service is absent."""
    mock_request = MagicMock()
    del mock_request.app.state.conversation_service

    with pytest.raises(HTTPException) as exc_info:
        get_conversation_service(mock_request)

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Conversation service is unavailable. Configure OPENAI_API_KEY to enable research chat." in exc_info.value.detail


def test_endpoints_return_503_not_500_when_conversation_service_uninitialized() -> None:
    """Verify all conversation endpoints return HTTP 503 (not 500) when conversation service is None."""
    app.dependency_overrides.pop(get_conversation_service, None)
    app.state.conversation_service = None

    test_client = TestClient(app, raise_server_exceptions=False)
    dummy_id = uuid.uuid4()
    expected_detail = "Conversation service is unavailable. Configure OPENAI_API_KEY to enable research chat."

    # 1. POST /api/v1/conversations
    r1 = test_client.post("/api/v1/conversations", json={"title": "Test"})
    assert r1.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert r1.json()["detail"] == expected_detail

    # 2. GET /api/v1/conversations/{id}
    r2 = test_client.get(f"/api/v1/conversations/{dummy_id}")
    assert r2.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert r2.json()["detail"] == expected_detail

    # 3. POST /api/v1/conversations/{id}/messages
    r3 = test_client.post(f"/api/v1/conversations/{dummy_id}/messages", json={"question": "Hello?"})
    assert r3.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert r3.json()["detail"] == expected_detail

    # 4. GET /api/v1/conversations/{id}/messages
    r4 = test_client.get(f"/api/v1/conversations/{dummy_id}/messages")
    assert r4.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert r4.json()["detail"] == expected_detail
