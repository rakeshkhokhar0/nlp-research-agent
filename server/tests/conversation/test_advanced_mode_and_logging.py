"""Tests for Advanced Mode answer generation, shared logging infrastructure, and correlation tracking."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from app.core.config import LoggingSettings
from app.core.logging_config import (
    CorrelationFilter,
    StructuredLogFormatter,
    correlation_context,
    get_correlation_ids,
    log_event,
    sanitize_log_text,
    sanitize_log_value,
    setup_logging,
)
from app.modules.conversation.llm_client import (
    LLMAnswerResult,
    LLMClientError,
    OpenAIResponseClient,
)
from app.modules.conversation.models import Conversation, Message
from app.modules.conversation.service import (
    GREETING_RESPONSE,
    NO_EVIDENCE_RESPONSE,
    UNSUPPORTED_SCOPE_RESPONSE,
    ConversationService,
)
from app.modules.retrieval.service import RetrievedChunk


@pytest.fixture
def sample_retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        paper_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
        chunk_index=0,
        content="The Transformer architecture is based entirely on self-attention mechanisms.",
        score=4.95,
        paper_title="Attention Is All You Need",
        publication_year=2017,
    )


@pytest.fixture
def mock_session_factory() -> MagicMock:
    factory = MagicMock()
    session = AsyncMock()
    tx = AsyncMock()
    tx.__aenter__.return_value = session
    tx.__aexit__.return_value = None
    session.begin = MagicMock(return_value=tx)

    cm = AsyncMock()
    cm.__aenter__.return_value = session
    cm.__aexit__.return_value = None
    factory.return_value = cm
    return factory


# 1. Normal vs Advanced Mode Tests


@pytest.mark.asyncio
async def test_normal_mode_preserves_existing_behavior(
    mock_session_factory: MagicMock,
    sample_retrieved_chunk: RetrievedChunk,
) -> None:
    """Verify Normal mode calls generate_grounded_answer with mode='normal'."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval = MagicMock()
    mock_retrieval.is_initialized = True
    mock_retrieval.search.return_value = [sample_retrieved_chunk]

    mock_llm = MagicMock(spec=OpenAIResponseClient)
    mock_llm.generate_grounded_answer = AsyncMock(
        return_value=LLMAnswerResult(
            answer="Normal grounded answer.",
            cited_chunk_ids=[str(sample_retrieved_chunk.chunk_id)],
            model_name="gpt-4o-mini",
            token_count=150,
        )
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval,
        llm_client=mock_llm,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        repo.create_message = AsyncMock(
            side_effect=[
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Q"),
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=2, role="assistant", content="A"),
            ]
        )
        repo.get_chunk_page_information = AsyncMock(return_value={sample_retrieved_chunk.chunk_id: 3})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(
            conversation_id=conv_id,
            question="What is the Transformer architecture?",
            top_k=5,
            mode="normal",
        )

        assert response.mode == "normal"
        assert response.answer == "Normal grounded answer."
        mock_llm.generate_grounded_answer.assert_awaited_once()
        _, call_kwargs = mock_llm.generate_grounded_answer.call_args
        assert call_kwargs["mode"] == "normal"
        assert call_kwargs["evidence_chunks"] == [sample_retrieved_chunk]


@pytest.mark.asyncio
async def test_advanced_mode_makes_single_llm_call_with_identical_retrieval(
    mock_session_factory: MagicMock,
    sample_retrieved_chunk: RetrievedChunk,
) -> None:
    """Verify Advanced mode triggers exactly ONE LLM call using the same retrieved evidence as Normal Mode."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval = MagicMock()
    mock_retrieval.is_initialized = True
    mock_retrieval.search.return_value = [sample_retrieved_chunk]

    mock_llm = MagicMock(spec=OpenAIResponseClient)
    mock_llm.generate_grounded_answer = AsyncMock(
        return_value=LLMAnswerResult(
            answer="Advanced detailed answer with conceptual depth.",
            cited_chunk_ids=[str(sample_retrieved_chunk.chunk_id)],
            model_name="gpt-4o-mini",
            token_count=350,
        )
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval,
        llm_client=mock_llm,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        repo.create_message = AsyncMock(
            side_effect=[
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Q"),
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=2, role="assistant", content="A"),
            ]
        )
        repo.get_chunk_page_information = AsyncMock(return_value={sample_retrieved_chunk.chunk_id: 3})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(
            conversation_id=conv_id,
            question="Analyze the Transformer architecture in depth.",
            top_k=5,
            mode="advanced",
        )

        assert response.mode == "advanced"
        assert response.answer == "Advanced detailed answer with conceptual depth."
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == sample_retrieved_chunk.chunk_id

        # Exactly ONE LLM call made
        mock_llm.generate_grounded_answer.assert_awaited_once()
        _, call_kwargs = mock_llm.generate_grounded_answer.call_args
        # Uses exactly the same retrieved chunks from search
        assert call_kwargs["evidence_chunks"] == [sample_retrieved_chunk]
        assert call_kwargs["mode"] == "advanced"


@pytest.mark.asyncio
async def test_both_modes_enforce_greeting_and_scope_restrictions(
    mock_session_factory: MagicMock,
) -> None:
    """Verify that in both normal and advanced modes, greetings and out-of-scope questions are handled without retrieval or LLM."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval = MagicMock()
    mock_llm = MagicMock()

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval,
        llm_client=mock_llm,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2, 3, 4])
        repo.create_message = AsyncMock(
            side_effect=[
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="hi"),
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=2, role="assistant", content=GREETING_RESPONSE),
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=3, role="user", content="capital of france"),
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=4, role="assistant", content=UNSUPPORTED_SCOPE_RESPONSE),
            ]
        )
        repo.update_conversation_last_message_at = AsyncMock()

        # 1. Greeting in advanced mode
        resp_greeting = await service.ask_question(conv_id, "Hello!", mode="advanced")
        assert resp_greeting.answer == GREETING_RESPONSE
        assert resp_greeting.mode == "advanced"
        mock_retrieval.search.assert_not_called()
        mock_llm.generate_grounded_answer.assert_not_called()

        # 2. Unsupported in advanced mode
        resp_unsupported = await service.ask_question(conv_id, "What is the capital of France?", mode="advanced")
        assert resp_unsupported.answer == UNSUPPORTED_SCOPE_RESPONSE
        assert resp_unsupported.mode == "advanced"
        mock_retrieval.search.assert_not_called()
        mock_llm.generate_grounded_answer.assert_not_called()


@pytest.mark.asyncio
async def test_citation_validation_rejects_hallucinated_ids(
    mock_session_factory: MagicMock,
    sample_retrieved_chunk: RetrievedChunk,
) -> None:
    """Verify that any cited chunk ID outside the retrieved evidence is filtered out."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval = MagicMock()
    mock_retrieval.is_initialized = True
    mock_retrieval.search.return_value = [sample_retrieved_chunk]

    fake_id = str(uuid.uuid4())
    mock_llm = MagicMock(spec=OpenAIResponseClient)
    mock_llm.generate_grounded_answer = AsyncMock(
        return_value=LLMAnswerResult(
            answer="Answer citing valid and hallucinated IDs.",
            cited_chunk_ids=[str(sample_retrieved_chunk.chunk_id), fake_id, "not-a-uuid"],
            model_name="gpt-4o-mini",
        )
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval,
        llm_client=mock_llm,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        repo.create_message = AsyncMock(
            side_effect=[
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Q"),
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=2, role="assistant", content="A"),
            ]
        )
        repo.get_chunk_page_information = AsyncMock(return_value={sample_retrieved_chunk.chunk_id: 1})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "Q", mode="normal")

        # Exactly 1 validated citation, hallucinated and malformed IDs dropped
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == sample_retrieved_chunk.chunk_id


# 2. Shared Logging, Sanitization, and Correlation Tests


def test_sanitize_log_text_redacts_credentials_and_tokens() -> None:
    """Verify secrets, Bearer tokens, and OpenAI keys are sanitized."""
    raw_log = (
        "Calling model with api_key='sk-proj-abc1234567890abcdef1234' "
        "and Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret.sig "
        "and password=mysecretpassword123"
    )
    sanitized = sanitize_log_text(raw_log)

    assert "sk-proj" not in sanitized
    assert "REDACTED" in sanitized
    assert "mysecretpassword123" not in sanitized
    assert "eyJhbGci" not in sanitized


def test_sanitize_log_value_handles_dicts_and_lists() -> None:
    """Verify recursive data structure sanitization."""
    payload = {
        "user": "alice",
        "api_key": "supersecretkey12345",
        "nested": {"client_secret": "topsecret_value", "normal_field": 42},
        "token_list": ["sk-test1234567890abcdef"],
    }
    sanitized = sanitize_log_value(payload)

    assert sanitized["user"] == "alice"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["client_secret"] == "[REDACTED]"
    assert sanitized["nested"]["normal_field"] == 42
    assert "sk-test" not in str(sanitized["token_list"])


@pytest.mark.asyncio
async def test_concurrent_tasks_preserve_isolated_correlation_ids() -> None:
    """Verify contextvars cleanly isolate request_id and conversation_id across concurrent coroutines."""

    async def worker(task_name: str, req_id: str, conv_id: str, job_id: str) -> dict[str, str | None]:
        with correlation_context(request_id=req_id, conversation_id=conv_id, ingestion_job_id=job_id):
            await asyncio.sleep(0.01)  # Context switch
            ids = get_correlation_ids()
            assert ids["request_id"] == req_id
            assert ids["conversation_id"] == conv_id
            assert ids["ingestion_job_id"] == job_id
            return ids

    results = await asyncio.gather(
        worker("A", "req-1", "conv-1", "job-1"),
        worker("B", "req-2", "conv-2", "job-2"),
        worker("C", "req-3", "conv-3", "job-3"),
    )

    assert results[0]["request_id"] == "req-1"
    assert results[1]["request_id"] == "req-2"
    assert results[2]["request_id"] == "req-3"


def test_log_event_failure_is_silent() -> None:
    """Verify log_event never raises exceptions even with faulty logger or settings."""
    broken_logger = MagicMock()
    broken_logger.isEnabledFor.return_value = True
    broken_logger.log.side_effect = RuntimeError("Disk full")

    # Should execute without throwing
    log_event(broken_logger, logging.INFO, "test_event", metric=123)


def test_structured_formatter_timestamp_uses_record_created() -> None:
    """Verify StructuredLogFormatter uses record.created rather than formatting time."""
    formatter = StructuredLogFormatter()
    fixed_time = 1700000000.123456  # 2023-11-14 22:13:20.123456 UTC
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.created = fixed_time
    output = formatter.format(record)

    expected_dt = datetime.fromtimestamp(fixed_time, tz=timezone.utc)
    expected_timestamp = expected_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"
    assert f"[{expected_timestamp}]" in output


def test_structured_formatter_sanitizes_exception_traceback() -> None:
    """Verify that exception tracebacks redact sk- keys, api_keys, and passwords."""
    formatter = StructuredLogFormatter()
    try:
        raise ValueError("Failed connecting to api_key: 'sk-proj-secret1234567890abcdef' with password='supersecret'")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="test_err_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=25,
            msg="Exception caught",
            args=(),
            exc_info=sys.exc_info(),
        )
    output = formatter.format(record)

    assert "sk-proj-secret" not in output
    assert "supersecret" not in output
    assert "[REDACTED" in output


def test_setup_logging_preserves_unrelated_handlers() -> None:
    """Verify setup_logging does not wipe unrelated existing root handlers."""
    root_logger = logging.getLogger()
    custom_handler = logging.NullHandler()
    root_logger.addHandler(custom_handler)

    try:
        setup_logging(LoggingSettings(log_level="DEBUG"))
        assert custom_handler in root_logger.handlers
        assert any(
            isinstance(getattr(h, "formatter", None), StructuredLogFormatter)
            for h in root_logger.handlers
        )
    finally:
        root_logger.removeHandler(custom_handler)


def test_nlp_trace_enabled_gates_trace_events_while_preserving_ordinary_logs() -> None:
    """Verify NLP_TRACE_ENABLED gates trace events while ordinary logs remain active."""
    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True

    settings_off = LoggingSettings(nlp_trace_enabled=False)
    settings_on = LoggingSettings(nlp_trace_enabled=True)

    # 1. Trace event with nlp_trace_enabled=False should be suppressed
    log_event(mock_logger, logging.INFO, "retrieval_query_processed", settings=settings_off)
    mock_logger.log.assert_not_called()

    # 2. Trace event with nlp_trace_enabled=True should log
    log_event(mock_logger, logging.INFO, "retrieval_query_processed", settings=settings_on)
    mock_logger.log.assert_called_once()
    mock_logger.reset_mock()

    # 3. Explicit is_trace=False should log even if nlp_trace_enabled=False
    log_event(mock_logger, logging.INFO, "app_status_event", is_trace=False, settings=settings_off)
    mock_logger.log.assert_called_once()
    mock_logger.reset_mock()

    # 4. Warnings and errors log regardless of trace flag
    log_event(mock_logger, logging.WARNING, "retrieval_query_warning", settings=settings_off)
    mock_logger.log.assert_called_once()


def test_nlp_trace_content_preview_controls_only_designated_preview_field() -> None:
    """Verify NLP_TRACE_CONTENT_PREVIEW only suppresses the designated preview field."""
    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True

    settings_no_preview = LoggingSettings(nlp_trace_enabled=True, nlp_trace_content_preview=False)
    settings_with_preview = LoggingSettings(nlp_trace_enabled=True, nlp_trace_content_preview=True)

    # Preview disabled: preview_field replaced by [PREVIEW_DISABLED], normal metrics preserved
    log_event(
        mock_logger,
        logging.INFO,
        "test_event",
        preview_field="chunk_content",
        chunk_content="This is detailed confidential paper text that is long.",
        paper_id="paper-123",
        settings=settings_no_preview,
    )
    msg = mock_logger.log.call_args[0][1]
    assert "chunk_content=[PREVIEW_DISABLED]" in msg
    assert "paper_id=paper-123" in msg

    mock_logger.reset_mock()

    # Preview enabled: preview_field contains truncated text
    log_event(
        mock_logger,
        logging.INFO,
        "test_event",
        preview_field="chunk_content",
        chunk_content="This is detailed confidential paper text that is long.",
        paper_id="paper-123",
        settings=settings_with_preview,
    )
    msg = mock_logger.log.call_args[0][1]
    assert "chunk_content=This is detailed confidential paper text that is long." in msg
    assert "paper_id=paper-123" in msg


def test_content_preview_redacts_credentials_when_preview_enabled() -> None:
    """Verify that OpenAI keys, bearer tokens, API keys, and passwords are redacted even with previews enabled."""
    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True

    settings_preview_on = LoggingSettings(nlp_trace_enabled=True, nlp_trace_content_preview=True)

    sensitive_content = (
        "Paper excerpt with secret key sk-proj-1234567890abcdef1234 and "
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.topsecret and "
        "api_key: 'prodkey123456' and password: 'adminpassword999'"
    )

    log_event(
        mock_logger,
        logging.INFO,
        "trace_chunk_event",
        is_trace=True,
        preview_field="chunk_preview",
        chunk_preview=sensitive_content,
        settings=settings_preview_on,
    )

    assert mock_logger.log.called
    logged_msg = mock_logger.log.call_args[0][1]

    # All secrets must be absent from the preview log
    assert "sk-proj-1234567890abcdef1234" not in logged_msg
    assert "eyJhbGci" not in logged_msg
    assert "prodkey123456" not in logged_msg
    assert "adminpassword999" not in logged_msg

    # Redaction placeholders must be present
    assert "[REDACTED" in logged_msg


def test_operational_events_emitted_when_nlp_trace_disabled() -> None:
    """Verify that essential operational lifecycle events remain available when NLP_TRACE_ENABLED=False."""
    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True
    settings_trace_off = LoggingSettings(nlp_trace_enabled=False)

    operational_events = [
        "ingestion_job_created",
        "ingestion_job_started",
        "ingestion_pdf_parsed",
        "ingestion_preprocessed",
        "ingestion_sections_detected",
        "ingestion_chunking_completed",
        "ingestion_embeddings_completed",
        "ingestion_retrieval_index_refreshed",
        "ingestion_job_completed",
        "retrieval_index_refresh_started",
        "retrieval_index_refresh_completed",
        "retrieval_ranking_completed",
        "conversation_created",
        "conversation_request_started",
        "greeting_returned",
        "no_evidence_returned",
        "conversation_message_completed",
    ]

    for ev in operational_events:
        log_event(mock_logger, logging.INFO, ev, settings=settings_trace_off, job_id="123")
        assert mock_logger.log.called, f"Expected operational event {ev} to be logged when trace is off"
        mock_logger.reset_mock()


def test_nlp_trace_events_suppressed_when_nlp_trace_disabled() -> None:
    """Verify that detailed NLP diagnostic trace events are suppressed when NLP_TRACE_ENABLED=False."""
    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True
    settings_trace_off = LoggingSettings(nlp_trace_enabled=False)

    nlp_trace_events = [
        "retrieval_query_diagnostics",
        "retrieval_candidates_evaluated",
        "retrieval_candidate_scoring_detail",
        "retrieval_ranking_diagnostics",
        "retrieval_query_processed",
        "retrieval_top_k_chunk_selected",
        "retrieval_lexical_scoring_completed",
        "ingestion_chunk_created",
        "ingestion_chunk_diagnostics",
    ]

    for ev in nlp_trace_events:
        log_event(mock_logger, logging.INFO, ev, settings=settings_trace_off, metric="detail")
        assert not mock_logger.log.called, f"Expected NLP trace event {ev} to be suppressed when trace is off"
        mock_logger.reset_mock()


@pytest.mark.asyncio
async def test_advanced_mode_citation_validation_and_grounding(
    mock_session_factory: MagicMock,
    sample_retrieved_chunk: RetrievedChunk,
) -> None:
    """Verify Advanced Mode validates citations against retrieved evidence and filters hallucinated IDs."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval = MagicMock()
    mock_retrieval.is_initialized = True
    mock_retrieval.search.return_value = [sample_retrieved_chunk]

    fake_id = str(uuid.uuid4())
    mock_llm = MagicMock(spec=OpenAIResponseClient)
    mock_llm.generate_grounded_answer = AsyncMock(
        return_value=LLMAnswerResult(
            answer="Advanced answer with valid and invalid citations.",
            cited_chunk_ids=[str(sample_retrieved_chunk.chunk_id), fake_id, "invalid-uuid"],
            model_name="gpt-4o-mini",
            token_count=210,
        )
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval,
        llm_client=mock_llm,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        repo.create_message = AsyncMock(
            side_effect=[
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Q"),
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=2, role="assistant", content="A"),
            ]
        )
        repo.get_chunk_page_information = AsyncMock(return_value={sample_retrieved_chunk.chunk_id: 2})
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "Q", mode="advanced")

        # Exactly 1 LLM call was made
        mock_llm.generate_grounded_answer.assert_awaited_once()
        assert mock_llm.generate_grounded_answer.call_args[1]["mode"] == "advanced"

        # Hallucinated and malformed citation IDs were filtered out
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == sample_retrieved_chunk.chunk_id
        assert response.citations[0].page_number == 2


@pytest.mark.asyncio
async def test_advanced_mode_handles_multiple_papers_with_single_call(
    mock_session_factory: MagicMock,
    sample_retrieved_chunk: RetrievedChunk,
) -> None:
    """Verify Advanced Mode passes all retrieved evidence to single LLM call and validates multi-paper citations."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    second_chunk = RetrievedChunk(
        chunk_id=uuid.UUID("33333333-3333-3333-3333-333333333333"),
        paper_id=uuid.UUID("44444444-4444-4444-4444-444444444444"),
        chunk_index=1,
        content="BERT uses bidirectional Transformers.",
        score=4.8,
        paper_title="BERT: Pre-training of Deep Bidirectional Transformers",
        publication_year=2018,
    )

    mock_retrieval = MagicMock()
    mock_retrieval.is_initialized = True
    mock_retrieval.search.return_value = [sample_retrieved_chunk, second_chunk]

    mock_llm = MagicMock(spec=OpenAIResponseClient)
    mock_llm.generate_grounded_answer = AsyncMock(
        return_value=LLMAnswerResult(
            answer="Advanced answer synthesizing Attention and BERT.",
            cited_chunk_ids=[str(sample_retrieved_chunk.chunk_id), str(second_chunk.chunk_id)],
            model_name="gpt-4o-mini",
            token_count=320,
        )
    )

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval,
        llm_client=mock_llm,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        repo.create_message = AsyncMock(
            side_effect=[
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Q"),
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=2, role="assistant", content="A"),
            ]
        )
        repo.get_chunk_page_information = AsyncMock(
            return_value={sample_retrieved_chunk.chunk_id: 1, second_chunk.chunk_id: 2}
        )
        repo.create_citations = AsyncMock()
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "Compare Attention and BERT.", mode="advanced")

        # Exactly 1 LLM call made
        mock_llm.generate_grounded_answer.assert_awaited_once()
        call_evidence = mock_llm.generate_grounded_answer.call_args[1]["evidence_chunks"]
        assert len(call_evidence) == 2
        assert sample_retrieved_chunk in call_evidence
        assert second_chunk in call_evidence
        assert mock_llm.generate_grounded_answer.call_args[1]["mode"] == "advanced"

        # Citations for both papers are validated
        assert len(response.citations) == 2
        assert {c.chunk_id for c in response.citations} == {sample_retrieved_chunk.chunk_id, second_chunk.chunk_id}


@pytest.mark.asyncio
async def test_advanced_mode_no_evidence_fallback(
    mock_session_factory: MagicMock,
) -> None:
    """Verify Advanced Mode returns no-evidence response without calling LLM when retrieval is empty."""
    conv_id = uuid.uuid4()
    active_conv = Conversation(id=conv_id, status="active")

    mock_retrieval = MagicMock()
    mock_retrieval.is_initialized = True
    mock_retrieval.search.return_value = []

    mock_llm = MagicMock(spec=OpenAIResponseClient)
    mock_llm.generate_grounded_answer = AsyncMock()

    service = ConversationService(
        session_factory=mock_session_factory,
        retrieval_service=mock_retrieval,
        llm_client=mock_llm,
    )

    with patch("app.modules.conversation.service.ConversationRepository") as mock_repo_cls:
        repo = mock_repo_cls.return_value
        repo.get_conversation = AsyncMock(return_value=active_conv)
        repo.get_recent_messages = AsyncMock(return_value=[])
        repo.get_next_sequence_number = AsyncMock(side_effect=[1, 2])
        repo.create_message = AsyncMock(
            side_effect=[
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=1, role="user", content="Q"),
                Message(id=uuid.uuid4(), conversation_id=conv_id, sequence_number=2, role="assistant", content=NO_EVIDENCE_RESPONSE),
            ]
        )
        repo.update_conversation_last_message_at = AsyncMock()

        response = await service.ask_question(conv_id, "Unindexed topic question", mode="advanced")

        assert response.answer == NO_EVIDENCE_RESPONSE
        assert response.mode == "advanced"
        assert response.citations == []
        mock_retrieval.search.assert_called_once()
        mock_llm.generate_grounded_answer.assert_not_called()
