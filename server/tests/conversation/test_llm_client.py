"""Unit tests for OpenAIResponseClient using mock AsyncOpenAI."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from app.core.config import LoggingSettings
from app.modules.conversation.llm_client import (
    ADVANCED_SYSTEM_INSTRUCTIONS,
    STRUCTURED_OUTPUT_TEXT_CONFIG,
    SYSTEM_INSTRUCTIONS,
    LLMAnswerResult,
    LLMClientError,
    LLMConfigurationError,
    OpenAIResponseClient,
)
from app.modules.retrieval.service import RetrievedChunk


@pytest.fixture
def mock_async_openai() -> MagicMock:
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        paper_id=uuid.uuid4(),
        chunk_index=0,
        content="Self-attention connects all tokens with O(1) sequential operations.",
        score=4.5,
        paper_title="Attention Is All You Need",
        publication_year=2017,
    )


@pytest.mark.asyncio
async def test_generate_grounded_answer_success(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify structured output parsing, answer text, and cited chunk IDs extraction."""
    expected_answer = "Self-attention requires O(1) sequential operations."
    cited_id = str(sample_chunk.chunk_id)

    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.error = None
    mock_response.output_text = json.dumps({
        "answer": expected_answer,
        "cited_chunk_ids": [cited_id],
    })
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 42

    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(
        client=mock_async_openai,
        default_model="gpt-test-model",
    )

    result = await llm_client.generate_grounded_answer(
        question="What are sequential operations for self-attention?",
        evidence_chunks=[sample_chunk],
        conversation_history=[{"role": "user", "content": "Hi"}],
    )

    assert isinstance(result, LLMAnswerResult)
    assert result.answer == expected_answer
    assert result.cited_chunk_ids == [cited_id]
    assert result.model_name == "gpt-test-model"
    assert result.token_count == 42

    # Verify input prompt formatting
    create_call = mock_async_openai.responses.create.call_args
    assert create_call is not None
    _, kwargs = create_call
    assert kwargs["model"] == "gpt-test-model"
    assert "Self-attention connects all tokens" in kwargs["input"]
    assert "Hi" in kwargs["input"]
    assert "untrusted data" in kwargs["instructions"].lower()


@pytest.mark.asyncio
async def test_generate_grounded_answer_empty_output_raises_error(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify empty output_text raises LLMClientError."""
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.error = None
    mock_response.output_text = "   "
    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai)

    with pytest.raises(LLMClientError) as exc_info:
        await llm_client.generate_grounded_answer(
            question="Question?",
            evidence_chunks=[sample_chunk],
        )

    assert "empty response" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_generate_grounded_answer_api_error_response_raises(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify error object on response raises LLMClientError."""
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.error = MagicMock()
    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai)

    with pytest.raises(LLMClientError) as exc_info:
        await llm_client.generate_grounded_answer(
            question="Question?",
            evidence_chunks=[sample_chunk],
        )

    assert "error" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_generate_grounded_answer_malformed_json_raises(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify non-JSON response raises LLMClientError."""
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.error = None
    mock_response.output_text = "This is not JSON at all."
    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai)

    with pytest.raises(LLMClientError) as exc_info:
        await llm_client.generate_grounded_answer(
            question="Question?",
            evidence_chunks=[sample_chunk],
        )

    assert "failed to parse" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_generate_grounded_answer_api_exception_handled(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify network or API exception is cleanly wrapped in LLMClientError without leaking secrets."""
    mock_async_openai.responses.create.side_effect = RuntimeError("Connection timeout secret_key_123")

    llm_client = OpenAIResponseClient(client=mock_async_openai)

    with pytest.raises(LLMClientError) as exc_info:
        await llm_client.generate_grounded_answer(
            question="Question?",
            evidence_chunks=[sample_chunk],
        )

    assert "OpenAI service communication failed." in str(exc_info.value)
    assert "secret_key_123" not in str(exc_info.value)
    assert "Connection timeout" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_grounded_answer_incomplete_status_raises(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify response.status == 'incomplete' raises LLMClientError."""
    mock_response = MagicMock()
    mock_response.status = "incomplete"
    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai)

    with pytest.raises(LLMClientError) as exc_info:
        await llm_client.generate_grounded_answer(
            question="Question?",
            evidence_chunks=[sample_chunk],
        )

    assert "incomplete" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_generate_grounded_answer_failed_status_raises(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify response.status == 'failed' raises LLMClientError."""
    mock_response = MagicMock()
    mock_response.status = "failed"
    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai)

    with pytest.raises(LLMClientError) as exc_info:
        await llm_client.generate_grounded_answer(
            question="Question?",
            evidence_chunks=[sample_chunk],
        )

    assert "failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_generate_grounded_answer_refusal_raises(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify response with refusal attribute raises fixed safe LLMClientError without refusal text."""
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = "I cannot fulfill this request because of policy XYZ-999."
    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai)

    with pytest.raises(LLMClientError) as exc_info:
        await llm_client.generate_grounded_answer(
            question="Question?",
            evidence_chunks=[sample_chunk],
        )

    assert str(exc_info.value) == "Model refused to answer the question."
    assert "XYZ-999" not in str(exc_info.value)
    assert "I cannot fulfill this request" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_grounded_answer_refusal_item_in_output_raises(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify refusal item in response.output raises fixed safe LLMClientError without refusal text."""
    mock_item = MagicMock()
    mock_item.type = "refusal"
    mock_item.refusal = "Safety policy refusal detail sensitive_data_456"
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.output = [mock_item]
    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai)

    with pytest.raises(LLMClientError) as exc_info:
        await llm_client.generate_grounded_answer(
            question="Question?",
            evidence_chunks=[sample_chunk],
        )

    assert str(exc_info.value) == "Model refused to answer the question."
    assert "sensitive_data_456" not in str(exc_info.value)

    assert "refused" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_generate_grounded_answer_empty_answer_in_json_raises(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify parsed answer that is empty or whitespace raises LLMClientError."""
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.error = None
    mock_response.output = []
    mock_response.output_text = json.dumps({"answer": "   ", "cited_chunk_ids": []})
    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai)

    with pytest.raises(LLMClientError) as exc_info:
        await llm_client.generate_grounded_answer(
            question="Question?",
            evidence_chunks=[sample_chunk],
        )

    assert "non-empty answer" in str(exc_info.value).lower()


def test_openai_client_empty_api_key_raises_configuration_error() -> None:
    """Verify empty or whitespace API key raises LLMConfigurationError."""
    with pytest.raises(LLMConfigurationError) as exc_info:
        OpenAIResponseClient(api_key="   ")
    assert "missing or empty" in str(exc_info.value).lower()


def test_system_instructions_evidence_rules() -> None:
    """Verify system instructions enforce that history is not evidence and chunks are sole evidence."""
    assert "Agent" in SYSTEM_INSTRUCTIONS
    assert "ONLY source of factual evidence" in SYSTEM_INSTRUCTIONS
    assert "NOT evidence and must never be treated as facts" in SYSTEM_INSTRUCTIONS
    assert "untrusted data" in SYSTEM_INSTRUCTIONS
    assert "specifically to assist with your available research papers" in SYSTEM_INSTRUCTIONS


def test_advanced_system_instructions_principles() -> None:
    """Verify ADVANCED_SYSTEM_INSTRUCTIONS provides flexibility without forcing length or rigid headings."""
    assert "Explain concepts in more detail when the question genuinely requires it" in ADVANCED_SYSTEM_INSTRUCTIONS
    assert "Use examples or structure when they improve clarity" in ADVANCED_SYSTEM_INSTRUCTIONS
    assert "Connect retrieved evidence when relevant and supported" in ADVANCED_SYSTEM_INSTRUCTIONS
    assert "Keep answers focused and avoid unnecessary detail" in ADVANCED_SYSTEM_INSTRUCTIONS
    assert "Do not require fixed headings, comprehensive analysis, or an artificially longer answer" in ADVANCED_SYSTEM_INSTRUCTIONS

    # Verify all 9 critical grounding and scope rules are intact
    assert "ONLY source of factual evidence" in ADVANCED_SYSTEM_INSTRUCTIONS
    assert "NOT evidence and must never be treated as facts" in ADVANCED_SYSTEM_INSTRUCTIONS
    assert "untrusted data" in ADVANCED_SYSTEM_INSTRUCTIONS
    assert "specifically to assist with your available research papers" in ADVANCED_SYSTEM_INSTRUCTIONS
    assert "Never invent, extrapolate, or hallucinate findings" in ADVANCED_SYSTEM_INSTRUCTIONS
    assert "return an empty list for 'cited_chunk_ids'" in ADVANCED_SYSTEM_INSTRUCTIONS


@pytest.mark.asyncio
async def test_close_calls_client_close(mock_async_openai: MagicMock) -> None:
    """Verify close() delegates to underlying AsyncOpenAI client."""
    llm_client = OpenAIResponseClient(client=mock_async_openai)
    await llm_client.close()
    mock_async_openai.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# LLM Request and Response Tracing Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_tracing_normal_mode_request_and_response(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify Normal Mode logs llm_request_started and llm_response_completed with accurate payload and tokens."""
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.error = None
    mock_response.output = []
    mock_response.output_text = json.dumps({
        "answer": "Grounded answer text.",
        "cited_chunk_ids": [str(sample_chunk.chunk_id)],
    })
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 120
    mock_response.usage.output_tokens = 45
    mock_response.usage.total_tokens = 165
    mock_response.model = "gpt-4o-mini-2024-07-18"

    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai, default_model="gpt-4o-mini")

    with patch("app.modules.conversation.llm_client.log_event") as mock_log_event:
        result = await llm_client.generate_grounded_answer(
            question="What is the Transformer architecture?",
            evidence_chunks=[sample_chunk],
            conversation_history=[{"role": "user", "content": "Hello"}],
        )

        assert result.answer == "Grounded answer text."
        assert result.cited_chunk_ids == [str(sample_chunk.chunk_id)]

        # Check logged events
        event_names = [call[0][2] for call in mock_log_event.call_args_list]
        assert "llm_request_started" in event_names
        assert "llm_response_completed" in event_names

        # Verify request event attributes
        req_call = next(c for c in mock_log_event.call_args_list if c[0][2] == "llm_request_started")
        req_kwargs = req_call[1]
        assert req_kwargs["stage"] == "grounded_answer"
        assert req_kwargs["model_name"] == "gpt-4o-mini"
        assert req_kwargs["system_instructions"] == SYSTEM_INSTRUCTIONS
        assert req_kwargs["user_question"] == "What is the Transformer architecture?"
        assert req_kwargs["conversation_history"] == [{"role": "user", "content": "Hello"}]
        assert req_kwargs["evidence_chunk_count"] == 1
        assert req_kwargs["total_evidence_characters"] == len(sample_chunk.content)
        assert req_kwargs["estimated_input_tokens"] > 0
        assert req_kwargs["evidence_details"][0]["chunk_id"] == str(sample_chunk.chunk_id)

        # Verify response event attributes
        resp_call = next(c for c in mock_log_event.call_args_list if c[0][2] == "llm_response_completed")
        resp_kwargs = resp_call[1]
        assert resp_kwargs["stage"] == "grounded_answer"
        assert resp_kwargs["model_name"] == "gpt-4o-mini-2024-07-18"
        assert resp_kwargs["input_tokens"] == 120
        assert resp_kwargs["output_tokens"] == 45
        assert resp_kwargs["total_tokens"] == 165
        assert resp_kwargs["status"] == "completed"
        assert resp_kwargs["parsed_answer"] == "Grounded answer text."
        assert resp_kwargs["cited_chunk_ids"] == [str(sample_chunk.chunk_id)]
        assert resp_kwargs["duration_ms"] >= 0.0

        # Verify exact payload captured matches the payload passed to the OpenAI API
        api_create_kwargs = mock_async_openai.responses.create.call_args[1]
        assert req_kwargs["api_request_payload"] == api_create_kwargs
        assert req_kwargs["api_request_payload"]["model"] == "gpt-4o-mini"
        assert req_kwargs["api_request_payload"]["instructions"] == SYSTEM_INSTRUCTIONS
        assert req_kwargs["api_request_payload"]["text"] == STRUCTURED_OUTPUT_TEXT_CONFIG
        assert req_kwargs["api_request_payload"]["input"] == api_create_kwargs["input"]
        assert "Conversation History:\nUser: Hello" in api_create_kwargs["input"]
        assert f"Chunk ID: {sample_chunk.chunk_id}" in api_create_kwargs["input"]
        assert f"Paper Title: {sample_chunk.paper_title}" in api_create_kwargs["input"]
        assert sample_chunk.content in api_create_kwargs["input"]
        assert "User Question: What is the Transformer architecture?" in api_create_kwargs["input"]


@pytest.mark.asyncio
async def test_llm_tracing_advanced_mode_single_stage(
    mock_async_openai: MagicMock, sample_chunk: RetrievedChunk
) -> None:
    """Verify Advanced Mode executes exactly one LLM call and logs exact request payload."""
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.error = None
    mock_response.output = []
    mock_response.output_text = json.dumps({
        "answer": "Advanced in-depth explanation with connections.",
        "cited_chunk_ids": [str(sample_chunk.chunk_id)],
    })
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 220
    mock_response.usage.output_tokens = 95
    mock_response.usage.total_tokens = 315
    mock_response.model = "gpt-4o-mini"

    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(client=mock_async_openai, default_model="gpt-4o-mini")

    with patch("app.modules.conversation.llm_client.log_event") as mock_log_event:
        # Run Advanced Mode single call
        result = await llm_client.generate_grounded_answer(
            question="Analyze Transformer mechanics in depth.",
            evidence_chunks=[sample_chunk],
            conversation_history=[{"role": "user", "content": "Hello"}],
            mode="advanced",
        )

        assert result.answer == "Advanced in-depth explanation with connections."
        assert result.cited_chunk_ids == [str(sample_chunk.chunk_id)]

        # Exactly ONE call to OpenAI Responses API
        mock_async_openai.responses.create.assert_awaited_once()

        # Check logged events: exactly one request_started and one response_completed
        started_calls = [c for c in mock_log_event.call_args_list if c[0][2] == "llm_request_started"]
        completed_calls = [c for c in mock_log_event.call_args_list if c[0][2] == "llm_response_completed"]
        assert len(started_calls) == 1
        assert len(completed_calls) == 1

        req_event = started_calls[0]
        assert req_event[1]["stage"] == "advanced_grounded_answer"
        assert req_event[1]["system_instructions"] == ADVANCED_SYSTEM_INSTRUCTIONS
        assert req_event[1]["mode"] == "advanced"

        # Verify exact request payload matches kwargs of responses.create()
        api_create_kwargs = mock_async_openai.responses.create.call_args[1]
        assert req_event[1]["api_request_payload"] == api_create_kwargs
        assert req_event[1]["api_request_payload"]["model"] == "gpt-4o-mini"
        assert req_event[1]["api_request_payload"]["instructions"] == ADVANCED_SYSTEM_INSTRUCTIONS
        assert req_event[1]["api_request_payload"]["text"] == STRUCTURED_OUTPUT_TEXT_CONFIG
        assert req_event[1]["api_request_payload"]["input"] == api_create_kwargs["input"]
        assert sample_chunk.content in api_create_kwargs["input"]

        resp_event = completed_calls[0]
        assert resp_event[1]["stage"] == "advanced_grounded_answer"
        assert resp_event[1]["input_tokens"] == 220
        assert resp_event[1]["output_tokens"] == 95
        assert resp_event[1]["total_tokens"] == 315
        assert resp_event[1]["parsed_answer"] == "Advanced in-depth explanation with connections."
        assert resp_event[1]["cited_chunk_ids"] == [str(sample_chunk.chunk_id)]


def test_llm_tracing_disabled_suppresses_content_keeps_operational_metadata() -> None:
    """Verify LLM_TRACE_ENABLED=False suppresses detailed prompts, evidence, and answers while keeping metadata."""
    from app.core.logging_config import log_event

    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True

    settings_off = LoggingSettings(llm_trace_enabled=False, llm_trace_full_content=False)

    log_event(
        mock_logger,
        logging.INFO,
        "llm_request_started",
        stage="grounded_answer",
        model_name="gpt-4o-mini",
        system_instructions="Top secret instructions that should be suppressed.",
        user_question="Confidential user question.",
        api_request_payload={"model": "gpt-4o-mini", "input": "Confidential input payload"},
        total_evidence_characters=1500,
        evidence_chunk_count=3,
        estimated_input_tokens=400,
        settings=settings_off,
    )

    logged_req = mock_logger.log.call_args[0][1]
    # Operational metadata must be present
    assert "stage=grounded_answer" in logged_req
    assert "model_name=gpt-4o-mini" in logged_req
    assert "total_evidence_characters=1500" in logged_req
    assert "evidence_chunk_count=3" in logged_req
    assert "estimated_input_tokens=400" in logged_req

    # Detailed content must be completely suppressed
    assert "system_instructions" not in logged_req
    assert "Top secret instructions" not in logged_req
    assert "user_question" not in logged_req
    assert "Confidential user question" not in logged_req
    assert "api_request_payload" not in logged_req
    assert "Confidential input payload" not in logged_req

    mock_logger.reset_mock()

    log_event(
        mock_logger,
        logging.INFO,
        "llm_response_completed",
        stage="grounded_answer",
        model_name="gpt-4o-mini",
        duration_ms=520.4,
        status="completed",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        raw_response='{"answer": "Confidential response answer"}',
        parsed_answer="Confidential response answer",
        cited_chunk_ids=["uuid-1", "uuid-2"],
        settings=settings_off,
    )

    logged_resp = mock_logger.log.call_args[0][1]
    # Operational metadata must be present
    assert "stage=grounded_answer" in logged_resp
    assert "duration_ms=520.4" in logged_resp
    assert "total_tokens=150" in logged_resp
    assert "status=completed" in logged_resp
    assert "cited_chunk_ids=['uuid-1', 'uuid-2']" in logged_resp

    # Detailed content must be completely suppressed
    assert "raw_response" not in logged_resp
    assert "parsed_answer" not in logged_resp
    assert "Confidential response answer" not in logged_resp


def test_llm_tracing_full_content_false_truncates_previews() -> None:
    """Verify LLM_TRACE_FULL_CONTENT=False emits truncated previews rather than full prompts or answers."""
    from app.core.logging_config import log_event

    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True

    settings_preview = LoggingSettings(llm_trace_enabled=True, llm_trace_full_content=False)

    long_instructions = "A" * 300
    long_question = "B" * 300

    log_event(
        mock_logger,
        logging.INFO,
        "llm_request_started",
        stage="grounded_answer",
        system_instructions=long_instructions,
        user_question=long_question,
        settings=settings_preview,
    )

    logged_req = mock_logger.log.call_args[0][1]
    assert "system_instructions=" in logged_req
    assert "user_question=" in logged_req
    # Must be truncated with ellipsis
    assert f"{'A' * 80}..." in logged_req
    assert ('A' * 300) not in logged_req
    assert f"{'B' * 80}..." in logged_req
    assert ('B' * 300) not in logged_req


def test_llm_tracing_secret_redaction_in_prompts_evidence_and_exceptions() -> None:
    """Verify OpenAI keys, Bearer tokens, passwords, and API keys are redacted even in full content mode."""
    from app.core.logging_config import log_event

    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True

    settings_full = LoggingSettings(llm_trace_enabled=True, llm_trace_full_content=True)

    sensitive_prompt = (
        "Instructions with sk-proj-1234567890abcdef12345678 and "
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.topsecret and "
        "password: 'secret_admin_pass' and api_key: 'top_secret_key'"
    )

    log_event(
        mock_logger,
        logging.INFO,
        "llm_request_started",
        stage="grounded_answer",
        system_instructions=sensitive_prompt,
        settings=settings_full,
    )

    logged_req = mock_logger.log.call_args[0][1]
    assert "sk-proj-1234567890abcdef12345678" not in logged_req
    assert "eyJhbGci" not in logged_req
    assert "secret_admin_pass" not in logged_req
    assert "top_secret_key" not in logged_req
    assert "[REDACTED" in logged_req

    mock_logger.reset_mock()

    # Exception logging redaction
    log_event(
        mock_logger,
        logging.ERROR,
        "llm_request_failed",
        stage="grounded_answer",
        error_message="Failed call with api_key: 'sk-proj-leak1234567890abcdef'",
        settings=settings_full,
    )

    logged_err = mock_logger.log.call_args[0][1]
    assert "sk-proj-leak" not in logged_err
    assert "[REDACTED" in logged_err


def test_llm_tracing_api_request_payload_controls_and_secret_redaction() -> None:
    """Verify api_request_payload obeys suppression, preview truncation, and secret redaction."""
    from app.core.logging_config import log_event

    mock_logger = MagicMock()
    mock_logger.isEnabledFor.return_value = True

    long_instructions = "Inst " * 40  # 200 chars
    long_input = "Input text " * 30    # 330 chars
    sensitive_input = (
        f"sk-proj-supersecretkey12345678 {long_input} and "
        "Authorization: Bearer secrettoken1234567890 and password: 'leak'"
    )

    request_payload = {
        "model": "gpt-4o-mini",
        "instructions": long_instructions,
        "input": sensitive_input,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "test_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "Long schema description " * 10},
                    },
                },
            },
        },
    }

    # 1. Tracing disabled: payload completely omitted
    settings_off = LoggingSettings(llm_trace_enabled=False, llm_trace_full_content=False)
    log_event(
        mock_logger,
        logging.INFO,
        "llm_request_started",
        stage="grounded_answer",
        model_name="gpt-4o-mini",
        api_request_payload=request_payload,
        settings=settings_off,
    )
    logged_off = mock_logger.log.call_args[0][1]
    assert "api_request_payload" not in logged_off
    assert "gpt-4o-mini" in logged_off

    mock_logger.reset_mock()

    # 2. Preview mode: nested values are sanitized and truncated with ellipsis
    settings_preview = LoggingSettings(llm_trace_enabled=True, llm_trace_full_content=False)
    log_event(
        mock_logger,
        logging.INFO,
        "llm_request_started",
        stage="grounded_answer",
        model_name="gpt-4o-mini",
        api_request_payload=request_payload,
        settings=settings_preview,
    )
    logged_preview = mock_logger.log.call_args[0][1]
    assert "api_request_payload=" in logged_preview
    # Must have bounded previews with ellipsis for long nested values
    assert "..." in logged_preview
    assert long_instructions not in logged_preview
    # Secrets must NEVER appear
    assert "sk-proj-supersecretkey" not in logged_preview
    assert "secrettoken1234567890" not in logged_preview
    assert "leak" not in logged_preview
    assert "[REDACTED" in logged_preview

    mock_logger.reset_mock()

    # 3. Full-content mode: full payload retained, secrets redacted
    settings_full = LoggingSettings(llm_trace_enabled=True, llm_trace_full_content=True)
    log_event(
        mock_logger,
        logging.INFO,
        "llm_request_started",
        stage="grounded_answer",
        model_name="gpt-4o-mini",
        api_request_payload=request_payload,
        settings=settings_full,
    )
    logged_full = mock_logger.log.call_args[0][1]
    assert "api_request_payload=" in logged_full
    # Full instructions must be present without ellipsis truncation
    assert long_instructions in logged_full
    # Input prefix retained
    assert "Input text" in logged_full
    # Secrets must NEVER appear even in full content mode
    assert "sk-proj-supersecretkey" not in logged_full
    assert "secrettoken1234567890" not in logged_full
    assert "leak" not in logged_full
    assert "[REDACTED_API_KEY]" in logged_full
    assert "[REDACTED_TOKEN]" in logged_full


@pytest.mark.asyncio
async def test_generate_general_answer_success(mock_async_openai: MagicMock) -> None:
    """Verify generate_general_answer parses JSON output and returns LLMAnswerResult."""
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.error = None
    mock_response.output_text = json.dumps({
        "answer": "This is a direct conversational answer.",
    })
    mock_response.usage = MagicMock()
    mock_response.usage.total_tokens = 25

    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(
        client=mock_async_openai,
        default_model="gpt-test-model",
    )

    result = await llm_client.generate_general_answer(
        question="What is deep learning?",
        conversation_history=[{"role": "user", "content": "Hello"}],
        conversation_summary="Introductory chat.",
    )

    assert result.answer == "This is a direct conversational answer."
    assert result.cited_chunk_ids == []
    assert result.model_name == "gpt-test-model"
    assert result.token_count == 25
    mock_async_openai.responses.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_summarize_messages_success(mock_async_openai: MagicMock) -> None:
    """Verify summarize_messages parses summary output and returns clean summary string."""
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.refusal = None
    mock_response.error = None
    mock_response.output_text = json.dumps({
        "summary": "The user asked about transformer models and attention mechanisms.",
    })

    mock_async_openai.responses.create.return_value = mock_response

    llm_client = OpenAIResponseClient(
        client=mock_async_openai,
        default_model="gpt-test-model",
    )

    summary = await llm_client.summarize_messages(
        messages=[
            {"role": "user", "content": "What is attention?"},
            {"role": "assistant", "content": "Attention is a mechanism..."},
        ],
        existing_summary="Previous intro.",
    )

    assert summary == "The user asked about transformer models and attention mechanisms."
    mock_async_openai.responses.create.assert_awaited_once()

