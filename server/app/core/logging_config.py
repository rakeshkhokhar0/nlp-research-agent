"""Shared logging infrastructure and context-aware correlation tracking for NLP Research Agent."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Generator

from app.core.config import LoggingSettings, logging_settings

# Asynchronous Context Variables for Correlation Tracking
request_id_ctx: ContextVar[str | None] = ContextVar("request_id_ctx", default=None)
conversation_id_ctx: ContextVar[str | None] = ContextVar("conversation_id_ctx", default=None)
ingestion_job_id_ctx: ContextVar[str | None] = ContextVar("ingestion_job_id_ctx", default=None)


def get_correlation_ids() -> dict[str, str | None]:
    """Retrieve current correlation identifiers from the active async context."""
    return {
        "request_id": request_id_ctx.get(),
        "conversation_id": conversation_id_ctx.get(),
        "ingestion_job_id": ingestion_job_id_ctx.get(),
    }


@contextmanager
def correlation_context(
    request_id: str | None = None,
    conversation_id: str | None = None,
    ingestion_job_id: str | None = None,
) -> Generator[dict[str, str | None], None, None]:
    """Context manager to set and cleanly reset correlation IDs in async execution.

    Args:
        request_id: Optional HTTP / conversation request identifier.
        conversation_id: Optional research conversation session identifier.
        ingestion_job_id: Optional PDF ingestion pipeline job identifier.

    Yields:
        Dictionary containing active correlation identifiers.
    """
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    try:
        if request_id is not None:
            tokens.append((request_id_ctx, request_id_ctx.set(str(request_id))))
        if conversation_id is not None:
            tokens.append((conversation_id_ctx, conversation_id_ctx.set(str(conversation_id))))
        if ingestion_job_id is not None:
            tokens.append((ingestion_job_id_ctx, ingestion_job_id_ctx.set(str(ingestion_job_id))))

        yield get_correlation_ids()
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


# Secret Scrubbing & Redaction
_SECRET_PATTERNS = [
    re.compile(r"(sk-[a-zA-Z0-9_-]{16,})", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)([^'\";\s]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(api[_-]?secret\s*[:=]\s*['\"]?)([^'\";\s]{8,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(password\s*[:=]\s*['\"]?)([^'\";\s]{4,})(['\"]?)", re.IGNORECASE),
    re.compile(r"(bearer\s+)([a-zA-Z0-9_.-]{16,})", re.IGNORECASE),
]


def sanitize_log_text(text: str) -> str:
    """Scrub sensitive keys, passwords, and tokens from log strings."""
    if not isinstance(text, str):
        return str(text)

    scrubbed = text
    for pattern in _SECRET_PATTERNS:
        if "bearer" in pattern.pattern.lower():
            scrubbed = pattern.sub(r"\1[REDACTED_TOKEN]", scrubbed)
        elif "api" in pattern.pattern.lower() or "password" in pattern.pattern.lower():
            scrubbed = pattern.sub(r"\1[REDACTED]\3", scrubbed)
        else:
            scrubbed = pattern.sub("[REDACTED_API_KEY]", scrubbed)
    return scrubbed


def sanitize_log_value(val: Any) -> Any:
    """Recursively scrub values in structured payloads."""
    if isinstance(val, str):
        return sanitize_log_text(val)
    if isinstance(val, dict):
        return {
            k: ("[REDACTED]" if any(s in str(k).lower() for s in ("secret", "key", "password", "token")) else sanitize_log_value(v))
            for k, v in val.items()
        }
    if isinstance(val, (list, tuple, set)):
        return [sanitize_log_value(item) for item in val]
    return val


def sanitize_log_preview_value(val: Any, max_len: int = 80) -> Any:
    """Recursively scrub secrets and truncate string values to bounded previews."""
    if isinstance(val, str):
        sanitized = sanitize_log_text(val).replace("\n", " ").strip()
        return f"{sanitized[:max_len]}..." if len(sanitized) > max_len else sanitized
    if isinstance(val, dict):
        return {
            k: (
                "[REDACTED]"
                if any(s in str(k).lower() for s in ("secret", "key", "password", "token"))
                else sanitize_log_preview_value(v, max_len)
            )
            for k, v in val.items()
        }
    if isinstance(val, (list, tuple, set)):
        return [sanitize_log_preview_value(item, max_len) for item in val]
    return sanitize_log_value(val)


# Structured Log Formatter & Filter
class CorrelationFilter(logging.Filter):
    """Logging filter that injects active correlation IDs into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "-"
        record.conversation_id = conversation_id_ctx.get() or "-"
        record.ingestion_job_id = ingestion_job_id_ctx.get() or "-"
        return True


class StructuredLogFormatter(logging.Formatter):
    """Produces consistent, structured log lines with ISO timestamps and correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        # Standard attributes
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        req_id = getattr(record, "request_id", "-")
        conv_id = getattr(record, "conversation_id", "-")
        job_id = getattr(record, "ingestion_job_id", "-")

        # ISO 8601 UTC timestamp derived from record.created
        record_dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        timestamp = record_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"

        # Format prefix
        prefix = (
            f"[{timestamp}] [{record.levelname:<5}] [{record.name}] "
            f"[req:{req_id} conv:{conv_id} job:{job_id}]"
        )

        message_clean = sanitize_log_text(record.message)
        formatted = f"{prefix} {message_clean}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            sanitized_exc = sanitize_log_text(record.exc_text)
            record.exc_text = sanitized_exc
            formatted = f"{formatted}\n{sanitized_exc}"

        return formatted


# Structured Event Logging Helper
_NLP_TRACE_EVENT_NAMES = {
    "retrieval_query_diagnostics",
    "retrieval_candidates_evaluated",
    "retrieval_candidate_scoring_detail",
    "retrieval_ranking_diagnostics",
    "retrieval_query_processed",
    "retrieval_top_k_chunk_selected",
    "retrieval_lexical_scoring_completed",
    "ingestion_chunk_created",
    "ingestion_chunk_diagnostics",
}

_DETAILED_LLM_FIELDS = {
    "api_request_payload",
    "system_instructions",
    "user_question",
    "conversation_history",
    "input_prompt",
    "grouped_evidence",
    "evidence_details",
    "analysis_summary",
    "raw_response",
    "parsed_answer",
    "parsed_claims",
    "cross_paper_comparisons",
    "conflicting_evidence",
    "evidence_gaps",
}


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    is_trace: bool | None = None,
    preview_field: str | None = None,
    settings: LoggingSettings | None = None,
    **metrics: Any,
) -> None:
    """Emit a structured event log safely without risk of raising exceptions.

    Args:
        logger: Target standard logger instance.
        level: logging level (e.g. logging.INFO, logging.DEBUG).
        event: Unique structured event identifier (e.g. 'retrieval_query_processed').
        is_trace: Optional explicit flag whether this event is an NLP trace event.
        preview_field: Optional key in metrics holding content preview text to sanitize/truncate.
        settings: LoggingSettings instance for trace and preview flags.
        metrics: Key-value attributes describing durations, counts, scores, or identifiers.
    """
    try:
        cfg = settings or logging_settings

        is_llm_event = event.startswith("llm_")

        # Determine whether this event is subject to NLP trace gating
        if is_trace is not None:
            is_nlp_trace = is_trace
        elif is_llm_event:
            is_nlp_trace = False
        else:
            # By default, DEBUG events and specific detailed NLP diagnostic events are traces;
            # essential operational lifecycle events remain active.
            is_nlp_trace = (
                level == logging.DEBUG
                or event in _NLP_TRACE_EVENT_NAMES
                or event.endswith(("_diagnostics", "_scoring_detail", "_trace"))
            )

        if is_nlp_trace and not cfg.nlp_trace_enabled and level < logging.WARNING:
            return

        if not logger.isEnabledFor(level):
            return

        cleaned_metrics: dict[str, Any] = {}
        for k, v in metrics.items():
            if is_llm_event and k in _DETAILED_LLM_FIELDS:
                if not cfg.llm_trace_enabled:
                    # Detailed prompt, instructions, evidence, and response content suppressed
                    continue
                if not cfg.llm_trace_full_content:
                    # Preview mode: sanitize and truncate recursively
                    cleaned_metrics[k] = sanitize_log_preview_value(v, max_len=80)
                else:
                    # Full content mode: full text sanitized against credential leaks
                    cleaned_metrics[k] = sanitize_log_value(v)
            elif k == preview_field:
                if cfg.nlp_trace_content_preview and isinstance(v, str):
                    # Ensure designated content preview passes through sanitize_log_text before truncation
                    sanitized_text = sanitize_log_text(v).replace("\n", " ").strip()
                    preview = sanitized_text[:60]
                    cleaned_metrics[k] = f"{preview}..." if len(sanitized_text) > 60 else preview
                else:
                    cleaned_metrics[k] = "[PREVIEW_DISABLED]"
            else:
                cleaned_metrics[k] = sanitize_log_value(v)

        kv_pairs = " ".join(f"{k}={v}" for k, v in cleaned_metrics.items())
        msg = f"event={event} {kv_pairs}".strip()
        logger.log(level, msg)
    except Exception:
        # Logging failures must NEVER break application pipelines
        pass


# Centralized Setup
def setup_logging(settings: LoggingSettings | None = None) -> None:
    """Configure standard logging with StructuredLogFormatter and CorrelationFilter."""
    cfg = settings or logging_settings
    log_level_name = getattr(cfg, "log_level", "INFO").upper()
    numeric_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Check if our handler is already attached
    has_structured_handler = any(
        isinstance(getattr(h, "formatter", None), StructuredLogFormatter)
        for h in root_logger.handlers
    )

    if not has_structured_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredLogFormatter())
        handler.addFilter(CorrelationFilter())
        handler.setLevel(numeric_level)
        root_logger.addHandler(handler)
    else:
        for h in root_logger.handlers:
            if isinstance(getattr(h, "formatter", None), StructuredLogFormatter):
                h.setLevel(numeric_level)
                h.addFilter(CorrelationFilter())
