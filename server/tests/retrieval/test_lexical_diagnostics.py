"""Tests for detailed lexical retrieval scoring diagnostics and trace gating."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import LoggingSettings
from app.modules.retrieval.lexical.engine import LexicalRetrievalEngine


def test_lexical_diagnostics_emitted_when_trace_and_debug_enabled() -> None:
    """Verify detailed diagnostics are emitted when NLP_TRACE_ENABLED=True and logger is at DEBUG level."""
    engine = LexicalRetrievalEngine()
    engine.add_document(1, "Deep learning models require extensive training on GPUs.")
    engine.add_document(2, "Natural language processing models use attention mechanisms.")

    with patch("app.modules.retrieval.lexical.engine.logging_settings") as mock_settings, \
         patch("app.modules.retrieval.lexical.engine.logger") as mock_logger, \
         patch("app.modules.retrieval.lexical.engine.log_event") as mock_log_event:

        mock_settings.nlp_trace_enabled = True
        mock_logger.isEnabledFor.return_value = True

        results = engine.search("attention mechanisms models", top_k=5)

        assert len(results) > 0

        # Check diagnostic events emitted
        event_names = [call[0][2] for call in mock_log_event.call_args_list]
        assert "retrieval_query_diagnostics" in event_names
        assert "retrieval_candidates_evaluated" in event_names
        assert "retrieval_candidate_scoring_detail" in event_names
        assert "retrieval_ranking_diagnostics" in event_names

        # Verify query diagnostics content
        query_diag_call = next(c for c in mock_log_event.call_args_list if c[0][2] == "retrieval_query_diagnostics")
        kwargs = query_diag_call[1]
        assert kwargs["is_trace"] is True
        assert kwargs["corpus_n"] == 2
        assert "query_tf" in kwargs
        assert "query_df" in kwargs
        assert "query_idf" in kwargs
        assert "query_raw_weights" in kwargs
        assert "query_l2_norm" in kwargs
        assert "query_normalized_weights" in kwargs

        # Verify candidate scoring detail content
        cand_detail_call = next(c for c in mock_log_event.call_args_list if c[0][2] == "retrieval_candidate_scoring_detail")
        c_kwargs = cand_detail_call[1]
        assert "doc_id" in c_kwargs
        assert "doc_tf" in c_kwargs
        assert "doc_l2_norm" in c_kwargs
        assert "per_term_cosine_contributions" in c_kwargs
        assert "cosine_similarity" in c_kwargs


def test_lexical_diagnostics_suppressed_when_trace_disabled() -> None:
    """Verify detailed diagnostics are NOT emitted when NLP_TRACE_ENABLED=False."""
    engine = LexicalRetrievalEngine()
    engine.add_document(1, "Deep learning models require extensive training on GPUs.")
    engine.add_document(2, "Natural language processing models use attention mechanisms.")

    with patch("app.modules.retrieval.lexical.engine.logging_settings") as mock_settings, \
         patch("app.modules.retrieval.lexical.engine.logger") as mock_logger, \
         patch("app.modules.retrieval.lexical.engine.log_event") as mock_log_event:

        mock_settings.nlp_trace_enabled = False
        mock_logger.isEnabledFor.return_value = True

        results = engine.search("attention mechanisms models", top_k=5)

        assert len(results) > 0

        event_names = [call[0][2] for call in mock_log_event.call_args_list]
        assert "retrieval_query_diagnostics" not in event_names
        assert "retrieval_candidate_scoring_detail" not in event_names
        assert "retrieval_ranking_diagnostics" not in event_names


def test_lexical_search_results_unaltered_by_diagnostics() -> None:
    """Verify that enabling diagnostics does not alter scores, rankings, or tie-breaking."""
    engine = LexicalRetrievalEngine()
    engine.add_document(1, "Transformer models rely on self-attention.")
    engine.add_document(2, "Convolutional models process grid-structured data.")

    # Search with diagnostics off
    results_off = engine.search("Transformer models", top_k=2)

    # Search with diagnostics on
    with patch("app.modules.retrieval.lexical.engine.logging_settings", LoggingSettings(nlp_trace_enabled=True)):
        results_on = engine.search("Transformer models", top_k=2)

    assert results_off == results_on
