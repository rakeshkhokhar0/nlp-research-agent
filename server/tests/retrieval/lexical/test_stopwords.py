"""Unit tests for lexical stopword filtering.

Tests conservative academic English stopword collection, preservation of negations
and academic terminology, custom stopword extensions, Unicode normalization,
and input validation.
"""

from __future__ import annotations

import unicodedata
import pytest

from app.modules.retrieval.lexical.stopwords import (
    DEFAULT_STOPWORDS,
    StopwordFilter,
    remove_stopwords,
)


class TestStopwordFilterCore:
    """Verify standard filtering behavior and explicit regression requirements."""

    def test_regression_the_model_is_not_learning(self):
        filterer = StopwordFilter()
        assert filterer.filter_tokens(
            ["the", "model", "is", "not", "learning"]
        ) == ["model", "not", "learning"]

    def test_all_five_negations_survive(self):
        filterer = StopwordFilter()
        negations = ["not", "no", "nor", "neither", "never"]
        # None should be in DEFAULT_STOPWORDS
        for neg in negations:
            assert neg not in DEFAULT_STOPWORDS, f"Negation '{neg}' must not be in DEFAULT_STOPWORDS"
            assert not filterer.is_stopword(neg)

        # All must survive filtering
        assert filterer.filter_tokens(negations) == negations

    def test_academic_and_technical_terms_survive(self):
        filterer = StopwordFilter()
        academic_terms = [
            "model",
            "method",
            "approach",
            "network",
            "learning",
            "algorithm",
            "data",
            "system",
            "research",
            "attention",
            "transformer",
            "embedding",
            "retrieval",
            "classification",
            "generation",
        ]
        for term in academic_terms:
            assert term not in DEFAULT_STOPWORDS, f"Academic term '{term}' must not be in DEFAULT_STOPWORDS"
            assert not filterer.is_stopword(term)

        assert filterer.filter_tokens(academic_terms) == academic_terms

    def test_common_function_words_are_removed(self):
        filterer = StopwordFilter()
        function_words = [
            "the", "is", "are", "was", "were", "of", "in", "on", "at",
            "to", "for", "and", "or", "but", "with", "from", "by", "as", "an", "a",
        ]
        for word in function_words:
            assert word in DEFAULT_STOPWORDS, f"Function word '{word}' should be in DEFAULT_STOPWORDS"
            assert filterer.is_stopword(word)

        assert filterer.filter_tokens(function_words) == []

    def test_case_insensitive_behavior(self):
        filterer = StopwordFilter()
        # Non-stopwords retain their original casing; stopwords are filtered case-insensitively
        input_tokens = ["THE", "Model", "IS", "NOT", "Learning"]
        result = filterer.filter_tokens(input_tokens)
        assert result == ["Model", "NOT", "Learning"]


class TestStopwordFilterCustomization:
    """Verify custom stopwords merging, Unicode normalization, and isolation."""

    def test_custom_stopwords_filtering(self):
        filterer = StopwordFilter(custom_stopwords={"customword"})
        assert filterer.filter_tokens(
            ["the", "customword", "model"]
        ) == ["model"]

    def test_custom_stopwords_sequence_types(self):
        filterer_list = StopwordFilter(custom_stopwords=["foo", "bar"])
        assert filterer_list.filter_tokens(["foo", "model", "bar"]) == ["model"]

    def test_custom_stopwords_do_not_modify_default_stopwords(self):
        initial_default_len = len(DEFAULT_STOPWORDS)
        filterer = StopwordFilter(custom_stopwords={"customterm123"})
        assert "customterm123" in filterer.stopwords
        assert "customterm123" not in DEFAULT_STOPWORDS
        assert len(DEFAULT_STOPWORDS) == initial_default_len

    def test_custom_stopwords_unicode_nfc_normalization(self):
        # Decomposed custom stopword 'e' + combining acute accent (\u0301)
        decomposed = "e\u0301cole"
        filterer = StopwordFilter(custom_stopwords={decomposed})

        # Composed input token: 'école'
        composed = unicodedata.normalize("NFC", decomposed)
        assert filterer.is_stopword(composed)
        assert filterer.filter_tokens(["the", composed, "model"]) == ["model"]


class TestStopwordFilterValidationAndImmutability:
    """Verify input validation, immutability, order preservation, and convenience functions."""

    def test_empty_input(self):
        filterer = StopwordFilter()
        assert filterer.filter_tokens([]) == []

    def test_order_preservation(self):
        filterer = StopwordFilter()
        tokens = ["transformer", "self-attention", "model", "not", "cnn"]
        assert filterer.filter_tokens(tokens) == tokens

    def test_input_immutability(self):
        filterer = StopwordFilter()
        tokens = ["the", "model", "is", "attention"]
        original = list(tokens)
        result = filterer.filter_tokens(tokens)
        assert result == ["model", "attention"]
        assert result is not tokens
        assert tokens == original

    def test_invalid_tokens_input_raises_type_error(self):
        filterer = StopwordFilter()
        with pytest.raises(TypeError) as exc:
            filterer.filter_tokens("not a list")  # type: ignore[arg-type]
        assert "tokens must be a list or tuple" in str(exc.value)

        with pytest.raises(TypeError) as exc:
            filterer.filter_tokens(["valid", 123])  # type: ignore[list-item]
        assert "must be a str" in str(exc.value)

        with pytest.raises(TypeError):
            filterer.filter_tokens(["valid", None])  # type: ignore[list-item]

    def test_invalid_custom_stopwords_raises_type_error(self):
        with pytest.raises(TypeError) as exc:
            StopwordFilter(custom_stopwords=123)  # type: ignore[arg-type]
        assert "custom_stopwords must be a set" in str(exc.value)

        with pytest.raises(TypeError) as exc:
            StopwordFilter(custom_stopwords={"valid", 456})  # type: ignore[arg-type]
        assert "custom_stopwords items must be str" in str(exc.value)

    def test_is_stopword_validation(self):
        filterer = StopwordFilter()
        with pytest.raises(TypeError) as exc:
            filterer.is_stopword(None)  # type: ignore[arg-type]
        assert "token must be a str" in str(exc.value)

    def test_remove_stopwords_convenience_function(self):
        result = remove_stopwords(["the", "model", "is", "active"])
        assert result == ["model", "active"]

        result_custom = remove_stopwords(["the", "custom", "model"], custom_stopwords={"custom"})
        assert result_custom == ["model"]
