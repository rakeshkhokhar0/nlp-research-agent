"""Unit tests for lexical tokenizer.

Tests Unicode-aware tokenization, technical term preservation, hyphenated
expressions, contractions, possessives, and punctuation delimitation.
"""

from __future__ import annotations

import unicodedata
import pytest

from app.modules.retrieval.lexical.tokenizer import Tokenizer, tokenize


class TestTokenizerTechnicalTerms:
    """Verify that technical terms, architectures, and compound expressions are preserved."""

    def test_hyphenated_technical_terms(self):
        text = (
            "The state-of-the-art model uses self-attention, TF-IDF, "
            "and cross-entropy loss for zero-shot word-level evaluation."
        )
        tokens = tokenize(text)
        assert "state-of-the-art" in tokens
        assert "self-attention" in tokens
        assert "TF-IDF" in tokens
        assert "cross-entropy" in tokens
        assert "zero-shot" in tokens
        assert "word-level" in tokens

    def test_alphanumeric_technical_models(self):
        text = "We compare BERT, GPT-4, Word2Vec, VGG-16, and ResNet50 against CNN and RNN baselines."
        tokens = tokenize(text)
        expected = [
            "We",
            "compare",
            "BERT",
            "GPT-4",
            "Word2Vec",
            "VGG-16",
            "and",
            "ResNet50",
            "against",
            "CNN",
            "and",
            "RNN",
            "baselines",
        ]
        assert tokens == expected

    def test_numbers_and_dimensions(self):
        text = "In 2024, our model processed 100k samples in 3D latent space."
        tokens = tokenize(text)
        assert "2024" in tokens
        assert "100k" in tokens
        assert "3D" in tokens


class TestTokenizerApostrophesAndContractions:
    """Verify contractions and possessives are captured intact for downstream normalization."""

    def test_ascii_contractions(self):
        text = "The model don't fail, doesn't diverge, won't crash, and isn't slow."
        tokens = tokenize(text)
        assert "don't" in tokens
        assert "doesn't" in tokens
        assert "won't" in tokens
        assert "isn't" in tokens

    def test_unicode_curly_apostrophe_contractions(self):
        # Using U+2019 typographic apostrophe
        text = "It’s clear that they’re not able to train if it doesn’t converge."
        tokens = tokenize(text)
        assert "It’s" in tokens
        assert "they’re" in tokens
        assert "doesn’t" in tokens

    def test_possessive_apostrophes(self):
        text = "The transformer's encoder and the attention’s weights reflect the authors' intent."
        tokens = tokenize(text)
        assert "transformer's" in tokens
        assert "attention’s" in tokens
        assert "authors'" in tokens


class TestTokenizerPunctuationAndDelimiters:
    """Verify punctuation acts strictly as delimiter and never appears as tokens."""

    def test_punctuation_is_delimited_and_discarded(self):
        text = "Attention (Vaswani et al., 2017): Is it all you need? Yes! [NeurIPS]."
        tokens = tokenize(text)
        assert "(" not in tokens
        assert ")" not in tokens
        assert ":" not in tokens
        assert "," not in tokens
        assert "." not in tokens
        assert "?" not in tokens
        assert "!" not in tokens
        assert "[" not in tokens
        assert "]" not in tokens
        assert tokens == [
            "Attention",
            "Vaswani",
            "et",
            "al",
            "2017",
            "Is",
            "it",
            "all",
            "you",
            "need",
            "Yes",
            "NeurIPS",
        ]

    def test_outer_hyphens_and_dashes_stripped(self):
        text = "Pre-processing and --eval-flag with em—dash and -leading trailing- hyphens."
        tokens = tokenize(text)
        # Internal hyphens preserved
        assert "Pre-processing" in tokens
        assert "eval-flag" in tokens
        # Standalone or boundary hyphens stripped
        assert "--" not in tokens
        assert "-" not in tokens
        assert "—" not in tokens
        assert "leading" in tokens
        assert "trailing" in tokens

    def test_punctuation_only_input_returns_empty_list(self):
        assert tokenize("... ,,, --- !!! ??? () [] {} :; ' '") == []


class TestTokenizerUnicode:
    """Verify Unicode awareness and pre-tokenization NFC normalization."""

    def test_unicode_accented_words(self):
        text = "Naïve Bayes, Möbius transformation, Schrödinger equation, and László Gärdenfors."
        tokens = tokenize(text)
        assert "Naïve" in tokens
        assert "Möbius" in tokens
        assert "Schrödinger" in tokens
        assert "László" in tokens
        assert "Gärdenfors" in tokens

    def test_unicode_combining_characters_composed_before_scanning(self):
        # Decomposed 'e' + combining acute accent (\u0301)
        decomposed = "e\u0301cole de recherche"
        # Ensure input really is decomposed before passing to tokenize
        assert "e\u0301" in decomposed

        tokens = tokenize(decomposed)
        # Should be composed into single precomposed character 'école'
        composed_word = unicodedata.normalize("NFC", decomposed).split()[0]
        assert tokens[0] == composed_word
        assert tokens == ["école", "de", "recherche"]

    def test_greek_and_foreign_scripts(self):
        text = "Parameters α and β in the latent Gaussian distribution."
        tokens = tokenize(text)
        assert "α" in tokens
        assert "β" in tokens


class TestTokenizerInputValidationAndClassInterface:
    """Verify input validation, empty inputs, determinism, and class interface."""

    def test_empty_and_whitespace_input(self):
        assert tokenize("") == []
        assert tokenize(" ") == []
        assert tokenize("   \t  \n  \r\n  ") == []

    def test_type_error_on_non_string_input(self):
        with pytest.raises(TypeError) as exc:
            tokenize(None)  # type: ignore[arg-type]
        assert "text must be a str" in str(exc.value)

        with pytest.raises(TypeError):
            tokenize(12345)  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            tokenize(["not", "a", "string"])  # type: ignore[arg-type]

    def test_deterministic_output(self):
        text = "Deterministic tokenization: BERT, GPT-4, and self-attention."
        result1 = tokenize(text)
        result2 = tokenize(text)
        assert result1 == result2
        assert result1 is not result2  # Fresh list

    def test_tokenizer_class_instance(self):
        tok = Tokenizer()
        text = "Testing the Tokenizer class instance."
        assert tok(text) == tokenize(text)
        assert tok.tokenize(text) == tokenize(text)
