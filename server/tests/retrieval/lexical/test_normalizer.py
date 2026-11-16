"""Unit tests for lexical normalizer.

Tests Unicode NFC normalization, lowercasing, contraction expansion,
possessive handling, and preservation of ambiguous contractions.
"""

from __future__ import annotations

import unicodedata
import pytest

from app.modules.retrieval.lexical.normalizer import (
    CONTRACTION_MAP,
    expand_contractions_and_possessives,
    normalize_token,
    normalize_tokens,
)


class TestNormalizerCaseAndUnicode:
    """Verify case normalization, NFC normalization, and technical token integrity."""

    def test_case_normalization(self):
        assert normalize_token("BERT") == "bert"
        assert normalize_token("TF-IDF") == "tf-idf"
        assert normalize_token("GPT-4") == "gpt-4"
        assert normalize_token("State-of-the-Art") == "state-of-the-art"
        assert normalize_token("Word2Vec") == "word2vec"
        assert normalize_token("VGG-16") == "vgg-16"

    def test_unicode_nfc_normalization(self):
        # Decomposed e + acute accent
        decomposed = "e\u0301"
        assert normalize_token(decomposed) == "é"
        assert normalize_token("MÖBIUS") == "möbius"
        assert normalize_token("SCHRÖDINGER") == "schrödinger"

    def test_unicode_lowercasing_introduces_combining_mark_remains_nfc(self):
        # 'İ' (U+0130, Latin capital letter I with dot above)
        # Its lowercase conversion introduces combining dot above: 'i\u0307'
        token = "İ"
        result = normalize_token(token)
        # Verify that lowercasing introduced the combining mark U+0307
        assert "\u0307" in result
        # Verify that the final returned string is strictly NFC-normalized
        assert unicodedata.is_normalized("NFC", result)
        assert result == unicodedata.normalize("NFC", unicodedata.normalize("NFC", token.strip()).lower())

    def test_normalize_tokens_sequence(self):
        tokens = ["BERT", "GPT-4", "Self-Attention", "NAÏVE"]
        result = normalize_tokens(tokens)
        assert result == ["bert", "gpt-4", "self-attention", "naïve"]
        # Ensure new list is returned (immutability)
        assert result is not tokens


class TestContractionExpansion:
    """Verify expansion of recognized contractions with ASCII and typographic apostrophes."""

    @pytest.mark.parametrize(
        "token,expected",
        [
            ("don't", ["do", "not"]),
            ("doesn't", ["does", "not"]),
            ("can't", ["can", "not"]),
            ("won't", ["will", "not"]),
            ("it's", ["it", "is"]),
            ("didn't", ["did", "not"]),
            ("isn't", ["is", "not"]),
            ("aren't", ["are", "not"]),
            ("wasn't", ["was", "not"]),
            ("weren't", ["were", "not"]),
            ("haven't", ["have", "not"]),
            ("hasn't", ["has", "not"]),
            ("hadn't", ["had", "not"]),
            ("shouldn't", ["should", "not"]),
            ("mustn't", ["must", "not"]),
            ("they're", ["they", "are"]),
            ("we're", ["we", "are"]),
            ("you're", ["you", "are"]),
            ("they've", ["they", "have"]),
            ("we've", ["we", "have"]),
            ("they'll", ["they", "will"]),
            ("we'll", ["we", "will"]),
        ],
    )
    def test_ascii_contractions_expanded(self, token: str, expected: list[str]):
        assert expand_contractions_and_possessives([token]) == expected

    @pytest.mark.parametrize(
        "token,expected",
        [
            ("don’t", ["do", "not"]),
            ("doesn’t", ["does", "not"]),
            ("can’t", ["can", "not"]),
            ("won’t", ["will", "not"]),
            ("it’s", ["it", "is"]),
            ("they’re", ["they", "are"]),
            ("we’ve", ["we", "have"]),
        ],
    )
    def test_unicode_curly_apostrophe_contractions_expanded(self, token: str, expected: list[str]):
        assert expand_contractions_and_possessives([token]) == expected

    def test_uppercase_contractions_expanded(self):
        assert expand_contractions_and_possessives(["DON'T", "IT'S"]) == ["do", "not", "it", "is"]


class TestPossessivesAndAmbiguousContractions:
    """Verify possessives are stripped and ambiguous contractions are preserved."""

    def test_possessive_stripping(self):
        # Regression case: transformer’s -> ["transformer"]
        assert expand_contractions_and_possessives(["transformer’s"]) == ["transformer"]
        # ASCII variant
        assert expand_contractions_and_possessives(["transformer's"]) == ["transformer"]
        assert expand_contractions_and_possessives(["model's"]) == ["model"]
        assert expand_contractions_and_possessives(["attention's"]) == ["attention"]

    def test_trailing_possessive_apostrophe(self):
        # Regression case: authors' -> ["authors"]
        assert expand_contractions_and_possessives(["authors'"]) == ["authors"]
        # Typographic apostrophe variant
        assert expand_contractions_and_possessives(["authors’"]) == ["authors"]
        assert expand_contractions_and_possessives(["transformers'"]) == ["transformers"]

    def test_ambiguous_contractions_preserved_unchanged(self):
        # Regression case: he'd -> ["he'd"]
        assert expand_contractions_and_possessives(["he'd"]) == ["he'd"]
        assert expand_contractions_and_possessives(["they'd"]) == ["they'd"]
        assert expand_contractions_and_possessives(["she'd"]) == ["she'd"]
        assert expand_contractions_and_possessives(["we'd"]) == ["we'd"]

    def test_unapproved_contractions_absent_and_unchanged(self):
        # Unapproved contractions: ain't, mightn't, needn't (both ASCII and curly apostrophes)
        unapproved = ["ain't", "ain’t", "mightn't", "mightn’t", "needn't", "needn’t"]
        for word in unapproved:
            assert word not in CONTRACTION_MAP, f"{word} must not be in CONTRACTION_MAP"
            assert expand_contractions_and_possessives([word]) == [word], f"{word} should remain unchanged"

    def test_contraction_takes_priority_over_possessive_stripping(self):
        # "it's" must expand to ["it", "is"] and NOT be stripped to ["it"]
        assert expand_contractions_and_possessives(["it's"]) == ["it", "is"]
        assert expand_contractions_and_possessives(["it’s"]) == ["it", "is"]
        assert expand_contractions_and_possessives(["that's"]) == ["that", "is"]
        assert expand_contractions_and_possessives(["what's"]) == ["what", "is"]


class TestNormalizerInputValidationAndImmutability:
    """Verify input validation, error handling, and return types."""

    def test_normalize_token_validation(self):
        with pytest.raises(TypeError) as exc:
            normalize_token(None)  # type: ignore[arg-type]
        assert "token must be a str" in str(exc.value)

        with pytest.raises(TypeError):
            normalize_token(123)  # type: ignore[arg-type]

    def test_normalize_tokens_validation(self):
        with pytest.raises(TypeError) as exc:
            normalize_tokens("not a list")  # type: ignore[arg-type]
        assert "tokens must be a list or tuple" in str(exc.value)

        with pytest.raises(TypeError) as exc:
            normalize_tokens(["valid", None])  # type: ignore[list-item]
        assert "must be a str" in str(exc.value)

    def test_expand_contractions_validation(self):
        with pytest.raises(TypeError) as exc:
            expand_contractions_and_possessives(123)  # type: ignore[arg-type]
        assert "tokens must be a list or tuple" in str(exc.value)

        with pytest.raises(TypeError) as exc:
            expand_contractions_and_possessives(["valid", 456])  # type: ignore[list-item]
        assert "must be a str" in str(exc.value)

    def test_immutability_and_empty_list(self):
        empty: list[str] = []
        assert normalize_tokens(empty) == []
        assert expand_contractions_and_possessives(empty) == []

        original = ["Attention", "Transformer's", "BERT"]
        expanded = expand_contractions_and_possessives(original)
        assert expanded == ["attention", "transformer", "bert"]
        assert expanded is not original
        assert original == ["Attention", "Transformer's", "BERT"]  # Unmodified
