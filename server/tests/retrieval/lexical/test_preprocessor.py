"""Unit tests for the lexical TextPreprocessor pipeline."""

import pytest

from app.modules.retrieval.lexical import (
    PorterStemmer,
    StopwordFilter,
    TextPreprocessor,
    Tokenizer,
    preprocess,
)


class TestBasicPipeline:
    """Tests for basic English sentences, academic text, and punctuation handling."""

    def test_normal_english_sentence(self) -> None:
        """Standard sentence is tokenized, normalized, stopword-filtered, and stemmed."""
        preprocessor = TextPreprocessor()
        text = "The quick brown fox jumps over the lazy dog."
        # 'the', 'over' are stopwords.
        # 'quick', 'brown', 'fox', 'jumps' -> 'jump', 'lazy' -> 'lazi', 'dog'.
        tokens = preprocessor.preprocess(text)
        assert tokens == ["quick", "brown", "fox", "jump", "lazi", "dog"]

    def test_academic_research_text(self) -> None:
        """Academic text retains research keywords while filtering stopwords."""
        preprocessor = TextPreprocessor()
        text = "We propose a novel attention-based neural network architecture for language modeling."
        # 'we', 'a', 'for' are stopwords.
        # 'attention-based' is preserved as hyphenated compound term.
        # 'neural', 'network', 'architecture' -> 'architectur', 'language' -> 'languag', 'modeling' -> 'model'.
        tokens = preprocessor.preprocess(text)
        assert "network" in tokens
        assert "attention-based" in tokens
        assert "propos" in tokens
        assert "novel" in tokens
        assert "we" not in tokens
        assert "a" not in tokens
        assert "for" not in tokens

    def test_text_containing_punctuation(self) -> None:
        """Punctuation acts strictly as delimiters and is never emitted as tokens."""
        preprocessor = TextPreprocessor()
        text = "Attention (specifically, multi-head self-attention): is key; right?!"
        tokens = preprocessor.preprocess(text)
        assert "multi-head" in tokens
        assert "self-attention" in tokens
        assert "(" not in tokens
        assert ")" not in tokens
        assert ":" not in tokens
        assert ";" not in tokens
        assert "!" not in tokens
        assert "?" not in tokens

    def test_uppercase_words(self) -> None:
        """Standard uppercase words are lowercased and stemmed appropriately."""
        preprocessor = TextPreprocessor()
        text = "ATTENTION IS ALL YOU NEED"
        # 'is', 'all', 'you' are stopwords.
        tokens = preprocessor.preprocess(text)
        assert tokens == ["attent", "need"]

    def test_hyphenated_technical_terms(self) -> None:
        """Compound technical terms with internal hyphens are preserved unstemmed."""
        preprocessor = TextPreprocessor()
        text = "Our state-of-the-art model uses cross-entropy loss in zero-shot classification."
        tokens = preprocessor.preprocess(text)
        assert "state-of-the-art" in tokens
        assert "cross-entropy" in tokens
        assert "zero-shot" in tokens
        assert "model" in tokens


class TestContractionHandling:
    """Tests for contraction expansion and negation retention."""

    def test_approved_negation_regression_expanded_vs_contracted(self) -> None:
        """Expanded and contracted negation sentences yield identical output."""
        preprocessor = TextPreprocessor()
        expanded_text = "The model does not support attention"
        contracted_text = "The model doesn't support attention"

        expanded_tokens = preprocessor.preprocess(expanded_text)
        contracted_tokens = preprocessor.preprocess(contracted_text)

        # Expected output from approved plan:
        # 'the' (stopword) -> removed
        # 'model' (preserved academic keyword) -> 'model'
        # 'does' (auxiliary stopword) -> removed
        # 'not' (negation) -> preserved!
        # 'support' -> 'support'
        # 'attention' -> 'attent'
        expected = ["model", "not", "support", "attent"]
        assert expanded_tokens == expected
        assert contracted_tokens == expected

    def test_various_contractions_expanded_and_filtered(self) -> None:
        """Common contractions are expanded, filtering auxiliaries while preserving negations."""
        preprocessor = TextPreprocessor()
        text = "We don't think it's working and won't believe they aren't."
        tokens = preprocessor.preprocess(text)
        # don't -> do (stopword), not (preserved)
        # it's -> it (stopword), is (stopword)
        # won't -> will (stopword), not (preserved)
        # aren't -> are (stopword), not (preserved)
        assert "not" in tokens
        assert tokens.count("not") == 3
        assert "do" not in tokens
        assert "is" not in tokens
        assert "will" not in tokens
        assert "are" not in tokens

    def test_curly_apostrophe_contractions(self) -> None:
        """Typographic apostrophe contractions are expanded identically to ASCII."""
        preprocessor = TextPreprocessor()
        ascii_text = "The model doesn't fail."
        curly_text = "The model doesn’t fail."
        assert preprocessor.preprocess(ascii_text) == preprocessor.preprocess(curly_text)

    def test_ambiguous_contractions_preserved(self) -> None:
        """Ambiguous contractions such as he'd and they'd are kept intact."""
        preprocessor = TextPreprocessor()
        tokens = preprocessor.preprocess("He'd said they'd come.")
        assert "he'd" in tokens
        assert "they'd" in tokens


class TestStopwordsAndStemmingIntegration:
    """Tests for stopword filtering followed by Porter stemming."""

    def test_academic_keywords_preserved_and_stemmed(self) -> None:
        """Academic terms are preserved from stopword removal and stemmed consistently."""
        preprocessor = TextPreprocessor()
        text = "The models, methods, approaches, networks, algorithms, and data systems."
        tokens = preprocessor.preprocess(text)
        assert tokens == [
            "model",
            "method",
            "approach",
            "network",
            "algorithm",
            "data",
            "system",
        ]

    def test_negations_never_removed_by_default(self) -> None:
        """Negations ('not', 'no', 'nor', 'neither', 'never') are retained in output."""
        preprocessor = TextPreprocessor()
        text = "There is no recurrence, neither convolutional nor recurrent."
        tokens = preprocessor.preprocess(text)
        assert "no" in tokens
        assert "neither" in tokens
        assert "nor" in tokens

    def test_possessive_chunk_and_query_matching(self) -> None:
        """Chunk possessives are stripped so query terms match chunk tokens."""
        preprocessor = TextPreprocessor()
        chunk = "The transformer's self-attention mechanism"
        query = "transformer self-attention"

        chunk_tokens = preprocessor.preprocess(chunk)
        query_tokens = preprocessor.preprocess(query)

        assert chunk_tokens == ["transform", "self-attention", "mechan"]
        assert query_tokens == ["transform", "self-attention"]
        # Verify query terms are represented in chunk tokens
        assert set(query_tokens).issubset(set(chunk_tokens))

    def test_plural_possessives(self) -> None:
        """Plural possessives ending with trailing apostrophes are stripped."""
        preprocessor = TextPreprocessor()
        tokens = preprocessor.preprocess("The authors' conclusions")
        assert tokens == ["author", "conclus"]


class TestTechnicalTermsAndGuards:
    """Tests for technical terminology, acronyms, digits, and guards."""

    def test_technical_acronyms_preserved(self) -> None:
        """Technical exceptions (BERT, CNN, RNN, LSTM, MLP) bypass stemming."""
        preprocessor = TextPreprocessor()
        text = "BERT and CNN models outperform LSTM and RNN baselines."
        tokens = preprocessor.preprocess(text)
        assert "bert" in tokens
        assert "cnn" in tokens
        assert "lstm" in tokens
        assert "rnn" in tokens
        assert "model" in tokens
        assert "outperform" in tokens

    def test_alphanumeric_and_digit_models(self) -> None:
        """Tokens containing digits (GPT-4, Word2Vec, 2024, 3D, V100) bypass stemming."""
        preprocessor = TextPreprocessor()
        text = "GPT-4 and Word2Vec were evaluated on V100 GPUs in 2024."
        tokens = preprocessor.preprocess(text)
        assert "gpt-4" in tokens
        assert "word2vec" in tokens
        assert "v100" in tokens
        assert "gpu" in tokens  # GPU is in technical exceptions
        assert "2024" in tokens


class TestEdgeCasesAndValidation:
    """Tests for empty strings, whitespace, invalid inputs, and immutability."""

    def test_empty_string(self) -> None:
        """Empty input returns an empty list."""
        preprocessor = TextPreprocessor()
        assert preprocessor.preprocess("") == []

    def test_whitespace_only_string(self) -> None:
        """Whitespace-only input returns an empty list."""
        preprocessor = TextPreprocessor()
        assert preprocessor.preprocess("   \t  \n \r\n  ") == []

    def test_only_stopwords_string(self) -> None:
        """Input containing only function words returns an empty list."""
        preprocessor = TextPreprocessor()
        text = "The and or but in on at with by to of"
        assert preprocessor.preprocess(text) == []

    def test_only_punctuation_string(self) -> None:
        """Input containing only punctuation delimiters returns an empty list."""
        preprocessor = TextPreprocessor()
        assert preprocessor.preprocess("... --- !!! ??? ,,, ;;; ::: () [] {}") == []

    @pytest.mark.parametrize("invalid_input", [None, 123, 45.6, [], {}, b"bytes"])
    def test_type_error_on_invalid_input(self, invalid_input: object) -> None:
        """Passing non-string input raises TypeError."""
        preprocessor = TextPreprocessor()
        with pytest.raises(TypeError, match="text must be a str"):
            preprocessor.preprocess(invalid_input)  # type: ignore[arg-type]

    def test_repeated_calls_deterministic(self) -> None:
        """Calling preprocess repeatedly produces identical output."""
        preprocessor = TextPreprocessor()
        text = "Deep neural networks learn hierarchical representations."
        first = preprocessor.preprocess(text)
        second = preprocessor.preprocess(text)
        third = preprocessor.preprocess(text)
        assert first == second == third

    def test_fresh_list_returned_every_call(self) -> None:
        """Every call returns a distinct list instance, preventing external mutation."""
        preprocessor = TextPreprocessor()
        text = "Attention mechanisms improve accuracy."
        first = preprocessor.preprocess(text)
        second = preprocessor.preprocess(text)
        assert first == second
        assert first is not second

    def test_callable_interface(self) -> None:
        """TextPreprocessor instance is callable and delegates to preprocess()."""
        preprocessor = TextPreprocessor()
        text = "Self-attention replaces recurrence in sequence transduction."
        assert preprocessor(text) == preprocessor.preprocess(text)

    def test_module_convenience_function(self) -> None:
        """Module-level preprocess() matches TextPreprocessor instance output."""
        text = "Transformer architectures revolutionize natural language processing."
        preprocessor = TextPreprocessor()
        assert preprocess(text) == preprocessor.preprocess(text)


class TestCustomComponentInjection:
    """Tests for passing custom components to TextPreprocessor."""

    def test_custom_stopword_filter(self) -> None:
        """Custom StopwordFilter can be injected into TextPreprocessor."""
        custom_stopwords = StopwordFilter(custom_stopwords={"novel", "architecture"})
        preprocessor = TextPreprocessor(stopword_filter=custom_stopwords)
        text = "We propose a novel architecture."
        tokens = preprocessor.preprocess(text)
        # 'novel' and 'architecture' should be filtered out by custom filter
        assert "novel" not in tokens
        assert "architectur" not in tokens
        assert tokens == ["propos"]

    def test_custom_stemmer_with_additional_exceptions(self) -> None:
        """Custom PorterStemmer can be injected with custom exceptions."""
        custom_stemmer = PorterStemmer(technical_exceptions={"mamba", "rwkv"})
        preprocessor = TextPreprocessor(stemmer=custom_stemmer)
        text = "Mamba and RWKV models process sequential data."
        tokens = preprocessor.preprocess(text)
        assert "mamba" in tokens
        assert "rwkv" in tokens

    def test_invalid_component_type_error(self) -> None:
        """Passing invalid component instances raises TypeError."""
        with pytest.raises(TypeError, match="tokenizer must be an instance of Tokenizer"):
            TextPreprocessor(tokenizer="not_a_tokenizer")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="stopword_filter must be an instance of StopwordFilter"):
            TextPreprocessor(stopword_filter="not_a_filter")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="stemmer must be an instance of PorterStemmer"):
            TextPreprocessor(stemmer="not_a_stemmer")  # type: ignore[arg-type]
