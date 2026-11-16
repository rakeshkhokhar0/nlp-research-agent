"""Unit tests for InvertedIndex in lexical retrieval."""

from unittest.mock import MagicMock
import pytest

from app.modules.retrieval.lexical import (
    InvertedIndex,
    TextPreprocessor,
)


class TestInvertedIndexAddAndSearch:
    """Tests for adding documents and searching single terms."""

    def test_add_single_document_and_search(self) -> None:
        """Adding a document allows searching for constituent terms."""
        index = InvertedIndex()
        index.add_document(1, "Attention is all you need.")
        # 'attent' and 'need' are indexed
        assert index.search("attention") == {1}
        assert index.search("need") == {1}
        assert index.search("attent") == {1}

    def test_add_multiple_documents(self) -> None:
        """Multiple documents are indexed and mapped to corresponding IDs."""
        index = InvertedIndex()
        index.add_document(1, "Transformer models rely on self-attention.")
        index.add_document(2, "Convolutional neural networks process image data.")
        index.add_document(3, "Language models utilize deep transformer architectures.")

        assert index.search("transformer") == {1, 3}
        assert index.search("model") == {1, 3}
        assert index.search("self-attention") == {1}
        assert index.search("network") == {2}
        assert index.search("image") == {2}
        assert len(index) == 3

    def test_term_occurring_in_multiple_documents(self) -> None:
        """Terms appearing across documents return all document IDs."""
        index = InvertedIndex()
        index.add_document(10, "Neural networks learn representations.")
        index.add_document(20, "Deep neural models optimize parameters.")
        index.add_document(30, "Graph neural networks model relationships.")

        results = index.search("neural")
        assert results == {10, 20, 30}

    def test_duplicate_occurrences_within_single_document(self) -> None:
        """Repeated terms within a document do not duplicate document IDs."""
        index = InvertedIndex()
        index.add_document(1, "attention attention attention model model")
        assert index.search("attention") == {1}
        assert index.search("model") == {1}
        assert index.document_terms[1] == {"attent", "model"}

    def test_search_normalization_and_stemming(self) -> None:
        """Search query terms are normalized, lowercased, and stemmed identically to documents."""
        index = InvertedIndex()
        index.add_document(1, "The transformer models were optimizing representations.")

        # Uppercase search
        assert index.search("TRANSFORMER") == {1}
        assert index.search("MODELS") == {1}
        # Inflected form stemming
        assert index.search("optimization") == {1}
        assert index.search("representation") == {1}

    def test_search_unknown_term_returns_empty_set(self) -> None:
        """Searching for a term not in the index returns an empty set."""
        index = InvertedIndex()
        index.add_document(1, "Natural language processing.")
        assert index.search("quantum") == set()
        assert index.search("superconductor") == set()

    def test_search_preprocessed_exact_lookup(self) -> None:
        """When preprocessed=True, performs an exact lookup on index posting keys."""
        index = InvertedIndex()
        index.add_document(1, "Word embeddings are dense representations.")
        # "embeddings" stems to "embed"
        assert "embed" in index.index
        assert index.search("embed", preprocessed=True) == {1}
        # A nonexistent term returns an empty set
        assert index.search("nonexistent", preprocessed=True) == set()

    def test_search_preprocessed_does_not_call_preprocessor(self) -> None:
        """Exact preprocessed lookup does not invoke the TextPreprocessor."""
        index = InvertedIndex()
        index.add_document(1, "Deep learning models.")

        spy = MagicMock(side_effect=index._preprocessor.preprocess)
        index._preprocessor.preprocess = spy

        result = index.search("learn", preprocessed=True)
        assert result == {1}
        assert spy.call_count == 0

    def test_search_raw_term_default_invokes_preprocessor(self) -> None:
        """Default raw-term lookup (preprocessed=False) invokes the preprocessor as before."""
        index = InvertedIndex()
        index.add_document(1, "Convolutional neural networks.")

        spy = MagicMock(side_effect=index._preprocessor.preprocess)
        index._preprocessor.preprocess = spy

        # preprocessed defaults to False
        assert index.search("CONVOLUTIONAL") == {1}
        assert spy.call_count == 1

        # Explicit preprocessed=False
        assert index.search("NETWORKS", preprocessed=False) == {1}
        assert spy.call_count == 2

    def test_search_returned_posting_set_is_independent_copy(self) -> None:
        """Modifying the returned posting set does not mutate the internal index structure."""
        index = InvertedIndex()
        index.add_document(1, "Transformer attention.")

        # With preprocessed=False
        postings_raw = index.search("transformer")
        assert postings_raw == {1}
        postings_raw.add(999)
        assert index.search("transformer") == {1}
        assert 999 not in index.index["transform"]

        # With preprocessed=True
        postings_prep = index.search("transform", preprocessed=True)
        assert postings_prep == {1}
        postings_prep.add(888)
        assert index.search("transform", preprocessed=True) == {1}
        assert 888 not in index.index["transform"]

    @pytest.mark.parametrize("invalid_flag", [1, 0, "True", "False", None, [True], 1.0])
    def test_search_invalid_preprocessed_type_raises_type_error(self, invalid_flag: object) -> None:
        """Passing non-boolean values for preprocessed raises TypeError."""
        index = InvertedIndex()
        with pytest.raises(TypeError, match="preprocessed must be a bool"):
            index.search("term", preprocessed=invalid_flag)  # type: ignore[arg-type]


class TestDocumentReplacementAndRemoval:
    """Tests for document updates, removals, and posting cleanup."""

    def test_replace_existing_document_removes_stale_terms(self) -> None:
        """Re-adding a document ID replaces its content and removes stale postings."""
        index = InvertedIndex()
        index.add_document(1, "Recurrent architectures process sequences.")
        assert index.search("recurrent") == {1}
        assert index.search("sequence") == {1}

        # Update document 1 with completely different text
        index.add_document(1, "Attention mechanisms utilize transformers.")
        assert index.search("attention") == {1}
        assert index.search("mechan") == {1}
        assert index.search("transformer") == {1}
        # Stale terms from previous content must be gone
        assert index.search("recurrent") == set()
        assert index.search("sequence") == set()
        assert "recurr" not in index.index
        assert "sequenc" not in index.index
        assert len(index) == 1

    def test_failed_replacement_preserves_existing_document_and_postings(self) -> None:
        """If preprocessing fails during replacement, existing document and postings remain intact."""

        class FailingPreprocessor(TextPreprocessor):
            def preprocess(self, text: str) -> list[str]:
                if "failing" in text:
                    raise RuntimeError("Simulated preprocessing failure during replacement")
                return super().preprocess(text)

        preprocessor = FailingPreprocessor()
        index = InvertedIndex(preprocessor=preprocessor)

        # 1. Add document ID 1 with valid text
        index.add_document(1, "Original valid text about transformers and attention.")

        # 2. Verify that its original terms are searchable
        assert index.search("transformer") == {1}
        assert index.search("attention") == {1}
        original_terms = set(index.document_terms[1])
        assert len(original_terms) > 0

        # 3 & 4. Attempt to replace document ID 1 with text that raises an exception during preprocessing
        with pytest.raises(RuntimeError, match="Simulated preprocessing failure during replacement"):
            index.add_document(1, "This is failing replacement text.")

        # 5 & 6. Verify that original document and its postings remain unchanged
        assert index.search("transformer") == {1}
        assert index.search("attention") == {1}
        assert 1 in index.document_terms
        assert index.document_terms[1] == original_terms
        for term in original_terms:
            assert 1 in index.index[term]
        assert len(index) == 1

    def test_remove_document(self) -> None:
        """Removing a document removes its ID from postings and document_terms."""
        index = InvertedIndex()
        index.add_document(1, "Deep learning models.")
        index.add_document(2, "Reinforcement learning agents.")

        assert index.search("learn") == {1, 2}
        assert index.search("model") == {1}

        index.remove_document(1)
        assert index.search("learn") == {2}
        assert index.search("model") == set()
        assert 1 not in index.document_terms
        assert 2 in index.document_terms
        assert len(index) == 1

    def test_remove_final_posting_deletes_term_from_index(self) -> None:
        """When the last document containing a term is removed, the term is deleted from index."""
        index = InvertedIndex()
        index.add_document(1, "unique_token_alpha")
        stemmed_term = list(index.document_terms[1])[0]
        assert stemmed_term in index.index

        index.remove_document(1)
        assert stemmed_term not in index.index
        assert index.search("unique_token_alpha") == set()
        assert len(index.index) == 0

    def test_remove_nonexistent_document_is_idempotent(self) -> None:
        """Removing a nonexistent document does not raise error and leaves index unchanged."""
        index = InvertedIndex()
        index.add_document(1, "Active document content.")
        index.remove_document(999)
        assert index.search("active") == {1}
        assert len(index) == 1


class TestDocumentFrequencyAndClear:
    """Tests for document frequency calculation and clear operations."""

    def test_get_document_frequency_known_and_unknown_terms(self) -> None:
        """get_document_frequency returns correct document count."""
        index = InvertedIndex()
        index.add_document(1, "Transformer architecture.")
        index.add_document(2, "Transformer attention mechanism.")
        index.add_document(3, "Convolutional neural network architecture.")

        assert index.get_document_frequency("transformer") == 2
        assert index.get_document_frequency("architecture") == 2
        assert index.get_document_frequency("attention") == 1
        assert index.get_document_frequency("network") == 1
        assert index.get_document_frequency("nonexistent") == 0

    def test_get_document_frequency_after_removal(self) -> None:
        """DF updates correctly when documents are removed."""
        index = InvertedIndex()
        index.add_document(1, "Supervised learning algorithms.")
        index.add_document(2, "Unsupervised learning methods.")
        assert index.get_document_frequency("learning") == 2

        index.remove_document(1)
        assert index.get_document_frequency("learning") == 1

        index.remove_document(2)
        assert index.get_document_frequency("learning") == 0

    def test_clear_operation(self) -> None:
        """clear() empties index and document_terms mappings."""
        index = InvertedIndex()
        index.add_document(1, "First paper chunk.")
        index.add_document(2, "Second paper chunk.")
        assert len(index) == 2
        assert len(index.index) > 0

        index.clear()
        assert len(index) == 0
        assert len(index.index) == 0
        assert len(index.document_terms) == 0
        assert index.search("paper") == set()
        assert index.get_document_frequency("paper") == 0


class TestEdgeCasesAndValidation:
    """Tests for empty inputs, multi-token queries, type validation, and immutability."""

    def test_empty_and_stopword_only_documents(self) -> None:
        """Empty or stopword-only documents index safely without creating postings."""
        index = InvertedIndex()
        index.add_document(1, "")
        index.add_document(2, "   \t\n  ")
        index.add_document(3, "The and or but in on at")

        assert index.document_terms[1] == set()
        assert index.document_terms[2] == set()
        assert index.document_terms[3] == set()
        assert len(index.index) == 0
        assert len(index) == 3

    def test_empty_and_stopword_search_query(self) -> None:
        """Empty or stopword-only queries produce 0 tokens and return empty set/0 DF."""
        index = InvertedIndex()
        index.add_document(1, "Language models.")

        assert index.search("") == set()
        assert index.search("   ") == set()
        assert index.search("the") == set()
        assert index.get_document_frequency("") == 0
        assert index.get_document_frequency("the") == 0

    def test_multi_token_search_raises_value_error(self) -> None:
        """Queries producing multiple tokens raise ValueError for search and DF."""
        index = InvertedIndex()
        index.add_document(1, "Attention mechanisms.")

        with pytest.raises(ValueError, match="Search query must produce at most one term"):
            index.search("attention mechanism")

        with pytest.raises(ValueError, match="Query must produce at most one term"):
            index.get_document_frequency("attention mechanism")

    @pytest.mark.parametrize(
        "invalid_id",
        [
            True,  # bool is subclass of int, must be rejected
            False,
            0,
            -1,
            -100,
            "1",
            1.5,
            None,
            [],
            {},
        ],
    )
    def test_invalid_document_id_validation(self, invalid_id: object) -> None:
        """Invalid document IDs raise TypeError or ValueError."""
        index = InvertedIndex()
        with pytest.raises((TypeError, ValueError)):
            index.add_document(invalid_id, "Valid text.")  # type: ignore[arg-type]

        with pytest.raises((TypeError, ValueError)):
            index.remove_document(invalid_id)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "invalid_text",
        [None, 123, 4.5, True, [], {}],
    )
    def test_invalid_text_and_term_types(self, invalid_text: object) -> None:
        """Non-string inputs raise TypeError."""
        index = InvertedIndex()
        with pytest.raises(TypeError, match="text must be a str"):
            index.add_document(1, invalid_text)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="term must be a str"):
            index.search(invalid_text)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="term must be a str"):
            index.get_document_frequency(invalid_text)  # type: ignore[arg-type]

    def test_fresh_set_behavior_does_not_mutate_index(self) -> None:
        """Modifying the returned set from search() does not mutate internal index."""
        index = InvertedIndex()
        index.add_document(1, "Scientific research paper.")
        results = index.search("research")
        assert results == {1}

        # Mutate the returned set
        results.add(999)
        results.discard(1)

        # Internal index must remain intact
        assert index.search("research") == {1}
        assert index.index["research"] == {1}

    def test_custom_preprocessor_injection(self) -> None:
        """Custom TextPreprocessor can be injected and used for indexing and search."""
        preprocessor = TextPreprocessor()
        index = InvertedIndex(preprocessor=preprocessor)
        index.add_document(1, "Transformer architecture.")
        assert index.search("transformer") == {1}

    def test_invalid_preprocessor_raises_type_error(self) -> None:
        """Passing non-TextPreprocessor instance raises TypeError."""
        with pytest.raises(TypeError, match="preprocessor must be an instance of TextPreprocessor"):
            InvertedIndex(preprocessor="invalid_preprocessor")  # type: ignore[arg-type]

    def test_mapping_consistency_across_complex_operations(self) -> None:
        """Internal mappings remain strictly consistent after adds, replaces, and removes."""
        index = InvertedIndex()
        # Add docs
        index.add_document(1, "Attention is all you need.")
        index.add_document(2, "BERT pre-training of deep bidirectional transformers.")
        index.add_document(3, "Attention and BERT models.")

        # Verify initial consistency
        for doc_id, terms in index.document_terms.items():
            for term in terms:
                assert doc_id in index.index[term]

        # Replace doc 2
        index.add_document(2, "GPT-4 and generative models.")
        for doc_id, terms in index.document_terms.items():
            for term in terms:
                assert doc_id in index.index[term]
        assert index.search("bidirectional") == set()

        # Remove doc 3
        index.remove_document(3)
        for doc_id, terms in index.document_terms.items():
            for term in terms:
                assert doc_id in index.index[term]

        # Verify no orphan terms with empty postings in index
        for term, postings in index.index.items():
            assert len(postings) > 0
            for doc_id in postings:
                assert doc_id in index.document_terms
                assert term in index.document_terms[doc_id]


class TestInvertedIndexTermFrequencies:
    """Tests for term frequency tracking, retrieval, and copy isolation."""

    def test_term_frequency_counts_in_added_document(self) -> None:
        """Term frequencies are counted accurately for preprocessed/stemmed terms."""
        index = InvertedIndex()
        index.add_document(1, "Neural networks learn representations. Deep neural networks learn.")

        freqs = index.get_document_term_frequencies(1)
        assert freqs["neural"] == 2
        assert freqs["network"] == 2
        assert freqs["learn"] == 2
        assert freqs["deep"] == 1
        assert freqs["represent"] == 1
        assert index.document_term_frequencies[1] == freqs

    def test_get_term_frequency_querying(self) -> None:
        """get_term_frequency correctly queries known, inflected, and unknown terms."""
        index = InvertedIndex()
        index.add_document(1, "Transformer models, transformer layers, and transformer attention.")

        assert index.get_term_frequency("transformer", 1) == 3
        assert index.get_term_frequency("TRANSFORMER", 1) == 3
        assert index.get_term_frequency("transformers", 1) == 3
        assert index.get_term_frequency("models", 1) == 1
        assert index.get_term_frequency("layer", 1) == 1
        assert index.get_term_frequency("attention", 1) == 1
        # Unknown term in indexed doc
        assert index.get_term_frequency("convolution", 1) == 0
        # Unknown document ID
        assert index.get_term_frequency("transformer", 999) == 0
        # Stopword query term produces 0
        assert index.get_term_frequency("the", 1) == 0
        assert index.get_term_frequency("", 1) == 0

    def test_get_term_frequency_multi_token_raises_value_error(self) -> None:
        """Passing multiple tokens to get_term_frequency raises ValueError."""
        index = InvertedIndex()
        index.add_document(1, "Neural network models.")
        with pytest.raises(ValueError, match="Query must produce at most one term"):
            index.get_term_frequency("neural networks", 1)

    @pytest.mark.parametrize("invalid_id", [True, False, 0, -1, -5, "1", 1.5, None])
    def test_get_term_frequency_invalid_document_id(self, invalid_id: object) -> None:
        """Invalid document_id raises TypeError or ValueError."""
        index = InvertedIndex()
        index.add_document(1, "Neural networks.")
        with pytest.raises((TypeError, ValueError)):
            index.get_term_frequency("neural", invalid_id)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_term", [None, 123, 4.5, True, ["term"]])
    def test_get_term_frequency_invalid_term_type(self, invalid_term: object) -> None:
        """Non-string term raises TypeError."""
        index = InvertedIndex()
        index.add_document(1, "Neural networks.")
        with pytest.raises(TypeError, match="term must be a str"):
            index.get_term_frequency(invalid_term, 1)  # type: ignore[arg-type]

    def test_get_document_term_frequencies_copy_isolation(self) -> None:
        """Mutating the dictionary returned by get_document_term_frequencies does not affect internal state."""
        index = InvertedIndex()
        index.add_document(1, "Neural networks.")

        freqs = index.get_document_term_frequencies(1)
        original_copy = dict(freqs)

        # Mutate the returned dictionary
        freqs["neural"] = 9999
        freqs["injected_term"] = 42

        # Verify internal mapping is unaffected
        assert index.document_term_frequencies[1] == original_copy
        assert index.get_document_term_frequencies(1) == original_copy
        assert index.get_term_frequency("neural", 1) == 1
        assert "injected_term" not in index.document_term_frequencies[1]

    def test_get_document_term_frequencies_unknown_and_empty_document(self) -> None:
        """Unknown document or empty/stopword-only document returns empty dictionary."""
        index = InvertedIndex()
        index.add_document(1, "The and of with.")  # stopword only

        assert index.get_document_term_frequencies(1) == {}
        assert index.get_document_term_frequencies(999) == {}

    @pytest.mark.parametrize("invalid_id", [True, False, 0, -1, "1", 2.5, None])
    def test_get_document_term_frequencies_invalid_id(self, invalid_id: object) -> None:
        """Invalid document_id raises TypeError or ValueError."""
        index = InvertedIndex()
        with pytest.raises((TypeError, ValueError)):
            index.get_document_term_frequencies(invalid_id)  # type: ignore[arg-type]

    def test_term_frequencies_updated_on_replacement(self) -> None:
        """Replacing a document replaces its term frequencies and removes stale terms."""
        index = InvertedIndex()
        index.add_document(1, "Recurrent recurrent architectures.")
        assert index.get_term_frequency("recurrent", 1) == 2
        assert index.get_term_frequency("transformer", 1) == 0

        # Replace document 1
        index.add_document(1, "Transformer transformer transformer models.")
        assert index.get_term_frequency("transformer", 1) == 3
        assert index.get_term_frequency("model", 1) == 1
        assert index.get_term_frequency("recurrent", 1) == 0
        assert "recurr" not in index.document_term_frequencies[1]

    def test_term_frequencies_removed_on_document_removal(self) -> None:
        """Removing a document cleans up document_term_frequencies completely."""
        index = InvertedIndex()
        index.add_document(1, "Attention mechanisms.")
        index.add_document(2, "Convolutional kernels.")

        assert 1 in index.document_term_frequencies
        index.remove_document(1)
        assert 1 not in index.document_term_frequencies
        assert 2 in index.document_term_frequencies
        assert index.get_term_frequency("attention", 1) == 0

    def test_term_frequencies_cleared_on_clear(self) -> None:
        """clear() removes all document_term_frequencies entries."""
        index = InvertedIndex()
        index.add_document(1, "Deep learning.")
        index.add_document(2, "Reinforcement learning.")

        assert len(index.document_term_frequencies) == 2
        index.clear()
        assert len(index.document_term_frequencies) == 0
        assert index.get_document_term_frequencies(1) == {}

    def test_preprocessor_property(self) -> None:
        """preprocessor property returns the configured TextPreprocessor instance."""
        custom_preprocessor = TextPreprocessor()
        index = InvertedIndex(preprocessor=custom_preprocessor)
        assert index.preprocessor is custom_preprocessor

        default_index = InvertedIndex()
        assert isinstance(default_index.preprocessor, TextPreprocessor)
