"""Unit tests for LexicalRetrievalEngine initialization, index building, and search."""

from unittest.mock import MagicMock
import pytest

from app.modules.retrieval.lexical import (
    CosineSimilarity,
    DocumentRanker,
    InvertedIndex,
    LexicalRetrievalEngine,
    TextPreprocessor,
    TfidfCalculator,
)


class TestLexicalRetrievalEngineInitialization:
    """Tests for engine instantiation and component wiring."""

    def test_constructor_initializes_all_five_components(self) -> None:
        """Engine constructor instantiates and exposes all five core components."""
        engine = LexicalRetrievalEngine()
        assert isinstance(engine.preprocessor, TextPreprocessor)
        assert isinstance(engine.index, InvertedIndex)
        assert isinstance(engine.tfidf, TfidfCalculator)
        assert isinstance(engine.similarity, CosineSimilarity)
        assert isinstance(engine.ranker, DocumentRanker)

    def test_shared_preprocessor_instance(self) -> None:
        """The preprocessor instance is shared directly with the inverted index."""
        engine = LexicalRetrievalEngine()
        assert engine.index.preprocessor is engine.preprocessor

    def test_tfidf_calculator_references_engine_index(self) -> None:
        """The TF-IDF calculator references the exact same inverted index instance."""
        engine = LexicalRetrievalEngine()
        assert engine.tfidf.index is engine.index

    def test_custom_preprocessor_injection(self) -> None:
        """A custom TextPreprocessor instance can be injected and wired to components."""
        custom_preprocessor = TextPreprocessor()
        engine = LexicalRetrievalEngine(preprocessor=custom_preprocessor)
        assert engine.preprocessor is custom_preprocessor
        assert engine.index.preprocessor is custom_preprocessor

    def test_invalid_preprocessor_raises_type_error(self) -> None:
        """Passing non-TextPreprocessor instance raises TypeError."""
        with pytest.raises(TypeError, match="preprocessor must be an instance of TextPreprocessor"):
            LexicalRetrievalEngine(preprocessor="invalid")  # type: ignore[arg-type]


class TestLexicalRetrievalEngineIndexBuilding:
    """Tests for build_index, add, remove, and clear operations."""

    def test_build_index_with_multiple_documents(self) -> None:
        """build_index populates inverted index, vocabulary terms, and term frequencies."""
        engine = LexicalRetrievalEngine()
        docs = {
            1: "Attention mechanisms in transformer architectures.",
            2: "BERT pre-training of deep bidirectional transformers.",
            3: "Convolutional neural networks for image classification.",
        }
        engine.build_index(docs)

        assert len(engine) == 3
        assert engine.index.search("transformer") == {1, 2}
        assert engine.index.search("attention") == {1}
        assert engine.index.search("bert") == {2}
        assert engine.index.search("convolutional") == {3}

        # Check term frequencies
        freqs_1 = engine.index.get_document_term_frequencies(1)
        assert freqs_1["attent"] == 1
        assert freqs_1["transform"] == 1

    def test_build_empty_index(self) -> None:
        """Building an index with an empty dictionary results in an empty index."""
        engine = LexicalRetrievalEngine()
        engine.build_index({})
        assert len(engine) == 0
        assert len(engine.index) == 0

    def test_rebuilding_replaces_old_documents(self) -> None:
        """Calling build_index again replaces previous contents completely without stale terms."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Recurrent neural architectures process sequential data.",
            2: "Long short-term memory recurrent units.",
        })
        assert engine.index.search("recurrent") == {1, 2}
        assert len(engine) == 2

        # Rebuild with disjoint documents
        engine.build_index({
            3: "Transformer attention mechanism utilizes multi-head layers.",
            4: "Generative pre-trained transformer models.",
        })
        assert len(engine) == 2
        # Old documents must be gone
        assert 1 not in engine.index.document_terms
        assert 2 not in engine.index.document_terms
        assert engine.index.search("recurrent") == set()
        assert "recurr" not in engine.index.index
        # New documents must be indexed
        assert engine.index.search("transformer") == {3, 4}
        assert engine.index.search("generative") == {4}

    @pytest.mark.parametrize("invalid_docs", [None, [1, 2], "not_dict", 123, True])
    def test_invalid_documents_container_type_rejected(self, invalid_docs: object) -> None:
        """Non-dict inputs to build_index raise TypeError."""
        engine = LexicalRetrievalEngine()
        with pytest.raises(TypeError, match="documents must be a dict"):
            engine.build_index(invalid_docs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_id", [True, False, 1.5, "1", None, ("tuple",)])
    def test_invalid_document_id_types_rejected(self, invalid_id: object) -> None:
        """Non-integer or boolean document IDs raise TypeError."""
        engine = LexicalRetrievalEngine()
        with pytest.raises(TypeError, match="Document ID must be an int"):
            engine.build_index({invalid_id: "Valid text."})  # type: ignore[dict-item]

    @pytest.mark.parametrize("non_positive_id", [0, -1, -50])
    def test_non_positive_document_ids_rejected(self, non_positive_id: int) -> None:
        """Zero or negative document IDs raise ValueError."""
        engine = LexicalRetrievalEngine()
        with pytest.raises(ValueError, match="Document ID must be a positive integer"):
            engine.build_index({non_positive_id: "Valid text."})

    @pytest.mark.parametrize("invalid_text", [123, 4.5, None, True, ["text"], {"k": "v"}])
    def test_invalid_document_text_types_rejected(self, invalid_text: object) -> None:
        """Non-string document text values raise TypeError."""
        engine = LexicalRetrievalEngine()
        with pytest.raises(TypeError, match="Document text for ID .* must be a str"):
            engine.build_index({1: invalid_text})  # type: ignore[dict-item]

    def test_add_document(self) -> None:
        """add_document delegates to inverted index and indexes document."""
        engine = LexicalRetrievalEngine()
        engine.add_document(1, "Neural network representation learning.")
        assert len(engine) == 1
        assert engine.index.search("neural") == {1}
        assert engine.index.search("network") == {1}

    def test_replace_existing_document(self) -> None:
        """Adding a document with an existing ID replaces it and clears stale terms."""
        engine = LexicalRetrievalEngine()
        engine.add_document(1, "Recurrent networks.")
        assert engine.index.search("recurrent") == {1}

        engine.add_document(1, "Transformer attention.")
        assert engine.index.search("recurrent") == set()
        assert engine.index.search("transformer") == {1}
        assert len(engine) == 1

    def test_remove_document(self) -> None:
        """remove_document delegates to inverted index and removes document."""
        engine = LexicalRetrievalEngine()
        engine.add_document(1, "Transformer models.")
        engine.add_document(2, "Convolutional models.")
        assert len(engine) == 2

        engine.remove_document(1)
        assert len(engine) == 1
        assert engine.index.search("transformer") == set()
        assert engine.index.search("convolutional") == {2}

    def test_clear_index(self) -> None:
        """clear empties the index and all tracking structures."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Attention mechanisms.",
            2: "Transformer models.",
        })
        assert len(engine) == 2

        engine.clear()
        assert len(engine) == 0
        assert len(engine.index.index) == 0
        assert len(engine.index.document_terms) == 0
        assert len(engine.index.document_term_frequencies) == 0

    def test_clear_preserves_component_instances(self) -> None:
        """clear resets index data without replacing component object instances."""
        engine = LexicalRetrievalEngine()
        engine.build_index({1: "Text content."})

        p_id = id(engine.preprocessor)
        idx_id = id(engine.index)
        tfidf_id = id(engine.tfidf)
        sim_id = id(engine.similarity)
        ranker_id = id(engine.ranker)

        engine.clear()

        assert id(engine.preprocessor) == p_id
        assert id(engine.index) == idx_id
        assert id(engine.tfidf) == tfidf_id
        assert id(engine.similarity) == sim_id
        assert id(engine.ranker) == ranker_id
        assert engine.tfidf.index is engine.index

    def test_failed_rebuild_does_not_destroy_existing_index(self) -> None:
        """If build_index fails validation or preprocessing, existing index remains intact."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Deep transformer architectures.",
            2: "Attention mechanisms.",
        })
        assert engine.index.search("transformer") == {1}
        assert engine.index.search("attention") == {2}
        assert len(engine) == 2

        # Attempt rebuild with invalid document text type
        with pytest.raises(TypeError, match="Document text for ID 4 must be a str"):
            engine.build_index({
                3: "Valid replacement text.",
                4: 12345,  # type: ignore[dict-item]
            })

        # Existing index must be completely preserved
        assert len(engine) == 2
        assert engine.index.search("transformer") == {1}
        assert engine.index.search("attention") == {2}
        assert 1 in engine.index.document_terms
        assert 2 in engine.index.document_terms
        assert 3 not in engine.index.document_terms

        # Attempt rebuild with invalid ID
        with pytest.raises(TypeError, match="Document ID must be an int"):
            engine.build_index({
                True: "Invalid ID type text.",  # type: ignore[dict-item]
            })

        # Existing index remains untouched
        assert len(engine) == 2
        assert engine.index.search("transformer") == {1}

    def test_failed_rebuild_during_preprocessing_does_not_destroy_index(self) -> None:
        """If preprocessing fails during build_index, the original index remains intact."""

        class FailingPreprocessor(TextPreprocessor):
            def preprocess(self, text: str) -> list[str]:
                if "failing" in text:
                    raise RuntimeError("Simulated failure in preprocessor")
                return super().preprocess(text)

        failing_preprocessor = FailingPreprocessor()
        engine = LexicalRetrievalEngine(preprocessor=failing_preprocessor)
        engine.build_index({1: "Initial valid text."})
        assert engine.index.search("initial") == {1}

        # Attempt rebuild with text that triggers preprocessor failure
        with pytest.raises(RuntimeError, match="Simulated failure in preprocessor"):
            engine.build_index({
                2: "Valid text before failing.",
                3: "failing text causes error.",
            })

        # Original index must be preserved intact
        assert len(engine) == 1
        assert engine.index.search("initial") == {1}
        assert 2 not in engine.index.document_terms
        assert 3 not in engine.index.document_terms


class TestLexicalRetrievalEngineSearch:
    """Tests for LexicalRetrievalEngine search functionality."""

    @pytest.fixture
    def populated_engine(self) -> LexicalRetrievalEngine:
        """Provide an engine populated with sample NLP research chunks."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Attention mechanisms and self-attention in transformer models.",
            2: "Bidirectional encoder representations from transformers for masked language modeling.",
            3: "Convolutional neural networks for deep visual recognition and image understanding.",
            4: "Generative pre-training with causal language modeling objectives.",
        })
        return engine

    # 1-6. Basic retrieval
    def test_search_against_multiple_documents(self, populated_engine: LexicalRetrievalEngine) -> None:
        """Searching a query against multiple indexed documents returns matching results."""
        results = populated_engine.search("transformer language modeling")
        assert len(results) > 0
        doc_ids = [doc_id for doc_id, _ in results]
        # Doc 1, 2, and 4 all discuss transformer or language modeling
        assert 2 in doc_ids
        assert 1 in doc_ids or 4 in doc_ids

    def test_search_returns_relevant_documents(self, populated_engine: LexicalRetrievalEngine) -> None:
        """Search returns documents relevant to the specific query terms."""
        results = populated_engine.search("convolutional image recognition")
        assert len(results) == 1
        doc_id, score = results[0]
        assert doc_id == 3
        assert score > 0.0

    def test_search_results_sorted_by_descending_score(self, populated_engine: LexicalRetrievalEngine) -> None:
        """Search results are ordered strictly by descending similarity score."""
        results = populated_engine.search("transformer language modeling", top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_results_structure_and_types(self, populated_engine: LexicalRetrievalEngine) -> None:
        """Each search result is a (doc_id, similarity_score) tuple with exact types."""
        results = populated_engine.search("transformer")
        assert isinstance(results, list)
        for doc_id, score in results:
            assert isinstance(doc_id, int)
            assert not isinstance(doc_id, bool)
            assert isinstance(score, float)
            assert score > 0.0

    def test_search_top_k_one_returns_single_result(self, populated_engine: LexicalRetrievalEngine) -> None:
        """top_k=1 limits the result list to exactly one highest-scoring document."""
        results = populated_engine.search("transformer", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], tuple)

    def test_search_larger_top_k_returns_all_eligible(self, populated_engine: LexicalRetrievalEngine) -> None:
        """When top_k exceeds candidate count, all positive-scoring candidates are returned."""
        results = populated_engine.search("convolutional", top_k=50)
        assert len(results) == 1
        assert results[0][0] == 3

    # 7-10. Candidate selection
    def test_candidate_selection_union_based(self) -> None:
        """Candidate selection uses union across query terms, not intersection."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Quantum computing algorithms.",
            2: "Neural networks representations.",
            3: "Astrophysics and planetary exploration.",
        })
        results = engine.search("quantum neural")
        doc_ids = {doc_id for doc_id, _ in results}
        assert doc_ids == {1, 2}
        assert 3 not in doc_ids

    def test_document_matching_only_one_query_term_is_eligible(self) -> None:
        """A document containing only one of several query terms is retrieved."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Transformer architectures.",
            2: "Recurrent networks.",
        })
        # Only "transformer" matches doc 1; "vision" and "robotics" match nothing
        results = engine.search("transformer vision robotics")
        assert len(results) == 1
        assert results[0][0] == 1

    def test_documents_with_no_matching_terms_excluded(self, populated_engine: LexicalRetrievalEngine) -> None:
        """Documents with no matching terms are excluded from search results."""
        results = populated_engine.search("convolutional")
        doc_ids = {doc_id for doc_id, _ in results}
        assert doc_ids == {3}
        assert 1 not in doc_ids
        assert 2 not in doc_ids
        assert 4 not in doc_ids

    def test_empty_document_in_index_excluded(self) -> None:
        """An empty document in the index is never returned as a match."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "",
            2: "Deep learning transformer models.",
        })
        results = engine.search("learning")
        assert len(results) == 1
        assert results[0][0] == 2

    # 11-17. Edge cases
    def test_empty_query_returns_empty_list(self, populated_engine: LexicalRetrievalEngine) -> None:
        """An empty string query returns an empty list without error."""
        assert populated_engine.search("") == []

    def test_whitespace_only_query_returns_empty_list(self, populated_engine: LexicalRetrievalEngine) -> None:
        """A whitespace-only query returns an empty list without error."""
        assert populated_engine.search("   \t  \n  ") == []

    def test_stopword_only_query_returns_empty_list(self, populated_engine: LexicalRetrievalEngine) -> None:
        """A query containing only stopwords produces an empty query vector and returns []."""
        assert populated_engine.search("the in and of for at") == []

    def test_unknown_query_terms_returns_empty_list(self, populated_engine: LexicalRetrievalEngine) -> None:
        """A query with terms not present in the index returns []."""
        assert populated_engine.search("xenon superconductivity crystallography") == []

    def test_empty_index_returns_empty_list(self) -> None:
        """Searching against an unpopulated index returns []."""
        engine = LexicalRetrievalEngine()
        assert engine.search("transformer") == []

    @pytest.mark.parametrize("invalid_query", [None, 123, 4.5, True, ["query"], {"q": "text"}])
    def test_invalid_query_type_raises_type_error(self, invalid_query: object) -> None:
        """Non-string query inputs raise TypeError."""
        engine = LexicalRetrievalEngine()
        with pytest.raises(TypeError, match="query must be a str"):
            engine.search(invalid_query)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_top_k", [True, False, 2.5, "5", [5], None])
    def test_invalid_top_k_type_raises_type_error(self, invalid_top_k: object) -> None:
        """Non-integer or boolean top_k inputs raise TypeError."""
        engine = LexicalRetrievalEngine()
        with pytest.raises(TypeError, match="top_k must be a positive int"):
            engine.search("query", top_k=invalid_top_k)  # type: ignore[arg-type]

    @pytest.mark.parametrize("non_positive_top_k", [0, -1, -100])
    def test_non_positive_top_k_raises_value_error(self, non_positive_top_k: int) -> None:
        """Zero or negative top_k inputs raise ValueError."""
        engine = LexicalRetrievalEngine()
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            engine.search("query", top_k=non_positive_top_k)

    # 18-22. Dynamic index behavior
    def test_search_after_add_document(self) -> None:
        """Newly added documents are immediately retrievable by search."""
        engine = LexicalRetrievalEngine()
        engine.add_document(1, "Deep reinforcement learning.")
        results_before = engine.search("transformer")
        assert results_before == []

        engine.add_document(2, "Transformer attention networks.")
        results_after = engine.search("transformer")
        assert len(results_after) == 1
        assert results_after[0][0] == 2

    def test_search_after_replace_existing_document(self) -> None:
        """Replacing a document updates search results and removes stale terms."""
        engine = LexicalRetrievalEngine()
        engine.add_document(1, "Recurrent architectures for sequential processing.")
        assert len(engine.search("recurrent")) == 1

        # Replace doc 1 with new content
        engine.add_document(1, "Convolutional vision architectures.")
        # Stale term should not match
        assert engine.search("recurrent") == []
        # New term should match
        results = engine.search("convolutional")
        assert len(results) == 1
        assert results[0][0] == 1

    def test_search_after_remove_document(self) -> None:
        """Removed documents are excluded from subsequent search results."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Transformer models.",
            2: "Transformer pipelines.",
        })
        assert len(engine.search("transformer")) == 2

        engine.remove_document(1)
        results = engine.search("transformer")
        assert len(results) == 1
        assert results[0][0] == 2

    def test_search_after_clear_index(self, populated_engine: LexicalRetrievalEngine) -> None:
        """Clearing the index results in subsequent searches returning empty lists."""
        assert len(populated_engine.search("transformer")) > 0
        populated_engine.clear()
        assert populated_engine.search("transformer") == []

    def test_search_after_rebuild_index(self) -> None:
        """Rebuilding the index replaces the search corpus atomically."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Recurrent neural architectures.",
            2: "Long short-term memory networks.",
        })
        assert len(engine.search("recurrent")) == 1

        engine.build_index({
            3: "Transformer attention layers.",
            4: "Self-attention transformer blocks.",
        })
        # Previous corpus terms no longer return matches
        assert engine.search("recurrent") == []
        # New corpus terms are searchable
        results = engine.search("transformer")
        doc_ids = {doc_id for doc_id, _ in results}
        assert doc_ids == {3, 4}

    # 23-27. Component integration and spies
    def test_delegation_to_tfidf_transform_query(self) -> None:
        """Query vector generation is delegated to TfidfCalculator.transform_query."""
        engine = LexicalRetrievalEngine()
        engine.build_index({1: "Transformer models."})

        original_transform = engine.tfidf.transform_query
        spy = MagicMock(side_effect=original_transform)
        engine.tfidf.transform_query = spy

        engine.search("transformer")
        spy.assert_called_once_with("transformer")

    def test_delegation_to_similarity(self) -> None:
        """Cosine similarity computation is delegated to CosineSimilarity.similarity."""
        engine = LexicalRetrievalEngine()
        engine.build_index({1: "Transformer models."})

        original_similarity = engine.similarity.similarity
        spy = MagicMock(side_effect=original_similarity)
        engine.similarity.similarity = spy

        engine.search("transformer")
        assert spy.call_count >= 1

    def test_delegation_to_ranker(self) -> None:
        """Final ranking is delegated to DocumentRanker.rank."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Transformer models.",
            2: "Transformer architectures.",
        })

        original_rank = engine.ranker.rank
        spy = MagicMock(side_effect=original_rank)
        engine.ranker.rank = spy

        results = engine.search("transformer", top_k=1)
        assert spy.call_count == 1
        call_args, call_kwargs = spy.call_args
        # First positional argument is similarity_scores dict
        assert isinstance(call_args[0], dict)
        assert call_kwargs.get("top_k") == 1
        assert len(results) == 1

    def test_query_vector_generated_only_once(self) -> None:
        """Query vector is generated exactly once per search call."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Deep transformer models.",
            2: "Transformer attention mechanisms.",
            3: "Generative transformer pre-training.",
        })

        original_transform = engine.tfidf.transform_query
        spy = MagicMock(side_effect=original_transform)
        engine.tfidf.transform_query = spy

        engine.search("transformer")
        assert spy.call_count == 1

    def test_document_vectors_generated_only_for_candidates(self) -> None:
        """Document TF-IDF vectors are generated only for candidate documents."""
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Attention transformer mechanisms.",
            2: "Transformer models.",
            3: "Convolutional neural vision.",
            4: "Astrophysics and cosmology.",
        })

        original_transform_doc = engine.tfidf.transform_document
        spy = MagicMock(side_effect=original_transform_doc)
        engine.tfidf.transform_document = spy

        # "transformer" only matches doc 1 and doc 2
        engine.search("transformer")

        called_doc_ids = {call[0][0] for call in spy.call_args_list}
        assert called_doc_ids == {1, 2}
        assert 3 not in called_doc_ids
        assert 4 not in called_doc_ids

    def test_search_tie_breaking_by_ascending_doc_id(self) -> None:
        """Equal similarity scores break ties deterministically by ascending document ID."""
        engine = LexicalRetrievalEngine()
        # Documents with identical text have identical TF-IDF weights and cosine similarity
        engine.build_index({
            2: "Transformer attention.",
            1: "Transformer attention.",
        })
        results = engine.search("transformer")
        assert len(results) == 2
        assert results[0][0] == 1
        assert results[1][0] == 2
        assert results[0][1] == results[1][1]

    def test_search_porter_stemmed_terms_without_double_preprocessing(self) -> None:
        """Searching for terms whose Porter stems undergo further stemming when re-processed

        (e.g., 'embeddings' -> 'embed' -> 'emb', or 'experimental' -> 'experiment' -> 'experi')
        must correctly retrieve matching documents without dropping true matches or
        selecting spurious candidates.
        """
        engine = LexicalRetrievalEngine()
        engine.build_index({
            1: "Word embeddings are dense vector representations.",
            2: "Experimental evaluation of retrieval algorithms.",
            3: "We embed the vectors into memory.",
        })

        # 1. 'embeddings' stems once to 'embed'. Must retrieve document 1.
        results_embeddings = engine.search("embeddings")
        assert len(results_embeddings) == 1
        assert results_embeddings[0][0] == 1
        assert results_embeddings[0][1] > 0.0

        # 2. 'experimental' stems once to 'experiment'. Must retrieve document 2.
        results_experimental = engine.search("experimental")
        assert len(results_experimental) == 1
        assert results_experimental[0][0] == 2
        assert results_experimental[0][1] > 0.0

        # 3. 'embeddings' does not retrieve a document containing only 'embed' when stems differ
        engine_embed_only = LexicalRetrievalEngine()
        engine_embed_only.build_index({
            10: "We embed the vectors into memory.",
        })
        # Document 10 has stem 'emb'; query 'embeddings' has stem 'embed'
        assert engine_embed_only.search("embeddings") == []

    def test_search_candidate_retrieval_uses_preprocessed_true(self) -> None:
        """Candidate retrieval in engine.search() delegates to index.search with preprocessed=True."""
        engine = LexicalRetrievalEngine()
        engine.build_index({1: "Word embeddings."})

        original_search = engine.index.search
        spy = MagicMock(side_effect=original_search)
        engine.index.search = spy

        engine.search("embeddings")
        assert spy.call_count == 1
        call_args, call_kwargs = spy.call_args
        assert call_args[0] == "embed"
        assert call_kwargs.get("preprocessed") is True
