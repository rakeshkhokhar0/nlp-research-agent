"""Unit tests for TfidfCalculator in lexical retrieval."""

import math
import pytest

from app.modules.retrieval.lexical import (
    InvertedIndex,
    TextPreprocessor,
    TfidfCalculator,
)


@pytest.fixture
def sample_index() -> InvertedIndex:
    """Fixture providing a 3-document populated InvertedIndex."""
    index = InvertedIndex()
    # Doc 1: neural(1), network(1), deep(1), learn(1)
    index.add_document(1, "Neural networks and deep learning.")
    # Doc 2: convolut(1), neural(1), network(1)
    index.add_document(2, "Convolutional neural networks.")
    # Doc 3: deep(1), reinforc(1), learn(1)
    index.add_document(3, "Deep reinforcement learning.")
    return index


class TestTfidfTermFrequencyAndIdf:
    """Tests for raw TF and smoothed IDF computation."""

    def test_raw_term_frequency(self, sample_index: InvertedIndex) -> None:
        """compute_tf returns accurate raw term frequencies."""
        calc = TfidfCalculator(sample_index)
        assert calc.compute_tf("neural", 1) == 1
        assert calc.compute_tf("neural", 2) == 1
        assert calc.compute_tf("neural", 3) == 0
        assert calc.compute_tf("convolut", 2) == 1
        assert calc.compute_tf("convolutional", 2) == 1
        assert calc.compute_tf("deep", 1) == 1
        assert calc.compute_tf("deep", 3) == 1
        assert calc.compute_tf("unknown", 1) == 0

    def test_raw_term_frequency_repeated_tokens(self) -> None:
        """Repeated tokens within a document return accurate counts."""
        index = InvertedIndex()
        index.add_document(1, "neural neural neural network network")
        calc = TfidfCalculator(index)
        assert calc.compute_tf("neural", 1) == 3
        assert calc.compute_tf("network", 1) == 2
        assert calc.compute_tf("model", 1) == 0

    def test_idf_formula_with_known_values(self, sample_index: InvertedIndex) -> None:
        """compute_idf matches mathematical formula: log((N + 1) / (DF + 1)) + 1."""
        calc = TfidfCalculator(sample_index)
        n = 3

        # 'neural' appears in Doc 1 and Doc 2 -> DF = 2
        expected_idf_neural = math.log((n + 1.0) / (2.0 + 1.0)) + 1.0  # log(4/3) + 1
        assert calc.compute_idf("neural") == pytest.approx(expected_idf_neural)
        assert calc.compute_idf("networks") == pytest.approx(expected_idf_neural)

        # 'convolut' appears only in Doc 2 -> DF = 1
        expected_idf_convolut = math.log((n + 1.0) / (1.0 + 1.0)) + 1.0  # log(4/2) + 1 = log(2) + 1
        assert calc.compute_idf("convolutional") == pytest.approx(expected_idf_convolut)
        assert calc.compute_idf("convolut") == pytest.approx(expected_idf_convolut)

    def test_idf_for_unknown_terms(self, sample_index: InvertedIndex) -> None:
        """Terms not present in the index have DF = 0 and valid positive IDF."""
        calc = TfidfCalculator(sample_index)
        n = 3
        # 'transformer' appears in 0 docs -> DF = 0
        expected_idf_unknown = math.log((n + 1.0) / (0.0 + 1.0)) + 1.0  # log(4/1) + 1 = log(4) + 1
        assert calc.compute_idf("transformer") == pytest.approx(expected_idf_unknown)
        assert calc.compute_idf("quantum") == pytest.approx(expected_idf_unknown)

    def test_zero_document_corpus(self) -> None:
        """Empty corpus (N = 0) returns log(1) + 1 = 1.0 without error."""
        index = InvertedIndex()
        calc = TfidfCalculator(index)
        # N = 0, DF = 0 -> log(1/1) + 1 = 1.0
        assert calc.compute_idf("neural") == pytest.approx(1.0)
        assert calc.compute_idf("unknown") == pytest.approx(1.0)

    def test_idf_updates_dynamically_after_addition_and_removal(self, sample_index: InvertedIndex) -> None:
        """IDF changes dynamically as documents are added or removed without explicit refitting."""
        calc = TfidfCalculator(sample_index)
        # Initially N = 3, 'convolut' DF = 1
        idf_before = calc.compute_idf("convolut")
        assert idf_before == pytest.approx(math.log(4.0 / 2.0) + 1.0)

        # Add Doc 4 with 'convolutional' -> N = 4, 'convolut' DF = 2
        sample_index.add_document(4, "Convolutional neural architecture.")
        idf_after_add = calc.compute_idf("convolut")
        expected_after_add = math.log((4.0 + 1.0) / (2.0 + 1.0)) + 1.0  # log(5/3) + 1
        assert idf_after_add == pytest.approx(expected_after_add)
        assert idf_after_add < idf_before

        # Remove Doc 4 and Doc 2 -> N = 2, 'convolut' DF = 0
        sample_index.remove_document(4)
        sample_index.remove_document(2)
        idf_after_rem = calc.compute_idf("convolut")
        expected_after_rem = math.log((2.0 + 1.0) / (0.0 + 1.0)) + 1.0  # log(3/1) + 1
        assert idf_after_rem == pytest.approx(expected_after_rem)


class TestTfidfWeightCalculation:
    """Tests for compute_weight(term, document_id)."""

    def test_tfidf_weight_calculation(self, sample_index: InvertedIndex) -> None:
        """compute_weight equals raw TF * smoothed IDF."""
        calc = TfidfCalculator(sample_index)
        # Doc 2 has 'convolut': TF = 1, IDF = log(2) + 1
        expected_weight = 1 * (math.log(2.0) + 1.0)
        assert calc.compute_weight("convolutional", 2) == pytest.approx(expected_weight)
        assert calc.compute_weight("convolut", 2) == pytest.approx(expected_weight)

    def test_weight_missing_term_and_unknown_document_returns_zero(self, sample_index: InvertedIndex) -> None:
        """compute_weight returns 0.0 for missing term or unknown document."""
        calc = TfidfCalculator(sample_index)
        assert calc.compute_weight("unknown", 1) == 0.0
        assert calc.compute_weight("neural", 999) == 0.0

    def test_weight_updates_on_document_replacement(self) -> None:
        """Replacing a document updates TF and compute_weight."""
        index = InvertedIndex()
        index.add_document(1, "recurrent recurrent network")
        calc = TfidfCalculator(index)

        weight_recurrent = calc.compute_weight("recurrent", 1)
        assert weight_recurrent > 0.0
        assert calc.compute_weight("transformer", 1) == 0.0

        # Replace document 1
        index.add_document(1, "transformer transformer transformer model")
        assert calc.compute_weight("recurrent", 1) == 0.0
        assert calc.compute_weight("transformer", 1) > 0.0
        assert calc.compute_tf("transformer", 1) == 3


class TestTfidfDocumentVectorGeneration:
    """Tests for transform_document(document_id)."""

    def test_document_vector_generation(self, sample_index: InvertedIndex) -> None:
        """transform_document returns dictionary of terms and normalized weights."""
        calc = TfidfCalculator(sample_index)
        vec2 = calc.transform_document(2)

        assert "convolut" in vec2
        assert "neural" in vec2
        assert "network" in vec2
        assert len(vec2) == 3

        # Doc 2 has convolut (DF=1) and neural, network (DF=2)
        # convolut IDF > neural IDF, so normalized weight for convolut should be higher
        assert vec2["convolut"] > vec2["neural"]
        assert vec2["neural"] == pytest.approx(vec2["network"])

    def test_document_vector_l2_normalization_magnitude(self, sample_index: InvertedIndex) -> None:
        """Document vectors have Euclidean length (L2 norm) approximately equal to 1.0."""
        calc = TfidfCalculator(sample_index)
        for doc_id in [1, 2, 3]:
            vec = calc.transform_document(doc_id)
            magnitude = math.sqrt(sum(w * w for w in vec.values()))
            assert magnitude == pytest.approx(1.0, abs=1e-6)

    def test_document_vector_empty_and_unknown_documents(self) -> None:
        """transform_document returns {} for empty, stopword-only, or unknown documents."""
        index = InvertedIndex()
        index.add_document(1, "the and of")  # stopword only
        calc = TfidfCalculator(index)

        assert calc.transform_document(1) == {}
        assert calc.transform_document(999) == {}

    def test_document_vector_updated_on_replacement(self) -> None:
        """Replaced document vector reflects new terms without stale entries."""
        index = InvertedIndex()
        index.add_document(1, "recurrent recurrent architectures")
        calc = TfidfCalculator(index)

        vec_initial = calc.transform_document(1)
        assert "recurr" in vec_initial
        assert "transform" not in vec_initial

        index.add_document(1, "transformer models attention")
        vec_replaced = calc.transform_document(1)
        assert "recurr" not in vec_replaced
        assert "transform" in vec_replaced
        assert "model" in vec_replaced
        assert "attent" in vec_replaced
        magnitude = math.sqrt(sum(w * w for w in vec_replaced.values()))
        assert magnitude == pytest.approx(1.0, abs=1e-6)


class TestTfidfQueryVectorGeneration:
    """Tests for transform_query(query)."""

    def test_query_vector_generation(self, sample_index: InvertedIndex) -> None:
        """transform_query preprocessed text and computes L2-normalized weights."""
        calc = TfidfCalculator(sample_index)
        vec = calc.transform_query("neural convolutional")

        assert "neural" in vec
        assert "convolut" in vec
        assert len(vec) == 2
        magnitude = math.sqrt(sum(w * w for w in vec.values()))
        assert magnitude == pytest.approx(1.0, abs=1e-6)

    def test_query_vector_repeated_query_terms(self, sample_index: InvertedIndex) -> None:
        """Repeated terms in query increase their relative weight."""
        calc = TfidfCalculator(sample_index)
        vec = calc.transform_query("neural neural convolutional")

        # 'neural' appears twice, 'convolut' once
        # Raw weights: 2 * IDF(neural) vs 1 * IDF(convolut)
        assert "neural" in vec
        assert "convolut" in vec
        assert vec["neural"] > vec["convolut"]
        magnitude = math.sqrt(sum(w * w for w in vec.values()))
        assert magnitude == pytest.approx(1.0, abs=1e-6)

    def test_query_vector_single_term(self, sample_index: InvertedIndex) -> None:
        """Single term query produces a unit vector with weight 1.0."""
        calc = TfidfCalculator(sample_index)
        vec = calc.transform_query("transformer")
        assert vec == {"transform": pytest.approx(1.0)}

    def test_query_vector_empty_and_stopword_only(self, sample_index: InvertedIndex) -> None:
        """Empty, whitespace, or stopword-only queries return empty dictionary {}."""
        calc = TfidfCalculator(sample_index)
        assert calc.transform_query("") == {}
        assert calc.transform_query("   ") == {}
        assert calc.transform_query("the is are was and of") == {}
        assert calc.transform_query("... ??? ---") == {}

    def test_query_vector_out_of_vocabulary_term(self, sample_index: InvertedIndex) -> None:
        """Query with term not present in corpus gets valid DF=0 IDF and unit norm."""
        calc = TfidfCalculator(sample_index)
        vec = calc.transform_query("quantum superconductor")
        assert "quantum" in vec
        assert "superconductor" in vec
        magnitude = math.sqrt(sum(w * w for w in vec.values()))
        assert magnitude == pytest.approx(1.0, abs=1e-6)


class TestTfidfConsistencyAndEdgeCases:
    """Tests for preprocessor consistency, zero corpus, and input validation."""

    def test_consistency_with_same_preprocessor(self) -> None:
        """TfidfCalculator shares exactly the same preprocessor instance as InvertedIndex."""
        custom_preprocessor = TextPreprocessor()
        index = InvertedIndex(preprocessor=custom_preprocessor)
        calc = TfidfCalculator(index)

        assert calc.index.preprocessor is custom_preprocessor
        # Index document and query with matching technical term
        index.add_document(1, "Using BERT-base and GPT-4 for zero-shot classification.")
        doc_vec = calc.transform_document(1)
        query_vec = calc.transform_query("BERT-base GPT-4")

        assert "bert-base" in doc_vec
        assert "gpt-4" in doc_vec
        assert "bert-base" in query_vec
        assert "gpt-4" in query_vec

    def test_empty_corpus_transform_query(self) -> None:
        """Query transformation works safely on an empty corpus."""
        index = InvertedIndex()
        calc = TfidfCalculator(index)

        vec = calc.transform_query("neural networks")
        assert "neural" in vec
        assert "network" in vec
        magnitude = math.sqrt(sum(w * w for w in vec.values()))
        assert magnitude == pytest.approx(1.0, abs=1e-6)

    def test_invalid_calculator_init(self) -> None:
        """Non-InvertedIndex argument to constructor raises TypeError."""
        with pytest.raises(TypeError, match="index must be an instance of InvertedIndex"):
            TfidfCalculator("invalid_index")  # type: ignore[arg-type]

    def test_multi_token_rejection_in_single_term_methods(self, sample_index: InvertedIndex) -> None:
        """Multi-token inputs are rejected by compute_idf, compute_tf, and compute_weight."""
        calc = TfidfCalculator(sample_index)
        with pytest.raises(ValueError, match="Query must produce at most one term"):
            calc.compute_idf("neural networks")

        with pytest.raises(ValueError, match="Query must produce at most one term"):
            calc.compute_tf("neural networks", 1)

        with pytest.raises(ValueError, match="Query must produce at most one term"):
            calc.compute_weight("neural networks", 1)

    @pytest.mark.parametrize("invalid_id", [True, False, 0, -1, -10, "1", 2.5, None])
    def test_invalid_document_id_validation(self, sample_index: InvertedIndex, invalid_id: object) -> None:
        """Invalid document_id raises TypeError or ValueError across all document methods."""
        calc = TfidfCalculator(sample_index)
        with pytest.raises((TypeError, ValueError)):
            calc.compute_tf("neural", invalid_id)  # type: ignore[arg-type]

        with pytest.raises((TypeError, ValueError)):
            calc.compute_weight("neural", invalid_id)  # type: ignore[arg-type]

        with pytest.raises((TypeError, ValueError)):
            calc.transform_document(invalid_id)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_term", [None, 123, 4.5, True, ["neural"]])
    def test_invalid_term_types(self, sample_index: InvertedIndex, invalid_term: object) -> None:
        """Non-string terms raise TypeError across compute_idf, compute_tf, and compute_weight."""
        calc = TfidfCalculator(sample_index)
        with pytest.raises(TypeError, match="term must be a str"):
            calc.compute_idf(invalid_term)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="term must be a str"):
            calc.compute_tf(invalid_term, 1)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="term must be a str"):
            calc.compute_weight(invalid_term, 1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_query", [None, 123, 4.5, True, ["query"]])
    def test_invalid_query_types(self, sample_index: InvertedIndex, invalid_query: object) -> None:
        """Non-string queries raise TypeError in transform_query."""
        calc = TfidfCalculator(sample_index)
        with pytest.raises(TypeError, match="query must be a str"):
            calc.transform_query(invalid_query)  # type: ignore[arg-type]
