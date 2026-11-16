"""Unit tests for CosineSimilarity in lexical retrieval."""

import copy
import math
import pytest

from app.modules.retrieval.lexical import (
    CosineSimilarity,
    cosine_similarity,
)


@pytest.fixture
def sim_calculator() -> CosineSimilarity:
    """Fixture returning a CosineSimilarity instance."""
    return CosineSimilarity()


class TestCosineSimilarityCalculations:
    """Tests for core mathematical dot product and similarity calculations."""

    def test_identical_vectors(self, sim_calculator: CosineSimilarity) -> None:
        """Identical L2-normalized vectors produce a similarity score of exactly 1.0."""
        vec1 = {"neural": 0.6, "network": 0.8}
        score = sim_calculator.similarity(vec1, vec1)
        assert score == pytest.approx(1.0)
        assert cosine_similarity(vec1, vec1) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, sim_calculator: CosineSimilarity) -> None:
        """Disjoint / orthogonal vectors produce a similarity score of 0.0."""
        query = {"neural": 0.6, "network": 0.8}
        doc = {"transformer": 0.5, "attention": 0.8660254037844386}
        assert sim_calculator.similarity(query, doc) == 0.0

    def test_partially_overlapping_vectors(self, sim_calculator: CosineSimilarity) -> None:
        """Partially overlapping vectors sum only the common terms."""
        query = {"neural": 0.6, "network": 0.8}
        doc = {"network": 0.5, "transformer": 0.8660254037844386}
        # Common term: 'network' -> 0.8 * 0.5 = 0.4
        expected = 0.8 * 0.5
        assert sim_calculator.similarity(query, doc) == pytest.approx(expected)

    def test_empty_query_vector(self, sim_calculator: CosineSimilarity) -> None:
        """Empty query vector returns 0.0."""
        doc = {"neural": 0.6, "network": 0.8}
        assert sim_calculator.similarity({}, doc) == 0.0

    def test_empty_document_vector(self, sim_calculator: CosineSimilarity) -> None:
        """Empty document vector returns 0.0."""
        query = {"neural": 0.6, "network": 0.8}
        assert sim_calculator.similarity(query, {}) == 0.0

    def test_both_vectors_empty(self, sim_calculator: CosineSimilarity) -> None:
        """Both vectors empty returns 0.0."""
        assert sim_calculator.similarity({}, {}) == 0.0

    def test_no_common_terms(self, sim_calculator: CosineSimilarity) -> None:
        """Non-empty vectors with no shared keys return 0.0."""
        v1 = {"a": 0.5, "b": 0.5}
        v2 = {"c": 0.7, "d": 0.7}
        assert sim_calculator.similarity(v1, v2) == 0.0

    def test_single_term_vectors(self, sim_calculator: CosineSimilarity) -> None:
        """Single term matching and non-matching vectors."""
        assert sim_calculator.similarity({"term": 1.0}, {"term": 1.0}) == pytest.approx(1.0)
        assert sim_calculator.similarity({"term1": 1.0}, {"term2": 1.0}) == 0.0

    def test_floating_point_precision(self, sim_calculator: CosineSimilarity) -> None:
        """Maintains standard floating-point precision across multi-term weights."""
        query = {"alpha": 0.123456789, "beta": 0.987654321}
        doc = {"alpha": 0.333333333, "beta": 0.666666667}
        expected = (0.123456789 * 0.333333333) + (0.987654321 * 0.666666667)
        assert sim_calculator.similarity(query, doc) == pytest.approx(expected, rel=1e-9)

    def test_unclamped_score_preservation(self, sim_calculator: CosineSimilarity) -> None:
        """Cosine similarity does not silently clamp scores above 1.0."""
        # Unclamped dot product test
        v1 = {"a": 2.0}
        v2 = {"a": 3.0}
        assert sim_calculator.similarity(v1, v2) == pytest.approx(6.0)

    def test_inputs_not_required_to_have_every_term_in_common(
        self, sim_calculator: CosineSimilarity
    ) -> None:
        """Asymmetric vectors with varying sizes compute dot product over intersection."""
        query = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
        doc = {"c": 0.5, "z": 0.9}
        # Only 'c' is shared: 0.3 * 0.5 = 0.15
        assert sim_calculator.similarity(query, doc) == pytest.approx(0.15)


class TestCosineSimilarityValidation:
    """Tests for input validation, immutability, and rejection of invalid types/values."""

    @pytest.mark.parametrize("invalid_input", [None, 123, 4.5, "vector", True, False, [1, 2], (1, 2)])
    def test_invalid_vector_types(self, sim_calculator: CosineSimilarity, invalid_input: object) -> None:
        """Non-dict inputs for query_vector or document_vector raise TypeError."""
        valid_vec = {"neural": 1.0}
        with pytest.raises(TypeError, match="must be a dict"):
            sim_calculator.similarity(invalid_input, valid_vec)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a dict"):
            sim_calculator.similarity(valid_vec, invalid_input)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_key", [123, None, True, False, 4.5, ("tuple",)])
    def test_non_string_keys_raise_type_error(
        self, sim_calculator: CosineSimilarity, invalid_key: object
    ) -> None:
        """Non-string dictionary keys raise TypeError."""
        bad_vec = {invalid_key: 1.0}  # type: ignore[dict-item]
        valid_vec = {"neural": 1.0}
        with pytest.raises(TypeError, match="All keys in .* must be str"):
            sim_calculator.similarity(bad_vec, valid_vec)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="All keys in .* must be str"):
            sim_calculator.similarity(valid_vec, bad_vec)  # type: ignore[arg-type]

    def test_empty_string_keys_raise_value_error(self, sim_calculator: CosineSimilarity) -> None:
        """Empty string keys raise ValueError."""
        bad_vec = {"": 1.0}
        valid_vec = {"neural": 1.0}
        with pytest.raises(ValueError, match="Keys in .* cannot be empty strings"):
            sim_calculator.similarity(bad_vec, valid_vec)

        with pytest.raises(ValueError, match="Keys in .* cannot be empty strings"):
            sim_calculator.similarity(valid_vec, bad_vec)

    @pytest.mark.parametrize("bool_val", [True, False])
    def test_boolean_values_rejected_with_type_error(
        self, sim_calculator: CosineSimilarity, bool_val: bool
    ) -> None:
        """Boolean values raise TypeError even though bool subclasses int in Python."""
        bad_vec = {"neural": bool_val}
        valid_vec = {"neural": 1.0}
        with pytest.raises(TypeError, match="All values in .* must be numeric"):
            sim_calculator.similarity(bad_vec, valid_vec)

        with pytest.raises(TypeError, match="All values in .* must be numeric"):
            sim_calculator.similarity(valid_vec, bad_vec)

    @pytest.mark.parametrize("non_numeric_val", [None, "1.0", [1.0], {"val": 1.0}])
    def test_non_numeric_values_raise_type_error(
        self, sim_calculator: CosineSimilarity, non_numeric_val: object
    ) -> None:
        """Non-numeric values raise TypeError."""
        bad_vec = {"neural": non_numeric_val}  # type: ignore[dict-item]
        valid_vec = {"neural": 1.0}
        with pytest.raises(TypeError, match="All values in .* must be numeric"):
            sim_calculator.similarity(bad_vec, valid_vec)  # type: ignore[arg-type]

    @pytest.mark.parametrize("neg_val", [-0.0001, -1.0, -100.5])
    def test_negative_values_raise_value_error(
        self, sim_calculator: CosineSimilarity, neg_val: float
    ) -> None:
        """Negative values raise ValueError."""
        bad_vec = {"neural": neg_val}
        valid_vec = {"neural": 1.0}
        with pytest.raises(ValueError, match="Values in .* must be non-negative"):
            sim_calculator.similarity(bad_vec, valid_vec)

        with pytest.raises(ValueError, match="Values in .* must be non-negative"):
            sim_calculator.similarity(valid_vec, bad_vec)

    @pytest.mark.parametrize("nan_or_inf", [float("nan"), float("inf"), float("-inf")])
    def test_nan_and_infinity_raise_value_error(
        self, sim_calculator: CosineSimilarity, nan_or_inf: float
    ) -> None:
        """NaN and infinite values raise ValueError."""
        bad_vec = {"neural": nan_or_inf}
        valid_vec = {"neural": 1.0}
        with pytest.raises(ValueError, match="Values in .* must be finite numbers"):
            sim_calculator.similarity(bad_vec, valid_vec)

        with pytest.raises(ValueError, match="Values in .* must be finite numbers"):
            sim_calculator.similarity(valid_vec, bad_vec)

    def test_input_dictionaries_remain_unchanged(self, sim_calculator: CosineSimilarity) -> None:
        """Neither input dictionary is mutated or modified by the similarity calculation."""
        query = {"neural": 0.6, "network": 0.8}
        doc = {"network": 0.5, "deep": 0.8660254037844386}

        query_copy = copy.deepcopy(query)
        doc_copy = copy.deepcopy(doc)

        sim_calculator.similarity(query, doc)

        assert query == query_copy
        assert doc == doc_copy
