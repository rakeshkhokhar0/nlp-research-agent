"""Unit tests for DocumentRanker in lexical retrieval."""

import copy
import math
import pytest

from app.modules.retrieval.lexical import (
    DocumentRanker,
    rank_documents,
)


@pytest.fixture
def ranker() -> DocumentRanker:
    """Fixture returning a DocumentRanker instance."""
    return DocumentRanker()


class TestDocumentRankerRanking:
    """Tests for document ranking ordering, slicing, and tie-breaking."""

    def test_correct_descending_ranking(self, ranker: DocumentRanker) -> None:
        """Documents are ranked in strictly descending order of similarity score."""
        scores = {1: 0.25, 2: 0.95, 3: 0.45, 4: 0.75}
        ranked = ranker.rank(scores)
        assert ranked == [(2, 0.95), (4, 0.75), (3, 0.45), (1, 0.25)]
        assert rank_documents(scores) == ranked

    def test_top_k_limits(self, ranker: DocumentRanker) -> None:
        """top_k restricts the number of returned results (top 1, top 3, default top 5)."""
        scores = {
            1: 0.1,
            2: 0.9,
            3: 0.3,
            4: 0.7,
            5: 0.5,
            6: 0.8,
            7: 0.2,
        }
        # Top 1
        assert ranker.rank(scores, top_k=1) == [(2, 0.9)]
        # Top 3
        assert ranker.rank(scores, top_k=3) == [(2, 0.9), (6, 0.8), (4, 0.7)]
        # Default top 5
        assert ranker.rank(scores) == [(2, 0.9), (6, 0.8), (4, 0.7), (5, 0.5), (3, 0.3)]

    def test_fewer_documents_than_k(self, ranker: DocumentRanker) -> None:
        """When fewer documents exist than top_k, all documents are returned sorted."""
        scores = {10: 0.4, 20: 0.8, 30: 0.6}
        ranked = ranker.rank(scores, top_k=10)
        assert len(ranked) == 3
        assert ranked == [(20, 0.8), (30, 0.6), (10, 0.4)]

    def test_k_greater_than_number_of_documents(self, ranker: DocumentRanker) -> None:
        """Large top_k does not produce index errors or pad results."""
        scores = {1: 0.9, 2: 0.5}
        assert ranker.rank(scores, top_k=100) == [(1, 0.9), (2, 0.5)]

    def test_empty_input(self, ranker: DocumentRanker) -> None:
        """Empty input dictionary returns an empty list."""
        assert ranker.rank({}) == []
        assert ranker.rank({}, top_k=1) == []
        assert ranker.rank({}, top_k=10) == []

    def test_equal_scores_deterministic_tie_breaking(self, ranker: DocumentRanker) -> None:
        """Equal similarity scores are broken deterministically by document ID ascending."""
        # All equal scores
        scores = {5: 0.85, 2: 0.85, 8: 0.85, 1: 0.85}
        ranked = ranker.rank(scores)
        assert ranked == [(1, 0.85), (2, 0.85), (5, 0.85), (8, 0.85)]

        # Mixed scores with a tie in the middle
        mixed_scores = {10: 0.9, 3: 0.8, 7: 0.8, 1: 0.95}
        mixed_ranked = ranker.rank(mixed_scores)
        assert mixed_ranked == [(1, 0.95), (10, 0.9), (3, 0.8), (7, 0.8)]

    def test_single_document_input(self, ranker: DocumentRanker) -> None:
        """Single-document dictionary returns a single tuple."""
        assert ranker.rank({42: 0.88}) == [(42, 0.88)]
        assert ranker.rank({42: 0.88}, top_k=1) == [(42, 0.88)]
        assert ranker.rank({42: 0.88}, top_k=10) == [(42, 0.88)]

    def test_integer_and_float_scores(self, ranker: DocumentRanker) -> None:
        """Integer scores (e.g. 0, 1) are accepted and returned as floats."""
        scores = {1: 1, 2: 0, 3: 0.75}
        ranked = ranker.rank(scores)
        assert ranked == [(1, 1.0), (3, 0.75), (2, 0.0)]
        assert all(isinstance(score, float) for _, score in ranked)

    def test_representable_large_integer_score(self, ranker: DocumentRanker) -> None:
        """Normal large integers representable as finite floats are accepted and ranked."""
        scores = {1: 10**100, 2: 10**50, 3: 10**150}
        ranked = ranker.rank(scores)
        assert ranked == [(3, float(10**150)), (1, float(10**100)), (2, float(10**50))]
        assert all(isinstance(score, float) for _, score in ranked)


class TestDocumentRankerValidation:
    """Tests for input validation, type safety, and immutability."""

    @pytest.mark.parametrize("invalid_dict", [None, [1, 2], "not_a_dict", 123, True, (1, 2)])
    def test_invalid_dictionary_types(self, ranker: DocumentRanker, invalid_dict: object) -> None:
        """Non-dict similarity_scores raises TypeError."""
        with pytest.raises(TypeError, match="similarity_scores must be a dict"):
            ranker.rank(invalid_dict)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_id", [True, False, 1.5, "1", None, ("tuple",)])
    def test_invalid_document_id_types(self, ranker: DocumentRanker, invalid_id: object) -> None:
        """Non-int or boolean document IDs raise TypeError."""
        with pytest.raises(TypeError, match="Document ID must be an int"):
            ranker.rank({invalid_id: 0.5})  # type: ignore[dict-item]

    @pytest.mark.parametrize("invalid_id", [0, -1, -50])
    def test_non_positive_document_ids(self, ranker: DocumentRanker, invalid_id: int) -> None:
        """Zero or negative document IDs raise ValueError."""
        with pytest.raises(ValueError, match="Document ID must be a positive integer"):
            ranker.rank({invalid_id: 0.5})

    @pytest.mark.parametrize("invalid_score", [True, False, None, "0.5", [0.5], {"score": 0.5}])
    def test_invalid_score_types(self, ranker: DocumentRanker, invalid_score: object) -> None:
        """Non-numeric or boolean scores raise TypeError."""
        with pytest.raises(TypeError, match="Similarity score for document .* must be numeric"):
            ranker.rank({1: invalid_score})  # type: ignore[dict-item]

    @pytest.mark.parametrize("neg_score", [-0.0001, -1.0, -100.0])
    def test_negative_scores(self, ranker: DocumentRanker, neg_score: float) -> None:
        """Negative scores raise ValueError."""
        with pytest.raises(ValueError, match="Similarity score for document .* must be non-negative"):
            ranker.rank({1: neg_score})

    @pytest.mark.parametrize("nan_or_inf", [float("nan"), float("inf"), float("-inf")])
    def test_nan_and_infinity_scores(self, ranker: DocumentRanker, nan_or_inf: float) -> None:
        """NaN and infinite scores raise ValueError."""
        with pytest.raises(ValueError, match="Similarity score for document .* must be finite"):
            ranker.rank({1: nan_or_inf})

    @pytest.mark.parametrize("overflow_score", [10**1000, 10**500, -(10**1000)])
    def test_unrepresentable_large_integer_score_raises_value_error(
        self, ranker: DocumentRanker, overflow_score: int
    ) -> None:
        """Integer scores too large to convert to a finite float raise ValueError instead of OverflowError."""
        with pytest.raises(ValueError, match="too large to convert to a finite float"):
            ranker.rank({1: overflow_score})

    @pytest.mark.parametrize("invalid_top_k", [True, False, None, "5", 2.5, [5]])
    def test_invalid_top_k_types(self, ranker: DocumentRanker, invalid_top_k: object) -> None:
        """Non-int or boolean top_k raises TypeError."""
        with pytest.raises(TypeError, match="top_k must be a positive int"):
            ranker.rank({1: 0.5}, top_k=invalid_top_k)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_top_k", [0, -1, -10])
    def test_non_positive_top_k(self, ranker: DocumentRanker, invalid_top_k: int) -> None:
        """Zero or negative top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            ranker.rank({1: 0.5}, top_k=invalid_top_k)

    def test_input_dictionary_remains_unchanged(self, ranker: DocumentRanker) -> None:
        """Input dictionary is not mutated during ranking."""
        scores = {1: 0.3, 2: 0.9, 3: 0.6}
        scores_copy = copy.deepcopy(scores)
        ranker.rank(scores, top_k=2)
        assert scores == scores_copy
