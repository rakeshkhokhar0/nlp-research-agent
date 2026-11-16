"""Document ranking and top-K selection for lexical retrieval.

Accepts precomputed document similarity scores, validates document IDs and scores,
orders documents descending by score with deterministic ascending document ID
tie-breaking, and selects the top-K candidates.
"""

from __future__ import annotations

import math
from typing import Mapping


def _validate_top_k(top_k: int) -> None:
    """Validate that top_k is a positive integer and not a boolean."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be a positive int, got {type(top_k).__name__}")
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer (> 0), got {top_k}")


def _validate_similarity_scores(similarity_scores: Mapping[int, float]) -> None:
    """Validate that similarity_scores mapping is well-formed.

    Args:
        similarity_scores: Dictionary mapping document ID to similarity score.

    Raises:
        TypeError: If similarity_scores is not a dict, keys are not ints (or are bools),
            or values are not ints or floats (or are bools).
        ValueError: If document IDs are not positive, or scores are negative, NaN,
            or infinite.
    """
    if not isinstance(similarity_scores, dict):
        raise TypeError(
            f"similarity_scores must be a dict, got {type(similarity_scores).__name__}"
        )

    for doc_id, score in similarity_scores.items():
        if isinstance(doc_id, bool) or not isinstance(doc_id, int):
            raise TypeError(
                f"Document ID must be an int, got {type(doc_id).__name__}"
            )
        if doc_id <= 0:
            raise ValueError(
                f"Document ID must be a positive integer (> 0), got {doc_id}"
            )

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(
                f"Similarity score for document {doc_id} must be numeric (int or float), got {type(score).__name__}"
            )
        try:
            float_score = float(score)
        except OverflowError:
            raise ValueError(
                f"Similarity score for document {doc_id} is too large to convert to a finite float"
            )

        if not math.isfinite(float_score):
            raise ValueError(
                f"Similarity score for document {doc_id} must be finite, got {score}"
            )
        if float_score < 0.0:
            raise ValueError(
                f"Similarity score for document {doc_id} must be non-negative, got {score}"
            )


class DocumentRanker:
    """Ranks documents by similarity score and extracts top-K matches."""

    def rank(
        self,
        similarity_scores: dict[int, float],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Rank documents by similarity score descending with document ID tie-breaking.

        Args:
            similarity_scores: Dictionary mapping document ID to similarity score.
            top_k: Maximum number of ranked results to return (default: 5).

        Returns:
            List of (document_id, score) tuples sorted by score descending,
            with ties broken by document_id ascending. Returns [] for empty input.

        Raises:
            TypeError: If similarity_scores is not a dict, top_k is not an int (or is bool),
                or keys/values have invalid types.
            ValueError: If top_k <= 0, document IDs <= 0, or scores are negative/NaN/inf.
        """
        _validate_similarity_scores(similarity_scores)
        _validate_top_k(top_k)

        if not similarity_scores:
            return []

        # Sort: primary descending by score, secondary ascending by doc_id
        sorted_items = sorted(
            similarity_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        return [(doc_id, float(score)) for doc_id, score in sorted_items[:top_k]]


def rank_documents(
    similarity_scores: dict[int, float],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Convenience function to rank documents using DocumentRanker."""
    return DocumentRanker().rank(similarity_scores, top_k=top_k)
