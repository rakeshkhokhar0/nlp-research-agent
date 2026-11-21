from __future__ import annotations

from collections import Counter
import math

from app.modules.retrieval.lexical.inverted_index import InvertedIndex


class TfidfCalculator:
    """Calculates TF-IDF weights and normalized sparse vector representations for retrieval."""

    def __init__(self, index: InvertedIndex) -> None:
        if not isinstance(index, InvertedIndex):
            raise TypeError(
                f"index must be an instance of InvertedIndex, got {type(index).__name__}"
            )
        self._index: InvertedIndex = index

    @property
    def index(self) -> InvertedIndex:
        """Return the underlying InvertedIndex instance."""
        return self._index

    def compute_idf(self, term: str) -> float:
        if not isinstance(term, str):
            raise TypeError(f"term must be a str, got {type(term).__name__}")

        df = self._index.get_document_frequency(term)
        n = len(self._index)

        return math.log((n + 1.0) / (df + 1.0)) + 1.0

    def compute_tf(self, term: str, document_id: int) -> int:
        return self._index.get_term_frequency(term, document_id)

    def compute_weight(self, term: str, document_id: int) -> float:
        tf = self.compute_tf(term, document_id)
        idf = self.compute_idf(term)
        return float(tf * idf)

    def transform_document(self, document_id: int) -> dict[str, float]:
        term_frequencies = self._index.get_document_term_frequencies(document_id)
        if not term_frequencies:
            return {}

        raw_weights: dict[str, float] = {}
        sum_squares = 0.0

        for term, tf in term_frequencies.items():
            idf = self.compute_idf(term)
            weight = tf * idf
            raw_weights[term] = weight
            sum_squares += weight * weight

        norm = math.sqrt(sum_squares)
        if norm == 0.0:
            return {}

        return {term: weight / norm for term, weight in raw_weights.items()}

    def transform_query(self, query: str) -> dict[str, float]:
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")

        tokens = self._index.preprocessor.preprocess(query)
        if not tokens:
            return {}

        term_counts = Counter(tokens)
        raw_weights: dict[str, float] = {}
        sum_squares = 0.0

        for term, tf in term_counts.items():
            idf = self.compute_idf(term)
            weight = tf * idf
            raw_weights[term] = weight
            sum_squares += weight * weight

        norm = math.sqrt(sum_squares)
        if norm == 0.0:
            return {}

        return {term: weight / norm for term, weight in raw_weights.items()}
