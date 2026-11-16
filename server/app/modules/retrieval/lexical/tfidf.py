"""TF-IDF (Term Frequency-Inverse Document Frequency) calculator for lexical retrieval.

Implements term weighting and L2-normalized vector generation for documents and queries
based on an underlying dynamic InvertedIndex.

Mathematical Specification:
1. Term Frequency (raw):
   TF(t, d) = frequency of term t in document d

2. Inverse Document Frequency (smoothed):
   IDF(t) = log((N + 1) / (DF(t) + 1)) + 1
   where:
   - N is the total number of indexed documents in the corpus
   - DF(t) is the document frequency (number of documents containing term t)
   - log is the natural logarithm (math.log)

3. Composite TF-IDF Weight:
   weight(t, d) = TF(t, d) * IDF(t)

4. Vector Normalization (L2 norm / Euclidean length):
   ||v||_2 = sqrt(sum(w_i^2))
   v_norm_i = w_i / ||v||_2
"""

from __future__ import annotations

from collections import Counter
import math

from app.modules.retrieval.lexical.inverted_index import InvertedIndex


class TfidfCalculator:
    """Calculates TF-IDF weights and normalized sparse vector representations for retrieval."""

    def __init__(self, index: InvertedIndex) -> None:
        """Initialize the TF-IDF calculator with an inverted index.

        Args:
            index: InvertedIndex instance containing indexed documents and vocabulary.

        Raises:
            TypeError: If index is not an instance of InvertedIndex.
        """
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
        """Compute the smoothed inverse document frequency (IDF) for a single term.

        Formula:
            IDF(t) = log((N + 1) / (DF(t) + 1)) + 1
        where N is the total number of indexed documents, DF(t) is the number of
        documents containing the preprocessed term, and log is the natural logarithm.

        Args:
            term: Single query term string.

        Returns:
            The smoothed IDF value as a float.

        Raises:
            TypeError: If term is not a string.
            ValueError: If term produces multiple preprocessed terms.
        """
        if not isinstance(term, str):
            raise TypeError(f"term must be a str, got {type(term).__name__}")

        df = self._index.get_document_frequency(term)
        n = len(self._index)

        return math.log((n + 1.0) / (df + 1.0)) + 1.0

    def compute_tf(self, term: str, document_id: int) -> int:
        """Compute the raw term frequency (TF) for a term in a document.

        Formula:
            TF(t, d) = frequency of term t in document d

        Args:
            term: Single query term string.
            document_id: Positive integer document identifier.

        Returns:
            Raw term frequency count in the document (0 if unknown document or term).

        Raises:
            TypeError: If term is not a string, or document_id is not an int (or is bool).
            ValueError: If document_id is not positive (> 0), or term produces multiple tokens.
        """
        return self._index.get_term_frequency(term, document_id)

    def compute_weight(self, term: str, document_id: int) -> float:
        """Compute the unnormalized TF-IDF weight for a term in a document.

        Formula:
            weight(t, d) = TF(t, d) * IDF(t)

        Args:
            term: Single query term string.
            document_id: Positive integer document identifier.

        Returns:
            The product of raw TF and smoothed IDF as a float (0.0 if TF is 0).

        Raises:
            TypeError: If term is not a string, or document_id is not an int (or is bool).
            ValueError: If document_id is not positive (> 0), or term produces multiple tokens.
        """
        tf = self.compute_tf(term, document_id)
        idf = self.compute_idf(term)
        return float(tf * idf)

    def transform_document(self, document_id: int) -> dict[str, float]:
        """Transform an indexed document into an L2-normalized TF-IDF vector.

        Retrieves the document's term frequencies, computes TF-IDF weights using
        the current corpus IDF values, and applies L2 normalization.

        Args:
            document_id: Positive integer document identifier.

        Returns:
            Dictionary mapping preprocessed terms to their L2-normalized weights.
            Returns {} if the document is empty or unknown.

        Raises:
            TypeError: If document_id is not an int (or is a bool).
            ValueError: If document_id is not positive (> 0).
        """
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
        """Transform a text query into an L2-normalized TF-IDF vector.

        Preprocesses the query using the inverted index's TextPreprocessor instance,
        counts query term frequencies, computes TF-IDF weights against current corpus
        IDF values, and applies L2 normalization.

        Args:
            query: Raw query text string.

        Returns:
            Dictionary mapping preprocessed terms to their L2-normalized weights.
            Returns {} if query produces zero terms (e.g. empty or stopword-only).

        Raises:
            TypeError: If query is not a string.
        """
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
