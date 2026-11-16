"""Inverted index implementation for lexical text retrieval.

Maintains bidirectional mappings between vocabulary terms and document IDs,
utilizing the custom TextPreprocessor pipeline for identical tokenization,
normalization, stopword filtering, and stemming across documents and search terms.
"""

from __future__ import annotations

from typing import Optional

from app.modules.retrieval.lexical.preprocessor import TextPreprocessor


def _validate_document_id(document_id: int) -> None:
    """Validate that document_id is a positive integer and not a boolean."""
    if isinstance(document_id, bool) or not isinstance(document_id, int):
        raise TypeError(
            f"document_id must be a positive int, got {type(document_id).__name__}"
        )
    if document_id <= 0:
        raise ValueError(
            f"document_id must be a positive integer (> 0), got {document_id}"
        )


class InvertedIndex:
    """Inverted index data structure for fast lexical term retrieval."""

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None) -> None:
        """Initialize the inverted index.

        Args:
            preprocessor: Optional TextPreprocessor instance. If None,
                a default TextPreprocessor is instantiated.

        Raises:
            TypeError: If preprocessor is not an instance of TextPreprocessor.
        """
        if preprocessor is not None and not isinstance(preprocessor, TextPreprocessor):
            raise TypeError(
                f"preprocessor must be an instance of TextPreprocessor, got {type(preprocessor).__name__}"
            )

        self._preprocessor: TextPreprocessor = (
            preprocessor if preprocessor is not None else TextPreprocessor()
        )
        self.index: dict[str, set[int]] = {}
        self.document_terms: dict[int, set[str]] = {}
        self.document_term_frequencies: dict[int, dict[str, int]] = {}

    @property
    def preprocessor(self) -> TextPreprocessor:
        """Return the configured TextPreprocessor instance."""
        return self._preprocessor

    def add_document(self, document_id: int, text: str) -> None:
        """Add or replace a document in the inverted index.

        Preprocesses text into unique tokens and term frequencies, updates term
        postings, and maintains document-to-term tracking. If document_id already
        exists, previous postings and frequencies are cleanly removed only after
        successful preprocessing to eliminate stale terms without risking partial
        state if preprocessing fails.

        Args:
            document_id: Positive integer document identifier.
            text: Raw document text to index.

        Raises:
            TypeError: If document_id is not an int (or is a bool) or text is not a str.
            ValueError: If document_id is not positive (> 0).
        """
        _validate_document_id(document_id)
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")

        # Preprocess text into lexical tokens
        tokens = self._preprocessor.preprocess(text)
        unique_terms = set(tokens)

        # Count frequencies of preprocessed terms
        term_frequencies: dict[str, int] = {}
        for token in tokens:
            term_frequencies[token] = term_frequencies.get(token, 0) + 1

        # If document already exists, remove it only after successful preprocessing
        if document_id in self.document_terms:
            self.remove_document(document_id)

        # Register document terms and frequencies
        self.document_terms[document_id] = unique_terms
        self.document_term_frequencies[document_id] = term_frequencies

        # Update inverted index postings
        for term in unique_terms:
            if term not in self.index:
                self.index[term] = set()
            self.index[term].add(document_id)

    def remove_document(self, document_id: int) -> None:
        """Remove a document from the inverted index.

        Removes document_id from all term postings, deletes terms whose posting
        sets become empty, and removes the document from the reverse mapping and
        term frequency tracking. Safe and idempotent if document_id does not exist.

        Args:
            document_id: Positive integer document identifier.

        Raises:
            TypeError: If document_id is not an int (or is a bool).
            ValueError: If document_id is not positive (> 0).
        """
        _validate_document_id(document_id)

        if document_id not in self.document_terms:
            return

        self.document_term_frequencies.pop(document_id, None)
        terms = self.document_terms.pop(document_id)
        for term in terms:
            postings = self.index.get(term)
            if postings is not None:
                postings.discard(document_id)
                if not postings:
                    del self.index[term]

    def search(self, term: str, preprocessed: bool = False) -> set[int]:
        """Search for documents containing the specified term.

        Args:
            term: Single search query term string.
            preprocessed: If True, term is treated as an already-preprocessed posting
                key and looked up directly without preprocessing. If False (default),
                term is preprocessed using the index's TextPreprocessor.

        Returns:
            A new set of matching document IDs (empty set if no match or if
            query produces zero terms).

        Raises:
            TypeError: If term is not a string, or preprocessed is not a bool.
            ValueError: If query produces multiple preprocessed terms.
        """
        if not isinstance(term, str):
            raise TypeError(f"term must be a str, got {type(term).__name__}")
        if not isinstance(preprocessed, bool):
            raise TypeError(
                f"preprocessed must be a bool, got {type(preprocessed).__name__}"
            )

        if preprocessed:
            postings = self.index.get(term)
            if postings is None:
                return set()
            return set(postings)

        tokens = self._preprocessor.preprocess(term)
        if len(tokens) > 1:
            raise ValueError(
                f"Search query must produce at most one term, got {len(tokens)}: {tokens}"
            )
        if len(tokens) == 0:
            return set()

        processed_term = tokens[0]
        postings = self.index.get(processed_term)
        if postings is None:
            return set()

        return set(postings)

    def get_document_frequency(self, term: str) -> int:
        """Return the document frequency (DF) for the specified term.

        Args:
            term: Single query term string.

        Returns:
            Number of unique documents containing the term (0 if unknown
            or if query produces zero terms).

        Raises:
            TypeError: If term is not a string.
            ValueError: If query produces multiple preprocessed terms.
        """
        if not isinstance(term, str):
            raise TypeError(f"term must be a str, got {type(term).__name__}")

        if term in self.index:
            return len(self.index[term])

        tokens = self._preprocessor.preprocess(term)
        if len(tokens) > 1:
            raise ValueError(
                f"Query must produce at most one term, got {len(tokens)}: {tokens}"
            )
        if len(tokens) == 0:
            return 0

        processed_term = tokens[0]
        postings = self.index.get(processed_term)
        return len(postings) if postings is not None else 0

    def get_term_frequency(self, term: str, document_id: int) -> int:
        """Return the raw term frequency (TF) for a term within a document.

        Preprocesses the term using the index's TextPreprocessor and returns
        its frequency in the specified document.

        Args:
            term: Single query term string.
            document_id: Positive integer document identifier.

        Returns:
            The raw term frequency count in the document (0 if document or term is
            unknown or if query produces zero terms).

        Raises:
            TypeError: If term is not a string, or document_id is not an int (or is bool).
            ValueError: If document_id is not positive (> 0), or term produces multiple tokens.
        """
        _validate_document_id(document_id)
        if not isinstance(term, str):
            raise TypeError(f"term must be a str, got {type(term).__name__}")

        doc_frequencies = self.document_term_frequencies.get(document_id)
        if doc_frequencies is None:
            return 0

        if term in doc_frequencies:
            return doc_frequencies[term]

        tokens = self._preprocessor.preprocess(term)
        if len(tokens) > 1:
            raise ValueError(
                f"Query must produce at most one term, got {len(tokens)}: {tokens}"
            )
        if len(tokens) == 0:
            return 0

        processed_term = tokens[0]
        return doc_frequencies.get(processed_term, 0)

    def get_document_term_frequencies(self, document_id: int) -> dict[str, int]:
        """Return a copy of the term frequencies dictionary for a document.

        Args:
            document_id: Positive integer document identifier.

        Returns:
            A new dictionary mapping preprocessed terms to their integer frequencies
            within the document (empty dictionary if document is unknown).

        Raises:
            TypeError: If document_id is not an int (or is a bool).
            ValueError: If document_id is not positive (> 0).
        """
        _validate_document_id(document_id)
        doc_frequencies = self.document_term_frequencies.get(document_id)
        if doc_frequencies is None:
            return {}
        return dict(doc_frequencies)

    def clear(self) -> None:
        """Clear all indexed documents, postings, and term frequencies."""
        self.index.clear()
        self.document_terms.clear()
        self.document_term_frequencies.clear()

    def __len__(self) -> int:
        """Return the number of documents currently indexed."""
        return len(self.document_terms)
