"""Conservative English stopword collection and filtering for research-paper retrieval.

Policy:
1. Conservative Grammatical Scope:
   Only high-frequency English closed-class grammatical words (articles, prepositions,
   coordinating conjunctions, pronouns, auxiliary verbs, and common function adverbs)
   are included in DEFAULT_STOPWORDS.

2. Strict Negation Preservation:
   Negations ("not", "no", "nor", "neither", "never") carry vital discriminating
   power in scientific literature (e.g., "not autoregressive", "no recurrence",
   "neither recurrent nor convolutional") and are strictly excluded from default
   stopwords.

3. Academic and Technical Vocabulary Preservation:
   Domain-specific academic and machine-learning terms must never be removed as
   stopwords:
   - "model", "method", "approach", "network", "learning", "algorithm",
     "data", "system", "research", "attention", "transformer", "embedding",
     "retrieval", "classification", "generation".

4. Extensibility & Normalization:
   StopwordFilter accepts optional custom stopwords. Custom stopwords are
   NFC-normalized and lowercased upon initialization, and matching is performed
   case-insensitively against incoming tokens.
"""

from __future__ import annotations

from typing import Sequence
import unicodedata

# Explicit, documented, conservative baseline English stopwords
DEFAULT_STOPWORDS: frozenset[str] = frozenset({
    # Articles
    "a",
    "an",
    "the",
    # Conjunctions (negations "nor" and "neither" are intentionally excluded)
    "and",
    "or",
    "but",
    "if",
    "while",
    "as",
    "than",
    "because",
    "since",
    "although",
    "though",
    "whether",
    # Prepositions
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "of",
    "off",
    "over",
    "under",
    "until",
    "onto",
    "upon",
    "within",
    "without",
    "via",
    "per",
    # Pronouns & Determiners
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "each",
    "every",
    "both",
    "all",
    "any",
    "some",
    "few",
    "more",
    "most",
    "other",
    "such",
    # Auxiliary & Copular Verbs
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "should",
    "could",
    "will",
    "shall",
    "can",
    "may",
    "might",
    "must",
    # Common Function Adverbs
    "there",
    "here",
    "when",
    "where",
    "why",
    "how",
    "then",
    "so",
    "too",
    "very",
    "just",
    "now",
    "also",
    "only",
})


class StopwordFilter:
    """Filter for removing conservative English stopwords in lexical retrieval."""

    def __init__(
        self,
        custom_stopwords: set[str] | Sequence[str] | None = None,
    ) -> None:
        """Initialize filter with default stopwords and optional custom additions.

        Args:
            custom_stopwords: Optional collection of additional stopword strings.
                These are normalized to lowercased Unicode NFC and merged with
                DEFAULT_STOPWORDS.

        Raises:
            TypeError: If custom_stopwords is not a collection of strings or contains
                non-string items.
        """
        # Independent mutable set starting from DEFAULT_STOPWORDS
        self._stopwords: set[str] = set(DEFAULT_STOPWORDS)

        if custom_stopwords is not None:
            if not isinstance(custom_stopwords, (set, frozenset, list, tuple)):
                raise TypeError(
                    f"custom_stopwords must be a set, frozenset, list, or tuple, got {type(custom_stopwords).__name__}"
                )
            for item in custom_stopwords:
                if not isinstance(item, str):
                    raise TypeError(f"custom_stopwords items must be str, got {type(item).__name__}")
                normalized = unicodedata.normalize(
                                "NFC",
                                unicodedata.normalize("NFC", item.strip()).lower(),
                            )
                if normalized:
                    self._stopwords.add(normalized)

    @property
    def stopwords(self) -> frozenset[str]:
        """Return an immutable view of active stopwords."""
        return frozenset(self._stopwords)

    def is_stopword(self, token: str) -> bool:
        """Check whether a token is a stopword (case-insensitive, NFC-normalized).

        Args:
            token: String token to test.

        Returns:
            True if token is recognized as a stopword, False otherwise.

        Raises:
            TypeError: If token is not a string.
        """
        if not isinstance(token, str):
            raise TypeError(f"token must be a str, got {type(token).__name__}")
        return unicodedata.normalize("NFC", unicodedata.normalize("NFC", token.strip()).lower(),) in self._stopwords

    def filter_tokens(self, tokens: Sequence[str]) -> list[str]:
        """Filter out stopwords from a token sequence.

        Preserves the original order and casing of all non-stopword tokens.

        Args:
            tokens: Sequence of token strings.

        Returns:
            Fresh list containing only non-stopword tokens.

        Raises:
            TypeError: If tokens is not a list or tuple, or contains non-string elements.
        """
        if not isinstance(tokens, (list, tuple)):
            raise TypeError(f"tokens must be a list or tuple of strings, got {type(tokens).__name__}")

        result: list[str] = []
        for i, token in enumerate(tokens):
            if not isinstance(token, str):
                raise TypeError(f"tokens[{i}] must be a str, got {type(token).__name__}")
            if not self.is_stopword(token):
                result.append(token)
        return result


def remove_stopwords(
    tokens: Sequence[str],
    custom_stopwords: set[str] | Sequence[str] | None = None,
) -> list[str]:
    """Convenience function to filter stopwords from a token sequence.

    Args:
        tokens: Sequence of token strings.
        custom_stopwords: Optional additional stopwords to include.

    Returns:
        Fresh list of non-stopword tokens.
    """
    filterer = StopwordFilter(custom_stopwords=custom_stopwords)
    return filterer.filter_tokens(tokens)
