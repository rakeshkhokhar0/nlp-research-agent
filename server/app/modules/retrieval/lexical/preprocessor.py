"""Text preprocessing pipeline orchestrator for lexical retrieval.

Coordinates Unicode tokenization, normalization (lowercasing, contraction expansion,
and selective possessive stripping), conservative stopword filtering, and Porter
stemming into a unified, deterministic pipeline used identically for indexing
research-paper chunks and evaluating search queries.
"""

from __future__ import annotations

from typing import Optional

from app.modules.retrieval.lexical.normalizer import expand_contractions_and_possessives
from app.modules.retrieval.lexical.stemmer import PorterStemmer
from app.modules.retrieval.lexical.stopwords import StopwordFilter
from app.modules.retrieval.lexical.tokenizer import Tokenizer


class TextPreprocessor:
    """Orchestrates tokenization, normalization, stopword filtering, and stemming."""

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        stopword_filter: Optional[StopwordFilter] = None,
        stemmer: Optional[PorterStemmer] = None,
    ) -> None:
        """Initialize the text preprocessor with pipeline components.

        Args:
            tokenizer: Optional Tokenizer instance. If None, a default Tokenizer is created.
            stopword_filter: Optional StopwordFilter instance. If None, a default StopwordFilter is created.
            stemmer: Optional PorterStemmer instance. If None, a default PorterStemmer is created.

        Raises:
            TypeError: If any provided component is not an instance of the expected class.
        """
        if tokenizer is not None and not isinstance(tokenizer, Tokenizer):
            raise TypeError(
                f"tokenizer must be an instance of Tokenizer, got {type(tokenizer).__name__}"
            )
        if stopword_filter is not None and not isinstance(stopword_filter, StopwordFilter):
            raise TypeError(
                f"stopword_filter must be an instance of StopwordFilter, got {type(stopword_filter).__name__}"
            )
        if stemmer is not None and not isinstance(stemmer, PorterStemmer):
            raise TypeError(
                f"stemmer must be an instance of PorterStemmer, got {type(stemmer).__name__}"
            )

        self._tokenizer: Tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self._stopword_filter: StopwordFilter = (
            stopword_filter if stopword_filter is not None else StopwordFilter()
        )
        self._stemmer: PorterStemmer = stemmer if stemmer is not None else PorterStemmer()

    def preprocess(self, text: str) -> list[str]:
        """Preprocess raw text through the full lexical pipeline.

        Pipeline stages:
        1. Tokenization: Unicode NFC normalization, hyphenated compound term extraction,
           and delimiter splitting.
        2. Normalization: Lowercase conversion, contraction expansion (e.g. doesn't -> does, not),
           and selective possessive suffix stripping.
        3. Stopword Filtering: Conservative grammatical stopword removal while strictly
           preserving negations (not, no, nor, neither, never) and research terms.
        4. Porter Stemming: Pure Python Porter 1980 stemming with technical term, digit,
           and hyphen guards.

        Args:
            text: Raw input string (document chunk or search query).

        Returns:
            Fresh list of processed lexical token strings.

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")

        if not text.strip():
            return []

        # Stage 1: Tokenize raw text
        tokens = self._tokenizer.tokenize(text)
        if not tokens:
            return []

        # Stage 2: Lowercase, expand contractions, and strip possessives
        normalized = expand_contractions_and_possessives(tokens)
        if not normalized:
            return []

        # Stage 3: Remove closed-class grammatical stopwords
        filtered = self._stopword_filter.filter_tokens(normalized)
        if not filtered:
            return []

        # Stage 4: Apply Porter stemming with guards
        stemmed = self._stemmer.stem_tokens(filtered)

        return stemmed

    def __call__(self, text: str) -> list[str]:
        """Delegate directly to preprocess().

        Args:
            text: Raw input string.

        Returns:
            Fresh list of processed lexical token strings.
        """
        return self.preprocess(text)


_DEFAULT_PREPROCESSOR = TextPreprocessor()


def preprocess(text: str) -> list[str]:
    """Convenience function to preprocess text using the default TextPreprocessor.

    Args:
        text: Raw input string (document chunk or search query).

    Returns:
        Fresh list of processed lexical token strings.
    """
    return _DEFAULT_PREPROCESSOR.preprocess(text)
