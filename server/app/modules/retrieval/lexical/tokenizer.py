"""Deterministic Unicode-aware tokenizer for lexical retrieval.

Tokenization Policy:
1. Unicode Normalization:
   The complete raw text is normalized using Unicode Canonical Composition (NFC)
   prior to token scanning. This guarantees that decomposed combining characters
   (such as 'e' followed by combining acute accent U+0301) are composed into single
   atomic characters ('é') before regex matching, preventing accents from being
   detached or discarded as punctuation.

2. Compound Hyphenated Technical Terms:
   Internal hyphenated expressions representing technical terminology, architectures,
   and domain phrases are preserved as single tokens:
   - "state-of-the-art", "self-attention", "TF-IDF", "GPT-4", "cross-entropy",
     "zero-shot", "word-level", "VGG-16".
   Leading and trailing hyphens (e.g. "--option", "pre-") are treated as delimiters
   and stripped.

3. Contractions and Possessives:
   Apostrophe-bearing tokens (both ASCII straight quote "'" and Unicode typographic
   apostrophe "’" U+2019) are extracted intact:
   - Contractions: "don't", "doesn't", "it's", "can't", "won't".
   - Possessives: "transformer's", "attention’s", "authors'".
   Apostrophes are retained on the tokens at this stage so that downstream
   normalization can deliberately disambiguate recognized contractions from
   possessives without premature truncation.

4. Punctuation as Delimiter:
   All standard punctuation marks (periods, commas, colons, semicolons, exclamation
   marks, question marks, parentheses, brackets, braces, quotation marks, and math
   operators) act strictly as token delimiters. Punctuation characters are discarded
   and never emitted as searchable tokens.

5. Numbers and Alphanumeric Identifiers:
   Standalone integers and numbers ("2024", "100k") and alphanumeric terms
   ("BERT", "Word2Vec", "CNN", "RNN", "ResNet50", "3D") are preserved as tokens.

Limitations:
- Mathematical equations: Operators and symbols (e.g., "+", "=", "∑") act as delimiters.
- URLs and email addresses: Punctuation inside URLs (e.g., "https://arxiv.org/abs/1706.03762")
  is treated as delimiters, splitting the URL into component tokens.
- Acronyms with internal periods: Terms like "e.g." or "i.e." are split into "e" and "g".
"""

from __future__ import annotations

import re
import unicodedata

# Token regex pattern:
# 1. Hyphenated compound terms: \b\w+(?:-\w+)+\b
# 2. Contractions and internal apostrophes: \b\w+(?:['\u2019]\w+)+\b
# 3. Trailing possessive apostrophes: \b\w+['\u2019]
# 4. Standard Unicode words and alphanumeric numbers: \b\w+\b
TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"""
    \b\w+(?:-\w+)+\b            # Internal hyphenated compound terms (e.g. self-attention, GPT-4)
    |
    \b\w+(?:['\u2019]\w+)+\b    # Words with internal apostrophes (e.g. don't, transformer's)
    |
    \b\w+['\u2019]              # Words with trailing possessive apostrophes (e.g. authors')
    |
    \b\w+\b                     # Standard Unicode words, acronyms, and integers
    """,
    re.VERBOSE | re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    """Tokenize input text into a deterministic list of lexical tokens.

    Args:
        text: Input string to tokenize.

    Returns:
        List of extracted token strings. Empty list if input is empty or whitespace.

    Raises:
        TypeError: If input text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")

    if not text.strip():
        return []

    # Step 1: Pre-tokenize Unicode NFC normalization to compose combining characters
    normalized_text = unicodedata.normalize("NFC", text)

    # Step 2: Extract tokens using compiled Unicode-aware regex
    return TOKEN_PATTERN.findall(normalized_text)


class Tokenizer:
    """Configurable tokenizer instance for lexical retrieval pipelines."""

    def __call__(self, text: str) -> list[str]:
        """Tokenize input text."""
        return tokenize(text)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize input text."""
        return tokenize(text)
