"""Normalization, contraction expansion, and possessive handling for lexical retrieval.

Policy:
1. Canonical Unicode Normalization & Lowercasing:
   All tokens are normalized using Unicode Canonical Composition (NFC) and
   converted to lowercase. Structural components (digits, internal hyphens in
   terms like "gpt-4", "tf-idf", "state-of-the-art") are preserved intact.

2. Contraction Expansion:
   Recognized English contractions with both ASCII "'" and typographic "’" (U+2019)
   apostrophes are expanded into constituent words before stopword filtering.
   This guarantees that negations ("not") are isolated and preserved for accurate
   semantic matching, while auxiliaries ("does", "is", "have") can be filtered
   by downstream stopword removal.
   Examples:
   - "don't" / "don’t" -> ["do", "not"]
   - "doesn't" / "doesn’t" -> ["does", "not"]
   - "can't" / "can’t" -> ["can", "not"]
   - "won't" / "won’t" -> ["will", "not"]
   - "it's" / "it’s" -> ["it", "is"]

3. Possessive vs Contraction Disambiguation:
   Contractions are matched first against the explicit CONTRACTION_MAP.
   Tokens that are NOT recognized contractions but have possessive endings:
   - Trailing "'s" or "’s" (e.g. "transformer's", "attention’s") -> suffix stripped
     to yield base token ("transformer", "attention").
   - Trailing apostrophe "'" or "’" (e.g. "authors'", "transformers’") -> apostrophe
     stripped to yield plural base ("authors", "transformers").

4. Ambiguous Contraction Preservation:
   Ambiguous contractions such as "'d" ("he'd", "they'd", "we'd", which can mean
   either "would" or "had") are intentionally unsupported for expansion to prevent
   speculative distortion in search queries. They are returned unchanged.
"""

from __future__ import annotations

from typing import Sequence
import unicodedata

# Base mapping of ASCII contraction pairs
_BASE_CONTRACTIONS: dict[str, list[str]] = {
    # Negations
    "don't": ["do", "not"],
    "doesn't": ["does", "not"],
    "didn't": ["did", "not"],
    "can't": ["can", "not"],
    "cannot": ["can", "not"],
    "couldn't": ["could", "not"],
    "won't": ["will", "not"],
    "wouldn't": ["would", "not"],
    "isn't": ["is", "not"],
    "aren't": ["are", "not"],
    "wasn't": ["was", "not"],
    "weren't": ["were", "not"],
    "haven't": ["have", "not"],
    "hasn't": ["has", "not"],
    "hadn't": ["had", "not"],
    "shouldn't": ["should", "not"],
    "mustn't": ["must", "not"],
    # Pronouns & Auxiliaries
    "it's": ["it", "is"],
    "that's": ["that", "is"],
    "there's": ["there", "is"],
    "here's": ["here", "is"],
    "what's": ["what", "is"],
    "who's": ["who", "is"],
    "where's": ["where", "is"],
    "when's": ["when", "is"],
    "why's": ["why", "is"],
    "how's": ["how", "is"],
    "let's": ["let", "us"],
    "i'm": ["i", "am"],
    "they're": ["they", "are"],
    "we're": ["we", "are"],
    "you're": ["you", "are"],
    "i've": ["i", "have"],
    "they've": ["they", "have"],
    "we've": ["we", "have"],
    "you've": ["you", "have"],
    "i'll": ["i", "will"],
    "they'll": ["they", "will"],
    "we'll": ["we", "will"],
    "you'll": ["you", "will"],
    "it'll": ["it", "will"],
    "that'll": ["that", "will"],
}

# Populate full CONTRACTION_MAP including typographic apostrophe (U+2019 ’) variants
CONTRACTION_MAP: dict[str, list[str]] = {}
for contraction, expansion in _BASE_CONTRACTIONS.items():
    CONTRACTION_MAP[contraction] = expansion.copy()
    if "'" in contraction:
        curly_variant = contraction.replace("'", "\u2019")
        CONTRACTION_MAP[curly_variant] = expansion.copy()


def normalize_token(token: str) -> str:
    """Normalize a single token to lowercased Unicode NFC string.

    Args:
        token: Input token string.

    Returns:
        Lowercased NFC normalized token with outer whitespace stripped.

    Raises:
        TypeError: If token is not a string.
    """
    if not isinstance(token, str):
        raise TypeError(f"token must be a str, got {type(token).__name__}")
    return unicodedata.normalize(
        "NFC",
        unicodedata.normalize("NFC", token.strip()).lower(),
    )


def normalize_tokens(tokens: Sequence[str]) -> list[str]:
    """Normalize a sequence of tokens to lowercased Unicode NFC strings.

    Args:
        tokens: Sequence of token strings.

    Returns:
        New list of normalized tokens (excluding any empty strings).

    Raises:
        TypeError: If tokens is not a sequence or contains non-string items.
    """
    if not isinstance(tokens, (list, tuple)):
        raise TypeError(f"tokens must be a list or tuple of strings, got {type(tokens).__name__}")

    result: list[str] = []
    for i, token in enumerate(tokens):
        if not isinstance(token, str):
            raise TypeError(f"tokens[{i}] must be a str, got {type(token).__name__}")
        norm = normalize_token(token)
        if norm:
            result.append(norm)
    return result


def expand_contractions_and_possessives(tokens: Sequence[str]) -> list[str]:
    """Expand recognized contractions and selectively strip possessive suffixes.

    Processing logic:
    1. Lowercase and NFC-normalize each token.
    2. If the token is found in CONTRACTION_MAP (e.g. "don't", "it's"), expand into
       constituent tokens (e.g. ["do", "not"], ["it", "is"]).
    3. If not a contraction, check for possessive endings:
       - Strip "'s" or "’s" (e.g. "transformer's" -> "transformer").
       - Strip trailing apostrophe "'" or "’" (e.g. "authors'" -> "authors").
    4. Ambiguous contractions (e.g. "he'd", "they'd") remain unchanged.

    Args:
        tokens: Sequence of token strings to process.

    Returns:
        Fresh list of expanded and normalized tokens.

    Raises:
        TypeError: If tokens is not a sequence or contains non-string items.
    """
    if not isinstance(tokens, (list, tuple)):
        raise TypeError(f"tokens must be a list or tuple of strings, got {type(tokens).__name__}")

    expanded: list[str] = []
    for i, token in enumerate(tokens):
        if not isinstance(token, str):
            raise TypeError(f"tokens[{i}] must be a str, got {type(token).__name__}")

        norm = normalize_token(token)
        if not norm:
            continue

        # Step 1: Check recognized contractions
        if norm in CONTRACTION_MAP:
            expanded.extend(CONTRACTION_MAP[norm])
        # Step 2: Strip possessive 's / ’s (if stem is at least 1 character)
        elif len(norm) > 2 and (norm.endswith("'s") or norm.endswith("\u2019s")):
            expanded.append(norm[:-2])
        # Step 3: Strip trailing possessive apostrophe (if stem is at least 1 character)
        elif len(norm) > 1 and (norm.endswith("'") or norm.endswith("\u2019")):
            expanded.append(norm[:-1])
        # Step 4: Keep unchanged (includes technical tokens, normal words, ambiguous contractions)
        else:
            expanded.append(norm)

    return expanded
