"""Pure Python implementation of Martin Porter's 1980 Stemming Algorithm.

Reference:
    Porter, M.F. (1980). "An algorithm for suffix stripping."
    Program: automated library and information systems, 14(3), pp. 130-137.

This implementation faithfully reproduces the original five-step algorithm
with the official Step 2 amendment (logi -> log), while adding critical
engineering guards for technical terminology, hyphenated terms, numeric
tokens, and unresolved contractions.
"""

from collections.abc import Iterable, Sequence
from typing import Optional

DEFAULT_TECHNICAL_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "api",
        "bert",
        "cnn",
        "cpu",
        "css",
        "dbms",
        "gan",
        "gpt",
        "gpu",
        "html",
        "json",
        "knn",
        "llm",
        "lstm",
        "mlp",
        "nlp",
        "os",
        "rnn",
        "sql",
        "svm",
        "tf-idf",
        "word2vec",
        "xml",
    }
)

_STEP_2_RULES: tuple[tuple[str, str], ...] = (
    ("ization", "ize"),
    ("ational", "ate"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("biliti", "ble"),
    ("tional", "tion"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("ation", "ate"),
    ("entli", "ent"),
    ("ousli", "ous"),
    ("alism", "al"),
    ("abli", "able"),
    ("alli", "al"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("ator", "ate"),
    ("logi", "log"),
    ("eli", "e"),
)

_STEP_3_RULES: tuple[tuple[str, str], ...] = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)

_STEP_4_SUFFIXES: tuple[str, ...] = (
    "ement",
    "ance",
    "ence",
    "able",
    "ible",
    "ment",
    "ant",
    "ent",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
    "ion",
    "al",
    "er",
    "ic",
    "ou",
)


def _is_consonant(word: str, i: int) -> bool:
    """Check if character at index i is a consonant according to Porter 1980.

    A letter is a consonant if it is not A, E, I, O, U, and not Y preceded
    by a consonant.
    """
    ch = word[i]
    if ch in "aeiou":
        return False
    if ch == "y":
        if i == 0:
            return True
        return not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Calculate the measure m of a stem according to [C](VC)^m[V]."""
    count = 0
    i = 0
    length = len(stem)
    # Skip optional leading consonants [C]
    while i < length and _is_consonant(stem, i):
        i += 1
    while i < length:
        # Scan vowels V
        while i < length and not _is_consonant(stem, i):
            i += 1
        # Scan consonants C
        if i < length:
            count += 1
            while i < length and _is_consonant(stem, i):
                i += 1
    return count


def _contains_vowel(stem: str) -> bool:
    """*v* condition: Stem contains at least one vowel."""
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    """*d condition: Stem ends with a double consonant (e.g., -TT, -SS)."""
    if len(stem) < 2:
        return False
    return stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)


def _ends_cvc(stem: str) -> bool:
    """*o condition: Stem ends cvc, where the second c is not W, X, or Y."""
    if len(stem) < 3:
        return False
    return (
        _is_consonant(stem, len(stem) - 3)
        and not _is_consonant(stem, len(stem) - 2)
        and _is_consonant(stem, len(stem) - 1)
        and stem[-1] not in ("w", "x", "y")
    )


def _step_1a(word: str) -> str:
    """Step 1a: Plural and inflectional s handling."""
    if word.endswith("sses"):
        return word[:-4] + "ss"
    if word.endswith("ies"):
        return word[:-3] + "i"
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step_1b(word: str) -> str:
    """Step 1b: Past participle and progressive handling."""
    if word.endswith("eed"):
        stem = word[:-3]
        if _measure(stem) > 0:
            return stem + "ee"
        return word

    extra_rule = False
    stem = word
    if word.endswith("ed"):
        candidate = word[:-2]
        if _contains_vowel(candidate):
            stem = candidate
            extra_rule = True
    elif word.endswith("ing"):
        candidate = word[:-3]
        if _contains_vowel(candidate):
            stem = candidate
            extra_rule = True

    if extra_rule:
        if stem.endswith("at") or stem.endswith("bl") or stem.endswith("iz"):
            return stem + "e"
        if _ends_double_consonant(stem) and stem[-1] not in ("l", "s", "z"):
            return stem[:-1]
        if _measure(stem) == 1 and _ends_cvc(stem):
            return stem + "e"
        return stem

    return word


def _step_1c(word: str) -> str:
    """Step 1c: Terminal y -> i if stem contains a vowel."""
    if word.endswith("y"):
        stem = word[:-1]
        if _contains_vowel(stem):
            return stem + "i"
    return word


def _step_2(word: str) -> str:
    """Step 2: Relational suffixes (m > 0), including logi -> log."""
    for suffix, replacement in _STEP_2_RULES:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            # For logi -> log, verify measure on stem + 'l' (word[:-3])
            # following standard Porter logi extension (e.g. geologi -> geolog).
            cond = _measure(word[:-3]) > 0 if suffix == "logi" else _measure(stem) > 0
            if cond:
                return stem + replacement
            return word
    return word


def _step_3(word: str) -> str:
    """Step 3: Derivational suffixes (m > 0)."""
    for suffix, replacement in _STEP_3_RULES:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if _measure(stem) > 0:
                return stem + replacement
            return word
    return word


def _step_4(word: str) -> str:
    """Step 4: Suffix removal in measure m > 1."""
    for suffix in _STEP_4_SUFFIXES:
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if suffix == "ion":
                if _measure(stem) > 1 and len(stem) > 0 and stem[-1] in ("s", "t"):
                    return stem
            else:
                if _measure(stem) > 1:
                    return stem
            return word
    return word


def _step_5a(word: str) -> str:
    """Step 5a: Remove trailing e if measure conditions are met.

    - Remove terminal e if measure of the remaining stem is greater than 1.
    - Remove terminal e if measure is exactly 1 and remaining stem does not satisfy *o.
    - Otherwise retain the terminal e.
    """
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)

        if m > 1:
            return stem

        if m == 1 and not _ends_cvc(stem):
            return stem

    return word


def _step_5b(word: str) -> str:
    """Step 5b: Reduce -ll to -l if measure m > 1."""
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        return word[:-1]
    return word


class PorterStemmer:
    """Pure Python implementation of Martin Porter's 1980 Stemming Algorithm.

    Supports configurable technical-term/acronym exceptions, preserving
    token casing for protected terms while lowercasing normal stemmed words.
    """

    def __init__(
        self,
        technical_exceptions: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize the stemmer with technical exceptions.

        Args:
            technical_exceptions: Optional iterable of additional technical words
                and acronyms that should bypass stemming, added to DEFAULT_TECHNICAL_EXCEPTIONS.

        Raises:
            TypeError: If technical_exceptions is not an iterable of strings.
        """
        # Start with an independent copy of DEFAULT_TECHNICAL_EXCEPTIONS
        self._technical_exceptions: set[str] = set(DEFAULT_TECHNICAL_EXCEPTIONS)

        # If custom exceptions are supplied, validate and merge them
        if technical_exceptions is not None:
            if not isinstance(technical_exceptions, Iterable) or isinstance(
                technical_exceptions, (str, bytes)
            ):
                raise TypeError(
                    "technical_exceptions must be an iterable of strings"
                )
            for item in technical_exceptions:
                if not isinstance(item, str):
                    raise TypeError(
                        f"technical_exception item must be a string, got {type(item).__name__}"
                    )
                self._technical_exceptions.add(item.lower().strip())

    @property
    def technical_exceptions(self) -> frozenset[str]:
        """Return the immutable set of technical exceptions."""
        return frozenset(self._technical_exceptions)

    def stem(self, word: str) -> str:
        """Stem a single word following Porter 1980 with guards.

        Args:
            word: The input string word/token.

        Returns:
            The stemmed token (lowercase for stemmed words, original casing for protected tokens).

        Raises:
            TypeError: If word is not a string.
        """
        if not isinstance(word, str):
            raise TypeError(f"word must be a string, got {type(word).__name__}")

        # Guard 1: Tokens with length <= 2 are returned unchanged
        if len(word) <= 2:
            return word

        # Guard 2: Tokens containing digits (e.g., gpt-4, word2vec, 2024, 3d, v100)
        if any(char.isdigit() for char in word):
            return word

        # Guard 3: Tokens containing internal hyphens (e.g., state-of-the-art, self-attention)
        if "-" in word:
            return word

        # Guard 4: Tokens containing apostrophes (e.g., he'd, they'd)
        if "'" in word or "’" in word:
            return word

        # Guard 5: Technical terms / acronyms (case-insensitive check, preserving casing)
        if word.lower() in self._technical_exceptions:
            return word

        # Convert normal word to lowercase for algorithmic stemming
        w = word.lower()

        # Step 1a: Plural handling
        w = _step_1a(w)
        # Step 1b: Past participles and progressive forms
        w = _step_1b(w)
        # Step 1c: Terminal y -> i
        w = _step_1c(w)
        # Step 2: Relational suffixes
        w = _step_2(w)
        # Step 3: Derivational suffixes
        w = _step_3(w)
        # Step 4: Suffix removal
        w = _step_4(w)
        # Step 5a: Trailing e removal
        w = _step_5a(w)
        # Step 5b: Final double -ll cleanup
        w = _step_5b(w)

        return w

    def stem_tokens(self, tokens: Sequence[str]) -> list[str]:
        """Stem a sequence of tokens, returning a fresh list.

        Args:
            tokens: Sequence of string tokens.

        Returns:
            A new list of stemmed tokens.

        Raises:
            TypeError: If tokens is not a sequence or contains non-string items.
        """
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            raise TypeError(
                f"tokens must be a Sequence of strings, got {type(tokens).__name__}"
            )
        return [self.stem(token) for token in tokens]


_DEFAULT_STEMMER = PorterStemmer()


def stem_word(word: str) -> str:
    """Stem a single word using the default PorterStemmer instance."""
    return _DEFAULT_STEMMER.stem(word)


def stem(word: str) -> str:
    """Alias for stem_word."""
    return _DEFAULT_STEMMER.stem(word)


def stem_tokens(tokens: Sequence[str]) -> list[str]:
    """Stem a sequence of tokens using the default PorterStemmer instance."""
    return _DEFAULT_STEMMER.stem_tokens(tokens)
