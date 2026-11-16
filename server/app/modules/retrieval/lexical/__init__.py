"""Lexical text processing package for information retrieval."""

from app.modules.retrieval.lexical.engine import LexicalRetrievalEngine
from app.modules.retrieval.lexical.inverted_index import InvertedIndex
from app.modules.retrieval.lexical.normalizer import (
    CONTRACTION_MAP,
    expand_contractions_and_possessives,
    normalize_token,
    normalize_tokens,
)
from app.modules.retrieval.lexical.preprocessor import (
    TextPreprocessor,
    preprocess,
)
from app.modules.retrieval.lexical.stemmer import (
    DEFAULT_TECHNICAL_EXCEPTIONS,
    PorterStemmer,
    stem,
    stem_tokens,
    stem_word,
)
from app.modules.retrieval.lexical.stopwords import (
    DEFAULT_STOPWORDS,
    StopwordFilter,
    remove_stopwords,
)
from app.modules.retrieval.lexical.ranking import (
    DocumentRanker,
    rank_documents,
)
from app.modules.retrieval.lexical.similarity import (
    CosineSimilarity,
    cosine_similarity,
)
from app.modules.retrieval.lexical.tfidf import TfidfCalculator
from app.modules.retrieval.lexical.tokenizer import Tokenizer, tokenize

__all__ = [
    "CONTRACTION_MAP",
    "CosineSimilarity",
    "DEFAULT_STOPWORDS",
    "DEFAULT_TECHNICAL_EXCEPTIONS",
    "DocumentRanker",
    "InvertedIndex",
    "LexicalRetrievalEngine",
    "PorterStemmer",
    "StopwordFilter",
    "TextPreprocessor",
    "TfidfCalculator",
    "Tokenizer",
    "cosine_similarity",
    "expand_contractions_and_possessives",
    "normalize_token",
    "normalize_tokens",
    "preprocess",
    "rank_documents",
    "remove_stopwords",
    "stem",
    "stem_tokens",
    "stem_word",
    "tokenize",
]
