"""Lexical retrieval engine coordinating text preprocessing, inverted index,
TF-IDF scoring, cosine similarity, and document ranking.
"""

from __future__ import annotations

from collections import Counter
import logging
import math
import time
from typing import Optional

from app.core.config import logging_settings
from app.core.logging_config import log_event
from app.modules.retrieval.lexical.inverted_index import InvertedIndex
from app.modules.retrieval.lexical.preprocessor import TextPreprocessor
from app.modules.retrieval.lexical.ranking import DocumentRanker
from app.modules.retrieval.lexical.similarity import CosineSimilarity
from app.modules.retrieval.lexical.tfidf import TfidfCalculator

logger = logging.getLogger(__name__)


class LexicalRetrievalEngine:
    """Coordinates preprocessing, inverted indexing, TF-IDF calculation, similarity,
    and ranking for lexical document retrieval.
    """

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None) -> None:
        """Initialize the lexical retrieval engine and its constituent components.

        Args:
            preprocessor: Optional TextPreprocessor instance. If None, a default
                TextPreprocessor is instantiated and shared across components.

        Raises:
            TypeError: If preprocessor is not an instance of TextPreprocessor.
        """
        if preprocessor is not None and not isinstance(preprocessor, TextPreprocessor):
            raise TypeError(
                f"preprocessor must be an instance of TextPreprocessor, got {type(preprocessor).__name__}"
            )

        self.preprocessor: TextPreprocessor = (
            preprocessor if preprocessor is not None else TextPreprocessor()
        )
        self.index: InvertedIndex = InvertedIndex(preprocessor=self.preprocessor)
        self.tfidf: TfidfCalculator = TfidfCalculator(index=self.index)
        self.similarity: CosineSimilarity = CosineSimilarity()
        self.ranker: DocumentRanker = DocumentRanker()

    def build_index(self, documents: dict[int, str]) -> None:
        """Build or replace the inverted index from a batch of documents.

        Constructs the new index atomically. If any document ID or text fails
        validation or preprocessing, the previous index remains untouched.

        Args:
            documents: Dictionary mapping positive integer document IDs to raw text.

        Raises:
            TypeError: If documents is not a dict, keys are not ints (or are bools),
                or values are not strings.
            ValueError: If document IDs are not positive integers (> 0).
        """
        if not isinstance(documents, dict):
            raise TypeError(
                f"documents must be a dict, got {type(documents).__name__}"
            )

        # Validate all inputs before attempting modification
        for doc_id, text in documents.items():
            if isinstance(doc_id, bool) or not isinstance(doc_id, int):
                raise TypeError(
                    f"Document ID must be an int, got {type(doc_id).__name__}"
                )
            if doc_id <= 0:
                raise ValueError(
                    f"Document ID must be a positive integer (> 0), got {doc_id}"
                )
            if not isinstance(text, str):
                raise TypeError(
                    f"Document text for ID {doc_id} must be a str, got {type(text).__name__}"
                )

        # Build into a temporary index to guarantee atomic all-or-nothing replacement
        temp_index = InvertedIndex(preprocessor=self.preprocessor)
        for doc_id, text in documents.items():
            temp_index.add_document(doc_id, text)

        # Atomically transfer state into self.index without replacing component instances
        self.index.index = temp_index.index
        self.index.document_terms = temp_index.document_terms
        self.index.document_term_frequencies = temp_index.document_term_frequencies

    def add_document(self, document_id: int, text: str) -> None:
        """Add or update a single document in the index.

        Args:
            document_id: Positive integer document identifier.
            text: Raw document text to index.

        Raises:
            TypeError: If document_id is not an int (or is bool) or text is not a str.
            ValueError: If document_id is not positive (> 0).
        """
        self.index.add_document(document_id, text)

    def remove_document(self, document_id: int) -> None:
        """Remove a single document from the index.

        Args:
            document_id: Positive integer document identifier.

        Raises:
            TypeError: If document_id is not an int (or is bool).
            ValueError: If document_id is not positive (> 0).
        """
        self.index.remove_document(document_id)

    def clear(self) -> None:
        """Clear all indexed documents and postings without replacing component instances."""
        self.index.clear()

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Search for documents relevant to the query and return ranked results.

        Coordinates TF-IDF query vector generation, candidate selection using the
        inverted index, candidate document TF-IDF calculation, cosine similarity
        scoring, and top-K ranking.

        Args:
            query: Raw search query text.
            top_k: Maximum number of ranked results to return (default: 5).

        Returns:
            List of (document_id, similarity_score) tuples ordered descending by score
            with ascending document ID tie-breaking. Returns an empty list if query is
            empty, produces no terms, matches no indexed documents, or index is empty.

        Raises:
            TypeError: If query is not a string, or top_k is a bool or not an int.
            ValueError: If top_k is not a positive integer (> 0).
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be a positive int, got {type(top_k).__name__}")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer (> 0), got {top_k}")

        if not query.strip():
            return []

        start_query_t = time.perf_counter()
        query_vector = self.tfidf.transform_query(query)
        query_duration_ms = (time.perf_counter() - start_query_t) * 1000.0

        trace_debug = getattr(logging_settings, "nlp_trace_enabled", False) and logger.isEnabledFor(logging.DEBUG)

        query_terms = list(query_vector.keys()) if query_vector else []
        term_idfs = {term: round(self.tfidf.compute_idf(term), 4) for term in query_terms}

        if trace_debug:
            tokens = self.preprocessor.preprocess(query)
            q_tf = dict(Counter(tokens))
            q_df = {t: self.index.get_document_frequency(t) for t in q_tf}
            q_idf = {t: round(self.tfidf.compute_idf(t), 4) for t in q_tf}
            q_raw = {t: round(q_tf[t] * self.tfidf.compute_idf(t), 4) for t in q_tf}
            q_sum_sq = sum((q_tf[t] * self.tfidf.compute_idf(t)) ** 2 for t in q_tf)
            q_norm = round(math.sqrt(q_sum_sq), 4)
            q_norm_weights = {t: round(query_vector[t], 4) for t in query_vector}

            log_event(
                logger,
                logging.DEBUG,
                "retrieval_query_diagnostics",
                is_trace=True,
                corpus_n=len(self.index),
                normalized_tokens=tokens,
                query_tf=q_tf,
                query_df=q_df,
                query_idf=q_idf,
                query_raw_weights=q_raw,
                query_l2_norm=q_norm,
                query_normalized_weights=q_norm_weights,
            )

        log_event(
            logger,
            logging.INFO,
            "retrieval_query_processed",
            is_trace=True,
            normalized_terms=query_terms,
            term_count=len(query_terms),
            idf_values=term_idfs,
            analyzer="lowercase_stopwords_stemmer",
            duration_ms=round(query_duration_ms, 2),
        )

        if not query_vector:
            return []

        candidates: set[int] = set()
        for term in query_vector:
            candidates.update(self.index.search(term, preprocessed=True))

        initial_candidates_count = len(candidates)
        if not candidates:
            log_event(
                logger,
                logging.INFO,
                "retrieval_candidates_evaluated",
                is_trace=True,
                initial_candidates=0,
                candidates_after_filtering=0,
            )
            return []

        if trace_debug:
            log_event(
                logger,
                logging.DEBUG,
                "retrieval_candidates_evaluated",
                is_trace=True,
                candidate_doc_ids=sorted(list(candidates)),
                candidate_count=initial_candidates_count,
            )

        scoring_start_t = time.perf_counter()
        similarity_scores: dict[int, float] = {}
        for doc_id in candidates:
            document_vector = self.tfidf.transform_document(doc_id)
            if not document_vector:
                continue

            score = self.similarity.similarity(query_vector, document_vector)
            if score > 0.0:
                similarity_scores[doc_id] = score

            if trace_debug:
                doc_term_freqs = self.index.get_document_term_frequencies(doc_id)
                doc_tf_query = {t: doc_term_freqs.get(t, 0) for t in query_terms}
                doc_raw_weights = {
                    t: round(tf * self.tfidf.compute_idf(t), 4)
                    for t, tf in doc_term_freqs.items()
                }
                doc_sum_sq = sum(
                    (tf * self.tfidf.compute_idf(t)) ** 2
                    for t, tf in doc_term_freqs.items()
                )
                doc_l2_norm = round(math.sqrt(doc_sum_sq), 4)
                doc_norm_weights = {t: round(document_vector.get(t, 0.0), 4) for t in query_terms}
                contributions = {
                    t: round(query_vector.get(t, 0.0) * document_vector.get(t, 0.0), 6)
                    for t in query_terms
                    if t in document_vector
                }
                log_event(
                    logger,
                    logging.DEBUG,
                    "retrieval_candidate_scoring_detail",
                    is_trace=True,
                    doc_id=doc_id,
                    doc_tf=doc_tf_query,
                    doc_l2_norm=doc_l2_norm,
                    doc_normalized_weights=doc_norm_weights,
                    per_term_cosine_contributions=contributions,
                    cosine_similarity=round(score, 6),
                )

        scoring_duration_ms = (time.perf_counter() - scoring_start_t) * 1000.0

        ranked_results = self.ranker.rank(similarity_scores, top_k=top_k)

        if trace_debug:
            log_event(
                logger,
                logging.DEBUG,
                "retrieval_ranking_diagnostics",
                is_trace=True,
                requested_top_k=top_k,
                total_candidates_scored=len(similarity_scores),
                ranked_results=[
                    {"rank": r, "doc_id": doc_id, "score": round(sc, 6)}
                    for r, (doc_id, sc) in enumerate(ranked_results, start=1)
                ],
                tie_breaking="descending score, then ascending doc_id",
            )

        log_event(
            logger,
            logging.INFO,
            "retrieval_lexical_scoring_completed",
            is_trace=True,
            algorithm="TF-IDF-Cosine",
            initial_candidates=initial_candidates_count,
            candidates_after_filtering=len(similarity_scores),
            ranked_count=len(ranked_results),
            requested_top_k=top_k,
            scoring_duration_ms=round(scoring_duration_ms, 2),
        )

        return ranked_results

    def __len__(self) -> int:
        """Return the number of documents currently indexed."""
        return len(self.index)
