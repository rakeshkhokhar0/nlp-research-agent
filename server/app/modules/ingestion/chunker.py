"""Research paper section-aware chunking engine.

This module provides deterministic, section-aware, paragraph-first chunking
for preprocessed and section-detected research papers. It respects section boundaries,
handles oversized paragraphs and sentences gracefully, supports configurable overlap,
preserves source tracking (page spans, paragraph indices, section hierarchy), and
maps directly to the PaperChunk schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Protocol, runtime_checkable

from app.modules.ingestion.section_detector import (
    DetectedSection,
    SectionDetectionResult,
    SectionParagraph,
)


# -------------------------------------------------------------------------
# Token Counter Protocol & Implementations
# -------------------------------------------------------------------------

@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for counting tokens and splitting text to token limits."""

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text string."""
        ...

    def split_text_to_token_limit(self, text: str, max_tokens: int) -> list[str]:
        """Split text into contiguous lossless fragments that each fit within max_tokens."""
        ...


class WhitespaceTokenCounter:
    """Deterministic token counter based on whitespace-separated words.

    Useful for testing and baseline execution without downloading ML models.
    Guarantees exact character and whitespace preservation when splitting.
    """

    @property
    def identity(self) -> str:
        """Return explicit identity string for whitespace token counter."""
        return "WhitespaceTokenCounter"

    def count_tokens(self, text: str) -> int:
        """Count words as tokens (empty text yields 0)."""
        if not text or not text.strip():
            return 0
        return len(text.split())

    def split_text_to_token_limit(self, text: str, max_tokens: int) -> list[str]:
        """Split text into pieces containing at most max_tokens words each.

        Losslessly slices original text at word boundaries so that:
        "".join(pieces) == text.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if not text:
            return []

        word_matches = list(re.finditer(r"\S+", text))
        if not word_matches:
            return [text]
        if len(word_matches) <= max_tokens:
            return [text]

        pieces: list[str] = []
        n_words = len(word_matches)
        for i in range(0, n_words, max_tokens):
            start_char = 0 if i == 0 else word_matches[i].start()
            end_word_idx = i + max_tokens
            if end_word_idx < n_words:
                end_char = word_matches[end_word_idx].start()
            else:
                end_char = len(text)
            pieces.append(text[start_char:end_char])
        return pieces


class CharacterTokenCounter:
    """Deterministic token counter based on character ratio (e.g. 4 chars per token).

    Useful for tests verifying precise token boundaries and fractional lengths.
    Guarantees exact character preservation when splitting.
    """

    def __init__(self, chars_per_token: int = 4) -> None:
        if chars_per_token <= 0:
            raise ValueError(f"chars_per_token must be positive, got {chars_per_token}")
        self.chars_per_token = chars_per_token

    @property
    def identity(self) -> str:
        """Return explicit identity string for character token counter."""
        return f"CharacterTokenCounter(chars_per_token={self.chars_per_token})"

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return (len(text) + self.chars_per_token - 1) // self.chars_per_token

    def split_text_to_token_limit(self, text: str, max_tokens: int) -> list[str]:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if not text:
            return []
        max_chars = max_tokens * self.chars_per_token
        if len(text) <= max_chars:
            return [text]

        pieces: list[str] = []
        for i in range(0, len(text), max_chars):
            pieces.append(text[i : i + max_chars])
        return pieces


class HuggingFaceTokenCounter:
    """Token counter adapter wrapping an already-loaded HuggingFace or transformers tokenizer.

    Operates strictly offline without downloading models or accessing the network.
    Accepts any pre-instantiated tokenizer object exposing `encode()` or `tokenize()`.
    """

    def __init__(self, tokenizer: Any) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer must not be None")
        self.tokenizer = tokenizer

    @property
    def identity(self) -> str:
        """Return explicit identity string including tokenizer model or class name."""
        tokenizer_name = (
            getattr(self.tokenizer, "name_or_path", None)
            or getattr(self.tokenizer, "__name__", None)
            or type(self.tokenizer).__name__
        )
        return f"HuggingFaceTokenCounter({tokenizer_name})"

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if hasattr(self.tokenizer, "encode"):
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except TypeError:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
        elif hasattr(self.tokenizer, "tokenize"):
            tokens = self.tokenizer.tokenize(text)
            return len(tokens)
        else:
            raise TypeError(
                f"Tokenizer of type {type(self.tokenizer).__name__} must implement encode() or tokenize()"
            )

    def split_text_to_token_limit(self, text: str, max_tokens: int) -> list[str]:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if not text:
            return []

        if self.count_tokens(text) <= max_tokens:
            return [text]

        word_matches = list(re.finditer(r"\S+", text))
        if not word_matches or len(word_matches) <= 1:
            # Single word or whitespace-only exceeding token limit: split by characters
            pieces: list[str] = []
            curr_start = 0
            n_chars = len(text)
            while curr_start < n_chars:
                low = curr_start + 1
                high = n_chars
                best_end = low
                while low <= high:
                    mid = (low + high) // 2
                    sub = text[curr_start:mid]
                    if self.count_tokens(sub) <= max_tokens:
                        best_end = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                pieces.append(text[curr_start:best_end])
                curr_start = best_end
            return pieces

        # Multi-word text: split at word boundaries losslessly
        pieces = []
        n_words = len(word_matches)
        start_word_idx = 0
        while start_word_idx < n_words:
            start_char = 0 if start_word_idx == 0 else word_matches[start_word_idx].start()
            low = start_word_idx
            high = n_words - 1
            best_word_idx = start_word_idx
            while low <= high:
                mid = (low + high) // 2
                end_char = word_matches[mid + 1].start() if mid + 1 < n_words else len(text)
                candidate = text[start_char:end_char]
                if self.count_tokens(candidate) <= max_tokens:
                    best_word_idx = mid
                    low = mid + 1
                else:
                    high = mid - 1

            if best_word_idx == start_word_idx:
                end_char = word_matches[start_word_idx + 1].start() if start_word_idx + 1 < n_words else len(text)
                single_word_slice = text[start_char:end_char]
                if self.count_tokens(single_word_slice) <= max_tokens:
                    pieces.append(single_word_slice)
                    start_word_idx += 1
                else:
                    # Single word slice itself exceeds max_tokens: character split
                    curr_c = start_char
                    while curr_c < end_char:
                        c_low = curr_c + 1
                        c_high = end_char
                        c_best = c_low
                        while c_low <= c_high:
                            c_mid = (c_low + c_high) // 2
                            c_sub = text[curr_c:c_mid]
                            if self.count_tokens(c_sub) <= max_tokens:
                                c_best = c_mid
                                c_low = c_mid + 1
                            else:
                                c_high = c_mid - 1
                        pieces.append(text[curr_c:c_best])
                        curr_c = c_best
                    start_word_idx += 1
            else:
                end_char = word_matches[best_word_idx + 1].start() if best_word_idx + 1 < n_words else len(text)
                pieces.append(text[start_char:end_char])
                start_word_idx = best_word_idx + 1

        return pieces


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for research paper chunking engine.

    Attributes:
        target_chunk_size: Desired chunk size in tokens.
        max_chunk_size: Hard upper bound for chunk size in tokens.
        overlap: Target overlap in tokens between adjacent chunks within the same section.
        min_chunk_size: Lower bound threshold below which non-final chunks are flagged.
        chunking_version: Identifier for the chunking strategy and parameter version.
    """

    target_chunk_size: int = 200
    max_chunk_size: int = 220
    overlap: int = 40
    min_chunk_size: int = 40
    chunking_version: str = "section-aware-v1"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.target_chunk_size <= 0:
            raise ValueError(f"target_chunk_size must be positive, got {self.target_chunk_size}")
        if self.max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {self.max_chunk_size}")
        if self.min_chunk_size <= 0:
            raise ValueError(f"min_chunk_size must be positive, got {self.min_chunk_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {self.overlap}")
        if self.max_chunk_size < self.target_chunk_size:
            raise ValueError(
                f"max_chunk_size ({self.max_chunk_size}) must be >= target_chunk_size ({self.target_chunk_size})"
            )
        if self.overlap >= self.max_chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be < max_chunk_size ({self.max_chunk_size})"
            )
        if self.min_chunk_size > self.target_chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) must be <= target_chunk_size ({self.target_chunk_size})"
            )


# -------------------------------------------------------------------------
# Output Data Structures
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class ResearchChunk:
    """An immutable chunk of research paper text mapped to the database schema.

    Attributes:
        chunk_index: 0-indexed sequential position across the entire paper.
        content: Text content of the chunk.
        token_count: Number of tokens measured by the injected TokenCounter.
        char_count: Number of characters in content string.
        page_start: 1-indexed source starting page number.
        page_end: 1-indexed source ending page number.
        section_title: Title of enclosing section (or 'Preamble' / None).
        content_hash: SHA-256 hexadecimal digest of content.
        metadata: JSON-serializable dictionary with structural and provenance data.
    """

    chunk_index: int
    content: str
    token_count: int
    char_count: int
    page_start: int
    page_end: int
    section_title: str | None
    content_hash: str
    metadata: dict[str, Any]


@dataclass
class ChunkingDiagnostics:
    """Diagnostics and performance metrics for a chunking execution.

    Attributes:
        input_paragraph_count: Total non-empty paragraphs received across preamble and sections.
        output_chunk_count: Total ResearchChunk instances generated.
        sections_processed: Total logical sections processed (including preamble if present).
        oversized_paragraphs_split: Count of paragraphs exceeding max_chunk_size that were split.
        total_overlap_tokens: Sum of overlap tokens generated across all chunk transitions.
        warnings: Quality and boundary warnings encountered during chunking.
    """

    input_paragraph_count: int
    output_chunk_count: int
    sections_processed: int
    oversized_paragraphs_split: int
    total_overlap_tokens: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class ChunkingResult:
    """Complete chunking result containing all generated chunks and diagnostics.

    Attributes:
        chunks: Chronologically ordered list of ResearchChunk objects.
        diagnostics: Quality diagnostics and metrics.
    """

    chunks: list[ResearchChunk]
    diagnostics: ChunkingDiagnostics

    def reconstruct_text(self) -> str:
        """Reconstruct document text from non-overlapping chunks using boundary separators.

        Preserves exact inter-paragraph boundaries ('\\n\\n') while never inserting
        separators across split intra-paragraph units.

        Returns:
            The reconstructed document text string.

        Raises:
            ValueError: If any chunk in the result has overlap_tokens > 0.
        """
        if any(c.metadata.get("overlap_tokens", 0) > 0 for c in self.chunks):
            raise ValueError("Text reconstruction requires overlap-free chunks (overlap=0).")
        return "".join(c.metadata.get("boundary_separator", "") + c.content for c in self.chunks)


# -------------------------------------------------------------------------
# Internal Fragment Representation & Boundary Logic
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class _TextUnit:
    """A minimal atomic chunking unit (paragraph, sentence, or sub-sentence fragment).

    Attributes:
        text: Exact string content of the unit.
        paragraph_index: Global paragraph index of origin.
        page_number: Source page number.
        token_count: Token count of this unit.
        is_paragraph_start: True if this unit begins a distinct paragraph.
    """

    text: str
    paragraph_index: int
    page_number: int
    token_count: int
    is_paragraph_start: bool = True


def _render_units_text(units: list[_TextUnit]) -> str:
    """Render a sequence of text units into a combined string.

    Distinguishes inter-paragraph boundaries (separated by '\\n\\n') from
    intra-paragraph continuations (joined with empty string so trailing whitespace
    and characters are preserved verbatim).
    """
    if not units:
        return ""
    parts: list[str] = []
    for i, u in enumerate(units):
        if i == 0:
            parts.append(u.text)
        elif u.is_paragraph_start:
            parts.append("\n\n")
            parts.append(u.text)
        else:
            parts.append(u.text)
    return "".join(parts)


# Common abbreviation prefixes that should not cause sentence splitting
_ABBREVIATIONS: tuple[str, ...] = (
    "e.g.", "i.e.", "et al.", "fig.", "figs.", "tab.", "tabs.", "eq.", "eqs.",
    "sec.", "secs.", "dr.", "prof.", "mr.", "mrs.", "ms.", "vs.", "al.",
    "dept.", "univ.", "vol.", "no.", "pp.", "p.", "ref.", "refs.", "cf.", "approx.",
)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving academic abbreviations and formulas.

    Guarantees that concatenating sentences losslessly reconstructs the original string:
    "".join(sentences) == text.
    """
    if not text:
        return []

    candidate_pattern = re.compile(r"([\.\?!]+)(\s+)(?=[A-Z0-9\"'\(]|$)")
    sentences: list[str] = []
    start_idx = 0

    for match in candidate_pattern.finditer(text):
        punct_end = match.end(1)
        prefix = text[start_idx:punct_end].strip()
        is_abbreviation = False

        lower_prefix = prefix.lower()
        for abbr in _ABBREVIATIONS:
            if lower_prefix.endswith(abbr):
                is_abbreviation = True
                break

        if re.search(r"\b[A-Z]\.$", prefix):
            is_abbreviation = True

        if is_abbreviation:
            continue

        end_idx = match.end()
        sentence = text[start_idx:end_idx]
        sentences.append(sentence)
        start_idx = end_idx

    if start_idx < len(text):
        remainder = text[start_idx:]
        sentences.append(remainder)

    return sentences if sentences else [text]


# -------------------------------------------------------------------------
# Research Chunking Engine
# -------------------------------------------------------------------------

class ResearchChunkingEngine:
    """Deterministic, section-aware chunking engine for research papers.

    Operates on SectionDetectionResult, processes preamble and sections independently,
    preserves paragraph ordering, handles oversized content, enforces hard maximum token limits,
    and guarantees traceable provenance and schema compatibility with PaperChunk.
    """

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        config: ChunkingConfig | None = None,
    ) -> None:
        """Initialize the chunking engine.

        Args:
            token_counter: Token counting implementation. Defaults to WhitespaceTokenCounter.
            config: Chunking configuration. Defaults to ChunkingConfig().
        """
        self.token_counter = token_counter if token_counter is not None else WhitespaceTokenCounter()
        self.config = config if config is not None else ChunkingConfig()

    @property
    def tokenizer_identity(self) -> str:
        """Return the explicit identity and configuration of the active tokenizer."""
        if hasattr(self.token_counter, "identity"):
            return str(self.token_counter.identity)
        return type(self.token_counter).__name__

    def chunk(self, detection_result: SectionDetectionResult) -> ChunkingResult:
        """Generate research chunks from section detection results.

        Args:
            detection_result: Source SectionDetectionResult containing preamble and sections.

        Returns:
            ChunkingResult containing ordered ResearchChunk list and diagnostics.

        Raises:
            TypeError: If detection_result is not an instance of SectionDetectionResult.
        """
        if not isinstance(detection_result, SectionDetectionResult):
            raise TypeError(
                f"Expected SectionDetectionResult instance, got {type(detection_result).__name__}"
            )

        all_chunks: list[ResearchChunk] = []
        global_chunk_idx = 0
        oversized_split_count = 0
        total_overlap_tokens = 0
        sections_processed_count = 0
        warnings: list[str] = []

        total_input_paragraphs = sum(
            1 for p in detection_result.preamble if p.text and p.text.strip()
        ) + sum(
            1 for s in detection_result.sections for p in s.paragraphs if p.text and p.text.strip()
        )

        # 1. Process Preamble as its own logical section
        if detection_result.preamble:
            preamble_chunks, splits, overlap_toks, warn = self._chunk_section_paragraphs(
                paragraphs=detection_result.preamble,
                section_title="Preamble",
                section_id=None,
                section_index=None,
                section_type="preamble",
                section_level=0,
                parent_section_id=None,
                start_chunk_index=global_chunk_idx,
            )
            all_chunks.extend(preamble_chunks)
            global_chunk_idx += len(preamble_chunks)
            oversized_split_count += splits
            total_overlap_tokens += overlap_toks
            sections_processed_count += 1
            warnings.extend(warn)

        # 2. Process each DetectedSection independently
        for sec in detection_result.sections:
            sec_chunks, splits, overlap_toks, warn = self._chunk_section_paragraphs(
                paragraphs=sec.paragraphs,
                section_title=sec.title.strip(),
                section_id=sec.section_id,
                section_index=sec.section_index,
                section_type=sec.section_type.value if hasattr(sec.section_type, "value") else str(sec.section_type),
                section_level=sec.level,
                parent_section_id=sec.parent_section_id,
                start_chunk_index=global_chunk_idx,
                fallback_heading_page=sec.heading_page,
            )
            all_chunks.extend(sec_chunks)
            global_chunk_idx += len(sec_chunks)
            oversized_split_count += splits
            total_overlap_tokens += overlap_toks
            sections_processed_count += 1
            warnings.extend(warn)

        # Assemble diagnostics
        diagnostics = ChunkingDiagnostics(
            input_paragraph_count=total_input_paragraphs,
            output_chunk_count=len(all_chunks),
            sections_processed=sections_processed_count,
            oversized_paragraphs_split=oversized_split_count,
            total_overlap_tokens=total_overlap_tokens,
            warnings=warnings,
        )

        return ChunkingResult(chunks=all_chunks, diagnostics=diagnostics)

    # -------------------------------------------------------------------------
    # Internal Section Chunking Logic
    # -------------------------------------------------------------------------

    def _chunk_section_paragraphs(
        self,
        paragraphs: list[SectionParagraph],
        section_title: str | None,
        section_id: str | None,
        section_index: int | None,
        section_type: str | None,
        section_level: int | None,
        parent_section_id: str | None,
        start_chunk_index: int,
        fallback_heading_page: int | None = None,
    ) -> tuple[list[ResearchChunk], int, int, list[str]]:
        """Chunk all paragraphs within a single section boundary."""
        chunks: list[ResearchChunk] = []
        splits_count = 0
        overlap_tokens_count = 0
        warnings: list[str] = []

        if not paragraphs:
            return chunks, splits_count, overlap_tokens_count, warnings

        # Step A: Convert paragraphs into atomic units (splitting oversized ones)
        units: list[_TextUnit] = []
        for p in paragraphs:
            text = p.text
            if not text or not text.strip():
                continue

            tok_count = self.token_counter.count_tokens(text)
            if tok_count <= self.config.max_chunk_size:
                units.append(
                    _TextUnit(
                        text=text,
                        paragraph_index=p.paragraph_index,
                        page_number=p.page_number,
                        token_count=tok_count,
                        is_paragraph_start=True,
                    )
                )
            else:
                splits_count += 1
                split_units = self._split_oversized_paragraph(p)
                units.extend(split_units)

        if not units:
            return chunks, splits_count, overlap_tokens_count, warnings

        # Step B: Group units into chunks with section-bounded overlap
        unit_idx = 0
        n_units = len(units)
        chunk_idx_in_sec = 0
        previous_chunk_units: list[_TextUnit] = []

        while unit_idx < n_units:
            current_overlap_units: list[_TextUnit] = []
            if previous_chunk_units and self.config.overlap > 0:
                current_overlap_units = self._select_overlap_units(
                    previous_chunk_units, self.config.overlap
                )

            next_unit = units[unit_idx]

            # If overlap + next_unit exceeds max_chunk_size, trim overlap from front
            while current_overlap_units:
                candidate_text = _render_units_text(current_overlap_units + [next_unit])
                if self.token_counter.count_tokens(candidate_text) <= self.config.max_chunk_size:
                    break
                current_overlap_units.pop(0)

            # Recalculate actual overlap tokens after trimming
            if current_overlap_units:
                actual_overlap_tokens = self.token_counter.count_tokens(
                    _render_units_text(current_overlap_units)
                )
            else:
                actual_overlap_tokens = 0

            overlap_tokens_count += actual_overlap_tokens

            current_units = list(current_overlap_units)
            current_units.append(next_unit)
            unit_idx += 1
            current_tokens = self.token_counter.count_tokens(_render_units_text(current_units))

            # Add more new units if possible and below target_chunk_size
            while unit_idx < n_units and current_tokens < self.config.target_chunk_size:
                candidate_units = current_units + [units[unit_idx]]
                candidate_tokens = self.token_counter.count_tokens(
                    _render_units_text(candidate_units)
                )
                if candidate_tokens <= self.config.max_chunk_size:
                    current_units = candidate_units
                    current_tokens = candidate_tokens
                    unit_idx += 1
                else:
                    break

            final_content = _render_units_text(current_units)
            final_token_count = self.token_counter.count_tokens(final_content)
            if final_token_count > self.config.max_chunk_size:
                raise ValueError(
                    f"Chunk token count ({final_token_count}) strictly exceeded max_chunk_size ({self.config.max_chunk_size})"
                )

            is_final_in_section = (unit_idx >= n_units)
            if not is_final_in_section and final_token_count < self.config.min_chunk_size:
                warnings.append(
                    f"Non-final chunk {start_chunk_index + len(chunks)} in section '{section_title or 'Untitled'}' "
                    f"is undersized: {final_token_count} tokens < min_chunk_size {self.config.min_chunk_size}"
                )

            final_char_count = len(final_content)
            content_hash = hashlib.sha256(final_content.encode("utf-8")).hexdigest()

            page_start = min(u.page_number for u in current_units)
            page_end = max(u.page_number for u in current_units)
            para_indices = sorted(list(set(u.paragraph_index for u in current_units)))

            global_chunk_idx = start_chunk_index + len(chunks)
            if global_chunk_idx == 0:
                boundary_separator = ""
            elif current_units and not current_units[0].is_paragraph_start:
                boundary_separator = ""
            else:
                boundary_separator = "\n\n"

            chunk_meta: dict[str, Any] = {
                "section_id": section_id,
                "section_index": section_index,
                "section_type": section_type,
                "section_level": section_level,
                "parent_section_id": parent_section_id,
                "paragraph_indices": para_indices,
                "chunk_index_in_section": chunk_idx_in_sec,
                "overlap_tokens": actual_overlap_tokens,
                "boundary_separator": boundary_separator,
                "prefix_separator": boundary_separator,
                "chunking_version": self.config.chunking_version,
            }

            chunk = ResearchChunk(
                chunk_index=start_chunk_index + len(chunks),
                content=final_content,
                token_count=final_token_count,
                char_count=final_char_count,
                page_start=page_start,
                page_end=page_end,
                section_title=section_title,
                content_hash=content_hash,
                metadata=chunk_meta,
            )
            chunks.append(chunk)
            chunk_idx_in_sec += 1
            previous_chunk_units = current_units

        return chunks, splits_count, overlap_tokens_count, warnings

    def _split_oversized_paragraph(self, p: SectionParagraph) -> list[_TextUnit]:
        """Split a single oversized paragraph into sentence and token pieces.

        Guarantees exact character preservation: "".join(u.text for u in units) == p.text.
        """
        units: list[_TextUnit] = []
        sentences = _split_into_sentences(p.text)

        first_unit = True
        for sentence in sentences:
            sent_tokens = self.token_counter.count_tokens(sentence)

            if sent_tokens <= self.config.max_chunk_size:
                units.append(
                    _TextUnit(
                        text=sentence,
                        paragraph_index=p.paragraph_index,
                        page_number=p.page_number,
                        token_count=sent_tokens,
                        is_paragraph_start=first_unit,
                    )
                )
                first_unit = False
            else:
                pieces = self.token_counter.split_text_to_token_limit(
                    sentence, self.config.max_chunk_size
                )
                if not pieces and sentence:
                    raise ValueError(
                        f"Token counter returned empty list for non-empty text: {sentence!r}"
                    )
                reconstructed = "".join(pieces)
                if reconstructed != sentence:
                    raise ValueError(
                        f"Token counter split dropped or altered characters: "
                        f"expected {len(sentence)} chars, got {len(reconstructed)}"
                    )
                for idx, piece in enumerate(pieces):
                    if not piece and len(pieces) > 1:
                        raise ValueError(
                            f"Token counter split returned empty fragment at index {idx} in multi-fragment result"
                        )
                    piece_tokens = self.token_counter.count_tokens(piece)
                    if piece_tokens > self.config.max_chunk_size:
                        raise ValueError(
                            f"Token counter split returned fragment exceeding max_chunk_size "
                            f"({piece_tokens} > {self.config.max_chunk_size}): {piece!r}"
                        )
                    units.append(
                        _TextUnit(
                            text=piece,
                            paragraph_index=p.paragraph_index,
                            page_number=p.page_number,
                            token_count=piece_tokens,
                            is_paragraph_start=first_unit,
                        )
                    )
                    first_unit = False

        return units

    def _select_overlap_units(
        self,
        previous_chunk_units: list[_TextUnit],
        target_overlap_tokens: int,
    ) -> list[_TextUnit]:
        """Select trailing units from the immediately preceding chunk up to target_overlap_tokens."""
        if not previous_chunk_units or target_overlap_tokens <= 0:
            return []

        selected: list[_TextUnit] = []
        for u in reversed(previous_chunk_units):
            candidate = [u] + selected
            cand_text = _render_units_text(candidate)
            cand_tokens = self.token_counter.count_tokens(cand_text)

            if cand_tokens <= target_overlap_tokens:
                selected = candidate
            else:
                break

        return selected
