"""Unit tests for Research Paper Chunking Engine.

Covers all 25 architectural requirements, edge cases, configuration validation,
section isolation, oversized content handling, overlap, and source tracking.
"""

from __future__ import annotations

import copy
import hashlib
import pytest

from app.modules.ingestion.chunker import (
    CharacterTokenCounter,
    ChunkingConfig,
    ChunkingDiagnostics,
    ChunkingResult,
    HuggingFaceTokenCounter,
    ResearchChunk,
    ResearchChunkingEngine,
    TokenCounter,
    WhitespaceTokenCounter,
)
from app.modules.ingestion.section_detector import (
    DetectedSection,
    SectionDetectionDiagnostics,
    SectionDetectionResult,
    SectionParagraph,
    SectionType,
)


# -------------------------------------------------------------------------
# Test Helpers and Fixtures
# -------------------------------------------------------------------------

def make_paragraph(text: str, page: int = 1, idx: int = 0) -> SectionParagraph:
    """Helper to create a SectionParagraph."""
    return SectionParagraph(text=text, page_number=page, paragraph_index=idx)


def make_section(
    title: str,
    paragraphs: list[SectionParagraph],
    sec_id: str = "sec_1",
    sec_idx: int = 0,
    sec_type: SectionType = SectionType.INTRODUCTION,
    level: int = 1,
    heading_page: int = 1,
    parent_id: str | None = None,
) -> DetectedSection:
    """Helper to create a DetectedSection."""
    return DetectedSection(
        section_id=sec_id,
        section_index=sec_idx,
        heading_paragraph_index=0,
        title=title,
        normalized_title=title.lower(),
        section_type=sec_type,
        level=level,
        parent_section_id=parent_id,
        heading_page=heading_page,
        start_page=heading_page,
        end_page=max([p.page_number for p in paragraphs], default=heading_page),
        paragraphs=paragraphs,
        confidence=1.0,
    )


def make_detection_result(
    sections: list[DetectedSection] | None = None,
    preamble: list[SectionParagraph] | None = None,
) -> SectionDetectionResult:
    """Helper to create a SectionDetectionResult."""
    secs = sections or []
    pre = preamble or []
    total_p = len(pre) + sum(len(s.paragraphs) for s in secs)
    return SectionDetectionResult(
        sections=secs,
        preamble=pre,
        diagnostics=SectionDetectionDiagnostics(
            total_paragraphs=total_p,
            detected_headings=len(secs),
            rejected_candidates=0,
            unclassified_headings=0,
            sections_created=len(secs),
        ),
    )


# -------------------------------------------------------------------------
# Test Suite
# -------------------------------------------------------------------------

class TestResearchChunkingEngine:
    """Comprehensive unit test suite for ResearchChunkingEngine."""

    @pytest.fixture
    def token_counter(self) -> WhitespaceTokenCounter:
        return WhitespaceTokenCounter()

    @pytest.fixture
    def default_config(self) -> ChunkingConfig:
        return ChunkingConfig(
            target_chunk_size=10,
            max_chunk_size=12,
            overlap=3,
            min_chunk_size=2,
            chunking_version="test-v1",
        )

    @pytest.fixture
    def engine(self, token_counter: WhitespaceTokenCounter, default_config: ChunkingConfig) -> ResearchChunkingEngine:
        return ResearchChunkingEngine(token_counter=token_counter, config=default_config)

    # 1. Empty detection result
    def test_empty_detection_result(self, engine: ResearchChunkingEngine) -> None:
        """Returns empty chunk list with zero diagnostics on empty input."""
        empty_result = make_detection_result()
        result = engine.chunk(empty_result)

        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 0
        assert result.diagnostics.input_paragraph_count == 0
        assert result.diagnostics.output_chunk_count == 0
        assert result.diagnostics.sections_processed == 0

    # 2. Preamble-only document
    def test_preamble_only_document(self, engine: ResearchChunkingEngine) -> None:
        """Preamble is processed as its own section with null section metadata."""
        preamble = [
            make_paragraph("Paper Title: Deep Reasoning", page=1, idx=0),
            make_paragraph("Author One and Author Two", page=1, idx=1),
        ]
        det = make_detection_result(preamble=preamble)
        result = engine.chunk(det)

        assert len(result.chunks) == 1
        chunk = result.chunks[0]
        assert chunk.chunk_index == 0
        assert chunk.section_title == "Preamble"
        assert chunk.metadata["section_type"] == "preamble"
        assert chunk.metadata["section_id"] is None
        assert chunk.metadata["paragraph_indices"] == [0, 1]
        assert "Deep Reasoning" in chunk.content

    # 3. One section with one paragraph
    def test_one_section_with_one_paragraph(self, engine: ResearchChunkingEngine) -> None:
        """Single section with a single small paragraph generates one chunk."""
        sec = make_section("1. Introduction", [make_paragraph("This is introductory text.", page=1, idx=1)])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert len(result.chunks) == 1
        chunk = result.chunks[0]
        assert chunk.chunk_index == 0
        assert chunk.section_title == "1. Introduction"
        assert chunk.content == "This is introductory text."
        assert chunk.metadata["section_id"] == "sec_1"
        assert chunk.metadata["section_index"] == 0
        assert chunk.metadata["paragraph_indices"] == [1]

    # 4. Multiple paragraphs grouped into one chunk
    def test_multiple_paragraphs_grouped_into_one_chunk(self, engine: ResearchChunkingEngine) -> None:
        """Consecutive paragraphs fitting within target_chunk_size are merged."""
        # 3 words + 3 words = 6 words <= target_chunk_size 10
        p1 = make_paragraph("One two three.", page=1, idx=1)
        p2 = make_paragraph("Four five six.", page=1, idx=2)
        sec = make_section("1. Methods", [p1, p2])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert len(result.chunks) == 1
        chunk = result.chunks[0]
        assert chunk.content == "One two three.\n\nFour five six."
        assert chunk.metadata["paragraph_indices"] == [1, 2]
        assert chunk.token_count == 6

    # 5. Paragraphs split across multiple chunks
    def test_paragraphs_split_across_multiple_chunks(self, engine: ResearchChunkingEngine) -> None:
        """Paragraphs are partitioned into multiple chunks when exceeding target size."""
        # Each paragraph has 6 words. Total = 18 words > target 10
        p1 = make_paragraph("Word one two three four five.", page=1, idx=1)  # 6 words
        p2 = make_paragraph("Word six seven eight nine ten.", page=1, idx=2)  # 6 words
        p3 = make_paragraph("Word eleven twelve thirteen fourteen fifteen.", page=2, idx=3)  # 6 words
        sec = make_section("1. Overview", [p1, p2, p3])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert len(result.chunks) >= 2
        for c in result.chunks:
            assert c.token_count <= 12  # max_chunk_size

    # 6. No mixing between sections
    def test_no_mixing_between_sections(self, engine: ResearchChunkingEngine) -> None:
        """Chunks never contain paragraphs from more than one section."""
        sec1 = make_section("1. Intro", [make_paragraph("Introductory sentence.", page=1, idx=1)], sec_id="sec_1", sec_idx=0)
        sec2 = make_section("2. Methods", [make_paragraph("Methodology sentence.", page=1, idx=2)], sec_id="sec_2", sec_idx=1)
        det = make_detection_result(sections=[sec1, sec2])
        result = engine.chunk(det)

        assert len(result.chunks) == 2
        assert result.chunks[0].section_title == "1. Intro"
        assert result.chunks[0].metadata["section_id"] == "sec_1"
        assert result.chunks[1].section_title == "2. Methods"
        assert result.chunks[1].metadata["section_id"] == "sec_2"
        assert "Methodology" not in result.chunks[0].content
        assert "Introductory" not in result.chunks[1].content

    # 7. Section ordering preserved
    def test_section_ordering_preserved(self, engine: ResearchChunkingEngine) -> None:
        """Chunks are created in the exact order of sections in SectionDetectionResult."""
        secs = [
            make_section(f"Section {i}", [make_paragraph(f"Content {i}", page=i+1, idx=i)], sec_id=f"sec_{i}", sec_idx=i)
            for i in range(4)
        ]
        det = make_detection_result(sections=secs)
        result = engine.chunk(det)

        assert len(result.chunks) == 4
        for i, c in enumerate(result.chunks):
            assert c.chunk_index == i
            assert c.section_title == f"Section {i}"
            assert c.metadata["section_index"] == i

    # 8. Oversized paragraph splitting
    def test_oversized_paragraph_splitting(self, engine: ResearchChunkingEngine) -> None:
        """An oversized paragraph exceeding max_chunk_size is split at sentence boundaries."""
        # 16 words > max_chunk_size 12. Two 8-word sentences.
        text = "This is the first sentence with eight words. This is the second sentence with eight words."
        sec = make_section("1. Analysis", [make_paragraph(text, page=1, idx=1)])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert len(result.chunks) >= 2
        assert result.diagnostics.oversized_paragraphs_split == 1
        for c in result.chunks:
            assert c.token_count <= 12
            assert c.metadata["paragraph_indices"] == [1]

    # 9. Oversized sentence splitting
    def test_oversized_sentence_splitting(self, engine: ResearchChunkingEngine) -> None:
        """A single sentence with no punctuation exceeding max_chunk_size is split losslessly."""
        # 20 words in one single sentence without periods
        words = [f"word{i}" for i in range(20)]
        huge_sentence = " ".join(words)
        sec = make_section("1. Data", [make_paragraph(huge_sentence, page=1, idx=1)])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert len(result.chunks) >= 2
        for c in result.chunks:
            assert c.token_count <= 12
        # Verify all words are represented across chunks
        all_content = " ".join(c.content for c in result.chunks)
        for w in words:
            assert w in all_content

    # 10. Preservation of punctuation and source text
    def test_preservation_of_punctuation_and_source_text(self) -> None:
        """Special punctuation, math equations, parentheses, and citations are preserved verbatim."""
        cfg = ChunkingConfig(target_chunk_size=30, max_chunk_size=40, overlap=5, min_chunk_size=5)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)
        text = "We observe $L(\\theta) \\approx 0.05$ (Smith et al., 2021; [1, 2]); results: positive!"
        sec = make_section("1. Results", [make_paragraph(text, page=1, idx=1)])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 1
        assert result.chunks[0].content == text

    # 11. Overlap between chunks within the same section
    def test_overlap_between_chunks(self) -> None:
        """Neighboring chunks within the same section share trailing overlap tokens."""
        cfg = ChunkingConfig(target_chunk_size=8, max_chunk_size=10, overlap=4, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)
        p1 = make_paragraph("Alpha beta gamma delta.", page=1, idx=1)      # 4 words
        p2 = make_paragraph("Epsilon zeta eta theta.", page=1, idx=2)      # 4 words
        p3 = make_paragraph("Iota kappa lambda mu.", page=1, idx=3)        # 4 words
        p4 = make_paragraph("Nu xi omicron pi.", page=1, idx=4)            # 4 words
        sec = make_section("1. Alphabet", [p1, p2, p3, p4])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 3
        # Chunk 0 has p1 + p2 (8 tokens)
        assert result.chunks[0].token_count == 8
        assert result.chunks[0].metadata["overlap_tokens"] == 0

        # Chunk 1 has overlap p2 (4 tokens) + p3 (4 tokens) = 8 tokens
        assert result.chunks[1].token_count == 8
        assert result.chunks[1].metadata["overlap_tokens"] == 4
        assert "Epsilon zeta eta theta." in result.chunks[0].content
        assert "Epsilon zeta eta theta." in result.chunks[1].content

        # Chunk 2 has overlap p3 (4 tokens) + p4 (4 tokens) = 8 tokens
        assert result.chunks[2].token_count == 8
        assert result.chunks[2].metadata["overlap_tokens"] == 4
        assert "Iota kappa lambda mu." in result.chunks[1].content
        assert "Iota kappa lambda mu." in result.chunks[2].content

        assert result.diagnostics.total_overlap_tokens == 8
        for c in result.chunks:
            assert c.token_count <= cfg.max_chunk_size

    # 12. No overlap across section boundaries
    def test_no_overlap_across_section_boundaries(self, engine: ResearchChunkingEngine) -> None:
        """First chunk of a new section has zero overlap from previous section."""
        sec1 = make_section("1. Intro", [make_paragraph("Alpha beta gamma delta epsilon zeta eta theta.", page=1, idx=1)])
        sec2 = make_section("2. Methods", [make_paragraph("Nu xi omicron pi rho sigma tau upsilon phi.", page=2, idx=2)], sec_id="sec_2", sec_idx=1)
        det = make_detection_result(sections=[sec1, sec2])
        result = engine.chunk(det)

        assert len(result.chunks) == 2
        # Section 2 chunk must have 0 overlap tokens
        assert result.chunks[1].metadata["overlap_tokens"] == 0
        assert "Alpha" not in result.chunks[1].content

    # 13. Page start and page end calculations
    def test_page_start_and_page_end_calculations(self, engine: ResearchChunkingEngine) -> None:
        """Chunk page_start and page_end reflect the pages of constituent paragraphs."""
        p1 = make_paragraph("Page two paragraph.", page=2, idx=1)
        p2 = make_paragraph("Page four paragraph.", page=4, idx=2)
        sec = make_section("1. MultiPage", [p1, p2])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert len(result.chunks) == 1
        assert result.chunks[0].page_start == 2
        assert result.chunks[0].page_end == 4

    # 14. Paragraph source tracking in metadata
    def test_paragraph_source_tracking(self, engine: ResearchChunkingEngine) -> None:
        """Chunks accurately record global paragraph indices in metadata."""
        p1 = make_paragraph("First paragraph text.", page=1, idx=10)
        p2 = make_paragraph("Second paragraph text.", page=1, idx=15)
        sec = make_section("1. Tracking", [p1, p2])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert result.chunks[0].metadata["paragraph_indices"] == [10, 15]

    # 15. Repeated paragraph indices for split paragraphs
    def test_repeated_paragraph_indices_for_split_paragraphs(self, engine: ResearchChunkingEngine) -> None:
        """Multiple chunks originating from one split paragraph share that paragraph index."""
        huge_text = "Sentence one is here. Sentence two is here. Sentence three is here. Sentence four is here."
        p = make_paragraph(huge_text, page=1, idx=7)
        sec = make_section("1. Split", [p])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert len(result.chunks) >= 2
        for c in result.chunks:
            assert c.metadata["paragraph_indices"] == [7]

    # 16. Unicode text handling
    def test_unicode_text_handling(self, engine: ResearchChunkingEngine) -> None:
        """Handles multilingual unicode, mathematical symbols, and non-ASCII glyphs correctly."""
        text = "BERT model (Bidirectional Encoder Representations from Transformers) 模型 评估 λ=0.01 €100."
        p = make_paragraph(text, page=1, idx=1)
        sec = make_section("1. Multilingual", [p])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert len(result.chunks) == 1
        assert "模型" in result.chunks[0].content
        assert "€100" in result.chunks[0].content
        assert result.chunks[0].char_count == len(text)

    # 17. Equations, citations, URLs, and references
    def test_equations_citations_urls_references(self) -> None:
        """Preserves DOI URLs, citations, and formulas without mangling."""
        cfg = ChunkingConfig(target_chunk_size=30, max_chunk_size=40, overlap=5, min_chunk_size=5)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)
        text = "See https://doi.org/10.1145/1234567 and Vaswani et al. (2017) [1] for details: $$\\nabla f(x) = 0$$."
        sec = make_section("References", [make_paragraph(text, page=1, idx=1)], sec_type=SectionType.REFERENCES)
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 1
        assert "https://doi.org/10.1145/1234567" in result.chunks[0].content
        assert "$$\\nabla f(x) = 0$$" in result.chunks[0].content

    # 18. Deterministic chunk indices and SHA-256 hashes
    def test_deterministic_chunk_indices_and_hashes(self, engine: ResearchChunkingEngine) -> None:
        """Identical inputs produce identical chunk indices and stable SHA-256 hashes."""
        p1 = make_paragraph("Deterministic content paragraph.", page=1, idx=1)
        sec = make_section("1. Determinism", [p1])
        det = make_detection_result(sections=[sec])

        res1 = engine.chunk(det)
        res2 = engine.chunk(det)

        assert len(res1.chunks) == len(res2.chunks) == 1
        assert res1.chunks[0].content_hash == res2.chunks[0].content_hash
        expected_hash = hashlib.sha256(p1.text.encode("utf-8")).hexdigest()
        assert res1.chunks[0].content_hash == expected_hash

    # 19. Configuration validation
    def test_configuration_validation(self) -> None:
        """Raises ValueError on non-positive limits, invalid maximums, or out-of-bound overlaps."""
        with pytest.raises(ValueError, match="target_chunk_size must be positive"):
            ChunkingConfig(target_chunk_size=0)

        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            ChunkingConfig(max_chunk_size=-5)

        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            ChunkingConfig(min_chunk_size=0)

        with pytest.raises(ValueError, match="overlap must be non-negative"):
            ChunkingConfig(overlap=-1)

        with pytest.raises(ValueError, match="must be >= target_chunk_size"):
            ChunkingConfig(target_chunk_size=100, max_chunk_size=80)

        with pytest.raises(ValueError, match="must be < max_chunk_size"):
            ChunkingConfig(target_chunk_size=100, max_chunk_size=100, overlap=100)

        with pytest.raises(ValueError, match="must be <= target_chunk_size"):
            ChunkingConfig(target_chunk_size=50, min_chunk_size=80)

    # 20. Token limit enforcement
    def test_token_limit_enforcement(self, engine: ResearchChunkingEngine) -> None:
        """No chunk exceeds max_chunk_size in tokens."""
        paras = [make_paragraph(f"Paragraph number {i} with several words for testing.", page=1, idx=i) for i in range(10)]
        sec = make_section("1. Section", paras)
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        for c in result.chunks:
            assert c.token_count <= engine.config.max_chunk_size

    # 21. No empty chunks generated
    def test_no_empty_chunks(self, engine: ResearchChunkingEngine) -> None:
        """Empty or whitespace-only paragraphs do not create blank chunks."""
        p1 = make_paragraph("   ", page=1, idx=0)
        p2 = make_paragraph("", page=1, idx=1)
        p3 = make_paragraph("Real content here.", page=1, idx=2)
        sec = make_section("1. Section", [p1, p2, p3])
        det = make_detection_result(sections=[sec])
        result = engine.chunk(det)

        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Real content here."
        assert len(result.chunks[0].content.strip()) > 0

    # 22. No input mutation
    def test_no_input_mutation(self, engine: ResearchChunkingEngine) -> None:
        """Input SectionDetectionResult and nested objects remain unmodified."""
        p = make_paragraph("Immutable text.", page=1, idx=1)
        sec = make_section("1. Section", [p])
        det = make_detection_result(sections=[sec])
        deep_copied = copy.deepcopy(det)

        _ = engine.chunk(det)

        assert det == deep_copied

    # 23. Empty sections handling
    def test_empty_sections_handling(self, engine: ResearchChunkingEngine) -> None:
        """Sections with zero body paragraphs are processed cleanly without generating empty chunks."""
        sec_empty = make_section("1. Empty Heading", [])
        sec_normal = make_section("2. Normal", [make_paragraph("Normal text.", page=1, idx=1)], sec_id="sec_2", sec_idx=1)
        det = make_detection_result(sections=[sec_empty, sec_normal])
        result = engine.chunk(det)

        assert len(result.chunks) == 1
        assert result.chunks[0].section_title == "2. Normal"

    # 24. Short final chunks allowed
    def test_short_final_chunks(self) -> None:
        """Remainder chunks smaller than min_chunk_size are preserved rather than discarded."""
        cfg = ChunkingConfig(target_chunk_size=10, max_chunk_size=12, overlap=2, min_chunk_size=5)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)
        p = make_paragraph("Short final text.", page=1, idx=1)  # 3 words < min_chunk_size 5
        sec = make_section("1. Short", [p])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Short final text."
        assert result.chunks[0].token_count < eng.config.min_chunk_size

    # 25. No infinite loops with edge-case overlap settings
    def test_no_infinite_loops_with_edge_case_overlap(self) -> None:
        """Engine makes forward progress even when overlap equals target size minus 1."""
        cfg = ChunkingConfig(target_chunk_size=5, max_chunk_size=6, overlap=4, min_chunk_size=1)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        paras = [make_paragraph(f"Word{i}", page=1, idx=i) for i in range(10)]
        sec = make_section("1. Progress", paras)
        det = make_detection_result(sections=[sec])

        result = eng.chunk(det)
        assert len(result.chunks) > 0
        # Indices must strictly increase
        for i, c in enumerate(result.chunks):
            assert c.chunk_index == i

    # 26. Source text reconstruction verification
    def test_source_text_reconstruction(self) -> None:
        """Verifies that all non-overlap text reconstructs source paragraphs using engine metadata."""
        cfg = ChunkingConfig(target_chunk_size=10, max_chunk_size=15, overlap=0, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        p1_text = "This is paragraph number one with several words."
        p2_text = "This is paragraph number two with other words."
        p1 = make_paragraph(p1_text, page=1, idx=0)
        p2 = make_paragraph(p2_text, page=1, idx=1)

        sec = make_section("1. Recon", [p1, p2])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        expected = f"{p1_text}\n\n{p2_text}"
        assert result.reconstruct_text() == expected
        assert "".join(c.metadata["boundary_separator"] + c.content for c in result.chunks) == expected
        assert p1_text in result.chunks[0].content
        assert p2_text in result.chunks[1].content

    # 27. CharacterTokenCounter verification
    def test_character_token_counter(self) -> None:
        """CharacterTokenCounter counts tokens and splits text deterministically."""
        counter = CharacterTokenCounter(chars_per_token=4)
        assert counter.count_tokens("") == 0
        assert counter.count_tokens("abcd") == 1
        assert counter.count_tokens("abcde") == 2

        pieces = counter.split_text_to_token_limit("abcdefghij", max_tokens=2)
        assert pieces == ["abcdefgh", "ij"]
        assert "".join(pieces) == "abcdefghij"

    # 28. Exact source text reconstruction with whitespace, tabs, and newlines
    def test_exact_source_text_reconstruction_with_whitespace_and_tabs(self) -> None:
        """Splitting oversized content preserves all characters, double spaces, and tabs."""
        cfg = ChunkingConfig(target_chunk_size=4, max_chunk_size=5, overlap=0, min_chunk_size=1)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        complex_text = "Word1   Word2\tWord3.  Word4   Word5\t\tWord6.  Word7   Word8."
        p = make_paragraph(complex_text, page=1, idx=0)
        sec = make_section("1. Complex", [p])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) >= 2
        # Reconstructing across overlap-free chunks reproduces the original string character-for-character
        reconstructed = "".join(c.content for c in result.chunks)
        assert reconstructed == complex_text

    # 29. Intra-paragraph vs inter-paragraph boundary distinctions
    def test_intra_paragraph_vs_inter_paragraph_boundaries(self) -> None:
        """Intra-paragraph sentences are not separated by \\n\\n while distinct paragraphs are."""
        cfg = ChunkingConfig(target_chunk_size=20, max_chunk_size=30, overlap=0, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        para1_text = "Sentence one of p1. Sentence two of p1."
        para2_text = "Sentence one of p2. Sentence two of p2."
        p1 = make_paragraph(para1_text, page=1, idx=0)
        p2 = make_paragraph(para2_text, page=1, idx=1)

        sec = make_section("1. Boundaries", [p1, p2])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 1
        content = result.chunks[0].content
        # Paragraphs separated by \n\n
        assert "\n\n" in content
        # Internal sentences within p1 and p2 have their single space, never \n\n
        assert "Sentence one of p1. Sentence two of p1." in content
        assert "Sentence one of p2. Sentence two of p2." in content
        assert content == f"{para1_text}\n\n{para2_text}"

    # 30. Hard maximum token limit strictly enforced
    def test_hard_token_limit_strict_enforcement(self) -> None:
        """Every chunk strictly obeys max_chunk_size across varying paragraph sizes."""
        cfg = ChunkingConfig(target_chunk_size=7, max_chunk_size=8, overlap=3, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        paras = [
            make_paragraph("Word " * 15, page=1, idx=0),
            make_paragraph("Another sentence with five words.", page=1, idx=1),
            make_paragraph("Third " * 20, page=2, idx=2),
        ]
        sec = make_section("1. Strict Limits", paras)
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) > 3
        for c in result.chunks:
            assert c.token_count <= cfg.max_chunk_size

    # 31. TokenCounter protocol validation
    def test_token_counter_contract_validation(self) -> None:
        """Engine raises ValueError when a custom token counter violates protocol contracts."""
        cfg = ChunkingConfig(target_chunk_size=5, max_chunk_size=6, overlap=0, min_chunk_size=2)

        class BrokenDropCharCounter:
            def count_tokens(self, text: str) -> int:
                return len(text.split())

            def split_text_to_token_limit(self, text: str, max_tokens: int) -> list[str]:
                # Violates contract: drops characters
                return ["truncated"]

        eng_broken_drop = ResearchChunkingEngine(token_counter=BrokenDropCharCounter(), config=cfg)
        p = make_paragraph("Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8", page=1, idx=0)
        det = make_detection_result(sections=[make_section("1. Test", [p])])

        with pytest.raises(ValueError, match="dropped or altered characters"):
            eng_broken_drop.chunk(det)

        class BrokenOversizedPieceCounter:
            def count_tokens(self, text: str) -> int:
                return len(text.split())

            def split_text_to_token_limit(self, text: str, max_tokens: int) -> list[str]:
                # Violates contract: returns piece exceeding max_tokens
                return [text]

        eng_broken_over = ResearchChunkingEngine(token_counter=BrokenOversizedPieceCounter(), config=cfg)
        with pytest.raises(ValueError, match="exceeding max_chunk_size"):
            eng_broken_over.chunk(det)

        class BrokenEmptyCounter:
            def count_tokens(self, text: str) -> int:
                return len(text.split())

            def split_text_to_token_limit(self, text: str, max_tokens: int) -> list[str]:
                # Violates contract: returns empty list
                return []

        eng_broken_empty = ResearchChunkingEngine(token_counter=BrokenEmptyCounter(), config=cfg)
        with pytest.raises(ValueError, match="empty list"):
            eng_broken_empty.chunk(det)

    # 32. Overlap trimming when candidate chunk exceeds max_chunk_size
    def test_overlap_trimming_when_exceeding_max_chunk_size(self) -> None:
        """Overlap is trimmed from the front so that new units fit within max_chunk_size."""
        cfg = ChunkingConfig(target_chunk_size=7, max_chunk_size=8, overlap=4, min_chunk_size=1)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        # First two paragraphs (3 words + 3 words = 6 words) form the first chunk
        p1 = make_paragraph("One two three.", page=1, idx=0)
        p2 = make_paragraph("Four five six.", page=1, idx=1)
        # Third paragraph is 7 words. Second chunk initially selects p2 (3 words) as overlap,
        # but 3 + 7 = 10 > 8 (max_chunk_size), forcing the trimming loop to pop p2.
        p3 = make_paragraph("Alpha beta gamma delta epsilon zeta eta.", page=1, idx=2)

        sec = make_section("1. Trim", [p1, p2, p3])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 2
        assert result.chunks[0].token_count == 6
        assert result.chunks[1].token_count == 7
        assert result.chunks[1].metadata["overlap_tokens"] == 0
        assert result.diagnostics.total_overlap_tokens == 0
        assert result.chunks[0].token_count <= cfg.max_chunk_size
        assert result.chunks[1].token_count <= cfg.max_chunk_size

    # 33. min_chunk_size diagnostic warnings
    def test_min_chunk_size_diagnostics_warning(self) -> None:
        """Emits diagnostic warning for non-final undersized chunks; suppresses for final chunks."""
        cfg = ChunkingConfig(target_chunk_size=10, max_chunk_size=10, overlap=0, min_chunk_size=5)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        # Para 1 has 2 words (< min_chunk_size 5).
        # Para 2 has 10 words (= max_chunk_size 10), so Para 2 cannot fit with Para 1 (2 + 10 = 12 > 10).
        # Para 1 is emitted as a standalone chunk, and it is NON-FINAL in the section.
        p1 = make_paragraph("Tiny chunk.", page=1, idx=0)  # 2 words
        p2 = make_paragraph("One two three four five six seven eight nine ten.", page=1, idx=1)  # 10 words
        p3 = make_paragraph("Final short.", page=1, idx=2)  # 2 words (final chunk in section)

        sec = make_section("1. Warnings", [p1, p2, p3])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 3
        assert result.chunks[0].token_count == 2
        assert result.chunks[1].token_count == 10
        assert result.chunks[2].token_count == 2

        # A warning is emitted for Chunk 0 (non-final undersized)
        assert len(result.diagnostics.warnings) == 1
        assert "undersized" in result.diagnostics.warnings[0]
        assert "chunk 0" in result.diagnostics.warnings[0].lower()
        # Final chunk 2 (also 2 words) did NOT emit a warning

    # 34. HuggingFaceTokenCounter offline adapter
    def test_huggingface_token_counter_offline(self) -> None:
        """HuggingFaceTokenCounter wraps pre-loaded tokenizers offline and splits losslessly."""
        class MockHFTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                if not text:
                    return []
                # Simple word-based encoding simulation
                return [i + 1 for i in range(len(text.split()))]

        mock_tok = MockHFTokenizer()
        counter = HuggingFaceTokenCounter(mock_tok)

        assert isinstance(counter, TokenCounter)
        assert counter.count_tokens("") == 0
        assert counter.count_tokens("one two three") == 3

        text = "Alpha beta gamma delta epsilon zeta."
        pieces = counter.split_text_to_token_limit(text, max_tokens=2)
        assert len(pieces) == 3
        assert "".join(pieces) == text
        for piece in pieces:
            assert counter.count_tokens(piece) <= 2

        # Tokenize fallback method
        class MockTokenizeOnlyTokenizer:
            def tokenize(self, text: str) -> list[str]:
                return text.split()

        tok_only_counter = HuggingFaceTokenCounter(MockTokenizeOnlyTokenizer())
        assert tok_only_counter.count_tokens("hello world") == 2

        with pytest.raises(ValueError, match="tokenizer must not be None"):
            HuggingFaceTokenCounter(None)

    # 35. Input paragraph diagnostics ignores empty and whitespace-only paragraphs
    def test_input_paragraph_diagnostics_ignores_empty_and_whitespace(self) -> None:
        """input_paragraph_count counts only non-empty paragraphs, ignoring empty and whitespace."""
        cfg = ChunkingConfig(target_chunk_size=10, max_chunk_size=12, overlap=0, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        p_empty = make_paragraph("", page=1, idx=0)
        p_whitespace = make_paragraph("   \n\t  ", page=1, idx=1)
        p_valid = make_paragraph("This is a valid non-empty paragraph.", page=1, idx=2)

        sec = make_section("1. Diagnostics", [p_empty, p_whitespace, p_valid])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert result.diagnostics.input_paragraph_count == 1
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "This is a valid non-empty paragraph."

    # 36. Reconstruction: Two paragraphs in one chunk
    def test_reconstruction_two_paragraphs_in_one_chunk(self) -> None:
        """Two paragraphs combined in a single chunk are separated by \\n\\n and reconstructed."""
        cfg = ChunkingConfig(target_chunk_size=10, max_chunk_size=12, overlap=0, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        p1 = make_paragraph("First paragraph text.", page=1, idx=0)
        p2 = make_paragraph("Second paragraph text.", page=1, idx=1)
        sec = make_section("1. Section", [p1, p2])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 1
        assert result.chunks[0].metadata["boundary_separator"] == ""
        assert result.chunks[0].metadata["prefix_separator"] == ""
        assert result.chunks[0].content == "First paragraph text.\n\nSecond paragraph text."
        assert result.reconstruct_text() == "First paragraph text.\n\nSecond paragraph text."
        assert "".join(c.metadata["boundary_separator"] + c.content for c in result.chunks) == "First paragraph text.\n\nSecond paragraph text."

    # 37. Reconstruction: Two paragraphs in separate chunks
    def test_reconstruction_two_paragraphs_in_separate_chunks(self) -> None:
        """Two paragraphs in separate chunks preserve \\n\\n boundary via boundary_separator metadata."""
        cfg = ChunkingConfig(target_chunk_size=5, max_chunk_size=5, overlap=0, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        p1 = make_paragraph("First paragraph with five words.", page=1, idx=0)  # 5 words
        p2 = make_paragraph("Second paragraph with five words.", page=1, idx=1)  # 5 words
        sec = make_section("1. Section", [p1, p2])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 2
        assert result.chunks[0].metadata["boundary_separator"] == ""
        assert result.chunks[1].metadata["boundary_separator"] == "\n\n"
        assert result.chunks[1].metadata["prefix_separator"] == "\n\n"

        expected = "First paragraph with five words.\n\nSecond paragraph with five words."
        assert result.reconstruct_text() == expected
        assert "".join(c.metadata["boundary_separator"] + c.content for c in result.chunks) == expected

    # 38. Reconstruction: One oversized paragraph split across chunks
    def test_reconstruction_one_oversized_paragraph_split_across_chunks(self) -> None:
        """An oversized paragraph split across chunks does NOT have \\n\\n inserted between its pieces."""
        cfg = ChunkingConfig(target_chunk_size=6, max_chunk_size=7, overlap=0, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        sentence1 = "Sentence one with exactly six words."  # 6 words
        sentence2 = "Sentence two with exactly six words."  # 6 words
        full_text = f"{sentence1} {sentence2}"
        p = make_paragraph(full_text, page=1, idx=0)

        sec = make_section("1. Oversized", [p])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert len(result.chunks) == 2
        # First chunk starts at document start
        assert result.chunks[0].metadata["boundary_separator"] == ""
        # Second chunk is intra-paragraph continuation: boundary_separator must be empty string
        assert result.chunks[1].metadata["boundary_separator"] == ""
        assert result.chunks[1].metadata["prefix_separator"] == ""

        assert result.reconstruct_text() == full_text
        assert "".join(c.metadata["boundary_separator"] + c.content for c in result.chunks) == full_text

    # 39. Reconstruction: Multiple paragraphs where one paragraph is split
    def test_reconstruction_multiple_paragraphs_with_one_split(self) -> None:
        """Boundary separators differentiate paragraph breaks from intra-paragraph splits accurately."""
        cfg = ChunkingConfig(target_chunk_size=6, max_chunk_size=7, overlap=0, min_chunk_size=1)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        p1 = make_paragraph("Intro paragraph here.", page=1, idx=0)  # 3 words
        p2_s1 = "Body paragraph sentence one is six."        # 6 words
        p2_s2 = "Body paragraph sentence two is six."        # 6 words
        p2 = make_paragraph(f"{p2_s1} {p2_s2}", page=1, idx=1)  # 12 words (split)
        p3 = make_paragraph("Conclusion paragraph here.", page=1, idx=2)  # 3 words

        sec = make_section("1. Mixed", [p1, p2, p3])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        # Expected chunks:
        # Chunk 0: p1 (3 words)
        # Chunk 1: p2_s1 (6 words)
        # Chunk 2: p2_s2 (6 words)
        # Chunk 3: p3 (3 words)
        assert len(result.chunks) == 4

        boundary_seps = [c.metadata["boundary_separator"] for c in result.chunks]
        assert boundary_seps == ["", "\n\n", "", "\n\n"]

        expected = f"{p1.text}\n\n{p2.text}\n\n{p3.text}"
        assert result.reconstruct_text() == expected
        assert "".join(c.metadata["boundary_separator"] + c.content for c in result.chunks) == expected

    # 40. Reconstruction: Exact reconstruction with overlap disabled
    def test_reconstruction_exact_with_overlap_disabled(self) -> None:
        """Deterministic reconstruction of complex multi-section document with overlap=0."""
        cfg = ChunkingConfig(target_chunk_size=8, max_chunk_size=10, overlap=0, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        preamble = [make_paragraph("Paper Title: Deep Retrieval Exploration", page=1, idx=0)]
        sec1_p1 = make_paragraph("First section paragraph one with seven words.", page=1, idx=1)
        sec1_p2 = make_paragraph("First section paragraph two with seven words.", page=1, idx=2)
        sec1 = make_section("1. Intro", [sec1_p1, sec1_p2], sec_id="sec_1", sec_idx=0)

        sec2_p1 = make_paragraph("Second section oversized sentence one. Second section oversized sentence two.", page=2, idx=3)
        sec2 = make_section("2. Methods", [sec2_p1], sec_id="sec_2", sec_idx=1)

        det = make_detection_result(sections=[sec1, sec2], preamble=preamble)
        result = eng.chunk(det)

        expected_full_text = (
            f"{preamble[0].text}\n\n"
            f"{sec1_p1.text}\n\n"
            f"{sec1_p2.text}\n\n"
            f"{sec2_p1.text}"
        )
        assert result.reconstruct_text() == expected_full_text
        assert "".join(c.metadata["boundary_separator"] + c.content for c in result.chunks) == expected_full_text

    # 41. Reconstruction raises ValueError when overlap is enabled
    def test_reconstruction_raises_error_with_overlap_enabled(self) -> None:
        """reconstruct_text raises ValueError if any chunk has overlap_tokens > 0."""
        cfg = ChunkingConfig(target_chunk_size=8, max_chunk_size=10, overlap=4, min_chunk_size=2)
        eng = ResearchChunkingEngine(token_counter=WhitespaceTokenCounter(), config=cfg)

        p1 = make_paragraph("Alpha beta gamma delta.", page=1, idx=0)
        p2 = make_paragraph("Epsilon zeta eta theta.", page=1, idx=1)
        p3 = make_paragraph("Iota kappa lambda mu.", page=1, idx=2)

        sec = make_section("1. Section", [p1, p2, p3])
        det = make_detection_result(sections=[sec])
        result = eng.chunk(det)

        assert any(c.metadata.get("overlap_tokens", 0) > 0 for c in result.chunks)
        with pytest.raises(ValueError, match="requires overlap-free chunks"):
            result.reconstruct_text()
