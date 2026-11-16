"""Unit tests for Research Paper Section Detection and Hierarchy Construction.

Tests cover:
- Standard academic paper section flow and classification.
- Numbered headings (decimal, roman numerals, lettered appendices).
- Structural hierarchy stack, parent section IDs, and subsection nesting.
- Hierarchy gaps and associated diagnostic warnings.
- Section continuing across multiple pages, and parent span propagation.
- Unknown/custom heading detection and classification as CUSTOM.
- Preamble handling for content before first heading.
- Sections without body paragraphs.
- Document with zero headings and empty documents.
- False positive rejection (ordinary prose, lists, citations, URLs, equations, references).
- Critical paragraph accounting invariant: no paragraph lost, none duplicated.
- Input immutability and type validation.
"""

from __future__ import annotations

import copy
import pytest

from app.modules.ingestion.preprocessor import (
    PreprocessedPage,
    PreprocessedResearchPaper,
    PreprocessingDiagnostics,
)
from app.modules.ingestion.section_detector import (
    DetectedSection,
    ResearchSectionDetector,
    SectionDetectionDiagnostics,
    SectionDetectionResult,
    SectionParagraph,
    SectionType,
)


def make_preprocessed_paper(
    pages_paragraphs: list[list[str]],
) -> PreprocessedResearchPaper:
    """Helper to build in-memory PreprocessedResearchPaper with specific paragraphs per page."""
    pages = [
        PreprocessedPage(
            page_number=idx + 1,
            original_text="\n\n".join(paras),
            cleaned_text="\n\n".join(paras),
            paragraphs=list(paras),
        )
        for idx, paras in enumerate(pages_paragraphs)
    ]
    return PreprocessedResearchPaper(
        pages=pages,
        full_text="\n\n\n".join(p.cleaned_text for p in pages if p.cleaned_text.strip()),
        diagnostics=PreprocessingDiagnostics(
            total_pages=len(pages),
            modified_pages=0,
            removed_repeated_lines=[],
            warnings=[],
        ),
    )


class TestResearchSectionDetector:
    """Comprehensive test suite for ResearchSectionDetector."""

    @pytest.fixture
    def detector(self) -> ResearchSectionDetector:
        """Fixture returning standard ResearchSectionDetector."""
        return ResearchSectionDetector()

    # 1. Standard academic paper structure
    def test_standard_academic_paper_structure(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Correctly detects and classifies full canonical research paper section flow."""
        pages = [
            [
                "Paper Title and Author Block",
                "Abstract",
                "This paper investigates neural symbolic reasoning for NLP.",
                "1. Introduction",
                "Large language models have achieved significant progress in benchmark reasoning.",
            ],
            [
                "2. Related Work",
                "Earlier studies examined symbolic logic integrations in recurrent networks.",
                "3. Methodology",
                "We introduce a neuro-symbolic loss function over discrete predicates.",
            ],
            [
                "4. Experiments",
                "Experiments are conducted on three standard reasoning benchmarks.",
                "5. Results",
                "Our model outperforms baseline systems by 4.2 points in accuracy.",
                "6. Discussion",
                "The findings confirm that symbolic constraints regularize latent features.",
                "7. Conclusion",
                "We presented a unified neuro-symbolic framework for deep NLP.",
                "References",
                "[1] Vaswani et al. Attention is all you need. NeurIPS, 2017.",
            ],
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert isinstance(result, SectionDetectionResult)
        assert len(result.sections) == 9

        expected_types = [
            SectionType.ABSTRACT,
            SectionType.INTRODUCTION,
            SectionType.RELATED_WORK,
            SectionType.METHODOLOGY,
            SectionType.EXPERIMENTS,
            SectionType.RESULTS,
            SectionType.DISCUSSION,
            SectionType.CONCLUSION,
            SectionType.REFERENCES,
        ]
        actual_types = [s.section_type for s in result.sections]
        assert actual_types == expected_types

        # Verify preamble captured the initial title/author block
        assert len(result.preamble) == 1
        assert result.preamble[0].text == "Paper Title and Author Block"

    # 2. Numbered headings and nested headings
    def test_numbered_and_nested_headings(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Establishes hierarchy levels 1, 2, 3 and parent IDs for nested decimal sections."""
        pages = [
            [
                "1. Introduction",
                "Introductory prose.",
                "1.1 Background",
                "Background prose.",
                "1.1.1 Theoretical Foundations",
                "Deep theory prose.",
                "1.2 Motivation",
                "Motivation prose.",
                "2. Methodology",
                "Methods prose.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 5

        sec1 = result.sections[0]
        sec1_1 = result.sections[1]
        sec1_1_1 = result.sections[2]
        sec1_2 = result.sections[3]
        sec2 = result.sections[4]

        # Levels
        assert sec1.level == 1
        assert sec1_1.level == 2
        assert sec1_1_1.level == 3
        assert sec1_2.level == 2
        assert sec2.level == 1

        # Parent relationships
        assert sec1.parent_section_id is None
        assert sec1_1.parent_section_id == sec1.section_id
        assert sec1_1_1.parent_section_id == sec1_1.section_id
        assert sec1_2.parent_section_id == sec1.section_id
        assert sec2.parent_section_id is None

        # Direct paragraphs (no duplication in parents)
        assert len(sec1.paragraphs) == 1
        assert sec1.paragraphs[0].text == "Introductory prose."
        assert len(sec1_1.paragraphs) == 1
        assert sec1_1.paragraphs[0].text == "Background prose."
        assert len(sec1_1_1.paragraphs) == 1
        assert sec1_1_1.paragraphs[0].text == "Deep theory prose."

    # 3. Roman numeral headings
    def test_roman_numeral_headings(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Detects and classifies IEEE-style Roman numeral headings as top-level sections."""
        pages = [
            [
                "I. INTRODUCTION",
                "Introduction body text.",
                "II. RELATED WORK",
                "Related work body text.",
                "III. PROPOSED APPROACH",
                "Approach body text.",
                "IV. EXPERIMENTAL EVALUATION",
                "Evaluation body text.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 4
        assert [s.level for s in result.sections] == [1, 1, 1, 1]
        assert [s.parent_section_id for s in result.sections] == [None, None, None, None]
        assert result.sections[0].section_type == SectionType.INTRODUCTION
        assert result.sections[1].section_type == SectionType.RELATED_WORK
        assert result.sections[2].section_type == SectionType.METHODOLOGY
        assert result.sections[3].section_type == SectionType.EXPERIMENTS

    # 4. Lettered appendices
    def test_lettered_appendices(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Detects lettered appendices and appendix subsections."""
        pages = [
            [
                "Appendix A: Proof of Main Theorems",
                "Detailed mathematical proofs.",
                "A.1 Lemma on Bounded Variance",
                "Proof of Lemma 1.",
                "Appendix B: Hyperparameter Tuning",
                "Grid search configuration table.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 3
        app_a = result.sections[0]
        sub_a1 = result.sections[1]
        app_b = result.sections[2]

        assert app_a.section_type == SectionType.APPENDIX
        assert app_a.level == 1
        assert app_a.parent_section_id is None

        assert sub_a1.level == 2
        assert sub_a1.parent_section_id == app_a.section_id

        assert app_b.section_type == SectionType.APPENDIX
        assert app_b.level == 1
        assert app_b.parent_section_id is None

    # 5. Unknown/custom headings
    def test_custom_domain_specific_headings(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Classifies confidently detected domain-specific headings as CUSTOM."""
        pages = [
            [
                "1. Introduction",
                "Intro text.",
                "1.1 Dataset Preparation and Cleaning",
                "Details on web scrape cleaning.",
                "1.2 Neural Radiance Fields Setup",
                "Ray casting camera configurations.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 3
        assert result.sections[0].section_type == SectionType.INTRODUCTION
        assert result.sections[1].section_type == SectionType.CUSTOM
        assert result.sections[1].normalized_title == "dataset preparation and cleaning"
        assert result.sections[2].section_type == SectionType.CUSTOM
        assert result.sections[2].normalized_title == "neural radiance fields setup"
        assert result.diagnostics.unclassified_headings == 2

    # 6. Section continuing across pages
    def test_section_continuing_across_pages(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Correctly expands section start_page and end_page across multiple pages."""
        pages = [
            [
                "3. Methodology",
                "Paragraph 1 of methodology on page 1.",
            ],
            [
                "Paragraph 2 of methodology on page 2.",
                "Paragraph 3 of methodology on page 2.",
            ],
            [
                "Paragraph 4 of methodology on page 3.",
                "4. Results",
                "First results paragraph on page 3.",
            ],
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 2
        method_sec = result.sections[0]
        results_sec = result.sections[1]

        assert method_sec.heading_page == 1
        assert method_sec.start_page == 1
        assert method_sec.end_page == 3
        assert len(method_sec.paragraphs) == 4

        assert results_sec.heading_page == 3
        assert results_sec.start_page == 3
        assert results_sec.end_page == 3
        assert len(results_sec.paragraphs) == 1

    # 7. Heading at the beginning of a new page
    def test_heading_at_beginning_of_new_page(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Correctly associates heading appearing as first paragraph on a page."""
        pages = [
            [
                "1. Introduction",
                "Introductory paragraph.",
            ],
            [
                "2. Related Work",
                "Related work discussion.",
            ],
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 2
        assert result.sections[1].title == "2. Related Work"
        assert result.sections[1].heading_page == 2
        assert result.sections[1].start_page == 2
        assert result.sections[1].end_page == 2

    # 8. Multiple sections with the same title
    def test_multiple_sections_with_same_title(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Assigns distinct section IDs to repeated titles without collision."""
        pages = [
            [
                "1. Model A",
                "Analysis",
                "Model A analysis text.",
                "2. Model B",
                "Analysis",
                "Model B analysis text.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 4
        analysis_sections = [s for s in result.sections if s.normalized_title == "analysis"]
        assert len(analysis_sections) == 2
        assert analysis_sections[0].section_id != analysis_sections[1].section_id
        assert analysis_sections[0].paragraphs[0].text == "Model A analysis text."
        assert analysis_sections[1].paragraphs[0].text == "Model B analysis text."

    # 9. Heading without body paragraphs
    def test_heading_without_body_paragraphs(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Preserves parent heading followed immediately by subsection heading."""
        pages = [
            [
                "3. Proposed Architecture",
                "3.1 Encoder Subnetwork",
                "Body text for encoder.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 2
        parent = result.sections[0]
        child = result.sections[1]

        assert parent.title == "3. Proposed Architecture"
        assert parent.paragraphs == []
        assert child.parent_section_id == parent.section_id
        assert len(child.paragraphs) == 1

        # Parent span inherits child's span
        assert parent.start_page == 1
        assert parent.end_page == 1

        # Warning generated for empty body paragraphs
        assert any("has no body paragraphs" in w for w in result.diagnostics.warnings)

    # 10. Text before the first heading
    def test_preamble_content_before_first_heading(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Assigns non-heading content before the first heading to preamble."""
        pages = [
            [
                "A Study of Neural Reasoning",
                "Author Name and Affiliation",
                "Correspondence: author@university.edu",
                "1. Introduction",
                "Main paper body text.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.preamble) == 3
        assert result.preamble[0].text == "A Study of Neural Reasoning"
        assert result.preamble[1].text == "Author Name and Affiliation"
        assert result.preamble[2].text == "Correspondence: author@university.edu"

        assert len(result.sections) == 1
        assert len(result.sections[0].paragraphs) == 1

    # 11. Document with no detected headings
    def test_document_with_no_detected_headings(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Assigns all paragraphs to preamble when no headings are found."""
        pages = [
            [
                "This is an unformatted meeting memo with no headings.",
                "Second paragraph detailing team discussions.",
                "Third paragraph listing action items.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 0
        assert len(result.preamble) == 3
        assert result.diagnostics.detected_headings == 0
        assert any("No section headings were detected" in w for w in result.diagnostics.warnings)

    # 12. Empty document
    def test_empty_document(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Handles empty pages list or pages with empty paragraph lists safely."""
        paper = make_preprocessed_paper([])
        result = detector.detect(paper)

        assert len(result.sections) == 0
        assert len(result.preamble) == 0
        assert result.diagnostics.total_paragraphs == 0

    # 13. References and bibliography entries
    def test_references_and_bibliography_entries(
        self, detector: ResearchSectionDetector
    ) -> None:
        """References heading is detected while citation entries are preserved as body paragraphs."""
        pages = [
            [
                "References",
                "[1] Vaswani, A., et al. Attention is all you need. NeurIPS 2017.",
                "[2] Devlin, J., et al. BERT: Pre-training deep bidirectional transformers. NAACL 2019.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        ref_sec = result.sections[0]
        assert ref_sec.section_type == SectionType.REFERENCES
        assert len(ref_sec.paragraphs) == 2
        assert ref_sec.paragraphs[0].text.startswith("[1] Vaswani")
        assert ref_sec.paragraphs[1].text.startswith("[2] Devlin")

    # 14. Acknowledgments and appendices
    def test_acknowledgments_and_appendices(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Classifies standard acknowledgment variations and appendices."""
        pages = [
            [
                "Acknowledgements",
                "This research was supported by National Science Foundation Grant 123456.",
                "Appendix",
                "Additional supplementary tables and figures.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 2
        assert result.sections[0].section_type == SectionType.ACKNOWLEDGMENTS
        assert result.sections[1].section_type == SectionType.APPENDIX

    # 15. Ordinary short paragraphs rejected as headings
    def test_ordinary_short_paragraphs_rejected(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Short sentences ending with periods are rejected as headings and kept as body."""
        pages = [
            [
                "1. Introduction",
                "This is brief.",
                "Another short remark.",
                "Yet another note.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        assert len(result.sections[0].paragraphs) == 3
        assert result.sections[0].paragraphs[0].text == "This is brief."
        assert result.diagnostics.rejected_candidates >= 1

    # 16. Numbered lists that resemble headings
    def test_numbered_lists_rejected_as_headings(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Numbered list items ending in punctuation are not mistaken for section headings."""
        pages = [
            [
                "1. Introduction",
                "Our approach addresses three core challenges:",
                "* 1. High computational memory footprint in attention maps.",
                "* 2. Slow training convergence in deep layers.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        assert len(result.sections[0].paragraphs) == 3
        assert "* 1. High computational" in result.sections[0].paragraphs[1].text

    # 17. Citation-heavy paragraphs rejected as headings
    def test_citation_heavy_paragraphs_rejected(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Paragraphs containing inline citations are rejected as headings."""
        pages = [
            [
                "2. Background",
                "Recent progress was documented in (Brown et al., 2020) and [12, 14].",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        assert len(result.sections[0].paragraphs) == 1
        assert "Recent progress was documented in (Brown et al., 2020)" in result.sections[0].paragraphs[0].text

    # 18. URLs and DOI identifiers rejected as headings
    def test_urls_and_doi_rejected_as_headings(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Standalone URL and DOI lines are rejected as headings."""
        pages = [
            [
                "1. Introduction",
                "https://github.com/organization/nlp-research-agent",
                "10.1145/3308558.3313528",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        assert len(result.sections[0].paragraphs) == 2

    # 19. Equations and math syntax rejected as headings
    def test_equations_rejected_as_headings(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Math equations are not mistaken for headings."""
        pages = [
            [
                "3. Methodology",
                "$$L(\\theta) = - \\sum_{i=1}^N \\log P(y_i | x_i; \\theta)$$",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        assert len(result.sections[0].paragraphs) == 1
        assert "$$L(\\theta)" in result.sections[0].paragraphs[0].text

    # 20. Hierarchy numbering gaps
    def test_hierarchy_numbering_gaps(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Handles numbering hierarchy gaps gracefully and records warning in diagnostics."""
        pages = [
            [
                "1. Introduction",
                "Intro text.",
                "1.1.1 Deep Subsection Gap",
                "Subsection text skipping level 2.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 2
        sec1 = result.sections[0]
        sec1_1_1 = result.sections[1]

        assert sec1.level == 1
        assert sec1_1_1.level == 3
        # Attached to nearest preceding lower level section
        assert sec1_1_1.parent_section_id == sec1.section_id

        # Warning recorded
        assert any("Hierarchy gap detected" in w for w in result.diagnostics.warnings)

    # 21. Critical paragraph accounting invariant
    def test_critical_paragraph_accounting_invariant(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Verifies that every single input paragraph is accounted for exactly once."""
        pages = [
            [
                "Preamble Paragraph 1",
                "Preamble Paragraph 2",
                "1. Introduction",
                "Intro Para 1",
                "Intro Para 2",
                "1.1 Motivation",
                "Motivation Para 1",
            ],
            [
                "2. Methodology",
                "Methodology Para 1",
                "2.1 Architecture",
                "Architecture Para 1",
                "Architecture Para 2",
            ],
        ]
        paper = make_preprocessed_paper(pages)
        total_input_paragraphs = sum(len(p.paragraphs) for p in paper.pages)

        result = detector.detect(paper)

        heading_count = len(result.sections)
        preamble_count = len(result.preamble)
        body_count = sum(len(s.paragraphs) for s in result.sections)

        total_accounted = heading_count + preamble_count + body_count
        assert total_accounted == total_input_paragraphs
        assert result.diagnostics.total_paragraphs == total_input_paragraphs

    # 22. Paragraph global indexing preservation
    def test_global_paragraph_indexing_preservation(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Preserves global paragraph indices across preamble and sections."""
        pages = [
            [
                "Preamble item",     # index 0
                "1. Introduction",   # index 1 (heading)
                "Intro body",        # index 2
            ],
            [
                "2. Methods",        # index 3 (heading)
                "Methods body 1",    # index 4
                "Methods body 2",    # index 5
            ],
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert result.preamble[0].paragraph_index == 0
        assert result.sections[0].paragraphs[0].paragraph_index == 2
        assert result.sections[1].paragraphs[0].paragraph_index == 4
        assert result.sections[1].paragraphs[1].paragraph_index == 5

    # 23. Input object immutability
    def test_input_object_immutability(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Verifies that the input PreprocessedResearchPaper is not mutated."""
        pages = [
            ["Title", "1. Introduction", "Body text."],
            ["2. Methods", "Methods text."],
        ]
        paper = make_preprocessed_paper(pages)
        deep_copied = copy.deepcopy(paper)

        detector.detect(paper)

        assert len(paper.pages) == len(deep_copied.pages)
        for i in range(len(paper.pages)):
            assert paper.pages[i].page_number == deep_copied.pages[i].page_number
            assert paper.pages[i].paragraphs == deep_copied.pages[i].paragraphs
            assert paper.pages[i].cleaned_text == deep_copied.pages[i].cleaned_text

    # 24. Confidence bounds
    def test_confidence_bounds(
        self, detector: ResearchSectionDetector
    ) -> None:
        """All detected sections have confidence scores strictly between 0.0 and 1.0."""
        pages = [
            [
                "1. Introduction",
                "Intro text.",
                "3.2 Dataset Preparation",
                "Prep text.",
                "REFERENCES",
                "[1] Ref entry.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        for s in result.sections:
            assert 0.0 <= s.confidence <= 1.0

    # 25. TypeError on invalid input type
    def test_type_error_on_invalid_input(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Raises TypeError when passed non-PreprocessedResearchPaper objects."""
        with pytest.raises(TypeError, match="Expected PreprocessedResearchPaper instance"):
            detector.detect("invalid string")  # type: ignore

        with pytest.raises(TypeError, match="Expected PreprocessedResearchPaper instance"):
            detector.detect(None)  # type: ignore

        with pytest.raises(TypeError, match="Expected PreprocessedResearchPaper instance"):
            detector.detect({"pages": []})  # type: ignore

    # 26. Numbered-list false positives reproduction case
    def test_numbered_list_false_positives_reproduction(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Sequential numbered list items inside a section are kept as body paragraphs and not CUSTOM sections."""
        pages = [
            [
                "1. Introduction",
                "1. High computational memory footprint",
                "2. Slow convergence during training",
                "3. Limited training data",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        intro = result.sections[0]
        assert intro.title == "1. Introduction"
        assert intro.section_type == SectionType.INTRODUCTION
        assert intro.level == 1
        assert len(intro.paragraphs) == 3

        # The three list items remain body paragraphs
        assert intro.paragraphs[0].text == "1. High computational memory footprint"
        assert intro.paragraphs[1].text == "2. Slow convergence during training"
        assert intro.paragraphs[2].text == "3. Limited training data"

        # None of them became CUSTOM sections
        custom_sections = [s for s in result.sections if s.section_type == SectionType.CUSTOM]
        assert len(custom_sections) == 0

    # 27. Sequential numbered list items without bullet markers
    def test_sequential_numbered_list_without_bullet_markers(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Sequential numbered list items without bullet markers remain body paragraphs."""
        pages = [
            [
                "2. Methodology",
                "Our pipeline consists of four sequential stages:",
                "1. Data ingestion and parsing",
                "2. Text normalization and cleaning",
                "3. Section detection and hierarchy extraction",
                "4. Chunking and indexing",
                "Subsequent sections explain each stage in detail.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        method = result.sections[0]
        assert method.title == "2. Methodology"
        assert len(method.paragraphs) == 6
        assert method.paragraphs[1].text == "1. Data ingestion and parsing"
        assert method.paragraphs[2].text == "2. Text normalization and cleaning"
        assert method.paragraphs[3].text == "3. Section detection and hierarchy extraction"
        assert method.paragraphs[4].text == "4. Chunking and indexing"

    # 28. Numbered list items containing full sentences rejected as headings
    def test_numbered_list_items_with_full_sentences_rejected(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Numbered list items containing complete sentences are rejected as headings."""
        pages = [
            [
                "1. Introduction",
                "1. The proposed algorithm converges in fewer epochs than baseline models.",
                "2. It requires significantly less GPU memory during distributed training.",
                "3. The learned representations generalize well across downstream domains.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        assert len(result.sections[0].paragraphs) == 3
        assert "converges in fewer epochs" in result.sections[0].paragraphs[0].text
        assert "significantly less GPU memory" in result.sections[0].paragraphs[1].text

    # 29. Numbered list items without sentence-ending punctuation rejected as headings
    def test_numbered_list_items_without_sentence_ending_punctuation_rejected(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Numbered list items lacking periods but using sentence-case wording are rejected as headings."""
        pages = [
            [
                "1. Introduction",
                "1. High computational overhead",
                "2. Memory bandwidth limitations",
                "3. Distributed communication bottlenecks",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        assert len(result.sections[0].paragraphs) == 3
        for p in result.sections[0].paragraphs:
            assert p.text.startswith(("1.", "2.", "3."))

    # 30. Legitimate custom numbered headings supported
    def test_legitimate_custom_numbered_heading_accepted(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Legitimate domain-specific custom numbered headings in title case are accepted."""
        pages = [
            [
                "1. Dataset Preparation",
                "We collected 100,000 multi-turn conversations from public repositories.",
                "2. Experimental Setup",
                "All models were evaluated on 8 NVIDIA H100 GPUs.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 2
        sec1 = result.sections[0]
        assert sec1.title == "1. Dataset Preparation"
        assert sec1.normalized_title == "dataset preparation"
        assert sec1.section_type == SectionType.CUSTOM
        assert sec1.level == 1
        assert len(sec1.paragraphs) == 1

        sec2 = result.sections[1]
        assert sec2.title == "2. Experimental Setup"
        assert sec2.section_type == SectionType.EXPERIMENTS
        assert sec2.level == 1
        assert len(sec2.paragraphs) == 1

    # 31. Legitimate nested custom headings supported
    def test_legitimate_nested_custom_heading_accepted(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Legitimate nested headings like 1.1 Dataset Preparation are accepted as subsections."""
        pages = [
            [
                "1. Methodology",
                "Overview of methodology.",
                "1.1 Dataset Preparation",
                "Dataset preparation details.",
                "1.2 Neural Radiance Fields Setup",
                "NeRF configuration details.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 3
        parent = result.sections[0]
        child1 = result.sections[1]
        child2 = result.sections[2]

        assert parent.title == "1. Methodology"
        assert parent.level == 1

        assert child1.title == "1.1 Dataset Preparation"
        assert child1.level == 2
        assert child1.parent_section_id == parent.section_id
        assert child1.section_type == SectionType.CUSTOM

        assert child2.title == "1.2 Neural Radiance Fields Setup"
        assert child2.level == 2
        assert child2.parent_section_id == parent.section_id
        assert child2.section_type == SectionType.CUSTOM

    # 32. Preserve heading paragraph positions
    def test_heading_paragraph_position_preservation(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Heading paragraph positions are zero-indexed, refer to global position, and are distinct from section_index."""
        pages = [
            [
                "Paper Title: Neural Reasoning",  # global 0 (preamble)
                "Author One, Author Two",         # global 1 (preamble)
                "1. Introduction",                 # global 2 (heading, section_index 0)
                "Introductory paragraph 1.",      # global 3 (body)
                "Introductory paragraph 2.",      # global 4 (body)
            ],
            [
                "2. Methodology",                  # global 5 (heading, section_index 1)
                "Methodology overview.",          # global 6 (body)
                "2.1 Analysis",                    # global 7 (heading, section_index 2)
                "Analysis part 1.",               # global 8 (body)
            ],
            [
                "3. Experiments",                  # global 9 (heading, section_index 3)
                "Experiment details.",            # global 10 (body)
                "3.1 Analysis",                    # global 11 (heading, section_index 4, repeated title)
                "Analysis part 2.",               # global 12 (body)
            ],
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.preamble) == 2
        assert result.preamble[0].paragraph_index == 0
        assert result.preamble[1].paragraph_index == 1

        assert len(result.sections) == 5

        sec0 = result.sections[0]
        assert sec0.section_index == 0
        assert sec0.heading_paragraph_index == 2
        assert sec0.title == "1. Introduction"
        assert sec0.heading_page == 1

        sec1 = result.sections[1]
        assert sec1.section_index == 1
        assert sec1.heading_paragraph_index == 5
        assert sec1.title == "2. Methodology"
        assert sec1.heading_page == 2

        sec2 = result.sections[2]
        assert sec2.section_index == 2
        assert sec2.heading_paragraph_index == 7
        assert sec2.title == "2.1 Analysis"
        assert sec2.heading_page == 2

        sec3 = result.sections[3]
        assert sec3.section_index == 3
        assert sec3.heading_paragraph_index == 9
        assert sec3.title == "3. Experiments"
        assert sec3.heading_page == 3

        sec4 = result.sections[4]
        assert sec4.section_index == 4
        assert sec4.heading_paragraph_index == 11
        assert sec4.title == "3.1 Analysis"
        assert sec4.heading_page == 3

    # 33. Preserve original paragraph and heading text verbatim
    def test_original_paragraph_and_heading_text_preserved(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Original paragraph and heading text with leading/trailing whitespace is preserved verbatim."""
        pages = [
            [
                "   Paper Title with leading spaces   ",
                "   1. Introduction   \n",
                "   This is an indented body paragraph with trailing whitespace.   ",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.preamble) == 1
        assert result.preamble[0].text == "   Paper Title with leading spaces   "

        assert len(result.sections) == 1
        sec = result.sections[0]
        # Title preserves verbatim string
        assert sec.title == "   1. Introduction   \n"
        # Normalized title is cleaned
        assert sec.normalized_title == "introduction"

        assert len(sec.paragraphs) == 1
        # Body paragraph preserves verbatim string
        assert sec.paragraphs[0].text == "   This is an indented body paragraph with trailing whitespace.   "

    # 34. Defensive handling of empty and whitespace-only paragraphs
    def test_whitespace_only_paragraphs_ignored_defensively(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Empty and whitespace-only paragraphs are excluded defensively, preserving the non-empty accounting invariant."""
        pages = [
            [
                "",
                "   ",
                "\t\n",
                "1. Introduction",
                "First body paragraph.",
                "   \n\t   ",
                "Second body paragraph.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        sec = result.sections[0]
        assert sec.title == "1. Introduction"
        assert sec.heading_paragraph_index == 0
        assert len(sec.paragraphs) == 2
        assert sec.paragraphs[0].text == "First body paragraph."
        assert sec.paragraphs[0].paragraph_index == 1
        assert sec.paragraphs[1].text == "Second body paragraph."
        assert sec.paragraphs[1].paragraph_index == 2

        # Diagnostic total_paragraphs counts only the non-empty paragraphs
        assert result.diagnostics.total_paragraphs == 3
        # Invariant holds for all non-empty paragraphs
        assert len(result.sections) + len(result.preamble) + sum(len(s.paragraphs) for s in result.sections) == 3

    # 35. Confidence scoring calibration
    def test_confidence_scoring_calibration(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Confidence scores are well-calibrated, bounded in [0.0, 1.0], and do not unnecessarily saturate at 1.0."""
        pages = [
            [
                "Introduction",              # canonical unnumbered (~0.70)
                "Introductory content.",
                "1. Introduction",           # canonical numbered (~0.95, not saturated at 1.0)
                "More intro content.",
                "1. Dataset Preparation",    # legitimate custom numbered title-case (~0.55)
                "Dataset content.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 3

        sec_unnum = result.sections[0]
        sec_num_canon = result.sections[1]
        sec_custom = result.sections[2]

        # Unnumbered canonical is high confidence but not saturated
        assert 0.60 <= sec_unnum.confidence <= 0.85
        # Numbered canonical is very high confidence but <= 1.0
        assert 0.85 <= sec_num_canon.confidence <= 1.0
        # Legitimate custom numbered is above MIN_HEADING_CONFIDENCE
        assert 0.50 <= sec_custom.confidence <= 0.70

        for s in result.sections:
            assert 0.0 <= s.confidence <= 1.0

    # 36. Sentence-case custom headings
    def test_sentence_case_custom_headings(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Legitimate sentence-case numbered headings are detected with correct hierarchy."""
        pages = [
            [
                "1. Dataset preparation",
                "Body paragraph describing dataset preparation.",
                "1.1 Data collection",
                "Body paragraph describing data collection.",
                "1.2 Feature extraction",
                "Body paragraph describing feature extraction.",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 3

        sec1 = result.sections[0]
        assert sec1.title == "1. Dataset preparation"
        assert sec1.normalized_title == "dataset preparation"
        assert sec1.section_type == SectionType.CUSTOM
        assert sec1.level == 1
        assert sec1.parent_section_id is None
        assert len(sec1.paragraphs) == 1

        sec1_1 = result.sections[1]
        assert sec1_1.title == "1.1 Data collection"
        assert sec1_1.normalized_title == "data collection"
        assert sec1_1.section_type == SectionType.CUSTOM
        assert sec1_1.level == 2
        assert sec1_1.parent_section_id == sec1.section_id
        assert len(sec1_1.paragraphs) == 1

        sec1_2 = result.sections[2]
        assert sec1_2.title == "1.2 Feature extraction"
        assert sec1_2.normalized_title == "feature extraction"
        assert sec1_2.section_type == SectionType.CUSTOM
        assert sec1_2.level == 2
        assert sec1_2.parent_section_id == sec1.section_id
        assert len(sec1_2.paragraphs) == 1

    # 37. Sentence-case numbered list rejected as headings
    def test_sentence_case_numbered_list_rejected(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Sentence-case numbered list items containing finite verbs are rejected and kept as body paragraphs."""
        pages = [
            [
                "1. Introduction",
                "1. Dataset preparation requires several preprocessing steps",
                "2. Data collection involves multiple public sources",
                "3. Feature extraction removes irrelevant information",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        # Only "1. Introduction" is detected
        assert len(result.sections) == 1
        intro = result.sections[0]
        assert intro.title == "1. Introduction"
        assert intro.section_type == SectionType.INTRODUCTION
        assert intro.level == 1
        assert len(intro.paragraphs) == 3

        # All 3 list items remain body paragraphs
        assert intro.paragraphs[0].text == "1. Dataset preparation requires several preprocessing steps"
        assert intro.paragraphs[1].text == "2. Data collection involves multiple public sources"
        assert intro.paragraphs[2].text == "3. Feature extraction removes irrelevant information"

        # None became custom sections
        assert len([s for s in result.sections if s.section_type == SectionType.CUSTOM]) == 0

    # 38. Adjacent custom headings preserved
    def test_adjacent_custom_headings_preserved(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Adjacent custom headings are not automatically classified as list items; plausible heading-only sections are preserved."""
        pages = [
            [
                "1. Dataset preparation",
                "2. Neural architecture",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 2

        sec1 = result.sections[0]
        assert sec1.title == "1. Dataset preparation"
        assert sec1.normalized_title == "dataset preparation"
        assert sec1.section_type == SectionType.CUSTOM
        assert sec1.level == 1
        assert len(sec1.paragraphs) == 0

        sec2 = result.sections[1]
        assert sec2.title == "2. Neural architecture"
        assert sec2.normalized_title == "neural architecture"
        assert sec2.section_type == SectionType.CUSTOM
        assert sec2.level == 1
        assert len(sec2.paragraphs) == 0

        # Diagnostics warn that heading-only sections have no body paragraphs
        assert any("has no body paragraphs" in w for w in result.diagnostics.warnings)

    # 39. Three adjacent custom headings detected
    def test_three_adjacent_custom_headings_detected(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Three adjacent plausible custom headings are not classified as a list solely due to sequence length."""
        pages = [
            [
                "1. Dataset preparation",
                "2. Neural architecture",
                "3. Training procedure",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 3

        sec1 = result.sections[0]
        assert sec1.title == "1. Dataset preparation"
        assert sec1.normalized_title == "dataset preparation"
        assert sec1.section_type == SectionType.CUSTOM
        assert sec1.level == 1
        assert len(sec1.paragraphs) == 0

        sec2 = result.sections[1]
        assert sec2.title == "2. Neural architecture"
        assert sec2.normalized_title == "neural architecture"
        assert sec2.section_type == SectionType.CUSTOM
        assert sec2.level == 1
        assert len(sec2.paragraphs) == 0

        sec3 = result.sections[2]
        assert sec3.title == "3. Training procedure"
        assert sec3.normalized_title == "training procedure"
        assert sec3.section_type == SectionType.CUSTOM
        assert sec3.level == 1
        assert len(sec3.paragraphs) == 0

    # 40. Three adjacent sentence-case numbered list items introduced by colon rejected
    def test_three_adjacent_sentence_case_numbered_list_items_rejected(
        self, detector: ResearchSectionDetector
    ) -> None:
        """Numbered list items introduced by a colon are rejected and kept as body paragraphs."""
        pages = [
            [
                "1. Introduction",
                "Our system has three major components:",
                "1. Data ingestion",
                "2. Neural processing",
                "3. Result generation",
            ]
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 1
        intro = result.sections[0]
        assert intro.title == "1. Introduction"
        assert intro.section_type == SectionType.INTRODUCTION
        assert intro.level == 1
        assert len(intro.paragraphs) == 4

        assert intro.paragraphs[0].text == "Our system has three major components:"
        assert intro.paragraphs[1].text == "1. Data ingestion"
        assert intro.paragraphs[2].text == "2. Neural processing"
        assert intro.paragraphs[3].text == "3. Result generation"

        # None became custom sections
        assert len([s for s in result.sections if s.section_type == SectionType.CUSTOM]) == 0

    # 41. Numbered custom headings after an earlier numbered section detected
    def test_numbered_custom_headings_after_earlier_numbered_section_detected(
        self, detector: ResearchSectionDetector
    ) -> None:
        """An earlier section beginning with 1. does not force a later numbered heading sequence to be classified as a list."""
        pages = [
            [
                "1. Introduction",
                "Introduction body text describing the overall project goals.",
                "2. Related Work",
                "Overview of prior approaches in the literature.",
            ],
            [
                "1. Dataset preparation",
                "Details of data preprocessing and corpus filtering.",
                "2. Neural architecture",
                "Model transformer layers and attention mechanism.",
                "3. Training procedure",
                "Optimization hyperparameters and hardware setup.",
            ],
        ]
        paper = make_preprocessed_paper(pages)
        result = detector.detect(paper)

        assert len(result.sections) == 5

        sec1 = result.sections[0]
        assert sec1.title == "1. Introduction"
        assert sec1.section_type == SectionType.INTRODUCTION

        sec2 = result.sections[1]
        assert sec2.title == "2. Related Work"
        assert sec2.section_type == SectionType.RELATED_WORK

        sec3 = result.sections[2]
        assert sec3.title == "1. Dataset preparation"
        assert sec3.normalized_title == "dataset preparation"
        assert sec3.section_type == SectionType.CUSTOM
        assert sec3.level == 1
        assert len(sec3.paragraphs) == 1

        sec4 = result.sections[3]
        assert sec4.title == "2. Neural architecture"
        assert sec4.normalized_title == "neural architecture"
        assert sec4.section_type == SectionType.CUSTOM
        assert sec4.level == 1
        assert len(sec4.paragraphs) == 1

        sec5 = result.sections[4]
        assert sec5.title == "3. Training procedure"
        assert sec5.normalized_title == "training procedure"
        assert sec5.section_type == SectionType.CUSTOM
        assert sec5.level == 1
        assert len(sec5.paragraphs) == 1
