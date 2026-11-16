"""Unit tests for the Research Paper Text Preprocessor.

Tests cover:
- Multi-page processing, empty page collection, empty pages, mixed pages.
- Line ending normalization (CRLF, CR -> LF).
- Whitespace normalization and collapsing excessive blank lines.
- Line-break hyphenation joining and preservation of legitimate compounds.
- Rejection of unsafe hyphenation joins (numbers, uppercase, headings, citations).
- Repeated header and footer detection and removal.
- Section heading, citation, DOI, URL, math, table, and list preservation.
- Single-line page guard preventing pages from becoming empty.
- Page number preservation, original text immutability, input object immutability.
- Preprocessing diagnostics and idempotency.
"""

from __future__ import annotations

import copy
import pytest

from app.modules.ingestion.parser import (
    AcademicMetadata,
    ParsedPage,
    ParsedResearchPaper,
    ParsingDiagnostics,
    PDFMetadata,
)
from app.modules.ingestion.preprocessor import (
    PreprocessedPage,
    PreprocessedResearchPaper,
    PreprocessingDiagnostics,
    ResearchTextPreprocessor,
)


def make_parsed_paper(
    pages_text: list[str],
    title: str = "Test Research Paper",
) -> ParsedResearchPaper:
    """Helper to construct an in-memory ParsedResearchPaper with specified page texts."""
    pages = [
        ParsedPage(
            page_number=i + 1,
            text=t,
            extraction_status="SUCCESS" if t.strip() else "NO_TEXT",
        )
        for i, t in enumerate(pages_text)
    ]
    return ParsedResearchPaper(
        pdf_metadata=PDFMetadata(title=title, page_count=len(pages)),
        academic_metadata=AcademicMetadata(title=title),
        pages=pages,
        diagnostics=ParsingDiagnostics(
            total_pages=len(pages),
            pages_with_text=sum(1 for p in pages if p.text.strip()),
            pages_without_text=sum(1 for p in pages if not p.text.strip()),
        ),
    )


class TestResearchTextPreprocessor:
    """Comprehensive test suite for ResearchTextPreprocessor."""

    @pytest.fixture
    def preprocessor(self) -> ResearchTextPreprocessor:
        """Fixture returning standard ResearchTextPreprocessor."""
        return ResearchTextPreprocessor()

    # 1. Valid multi-page input
    def test_valid_multi_page_input(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Processes valid multi-page document, populating pages, full_text, and diagnostics."""
        paper = make_parsed_paper([
            "1. Introduction\nThis is the first page introduction text.",
            "2. Methodology\nThis is the second page methodology text.",
        ])

        result = preprocessor.preprocess(paper)

        assert isinstance(result, PreprocessedResearchPaper)
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1
        assert result.pages[1].page_number == 2
        assert "first page introduction" in result.pages[0].cleaned_text
        assert "second page methodology" in result.pages[1].cleaned_text
        assert result.diagnostics.total_pages == 2
        assert result.diagnostics.modified_pages == 2
        assert preprocessor.page_separator in result.full_text

    # 2. Empty page collection
    def test_empty_page_collection(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Empty page collection returns empty result with appropriate warning and zero totals."""
        paper = make_parsed_paper([])

        result = preprocessor.preprocess(paper)

        assert len(result.pages) == 0
        assert result.full_text == ""
        assert result.diagnostics.total_pages == 0
        assert result.diagnostics.modified_pages == 0
        assert any("zero pages" in w.lower() for w in result.diagnostics.warnings)

    # 3. Empty pages
    def test_empty_pages(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Pages with zero extractable text remain empty output pages with warnings."""
        paper = make_parsed_paper(["", "   \n\t  "])

        result = preprocessor.preprocess(paper)

        assert len(result.pages) == 2
        assert result.pages[0].cleaned_text == ""
        assert result.pages[0].paragraphs == []
        assert result.pages[1].cleaned_text == ""
        assert result.pages[1].paragraphs == []
        assert result.full_text == ""
        assert len(result.diagnostics.warnings) == 2
        assert "Page 1 contains no text." in result.diagnostics.warnings
        assert "Page 2 contains no text." in result.diagnostics.warnings

    # 4. Mixed populated and empty pages
    def test_mixed_populated_and_empty_pages(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Mixed populated and blank pages retain original page numbers and join only non-empty pages."""
        paper = make_parsed_paper([
            "Page one content with introduction.",
            "",
            "Page three content with conclusion.",
        ])

        result = preprocessor.preprocess(paper)

        assert len(result.pages) == 3
        assert result.pages[0].page_number == 1
        assert result.pages[0].cleaned_text != ""
        assert result.pages[1].page_number == 2
        assert result.pages[1].cleaned_text == ""
        assert result.pages[2].page_number == 3
        assert result.pages[2].cleaned_text != ""

        # full_text only contains non-empty pages
        assert "Page one content" in result.full_text
        assert "Page three content" in result.full_text
        assert result.full_text == f"{result.pages[0].cleaned_text}\n\n\n{result.pages[2].cleaned_text}"

    # 5. CRLF and CR line-ending normalization
    def test_crlf_and_cr_line_ending_normalization(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Normalizes Windows CRLF and legacy Mac CR line endings to LF."""
        paper = make_parsed_paper([
            "Line one.\r\nLine two.\rLine three.\nLine four.",
        ])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "\r\n" not in cleaned
        assert "\r" not in cleaned
        assert "Line one." in cleaned
        assert "Line four." in cleaned

    # 6. Multiple spaces and tabs
    def test_multiple_spaces_and_tabs(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Collapses consecutive horizontal spaces and tabs into a single space."""
        paper = make_parsed_paper([
            "This    sentence  has \t\t unnecessary \t   spaces.",
        ])

        result = preprocessor.preprocess(paper)

        assert result.pages[0].cleaned_text == "This sentence has unnecessary spaces."

    # 7. Excessive blank lines
    def test_excessive_blank_lines(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Reduces 3 or more consecutive newlines to a standard 2-newline paragraph break."""
        paper = make_parsed_paper([
            "Paragraph one.\n\n\n\n\nParagraph two.\n\n\n\nParagraph three.",
        ])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "\n\n\n" not in cleaned
        assert "Paragraph one.\n\nParagraph two.\n\nParagraph three." == cleaned
        assert len(result.pages[0].paragraphs) == 3

    # 8. Paragraph boundary preservation
    def test_paragraph_boundary_preservation(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Preserves intentional blank-line paragraph breaks across the page."""
        paper = make_parsed_paper([
            "First distinct paragraph detailing theory.\n\n"
            "Second distinct paragraph presenting data.\n\n"
            "Third distinct paragraph summarizing findings.",
        ])

        result = preprocessor.preprocess(paper)

        assert len(result.pages[0].paragraphs) == 3
        assert result.pages[0].paragraphs[0] == "First distinct paragraph detailing theory."
        assert result.pages[0].paragraphs[1] == "Second distinct paragraph presenting data."
        assert result.pages[0].paragraphs[2] == "Third distinct paragraph summarizing findings."

    # 9. Line-wrapped paragraph reconstruction
    def test_line_wrapped_paragraph_reconstruction(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Converts soft line wraps within a paragraph into continuous flowing text."""
        paper = make_parsed_paper([
            "Deep neural networks have demonstrated remarkable success\n"
            "across diverse domains of natural language processing.\n"
            "Their scalability enables significant performance improvements.",
        ])

        result = preprocessor.preprocess(paper)

        expected = (
            "Deep neural networks have demonstrated remarkable success "
            "across diverse domains of natural language processing. "
            "Their scalability enables significant performance improvements."
        )
        assert result.pages[0].cleaned_text == expected
        assert len(result.pages[0].paragraphs) == 1
        assert result.pages[0].paragraphs[0] == expected

    # 10. Correct hyphenation joining
    def test_correct_hyphenation_joining(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Correctly rejoins words broken across line breaks with hyphens."""
        paper = make_parsed_paper([
            "We explore natural language process-\n"
            "ing and representations.\n\n"
            "The infor-\n"
            "mation retrieval engine achieves high accuracy.",
        ])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "processing" in cleaned
        assert "process-" not in cleaned
        assert "information" in cleaned
        assert "infor-" not in cleaned

    # 11. Preservation of legitimate hyphenated compounds
    def test_preservation_of_legitimate_hyphenated_compounds(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Preserves legitimate hyphenated compounds on the same line and when line-broken."""
        paper = make_parsed_paper([
            "We propose state-of-the-art models for self-\n"
            "attention networks and cross-\n"
            "lingual transfer learning.",
        ])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        # Inline compound remains intact
        assert "state-of-the-art models" in cleaned
        # Broken prefix compounds retain hyphen
        assert "self-attention" in cleaned
        assert "cross-lingual" in cleaned

    # 12. Rejection of unsafe hyphenation joins
    def test_rejection_of_unsafe_hyphenation_joins(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Rejects hyphen joins involving numbers, uppercase tokens, citations, headings, or URLs."""
        paper = make_parsed_paper([
            "The historical period was 1990-\n"
            "2020 in European politics.\n\n"
            "The agency was NASA-\n"
            "JPL joint initiative.\n\n"
            "Refer to equation-\n"
            "[1] for details.\n\n"
            "Methodology overview-\n"
            "1. Introduction\n\n"
            "Visit https://example.com/api-\n"
            "v1 for documentation.",
        ])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        # Numbers not joined
        assert "1990-" in cleaned
        assert "2020" in cleaned
        # Uppercase not joined
        assert "NASA-" in cleaned
        assert "JPL" in cleaned
        # Bracketed citation not joined
        assert "[1]" in cleaned
        # Heading not joined
        assert "1. Introduction" in cleaned
        # URL not corrupted
        assert "https://example.com/api-" in cleaned

    # 13. Repeated header detection
    def test_repeated_header_detection(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Detects and removes running headers repeated across multiple pages."""
        header_text = "Proceedings of the 61st Annual Meeting of the ACL"
        paper = make_parsed_paper([
            f"{header_text}\n\n1. Introduction\nFirst page body text.",
            f"{header_text}\n\n2. Methods\nSecond page body text.",
            f"{header_text}\n\n3. Results\nThird page body text.",
        ])

        result = preprocessor.preprocess(paper)

        # Removed from all pages
        for p in result.pages:
            assert header_text not in p.cleaned_text
            assert "body text" in p.cleaned_text

        # Recorded in diagnostics
        assert header_text in result.diagnostics.removed_repeated_lines
        assert len(result.diagnostics.removed_repeated_lines) == 1

    # 14. Repeated footer detection
    def test_repeated_footer_detection(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Detects and removes running footers repeated across multiple pages."""
        footer_text = "Confidential - Submitted to NeurIPS 2023"
        paper = make_parsed_paper([
            f"1. Introduction\nPage 1 body.\n\n{footer_text}",
            f"2. Methods\nPage 2 body.\n\n{footer_text}",
            f"3. Results\nPage 3 body.\n\n{footer_text}",
        ])

        result = preprocessor.preprocess(paper)

        for p in result.pages:
            assert footer_text not in p.cleaned_text
            assert "body." in p.cleaned_text

        assert footer_text in result.diagnostics.removed_repeated_lines

    # 15. Unique lines not being removed
    def test_unique_lines_not_being_removed(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Lines that appear only once at the top or bottom of a page are never removed."""
        paper = make_parsed_paper([
            "Unique Top Banner Page 1\nBody text page 1.",
            "Different Top Banner Page 2\nBody text page 2.",
        ])

        result = preprocessor.preprocess(paper)

        assert "Unique Top Banner Page 1" in result.pages[0].cleaned_text
        assert "Different Top Banner Page 2" in result.pages[1].cleaned_text
        assert len(result.diagnostics.removed_repeated_lines) == 0

    # 16. Section headings not being removed
    def test_section_headings_not_being_removed(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Section headings are protected from repeated header/footer removal even if repeated."""
        paper = make_parsed_paper([
            "References\n[1] Smith et al., 2020.",
            "References\n[2] Doe et al., 2021.",
        ])

        result = preprocessor.preprocess(paper)

        assert "References" in result.pages[0].cleaned_text
        assert "References" in result.pages[1].cleaned_text
        assert "References" not in result.diagnostics.removed_repeated_lines

    # 17. Pages not becoming empty after header/footer removal
    def test_pages_not_becoming_empty_after_header_removal(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """A page with only one line matching a repeated header is preserved and not made empty."""
        header_text = "Journal of Machine Learning Research"
        paper = make_parsed_paper([
            f"{header_text}\nContent on page 1.",
            f"{header_text}\nContent on page 2.",
            f"{header_text}",  # Single line page
        ])

        result = preprocessor.preprocess(paper)

        # Pages 1 and 2 had content, so header was removed
        assert header_text not in result.pages[0].cleaned_text
        assert header_text not in result.pages[1].cleaned_text
        # Page 3 had ONLY the header line, so it must not be stripped to empty
        assert header_text in result.pages[2].cleaned_text
        assert result.pages[2].cleaned_text != ""

    # 18. Citation preservation
    def test_citation_preservation(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Preserves both inline author-year citations and numeric bracketed citations."""
        text = (
            "Prior work by (Brown et al., 2020) and (Vaswani et al., 2017) established\n"
            "foundational attention mechanisms [1], which were later expanded [2, 3]."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "(Brown et al., 2020)" in cleaned
        assert "(Vaswani et al., 2017)" in cleaned
        assert "[1]" in cleaned
        assert "[2, 3]" in cleaned

    # 19. DOI and URL preservation
    def test_doi_and_url_preservation(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Preserves DOI identifiers and URLs without stripping or breaking them."""
        text = (
            "The repository is available at https://github.com/organization/repo-name\n"
            "and the published article is indexed at https://doi.org/10.1145/3308558.3313528."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "https://github.com/organization/repo-name" in cleaned
        assert "https://doi.org/10.1145/3308558.3313528" in cleaned

    # 20. Mathematical equation and symbol preservation
    def test_mathematical_equation_and_symbol_preservation(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Preserves mathematical equations, Greek letters, and summation symbols."""
        text = (
            "The optimization objective is defined as:\n"
            "$$L(\\theta) = - \\sum_{i=1}^N \\log P(y_i | x_i; \\theta)$$\n"
            "where $\\theta$ denotes parameters and $\\alpha, \\beta > 0$ are weights."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "$$L(\\theta) = - \\sum_{i=1}^N \\log P(y_i | x_i; \\theta)$$" in cleaned
        assert "\\alpha, \\beta > 0" in cleaned

    # 21. Reference text preservation
    def test_reference_text_preservation(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Preserves bibliographic reference entries intact."""
        text = (
            "References\n\n"
            "[1] A. Vaswani, N. Shazeer, N. Parmar, et al. Attention is all you need. NeurIPS, 2017.\n\n"
            "[2] J. Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirectional transformers. NAACL, 2019."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "[1] A. Vaswani" in cleaned
        assert "[2] J. Devlin" in cleaned
        assert "NeurIPS, 2017." in cleaned

    # 22. Table text preservation
    def test_table_text_preservation(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Preserves markdown / ASCII table structure without merging rows into paragraphs."""
        text = (
            "Experimental results are summarized in Table 1:\n\n"
            "| Model | Accuracy | F1-Score |\n"
            "|---|---|---|\n"
            "| Baseline | 82.4 | 81.9 |\n"
            "| Proposed | 89.7 | 89.2 |\n\n"
            "As observed, the proposed architecture outperforms the baseline."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "| Model | Accuracy | F1-Score |" in cleaned
        assert "| Proposed | 89.7 | 89.2 |" in cleaned
        # Table lines are not flattened into running prose
        assert "\n| Model |" in cleaned or cleaned.startswith("| Model |")

    # 23. Numbered and bulleted list preservation
    def test_numbered_and_bulleted_list_preservation(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Preserves numbered and bulleted list items on separate lines."""
        text = (
            "We make three key contributions:\n"
            "* A self-supervised representation learning objective.\n"
            "* Empirical verification across eight benchmark datasets.\n"
            "* An extensive ablation study investigating hyperparameter sensitivity."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "* A self-supervised" in cleaned
        assert "* Empirical verification" in cleaned
        assert "* An extensive ablation" in cleaned
        # List items remain on separate lines
        assert "\n* " in cleaned

    # 24. Page-number preservation
    def test_page_number_preservation(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Output pages retain exact original 1-indexed page numbering."""
        paper = make_parsed_paper(["Page 1", "Page 2", "Page 3", "Page 4"])

        result = preprocessor.preprocess(paper)

        for i, page in enumerate(result.pages):
            assert page.page_number == i + 1

    # 25. Original text immutability
    def test_original_text_immutability(self, preprocessor: ResearchTextPreprocessor) -> None:
        """PreprocessedPage.original_text preserves exact input characters without modification."""
        raw_text = "Original raw   text \r\n with CRLF  and  spaces."
        paper = make_parsed_paper([raw_text])

        result = preprocessor.preprocess(paper)

        assert result.pages[0].original_text == raw_text
        assert result.pages[0].cleaned_text != raw_text

    # 26. Input object immutability
    def test_input_object_immutability(self, preprocessor: ResearchTextPreprocessor) -> None:
        """The input ParsedResearchPaper and its ParsedPage objects are not mutated."""
        raw_pages = [
            "Page 1 with hyphen-\nation and CRLF.\r\n",
            "Page 2 with trailing footer.\nFooter Line",
        ]
        paper = make_parsed_paper(raw_pages)
        deep_copied_paper = copy.deepcopy(paper)

        preprocessor.preprocess(paper)

        # Compare paper attributes against deep copy
        assert len(paper.pages) == len(deep_copied_paper.pages)
        for i in range(len(paper.pages)):
            assert paper.pages[i].page_number == deep_copied_paper.pages[i].page_number
            assert paper.pages[i].text == deep_copied_paper.pages[i].text
            assert paper.pages[i].extraction_status == deep_copied_paper.pages[i].extraction_status

    # 27. Diagnostics correctness
    def test_diagnostics_correctness(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Diagnostics accurately reflect total pages, modified count, removed lines, and warnings."""
        header = "Conference on Empirical Methods in Natural Language Processing"
        paper = make_parsed_paper([
            f"{header}\nFirst page with CRLF.\r\n",
            "",  # Empty page
            f"{header}\nThird page with regular clean text.",
        ])

        result = preprocessor.preprocess(paper)

        assert result.diagnostics.total_pages == 3
        # Page 1 modified (header + CRLF), Page 2 was empty/unmodified, Page 3 modified (header)
        assert result.diagnostics.modified_pages >= 2
        assert header in result.diagnostics.removed_repeated_lines
        assert any("Page 2 contains no text." in w for w in result.diagnostics.warnings)

    # 28. Idempotency
    def test_idempotency(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Running the preprocessor on already-cleaned pages produces identical output."""
        paper = make_parsed_paper([
            "1. Introduction\n"
            "This is a line of text that wraps\n"
            "onto the next line with process-\n"
            "ing.\n\n"
            "2. Methodology\n"
            "Another paragraph with multiple sentences. Here is the second sentence.",
        ])

        first_pass = preprocessor.preprocess(paper)

        # Construct a second ParsedResearchPaper using the cleaned text from the first pass
        second_paper = make_parsed_paper([p.cleaned_text for p in first_pass.pages])
        second_pass = preprocessor.preprocess(second_paper)

        assert len(first_pass.pages) == len(second_pass.pages)
        for i in range(len(first_pass.pages)):
            assert first_pass.pages[i].cleaned_text == second_pass.pages[i].cleaned_text
            assert first_pass.pages[i].paragraphs == second_pass.pages[i].paragraphs
        assert first_pass.full_text == second_pass.full_text

    # 29. TypeError on invalid input type
    def test_type_error_on_invalid_input(self, preprocessor: ResearchTextPreprocessor) -> None:
        """Raises TypeError when passed non-ParsedResearchPaper objects."""
        with pytest.raises(TypeError, match="Expected ParsedResearchPaper instance"):
            preprocessor.preprocess("invalid string input")  # type: ignore

        with pytest.raises(TypeError, match="Expected ParsedResearchPaper instance"):
            preprocessor.preprocess(None)  # type: ignore

        with pytest.raises(TypeError, match="Expected ParsedResearchPaper instance"):
            preprocessor.preprocess({"pages": []})  # type: ignore

    # -------------------------------------------------------------------------
    # Targeted Tests: Paragraph Boundary Preservation with Header/Footer Detection
    # -------------------------------------------------------------------------

    def test_multipage_multiple_paragraphs_no_repeated_headers(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Multi-page document with multiple paragraphs and no repeated headers preserves paragraph breaks."""
        page1 = "Page 1 Paragraph 1.\n\nPage 1 Paragraph 2."
        page2 = "Page 2 Paragraph 1.\n\nPage 2 Paragraph 2."
        paper = make_parsed_paper([page1, page2])

        result = preprocessor.preprocess(paper)

        assert len(result.pages[0].paragraphs) == 2
        assert result.pages[0].paragraphs[0] == "Page 1 Paragraph 1."
        assert result.pages[0].paragraphs[1] == "Page 1 Paragraph 2."
        assert len(result.pages[1].paragraphs) == 2
        assert result.pages[1].paragraphs[0] == "Page 2 Paragraph 1."
        assert result.pages[1].paragraphs[1] == "Page 2 Paragraph 2."

    def test_multipage_repeated_headers_with_multiple_paragraphs(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Repeated headers are removed while preserving blank-line paragraph separators on each page."""
        header = "Conference on Neural Information Processing Systems"
        page1 = f"{header}\n\nPage 1 Paragraph 1.\n\nPage 1 Paragraph 2."
        page2 = f"{header}\n\nPage 2 Paragraph 1.\n\nPage 2 Paragraph 2."
        paper = make_parsed_paper([page1, page2])

        result = preprocessor.preprocess(paper)

        assert header in result.diagnostics.removed_repeated_lines
        assert len(result.pages[0].paragraphs) == 2
        assert result.pages[0].paragraphs[0] == "Page 1 Paragraph 1."
        assert result.pages[0].paragraphs[1] == "Page 1 Paragraph 2."
        assert len(result.pages[1].paragraphs) == 2
        assert result.pages[1].paragraphs[0] == "Page 2 Paragraph 1."
        assert result.pages[1].paragraphs[1] == "Page 2 Paragraph 2."

    def test_multipage_repeated_footers_with_multiple_paragraphs(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Repeated footers are removed while preserving blank-line paragraph separators on each page."""
        footer = "Page Footer - Confidential Draft"
        page1 = f"Page 1 Paragraph 1.\n\nPage 1 Paragraph 2.\n\n{footer}"
        page2 = f"Page 2 Paragraph 1.\n\nPage 2 Paragraph 2.\n\n{footer}"
        paper = make_parsed_paper([page1, page2])

        result = preprocessor.preprocess(paper)

        assert footer in result.diagnostics.removed_repeated_lines
        assert len(result.pages[0].paragraphs) == 2
        assert result.pages[0].paragraphs[0] == "Page 1 Paragraph 1."
        assert result.pages[0].paragraphs[1] == "Page 1 Paragraph 2."
        assert len(result.pages[1].paragraphs) == 2
        assert result.pages[1].paragraphs[0] == "Page 2 Paragraph 1."
        assert result.pages[1].paragraphs[1] == "Page 2 Paragraph 2."

    def test_repeated_header_removal_preserves_paragraph_counts(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Verifies paragraph counts remain strictly correct after repeated header removal."""
        header = "International Conference on Learning Representations"
        page1 = f"{header}\n\nParagraph 1.\n\nParagraph 2.\n\nParagraph 3."
        page2 = f"{header}\n\nParagraph 4.\n\nParagraph 5."
        paper = make_parsed_paper([page1, page2])

        result = preprocessor.preprocess(paper)

        assert len(result.pages[0].paragraphs) == 3
        assert len(result.pages[1].paragraphs) == 2
        assert "\n\n" in result.pages[0].cleaned_text
        assert "\n\n" in result.pages[1].cleaned_text

    # -------------------------------------------------------------------------
    # Targeted Tests: List-Item Structure Preservation
    # -------------------------------------------------------------------------

    def test_multiple_independent_list_items(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Preserves multiple independent list items as separate structural blocks."""
        text = (
            "* First independent contribution item.\n"
            "* Second independent contribution item.\n"
            "* Third independent contribution item."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        paragraphs = result.pages[0].paragraphs
        assert len(paragraphs) == 3
        assert paragraphs[0] == "* First independent contribution item."
        assert paragraphs[1] == "* Second independent contribution item."
        assert paragraphs[2] == "* Third independent contribution item."

    def test_list_item_followed_by_explanatory_paragraph(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Ensures a list item is not merged with a following explanatory prose paragraph."""
        text = (
            "* Primary finding demonstrating convergence on benchmark tasks.\n\n"
            "This empirical result validates the underlying theoretical foundation "
            "discussed in Section 3."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        paragraphs = result.pages[0].paragraphs
        assert len(paragraphs) == 2
        assert paragraphs[0] == "* Primary finding demonstrating convergence on benchmark tasks."
        assert paragraphs[1].startswith("This empirical result validates")

    def test_wrapped_list_item_continuation(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Joins line-wrapped continuation of a list item when reliable lowercase continuation exists."""
        text = (
            "* A self-supervised representation learning objective that optimizes\n"
            "mutual information across multimodal representations."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        paragraphs = result.pages[0].paragraphs
        assert len(paragraphs) == 1
        expected = (
            "* A self-supervised representation learning objective that optimizes "
            "mutual information across multimodal representations."
        )
        assert paragraphs[0] == expected

    def test_numbered_list_items_preserved(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Preserves numbered list items as distinct structural elements."""
        text = (
            "1. First data preprocessing step.\n"
            "2. Second feature extraction step.\n"
            "3. Third classifier training step."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        paragraphs = result.pages[0].paragraphs
        assert len(paragraphs) == 3
        assert paragraphs[0] == "1. First data preprocessing step."
        assert paragraphs[1] == "2. Second feature extraction step."
        assert paragraphs[2] == "3. Third classifier training step."

    def test_list_items_followed_by_section_heading(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """List items preceding section headings remain distinct from the heading."""
        text = (
            "* Final observation regarding training stability.\n\n"
            "2. Methodology\n\n"
            "Our model architecture consists of three core components."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        paragraphs = result.pages[0].paragraphs
        assert len(paragraphs) == 3
        assert paragraphs[0] == "* Final observation regarding training stability."
        assert paragraphs[1] == "2. Methodology"
        assert paragraphs[2].startswith("Our model architecture")

    # -------------------------------------------------------------------------
    # Targeted Tests: URL and Email Address Dehyphenation Protection
    # -------------------------------------------------------------------------

    def test_url_ending_in_hyphen_with_lowercase_continuation(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """URL ending in a hyphen followed by lowercase continuation is not corrupted."""
        text = (
            "Documentation available at https://example.com/api-\n"
            "version for details on endpoint configurations."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "https://example.com/api-" in cleaned
        assert "version" in cleaned
        assert "https://example.com/apiversion" not in cleaned

    def test_url_with_line_break_in_path(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """URL with a hyphenated directory path across line break is preserved."""
        text = (
            "Code repository: https://github.com/organization/nlp-research-\n"
            "agent for reproducibility."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "https://github.com/organization/nlp-research-" in cleaned
        assert "agent" in cleaned

    def test_email_address_dehyphenation_protection(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Email address with hyphenated domain or local part is not corrupted."""
        text = (
            "Contact author at support@domain-services-\n"
            "cloud.com for correspondence."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "support@domain-services-" in cleaned
        assert "cloud.com" in cleaned

    def test_normal_word_dehyphenation_still_works(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Standard word dehyphenation continues to operate accurately alongside URL protection."""
        text = (
            "The neural network performs text process-\n"
            "ing and feature representa-\n"
            "tion learning."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "processing" in cleaned
        assert "representation" in cleaned
        assert "process-" not in cleaned
        assert "representa-" not in cleaned

    def test_legitimate_compounds_remain_correct(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Legitimate hyphenated compounds like self-attention remain correct."""
        text = (
            "We employ multi-head self-\n"
            "attention and cross-\n"
            "layer connections in our state-of-the-art model."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "self-attention" in cleaned
        assert "cross-layer" in cleaned
        assert "state-of-the-art" in cleaned

    def test_bare_domain_url_dehyphenation_protection(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Bare domain-based URL ending with hyphen is protected from incorrect dehyphenation."""
        text = (
            "Refer to the documentation on example.com/api-\n"
            "version for parameter details."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "example.com/api-" in cleaned
        assert "version" in cleaned
        assert "example.com/apiversion" not in cleaned

    def test_bare_doi_identifier_dehyphenation_protection(
        self, preprocessor: ResearchTextPreprocessor
    ) -> None:
        """Bare DOI identifier ending with hyphen is protected from incorrect dehyphenation."""
        text = (
            "The publication identifier is 10.1234/research-paper-\n"
            "identifier recorded in the metadata repository."
        )
        paper = make_parsed_paper([text])

        result = preprocessor.preprocess(paper)

        cleaned = result.pages[0].cleaned_text
        assert "10.1234/research-paper-" in cleaned
        assert "identifier" in cleaned
        assert "10.1234/research-paperidentifier" not in cleaned
