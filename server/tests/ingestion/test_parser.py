"""Unit tests for Research Paper PDF parsing and metadata extraction layer.

Tests:
- Input validation (bytes type, empty bytes, %PDF- signature).
- Document handling (multi-page, page numbering, embedded metadata, empty pages, mixed pages).
- Error conditions (corrupted PDF, zero pages, encrypted/password-protected, no extractable text).
- Page extraction failures (ERROR status) vs genuinely empty pages (NO_TEXT status).
- Academic metadata extraction (title, authors, affiliations, abstract, DOI, keywords, year, venue, publisher).
- Single-author, multiline title, and titles resembling person names.
- Affiliation association (shared, numbered mapping, ambiguous, absent).
- Publication year vs received/accepted dates.
- Parsing diagnostics (page counts, error counts, warnings).
All PDFs are generated in-memory via PyMuPDF without external network calls or temporary files.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from app.core.exceptions import (
    PDFEncryptedError,
    PDFNoExtractableTextError,
    PDFParsingError,
)
from app.modules.ingestion.parser import (
    AcademicMetadata,
    AuthorCandidate,
    ParsedPage,
    ParsedResearchPaper,
    ParsingDiagnostics,
    PDFMetadata,
    PDFParser,
    ResearchMetadataExtractor,
)


def create_synthetic_pdf(
    pages_text: list[str],
    metadata: dict[str, str] | None = None,
) -> bytes:
    """Helper to generate an in-memory PDF with specified page text and metadata."""
    doc = pymupdf.open()
    if metadata:
        doc.set_metadata(metadata)
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_encrypted_pdf(text: str = "Secret Content") -> bytes:
    """Helper to generate an in-memory encrypted PDF."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner_password",
        user_pw="user_password",
    )
    doc.close()
    return pdf_bytes


@pytest.fixture
def parser() -> PDFParser:
    """Fixture providing a fresh PDFParser instance."""
    return PDFParser()


@pytest.fixture
def extractor() -> ResearchMetadataExtractor:
    """Fixture providing a fresh ResearchMetadataExtractor instance."""
    return ResearchMetadataExtractor()


# =========================================================================
# 1. PDF Parsing & Extraction Tests
# =========================================================================


class TestPDFParserExtraction:
    """Tests for PDF loading, text extraction, page numbering, and error handling."""

    def test_valid_pdf_multi_page_extraction(self, parser: PDFParser) -> None:
        """Valid multi-page PDF extracts correct pages, order, and 1-indexed numbering."""
        pdf_bytes = create_synthetic_pdf([
            "Page 1 Content\nIntroduction to Machine Learning",
            "Page 2 Content\nMethodology and Architecture",
            "Page 3 Content\nResults and Evaluation",
        ])

        result = parser.parse(pdf_bytes)

        assert isinstance(result, ParsedResearchPaper)
        assert len(result.pages) == 3

        assert result.pages[0].page_number == 1
        assert "Page 1 Content" in result.pages[0].text
        assert result.pages[0].extraction_status == "SUCCESS"

        assert result.pages[1].page_number == 2
        assert "Page 2 Content" in result.pages[1].text
        assert result.pages[1].extraction_status == "SUCCESS"

        assert result.pages[2].page_number == 3
        assert "Page 3 Content" in result.pages[2].text
        assert result.pages[2].extraction_status == "SUCCESS"

        assert result.diagnostics.total_pages == 3
        assert result.diagnostics.pages_with_text == 3
        assert result.diagnostics.pages_without_text == 0
        assert result.diagnostics.pages_with_errors == 0
        assert len(result.diagnostics.warnings) == 0

    def test_embedded_metadata_extraction(self, parser: PDFParser) -> None:
        """Embedded document catalog metadata is correctly extracted into PDFMetadata."""
        meta_dict = {
            "title": "Attention Mechanisms in Deep Learning",
            "author": "Jane Doe; John Smith",
            "subject": "Natural Language Processing",
            "keywords": "attention, transformer, nlp",
            "creator": "LaTeX with hyperref",
            "producer": "pdfTeX-1.40.24",
        }
        pdf_bytes = create_synthetic_pdf(["Sample page text with content."], metadata=meta_dict)

        result = parser.parse(pdf_bytes)

        assert isinstance(result.pdf_metadata, PDFMetadata)
        assert result.pdf_metadata.title == "Attention Mechanisms in Deep Learning"
        assert result.pdf_metadata.author == "Jane Doe; John Smith"
        assert result.pdf_metadata.subject == "Natural Language Processing"
        assert result.pdf_metadata.keywords == "attention, transformer, nlp"
        assert result.pdf_metadata.creator == "LaTeX with hyperref"
        assert result.pdf_metadata.producer == "pdfTeX-1.40.24"
        assert result.pdf_metadata.page_count == 1

    def test_mixed_text_and_empty_scanned_pages(self, parser: PDFParser) -> None:
        """PDF with text on some pages and empty text on others records diagnostics and warnings."""
        pdf_bytes = create_synthetic_pdf([
            "First page with valid text.",
            "",  # Scanned or image-only page without text
            "Third page with further text.",
        ])

        result = parser.parse(pdf_bytes)

        assert len(result.pages) == 3
        assert result.pages[0].extraction_status == "SUCCESS"
        assert result.pages[1].extraction_status == "NO_TEXT"
        assert result.pages[1].text == ""
        assert result.pages[2].extraction_status == "SUCCESS"

        assert result.diagnostics.total_pages == 3
        assert result.diagnostics.pages_with_text == 2
        assert result.diagnostics.pages_without_text == 1
        assert result.diagnostics.pages_with_errors == 0
        assert any("Page 2 contains no extractable text" in w for w in result.diagnostics.warnings)

    def test_mixed_success_no_text_and_extraction_error(self, parser: PDFParser) -> None:
        """Page extraction failure, genuinely empty page, and successful page are all distinguishable."""
        page1_mock = MagicMock()
        page1_mock.get_text.return_value = "Page 1 successfully extracted text."

        page2_mock = MagicMock()
        page2_mock.get_text.return_value = ""  # Genuinely empty / scanned page

        page3_mock = MagicMock()
        page3_mock.get_text.side_effect = RuntimeError("Font descriptor parse error on page 3")

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.needs_pass = False
        mock_doc.page_count = 3
        mock_doc.metadata = {}
        mock_doc.__getitem__.side_effect = [page1_mock, page2_mock, page3_mock]

        with patch("pymupdf.open", return_value=mock_doc):
            result = parser.parse(b"%PDF-1.4 mock stream")

            assert result.diagnostics.total_pages == 3
            assert result.diagnostics.pages_with_text == 1
            assert result.diagnostics.pages_without_text == 1
            assert result.diagnostics.pages_with_errors == 1

            assert result.pages[0].extraction_status == "SUCCESS"
            assert result.pages[0].text == "Page 1 successfully extracted text."

            assert result.pages[1].extraction_status == "NO_TEXT"
            assert result.pages[1].text == ""

            assert result.pages[2].extraction_status == "ERROR"
            assert result.pages[2].text == ""

            assert any("Page 3: Extraction error" in w for w in result.diagnostics.warnings)
            assert any("Page 2 contains no extractable text" in w for w in result.diagnostics.warnings)

    def test_all_pages_empty_raises_pdf_no_extractable_text_error(self, parser: PDFParser) -> None:
        """When all pages contain zero extractable text (no errors), raise PDFNoExtractableTextError."""
        pdf_bytes = create_synthetic_pdf(["", "", ""])

        with pytest.raises(PDFNoExtractableTextError, match="No extractable text found across all 3 page"):
            parser.parse(pdf_bytes)

    def test_all_pages_error_raises_pdf_parsing_error(self, parser: PDFParser) -> None:
        """When extraction fails on all pages due to exceptions, raise PDFParsingError."""
        page_mock = MagicMock()
        page_mock.get_text.side_effect = RuntimeError("Corrupted page stream")

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.needs_pass = False
        mock_doc.page_count = 2
        mock_doc.metadata = {}
        mock_doc.__getitem__.side_effect = [page_mock, page_mock]

        with patch("pymupdf.open", return_value=mock_doc):
            with pytest.raises(PDFParsingError, match="encountered extraction errors"):
                parser.parse(b"%PDF-1.4 mock stream")

    def test_encrypted_pdf_raises_pdf_encrypted_error(self, parser: PDFParser) -> None:
        """Password-protected or encrypted PDF raises PDFEncryptedError."""
        encrypted_bytes = create_encrypted_pdf("Confidential Research")

        with pytest.raises(PDFEncryptedError, match="password-protected or encrypted"):
            parser.parse(encrypted_bytes)

    def test_corrupted_pdf_raises_pdf_parsing_error(self, parser: PDFParser) -> None:
        """Corrupted PDF stream raises PDFParsingError."""
        corrupted_bytes = b"%PDF-1.4\nCorrupted content without valid xref or objects %%EOF"

        with pytest.raises(PDFParsingError, match="Failed to open PDF document"):
            parser.parse(corrupted_bytes)

    def test_empty_bytes_raises_pdf_parsing_error(self, parser: PDFParser) -> None:
        """Empty bytes input raises PDFParsingError."""
        with pytest.raises(PDFParsingError, match="PDF input is empty"):
            parser.parse(b"")

    @pytest.mark.parametrize(
        "invalid_input",
        ["string_data", 12345, None, ["list"], {"key": "val"}, bytearray(b"%PDF-1.4")],
    )
    def test_non_bytes_input_raises_pdf_parsing_error(
        self,
        parser: PDFParser,
        invalid_input: Any,
    ) -> None:
        """Non-bytes inputs (including bytearray) raise PDFParsingError."""
        with pytest.raises(PDFParsingError, match="PDF input must be a bytes instance"):
            parser.parse(invalid_input)

    def test_missing_pdf_magic_header_raises_pdf_parsing_error(self, parser: PDFParser) -> None:
        """Input lacking %PDF- header signature raises PDFParsingError."""
        with pytest.raises(PDFParsingError, match="Invalid PDF header: content must start with b'%PDF-'"):
            parser.parse(b"NOT_A_PDF_STREAM")

    def test_zero_pages_raises_pdf_parsing_error(self, parser: PDFParser) -> None:
        """A PDF with zero pages raises PDFParsingError."""
        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.needs_pass = False
        mock_doc.page_count = 0
        mock_doc.metadata = {}

        with patch("pymupdf.open", return_value=mock_doc):
            with pytest.raises(PDFParsingError, match="contains zero pages"):
                parser.parse(b"%PDF-1.4 mock stream")
            mock_doc.close.assert_called_once()

    def test_document_is_closed_on_page_extraction_failure(self, parser: PDFParser) -> None:
        """PDF document is safely closed in finally block even if page extraction raises an error."""
        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.needs_pass = False
        mock_doc.page_count = 1
        mock_doc.metadata = {}
        mock_doc.__getitem__.side_effect = RuntimeError("Memory corruption reading page")

        with patch("pymupdf.open", return_value=mock_doc):
            with pytest.raises(PDFParsingError, match="encountered extraction errors"):
                parser.parse(b"%PDF-1.4 mock stream")
            mock_doc.close.assert_called_once()


# =========================================================================
# 2. Academic Metadata Extraction Tests
# =========================================================================


class TestAcademicMetadataExtraction:
    """Tests for heuristic extraction of title, authors, abstract, DOI, keywords, and publication details."""

    def test_academic_metadata_full_paper_extraction(self, parser: PDFParser) -> None:
        """Extracts complete academic metadata from a typical research paper layout."""
        page_1_text = (
            "arXiv:2301.05678v2 [cs.CL] 15 Feb 2023\n"
            "Neural Symbolic Reasoning for Deep Language Understanding\n"
            "Alice Johnson1, Bob Williams2, and Carol Davis1\n"
            "1Department of Computer Science, Stanford University\n"
            "2Google DeepMind\n"
            "{alice, carol}@cs.stanford.edu, bob@google.com\n"
            "Abstract\n"
            "This paper explores the synthesis of neural language models and symbolic reasoning. "
            "We demonstrate substantial improvements in multi-step logical deduction tasks.\n"
            "Keywords: neural-symbolic reasoning, language models, formal logic, automated deduction\n"
            "1. Introduction\n"
            "Large language models have achieved remarkable success across diverse tasks..."
        )
        page_2_text = (
            "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics\n"
            "DOI: 10.18653/v1/2023.acl-long.42\n"
            "Published: 15 July 2023\n"
            "© 2023 Association for Computational Linguistics\n"
            "2. Methodology and Implementation..."
        )

        pdf_bytes = create_synthetic_pdf([page_1_text, page_2_text])
        result = parser.parse(pdf_bytes)

        meta = result.academic_metadata
        assert isinstance(meta, AcademicMetadata)

        # Title
        assert meta.title == "Neural Symbolic Reasoning for Deep Language Understanding"

        # Authors and affiliations
        assert len(meta.authors) == 3
        author_names = [a.name for a in meta.authors]
        assert author_names == ["Alice Johnson", "Bob Williams", "Carol Davis"]
        assert meta.authors[0].affiliation == "Department of Computer Science, Stanford University"
        assert meta.authors[1].affiliation == "Google DeepMind"
        assert meta.authors[2].affiliation == "Department of Computer Science, Stanford University"

        # Abstract
        assert meta.abstract is not None
        assert meta.abstract.startswith("This paper explores the synthesis of neural language models")
        assert "multi-step logical deduction tasks." in meta.abstract

        # DOI
        assert meta.doi == "10.18653/v1/2023.acl-long.42"

        # Keywords
        assert len(meta.keywords) == 4
        assert "neural-symbolic reasoning" in meta.keywords
        assert "language models" in meta.keywords
        assert "formal logic" in meta.keywords
        assert "automated deduction" in meta.keywords

        # Publication details
        assert meta.publication_year == 2023
        assert meta.venue is not None
        assert "Association for Computational Linguistics" in meta.venue

    def test_single_author_two_word_name(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Single author with two-word name is recognized and not appended to the title."""
        page_text = (
            "Neural Radiance Fields for Novel View Synthesis\n"
            "Ben Mildenhall\n"
            "University of California, Berkeley\n"
            "Abstract\n"
            "We present a method that achieves state-of-the-art results for novel view synthesis."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.title == "Neural Radiance Fields for Novel View Synthesis"
        assert len(meta.authors) == 1
        assert meta.authors[0].name == "Ben Mildenhall"
        assert meta.authors[0].affiliation == "University of California, Berkeley"

    def test_single_author_without_affiliation(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Single author without affiliation followed directly by abstract."""
        page_text = (
            "Fast Approximate Nearest Neighbor Search\n"
            "John Smith\n"
            "Abstract\n"
            "Nearest neighbor search is a fundamental algorithmic primitive in machine learning."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.title == "Fast Approximate Nearest Neighbor Search"
        assert len(meta.authors) == 1
        assert meta.authors[0].name == "John Smith"
        assert meta.authors[0].affiliation is None

    def test_multiline_title_followed_by_multiple_authors(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Multiline title correctly separated from author lines."""
        page_text = (
            "Deep Residual Learning\n"
            "for Image Recognition\n"
            "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun\n"
            "Microsoft Research\n"
            "Abstract\n"
            "Deeper neural networks are more difficult to train."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.title == "Deep Residual Learning for Image Recognition"
        assert len(meta.authors) == 4
        author_names = [a.name for a in meta.authors]
        assert author_names == ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"]
        assert all(a.affiliation == "Microsoft Research" for a in meta.authors)

    def test_title_looks_like_person_name(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """A paper title that resembles a person's name is not mistaken for an author."""
        page_text = (
            "Alan Turing\n"
            "John Smith\n"
            "University of Cambridge\n"
            "Abstract\n"
            "A historical retrospective on the computational contributions of Alan Turing."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.title == "Alan Turing"
        assert len(meta.authors) == 1
        assert meta.authors[0].name == "John Smith"
        assert meta.authors[0].affiliation == "University of Cambridge"

    def test_three_authors_one_shared_affiliation(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Three authors sharing one unambiguous affiliation."""
        page_text = (
            "Scalable Deep Clustering\n"
            "Alice Johnson, Bob Williams, Carol Davis\n"
            "Department of Computer Science, Stanford University\n"
            "Abstract\n"
            "We propose a scalable clustering algorithm for large-scale embedding spaces."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert len(meta.authors) == 3
        assert all(a.affiliation == "Department of Computer Science, Stanford University" for a in meta.authors)

    def test_three_authors_two_numbered_affiliations(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Authors with numbered markers mapped to corresponding numbered affiliations."""
        page_text = (
            "Reinforcement Learning with Verifiable Constraints\n"
            "Alice Johnson1, Bob Williams2, Carol Davis1\n"
            "1Department of Computer Science, Stanford University\n"
            "2Google DeepMind\n"
            "Abstract\n"
            "Constrained Markov decision processes provide a principled framework for safe exploration."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert len(meta.authors) == 3
        assert meta.authors[0].name == "Alice Johnson"
        assert meta.authors[0].affiliation == "Department of Computer Science, Stanford University"
        assert meta.authors[1].name == "Bob Williams"
        assert meta.authors[1].affiliation == "Google DeepMind"
        assert meta.authors[2].name == "Carol Davis"
        assert meta.authors[2].affiliation == "Department of Computer Science, Stanford University"

    def test_multiple_affiliations_without_clear_mapping_leaves_none(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Multiple affiliations without clear numbered markers are not assigned to avoid false associations."""
        page_text = (
            "Distributed Consensus in Multi-Agent Systems\n"
            "Alice Johnson, Bob Williams, Carol Davis\n"
            "Stanford University\n"
            "Google DeepMind\n"
            "Abstract\n"
            "We analyze convergence bounds for distributed consensus under adversarial delays."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert len(meta.authors) == 3
        # Affiliation mapping is ambiguous, so left as None conservatively
        assert all(a.affiliation is None for a in meta.authors)

    def test_authors_without_affiliations_leaves_none(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Authors without affiliations have affiliation set to None."""
        page_text = (
            "Kernel Methods for High-Dimensional Density Estimation\n"
            "Alice Johnson, Bob Williams, and Carol Davis\n"
            "Abstract\n"
            "Nonparametric density estimation faces severe curse-of-dimensionality challenges."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert len(meta.authors) == 3
        assert all(a.affiliation is None for a in meta.authors)

    def test_title_skips_common_headers_and_page_numbers(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Title extraction ignores arXiv tags, journal headers, and page numbers."""
        page_text = (
            "arXiv:2104.12345v1 [cs.AI]\n"
            "IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE\n"
            "Vol. 44, No. 12, December 2022\n"
            "Self-Supervised Representation Learning for Graph Neural Networks\n"
            "David Miller and Sarah Connor\n"
            "Abstract\n"
            "We propose a self-supervised approach for learning node representations on graphs."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.title == "Self-Supervised Representation Learning for Graph Neural Networks"

    def test_title_fallback_to_pdf_metadata(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """When page text has no clear title, fall back to embedded PDF metadata title."""
        page_text = (
            "Abstract\n"
            "A comprehensive study of optimization landscapes in high-dimensional deep neural networks."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        pdf_meta = PDFMetadata(title="Optimization Landscapes of Deep Neural Networks")

        meta = extractor.extract(pages, pdf_meta)
        assert meta.title == "Optimization Landscapes of Deep Neural Networks"

    def test_generic_pdf_metadata_title_ignored(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Generic placeholder titles like 'untitled' or 'Microsoft Word' are rejected."""
        page_text = "Abstract\nSimple abstract without a recognizable title above it."
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        pdf_meta = PDFMetadata(title="Microsoft Word - Document1.docx")

        meta = extractor.extract(pages, pdf_meta)
        assert meta.title is None

    def test_abstract_with_summary_heading(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Summary heading is detected and extracted as the abstract."""
        page_text = (
            "Scalable Graph Analytics at Web Scale\n"
            "Author One\n"
            "Summary:\n"
            "In this work, we present a distributed graph analytics engine designed for petabyte-scale graphs.\n"
            "1. Introduction\n"
            "Graph-structured data is ubiquitous across social networks and bioinformatics."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.abstract is not None
        assert meta.abstract.startswith("In this work, we present a distributed graph analytics engine")

    @pytest.mark.parametrize(
        "raw_doi_text,expected_doi",
        [
            ("https://doi.org/10.1016/j.artint.2023.103980", "10.1016/j.artint.2023.103980"),
            ("http://dx.doi.org/10.1109/TPAMI.2022.3150241.", "10.1109/tpami.2022.3150241"),
            ("doi: 10.1145/3308558.3313528;", "10.1145/3308558.3313528"),
            ("Published under DOI: 10.48550/arXiv.2301.00001)", "10.48550/arxiv.2301.00001"),
        ],
    )
    def test_doi_extraction_and_normalization(
        self,
        extractor: ResearchMetadataExtractor,
        raw_doi_text: str,
        expected_doi: str,
    ) -> None:
        """DOIs with URL prefixes or punctuation are normalized to clean lowercase identifier."""
        page_text = f"Sample Research Paper\n{raw_doi_text}\nAbstract\nBrief summary text for extraction testing."
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.doi == expected_doi

    def test_keywords_with_index_terms_heading(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Index Terms heading is recognized and parsed into individual keyword strings."""
        page_text = (
            "Paper Title\nAuthor Name\nAbstract\nShort abstract text.\n"
            "Index Terms—Deep learning, convolutional networks, image classification; transfer learning.\n"
            "I. Introduction\n"
            "Introduction text follows here."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert len(meta.keywords) == 4
        assert "Deep learning" in meta.keywords
        assert "convolutional networks" in meta.keywords
        assert "image classification" in meta.keywords
        assert "transfer learning" in meta.keywords

    def test_received_year_different_from_published_year(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Received year is not used as publication year when explicit publication year is present."""
        page_text = (
            "Adaptive Optimization for Language Modeling\n"
            "Author Name\n"
            "Received: 14 October 2021; Published: 20 March 2023\n"
            "Abstract\nValid abstract text."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.publication_year == 2023
        assert "2023" in (meta.publication_date or "")

    def test_accepted_year_different_from_published_year(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Accepted year is not used as publication year when explicit published year is present."""
        page_text = (
            "Generalization Guarantees for Attention Models\n"
            "Author Name\n"
            "Accepted: 18 December 2022; Published online: 15 January 2024\n"
            "Abstract\nValid abstract text."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.publication_year == 2024
        assert "2024" in (meta.publication_date or "")

    def test_only_received_and_accepted_years_keeps_uncertain_year_none(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """When text only contains submission/review dates and no published date, year remains None."""
        page_text = (
            "Sparse Matrix Decomposition\n"
            "Author Name\n"
            "Received: 10 May 2021; Accepted: 12 August 2022\n"
            "Abstract\nValid abstract text."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.publication_year is None
        assert meta.publication_date is None

    def test_publication_details_from_copyright_and_publisher(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Publisher and year are identified from standard copyright lines."""
        page_text = (
            "Efficient Sparse Attention Transformers\n"
            "Author A and Author B\n"
            "Abstract\nValid abstract text.\n"
            "© 2021 IEEE. Personal use of this material is permitted."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.publication_year == 2021
        assert meta.publisher == "IEEE"

    def test_venue_detection_neurips_acronym(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Known conference acronyms like NeurIPS are identified as venues."""
        page_text = (
            "36th Conference on Neural Information Processing Systems (NeurIPS 2022)\n"
            "Contrastive Learning of Visual Representations\n"
            "Author A\nAbstract\nAbstract content here."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.venue is not None
        assert "NeurIPS" in meta.venue

    def test_missing_metadata_leaves_fields_none_or_empty(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """When text does not contain academic metadata, fields remain None or empty list conservatively."""
        raw_text = "Just some arbitrary document notes without title, author, or abstract sections."
        pages = [ParsedPage(page_number=1, text=raw_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.abstract is None
        assert meta.doi is None
        assert meta.keywords == []
        assert meta.publication_year is None
        assert meta.venue is None
        assert meta.publisher is None

    def test_empty_pages_list_returns_empty_academic_metadata(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Empty page list returns default AcademicMetadata."""
        meta = extractor.extract([])
        assert isinstance(meta, AcademicMetadata)
        assert meta.title is None
        assert meta.authors == []

    def test_multiline_title_containing_research_is_preserved(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Multiline title containing 'Research' is preserved and not mistaken for an affiliation."""
        page_text = (
            "A Survey of Modern Machine Learning\n"
            "Research Methods for Large Language Models\n"
            "John Smith\n"
            "University of Delhi\n"
            "Abstract\n"
            "We present a comprehensive survey of methods for large language models."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.title == "A Survey of Modern Machine Learning Research Methods for Large Language Models"
        assert len(meta.authors) == 1
        assert meta.authors[0].name == "John Smith"
        assert meta.authors[0].affiliation == "University of Delhi"

    def test_multiline_title_containing_laboratory_or_lab_is_preserved(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Multiline titles containing 'Laboratory' or 'Lab' in academic context remain part of the title."""
        # Case A: Title containing 'A Laboratory Evaluation...'
        page_text_lab1 = (
            "Speech and Language Processing\n"
            "A Laboratory Evaluation of Machine Translation\n"
            "Jane Doe\n"
            "University of Cambridge\n"
            "Abstract\n"
            "We evaluate machine translation systems in controlled laboratory conditions."
        )
        pages1 = [ParsedPage(page_number=1, text=page_text_lab1, extraction_status="SUCCESS")]
        meta1 = extractor.extract(pages1)

        assert meta1.title == "Speech and Language Processing A Laboratory Evaluation of Machine Translation"
        assert len(meta1.authors) == 1
        assert meta1.authors[0].name == "Jane Doe"
        assert meta1.authors[0].affiliation == "University of Cambridge"

        # Case B: Title containing 'Lab Experiments in...'
        page_text_lab2 = (
            "Deep Learning in Natural Language Processing\n"
            "Lab Experiments in Speech Recognition\n"
            "Robert Brown\n"
            "Stanford University\n"
            "Abstract\n"
            "This study investigates acoustic modeling through lab experiments."
        )
        pages2 = [ParsedPage(page_number=1, text=page_text_lab2, extraction_status="SUCCESS")]
        meta2 = extractor.extract(pages2)

        assert meta2.title == "Deep Learning in Natural Language Processing Lab Experiments in Speech Recognition"
        assert len(meta2.authors) == 1
        assert meta2.authors[0].name == "Robert Brown"
        assert meta2.authors[0].affiliation == "Stanford University"

    def test_genuine_affiliation_containing_microsoft_research(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Genuine affiliation lines containing 'Microsoft Research' are correctly recognized."""
        page_text = (
            "Representation Learning for Natural Language\n"
            "Kaiming He, Jian Sun\n"
            "Microsoft Research\n"
            "Abstract\n"
            "We propose novel representations for language modeling."
        )
        pages = [ParsedPage(page_number=1, text=page_text, extraction_status="SUCCESS")]
        meta = extractor.extract(pages)

        assert meta.title == "Representation Learning for Natural Language"
        assert len(meta.authors) == 2
        assert [a.name for a in meta.authors] == ["Kaiming He", "Jian Sun"]
        assert all(a.affiliation == "Microsoft Research" for a in meta.authors)

    def test_genuine_affiliation_containing_artificial_intelligence_laboratory(
        self,
        extractor: ResearchMetadataExtractor,
    ) -> None:
        """Genuine affiliation lines containing 'Artificial Intelligence Laboratory' are recognized."""
        # Single author with Artificial Intelligence Laboratory
        page_text_single = (
            "Autonomous Symbolic Deduction Systems\n"
            "Alice Johnson\n"
            "Artificial Intelligence Laboratory\n"
            "Abstract\n"
            "We present formal deduction systems in autonomous environments."
        )
        pages_single = [ParsedPage(page_number=1, text=page_text_single, extraction_status="SUCCESS")]
        meta_single = extractor.extract(pages_single)

        assert meta_single.title == "Autonomous Symbolic Deduction Systems"
        assert len(meta_single.authors) == 1
        assert meta_single.authors[0].name == "Alice Johnson"
        assert meta_single.authors[0].affiliation == "Artificial Intelligence Laboratory"

        # Multiple authors with numbered affiliation
        page_text_numbered = (
            "Collaborative Multi-Agent Verification\n"
            "Alice Johnson1, Bob Williams2\n"
            "1Artificial Intelligence Laboratory\n"
            "2Microsoft Research\n"
            "Abstract\n"
            "Formal multi-agent verification methods."
        )
        pages_numbered = [ParsedPage(page_number=1, text=page_text_numbered, extraction_status="SUCCESS")]
        meta_numbered = extractor.extract(pages_numbered)

        assert meta_numbered.title == "Collaborative Multi-Agent Verification"
        assert len(meta_numbered.authors) == 2
        assert meta_numbered.authors[0].affiliation == "Artificial Intelligence Laboratory"
        assert meta_numbered.authors[1].affiliation == "Microsoft Research"
