"""Research paper PDF parsing and academic metadata extraction layer.

This module provides deterministic, in-memory PDF parsing using PyMuPDF (pymupdf),
extracting embedded PDF metadata, page-by-page text, parsing diagnostics, and academic
metadata candidates (title, authors, abstract, DOI, keywords, publication details).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import pymupdf

from app.core.exceptions import (
    PDFEncryptedError,
    PDFNoExtractableTextError,
    PDFParsingError,
)

# Common affiliation and organization indicator keywords
AFFILIATION_KEYWORDS: tuple[str, ...] = (
    "university",
    "department",
    "dept.",
    "dept",
    "institute",
    "laboratory",
    "lab",
    "laboratories",
    "faculty",
    "school",
    "college",
    "center",
    "centre",
    "inc.",
    "inc",
    "corp.",
    "corp",
    "corporation",
    "ltd",
    "research",
    "google",
    "microsoft",
    "meta",
    "amazon",
    "apple",
    "openai",
    "deepmind",
    "ibm",
    "intel",
    "adobe",
    "telecom",
    "polytechnic",
    "hospital",
    "academy",
    "sciences",
)

# Strong affiliation indicators with explicit word boundaries
STRONG_AFFILIATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Academic institutions and organizational units
    re.compile(r"\buniversit(?:y|ies|ät|e|ad|aria)?\b", re.IGNORECASE),
    re.compile(r"\bdepartment(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bdept(?:s|\.)?\b", re.IGNORECASE),
    re.compile(r"\bdivision(?:s)?\b", re.IGNORECASE),
    re.compile(r"\binstitut(?:e|es|ion|ions|o)?\b", re.IGNORECASE),
    re.compile(r"\bpolytechnic\b", re.IGNORECASE),
    re.compile(r"\bcollege(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bfaculty\b", re.IGNORECASE),
    re.compile(r"\bschool\s+of\b", re.IGNORECASE),
    re.compile(r"\bhospital(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bacademy\b", re.IGNORECASE),
    re.compile(r"\b(?:center|centre)\s+(?:for|of)\b", re.IGNORECASE),
    # Known major industrial / tech research organizations
    re.compile(r"\bgoogle\b", re.IGNORECASE),
    re.compile(r"\bdeepmind\b", re.IGNORECASE),
    re.compile(r"\bmicrosoft\b", re.IGNORECASE),
    re.compile(r"\bopenai\b", re.IGNORECASE),
    re.compile(r"\bmeta\s+ai\b", re.IGNORECASE),
    re.compile(r"\bibm\b", re.IGNORECASE),
    re.compile(r"\bintel\b", re.IGNORECASE),
    re.compile(r"\badobe\b", re.IGNORECASE),
    re.compile(r"\bamazon(?:\s+science|\s+web\s+services|\s+aws|\s+ai|\s+research)?\b", re.IGNORECASE),
    # Corporate legal entity designations
    re.compile(r"\b(?:inc|corp|corporation|gmbh|llc|ltd)\b", re.IGNORECASE),
)

# Common title words, articles, and research topic connectors that contradict affiliation lines
TITLE_INDICATOR_WORDS: re.Pattern[str] = re.compile(
    r"\b(?:evaluation|evaluations|experiments?|measurement|measurements|studies|study|methods?|methodology|"
    r"benchmarking|benchmark|survey|surveys|analysis|analyses|investigation|investigations|approach|approaches|"
    r"systems?|models?|frameworks?|review|advances|techniques?|algorithms?|findings?|perspectives?|challenges?)\b",
    re.IGNORECASE,
)

TITLE_CONNECTOR_WORDS: re.Pattern[str] = re.compile(
    r"\b(?:towards?|via|with|using|between|across|through)\b",
    re.IGNORECASE,
)

# Known major academic publishers
KNOWN_PUBLISHERS: tuple[str, ...] = (
    "IEEE",
    "ACM",
    "Springer",
    "Elsevier",
    "Wiley",
    "Nature",
    "Science",
    "OpenReview",
    "arXiv",
    "PMLR",
    "Oxford University Press",
    "Cambridge University Press",
    "MIT Press",
)

# Known academic venues and conferences
KNOWN_VENUE_PATTERNS: tuple[str, ...] = (
    r"(?i)\bProceedings of (?:the )?[A-Za-z0-9\s,–-]+",
    r"(?i)\bIEEE Transactions on [A-Za-z0-9\s,–-]+",
    r"(?i)\bACM Transactions on [A-Za-z0-9\s,–-]+",
    r"(?i)\b(?:NeurIPS|ICLR|ICML|ACL|EMNLP|NAACL|EACL|CVPR|ICCV|ECCV|KDD|AAAI|IJCAI|SIGIR|WWW|Interspeech)\b",
)


@dataclass
class PDFMetadata:
    """Embedded metadata extracted directly from the PDF document catalog."""

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    page_count: int = 0


@dataclass
class AuthorCandidate:
    """An academic author candidate with optional institution or affiliation."""

    name: str
    affiliation: str | None = None


@dataclass
class AcademicMetadata:
    """Extracted academic metadata candidates from the research paper text."""

    title: str | None = None
    authors: list[AuthorCandidate] = field(default_factory=list)
    abstract: str | None = None
    doi: str | None = None
    keywords: list[str] = field(default_factory=list)
    publication_date: str | None = None
    publication_year: int | None = None
    venue: str | None = None
    publisher: str | None = None
    language: str | None = None
    paper_type: str | None = None


@dataclass
class ParsedPage:
    """Extracted text and status for an individual page."""

    page_number: int  # 1-indexed
    text: str
    extraction_status: str  # "SUCCESS", "NO_TEXT", or "ERROR"


@dataclass
class ParsingDiagnostics:
    """Diagnostics and extraction quality metrics for the parsed document."""

    total_pages: int
    pages_with_text: int
    pages_without_text: int
    pages_with_errors: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedResearchPaper:
    """Complete parsed research paper representation."""

    pdf_metadata: PDFMetadata
    academic_metadata: AcademicMetadata
    pages: list[ParsedPage]
    diagnostics: ParsingDiagnostics


class ResearchMetadataExtractor:
    """Deterministic academic metadata extractor using regular expressions and heuristics."""

    def extract(
        self,
        pages: list[ParsedPage],
        pdf_metadata: PDFMetadata | None = None,
    ) -> AcademicMetadata:
        """Extract academic metadata candidates from the parsed pages."""
        if not pages:
            return AcademicMetadata()

        first_page_text = pages[0].text if pages else ""
        second_page_text = pages[1].text if len(pages) > 1 else ""
        combined_initial_text = f"{first_page_text}\n{second_page_text}"
        full_text = "\n".join(p.text for p in pages)

        title, author_lines = self._extract_title_and_author_lines(first_page_text, pdf_metadata)
        authors = self._extract_authors_from_lines(author_lines)
        abstract = self._extract_abstract(combined_initial_text)
        doi = self._extract_doi(full_text)
        keywords = self._extract_keywords(combined_initial_text)
        year, date_str = self._extract_publication_date(combined_initial_text)
        venue = self._extract_venue(combined_initial_text)
        publisher = self._extract_publisher(combined_initial_text)

        return AcademicMetadata(
            title=title,
            authors=authors,
            abstract=abstract,
            doi=doi,
            keywords=keywords,
            publication_date=date_str,
            publication_year=year,
            venue=venue,
            publisher=publisher,
        )

    def _extract_title_and_author_lines(
        self,
        first_page_text: str,
        pdf_metadata: PDFMetadata | None = None,
    ) -> tuple[str | None, list[str]]:
        """Partition pre-abstract text into title and author lines consistently."""
        if not first_page_text.strip():
            # Fallback to embedded title if plausible
            fallback_title = None
            if pdf_metadata and pdf_metadata.title:
                candidate = pdf_metadata.title.strip()
                if candidate and not self._is_generic_title(candidate):
                    fallback_title = candidate
            return fallback_title, []

        abstract_match = re.search(r"(?i)\b(abstract|summary)\b", first_page_text)
        pre_abstract_text = first_page_text[: abstract_match.start()] if abstract_match else first_page_text

        lines = [line.strip() for line in pre_abstract_text.splitlines() if line.strip()]

        # Skip initial header noise
        idx = 0
        while idx < len(lines) and self._is_header_noise(lines[idx]):
            idx += 1

        remaining = lines[idx:]
        if not remaining:
            fallback_title = None
            if pdf_metadata and pdf_metadata.title:
                candidate = pdf_metadata.title.strip()
                if candidate and not self._is_generic_title(candidate):
                    fallback_title = candidate
            return fallback_title, []

        title_lines: list[str] = [remaining[0]]
        k = 1

        while k < len(remaining):
            curr = remaining[k]
            prev = remaining[k - 1]

            # 1. Immediate termination: affiliation or email line
            if self._is_affiliation_or_email(curr):
                break

            # 2. Immediate termination: footnote markers attached to names
            if re.search(r"[A-Z][a-z]+[\*†‡§\d]", curr):
                break

            # 3. Check author candidates
            names_with_markers = self._parse_author_line(curr)
            if len(names_with_markers) >= 2:
                # Multiple author names (e.g. "John Smith and Jane Doe" or "John Smith, Jane Doe")
                break
            if len(names_with_markers) == 1:
                # Single person name (e.g. "John Smith")
                # Title cannot continue if previous line didn't end with connector and next line confirms author block
                if not prev.endswith((":", "-", "—")):
                    next_line = remaining[k + 1] if k + 1 < len(remaining) else None
                    if (
                        next_line is None
                        or self._is_affiliation_or_email(next_line)
                        or len(self._parse_author_line(next_line)) >= 1
                    ):
                        break

            if len(title_lines) >= 3:
                break

            title_lines.append(curr)
            k += 1

        author_lines = remaining[k:]
        title_str = " ".join(title_lines).strip()

        if not (6 <= len(title_str) <= 350 and not self._is_generic_title(title_str)):
            title_str = None
            if pdf_metadata and pdf_metadata.title:
                candidate = pdf_metadata.title.strip()
                if candidate and not self._is_generic_title(candidate):
                    title_str = candidate

        return title_str, author_lines

    def _extract_authors_from_lines(self, author_lines: list[str]) -> list[AuthorCandidate]:
        """Extract author candidates and associate affiliations conservatively."""
        if not author_lines:
            return []

        author_candidates: list[tuple[str, str | None]] = []
        affiliations: list[tuple[str | None, str]] = []

        for line in author_lines:
            if self._is_affiliation_or_email(line):
                # Check for numbered marker e.g. "1Department of Computer Science..."
                m = re.match(r"^(\d+)[\s\.\-\:]*(.+)$", line.strip())
                if m:
                    marker = m.group(1).strip()
                    affil_text = m.group(2).strip()
                    if self._is_affiliation_line(affil_text):
                        affiliations.append((marker, affil_text))
                else:
                    affil_clean = re.sub(r"[\s\*†‡§]+", " ", line).strip()
                    # Only keep if contains organizational keywords or is non-email affiliation
                    if affil_clean and self._is_affiliation_line(affil_clean):
                        affiliations.append((None, affil_clean))
            else:
                names = self._parse_author_line(line)
                if names:
                    author_candidates.extend(names)

        if not author_candidates:
            return []

        # 1. Numbered / superscript marker mapping:
        marker_map = {m: text for m, text in affiliations if m is not None}
        has_numbered_mapping = any(m is not None for _, m in author_candidates) and bool(marker_map)

        results: list[AuthorCandidate] = []
        if has_numbered_mapping:
            for name, marker in author_candidates:
                affil = marker_map.get(marker) if marker else None
                results.append(AuthorCandidate(name=name, affiliation=affil))
            return results

        # 2. Single unambiguous shared affiliation:
        if len(affiliations) == 1:
            shared_affil = affiliations[0][1]
            for name, _ in author_candidates:
                results.append(AuthorCandidate(name=name, affiliation=shared_affil))
            return results

        # 3. Multiple affiliations without clear author mapping (ambiguous mapping) or no affiliations:
        # Leave affiliation as None to avoid fabricating incorrect associations
        for name, _ in author_candidates:
            results.append(AuthorCandidate(name=name, affiliation=None))

        return results

    def _is_header_noise(self, line: str) -> bool:
        """Check if a line is common academic header noise."""
        lower = line.lower()
        if re.match(r"^arxiv:\s*\d{4}\.\d{4,5}", lower):
            return True
        if lower.startswith(("https://", "http://", "doi:", "isbn", "issn", "vol.", "volume", "issue", "no.")):
            return True
        if re.match(r"^(?:page\s+)?\d+$", lower):
            return True
        if any(h in lower for h in ("under review", "preprint", "draft", "copyright", "all rights reserved")):
            return True
        if any(lower.startswith(prefix) for prefix in ("proceedings of", "ieee transactions", "acm transactions")):
            return True
        return False

    def _is_email_line(self, line: str) -> bool:
        """Check if a line contains an email address."""
        return "@" in line and not line.strip().lower().startswith(("http://", "https://"))

    def _is_laboratory_affiliation(self, text: str) -> bool:
        """Check if text represents an academic or industrial laboratory entity.

        Distinguishes institutional lab names (e.g., 'Artificial Intelligence Laboratory')
        from research paper titles containing 'Laboratory' or 'Lab' in their academic
        meaning (e.g., 'A Laboratory Evaluation of Machine Translation').
        """
        cleaned = re.sub(r"^[\d\*\†\‡\§\s\.\-\:\[\]\(\)]+", "", text).strip()
        if not cleaned:
            return False

        # Obvious title-starting articles e.g. "A Laboratory...", "The Lab..."
        if re.match(r"^(?:a|an|the)\s+", cleaned, re.IGNORECASE):
            return False

        # Academic method/evaluation words or title connectors indicate title lines
        if TITLE_INDICATOR_WORDS.search(cleaned) or TITLE_CONNECTOR_WORDS.search(cleaned):
            return False

        # Laboratory head noun ending e.g. "Artificial Intelligence Laboratory" or "Speech Processing Lab"
        if re.search(r"\b(?:laboratory|laboratories|labs?)\b[\s\*†‡§\d\.\,\-]*$", cleaned, re.IGNORECASE):
            return True

        # Institutional lab prefix e.g. "Laboratory of Computer Science", "Lab for Information Systems"
        if re.match(r"^(?:the\s+)?\b(?:laboratory|laboratories|labs?)\s+(?:of|for)\b", cleaned, re.IGNORECASE):
            return True

        return False

    def _is_affiliation_line(self, text: str) -> bool:
        """Check if a line represents an academic or corporate affiliation."""
        cleaned = re.sub(r"^[\d\*\†\‡\§\s\.\-\:\[\]\(\)]+", "", text).strip()
        if not cleaned:
            return False

        if self._is_email_line(cleaned):
            return False

        # 1. Strong affiliation indicators with explicit word boundaries
        for pattern in STRONG_AFFILIATION_PATTERNS:
            if pattern.search(cleaned):
                return True

        # 2. Institutional laboratory affiliations (treating 'lab'/'laboratory' as weak indicators)
        if self._is_laboratory_affiliation(cleaned):
            return True

        return False

    def _is_affiliation_or_email(self, line: str) -> bool:
        """Check if a line is an affiliation, department, or email address."""
        return self._is_email_line(line) or self._is_affiliation_line(line)

    def _parse_author_line(self, line: str) -> list[tuple[str, str | None]]:
        """Parse potential person names and optional numbered markers from an author line."""
        cleaned_line = re.sub(r"[\*†‡§]+", "", line).strip()
        if not cleaned_line or len(cleaned_line) > 160:
            return []

        tokens = re.split(r",|\band\b|&", cleaned_line, flags=re.IGNORECASE)
        valid_authors: list[tuple[str, str | None]] = []

        for token in tokens:
            t = token.strip()
            if not t:
                continue

            # Check if token ends with digit marker e.g. "Alice Johnson1"
            m = re.search(r"^([A-Za-z\.\-\s]+?)(\d+)$", t)
            if m:
                name = m.group(1).strip()
                marker = m.group(2).strip()
            else:
                name = t
                marker = None

            words = name.split()
            # Validate person name shape: 2 to 4 words, each capitalized or initial
            if 2 <= len(words) <= 4:
                if all(re.match(r"^[A-Z][a-zA-Z\.\-]*$", w) for w in words):
                    if not self._is_affiliation_line(name):
                        valid_authors.append((name, marker))

        return valid_authors

    def _is_generic_title(self, title: str) -> bool:
        """Check if title is a generic tool/software placeholder."""
        lower = title.lower()
        generic_markers = (
            "untitled",
            "microsoft word",
            "latex",
            "default",
            "paper.pdf",
            "document",
            "presentation",
        )
        return any(lower.startswith(m) or lower == m for m in generic_markers)

    def _extract_abstract(self, text: str) -> str | None:
        """Extract abstract text following an Abstract/Summary heading."""
        pattern = (
            r"(?is)(?:^|\n)\s*(?:abstract|summary)\s*[:—\-\.]*\s*"
            r"(.*?)"
            r"(?=\n\s*(?:(?:[1I]\.?|1\.0)?\s*Introduction\b|Keywords\b|Key\s*words\b|Index\s*terms\b|ACM\s+Reference\b|[1I]\s+[A-Z]|\Z))"
        )
        match = re.search(pattern, text)
        if match:
            abstract_content = match.group(1).strip()
            # Normalize whitespace
            abstract_clean = re.sub(r"\s+", " ", abstract_content).strip()
            if len(abstract_clean) >= 20:
                return abstract_clean
        return None

    def _extract_doi(self, text: str) -> str | None:
        """Extract and normalize DOI from the document text."""
        # Match standard DOI with optional URL or doi: prefix
        pattern = r"(?i)(?:https?://(?:dx\.)?doi\.org/|doi:\s*)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)"
        match = re.search(pattern, text)
        if not match:
            # Fallback to bare DOI pattern
            bare_pattern = r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)"
            match = re.search(bare_pattern, text)

        if match:
            raw_doi = match.group(1).strip()
            # Strip trailing punctuation
            raw_doi = re.sub(r"[\.,;:>\)\]\"']+$", "", raw_doi)
            if raw_doi:
                return raw_doi.lower()

        return None

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract author keywords following standard headings."""
        pattern = (
            r"(?is)(?:keywords|key\s*words|index\s*terms)[\s:—\-\.]+"
            r"(.+?)"
            r"(?=\n\s*(?:(?:[1I]\.?|1\.0)?\s*Introduction\b|Introduction\b|ACM\s+Reference\b|[1I]\s+[A-Z]|\n\n|\Z))"
        )
        match = re.search(pattern, text)
        if not match:
            return []

        raw_keywords = match.group(1).strip()
        tokens = re.split(r"[,;•—]", raw_keywords)
        keywords: list[str] = []

        for token in tokens:
            kw = token.strip().strip("'\"").rstrip(".").strip()
            if kw and 2 <= len(kw) <= 80:
                kw_clean = re.sub(r"\s+", " ", kw)
                keywords.append(kw_clean)

        return keywords

    def _extract_publication_date(self, text: str) -> tuple[int | None, str | None]:
        """Extract publication year and date string conservatively from text.

        Distinguishes publication dates from submission/received/accepted dates.
        Only explicit publication, copyright, or verified conference/arXiv publication
        information is considered.
        """
        year: int | None = None
        date_str: str | None = None

        # 1. Look for explicit publication date: "Published: August 2024", "Published online: 2024"
        pub_match = re.search(
            r"(?i)\b(?:published(?:\s+online)?|appeared\s+in|publication\s+date)[\s:—\-]+([A-Za-z]+\s+[12]\d{3}|\d{1,2}\s+[A-Za-z]+\s+[12]\d{3}|[12]\d{3})",
            text,
        )
        if pub_match:
            date_str = pub_match.group(1).strip()
            year_match = re.search(r"\b([12]\d{3})\b", date_str)
            if year_match:
                cand = int(year_match.group(1))
                if 1900 <= cand <= 2030:
                    year = cand

        # 2. Look for explicit copyright notice: © 2024 or Copyright 2024
        if year is None:
            copyright_match = re.search(r"(?i)(?:©|Copyright|\(c\))\s*([12]\d{3})", text)
            if copyright_match:
                cand = int(copyright_match.group(1))
                if 1900 <= cand <= 2030:
                    year = cand
                    if date_str is None:
                        date_str = str(year)

        # 3. Look for arXiv publication date in header stamp: arXiv:2301.12345v1 [cs.CL] 15 Jan 2023
        if year is None:
            arxiv_date_match = re.search(
                r"(?i)arxiv:\s*\d{4}\.\d{4,5}(?:v\d+)?\s+\[.*?\]\s+(\d{1,2}\s+[A-Za-z]+\s+([12]\d{3}))",
                text,
            )
            if arxiv_date_match:
                date_str = arxiv_date_match.group(1).strip()
                cand = int(arxiv_date_match.group(2))
                if 1900 <= cand <= 2030:
                    year = cand

        # 4. Look for conference/proceedings publication year in header: "Proceedings of ... (2024)"
        if year is None:
            conf_year_match = re.search(
                r"(?i)\b(?:Proceedings of (?:the )?[A-Za-z0-9\s,–-]+?|Conference on [A-Za-z0-9\s,–-]+?)\b.*?([12]\d{3})",
                text[:1000],
            )
            if conf_year_match:
                cand = int(conf_year_match.group(1))
                if 1900 <= cand <= 2030:
                    year = cand

        return year, date_str

    def _extract_venue(self, text: str) -> str | None:
        """Extract conference, journal, or venue name if clearly identifiable."""
        for pattern in KNOWN_VENUE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                venue_str = match.group(0).strip()
                venue_clean = re.sub(r"\s+", " ", venue_str).strip()
                if 3 <= len(venue_clean) <= 150:
                    return venue_clean
        return None

    def _extract_publisher(self, text: str) -> str | None:
        """Extract publisher name if clearly identifiable in copyright/header lines."""
        for publisher in KNOWN_PUBLISHERS:
            pattern = rf"(?i)(?:©|Copyright|Published by|Published in|By)\s+.*?\b{re.escape(publisher)}\b"
            if re.search(pattern, text):
                return publisher
            header_pattern = rf"(?i)^[^\n]*\b{re.escape(publisher)}\b[^\n]*$"
            if re.search(header_pattern, text[:1000], re.MULTILINE):
                return publisher
        return None


class PDFParser:
    """PDF parser component utilizing PyMuPDF for in-memory text and metadata extraction."""

    def __init__(self, metadata_extractor: ResearchMetadataExtractor | None = None) -> None:
        """Initialize PDF parser with optional research metadata extractor."""
        self.metadata_extractor = metadata_extractor or ResearchMetadataExtractor()

    def parse(self, pdf_bytes: bytes) -> ParsedResearchPaper:
        """Parse raw PDF bytes into structured document, pages, diagnostics, and metadata.

        Args:
            pdf_bytes: Raw bytes of the PDF file.

        Returns:
            ParsedResearchPaper with pdf_metadata, academic_metadata, pages, and diagnostics.

        Raises:
            PDFParsingError: If bytes are invalid, empty, lack %PDF- header, corrupted,
                             or if extraction failures prevent document assessment.
            PDFEncryptedError: If document is password-protected or encrypted.
            PDFNoExtractableTextError: If document contains pages but zero extractable text.
        """
        self._validate_input(pdf_bytes)

        doc: pymupdf.Document | None = None
        try:
            try:
                doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            except Exception as e:
                raise PDFParsingError(f"Failed to open PDF document: {e}") from e

            if doc.is_encrypted or getattr(doc, "needs_pass", False):
                raise PDFEncryptedError("PDF document is password-protected or encrypted.")

            total_pages = doc.page_count
            if total_pages == 0:
                raise PDFParsingError("PDF document contains zero pages.")

            # Extract embedded PDF metadata
            raw_meta = doc.metadata or {}
            pdf_meta = PDFMetadata(
                title=self._clean_str(raw_meta.get("title")),
                author=self._clean_str(raw_meta.get("author")),
                subject=self._clean_str(raw_meta.get("subject")),
                keywords=self._clean_str(raw_meta.get("keywords")),
                creator=self._clean_str(raw_meta.get("creator")),
                producer=self._clean_str(raw_meta.get("producer")),
                creation_date=self._clean_str(raw_meta.get("creationDate")),
                modification_date=self._clean_str(raw_meta.get("modDate")),
                page_count=total_pages,
            )

            # Extract text page-by-page
            pages: list[ParsedPage] = []
            warnings: list[str] = []
            pages_with_text = 0
            pages_without_text = 0
            pages_with_errors = 0

            for idx in range(total_pages):
                page_num = idx + 1
                try:
                    page = doc[idx]
                    page_text = page.get_text()
                    extraction_error = None
                except Exception as e:
                    page_text = ""
                    extraction_error = str(e)

                if extraction_error is not None:
                    pages.append(
                        ParsedPage(
                            page_number=page_num,
                            text="",
                            extraction_status="ERROR",
                        )
                    )
                    pages_with_errors += 1
                    warnings.append(f"Page {page_num}: Extraction error: {extraction_error}")
                elif page_text and page_text.strip():
                    pages.append(
                        ParsedPage(
                            page_number=page_num,
                            text=page_text,
                            extraction_status="SUCCESS",
                        )
                    )
                    pages_with_text += 1
                else:
                    pages.append(
                        ParsedPage(
                            page_number=page_num,
                            text="",
                            extraction_status="NO_TEXT",
                        )
                    )
                    pages_without_text += 1
                    warnings.append(f"Page {page_num} contains no extractable text.")

            # If all pages failed extraction due to exceptions
            if pages_with_errors > 0 and pages_with_text == 0:
                raise PDFParsingError(
                    f"Failed to extract text from PDF document: {pages_with_errors} page(s) encountered extraction errors."
                )

            # If all pages succeeded without errors but contained zero extractable text (e.g. scanned)
            if pages_with_text == 0:
                raise PDFNoExtractableTextError(
                    f"No extractable text found across all {total_pages} page(s) of the PDF document."
                )

            diagnostics = ParsingDiagnostics(
                total_pages=total_pages,
                pages_with_text=pages_with_text,
                pages_without_text=pages_without_text,
                pages_with_errors=pages_with_errors,
                warnings=warnings,
            )

            # Extract academic metadata candidates
            academic_meta = self.metadata_extractor.extract(pages, pdf_meta)

            return ParsedResearchPaper(
                pdf_metadata=pdf_meta,
                academic_metadata=academic_meta,
                pages=pages,
                diagnostics=diagnostics,
            )

        finally:
            if doc is not None:
                doc.close()

    @staticmethod
    def _validate_input(pdf_bytes: bytes) -> None:
        """Validate input types and initial PDF signature."""
        if type(pdf_bytes) is not bytes:
            raise PDFParsingError("PDF input must be a bytes instance.")
        if not pdf_bytes:
            raise PDFParsingError("PDF input is empty.")
        if not pdf_bytes.startswith(b"%PDF-"):
            raise PDFParsingError("Invalid PDF header: content must start with b'%PDF-'.")

    @staticmethod
    def _clean_str(val: Any) -> str | None:
        """Sanitize metadata strings to None if empty or whitespace."""
        if isinstance(val, str):
            s = val.strip()
            return s if s else None
        return None
