"""Research paper text preprocessing layer.

This module provides deterministic, rule-based text preprocessing for parsed research papers,
transforming raw page text from PyMuPDF into clean, structured paragraphs while strictly
preserving academic citations, equations, tables, lists, section headings, and references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from app.modules.ingestion.parser import ParsedPage, ParsedResearchPaper


@dataclass
class PreprocessedPage:
    """Represents a single preprocessed research paper page.

    Attributes:
        page_number: 1-indexed page number matching the original document.
        original_text: Exact, unmodified text extracted from the parser.
        cleaned_text: Cleaned and normalized text for this page.
        paragraphs: List of reconstructed paragraph strings for this page.
    """

    page_number: int
    original_text: str
    cleaned_text: str
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class PreprocessingDiagnostics:
    """Diagnostics and quality metrics for the text preprocessing step.

    Attributes:
        total_pages: Number of input pages processed.
        modified_pages: Number of pages whose cleaned text differs from original text.
        removed_repeated_lines: Unique candidate header/footer lines removed across pages.
        warnings: Non-fatal quality warnings encountered during preprocessing.
    """

    total_pages: int
    modified_pages: int
    removed_repeated_lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessedResearchPaper:
    """Preprocessed research paper with page-by-page cleaned content and diagnostics.

    Attributes:
        pages: List of preprocessed pages preserving page numbering.
        full_text: Combined cleaned text across all non-empty pages.
        diagnostics: Quality metrics and diagnostics for the preprocessing run.
    """

    pages: list[PreprocessedPage]
    full_text: str
    diagnostics: PreprocessingDiagnostics


# Prefixes that take a legitimate hyphen when line-wrapped (e.g. self-attention)
HYPHENATED_PREFIXES: frozenset[str] = frozenset({
    "all",
    "co",
    "cross",
    "ex",
    "half",
    "multi",
    "non",
    "pre",
    "pseudo",
    "quasi",
    "re",
    "self",
    "semi",
    "sub",
    "well",
})

# Standard section headings in research papers that must not be removed or merged
SECTION_HEADING_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:"
    r"(?:\d+(?:\.\d+)*\.?\s+)"  # e.g., "1. ", "2.1 ", "1.1.2. "
    r"|(?:[IVXLCDM]+\.?\s+)"    # e.g., "I. ", "II. "
    r"|(?:appendix\s+[A-Z0-9\.\s:]+)"  # e.g., "Appendix A"
    r")?"
    r"(?:abstract|summary|introduction|related\s+work|background|preliminaries|"
    r"methodology|methods|method|model|architecture|experimental\s+setup|"
    r"experiments?|results?|evaluation|discussion|conclusions?|future\s+work|"
    r"references|bibliography|acknowledg(?:e)?ments?)\b",
    re.IGNORECASE,
)

# Numbered or bulleted list item indicators
LIST_ITEM_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:[\*\-\+•◦▪]|\(?\d+[\.\)]|\(?[a-zA-Z][\.\)])\s+",
)

# Reference entry prefix (e.g., "[1] Smith, J...", "[42] Doe, A...")
REFERENCE_ENTRY_PATTERN: re.Pattern[str] = re.compile(
    r"^\[\d+\]\s+[A-Z]",
)


class ResearchTextPreprocessor:
    """Deterministic, rule-based text preprocessor for parsed research papers.

    Processes parsed paper pages through a 5-stage pipeline:
        Stage A: Normalize line endings (CRLF/CR -> LF).
        Stage B: Detect and remove repeated headers and footers conservatively.
        Stage C: Correct line-break hyphenation while preserving compounds.
        Stage D: Normalize whitespace and collapse excessive blank lines.
        Stage E: Reconstruct paragraphs from PDF line wraps.
    """

    def __init__(self, page_separator: str = "\n\n\n") -> None:
        """Initialize preprocessor with configurable page separator."""
        self.page_separator = page_separator

    def preprocess(self, parsed_paper: ParsedResearchPaper) -> PreprocessedResearchPaper:
        """Preprocess a parsed research paper and return cleaned pages and diagnostics.

        Args:
            parsed_paper: The parsed research paper representation.

        Returns:
            PreprocessedResearchPaper containing cleaned pages, full text, and diagnostics.

        Raises:
            TypeError: If parsed_paper is not an instance of ParsedResearchPaper.
        """
        if not isinstance(parsed_paper, ParsedResearchPaper):
            raise TypeError(
                f"Expected ParsedResearchPaper instance, got {type(parsed_paper).__name__}"
            )

        if not parsed_paper.pages:
            return PreprocessedResearchPaper(
                pages=[],
                full_text="",
                diagnostics=PreprocessingDiagnostics(
                    total_pages=0,
                    modified_pages=0,
                    removed_repeated_lines=[],
                    warnings=["Parsed research paper contains zero pages."],
                ),
            )

        warnings: list[str] = []

        # Stage A: Normalize line endings per page
        stage_a_texts: list[str] = []
        for p in parsed_paper.pages:
            orig = p.text
            if not orig.strip():
                warnings.append(f"Page {p.page_number} contains no text.")
            stage_a_texts.append(self._normalize_line_endings(orig))

        # Stage B: Detect and remove repeated headers and footers across pages
        stage_b_texts, removed_repeated_lines = self._remove_repeated_headers_footers(stage_a_texts)

        # Stages C, D, E: Per-page line dehyphenation, whitespace normalization, and paragraph reconstruction
        preprocessed_pages: list[PreprocessedPage] = []
        modified_pages_count = 0

        for idx, p in enumerate(parsed_paper.pages):
            original_text = p.text
            raw_page_text = stage_b_texts[idx]

            if not raw_page_text.strip():
                # Preserve genuinely empty pages
                preprocessed_pages.append(
                    PreprocessedPage(
                        page_number=p.page_number,
                        original_text=original_text,
                        cleaned_text="",
                        paragraphs=[],
                    )
                )
                if original_text != "":
                    modified_pages_count += 1
                continue

            # Stage C: Correct line-break hyphenation
            dehyphenated_text = self._dehyphenate_text(raw_page_text)

            # Stage D: Normalize whitespace
            whitespace_normalized = self._normalize_whitespace(dehyphenated_text)

            # Stage E: Reconstruct paragraphs
            cleaned_text, paragraphs = self._reconstruct_paragraphs(whitespace_normalized)

            if cleaned_text != original_text:
                modified_pages_count += 1

            preprocessed_pages.append(
                PreprocessedPage(
                    page_number=p.page_number,
                    original_text=original_text,
                    cleaned_text=cleaned_text,
                    paragraphs=paragraphs,
                )
            )

        # Build full_text joining non-empty cleaned page texts in page order
        non_empty_page_texts = [p.cleaned_text for p in preprocessed_pages if p.cleaned_text.strip()]
        full_text = self.page_separator.join(non_empty_page_texts)

        diagnostics = PreprocessingDiagnostics(
            total_pages=len(parsed_paper.pages),
            modified_pages=modified_pages_count,
            removed_repeated_lines=removed_repeated_lines,
            warnings=warnings,
        )

        return PreprocessedResearchPaper(
            pages=preprocessed_pages,
            full_text=full_text,
            diagnostics=diagnostics,
        )

    # -------------------------------------------------------------------------
    # Stage A: Line Endings
    # -------------------------------------------------------------------------

    def _normalize_line_endings(self, text: str) -> str:
        """Convert CRLF and CR to standard LF newlines."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    # -------------------------------------------------------------------------
    # Stage B: Repeated Headers & Footers
    # -------------------------------------------------------------------------

    def _remove_repeated_headers_footers(
        self,
        page_texts: list[str],
    ) -> tuple[list[str], list[str]]:
        """Identify and remove repeated header and footer lines across pages.

        Only removes candidates that:
        - Appear on multiple pages (>= 2).
        - Are not section headings, citations, equations, or tables.
        - Would not leave the page completely devoid of content.

        Preserves the original line structure including all blank lines.
        """
        if len(page_texts) < 2:
            return list(page_texts), []

        first_lines_by_page: list[str | None] = []
        last_lines_by_page: list[str | None] = []

        for text in page_texts:
            raw_lines = text.split("\n")
            first_idx = next((i for i, ln in enumerate(raw_lines) if ln.strip()), None)
            last_idx = next((i for i in range(len(raw_lines) - 1, -1, -1) if raw_lines[i].strip()), None)
            first_lines_by_page.append(raw_lines[first_idx].strip() if first_idx is not None else None)
            last_lines_by_page.append(raw_lines[last_idx].strip() if last_idx is not None else None)

        # Count occurrences of normalized header and footer candidate lines
        header_counts: dict[str, int] = {}
        header_originals: dict[str, str] = {}
        footer_counts: dict[str, int] = {}
        footer_originals: dict[str, str] = {}

        for fl in first_lines_by_page:
            if fl and self._is_candidate_header_or_footer(fl):
                norm = " ".join(fl.split())
                header_counts[norm] = header_counts.get(norm, 0) + 1
                if norm not in header_originals:
                    header_originals[norm] = fl

        for ll in last_lines_by_page:
            if ll and self._is_candidate_header_or_footer(ll):
                norm = " ".join(ll.split())
                footer_counts[norm] = footer_counts.get(norm, 0) + 1
                if norm not in footer_originals:
                    footer_originals[norm] = ll

        # Verified repeated headers and footers (appearing on at least 2 pages)
        valid_headers = {norm for norm, count in header_counts.items() if count >= 2}
        valid_footers = {norm for norm, count in footer_counts.items() if count >= 2}

        removed_lines_set: set[str] = set()
        removed_lines_list: list[str] = []

        cleaned_texts: list[str] = []

        for idx, text in enumerate(page_texts):
            raw_lines = list(text.split("\n"))
            non_empty_indices = [i for i, ln in enumerate(raw_lines) if ln.strip()]
            if not non_empty_indices:
                cleaned_texts.append(text)
                continue

            # Check header removal
            if non_empty_indices:
                first_idx = non_empty_indices[0]
                first_norm = " ".join(raw_lines[first_idx].split())
                if first_norm in valid_headers:
                    # Guard: do not remove if doing so leaves page with no content
                    if len(non_empty_indices) > 1:
                        orig = raw_lines[first_idx].strip()
                        if orig not in removed_lines_set:
                            removed_lines_set.add(orig)
                            removed_lines_list.append(orig)
                        raw_lines.pop(first_idx)
                        non_empty_indices = [i for i, ln in enumerate(raw_lines) if ln.strip()]

            # Check footer removal
            if non_empty_indices:
                last_idx = non_empty_indices[-1]
                last_norm = " ".join(raw_lines[last_idx].split())
                if last_norm in valid_footers:
                    # Guard: do not remove if doing so leaves page with no content
                    if len(non_empty_indices) > 1:
                        orig = raw_lines[last_idx].strip()
                        if orig not in removed_lines_set:
                            removed_lines_set.add(orig)
                            removed_lines_list.append(orig)
                        raw_lines.pop(last_idx)

            cleaned_texts.append("\n".join(raw_lines))

        return cleaned_texts, removed_lines_list

    def _is_candidate_header_or_footer(self, line: str) -> bool:
        """Check if a line can safely be considered a candidate header or footer."""
        cleaned = line.strip()
        if not cleaned:
            return False

        # Never treat section headings as headers/footers
        if self._is_section_heading(cleaned):
            return False

        # Never treat reference entries or citations as headers/footers
        if REFERENCE_ENTRY_PATTERN.match(cleaned):
            return False

        # Never treat equations or tables as headers/footers
        if self._is_equation_or_math(cleaned) or self._is_table_or_structure(cleaned):
            return False

        # Too long to be a standard header/footer
        if len(cleaned) > 220:
            return False

        return True

    # -------------------------------------------------------------------------
    # Stage C: Correct Line-Break Hyphenation
    # -------------------------------------------------------------------------

    def _dehyphenate_text(self, text: str) -> str:
        """Correct line-break hyphenation within page text.

        Only joins words when a hyphen appears at the end of a line and the next
        line begins with a plausible lowercase word continuation. Preserves legitimate
        hyphenated compounds on the same line and avoids joining across headings,
        list items, or citations.
        """
        lines = text.split("\n")
        if len(lines) < 2:
            return text

        result_lines: list[str] = []
        i = 0

        while i < len(lines):
            curr_line = lines[i]

            # Check if current line ends with a word followed by hyphen
            m = re.search(r"(\b[a-zA-Z]{2,})-$", curr_line)
            if not m or i + 1 >= len(lines):
                result_lines.append(curr_line)
                i += 1
                continue

            next_line = lines[i + 1].strip()
            # If next line is empty, it's a paragraph break; do not join across paragraphs
            if not next_line:
                result_lines.append(curr_line)
                i += 1
                continue

            # Plausible continuation: begins with lowercase alphabetic characters
            next_m = re.match(r"^([a-z]{2,})\b(.*)$", next_line)
            if not next_m:
                result_lines.append(curr_line)
                i += 1
                continue

            # Do not join if trailing hyphen is part of a URL, DOI, or email address
            if self._is_url_or_email_hyphen(curr_line, next_line):
                result_lines.append(curr_line)
                i += 1
                continue

            # Do not join if next line is a list item, heading, equation, or table
            if (
                self._is_list_item(next_line)
                or self._is_section_heading(next_line)
                or self._is_equation_or_math(next_line)
                or self._is_table_or_structure(next_line)
            ):
                result_lines.append(curr_line)
                i += 1
                continue

            word1 = m.group(1)
            word2 = next_m.group(1)
            rest_of_next_line = next_m.group(2).strip()

            # Preserve legitimate hyphenated compounds if word1 is a recognized prefix
            if word1.lower() in HYPHENATED_PREFIXES:
                joined_word = f"{word1}-{word2}"
            else:
                joined_word = f"{word1}{word2}"

            # Replace trailing "word1-" on curr_line with joined_word
            line_prefix = curr_line[:-len(m.group(0))]
            new_curr_line = f"{line_prefix}{joined_word}"

            result_lines.append(new_curr_line)

            # Advance next line: if it had subsequent words, keep them; else drop empty continuation line
            if rest_of_next_line:
                lines[i + 1] = rest_of_next_line
                i += 1
            else:
                i += 2

        return "\n".join(result_lines)

    def _is_url_or_email_hyphen(self, curr_line: str, next_line: str) -> bool:
        """Check if trailing hyphen is part of a URL, bare domain URL, DOI, or email address."""
        tokens = curr_line.split()
        if not tokens:
            return False
        trailing_token = tokens[-1].lower()

        # Check for URL schemes and domains in trailing token
        if "://" in trailing_token or "doi.org" in trailing_token:
            return True
        if trailing_token.startswith(("http:", "https:", "www.", "ftp:", "doi:", "mailto:")):
            return True

        # Check for email pattern in trailing token
        if "@" in trailing_token:
            return True

        # Check for bare or prefixed DOI pattern: 10.<registrant>/<suffix>-
        if re.search(r"\b10\.\d{4,9}/[^\s]*-$", curr_line):
            return True

        # Check for bare domain-based URLs (e.g., example.com/api-, sub.domain.org/path-)
        if re.search(
            r"\b[a-zA-Z0-9\.\-]+\.(?:com|org|edu|gov|net|io|ai|co|info|dev|app)(?:/[^\s]*)?-$",
            curr_line,
            re.IGNORECASE,
        ):
            return True

        # Regex check for URL/email pattern leading up to the hyphen at end of line
        if re.search(r"(?:https?://|www\.|[\w\.-]+@)[^\s]*-$", curr_line, re.IGNORECASE):
            return True

        # Check if continuation line starts with a URL path or top-level domain
        if re.match(r"^(?:/|\.com|\.org|\.edu|\.gov|\.net|\.io|\.ai)\b", next_line.strip(), re.IGNORECASE):
            return True

        return False

    # -------------------------------------------------------------------------
    # Stage D: Normalize Whitespace
    # -------------------------------------------------------------------------

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse redundant horizontal spaces and reduce excessive blank lines."""
        # 1. Collapse multiple spaces and tabs to a single space
        text = re.sub(r"[ \t]+", " ", text)

        # 2. Strip leading/trailing space from individual lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 3. Collapse 3 or more consecutive newlines into 2 (standard paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # -------------------------------------------------------------------------
    # Stage E: Reconstruct Paragraphs
    # -------------------------------------------------------------------------

    def _reconstruct_paragraphs(self, text: str) -> tuple[str, list[str]]:
        """Reconstruct running paragraphs from PDF line-wrapping.

        Preserves blank-line paragraph boundaries, section headings, list items,
        standalone equations, and table rows without merging them into running text.

        Returns:
            Tuple of (cleaned_text, list_of_paragraphs).
        """
        if not text:
            return "", []

        # Split into coarse blocks separated by blank lines
        raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        reconstructed_blocks: list[str] = []

        for block in raw_blocks:
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            if not lines:
                continue

            current_block_lines: list[str] = []

            for line in lines:
                if not current_block_lines:
                    current_block_lines.append(line)
                    continue

                prev_line = current_block_lines[-1]

                # Check if safe to join line with prev_line
                if self._can_join_lines(prev_line, line):
                    current_block_lines[-1] = f"{prev_line} {line}"
                else:
                    current_block_lines.append(line)

            # Partition lines in this block so headings and standalone equations remain distinct paragraphs
            sub_paragraphs: list[str] = []
            cur_para: list[str] = []

            for ln in current_block_lines:
                if self._is_isolated_element(ln):
                    if cur_para:
                        sub_paragraphs.append("\n".join(cur_para))
                        cur_para = []
                    sub_paragraphs.append(ln)
                else:
                    cur_para.append(ln)

            if cur_para:
                sub_paragraphs.append("\n".join(cur_para))

            reconstructed_blocks.extend(sub_paragraphs)

        cleaned_text = "\n\n".join(reconstructed_blocks)
        return cleaned_text, reconstructed_blocks

    def _can_join_lines(self, prev_line: str, curr_line: str) -> bool:
        """Determine whether curr_line can be safely folded into prev_line."""
        if not prev_line or not curr_line:
            return False

        # Do not merge if either is a section heading
        if self._is_section_heading(prev_line) or self._is_section_heading(curr_line):
            return False

        # Do not merge into list items or reference entries
        if self._is_list_item(curr_line) or bool(REFERENCE_ENTRY_PATTERN.match(curr_line)):
            return False

        # Do not merge if prev_line ends with a colon introducing a list/equation
        if prev_line.endswith(":"):
            return False

        # Do not merge standalone equations or table structures
        if (
            self._is_equation_or_math(prev_line)
            or self._is_equation_or_math(curr_line)
            or self._is_table_or_structure(prev_line)
            or self._is_table_or_structure(curr_line)
        ):
            return False

        # If prev_line is a list item:
        if self._is_list_item(prev_line):
            # Only join if there is reliable evidence that curr_line is a wrapped continuation:
            # - prev_line must not end with sentence/item-ending punctuation
            # - curr_line must begin with lowercase letter
            if prev_line.endswith((".", "!", "?", ":", ";")):
                return False
            if not re.match(r"^[a-z]", curr_line):
                return False

        return True

    def _is_isolated_element(self, line: str) -> bool:
        """Check if a line should stand alone as its own paragraph (e.g. heading, list item, equation)."""
        return (
            self._is_section_heading(line)
            or self._is_list_item(line)
            or bool(REFERENCE_ENTRY_PATTERN.match(line))
            or self._is_equation_or_math(line)
            or self._is_table_or_structure(line)
        )

    # -------------------------------------------------------------------------
    # Element Classification Helpers
    # -------------------------------------------------------------------------

    def _is_section_heading(self, line: str) -> bool:
        """Check if line is a standard or numbered academic section heading."""
        cleaned = line.strip()
        if not cleaned:
            return False

        # Standard named or numbered academic headings (e.g., "1. Introduction", "References")
        if SECTION_HEADING_PATTERN.match(cleaned):
            return True

        # Numbered headings (e.g., "1.2 Model Architecture", "3 Results and Analysis")
        # Ensure not ending with terminal period/punctuation which indicates a numbered list/sentence
        if re.match(r"^\d+(?:\.\d+)+\s+[A-Z][a-zA-Z\s]{1,50}$", cleaned):
            return True
        if re.match(r"^\d+\.?\s+[A-Z][a-zA-Z\s]{1,50}$", cleaned) and not cleaned.endswith((".", ";", ":", ",")):
            return True

        # Roman numeral headings (e.g., "II. RELATED WORK")
        if re.match(r"^[IVXLCDM]+\.\s+[A-Z][a-zA-Z\s]{1,50}$", cleaned) and not cleaned.endswith((".", ";", ":", ",")):
            return True

        return False

    def _is_list_item(self, line: str) -> bool:
        """Check if line begins with a bullet or numbered list marker."""
        cleaned = line.strip()
        if not cleaned or self._is_section_heading(cleaned):
            return False
        return bool(LIST_ITEM_PATTERN.match(cleaned))

    def _is_equation_or_math(self, line: str) -> bool:
        """Check if line represents a standalone equation or prominent mathematical expression."""
        cleaned = line.strip()
        if not cleaned:
            return False

        # Display math delimiters
        if cleaned.startswith("$$") and cleaned.endswith("$$") and len(cleaned) >= 4:
            return True
        if cleaned.startswith(r"\[") and cleaned.endswith(r"\]"):
            return True
        if re.match(r"^\\begin\{(?:equation|align|gather|multline)\*?\}", cleaned):
            return True
        if re.search(r"\\end\{(?:equation|align|gather|multline)\*?\}$", cleaned):
            return True

        # LaTeX mathematical command symbols
        if any(cmd in cleaned for cmd in (r"\sum", r"\int", r"\prod", r"\frac", r"\sqrt", r"\mathbb", r"\mathbf")):
            return True

        # Standalone equation with equality or relation operator (e.g., "E = mc^2", "L(θ) = - \log P(x)")
        if re.search(r"[a-zA-Z\(\)\}\]]\s*(?:=|\\approx|\\le|\\ge|\\leq|\\geq)\s*[-+0-9a-zA-Z\\\{]", cleaned):
            # Guard against regular prose containing "=" e.g. "setting temperature = 0.7"
            if len(cleaned) < 120 and (cleaned.endswith((".", ",", ";")) or not cleaned.endswith((".", "!", "?"))):
                if any(op in cleaned for op in ("^", "_", "\\", "+", "*", "/", "θ", "α", "β", "γ", "λ", "σ")):
                    return True

        return False

    def _is_table_or_structure(self, line: str) -> bool:
        """Check if line is a table row or structured table separator."""
        cleaned = line.strip()
        if not cleaned:
            return False

        # Markdown / pipe table row
        if cleaned.count("|") >= 2:
            return True

        # Table rule separator
        if re.match(r"^[\s\|\+\-\:]{3,}$", cleaned):
            return True

        return False
