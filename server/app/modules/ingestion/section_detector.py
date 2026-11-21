"""Research paper section detection and hierarchy construction layer.

This module provides deterministic, rule-based section detection for preprocessed research papers,
identifying section headings, classifying them into standard research categories, determining
heading hierarchy (parent-child relationships), and associating body paragraphs while preserving
page numbers, document order, and unclassified domain-specific headings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from app.modules.ingestion.preprocessor import PreprocessedPage, PreprocessedResearchPaper


class SectionType(str, Enum):
    """Standard academic research paper section categories."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    LIMITATIONS = "limitations"
    FUTURE_WORK = "future_work"
    ACKNOWLEDGMENTS = "acknowledgments"
    REFERENCES = "references"
    APPENDIX = "appendix"
    CUSTOM = "custom"


@dataclass(frozen=True)
class SectionParagraph:
    """An individual body paragraph belonging to a section or document preamble.

    Attributes:
        text: Original paragraph text.
        page_number: Source page number (1-indexed).
        paragraph_index: Global 0-indexed position in the flattened document.
    """

    text: str
    page_number: int
    paragraph_index: int


@dataclass
class DetectedSection:
    """A detected research paper section with title, hierarchy, and assigned paragraphs.

    Attributes:
        section_id: Unique identifier for this section (e.g. 'sec_1').
        section_index: 0-indexed order of this section in the detected sections list.
        heading_paragraph_index: 0-indexed global position of the heading in the document.
        title: Exact original heading text (unstripped).
        normalized_title: Normalized heading string for classification and comparison.
        section_type: Categorized section type.
        level: Structural hierarchy level (1 for top-level, 2 for subsection, etc.).
        parent_section_id: Section ID of the enclosing parent section, or None for top-level.
        heading_page: Page number where the heading appears.
        start_page: First page of this section's content (including subsections).
        end_page: Last page of this section's content (including subsections).
        paragraphs: List of direct body paragraphs belonging to this section.
        confidence: Normalized detection confidence score (0.0 to 1.0).
        metadata: Additional structural metadata (e.g. explicit numbering, child count).
    """

    section_id: str
    section_index: int
    heading_paragraph_index: int

    title: str
    normalized_title: str
    section_type: SectionType

    level: int
    parent_section_id: str | None

    heading_page: int
    start_page: int
    end_page: int

    paragraphs: list[SectionParagraph] = field(default_factory=list)

    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionDetectionDiagnostics:
    """Diagnostics and quality metrics for the section detection run.

    Attributes:
        total_paragraphs: Total count of non-empty input paragraphs evaluated across all pages.
        detected_headings: Total number of headings detected.
        rejected_candidates: Paragraphs evaluated as candidate headings but rejected.
        unclassified_headings: Number of detected headings categorized as CUSTOM.
        sections_created: Total DetectedSection instances created.
        warnings: Quality and structural anomaly warnings encountered.
    """

    total_paragraphs: int
    detected_headings: int
    rejected_candidates: int
    unclassified_headings: int
    sections_created: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class SectionDetectionResult:
    """Complete section detection result containing detected sections and preamble.

    Attributes:
        sections: Ordered list of detected research sections.
        preamble: Non-heading paragraphs occurring before the first section heading.
        diagnostics: Quality metrics and structural warnings.
    """

    sections: list[DetectedSection]
    preamble: list[SectionParagraph]
    diagnostics: SectionDetectionDiagnostics


# Detection Constants and Thresholds

MIN_HEADING_CONFIDENCE: float = 0.50
MAX_HEADING_CHARS: int = 140
MAX_HEADING_WORDS: int = 15

# Positive signal weights
WEIGHT_CANONICAL_ALIAS: float = 0.40
WEIGHT_HEADING_KEYWORD: float = 0.20
WEIGHT_HIERARCHICAL_NUMBERING: float = 0.30
WEIGHT_NUMBERED_HEADING: float = 0.25
WEIGHT_NUMBERED_GENERIC: float = 0.10
WEIGHT_ALL_CAPS: float = 0.25
WEIGHT_TITLE_CASE: float = 0.20
WEIGHT_SENTENCE_CASE_HEADING: float = 0.15  # For concise, verb-free sentence-case heading phrases
WEIGHT_BREVITY_SHORT: float = 0.10   # <= 40 chars
WEIGHT_BREVITY_MEDIUM: float = 0.05  # <= 80 chars

# Negative signal penalties
PENALTY_SENTENCE_PERIOD: float = 0.35
PENALTY_INLINE_CITATION: float = 0.45
PENALTY_URL_OR_DOI: float = 0.50
PENALTY_BULLET_MARKER: float = 0.50
PENALTY_PROSE_PHRASE: float = 0.50
PENALTY_LOWERCASE_START: float = 0.50
PENALTY_MATH_SYNTAX: float = 0.50
PENALTY_SEQUENTIAL_LIST: float = 0.45
PENALTY_SENTENCE_CASE_NUMBERED: float = 0.35  # For sentence-like prose or verb-bearing clauses in custom items

# Standard section aliases mapped to SectionType categories
SECTION_CATEGORY_ALIASES: dict[SectionType, tuple[str, ...]] = {
    SectionType.ABSTRACT: (
        "abstract",
        "summary",
        "executive summary",
    ),
    SectionType.INTRODUCTION: (
        "introduction",
        "overview",
        "introductory remarks",
    ),
    SectionType.RELATED_WORK: (
        "related work",
        "related works",
        "prior work",
        "prior works",
        "literature review",
        "related literature",
    ),
    SectionType.BACKGROUND: (
        "background",
        "preliminaries",
        "background and preliminaries",
        "theoretical background",
        "problem formulation",
        "problem setup",
    ),
    SectionType.METHODOLOGY: (
        "methodology",
        "method",
        "methods",
        "proposed method",
        "proposed methodology",
        "proposed approach",
        "our approach",
        "model",
        "model architecture",
        "system architecture",
        "framework",
        "technical approach",
    ),
    SectionType.EXPERIMENTS: (
        "experiments",
        "experiment",
        "experimental setup",
        "experimental design",
        "empirical evaluation",
        "experimental evaluation",
        "experiments and results",
    ),
    SectionType.RESULTS: (
        "results",
        "findings",
        "empirical results",
        "experimental results",
        "main results",
    ),
    SectionType.DISCUSSION: (
        "discussion",
        "discussion and analysis",
        "analysis",
        "ablation study",
        "ablation studies",
    ),
    SectionType.CONCLUSION: (
        "conclusion",
        "conclusions",
        "concluding remarks",
        "conclusion and future work",
        "conclusions and future work",
        "summary and conclusion",
    ),
    SectionType.LIMITATIONS: (
        "limitations",
        "limitation",
        "limitations and ethical considerations",
        "ethical considerations",
        "broader impact",
        "broader impacts",
    ),
    SectionType.FUTURE_WORK: (
        "future work",
        "future works",
        "future directions",
        "future research",
    ),
    SectionType.ACKNOWLEDGMENTS: (
        "acknowledgments",
        "acknowledgements",
        "acknowledgment",
        "acknowledgement",
    ),
    SectionType.REFERENCES: (
        "references",
        "reference",
        "bibliography",
        "works cited",
    ),
    SectionType.APPENDIX: (
        "appendix",
        "appendices",
        "supplementary material",
        "supplementary materials",
        "supplement",
    ),
}

# Regex for common prose phrases that disqualify a paragraph from being a heading
PROSE_INDICATOR_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:in this (?:paper|work|section|study)|we (?:propose|present|show|demonstrate|introduce|evaluate|describe|find))\b",
    re.IGNORECASE,
)

# Regex for finite verbs and predicate indicators that characterize clauses and sentences
VERB_PREDICATE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:"
    r"is|are|was|were|have|has|had|can|could|will|would|should|must|may|might|"
    r"requires|involves|removes|converges|shows|demonstrates|presents|provides|"
    r"includes|contains|achieves|uses|indicates|suggests|generalizes|consists|"
    r"evaluates|illustrates|corresponds|represents|applies|allows|improves|"
    r"yields|outperforms|leads|depends|focuses|serves|differs|enables|"
    r"it|they|we"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _FlattenedParagraph:
    """Internal flattened paragraph representation with page and global ordering."""

    text: str
    page_number: int
    global_index: int


@dataclass
class _HeadingCandidate:
    """Internal representation of a scored heading candidate."""

    paragraph: _FlattenedParagraph
    title: str
    normalized_title: str
    section_type: SectionType
    level: int
    numbering: str | None
    confidence: float


class ResearchSectionDetector:
    """Deterministic, rule-based section detector for preprocessed research papers.

    Evaluates paragraphs across multiple positive and negative structural signals,
    determines hierarchical relationships using a section stack, associates body
    paragraphs, and computes comprehensive multi-page section spans.
    """

    def detect(self, paper: PreprocessedResearchPaper) -> SectionDetectionResult:
        """Detect and organize sections from a preprocessed research paper.

        Args:
            paper: Input preprocessed paper containing page-by-page paragraphs.

        Returns:
            SectionDetectionResult with detected sections, preamble, and diagnostics.

        Raises:
            TypeError: If paper is not an instance of PreprocessedResearchPaper.
        """
        if not isinstance(paper, PreprocessedResearchPaper):
            raise TypeError(
                f"Expected PreprocessedResearchPaper instance, got {type(paper).__name__}"
            )

        flattened = self._flatten_paragraphs(paper.pages)
        if not flattened:
            return SectionDetectionResult(
                sections=[],
                preamble=[],
                diagnostics=SectionDetectionDiagnostics(
                    total_paragraphs=0,
                    detected_headings=0,
                    rejected_candidates=0,
                    unclassified_headings=0,
                    sections_created=0,
                    warnings=["Document contains zero paragraphs."],
                ),
            )

        warnings: list[str] = []
        rejected_candidates_count = 0

        sequential_list_indices = self._identify_sequential_list_indices(flattened)

        heading_map: dict[int, _HeadingCandidate] = {}

        for p in flattened:
            candidate, is_rejected = self._evaluate_paragraph(p, sequential_list_indices)
            if candidate is not None:
                heading_map[p.global_index] = candidate
            elif is_rejected:
                rejected_candidates_count += 1

        if not heading_map:
            warnings.append("No section headings were detected in the document.")

        sections, unclassified_count = self._build_sections_and_hierarchy(heading_map, warnings)

        preamble = self._assign_paragraphs(flattened, heading_map, sections)

        self._calculate_section_spans(sections)

        for s in sections:
            if not s.paragraphs:
                warnings.append(f"Section '{s.title}' on page {s.heading_page} has no body paragraphs.")

        diagnostics = SectionDetectionDiagnostics(
            total_paragraphs=len(flattened),
            detected_headings=len(heading_map),
            rejected_candidates=rejected_candidates_count,
            unclassified_headings=unclassified_count,
            sections_created=len(sections),
            warnings=warnings,
        )

        return SectionDetectionResult(
            sections=sections,
            preamble=preamble,
            diagnostics=diagnostics,
        )

    def _flatten_paragraphs(self, pages: list[PreprocessedPage]) -> list[_FlattenedParagraph]:
        """Flatten page paragraphs into an ordered sequence with global indices.

        Preserves original input paragraph text verbatim without stripping.
        Paragraphs that are completely empty or contain only whitespace are
        ignored defensively, consistent with the preprocessor output contract.
        """
        flattened: list[_FlattenedParagraph] = []
        global_idx = 0

        for page in pages:
            for para_text in page.paragraphs:
                if not para_text or not para_text.strip():
                    continue
                flattened.append(
                    _FlattenedParagraph(
                        text=para_text,
                        page_number=page.page_number,
                        global_index=global_idx,
                    )
                )
                global_idx += 1

        return flattened

    def _identify_sequential_list_indices(
        self,
        flattened: list[_FlattenedParagraph],
    ) -> set[int]:
        """Identify global paragraph indices that belong to sequential numbered lists.

        Detects sequences of adjacent paragraphs that follow list numbering patterns
        (e.g., 1. ..., 2. ..., 3. ...), distinguishing them from legitimate adjacent
        heading-only sections using contextual and structural indicators.
        """
        list_indices: set[int] = set()
        n_paras = len(flattened)
        if n_paras < 2:
            return list_indices

        # Extract decimal integers for each paragraph if it starts with a simple integer marker
        # Matches "1. ", "2. ", "1) ", "2) " but not "1.1 " or "Section 1:"
        int_items: dict[int, int] = {}
        for idx, p in enumerate(flattened):
            cleaned = p.text.strip()
            m = re.match(r"^(\d{1,3})[\.\)]\s+", cleaned)
            if m and not re.match(r"^\d+\.\d+", cleaned):
                int_items[idx] = int(m.group(1))

        i = 0
        while i < n_paras - 1:
            if i in int_items:
                seq = [i]
                curr_val = int_items[i]
                curr_idx = i
                while curr_idx + 1 < n_paras and curr_idx + 1 in int_items:
                    next_val = int_items[curr_idx + 1]
                    if next_val == curr_val + 1:
                        seq.append(curr_idx + 1)
                        curr_val = next_val
                        curr_idx += 1
                    else:
                        break

                if len(seq) >= 2:
                    if self._is_numbered_list_sequence(flattened, seq):
                        for s_idx in seq:
                            p_text = flattened[s_idx].text.strip()
                            norm = self._normalize_title(p_text)
                            cat = self._classify_title(norm)
                            # Canonical sections like "1. Introduction", "2. Related Work" are preserved
                            if cat == SectionType.CUSTOM:
                                list_indices.add(s_idx)
                    i = curr_idx + 1
                    continue
            i += 1

        return list_indices

    def _is_numbered_list_sequence(
        self,
        flattened: list[_FlattenedParagraph],
        seq: list[int],
    ) -> bool:
        """Determine whether an adjacent sequence of sequentially numbered paragraphs is a list."""
        # 1. If any item is a canonical section heading, it's not a list
        for idx in seq:
            p_text = flattened[idx].text.strip()
            norm = self._normalize_title(p_text)
            if self._classify_title(norm) != SectionType.CUSTOM:
                return False

        first_idx = seq[0]

        # 2. Check preceding context for list indicators (e.g. colon or introductory wording)
        if first_idx > 0:
            prev_text = flattened[first_idx - 1].text.strip()
            if prev_text.endswith(":") or re.search(
                r"\b(?:following|components|requirements|challenges|stages|steps|reasons|goals|tasks|findings|aspects|criteria|properties)\b",
                prev_text,
                re.IGNORECASE,
            ):
                return True

        # 3. Check if any item in the sequence contains finite verbs or sentence punctuation
        for idx in seq:
            text = flattened[idx].text.strip()
            if self._ends_with_sentence_period(text):
                return True
            if VERB_PREDICATE_PATTERN.search(text):
                return True
            if not self._is_plausible_heading_phrase(text):
                return True

        # 4. Relevant local numbering restart check:
        # A sequence starting with 1. is a list restart ONLY if supported by relevant local structural context:
        # Scan backward to find the immediately enclosing heading. If the most recent heading was numbered 1,
        # then restarting at 1 within that section indicates an embedded list.
        first_text = flattened[first_idx].text.strip()
        if re.match(r"^1[\.\)]\s+", first_text) and first_idx > 0:
            for p_idx in range(first_idx - 1, -1, -1):
                past_text = flattened[p_idx].text.strip()
                norm_past = self._normalize_title(past_text)
                is_canon = self._classify_title(norm_past) != SectionType.CUSTOM
                m_past_num = re.match(
                    r"^(?:section\s+)?(\d+(?:\.\d+)*|[ivxlcdm]+)[\.:\s-]",
                    past_text,
                    re.IGNORECASE,
                )
                if is_canon or m_past_num:
                    if m_past_num and m_past_num.group(1) == "1":
                        return True
                    break

        # 5. If all items in seq are concise, plausible heading noun phrases without verbs,
        # sentence periods, or list introductory context, it is a sequence of headings, NOT a list.
        return False

    def _evaluate_paragraph(
        self,
        paragraph: _FlattenedParagraph,
        sequential_list_indices: set[int] | None = None,
    ) -> tuple[_HeadingCandidate | None, bool]:
        """Evaluate a paragraph against positive and negative heading signals.

        Returns:
            Tuple of (HeadingCandidate or None, was_candidate_rejected: bool).
        """
        eval_text = paragraph.text.strip()
        words = eval_text.split()

        # Hard disqualifications
        if len(eval_text) > MAX_HEADING_CHARS or len(words) > MAX_HEADING_WORDS:
            return None, False

        score = 0.0
        had_positive_signal = False

        norm_title = self._normalize_title(eval_text)
        section_type = self._classify_title(norm_title)
        num_str, level = self._extract_numbering_and_level(eval_text)

        is_title = self._is_title_case(eval_text)
        is_all_caps = eval_text.isupper() and len(eval_text) >= 3
        is_sentence = self._is_sentence_case(eval_text)
        has_keyword = self._contains_heading_keyword(norm_title)
        is_plausible_heading = self._is_plausible_heading_phrase(eval_text)

        # 1. Numbering signals
        if num_str is not None:
            # Hierarchical numbering (e.g. 1.1, 1.2.3, A.1, Section 1:)
            if (level is not None and level >= 2) or re.match(r"^section\s+\d+", eval_text, re.IGNORECASE):
                score += WEIGHT_HIERARCHICAL_NUMBERING
                had_positive_signal = True
            elif is_title or is_all_caps or section_type != SectionType.CUSTOM or has_keyword or is_plausible_heading:
                score += WEIGHT_NUMBERED_HEADING
                had_positive_signal = True
            else:
                score += WEIGHT_NUMBERED_GENERIC
                had_positive_signal = True

        # 2. Vocabulary signals
        if section_type != SectionType.CUSTOM:
            score += WEIGHT_CANONICAL_ALIAS
            had_positive_signal = True
        elif has_keyword:
            score += WEIGHT_HEADING_KEYWORD
            had_positive_signal = True

        # 3. Capitalization & typographical signals
        if is_all_caps:
            score += WEIGHT_ALL_CAPS
            had_positive_signal = True
        elif is_title:
            score += WEIGHT_TITLE_CASE
            had_positive_signal = True
        elif is_plausible_heading:
            score += WEIGHT_SENTENCE_CASE_HEADING
            had_positive_signal = True

        # 4. Brevity signals
        if len(eval_text) <= 40:
            score += WEIGHT_BREVITY_SHORT
            had_positive_signal = True
        elif len(eval_text) <= 80:
            score += WEIGHT_BREVITY_MEDIUM
            had_positive_signal = True

        # Negative signals / penalties
        # Sequential list penalty
        if sequential_list_indices and paragraph.global_index in sequential_list_indices:
            score -= PENALTY_SEQUENTIAL_LIST

        # Sentence-like prose penalty (finite verbs, clause structure, or non-heading phrasing in custom numbered items)
        if num_str is not None and (not is_plausible_heading or VERB_PREDICATE_PATTERN.search(eval_text)) and section_type == SectionType.CUSTOM and not has_keyword:
            score -= PENALTY_SENTENCE_CASE_NUMBERED

        # Sentence-ending period after words (excluding single-word numbers e.g. '1.')
        if self._ends_with_sentence_period(eval_text):
            score -= PENALTY_SENTENCE_PERIOD

        # Inline citations e.g. (Smith et al., 2020) or [1, 2]
        if re.search(r"\([A-Z][a-zA-Z\s]+,\s*\d{4}\)|\[\d+(?:,\s*\d+)*\]|\bet al\.", eval_text):
            score -= PENALTY_INLINE_CITATION

        # URLs or DOIs
        if re.search(r"https?://|www\.|10\.\d{4,9}/", eval_text):
            score -= PENALTY_URL_OR_DOI

        # Bullet list markers
        if re.match(r"^[\*\-\+•◦▪]\s+", eval_text):
            score -= PENALTY_BULLET_MARKER

        # Common prose phrases
        if PROSE_INDICATOR_PATTERN.search(eval_text):
            score -= PENALTY_PROSE_PHRASE

        # Starts with lowercase
        if re.match(r"^[a-z]", eval_text):
            score -= PENALTY_LOWERCASE_START

        # Math and equation syntax
        if any(sym in eval_text for sym in ("$$", r"\sum", r"\int", r"\frac", r"\mathbb")):
            score -= PENALTY_MATH_SYNTAX

        confidence = max(0.0, min(1.0, score))

        if confidence >= MIN_HEADING_CONFIDENCE:
            # Fallback level for unnumbered headings
            if level is None:
                level = 1

            candidate = _HeadingCandidate(
                paragraph=paragraph,
                title=paragraph.text,  # Verbatim original heading string preserved
                normalized_title=norm_title,
                section_type=section_type,
                level=level,
                numbering=num_str,
                confidence=round(confidence, 2),
            )
            return candidate, False

        # If it had positive signals but fell below threshold, record as rejected candidate
        if had_positive_signal:
            return None, True

        return None, False

    def _ends_with_sentence_period(self, text: str) -> bool:
        """Check if text ends with a period preceded by ordinary sentence words."""
        cleaned = text.strip()
        if not cleaned.endswith((".", "!", "?", ";")):
            return False

        # A period immediately following a single number e.g. "1." or "2.1." is numbering, not sentence period
        if re.match(r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.$", cleaned):
            return False

        return True

    def _is_title_case(self, text: str) -> bool:
        """Check if heading text exhibits title-case capitalization."""
        cleaned = re.sub(
            r"^(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\.:\s-]+",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).strip()
        raw_words = [re.sub(r"^[^\w]+|[^\w]+$", "", w) for w in cleaned.split()]
        words = [w for w in raw_words if w and w.isalpha()]
        if not words:
            return False

        stopwords = {
            "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at",
            "to", "from", "by", "with", "in", "of", "into", "onto", "as",
        }
        capitalized_count = 0
        for i, w in enumerate(words):
            if w[0].isupper():
                capitalized_count += 1
            elif i > 0 and w.lower() in stopwords:
                capitalized_count += 1

        return (capitalized_count / len(words)) >= 0.75

    def _is_sentence_case(self, text: str) -> bool:
        """Check if heading text exhibits sentence-case phrasing (indicative of prose/list)."""
        cleaned = re.sub(
            r"^(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\.:\s-]+",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).strip()
        raw_words = [re.sub(r"^[^\w]+|[^\w]+$", "", w) for w in cleaned.split()]
        words = [w for w in raw_words if w and w.isalpha()]
        if len(words) < 2:
            return False

        stopwords = {
            "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at",
            "to", "from", "by", "with", "in", "of", "into", "onto", "as",
        }
        if not words[0][0].isupper():
            return False

        content_words = [w for w in words[1:] if w.lower() not in stopwords]
        if not content_words:
            return False

        lowercase_content_count = sum(1 for w in content_words if w[0].islower())
        return (lowercase_content_count / len(content_words)) >= 0.50

    def _is_plausible_heading_phrase(self, text: str) -> bool:
        """Check if unnumbered text resembles a concise academic heading noun phrase."""
        cleaned = re.sub(
            r"^(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\.:\s-]+",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).strip()
        raw_words = [re.sub(r"^[^\w]+|[^\w]+$", "", w) for w in cleaned.split()]
        words = [w for w in raw_words if w and w.isalpha()]
        if not words:
            return False

        # First word must be capitalized
        if not words[0][0].isupper():
            return False

        # Headings are concise: rarely more than 6 content words
        if len(words) > 6 or len(cleaned) > 60:
            return False

        # Cannot contain finite verbs / predicates or subject pronouns
        if VERB_PREDICATE_PATTERN.search(cleaned):
            return False

        # Cannot end with sentence punctuation
        if cleaned.endswith((".", "!", "?", ";", ":")):
            return False

        return True

    def _contains_heading_keyword(self, normalized_title: str) -> bool:
        """Check if normalized title contains a standard section keyword."""
        keywords = (
            "abstract", "introduction", "related", "background", "preliminary",
            "method", "model", "experiment", "result", "discussion", "analysis",
            "conclusion", "limitation", "reference", "appendix", "acknowledgment",
        )
        return any(k in normalized_title for k in keywords)

    def _normalize_title(self, title: str) -> str:
        """Normalize heading text for comparison and classification."""
        t = " ".join(title.lower().split()).strip()
        # Strip trailing periods/colons
        t = re.sub(r"[\.:\s-]+$", "", t).strip()
        # Strip leading numbering (e.g., "1. ", "1.1 ", "IV. ", "Section 2: ")
        stripped = re.sub(
            r"^(?:section\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\.:\s-]+",
            "",
            t,
            flags=re.IGNORECASE,
        ).strip()
        if stripped:
            return stripped
        return t

    def _classify_title(self, normalized_title: str) -> SectionType:
        """Classify normalized heading title into a standard SectionType or CUSTOM."""
        # 1. Exact alias match
        for sec_type, aliases in SECTION_CATEGORY_ALIASES.items():
            if normalized_title in aliases:
                return sec_type

        # 2. Appendix prefix match (e.g., "appendix a", "appendix 1", "appendix: proofs")
        if normalized_title.startswith("appendix"):
            return SectionType.APPENDIX

        return SectionType.CUSTOM

    def _extract_numbering_and_level(self, text: str) -> tuple[str | None, int | None]:
        """Extract explicit numbering and determine structural hierarchy level."""
        cleaned = text.strip()

        # 1. Decimal numbering: 1, 1.1, 1.1.1, Section 1: ...
        m_dec = re.match(r"^(?:section\s+)?(\d+(?:\.\d+)*)[\.:\s-]", cleaned, re.IGNORECASE)
        if m_dec:
            num_str = m_dec.group(1)
            parts = [p for p in num_str.split(".") if p]
            level = len(parts)
            return num_str, level

        # 2. Roman numerals: I. Introduction, IV. Results
        m_rom = re.match(r"^(?:section\s+)?([IVXLCDM]+)[\.:\s-]", cleaned, re.IGNORECASE)
        if m_rom:
            num_str = m_rom.group(1)
            return num_str, 1

        # 3. Lettered Appendix: Appendix A, Appendix B: ...
        m_app = re.match(r"^appendix\s+([A-Z0-9]+)[\.:\s-]?", cleaned, re.IGNORECASE)
        if m_app:
            return m_app.group(1), 1

        # 4. Appendix lettered subsection: A.1 Proof of Theorem
        m_app_sub = re.match(r"^([A-Z])\.(\d+)[\.:\s-]", cleaned)
        if m_app_sub:
            return f"{m_app_sub.group(1)}.{m_app_sub.group(2)}", 2

        return None, None

    def _build_sections_and_hierarchy(
        self,
        heading_map: dict[int, _HeadingCandidate],
        warnings: list[str],
    ) -> tuple[list[DetectedSection], int]:
        """Build DetectedSection objects and establish parent-child relationships."""
        sections: list[DetectedSection] = []
        hierarchy_stack: list[DetectedSection] = []
        unclassified_count = 0

        # Sort candidate headings in document order
        sorted_candidates = [heading_map[idx] for idx in sorted(heading_map.keys())]

        for sec_idx, cand in enumerate(sorted_candidates):
            if cand.section_type == SectionType.CUSTOM:
                unclassified_count += 1

            sec_id = f"sec_{sec_idx + 1}"

            # Establish hierarchy using hierarchy stack
            # Lower-level or same-level closes deeper sections
            while hierarchy_stack and hierarchy_stack[-1].level >= cand.level:
                hierarchy_stack.pop()

            parent_id: str | None = None
            if hierarchy_stack:
                parent = hierarchy_stack[-1]
                parent_id = parent.section_id

                # Check for structural hierarchy jumps (e.g. level 1 directly to level 3)
                if cand.level > parent.level + 1:
                    warnings.append(
                        f"Hierarchy gap detected: section '{cand.title}' (level {cand.level}) "
                        f"attached to parent '{parent.title}' (level {parent.level})."
                    )

            meta: dict[str, Any] = {}
            if cand.numbering:
                meta["numbering"] = cand.numbering
            if cand.section_type == SectionType.APPENDIX:
                meta["is_appendix"] = True

            sec = DetectedSection(
                section_id=sec_id,
                section_index=sec_idx,
                heading_paragraph_index=cand.paragraph.global_index,
                title=cand.title,
                normalized_title=cand.normalized_title,
                section_type=cand.section_type,
                level=cand.level,
                parent_section_id=parent_id,
                heading_page=cand.paragraph.page_number,
                start_page=cand.paragraph.page_number,
                end_page=cand.paragraph.page_number,
                paragraphs=[],
                confidence=cand.confidence,
                metadata=meta,
            )

            sections.append(sec)
            hierarchy_stack.append(sec)

        return sections, unclassified_count

    def _assign_paragraphs(
        self,
        flattened: list[_FlattenedParagraph],
        heading_map: dict[int, _HeadingCandidate],
        sections: list[DetectedSection],
    ) -> list[SectionParagraph]:
        """Assign non-heading paragraphs to preamble or exactly one section's direct paragraphs."""
        preamble: list[SectionParagraph] = []
        section_by_start_idx = {
            cand.paragraph.global_index: sections[i]
            for i, cand in enumerate(sorted(heading_map.values(), key=lambda c: c.paragraph.global_index))
        }

        current_section: DetectedSection | None = None

        for p in flattened:
            if p.global_index in section_by_start_idx:
                current_section = section_by_start_idx[p.global_index]
                # Heading is represented by DetectedSection.title, not added as a body paragraph
                continue

            sec_para = SectionParagraph(
                text=p.text,
                page_number=p.page_number,
                paragraph_index=p.global_index,
            )

            if current_section is None:
                preamble.append(sec_para)
            else:
                current_section.paragraphs.append(sec_para)

        return preamble

    def _calculate_section_spans(self, sections: list[DetectedSection]) -> None:
        """Compute start_page and end_page for all sections, including descendant spans."""
        section_by_id = {s.section_id: s for s in sections}

        # 1. Base spans from direct body paragraphs and heading page
        for s in sections:
            if s.paragraphs:
                min_page = min(p.page_number for p in s.paragraphs)
                max_page = max(p.page_number for p in s.paragraphs)
                s.start_page = min(s.heading_page, min_page)
                s.end_page = max(s.heading_page, max_para_page := max_page)
            else:
                s.start_page = s.heading_page
                s.end_page = s.heading_page

        # 2. Propagate descendant spans upwards to parents in reverse order
        for s in reversed(sections):
            if s.parent_section_id and s.parent_section_id in section_by_id:
                parent = section_by_id[s.parent_section_id]
                parent.start_page = min(parent.start_page, s.start_page)
                parent.end_page = max(parent.end_page, s.end_page)
                parent.metadata["has_children"] = True
