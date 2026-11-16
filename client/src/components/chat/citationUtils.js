/**
 * Citation utilities for formatting and deduplicating research citations.
 */

/**
 * Deduplicates citations by paper title and page number to prevent redundant references.
 *
 * @param {Array<Object>} citations - Raw citation objects from backend API
 * @returns {Array<Object>} Deduplicated citations with only title and page fields preserved
 */
export function getUniqueCitations(citations) {
  if (!Array.isArray(citations)) return [];
  const seen = new Set();
  const unique = [];

  for (const c of citations) {
    if (!c || !c.paper_title) continue;
    const pageKey = c.page_number != null ? String(c.page_number) : 'none';
    const key = `${c.paper_title.trim()}:::${pageKey}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push({
        paper_title: c.paper_title.trim(),
        page_number: c.page_number ?? null,
      });
    }
  }

  return unique;
}

/**
 * Formats a clean reference display string:
 * e.g., "Attention Is All You Need — Page 3" or "Attention Is All You Need"
 *
 * Excludes chunk IDs, relevance scores, embedding data, or chunk text.
 *
 * @param {Object} citation
 * @returns {string}
 */
export function formatCitationText(citation) {
  if (!citation || !citation.paper_title) return '';
  const title = citation.paper_title.trim();
  if (citation.page_number != null) {
    return `${title} — Page ${citation.page_number}`;
  }
  return title;
}
