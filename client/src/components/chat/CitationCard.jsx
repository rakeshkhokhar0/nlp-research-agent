import React from 'react';

/**
 * Renders a clean reference line displaying strictly paper title and page number.
 * Conforms to product presentation requirements:
 * Example:
 * - Attention Is All You Need — Page 3
 */
export default function CitationCard({ citation }) {
  if (!citation || !citation.paper_title) return null;

  const { paper_title, page_number } = citation;

  return (
    <li className="flex items-center gap-1.5 py-0.5 text-xs text-slate-700">
      <span className="text-slate-400 font-bold">&bull;</span>
      <span className="font-medium text-slate-800">{paper_title}</span>
      {page_number != null && (
        <span className="text-slate-500 font-normal">&mdash; Page {page_number}</span>
      )}
    </li>
  );
}
