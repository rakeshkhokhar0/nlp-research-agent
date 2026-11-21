import React from 'react';

/**
 * Formats byte count into human-readable string (KB, MB).
 */
function formatFileSize(bytes) {
  if (typeof bytes !== 'number' || isNaN(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const size = (bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1);
  return `${size} ${units[i]}`;
}

/**
 * Formats ISO date string into readable date.
 */
function formatPaperDate(isoString) {
  if (!isoString) return 'Recently';
  try {
    const d = new Date(isoString);
    return d.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return 'Recently';
  }
}

/**
 * Maps status to badge styling and label.
 */
function getStatusBadge(status) {
  switch (status?.toLowerCase()) {
    case 'completed':
      return {
        label: 'Indexed',
        className: 'bg-emerald-50 text-emerald-700 border-emerald-200',
        dotColor: 'bg-emerald-500',
      };
    case 'processing':
      return {
        label: 'Processing',
        className: 'bg-blue-50 text-blue-700 border-blue-200',
        dotColor: 'bg-blue-500 animate-pulse',
      };
    case 'failed':
      return {
        label: 'Failed',
        className: 'bg-rose-50 text-rose-700 border-rose-200',
        dotColor: 'bg-rose-500',
      };
    default:
      return {
        label: status || 'Available',
        className: 'bg-slate-100 text-slate-700 border-slate-200',
        dotColor: 'bg-slate-400',
      };
  }
}

/**
 * Card representing an individual research paper in the catalog.
 *
 * @param {object} props
 * @param {object} props.paper - The paper data object
 * @param {Function} props.onViewDetails - Handler to view detailed metadata
 * @param {Function} props.onStartChat - Handler to start a research conversation
 * @param {Function} [props.onDelete] - Handler to trigger paper deletion confirmation
 */
export default function PaperCard({ paper, onViewDetails, onStartChat, onDelete }) {
  const statusInfo = getStatusBadge(paper.ingestion_status);

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 p-5 shadow-2xs hover:shadow-xs hover:border-indigo-200 transition-all flex flex-col justify-between group">
      <div>
        {/* Top Header: Ingestion Status Badge, Date & Delete Action */}
        <div className="flex items-center justify-between gap-2 mb-3">
          <div
            className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${statusInfo.className}`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${statusInfo.dotColor}`} />
            <span>{statusInfo.label}</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[11px] text-slate-400 font-medium">
              {formatPaperDate(paper.created_at)}
            </span>
            {onDelete && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(paper);
                }}
                title="Delete paper"
                className="p-1 rounded text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors cursor-pointer"
                aria-label={`Delete ${paper.title}`}
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Paper Title */}
        <h3
          onClick={() => onViewDetails(paper)}
          className="text-sm font-semibold text-slate-900 group-hover:text-indigo-600 transition-colors line-clamp-2 leading-snug cursor-pointer mb-2"
          title={paper.title}
        >
          {paper.title}
        </h3>

        {/* File & Metadata Info */}
        <div className="space-y-1.5 mb-4">
          <div className="flex items-center gap-2 text-xs text-slate-500 min-w-0">
            <svg className="w-3.5 h-3.5 text-slate-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <span className="truncate font-mono text-[11px]">
              {paper.original_filename || 'document.pdf'}
            </span>
            {paper.file_size_bytes && (
              <span className="text-[11px] text-slate-400 shrink-0">
                ({formatFileSize(paper.file_size_bytes)})
              </span>
            )}
          </div>

          {paper.doi && (
            <div className="flex items-center gap-1.5 text-[11px] text-slate-400 font-mono truncate">
              <span className="text-slate-500 font-medium shrink-0">DOI:</span>
              <span className="truncate">{paper.doi}</span>
            </div>
          )}
        </div>

        {/* Extracted Stats Grid (Chunks & Pages) */}
        <div className="grid grid-cols-2 gap-2 py-2.5 px-3 bg-slate-50/80 rounded-xl border border-slate-100 text-xs mb-4">
          <div>
            <span className="text-[10px] text-slate-400 uppercase font-semibold block tracking-wider">
              Chunks
            </span>
            <span className="font-bold text-slate-800 font-mono text-sm">
              {paper.chunk_count ?? 0}
            </span>
          </div>

          <div>
            <span className="text-[10px] text-slate-400 uppercase font-semibold block tracking-wider">
              Pages
            </span>
            <span className="font-bold text-slate-800 font-mono text-sm">
              {paper.page_count ? paper.page_count : '—'}
            </span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="pt-2 border-t border-slate-100 flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => onViewDetails(paper)}
          className="text-xs font-semibold text-slate-600 hover:text-slate-900 px-2.5 py-1.5 rounded-lg hover:bg-slate-100 transition-colors cursor-pointer"
        >
          View Details
        </button>

        <button
          type="button"
          onClick={() => onStartChat(paper)}
          className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 px-3 py-1.5 rounded-lg border border-indigo-200 transition-colors cursor-pointer"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
          <span>Ask Questions</span>
        </button>
      </div>
    </div>
  );
}
