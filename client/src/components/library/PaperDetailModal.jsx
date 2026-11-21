import React, { useEffect, useState } from 'react';
import { getPaperDetails } from '../../api/papersApi';

function formatFileSize(bytes) {
  if (typeof bytes !== 'number' || isNaN(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const size = (bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1);
  return `${size} ${units[i]}`;
}

function formatPaperDate(isoString) {
  if (!isoString) return 'Not available';
  try {
    const d = new Date(isoString);
    return d.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  } catch {
    return 'Not available';
  }
}

/**
 * Modal dialog for inspecting complete paper details and initiating research conversations.
 *
 * @param {object} props
 * @param {boolean} props.isOpen - Whether the modal is visible
 * @param {string|null} props.paperId - UUID of the selected paper
 * @param {Function} props.onClose - Dismiss modal handler
 * @param {Function} props.onStartChat - Initiate research conversation handler
 * @param {Function} [props.onDelete] - Initiate paper deletion handler
 */
export default function PaperDetailModal({ isOpen, paperId, onClose, onStartChat, onDelete }) {
  const [details, setDetails] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!isOpen || !paperId) {
      setDetails(null);
      setError(null);
      return;
    }

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    getPaperDetails(paperId)
      .then((data) => {
        if (isMounted) {
          setDetails(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Failed to load paper details.');
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [isOpen, paperId]);

  if (!isOpen) return null;

  const authorsText = details?.authors?.length
    ? details.authors
        .map((a) => (typeof a === 'string' ? a : a.name || 'Unknown Author'))
        .join(', ')
    : null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4 sm:p-6 animate-fade-in">
      {/* Backdrop click to close */}
      <div className="fixed inset-0" onClick={onClose} aria-hidden="true" />

      {/* Modal Card */}
      <div className="relative bg-white rounded-2xl shadow-xl border border-slate-200/90 max-w-2xl w-full max-h-[90vh] flex flex-col overflow-hidden z-10">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <span className="text-[10px] font-semibold tracking-wider uppercase text-indigo-600 block mb-0.5">
              Research Paper Details
            </span>
            <h2 className="text-base sm:text-lg font-bold text-slate-900 leading-snug line-clamp-2">
              {details?.title || 'Loading paper details...'}
            </h2>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer shrink-0"
            aria-label="Close modal"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {isLoading ? (
            <div className="py-12 flex flex-col items-center justify-center text-slate-400 gap-3">
              <svg className="animate-spin h-7 w-7 text-indigo-600" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              <p className="text-xs font-medium text-slate-500">Retrieving document metadata...</p>
            </div>
          ) : error ? (
            <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl space-y-2">
              <div className="flex items-center gap-2 text-rose-800 font-semibold text-xs">
                <svg className="w-4 h-4 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Error Loading Details</span>
              </div>
              <p className="text-xs text-rose-700">{error}</p>
            </div>
          ) : details ? (
            <>
              {/* Document Overview Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-50 p-4 rounded-xl border border-slate-100">
                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400 block tracking-wider">
                    Indexed Chunks
                  </span>
                  <span className="text-base font-bold text-slate-900 font-mono">
                    {details.chunk_count ?? 0}
                  </span>
                </div>

                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400 block tracking-wider">
                    Total Pages
                  </span>
                  <span className="text-base font-bold text-slate-900 font-mono">
                    {details.page_count ? details.page_count : '—'}
                  </span>
                </div>

                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400 block tracking-wider">
                    Status
                  </span>
                  <span className="text-xs font-semibold text-emerald-700 block mt-1">
                    {details.ingestion_status === 'completed' ? 'Indexed & Ready' : (details.ingestion_status || 'Available')}
                  </span>
                </div>

                <div>
                  <span className="text-[10px] uppercase font-semibold text-slate-400 block tracking-wider">
                    Upload Date
                  </span>
                  <span className="text-xs font-semibold text-slate-800 block mt-1">
                    {formatPaperDate(details.created_at)}
                  </span>
                </div>
              </div>

              {/* Source File Information */}
              <div className="space-y-2">
                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Source File Information
                </h4>
                <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl space-y-1.5 text-xs">
                  <div className="flex items-center justify-between gap-4">
                    <span className="text-slate-500">Original Filename:</span>
                    <span className="font-mono text-slate-800 font-medium truncate">
                      {details.original_filename || 'Unknown'}
                    </span>
                  </div>
                  {details.file_size_bytes && (
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-slate-500">File Size:</span>
                      <span className="font-mono text-slate-800">
                        {formatFileSize(details.file_size_bytes)}
                      </span>
                    </div>
                  )}
                  {details.doi && (
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-slate-500">DOI:</span>
                      <span className="font-mono text-slate-800 truncate">
                        {details.doi}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Metadata Details (Authors, Venue, Year) */}
              {(authorsText || details.venue || details.publication_year) && (
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                    Academic Citation Metadata
                  </h4>
                  <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl space-y-2 text-xs">
                    {authorsText && (
                      <div>
                        <span className="text-slate-500 block mb-0.5">Authors:</span>
                        <p className="font-medium text-slate-800">{authorsText}</p>
                      </div>
                    )}
                    <div className="grid grid-cols-2 gap-3 pt-1">
                      {details.publication_year && (
                        <div>
                          <span className="text-slate-500 block mb-0.5">Year:</span>
                          <span className="font-semibold text-slate-800 font-mono">
                            {details.publication_year}
                          </span>
                        </div>
                      )}
                      {details.venue && (
                        <div>
                          <span className="text-slate-500 block mb-0.5">Venue:</span>
                          <span className="font-medium text-slate-800 truncate block">
                            {details.venue}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Abstract */}
              {details.abstract ? (
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                    Abstract
                  </h4>
                  <div className="p-4 bg-slate-50/70 border border-slate-200/80 rounded-xl text-xs text-slate-700 leading-relaxed max-h-48 overflow-y-auto">
                    {details.abstract}
                  </div>
                </div>
              ) : (
                <div className="p-3.5 bg-slate-50 border border-dashed border-slate-200 rounded-xl text-xs text-slate-400 text-center">
                  No abstract extracted for this document.
                </div>
              )}
            </>
          ) : null}
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-4 border-t border-slate-100 bg-slate-50/50 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:text-slate-800 hover:bg-slate-200/60 transition-colors cursor-pointer"
            >
              Close
            </button>
            {details && onDelete && (
              <button
                type="button"
                onClick={() => {
                  onClose();
                  onDelete(details);
                }}
                className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold text-rose-600 hover:text-rose-700 hover:bg-rose-50 transition-colors cursor-pointer"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                <span>Delete Paper</span>
              </button>
            )}
          </div>

          {details && (
            <button
              type="button"
              onClick={() => {
                onClose();
                onStartChat(details);
              }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 shadow-xs transition-colors cursor-pointer"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <span>Ask Questions About This Paper</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
