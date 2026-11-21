import React, { useState, useRef, useEffect, useCallback } from 'react';
import { uploadPaper, getIngestionJob, validatePdfFile } from '../../api/ingestionApi';
import { listPapers, deletePaper } from '../../api/papersApi';
import { useConversation } from '../../context/ConversationContext';
import PaperCard from './PaperCard';
import PaperDetailModal from './PaperDetailModal';

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
 * Safely extracts a readable string from any error object, string, or array.
 */
function formatDisplayError(err) {
  if (!err) return '';
  if (typeof err === 'string') return err;
  if (typeof err?.message === 'string') return err.message;
  if (Array.isArray(err)) {
    return err
      .map((item) => (typeof item === 'string' ? item : item.msg || JSON.stringify(item)))
      .join('; ');
  }
  if (typeof err === 'object') {
    return err.msg || err.detail || err.message || JSON.stringify(err);
  }
  return String(err);
}

/**
 * Status badge styling and label mapping for real backend job states.
 */
function getStatusBadge(status, stage) {
  switch (status?.toLowerCase()) {
    case 'completed':
      return {
        label: 'Indexed & Ready',
        className: 'bg-emerald-50 text-emerald-700 border-emerald-200',
        dotColor: 'bg-emerald-500',
      };
    case 'processing':
      return {
        label: stage ? `Processing: ${stage}` : 'Processing',
        className: 'bg-blue-50 text-blue-700 border-blue-200',
        dotColor: 'bg-blue-500 animate-pulse',
      };
    case 'pending':
      return {
        label: 'Queued',
        className: 'bg-amber-50 text-amber-700 border-amber-200',
        dotColor: 'bg-amber-500 animate-pulse',
      };
    case 'failed':
      return {
        label: 'Failed',
        className: 'bg-rose-50 text-rose-700 border-rose-200',
        dotColor: 'bg-rose-500',
      };
    case 'cancelled':
      return {
        label: 'Cancelled',
        className: 'bg-slate-100 text-slate-700 border-slate-300',
        dotColor: 'bg-slate-400',
      };
    default:
      return {
        label: status || 'Unknown',
        className: 'bg-slate-50 text-slate-600 border-slate-200',
        dotColor: 'bg-slate-400',
      };
  }
}

/**
 * Complete Research Library component featuring:
 * 1. Paper Catalog with search, status badges, chunk counts, and paper details inspection
 * 2. PDF upload interface with drag & drop, manual title, DOI, and live ingestion tracking
 * 3. Seamless transition to Research Chat for grounded question answering
 *
 * @param {object} props
 * @param {Function} props.onNavigateToChat - Callback to switch back to the chat workspace
 */
export default function ResearchLibrary({ onNavigateToChat }) {
  const { startNewConversation, setInputPrompt } = useConversation();

  const [activeTab, setActiveTab] = useState('catalog');

  const [papers, setPapers] = useState([]);
  const [isLoadingPapers, setIsLoadingPapers] = useState(false);
  const [papersError, setPapersError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPaperId, setSelectedPaperId] = useState(null);

  const [paperToDelete, setPaperToDelete] = useState(null);
  const [isDeletingPaper, setIsDeletingPaper] = useState(false);
  const [deleteError, setDeleteError] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [paperTitle, setPaperTitle] = useState('');
  const [doi, setDoi] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);

  const [isUploading, setIsUploading] = useState(false);
  const [activeJob, setActiveJob] = useState(null);
  const [jobError, setJobError] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);

  const [sessionPapers, setSessionPapers] = useState([]);

  const fileInputRef = useRef(null);
  const pollingTimerRef = useRef(null);

  const fetchPapers = useCallback(async (query = '') => {
    setIsLoadingPapers(true);
    setPapersError(null);
    try {
      const data = await listPapers({ limit: 100, search: query });
      setPapers(data.items || []);
    } catch (err) {
      setPapersError(formatDisplayError(err) || 'Failed to load research papers.');
    } finally {
      setIsLoadingPapers(false);
    }
  }, []);

  useEffect(() => {
    fetchPapers();
  }, [fetchPapers]);

  // Cleanup polling timer on unmount
  useEffect(() => {
    return () => {
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
        pollingTimerRef.current = null;
      }
    };
  }, []);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    fetchPapers(searchQuery);
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    fetchPapers('');
  };

  const handleInitiateDelete = (paper) => {
    setDeleteError(null);
    setPaperToDelete(paper);
  };

  const handleConfirmDelete = async () => {
    if (!paperToDelete || isDeletingPaper) return;
    setIsDeletingPaper(true);
    setDeleteError(null);
    try {
      await deletePaper(paperToDelete.id);
      setPapers((prev) => prev.filter((p) => p.id !== paperToDelete.id));
      setSessionPapers((prev) => prev.filter((p) => p.paper_id !== paperToDelete.id));
      if (selectedPaperId === paperToDelete.id) {
        setSelectedPaperId(null);
      }
      setPaperToDelete(null);
      // Refresh list to keep total counts and pagination accurate
      fetchPapers(searchQuery);
    } catch (err) {
      setDeleteError(formatDisplayError(err) || 'Failed to delete research paper.');
    } finally {
      setIsDeletingPaper(false);
    }
  };

  const handleFileChange = (file) => {
    setValidationError(null);
    setJobError(null);
    if (!file) {
      setSelectedFile(null);
      return;
    }

    try {
      validatePdfFile(file);
      setSelectedFile(file);
    } catch (err) {
      setValidationError(err.message);
      setSelectedFile(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      handleFileChange(file);
    }
  };

  const handleClearSelected = () => {
    setSelectedFile(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const startPolling = (jobId) => {
    if (pollingTimerRef.current) {
      clearInterval(pollingTimerRef.current);
    }

    pollingTimerRef.current = setInterval(async () => {
      try {
        const job = await getIngestionJob(jobId);
        setActiveJob(job);

        const currentStatus = job.status?.toLowerCase();
        if (currentStatus === 'completed' || currentStatus === 'failed' || currentStatus === 'cancelled') {
          clearInterval(pollingTimerRef.current);
          pollingTimerRef.current = null;
          setIsUploading(false);

          if (currentStatus === 'failed') {
            setJobError(job.error_message || 'Ingestion job failed during processing.');
          } else if (currentStatus === 'completed') {
            // Refresh catalog list so new paper is immediately visible
            fetchPapers(searchQuery);
          }
        }
      } catch (err) {
        console.warn('Failed to poll ingestion job status:', err);
      }
    }, 1500);
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile || isUploading) return;

    try {
      setIsUploading(true);
      setValidationError(null);
      setJobError(null);
      setActiveJob(null);
      setUploadResult(null);

      // Perform upload
      const result = await uploadPaper(selectedFile, { title: paperTitle, doi });
      setUploadResult(result);

      const paperRecord = {
        id: result.paper_id,
        title: result.title || paperTitle.trim() || selectedFile.name,
        filename: selectedFile.name,
        size: selectedFile.size,
        chunksCreated: result.chunks_created,
        embeddingsCreated: result.embeddings_created,
        status: result.status,
        timestamp: new Date().toISOString(),
      };

      setSessionPapers((prev) => [paperRecord, ...prev]);

      // If status is already completed (synchronous ingestion)
      if (result.status?.toLowerCase() === 'completed') {
        setActiveJob({
          job_id: result.job_id,
          paper_id: result.paper_id,
          status: 'completed',
          current_stage: 'indexed',
        });
        setIsUploading(false);
        // Refresh catalog list
        fetchPapers(searchQuery);
      } else if (result.job_id) {
        // Asynchronous ingestion: poll job status endpoint
        setActiveJob({
          job_id: result.job_id,
          paper_id: result.paper_id,
          status: result.status || 'processing',
          current_stage: 'initializing',
        });
        startPolling(result.job_id);
      } else {
        setIsUploading(false);
      }
    } catch (err) {
      setIsUploading(false);
      setJobError(formatDisplayError(err) || 'Failed to upload paper. Please check server connection.');
    }
  };

  const handleStartChatWithPaper = (paper = null) => {
    startNewConversation();
    if (paper?.title) {
      setInputPrompt(`Can you summarize the core contributions of "${paper.title}"?`);
    }
    if (onNavigateToChat) {
      onNavigateToChat();
    }
  };

  const handleResetForAnotherUpload = () => {
    setSelectedFile(null);
    setPaperTitle('');
    setDoi('');
    setValidationError(null);
    setJobError(null);
    setActiveJob(null);
    setUploadResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 p-4 sm:p-6 lg:p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Main Page Header */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-2xs">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div className="flex items-start gap-4">
              <div className="h-12 w-12 rounded-xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 shrink-0 shadow-2xs">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.75}
                    d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                  />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900 tracking-tight">
                  Research Library & Knowledge Base
                </h1>
                <p className="text-xs sm:text-sm text-slate-500 mt-1 leading-relaxed">
                  Browse, search, and inspect uploaded research papers indexed into the lexical and semantic retrieval engine.
                </p>
              </div>
            </div>

            {onNavigateToChat && (
              <button
                type="button"
                onClick={() => onNavigateToChat()}
                className="self-start sm:self-center inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-200 transition-colors cursor-pointer shrink-0"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                <span>Back to Research Chat</span>
              </button>
            )}
          </div>

          {/* Navigation Tabs (Catalog vs Upload) */}
          <div className="flex items-center gap-2 mt-6 pt-5 border-t border-slate-100">
            <button
              type="button"
              onClick={() => setActiveTab('catalog')}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                activeTab === 'catalog'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'bg-slate-100 text-slate-600 hover:text-slate-900 hover:bg-slate-200/80'
              }`}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              <span>Library Catalog</span>
              <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                activeTab === 'catalog' ? 'bg-indigo-700 text-indigo-100' : 'bg-slate-200 text-slate-700'
              }`}>
                {papers.length}
              </span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab('upload')}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                activeTab === 'upload'
                  ? 'bg-indigo-600 text-white shadow-xs'
                  : 'bg-slate-100 text-slate-600 hover:text-slate-900 hover:bg-slate-200/80'
              }`}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              <span>Upload New Paper</span>
            </button>
          </div>
        </div>

        {/* View 1: Library Catalog */}
        {activeTab === 'catalog' && (
          <div className="space-y-5">
            {/* Catalog Controls Bar: Search & Refresh */}
            <div className="bg-white rounded-2xl p-4 border border-slate-200/80 shadow-2xs flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
              {/* Search Form */}
              <form onSubmit={handleSearchSubmit} className="flex-1 relative flex items-center">
                <svg
                  className="w-4 h-4 text-slate-400 absolute left-3.5 pointer-events-none"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  placeholder="Search papers by title or DOI..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full text-xs pl-10 pr-10 py-2.5 rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all font-medium"
                />
                {searchQuery && (
                  <button
                    type="button"
                    onClick={handleClearSearch}
                    className="absolute right-3 p-1 rounded-md text-slate-400 hover:text-slate-600 cursor-pointer"
                    title="Clear search"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </form>

              {/* Actions: Refresh & Upload */}
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => fetchPapers(searchQuery)}
                  disabled={isLoadingPapers}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-200 transition-colors cursor-pointer disabled:opacity-50"
                  title="Refresh library catalog"
                >
                  <svg
                    className={`w-3.5 h-3.5 ${isLoadingPapers ? 'animate-spin text-indigo-600' : 'text-slate-500'}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <span>Refresh</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveTab('upload')}
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 shadow-xs transition-colors cursor-pointer"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  <span>Upload Paper</span>
                </button>
              </div>
            </div>

            {/* Delete Error Banner */}
            {deleteError && !paperToDelete && (
              <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl flex items-center justify-between gap-3 text-xs text-rose-800 animate-fade-in">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-rose-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{deleteError}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setDeleteError(null)}
                  className="text-rose-500 hover:text-rose-700 font-bold px-1.5 py-0.5 rounded cursor-pointer"
                  aria-label="Dismiss error"
                >
                  ×
                </button>
              </div>
            )}

            {/* Error State */}
            {papersError && (
              <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl flex items-center justify-between gap-3 text-xs text-rose-800">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-rose-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{papersError}</span>
                </div>
                <button
                  type="button"
                  onClick={() => fetchPapers(searchQuery)}
                  className="px-2.5 py-1 rounded-lg bg-rose-100 hover:bg-rose-200 font-semibold text-rose-900 transition-colors cursor-pointer shrink-0"
                >
                  Retry
                </button>
              </div>
            )}

            {/* Loading State */}
            {isLoadingPapers ? (
              <div className="py-16 bg-white rounded-2xl border border-slate-200/80 p-8 text-center space-y-3">
                <svg className="animate-spin h-8 w-8 text-indigo-600 mx-auto" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <p className="text-sm font-semibold text-slate-700">Loading Research Papers...</p>
                <p className="text-xs text-slate-400">Fetching document catalog from PostgreSQL</p>
              </div>
            ) : papers.length === 0 ? (
              /* Empty Catalog State */
              <div className="bg-white rounded-2xl border border-slate-200/80 p-12 text-center max-w-md mx-auto space-y-4">
                <div className="h-16 w-16 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 mx-auto">
                  <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    {searchQuery ? 'No Matching Papers Found' : 'Your Research Library is Empty'}
                  </h3>
                  <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                    {searchQuery
                      ? `No research papers matched "${searchQuery}". Try a different search term or clear the filter.`
                      : 'Upload research paper PDFs to extract text, compute vector embeddings, and start evidence-grounded research chats.'}
                  </p>
                </div>
                {searchQuery ? (
                  <button
                    type="button"
                    onClick={handleClearSearch}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 border border-slate-200 transition-colors cursor-pointer"
                  >
                    Clear Search Query
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => setActiveTab('upload')}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 shadow-xs transition-colors cursor-pointer"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                    <span>Upload Your First Paper</span>
                  </button>
                )}
              </div>
            ) : (
              /* Papers Grid */
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {papers.map((paper) => (
                  <PaperCard
                    key={paper.id}
                    paper={paper}
                    onViewDetails={(p) => setSelectedPaperId(p.id)}
                    onStartChat={handleStartChatWithPaper}
                    onDelete={handleInitiateDelete}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* View 2: Upload New Research Paper */}
        {activeTab === 'upload' && (
          <div className="space-y-6">
            <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-2xs space-y-5">
              <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
                    Upload New Research Paper
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Select or drag-and-drop a research paper PDF up to 50 MB.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={() => setActiveTab('catalog')}
                  className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 transition-colors cursor-pointer"
                >
                  &larr; View Catalog ({papers.length})
                </button>
              </div>

              <form onSubmit={handleUpload} className="space-y-4">
                {/* Drag & Drop File Zone */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                    isDragging
                      ? 'border-indigo-500 bg-indigo-50/60 ring-2 ring-indigo-200'
                      : selectedFile
                      ? 'border-indigo-300 bg-indigo-50/20'
                      : 'border-slate-300 hover:border-indigo-400 bg-slate-50/50 hover:bg-slate-50'
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,application/pdf"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files && e.target.files.length > 0) {
                        handleFileChange(e.target.files[0]);
                      }
                    }}
                  />

                  {selectedFile ? (
                    /* Selected File Card */
                    <div className="flex items-center justify-between bg-white p-3.5 rounded-xl border border-indigo-200 shadow-2xs max-w-lg mx-auto">
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="h-10 w-10 rounded-lg bg-indigo-100 text-indigo-700 flex items-center justify-center shrink-0">
                          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </div>
                        <div className="text-left min-w-0">
                          <p className="text-xs font-semibold text-slate-800 truncate">
                            {paperTitle.trim() || selectedFile.name}
                          </p>
                          <p className="text-[11px] text-slate-500 font-mono">
                            {paperTitle.trim()
                              ? `PDF: ${selectedFile.name} • ${formatFileSize(selectedFile.size)}`
                              : `${formatFileSize(selectedFile.size)} • Ready for upload`}
                          </p>
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleClearSelected();
                        }}
                        title="Remove file"
                        className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-slate-100 transition-colors"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ) : (
                    /* Empty Upload Prompt */
                    <div className="space-y-2">
                      <div className="h-12 w-12 rounded-xl bg-slate-100 text-slate-400 mx-auto flex items-center justify-center">
                        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                      </div>
                      <div>
                        <p className="text-xs sm:text-sm font-semibold text-slate-700">
                          Click to browse or drag and drop paper PDF
                        </p>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Standard academic PDFs (.pdf) up to 50 MB
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Validation Error Banner */}
                {validationError && (
                  <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl flex items-center gap-2.5 text-xs text-rose-800">
                    <svg className="w-4 h-4 text-rose-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>{validationError}</span>
                  </div>
                )}

                {/* Optional Research Paper Name Field */}
                <div className="max-w-md">
                  <label htmlFor="paper-title" className="block text-xs font-medium text-slate-700 mb-1">
                    Research Paper Name <span className="text-slate-400 font-normal">(optional)</span>
                  </label>
                  <input
                    id="paper-title"
                    type="text"
                    placeholder="e.g. Attention Is All You Need"
                    value={paperTitle}
                    onChange={(e) => setPaperTitle(e.target.value)}
                    disabled={isUploading}
                    className="w-full text-xs px-3 py-2 rounded-lg border border-slate-200 bg-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all font-medium text-slate-800"
                  />
                  <p className="text-[11px] text-slate-400 mt-1">
                    Manually enter paper title (takes precedence over automatically extracted PDF metadata).
                  </p>
                </div>

                {/* Optional DOI Field */}
                <div className="max-w-md">
                  <label htmlFor="paper-doi" className="block text-xs font-medium text-slate-700 mb-1">
                    Paper DOI <span className="text-slate-400 font-normal">(optional)</span>
                  </label>
                  <input
                    id="paper-doi"
                    type="text"
                    placeholder="e.g. 10.1145/3065386 or 10.48550/arXiv.1706.03762"
                    value={doi}
                    onChange={(e) => setDoi(e.target.value)}
                    disabled={isUploading}
                    className="w-full text-xs px-3 py-2 rounded-lg border border-slate-200 bg-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all font-mono"
                  />
                </div>

                {/* Ingestion Submit Button */}
                <div className="flex items-center gap-3 pt-2">
                  <button
                    type="submit"
                    disabled={!selectedFile || isUploading}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-xs text-white bg-indigo-600 hover:bg-indigo-700 shadow-xs hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isUploading ? (
                      <>
                        <svg className="animate-spin -ml-0.5 w-4 h-4 text-white" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                        </svg>
                        <span>Ingesting Paper...</span>
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        <span>Upload & Ingest Paper</span>
                      </>
                    )}
                  </button>

                  {selectedFile && !isUploading && (
                    <button
                      type="button"
                      onClick={handleClearSelected}
                      className="px-3.5 py-2.5 rounded-xl font-medium text-xs text-slate-600 hover:text-slate-800 hover:bg-slate-100 transition-colors cursor-pointer"
                    >
                      Cancel
                    </button>
                  )}
                </div>
              </form>
            </div>

            {/* Live Ingestion Job Status Display */}
            {(isUploading || activeJob || jobError || uploadResult) && (
              <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-2xs space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                    <span>Ingestion Pipeline Status</span>
                    {isUploading && (
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-indigo-50 text-indigo-700">
                        Live
                      </span>
                    )}
                  </h2>

                  {activeJob && (
                    <div
                      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${
                        getStatusBadge(activeJob.status, activeJob.current_stage).className
                      }`}
                    >
                      <span
                        className={`h-2 w-2 rounded-full ${
                          getStatusBadge(activeJob.status, activeJob.current_stage).dotColor
                        }`}
                      />
                      <span>{getStatusBadge(activeJob.status, activeJob.current_stage).label}</span>
                    </div>
                  )}
                </div>

                {/* Error State */}
                {jobError && (
                  <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl space-y-2">
                    <div className="flex items-center gap-2 text-rose-800 font-semibold text-xs">
                      <svg className="w-4 h-4 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      <span>Ingestion Pipeline Error</span>
                    </div>
                    <p className="text-xs text-rose-700 font-mono break-all">{formatDisplayError(jobError)}</p>
                    <button
                      type="button"
                      onClick={handleResetForAnotherUpload}
                      className="mt-2 text-xs font-semibold text-rose-700 underline hover:text-rose-900 cursor-pointer"
                    >
                      Try uploading another file
                    </button>
                  </div>
                )}

                {/* In-Progress Progress Stage Tracker */}
                {isUploading && !jobError && (
                  <div className="p-4 bg-indigo-50/50 rounded-xl border border-indigo-100 space-y-3">
                    <div className="flex items-center justify-between text-xs text-indigo-900 font-medium">
                      <span className="flex items-center gap-2">
                        <svg className="animate-spin w-4 h-4 text-indigo-600" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                        </svg>
                        <span>Parsing document, detecting sections, computing chunk embeddings...</span>
                      </span>
                      <span className="font-mono text-indigo-700 text-[11px]">
                        Stage: {activeJob?.current_stage || 'processing'}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500">
                      The retrieval engine will automatically refresh its TF-IDF index upon completion.
                    </p>
                  </div>
                )}

                {/* Completed Success Summary */}
                {activeJob?.status?.toLowerCase() === 'completed' && uploadResult && (
                  <div className="space-y-4">
                    <div className="p-4 bg-emerald-50/60 border border-emerald-200 rounded-xl space-y-2">
                      <div className="flex items-center gap-2 text-emerald-900 font-semibold text-xs">
                        <svg className="w-4 h-4 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span>Document Successfully Ingested & Indexed</span>
                      </div>
                      <p className="text-xs text-emerald-800">
                        <strong className="font-semibold text-emerald-950">
                          {uploadResult.title || paperTitle.trim() || selectedFile?.name}
                        </strong>{' '}
                        is now fully indexed. The retrieval service can query its chunks and citations in both Normal and Advanced research modes.
                      </p>
                    </div>

                    {/* Metrics Breakdown Grid */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                        <p className="text-[11px] text-slate-500 font-medium">Chunks Created</p>
                        <p className="text-lg font-bold text-slate-900 font-mono mt-0.5">
                          {uploadResult.chunks_created ?? 0}
                        </p>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                        <p className="text-[11px] text-slate-500 font-medium">Embeddings Created</p>
                        <p className="text-lg font-bold text-slate-900 font-mono mt-0.5">
                          {uploadResult.embeddings_created ?? 0}
                        </p>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                        <p className="text-[11px] text-slate-500 font-medium">Paper Reused</p>
                        <p className="text-sm font-semibold text-slate-800 mt-1">
                          {uploadResult.paper_reused ? 'Yes (Cached)' : 'No (New)'}
                        </p>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-xl border border-slate-200/60">
                        <p className="text-[11px] text-slate-500 font-medium">Retrieval Status</p>
                        <p className="text-sm font-semibold text-emerald-700 mt-1">
                          Refreshed & Ready
                        </p>
                      </div>
                    </div>

                    {/* Next Steps Actions */}
                    <div className="flex flex-wrap items-center gap-3 pt-2">
                      <button
                        type="button"
                        onClick={() => handleStartChatWithPaper(uploadResult)}
                        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-xs text-white bg-indigo-600 hover:bg-indigo-700 shadow-xs hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all cursor-pointer"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                        <span>Start Research Chat</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          setActiveTab('catalog');
                          fetchPapers(searchQuery);
                        }}
                        className="px-4 py-2.5 rounded-xl font-medium text-xs text-indigo-700 hover:text-indigo-900 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 transition-colors cursor-pointer"
                      >
                        View in Catalog ({papers.length})
                      </button>

                      <button
                        type="button"
                        onClick={handleResetForAnotherUpload}
                        className="px-4 py-2.5 rounded-xl font-medium text-xs text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 border border-slate-200 transition-colors cursor-pointer"
                      >
                        Upload Another Paper
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Ingested Papers in Session */}
            {sessionPapers.length > 0 && (
              <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-2xs space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
                    Papers Ingested This Session ({sessionPapers.length})
                  </h2>
                </div>

                <div className="divide-y divide-slate-100">
                  {sessionPapers.map((paper, idx) => (
                    <div key={idx} className="py-3 flex items-center justify-between gap-4">
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="h-8 w-8 rounded-lg bg-slate-100 text-slate-600 flex items-center justify-center shrink-0">
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </div>
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-slate-800 truncate">
                            {paper.title || paper.filename}
                          </p>
                          <p className="text-[11px] text-slate-400 font-mono">
                            {paper.title && paper.title !== paper.filename ? `${paper.filename} • ` : ''}
                            {formatFileSize(paper.size)} &bull; {paper.chunksCreated} chunks &bull; {paper.embeddingsCreated} embeddings
                          </p>
                        </div>
                      </div>

                      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 shrink-0">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                        <span>Indexed</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Paper Details Modal */}
      <PaperDetailModal
        isOpen={Boolean(selectedPaperId)}
        paperId={selectedPaperId}
        onClose={() => setSelectedPaperId(null)}
        onStartChat={handleStartChatWithPaper}
        onDelete={handleInitiateDelete}
      />

      {/* Delete Confirmation Modal */}
      {paperToDelete && (
        <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4 sm:p-6 animate-fade-in">
          <div
            className="fixed inset-0"
            onClick={() => !isDeletingPaper && setPaperToDelete(null)}
            aria-hidden="true"
          />

          <div className="relative bg-white rounded-2xl shadow-xl border border-slate-200/80 w-full max-w-lg overflow-hidden flex flex-col z-10 animate-scale-up">
            {/* Dialog Header */}
            <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-xl bg-rose-50 border border-rose-100 flex items-center justify-center text-rose-600 shrink-0">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900">Delete Research Paper</h3>
                  <p className="text-xs text-slate-500">Confirm permanent deletion from library</p>
                </div>
              </div>

              <button
                type="button"
                disabled={isDeletingPaper}
                onClick={() => setPaperToDelete(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer disabled:opacity-50"
                aria-label="Close confirmation dialog"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Dialog Body */}
            <div className="p-6 space-y-4">
              {deleteError && (
                <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-700 flex items-center justify-between">
                  <span>{deleteError}</span>
                  <button
                    type="button"
                    onClick={() => setDeleteError(null)}
                    className="text-rose-500 hover:text-rose-700 font-bold px-1"
                  >
                    ×
                  </button>
                </div>
              )}

              <div className="p-3 bg-slate-50 border border-slate-200/80 rounded-xl">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                  Paper Title
                </span>
                <p className="text-xs font-semibold text-slate-900 leading-snug">
                  {paperToDelete.title}
                </p>
                {paperToDelete.original_filename && (
                  <p className="text-[11px] text-slate-500 font-mono mt-1">
                    File: {paperToDelete.original_filename}
                  </p>
                )}
              </div>

              <div className="text-xs text-slate-600 space-y-2 leading-relaxed">
                <p>
                  This operation will <strong className="text-rose-700">permanently delete</strong> this research paper, its parsed evidence chunks, vector embeddings, and associated Cloudinary PDF storage.
                </p>
                <p className="text-slate-500">
                  Subsequent research questions and search queries will no longer be able to retrieve evidence from this paper. This action cannot be undone.
                </p>
              </div>

              {['pending', 'processing'].includes(paperToDelete.ingestion_status?.toLowerCase()) && (
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-800 flex items-start gap-2">
                  <svg className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <div>
                    <span className="font-semibold block">Ingestion In Progress</span>
                    <span>This paper is actively undergoing ingestion. Deletion is disabled until ingestion concludes.</span>
                  </div>
                </div>
              )}
            </div>

            {/* Dialog Footer */}
            <div className="px-6 py-4 border-t border-slate-100 bg-slate-50/50 flex items-center justify-end gap-3">
              <button
                type="button"
                disabled={isDeletingPaper}
                onClick={() => setPaperToDelete(null)}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:text-slate-800 hover:bg-slate-200/60 transition-colors cursor-pointer disabled:opacity-50"
              >
                Cancel
              </button>

              <button
                type="button"
                disabled={isDeletingPaper || ['pending', 'processing'].includes(paperToDelete.ingestion_status?.toLowerCase())}
                onClick={handleConfirmDelete}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold text-white bg-rose-600 hover:bg-rose-700 shadow-xs transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isDeletingPaper ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Deleting...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    <span>Permanently Delete</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
