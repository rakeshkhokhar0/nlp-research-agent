import React, { useState, useEffect } from 'react';
import { useConversation } from '../../context/ConversationContext';
import { checkHealth } from '../../api/client';

/**
 * Top header for the research workspace displaying active conversation info, mode, and library actions.
 *
 * @param {object} props
 * @param {Function} props.onToggleSidebar - Mobile sidebar toggle
 * @param {string} [props.activeView='chat'] - Currently active view ('chat' | 'library')
 * @param {Function} [props.onSelectView] - Callback to switch views
 */
export default function ChatHeader({ onToggleSidebar, activeView = 'chat', onSelectView }) {
  const { activeConversation, startNewConversation, isLoading, researchMode, chatType } = useConversation();
  const [backendOnline, setBackendOnline] = useState(null);

  useEffect(() => {
    let isMounted = true;
    checkHealth(3000)
      .then(() => {
        if (isMounted) setBackendOnline(true);
      })
      .catch(() => {
        if (isMounted) setBackendOnline(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const isLibraryView = activeView === 'library';

  return (
    <header className="h-16 bg-white border-b border-slate-200 px-4 sm:px-6 flex items-center justify-between shrink-0 shadow-2xs">
      <div className="flex items-center gap-3 min-w-0">
        {/* Mobile Sidebar Toggle Button */}
        <button
          type="button"
          onClick={onToggleSidebar}
          aria-label="Toggle sidebar"
          className="lg:hidden p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 focus:outline-none cursor-pointer"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        {/* View / Conversation Title */}
        <div className="min-w-0">
          {isLibraryView ? (
            <>
              <h2 className="text-sm sm:text-base font-semibold text-slate-900 truncate">
                Research Library & Document Ingestion
              </h2>
              <p className="text-xs text-slate-500 truncate hidden sm:block">
                Upload and index research papers into the shared lexical and semantic retrieval engine
              </p>
            </>
          ) : (
            <>
              <h2 className="text-sm sm:text-base font-semibold text-slate-900 truncate">
                {activeConversation?.title || 'New Research Conversation'}
              </h2>
              <p className="text-xs text-slate-500 truncate hidden sm:block">
                {activeConversation?.id
                  ? `Session ID: ${activeConversation.id}`
                  : 'Ask a question to start an active research session with the Agent'}
              </p>
            </>
          )}
        </div>
      </div>

      {/* Right Action & Status Area */}
      <div className="flex items-center gap-3 shrink-0">
        {/* Active Chat Type and Research Mode Badge (only in chat view) */}
        {!isLibraryView && (
          <div className="hidden md:flex items-center gap-2">
            <div className="flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border border-slate-200 bg-slate-50">
              <span
                className={`h-2 w-2 rounded-full ${
                  chatType === 'general' ? 'bg-emerald-500' : 'bg-indigo-600'
                }`}
              />
              <span className="text-slate-600">
                Type: <strong className="capitalize text-slate-800">{chatType === 'general' ? 'General' : 'Paper'}</strong>
              </span>
            </div>

            {chatType === 'paper' && (
              <div className="flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border border-slate-200 bg-slate-50">
                <span
                  className={`h-2 w-2 rounded-full ${
                    researchMode === 'advanced' ? 'bg-indigo-600' : 'bg-slate-400'
                  }`}
                />
                <span className="text-slate-600">
                  Mode: <strong className="capitalize text-slate-800">{researchMode}</strong>
                </span>
              </div>
            )}
          </div>
        )}

        {/* Backend Status Badge */}
        <div className="hidden sm:flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border border-slate-200 bg-slate-50">
          <span
            className={`h-2 w-2 rounded-full ${
              backendOnline === true
                ? 'bg-emerald-500'
                : backendOnline === false
                ? 'bg-amber-500'
                : 'bg-slate-300 animate-pulse'
            }`}
          />
          <span className="text-slate-600">
            {backendOnline === true
              ? 'Backend Online'
              : backendOnline === false
              ? 'Backend Offline'
              : 'Connecting...'}
          </span>
        </div>

        {/* View Switch / Action Buttons */}
        {isLibraryView ? (
          <button
            type="button"
            onClick={() => onSelectView && onSelectView('chat')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 focus:outline-none transition-colors cursor-pointer"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <span>Research Chat</span>
          </button>
        ) : (
          <>
            {/* Upload Paper Button */}
            <button
              type="button"
              onClick={() => onSelectView && onSelectView('library')}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-700 hover:text-indigo-600 bg-slate-100 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 transition-colors cursor-pointer"
              title="Upload academic paper PDF to library"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              <span className="hidden sm:inline">Upload Paper</span>
            </button>

            {/* New Chat Button */}
            <button
              type="button"
              onClick={() => startNewConversation()}
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 focus:outline-none transition-colors cursor-pointer disabled:opacity-50"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              <span className="hidden sm:inline">New Chat</span>
            </button>
          </>
        )}
      </div>
    </header>
  );
}
