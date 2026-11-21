import React from 'react';
import { useConversation } from '../../context/ConversationContext';

/**
 * Formats timestamps for display in sidebar.
 */
function formatDateLabel(isoString) {
  if (!isoString) return '';
  try {
    const d = new Date(isoString);
    const now = new Date();
    const diffDays = Math.floor((now - d) / (1000 * 60 * 60 * 24));
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
}

/**
 * Sidebar containing brand header, navigation tabs, New Chat action, and recent conversation sessions.
 */
export default function Sidebar({ isOpen, onClose, activeView = 'chat', onSelectView }) {
  const {
    conversations,
    isConversationsLoading,
    conversationsError,
    reloadConversations,
    activeConversationId,
    selectConversation,
    startNewConversation,
    deleteConversation,
    removeConversation,
    deletingConversationIds,
    isLoading,
  } = useConversation();

  const [deleteError, setDeleteError] = React.useState(null);

  const handleDelete = async (e, convId) => {
    e.stopPropagation();
    if (deletingConversationIds?.has(convId)) return;

    setDeleteError(null);
    try {
      const fn = deleteConversation || removeConversation;
      await fn(convId);
    } catch (err) {
      setDeleteError(err.message || 'Failed to delete conversation.');
    }
  };

  const handleSelect = (convId) => {
    if (onSelectView) onSelectView('chat');
    selectConversation(convId);
    if (onClose) onClose();
  };

  const handleNewChat = () => {
    if (onSelectView) onSelectView('chat');
    startNewConversation();
    if (onClose) onClose();
  };

  const handleNavClick = (view) => {
    if (onSelectView) onSelectView(view);
    if (onClose) onClose();
  };

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-30 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed lg:static inset-y-0 left-0 z-40 w-72 bg-white border-r border-slate-200 flex flex-col transition-transform duration-200 ease-in-out shrink-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Brand Header */}
        <div className="h-16 px-5 border-b border-slate-200 flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-3">
            <div className="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-sm shadow-xs">
              NA
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-900 leading-tight">
                NLP Research Agent
              </h1>
              <p className="text-[11px] text-slate-500">Academic Intelligence</p>
            </div>
          </div>

          {/* Close button on mobile */}
          <button
            type="button"
            onClick={onClose}
            className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100"
            aria-label="Close sidebar"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Navigation Mode Selector */}
        <div className="p-3 pb-0 shrink-0 space-y-1">
          <button
            type="button"
            onClick={() => handleNavClick('chat')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-semibold transition-colors cursor-pointer ${
              activeView === 'chat'
                ? 'bg-indigo-50 text-indigo-700 border border-indigo-200/80 shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <span>Research Chat</span>
          </button>

          <button
            type="button"
            onClick={() => handleNavClick('library')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-semibold transition-colors cursor-pointer ${
              activeView === 'library'
                ? 'bg-indigo-50 text-indigo-700 border border-indigo-200/80 shadow-2xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            <span>Paper Library & Upload</span>
          </button>
        </div>

        {/* New Chat Primary Button */}
        <div className="p-3 shrink-0">
          <button
            type="button"
            onClick={handleNewChat}
            disabled={isLoading}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl font-medium text-xs text-white bg-indigo-600 hover:bg-indigo-700 shadow-xs hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all cursor-pointer disabled:opacity-50"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New Chat</span>
          </button>
        </div>

        {/* Recent Conversations List */}
        <div className="flex-1 overflow-y-auto px-3 py-2">
          <div className="px-2 pb-2 text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
            <span>Recent Chats</span>
            <span className="font-mono text-slate-400">{conversations.length}</span>
          </div>

          {deleteError && (
            <div className="mx-1 mb-2 p-2 rounded-lg bg-rose-50 border border-rose-200 text-[11px] text-rose-700 flex items-center justify-between animate-fade-in">
              <span className="truncate pr-1">{deleteError}</span>
              <button
                type="button"
                onClick={() => setDeleteError(null)}
                className="text-rose-500 hover:text-rose-700 font-bold px-1 rounded cursor-pointer shrink-0"
                aria-label="Dismiss error"
              >
                ×
              </button>
            </div>
          )}

          {isConversationsLoading && conversations.length === 0 ? (
            /* Loading State */
            <div className="px-3 py-8 text-center text-slate-400">
              <div className="w-5 h-5 mx-auto mb-2 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
              <p className="text-xs font-medium text-slate-600">Loading conversations...</p>
            </div>
          ) : conversationsError && conversations.length === 0 ? (
            /* Error State */
            <div className="px-3 py-6 text-center text-slate-500">
              <svg className="w-6 h-6 mx-auto mb-2 text-rose-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <p className="text-xs font-medium text-slate-700 mb-1">Failed to load history</p>
              <p className="text-[11px] text-slate-400 mb-3 line-clamp-2">{conversationsError}</p>
              <button
                type="button"
                onClick={reloadConversations}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 transition-colors cursor-pointer"
              >
                Retry
              </button>
            </div>
          ) : conversations.length === 0 ? (
            /* Empty State */
            <div className="px-3 py-8 text-center text-slate-400">
              <svg className="w-8 h-8 mx-auto mb-2 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              <p className="text-xs font-medium text-slate-600">No conversations yet</p>
              <p className="text-[11px] text-slate-400 mt-0.5">Start a new chat to begin research.</p>
            </div>
          ) : (
            <div className="space-y-1">
              {conversations.map((conv) => {
                const isActive = conv.id === activeConversationId;
                return (
                  <div
                    key={conv.id}
                    className={`group relative flex items-center rounded-xl p-2.5 text-xs transition-colors cursor-pointer ${
                      isActive
                        ? 'bg-indigo-50/80 text-indigo-950 font-medium border border-indigo-200/80'
                        : 'text-slate-700 hover:bg-slate-100 hover:text-slate-900'
                    }`}
                    onClick={() => handleSelect(conv.id)}
                  >
                    <svg
                      className={`w-4 h-4 shrink-0 mr-2.5 ${isActive ? 'text-indigo-600' : 'text-slate-400 group-hover:text-slate-600'}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.75} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>

                    <div className="flex-1 min-w-0 pr-6">
                      <p className="truncate leading-snug">{conv.title || 'Untitled Chat'}</p>
                      <p className="text-[10px] text-slate-400 font-mono mt-0.5 flex items-center gap-1.5">
                        <span>{formatDateLabel(conv.lastMessageAt || conv.createdAt)}</span>
                        <span className="text-slate-300">&bull;</span>
                        <span className={conv.chatType === 'general' ? 'text-emerald-600 font-medium' : 'text-indigo-600 font-medium'}>
                          {conv.chatType === 'general' ? 'General' : 'Paper'}
                        </span>
                      </p>
                    </div>

                    {/* Delete conversation action */}
                    <button
                      type="button"
                      disabled={deletingConversationIds?.has(conv.id)}
                      onClick={(e) => handleDelete(e, conv.id)}
                      title={deletingConversationIds?.has(conv.id) ? "Deleting chat..." : "Delete chat"}
                      className={`absolute right-2 p-1 rounded text-slate-400 hover:text-rose-600 transition-opacity focus:opacity-100 cursor-pointer ${
                        deletingConversationIds?.has(conv.id)
                          ? 'opacity-100 cursor-not-allowed text-rose-500'
                          : 'opacity-0 group-hover:opacity-100'
                      }`}
                      aria-label="Delete chat"
                    >
                      {deletingConversationIds?.has(conv.id) ? (
                        <div className="w-3.5 h-3.5 border-2 border-rose-500 border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer Disclaimer */}
        <div className="p-3 border-t border-slate-100 bg-slate-50/50 text-[10px] text-slate-400 text-center">
          Grounded evidence synthesis &bull; Citations verified
        </div>
      </aside>
    </>
  );
}
