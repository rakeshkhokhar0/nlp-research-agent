import React, { useEffect, useRef } from 'react';
import { useConversation } from '../../context/ConversationContext';
import MessageBubble from './MessageBubble';
import WelcomeScreen from './WelcomeScreen';
import TypingIndicator from './TypingIndicator';

/**
 * Scrollable message viewport rendering message history, welcome screen,
 * typing indicator, and actionable error alerts.
 */
export default function ChatWindow({ onNavigateToLibrary }) {
  const { messages, isLoading, isHistoryLoading, error, clearError, reloadHistory } = useConversation();
  const messagesEndRef = useRef(null);

  // Auto-scroll to the bottom when messages update or assistant begins typing
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 bg-slate-50 flex flex-col">
      {/* Error Banner */}
      {error && (
        <div className="mb-4 p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start justify-between gap-3 shadow-xs shrink-0 max-w-4xl mx-auto w-full">
          <div className="flex items-start gap-2.5">
            <svg className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <p className="font-semibold text-rose-900 mb-0.5">Communication Error</p>
              <p className="text-rose-700 leading-relaxed">{error}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {reloadHistory && (
              <button
                type="button"
                onClick={reloadHistory}
                className="px-2 py-1 rounded bg-rose-100 hover:bg-rose-200 text-rose-800 font-medium transition-colors cursor-pointer"
              >
                Retry
              </button>
            )}
            <button
              type="button"
              onClick={clearError}
              aria-label="Dismiss error"
              className="text-rose-500 hover:text-rose-700 p-1 cursor-pointer"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* History Loading Spinner */}
      {isHistoryLoading ? (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-400 gap-2">
          <svg className="animate-spin h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
          </svg>
          <p className="text-xs font-medium text-slate-500">Loading conversation history...</p>
        </div>
      ) : messages.length === 0 && !isLoading ? (
        /* Empty State / Welcome Screen */
        <WelcomeScreen onNavigateToUpload={onNavigateToLibrary} />
      ) : (
        /* Message Stream */
        <div className="max-w-4xl mx-auto w-full">
          {messages.map((message) => (
            <MessageBubble key={message.id || message.sequence_number} message={message} />
          ))}

          {/* Typing Indicator while generation request is pending */}
          {isLoading && <TypingIndicator />}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      )}
    </div>
  );
}
