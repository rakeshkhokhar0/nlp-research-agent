import React from 'react';
import CitationCard from './CitationCard';
import { getUniqueCitations } from './citationUtils';

/**
 * Formats an ISO date string into readable local time.
 */
function formatTime(isoString) {
  if (!isoString) return '';
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

/**
 * Renders an individual chat message bubble (User or Agent).
 */
export default function MessageBubble({ message }) {
  if (!message) return null;

  const isUser = message.role === 'user';
  const isFailed = message.status === 'failed';
  const isGeneral = message.chat_type === 'general' || message.metadata?.chat_type === 'general';
  const uniqueCitations = !isUser && !isGeneral ? getUniqueCitations(message.citations) : [];

  return (
    <div className={`flex items-start gap-3 my-4 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar displaying Agent identity */}
      <div
        className={`h-8 px-2 rounded-lg flex items-center justify-center font-bold text-xs shrink-0 shadow-2xs ${
          isUser
            ? 'bg-slate-700 text-white min-w-8'
            : 'bg-indigo-600 text-white min-w-14'
        }`}
      >
        {isUser ? 'You' : 'Agent'}
      </div>

      {/* Bubble Content */}
      <div className={`flex flex-col max-w-2xl ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
            isUser
              ? isFailed
                ? 'bg-rose-50 text-rose-900 border border-rose-200 rounded-tr-sm'
                : 'bg-indigo-600 text-white rounded-tr-sm shadow-xs'
              : 'bg-white border border-slate-200 text-slate-800 rounded-tl-sm shadow-xs'
          }`}
        >
          {/* Main Message Text (Agent Response or User Message) */}
          <div className="whitespace-pre-wrap break-words">{message.content}</div>

          {/* Clean References Block: Paper Title and Page Number only */}
          {!isUser && uniqueCitations.length > 0 && (
            <div className="mt-3 pt-2.5 border-t border-slate-100">
              <div className="text-xs font-semibold text-slate-700 mb-1">References:</div>
              <ul className="space-y-0.5">
                {uniqueCitations.map((citation, index) => (
                  <CitationCard key={index} citation={citation} />
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Timestamp & Status Metadata */}
        <div className="flex items-center gap-2 mt-1 px-1 text-[11px] text-slate-400 font-mono">
          <span>{formatTime(message.created_at)}</span>
          {isFailed && (
            <span className="text-rose-600 font-sans font-medium flex items-center gap-1">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Delivery failed
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
