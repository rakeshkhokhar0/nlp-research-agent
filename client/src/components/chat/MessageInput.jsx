import React, { useState, useEffect, useRef } from 'react';
import { useConversation } from '../../context/ConversationContext';

/**
 * Chat input box with multiline textarea, auto-grow, Enter to send, and Shift+Enter for newline.
 */
export default function MessageInput() {
  const {
    askQuestion,
    isLoading,
    inputPrompt,
    setInputPrompt,
    researchMode,
    setResearchMode,
    chatType,
    setChatType,
    messages,
  } = useConversation();
  const [text, setText] = useState('');
  const textareaRef = useRef(null);

  // Sync when a suggested prompt was clicked
  useEffect(() => {
    if (inputPrompt) {
      setText(inputPrompt);
      setInputPrompt('');
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  }, [inputPrompt, setInputPrompt]);

  // Adjust textarea height automatically based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 160)}px`;
    }
  }, [text]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || isLoading) return;

    askQuestion(trimmed);
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const isSendDisabled = !text.trim() || isLoading;
  const isConversationLocked = messages && messages.length > 0;

  return (
    <div className="border-t border-slate-200 bg-white p-4">
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
        {/* Chat Type and Research Mode Selectors */}
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2 px-1">
          <div className="flex items-center gap-2">
            {/* Chat Type Selector */}
            <div className="flex items-center gap-1 p-0.5 bg-slate-100 border border-slate-200 rounded-xl">
              <button
                type="button"
                onClick={() => setChatType('paper')}
                disabled={isLoading || (isConversationLocked && chatType !== 'paper')}
                title={
                  isConversationLocked && chatType !== 'paper'
                    ? 'This conversation is locked to General Chat'
                    : 'Grounded question answering from indexed research papers'
                }
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  chatType === 'paper'
                    ? 'bg-white text-indigo-700 shadow-2xs font-bold'
                    : 'text-slate-600 hover:text-slate-900 disabled:opacity-40 disabled:cursor-not-allowed'
                }`}
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                Chat with Paper
              </button>

              <button
                type="button"
                onClick={() => setChatType('general')}
                disabled={isLoading || (isConversationLocked && chatType !== 'general')}
                title={
                  isConversationLocked && chatType !== 'general'
                    ? 'This conversation is locked to Chat with Paper'
                    : 'Conversational chat with context memory without research retrieval'
                }
                className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  chatType === 'general'
                    ? 'bg-white text-emerald-700 shadow-2xs font-bold'
                    : 'text-slate-600 hover:text-slate-900 disabled:opacity-40 disabled:cursor-not-allowed'
                }`}
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                General Chat
              </button>
            </div>

            {/* Research Mode Selector (Visible only when Chat with Paper is selected) */}
            {chatType === 'paper' && (
              <div className="flex items-center gap-1 p-0.5 bg-slate-100 border border-slate-200 rounded-xl">
                <button
                  type="button"
                  onClick={() => setResearchMode('normal')}
                  disabled={isLoading}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                    researchMode === 'normal'
                      ? 'bg-white text-indigo-700 shadow-2xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Normal
                </button>
                <button
                  type="button"
                  onClick={() => setResearchMode('advanced')}
                  disabled={isLoading}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                    researchMode === 'advanced'
                      ? 'bg-indigo-600 text-white shadow-2xs font-bold'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Advanced
                </button>
              </div>
            )}
          </div>

          {/* Context and Mode description */}
          <span className="text-[11px] text-slate-500 font-medium hidden sm:inline">
            {chatType === 'general'
              ? 'Context-aware conversation • No paper retrieval'
              : researchMode === 'advanced'
              ? 'Flexible in-depth research • Evidence-grounded explanations'
              : 'Direct grounded synthesis • Top-k retrieved chunks'}
          </span>
        </div>

        <div className="relative flex items-end bg-slate-50 border border-slate-300 rounded-2xl shadow-xs focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-500/20 transition-all p-2">
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder={
              chatType === 'general'
                ? 'Ask a general conceptual or follow-up question... (Shift + Enter for new line)'
                : researchMode === 'advanced'
                ? 'Ask the Agent an in-depth research question (Advanced mode analyzes claims & gaps)...'
                : 'Ask the Agent a research question about indexed papers... (Shift + Enter for new line)'
            }
            className="w-full resize-none bg-transparent px-3 py-2 text-sm text-slate-800 placeholder-slate-400 focus:outline-none max-h-40 disabled:opacity-50"
          />

          <button
            type="submit"
            disabled={isSendDisabled}
            aria-label="Send message"
            className={`shrink-0 h-9 w-9 rounded-xl flex items-center justify-center transition-all disabled:opacity-40 disabled:cursor-not-allowed text-white shadow-xs focus:outline-none cursor-pointer ${
              chatType === 'general'
                ? 'bg-emerald-600 hover:bg-emerald-700'
                : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
          >
            {isLoading ? (
              <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
              </svg>
            ) : (
              <svg className="w-4 h-4 transform rotate-90" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
              </svg>
            )}
          </button>
        </div>

        <div className="flex items-center justify-between mt-2 px-1 text-[11px] text-slate-400">
          <span>
            Press <kbd className="px-1 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-600 font-mono">Enter</kbd> to send, <kbd className="px-1 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-600 font-mono">Shift + Enter</kbd> for new line
          </span>
          <span>
            Type: <strong className="capitalize text-slate-600">{chatType === 'general' ? 'General Chat' : 'Chat with Paper'}</strong>
            {chatType === 'paper' && (
              <> &bull; Mode: <strong className="capitalize text-slate-600">{researchMode}</strong> &bull; top-k: 5</>
            )}
          </span>
        </div>
      </form>
    </div>
  );
}
