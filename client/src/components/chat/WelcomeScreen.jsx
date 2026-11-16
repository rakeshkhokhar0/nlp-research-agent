import React from 'react';
import { useConversation } from '../../context/ConversationContext';

const SUGGESTED_PROMPTS = [
  {
    title: 'Main Contribution',
    prompt: 'What is the main contribution of this paper?',
    icon: 'M13 10V3L4 14h7v7l9-11h-7z',
  },
  {
    title: 'Methodology & Architecture',
    prompt: 'Explain the methodology and model architecture used in this research.',
    icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10',
  },
  {
    title: 'Key Findings & Results',
    prompt: 'What are the key findings and quantitative results reported by the authors?',
    icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
  },
  {
    title: 'Limitations & Future Work',
    prompt: 'What limitations and future research directions do the authors mention?',
    icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
  },
];

/**
 * Welcome screen shown when a conversation has no messages yet.
 * Suggested prompts populate the input without auto-submitting.
 */
export default function WelcomeScreen({ onNavigateToUpload }) {
  const { setInputPrompt } = useConversation();

  const handleSelectPrompt = (promptText) => {
    setInputPrompt(promptText);
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-2xl mx-auto my-auto">
      {/* Icon */}
      <div className="h-16 w-16 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 mb-6 shadow-xs">
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.75}
            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
          />
        </svg>
      </div>

      {/* Title & Description */}
      <h2 className="text-xl font-bold text-slate-900 mb-2">
        NLP Research Agent
      </h2>
      <p className="text-sm text-slate-600 mb-6 max-w-lg leading-relaxed">
        Ask detailed questions about your available research papers. The Agent synthesizes answers grounded exclusively in verified research evidence.
      </p>

      {/* Quick Upload Action */}
      {onNavigateToUpload && (
        <button
          type="button"
          onClick={onNavigateToUpload}
          className="mb-8 inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200/80 shadow-2xs transition-colors cursor-pointer"
        >
          <svg className="w-4 h-4 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
          <span>Upload Academic Paper (PDF) to Library</span>
        </button>
      )}

      {/* Suggested Prompts Grid */}
      <div className="w-full text-left">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Suggested Question Prompts
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {SUGGESTED_PROMPTS.map((item, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleSelectPrompt(item.prompt)}
              className="p-3.5 rounded-xl bg-white border border-slate-200 hover:border-indigo-300 hover:bg-indigo-50/30 transition-all text-left shadow-2xs group cursor-pointer"
            >
              <div className="flex items-center gap-2 mb-1">
                <svg className="w-4 h-4 text-indigo-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.icon} />
                </svg>
                <span className="text-xs font-semibold text-slate-800 group-hover:text-indigo-600 transition-colors">
                  {item.title}
                </span>
              </div>
              <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed">
                {item.prompt}
              </p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
