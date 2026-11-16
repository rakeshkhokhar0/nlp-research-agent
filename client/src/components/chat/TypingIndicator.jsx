import React from 'react';

/**
 * Animated typing indicator displayed while waiting for Agent response.
 */
export default function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 my-4">
      {/* Agent Avatar */}
      <div className="h-8 px-2 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-bold text-xs shrink-0 shadow-2xs min-w-14">
        Agent
      </div>

      <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-xs max-w-md">
        <div className="flex items-center space-x-1.5 mb-1">
          <div className="h-2 w-2 rounded-full bg-indigo-500 animate-bounce"></div>
          <div className="h-2 w-2 rounded-full bg-indigo-500 animate-bounce [animation-delay:0.2s]"></div>
          <div className="h-2 w-2 rounded-full bg-indigo-500 animate-bounce [animation-delay:0.4s]"></div>
        </div>
        <p className="text-xs text-slate-500 font-medium">
          Agent is analyzing research papers &amp; synthesizing answer...
        </p>
      </div>
    </div>
  );
}
