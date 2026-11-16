import React, { useState } from 'react';
import Sidebar from './Sidebar';
import ChatHeader from '../chat/ChatHeader';
import ChatWindow from '../chat/ChatWindow';
import MessageInput from '../chat/MessageInput';
import ResearchLibrary from '../library/ResearchLibrary';

/**
 * Main chat layout combining responsive sidebar, header, research workspace, and library document uploader.
 */
export default function ChatLayout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeView, setActiveView] = useState('chat'); // 'chat' | 'library'

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50 font-sans text-slate-800 antialiased">
      {/* Sidebar with mobile drawer support and view switching */}
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeView={activeView}
        onSelectView={setActiveView}
      />

      {/* Main Workspace Interface */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        <ChatHeader
          onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
          activeView={activeView}
          onSelectView={setActiveView}
        />

        {activeView === 'library' ? (
          <ResearchLibrary onNavigateToChat={() => setActiveView('chat')} />
        ) : (
          <>
            <ChatWindow onNavigateToLibrary={() => setActiveView('library')} />
            <MessageInput />
          </>
        )}
      </div>
    </div>
  );
}
