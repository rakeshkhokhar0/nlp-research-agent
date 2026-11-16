import React from 'react';
import { ConversationProvider } from '../context/ConversationContext';
import ChatLayout from '../components/layout/ChatLayout';

/**
 * Chat page view wrapping the research workspace in ConversationProvider.
 */
export default function ChatPage() {
  return (
    <ConversationProvider>
      <ChatLayout />
    </ConversationProvider>
  );
}
