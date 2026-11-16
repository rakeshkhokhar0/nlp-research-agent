import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import * as conversationApi from '../api/conversationApi';

const STORAGE_KEY = 'nlp_research_conversations';
const ACTIVE_CONV_KEY = 'nlp_research_active_conv_id';

const ConversationContext = createContext(null);

export function ConversationProvider({ children }) {
  const [conversations, setConversations] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  const [activeConversationId, setActiveConversationId] = useState(() => {
    try {
      return localStorage.getItem(ACTIVE_CONV_KEY) || null;
    } catch {
      return null;
    }
  });

  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inputPrompt, setInputPrompt] = useState('');
  const [researchMode, setResearchMode] = useState('normal');
  const [chatType, setChatType] = useState('paper'); // 'paper' | 'general'

  // Ref to prevent double-execution in StrictMode or rapid clicks
  const isCreatingRef = useRef(false);

  // Sync conversations to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (err) {
      console.error('Failed to persist conversations to localStorage:', err);
    }
  }, [conversations]);

  // Sync activeConversationId to localStorage
  useEffect(() => {
    try {
      if (activeConversationId) {
        localStorage.setItem(ACTIVE_CONV_KEY, activeConversationId);
      } else {
        localStorage.removeItem(ACTIVE_CONV_KEY);
      }
    } catch (err) {
      console.error('Failed to persist active conversation ID:', err);
    }
  }, [activeConversationId]);

  // Load message history when active conversation changes
  const loadConversationHistory = useCallback(async (convId) => {
    if (!convId) {
      setMessages([]);
      return;
    }

    setIsHistoryLoading(true);
    setError(null);

    try {
      const history = await conversationApi.getConversationHistory(convId);
      setMessages(history);

      // Also sync conversation metadata to determine established chat_type
      try {
        const convDetails = await conversationApi.getConversation(convId);
        if (convDetails?.chat_type) {
          setChatType(convDetails.chat_type);
          setConversations((prev) =>
            prev.map((c) =>
              c.id === convId ? { ...c, chatType: convDetails.chat_type } : c
            )
          );
        }
      } catch {
        // Fallback silently if getConversation details fails
      }
    } catch (err) {
      const isNotFound = err.message?.includes('not found') || err.message?.includes('404');
      if (isNotFound) {
        setError(`Conversation not found on server. It may have expired or been cleared.`);
      } else {
        setError(err.message || 'Failed to load conversation history from server.');
      }
      setMessages([]);
    } finally {
      setIsHistoryLoading(false);
    }
  }, []);

  // Fetch history on initial mount if an active conversation exists
  useEffect(() => {
    if (activeConversationId) {
      loadConversationHistory(activeConversationId);
    } else if (conversations.length > 0) {
      // If active conversation is not set but conversations exist, activate the first one
      setActiveConversationId(conversations[0].id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeConversationId]);

  /**
   * Starts a new conversation session via backend POST /conversations.
   *
   * @param {string|null} [title=null]
   * @param {string|null} [type=null]
   * @returns {Promise<string>} The new conversation ID
   */
  const startNewConversation = useCallback(async (title = null, type = null) => {
    if (isCreatingRef.current) return null;
    isCreatingRef.current = true;
    setIsLoading(true);
    setError(null);

    const effectiveType = type || chatType || 'paper';
    const defaultTitle = effectiveType === 'general' ? 'New General Chat' : 'New Research Chat';

    try {
      const conv = await conversationApi.createConversation(title || defaultTitle, {
        chat_type: effectiveType,
      });
      const newEntry = {
        id: conv.id,
        title: conv.title || defaultTitle,
        chatType: conv.chat_type || effectiveType,
        createdAt: conv.created_at,
        lastMessageAt: conv.last_message_at || conv.created_at,
      };

      setConversations((prev) => [newEntry, ...prev.filter((c) => c.id !== conv.id)]);
      setActiveConversationId(conv.id);
      setChatType(conv.chat_type || effectiveType);
      setMessages([]);
      return conv.id;
    } catch (err) {
      const msg = err.message || 'Failed to create conversation on backend.';
      setError(msg);
      throw err;
    } finally {
      setIsLoading(false);
      isCreatingRef.current = false;
    }
  }, [chatType]);

  /**
   * Selects an existing conversation and loads its history.
   */
  const selectConversation = useCallback(
    (convId) => {
      if (convId === activeConversationId) return;
      setActiveConversationId(convId);
      setError(null);
      const existing = conversations.find((c) => c.id === convId);
      if (existing?.chatType) {
        setChatType(existing.chatType);
      }
    },
    [activeConversationId, conversations]
  );

  /**
   * Removes a conversation entry from local tracking.
   */
  const removeConversation = useCallback(
    (convId) => {
      setConversations((prev) => prev.filter((c) => c.id !== convId));
      if (activeConversationId === convId) {
        const remaining = conversations.filter((c) => c.id !== convId);
        setActiveConversationId(remaining.length > 0 ? remaining[0].id : null);
        setMessages([]);
      }
    },
    [activeConversationId, conversations]
  );

  /**
   * Submits a question to the active conversation.
   * If no active conversation exists, creates one first.
   */
  const askQuestion = useCallback(
    async (questionText) => {
      const trimmed = questionText?.trim();
      if (!trimmed || isLoading) return;

      setError(null);

      let convId = activeConversationId;

      // 1. If no conversation is active, create one now
      if (!convId) {
        try {
          // Generate a friendly initial title based on the question
          const initialTitle = trimmed.length > 45 ? `${trimmed.slice(0, 42)}...` : trimmed;
          convId = await startNewConversation(initialTitle);
        } catch (err) {
          setError(`Could not initialize conversation: ${err.message}`);
          return;
        }
      }

      // 2. Optimistic user message rendering
      const tempUserMsgId = `temp-${Date.now()}`;
      const tempUserMessage = {
        id: tempUserMsgId,
        conversation_id: convId,
        sequence_number: messages.length + 1,
        role: 'user',
        content: trimmed,
        status: 'completed',
        created_at: new Date().toISOString(),
        citations: [],
      };

      setMessages((prev) => [...prev, tempUserMessage]);
      setIsLoading(true);

      // 3. Send message to backend
      try {
        const chatResponse = await conversationApi.sendMessage(convId, {
          question: trimmed,
          top_k: 5,
          mode: researchMode,
          chat_type: chatType,
        });

        // 4. Create assistant message object from real ChatResponse
        const assistantMessage = {
          id: chatResponse.assistant_message_id,
          conversation_id: chatResponse.conversation_id,
          sequence_number: messages.length + 2,
          role: 'assistant',
          content: chatResponse.answer,
          status: 'completed',
          model_name: chatResponse.model_name,
          chat_type: chatResponse.chat_type || chatType,
          citations: chatResponse.citations || [],
          created_at: chatResponse.created_at,
        };

        // Update the temporary user message with real ID and append assistant answer
        setMessages((prev) => [
          ...prev.map((m) =>
            m.id === tempUserMsgId
              ? { ...m, id: chatResponse.user_message_id, chat_type: chatType }
              : m
          ),
          assistantMessage,
        ]);

        // 5. Update local conversation title & timestamp if it was a default title
        setConversations((prev) =>
          prev.map((c) => {
            if (c.id === convId) {
              const shouldUpdateTitle =
                c.title === 'New Chat' ||
                c.title === 'New General Chat' ||
                c.title === 'New Research Chat';
              const derivedTitle =
                trimmed.length > 45 ? `${trimmed.slice(0, 42)}...` : trimmed;
              return {
                ...c,
                title: shouldUpdateTitle ? derivedTitle : c.title,
                chatType: chatResponse.chat_type || chatType,
                lastMessageAt: chatResponse.created_at || new Date().toISOString(),
              };
            }
            return c;
          })
        );
      } catch (err) {
        // Keep user message visible and show actionable error
        const errorMessage =
          err.message || 'An error occurred while generating the answer.';
        setError(errorMessage);

        // Preserve question in inputPrompt so user does not lose their draft
        setInputPrompt(trimmed);

        // Mark temporary user message as failed or leave visible with error state
        setMessages((prev) =>
          prev.map((m) =>
            m.id === tempUserMsgId ? { ...m, status: 'failed' } : m
          )
        );
      } finally {
        setIsLoading(false);
      }
    },
    [activeConversationId, isLoading, messages.length, researchMode, setInputPrompt, startNewConversation]
  );

  const activeConversation = conversations.find((c) => c.id === activeConversationId) || null;

  const value = {
    conversations,
    activeConversationId,
    activeConversation,
    messages,
    isLoading,
    isHistoryLoading,
    error,
    inputPrompt,
    setInputPrompt,
    researchMode,
    setResearchMode,
    chatType,
    setChatType,
    startNewConversation,
    selectConversation,
    removeConversation,
    askQuestion,
    clearError: () => setError(null),
    reloadHistory: () => activeConversationId && loadConversationHistory(activeConversationId),
  };

  return (
    <ConversationContext.Provider value={value}>
      {children}
    </ConversationContext.Provider>
  );
}

export function useConversation() {
  const context = useContext(ConversationContext);
  if (!context) {
    throw new Error('useConversation must be used within a ConversationProvider');
  }
  return context;
}
