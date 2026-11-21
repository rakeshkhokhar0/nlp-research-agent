import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import * as conversationApi from '../api/conversationApi';

const ACTIVE_CONV_KEY = 'nlp_research_active_conv_id';

const ConversationContext = createContext(null);

export function ConversationProvider({ children }) {
  const [conversations, setConversations] = useState([]);
  const [isConversationsLoading, setIsConversationsLoading] = useState(true);
  const [conversationsError, setConversationsError] = useState(null);

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
  const [deletingConversationIds, setDeletingConversationIds] = useState(() => new Set());

  // Ref to prevent double-execution in StrictMode or rapid clicks
  const isCreatingRef = useRef(false);
  const deletingRef = useRef(new Set());

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

  // Fetch conversation list from backend
  const loadConversations = useCallback(async (preferredActiveId = null) => {
    setIsConversationsLoading(true);
    setConversationsError(null);

    try {
      const list = await conversationApi.listConversations();
      const mapped = list.map((c) => ({
        id: c.id,
        title: c.title || (c.chat_type === 'general' ? 'New General Chat' : 'New Research Chat'),
        chatType: c.chat_type || 'paper',
        status: c.status,
        createdAt: c.created_at,
        lastMessageAt: c.last_message_at || c.created_at,
      }));

      // Sort by most recent activity descending
      mapped.sort(
        (a, b) =>
          new Date(b.lastMessageAt || b.createdAt) -
          new Date(a.lastMessageAt || a.createdAt)
      );

      setConversations(mapped);

      // Determine active conversation:
      // candidate: preferredActiveId || activeConversationId || localStorage
      let candidateId = preferredActiveId;
      if (!candidateId) {
        try {
          candidateId = localStorage.getItem(ACTIVE_CONV_KEY);
        } catch {
          candidateId = null;
        }
      }
      if (!candidateId && activeConversationId) {
        candidateId = activeConversationId;
      }

      if (candidateId && mapped.some((c) => c.id === candidateId)) {
        setActiveConversationId(candidateId);
        const target = mapped.find((c) => c.id === candidateId);
        if (target?.chatType) {
          setChatType(target.chatType);
        }
      } else if (mapped.length > 0) {
        // Fresh browser profile or invalid stored ID: default to the most recently active conversation
        setActiveConversationId(mapped[0].id);
        if (mapped[0].chatType) {
          setChatType(mapped[0].chatType);
        }
      } else {
        setActiveConversationId(null);
        setMessages([]);
      }
      return mapped;
    } catch (err) {
      console.error('Failed to load conversations from backend:', err);
      setConversationsError(err.message || 'Failed to load conversations from server.');
      return [];
    } finally {
      setIsConversationsLoading(false);
    }
  }, [activeConversationId]);

  // Load conversations on initial mount
  useEffect(() => {
    loadConversations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch history whenever activeConversationId changes
  useEffect(() => {
    if (activeConversationId) {
      loadConversationHistory(activeConversationId);
    } else {
      setMessages([]);
    }
  }, [activeConversationId, loadConversationHistory]);

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
   * Permanently deletes a conversation from the backend and updates local state.
   */
  const deleteConversation = useCallback(
    async (convId) => {
      if (!convId) return;
      if (deletingRef.current.has(convId)) {
        // Prevent duplicate delete requests while one is in progress
        return;
      }

      deletingRef.current.add(convId);
      setDeletingConversationIds((prev) => new Set(prev).add(convId));
      setError(null);

      try {
        await conversationApi.deleteConversation(convId);

        // Remove from conversation list upon successful confirmation
        setConversations((prev) => prev.filter((c) => c.id !== convId));

        // If the deleted conversation was active, clear active state and return to new-chat state
        if (activeConversationId === convId) {
          setActiveConversationId(null);
          setMessages([]);
        }
      } catch (err) {
        const errorMsg = err.message || 'Failed to delete conversation from server.';
        setError(errorMsg);
        throw err;
      } finally {
        deletingRef.current.delete(convId);
        setDeletingConversationIds((prev) => {
          const next = new Set(prev);
          next.delete(convId);
          return next;
        });
      }
    },
    [activeConversationId]
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

      if (!convId) {
        try {
          const initialTitle = trimmed.length > 45 ? `${trimmed.slice(0, 42)}...` : trimmed;
          convId = await startNewConversation(initialTitle);
        } catch (err) {
          setError(`Could not initialize conversation: ${err.message}`);
          return;
        }
      }

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

      try {
        const chatResponse = await conversationApi.sendMessage(convId, {
          question: trimmed,
          top_k: 5,
          mode: researchMode,
          chat_type: chatType,
        });

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

        setMessages((prev) => [
          ...prev.map((m) =>
            m.id === tempUserMsgId
              ? { ...m, id: chatResponse.user_message_id, chat_type: chatType }
              : m
          ),
          assistantMessage,
        ]);

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
        const errorMessage =
          err.message || 'An error occurred while generating the answer.';
        setError(errorMessage);

        setInputPrompt(trimmed);

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
    isConversationsLoading,
    conversationsError,
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
    deleteConversation,
    removeConversation: deleteConversation,
    deletingConversationIds,
    askQuestion,
    clearError: () => setError(null),
    reloadConversations: () => loadConversations(activeConversationId),
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
