/**
 * API client methods for research conversations and grounded question-answering.
 * Interacts with FastAPI backend conversation endpoints.
 */

import apiClient from './client.js';

/**
 * Creates a new research conversation session on the backend.
 *
 * @param {string|null} [title=null] - Optional title for the conversation
 * @param {object|null} [metadata=null] - Optional metadata dictionary
 * @returns {Promise<import('./types').ConversationResponse>} Created conversation metadata
 */
export async function createConversation(title = null, metadata = null) {
  const payload = {};
  if (title) payload.title = title;
  if (metadata) payload.metadata = metadata;

  const response = await apiClient.post('/conversations', payload);
  return response.data;
}

/**
 * Retrieves all conversations from the backend ordered by most recent activity.
 *
 * @param {object} [options={}] - Query options
 * @param {number} [options.limit=50] - Maximum number of conversations to retrieve
 * @param {number} [options.offset=0] - Number of conversations to skip
 * @param {string|null} [options.status='active'] - Status filter ('active', 'archived')
 * @returns {Promise<Array<import('./types').ConversationResponse>>} List of conversations
 */
export async function listConversations({ limit = 50, offset = 0, status = 'active' } = {}) {
  const params = {
    limit: Math.max(1, Math.min(Number(limit) || 50, 100)),
    offset: Math.max(0, Number(offset) || 0),
  };
  if (status) {
    params.status = status;
  }
  const response = await apiClient.get('/conversations', { params });
  return Array.isArray(response.data) ? response.data : (response.data?.items || []);
}

/**
 * Retrieves metadata for a specific conversation by UUID.
 *
 * @param {string} conversationId - UUID of the conversation
 * @returns {Promise<import('./types').ConversationResponse>} Conversation metadata
 */
export async function getConversation(conversationId) {
  if (!conversationId) {
    throw new Error('conversationId is required');
  }
  const response = await apiClient.get(`/conversations/${conversationId}`);
  return response.data;
}

/**
 * Permanently deletes a conversation by UUID on the backend.
 *
 * @param {string} conversationId - UUID of the conversation
 * @returns {Promise<void>}
 */
export async function deleteConversation(conversationId) {
  if (!conversationId) {
    throw new Error('conversationId is required');
  }
  await apiClient.delete(`/conversations/${conversationId}`);
}

/**
 * Sends a research question within an active conversation and returns the grounded answer with citations.
 *
 * @param {string} conversationId - UUID of the conversation
 * @param {object} payload - Question payload
 * @param {string} payload.question - Research question text
 * @param {number} [payload.top_k=5] - Number of evidence chunks to retrieve (1..20)
 * @param {string[]|null} [payload.paper_ids=null] - Optional UUID array to filter papers
 * @param {string} [payload.mode='normal'] - Research answer mode ('normal' or 'advanced')
 * @param {string} [payload.chat_type='paper'] - Chat type ('general' or 'paper')
 * @returns {Promise<import('./types').ChatResponse>} Grounded answer and citations
 */
export async function sendMessage(conversationId, { question, top_k = 5, paper_ids = null, mode = 'normal', chat_type = null }) {
  if (!conversationId) {
    throw new Error('conversationId is required to send a message');
  }
  if (!question || !question.trim()) {
    throw new Error('Question cannot be empty');
  }

  const payload = {
    question: question.trim(),
    top_k: Number(top_k) || 5,
    mode: mode === 'advanced' ? 'advanced' : 'normal',
  };
  if (chat_type) {
    payload.chat_type = chat_type;
  }
  if (Array.isArray(paper_ids) && paper_ids.length > 0) {
    payload.paper_ids = paper_ids;
  }

  const response = await apiClient.post(`/conversations/${conversationId}/messages`, payload);
  return response.data;
}

/**
 * Retrieves the full chronological message history and citations for a conversation.
 *
 * @param {string} conversationId - UUID of the conversation
 * @returns {Promise<Array<import('./types').ConversationMessageResponse>>} Ordered list of messages
 */
export async function getConversationHistory(conversationId) {
  if (!conversationId) {
    throw new Error('conversationId is required to retrieve history');
  }
  const response = await apiClient.get(`/conversations/${conversationId}/messages`);
  return response.data;
}
