import test from 'node:test';
import assert from 'node:assert/strict';
import apiClient from './client.js';
import * as conversationApi from './conversationApi.js';

test('conversationApi - createConversation sends POST /conversations with title', async (t) => {
  const originalPost = apiClient.post;
  let interceptedUrl = null;
  let interceptedPayload = null;

  apiClient.post = async (url, payload) => {
    interceptedUrl = url;
    interceptedPayload = payload;
    return {
      data: {
        id: '12345678-1234-5678-1234-567812345678',
        title: payload.title || 'New Chat',
        status: 'active',
        created_at: '2026-11-15T00:00:00Z',
        updated_at: null,
        last_message_at: null,
      },
    };
  };

  t.after(() => {
    apiClient.post = originalPost;
  });

  const result = await conversationApi.createConversation('Transformer Research');

  assert.equal(interceptedUrl, '/conversations');
  assert.deepEqual(interceptedPayload, { title: 'Transformer Research' });
  assert.equal(result.id, '12345678-1234-5678-1234-567812345678');
  assert.equal(result.title, 'Transformer Research');
});

test('conversationApi - getConversation sends GET /conversations/{id}', async (t) => {
  const originalGet = apiClient.get;
  let interceptedUrl = null;

  apiClient.get = async (url) => {
    interceptedUrl = url;
    return {
      data: {
        id: 'conv-abc',
        title: 'Attention Study',
        status: 'active',
        created_at: '2026-11-15T00:00:00Z',
      },
    };
  };

  t.after(() => {
    apiClient.get = originalGet;
  });

  const result = await conversationApi.getConversation('conv-abc');

  assert.equal(interceptedUrl, '/conversations/conv-abc');
  assert.equal(result.id, 'conv-abc');
});

test('conversationApi - sendMessage sends POST /conversations/{id}/messages with valid payload', async (t) => {
  const originalPost = apiClient.post;
  let interceptedUrl = null;
  let interceptedPayload = null;

  apiClient.post = async (url, payload) => {
    interceptedUrl = url;
    interceptedPayload = payload;
    return {
      data: {
        conversation_id: 'conv-abc',
        user_message_id: 'user-msg-1',
        assistant_message_id: 'asst-msg-1',
        answer: 'Self-attention connects all positions with O(1) sequential operations.',
        citations: [
          {
            chunk_id: 'chunk-123',
            paper_id: 'paper-456',
            paper_title: 'Attention Is All You Need',
            citation_order: 1,
            page_number: 3,
            relevance_score: 4.85,
            quote_text: 'Self-attention allows connecting representations.',
          },
        ],
        model_name: 'gpt-4o-mini',
        created_at: '2026-11-15T00:01:00Z',
      },
    };
  };

  t.after(() => {
    apiClient.post = originalPost;
  });

  const result = await conversationApi.sendMessage('conv-abc', {
    question: 'How does self-attention work?',
    top_k: 5,
  });

  assert.equal(interceptedUrl, '/conversations/conv-abc/messages');
  assert.deepEqual(interceptedPayload, {
    question: 'How does self-attention work?',
    top_k: 5,
    mode: 'normal',
  });
  assert.equal(result.answer, 'Self-attention connects all positions with O(1) sequential operations.');
  assert.equal(result.citations.length, 1);
  assert.equal(result.citations[0].paper_title, 'Attention Is All You Need');
  assert.equal(result.citations[0].page_number, 3);
});

test('conversationApi - sendMessage forwards advanced mode in payload', async (t) => {
  const originalPost = apiClient.post;
  let interceptedPayload = null;

  apiClient.post = async (url, payload) => {
    interceptedPayload = payload;
    return {
      data: {
        conversation_id: 'conv-abc',
        user_message_id: 'user-msg-2',
        assistant_message_id: 'asst-msg-2',
        answer: 'Detailed cross-paper analysis.',
        citations: [],
        model_name: 'gpt-4o-mini',
        mode: 'advanced',
        created_at: '2026-11-15T00:02:00Z',
      },
    };
  };

  t.after(() => {
    apiClient.post = originalPost;
  });

  const result = await conversationApi.sendMessage('conv-abc', {
    question: 'Compare Transformer and RNN architectures',
    top_k: 5,
    mode: 'advanced',
  });

  assert.equal(interceptedPayload.mode, 'advanced');
  assert.equal(result.mode, 'advanced');
});

test('conversationApi - sendMessage rejects empty question without network call', async () => {
  await assert.rejects(
    async () => {
      await conversationApi.sendMessage('conv-abc', { question: '   ' });
    },
    {
      message: 'Question cannot be empty',
    }
  );
});

test('conversationApi - sendMessage rejects missing conversationId without network call', async () => {
  await assert.rejects(
    async () => {
      await conversationApi.sendMessage('', { question: 'Valid question?' });
    },
    {
      message: 'conversationId is required to send a message',
    }
  );
});

test('conversationApi - getConversationHistory sends GET /conversations/{id}/messages', async (t) => {
  const originalGet = apiClient.get;
  let interceptedUrl = null;

  apiClient.get = async (url) => {
    interceptedUrl = url;
    return {
      data: [
        {
          id: 'msg-1',
          conversation_id: 'conv-abc',
          sequence_number: 1,
          role: 'user',
          content: 'What is attention?',
          status: 'completed',
          citations: [],
          created_at: '2026-11-15T00:00:00Z',
        },
        {
          id: 'msg-2',
          conversation_id: 'conv-abc',
          sequence_number: 2,
          role: 'assistant',
          content: 'Attention is all you need.',
          status: 'completed',
          citations: [],
          created_at: '2026-11-15T00:01:00Z',
        },
      ],
    };
  };

  t.after(() => {
    apiClient.get = originalGet;
  });

  const history = await conversationApi.getConversationHistory('conv-abc');

  assert.equal(interceptedUrl, '/conversations/conv-abc/messages');
  assert.equal(history.length, 2);
  assert.equal(history[0].role, 'user');
  assert.equal(history[1].role, 'assistant');
});
