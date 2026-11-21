import test from 'node:test';
import assert from 'node:assert/strict';
import apiClient from './client.js';
import { listPapers, getPaperDetails, deletePaper } from './papersApi.js';

test('papersApi - listPapers sends GET /papers with default pagination params', async (t) => {
  const originalGet = apiClient.get;
  let interceptedUrl = null;
  let interceptedConfig = null;

  apiClient.get = async (url, config) => {
    interceptedUrl = url;
    interceptedConfig = config;
    return {
      data: {
        items: [],
        total: 0,
        limit: 50,
        offset: 0,
      },
    };
  };

  t.after(() => {
    apiClient.get = originalGet;
  });

  const res = await listPapers();

  assert.equal(interceptedUrl, '/papers');
  assert.deepEqual(interceptedConfig.params, { limit: 50, offset: 0 });
  assert.deepEqual(res.items, []);
  assert.equal(res.total, 0);
});

test('papersApi - listPapers forwards search query param', async (t) => {
  const originalGet = apiClient.get;
  let interceptedConfig = null;

  apiClient.get = async (url, config) => {
    interceptedConfig = config;
    return {
      data: {
        items: [
          {
            id: 'paper-123',
            title: 'Attention Is All You Need',
            chunk_count: 42,
            page_count: 12,
            ingestion_status: 'completed',
          },
        ],
        total: 1,
        limit: 20,
        offset: 0,
      },
    };
  };

  t.after(() => {
    apiClient.get = originalGet;
  });

  const res = await listPapers({ limit: 20, offset: 0, search: 'Attention' });

  assert.equal(interceptedConfig.params.search, 'Attention');
  assert.equal(interceptedConfig.params.limit, 20);
  assert.equal(res.items.length, 1);
  assert.equal(res.items[0].title, 'Attention Is All You Need');
});

test('papersApi - getPaperDetails sends GET /papers/{paper_id}', async (t) => {
  const originalGet = apiClient.get;
  let interceptedUrl = null;

  apiClient.get = async (url) => {
    interceptedUrl = url;
    return {
      data: {
        id: 'paper-uuid-999',
        title: 'BERT: Pre-training of Deep Bidirectional Transformers',
        chunk_count: 55,
        files: [
          {
            id: 'file-1',
            original_filename: 'bert.pdf',
          },
        ],
      },
    };
  };

  t.after(() => {
    apiClient.get = originalGet;
  });

  const res = await getPaperDetails('paper-uuid-999');

  assert.equal(interceptedUrl, '/papers/paper-uuid-999');
  assert.equal(res.id, 'paper-uuid-999');
  assert.equal(res.files.length, 1);
});

test('papersApi - getPaperDetails rejects missing paperId without network call', async () => {
  await assert.rejects(
    async () => getPaperDetails(''),
    /paperId is required/
  );
  await assert.rejects(
    async () => getPaperDetails(null),
    /paperId is required/
  );
});

test('papersApi - deletePaper sends DELETE /papers/{paper_id}', async (t) => {
  const originalDelete = apiClient.delete;
  let interceptedUrl = null;

  apiClient.delete = async (url) => {
    interceptedUrl = url;
    return { data: null, status: 204 };
  };

  t.after(() => {
    apiClient.delete = originalDelete;
  });

  await deletePaper('paper-to-delete-123');
  assert.equal(interceptedUrl, '/papers/paper-to-delete-123');
});

test('papersApi - deletePaper rejects missing paperId without network call', async () => {
  await assert.rejects(
    async () => deletePaper(''),
    /paperId is required/
  );
  await assert.rejects(
    async () => deletePaper(null),
    /paperId is required/
  );
});

test('papersApi - deletePaper propagates error when backend fails', async (t) => {
  const originalDelete = apiClient.delete;

  apiClient.delete = async () => {
    const error = new Error('Conflict: Ingestion active');
    error.status = 409;
    throw error;
  };

  t.after(() => {
    apiClient.delete = originalDelete;
  });

  await assert.rejects(
    async () => deletePaper('paper-123'),
    /Conflict: Ingestion active/
  );
});
