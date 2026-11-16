import test from 'node:test';
import assert from 'node:assert/strict';
import apiClient from './client.js';
import {
  validatePdfFile,
  uploadPaper,
  getIngestionJob,
  MAX_FILE_SIZE_BYTES,
} from './ingestionApi.js';

test('ingestionApi - validatePdfFile accepts valid PDF File/Blob', () => {
  const validFile = {
    name: 'attention_is_all_you_need.pdf',
    type: 'application/pdf',
    size: 1024 * 500, // 500 KB
  };
  assert.doesNotThrow(() => validatePdfFile(validFile));
});

test('ingestionApi - validatePdfFile rejects missing file', () => {
  assert.throws(
    () => validatePdfFile(null),
    /A PDF file must be provided/
  );
  assert.throws(
    () => validatePdfFile(undefined),
    /A PDF file must be provided/
  );
});

test('ingestionApi - validatePdfFile rejects non-PDF extension', () => {
  const invalidExt = {
    name: 'research_notes.docx',
    type: 'application/pdf',
    size: 2048,
  };
  assert.throws(
    () => validatePdfFile(invalidExt),
    /Only PDF documents \(\.pdf\) are supported/
  );
});

test('ingestionApi - validatePdfFile rejects non-PDF mime type', () => {
  const invalidMime = {
    name: 'document.pdf',
    type: 'image/jpeg',
    size: 2048,
  };
  assert.throws(
    () => validatePdfFile(invalidMime),
    /MIME type must be application\/pdf/
  );
});

test('ingestionApi - validatePdfFile rejects 0-byte empty file', () => {
  const emptyFile = {
    name: 'empty.pdf',
    type: 'application/pdf',
    size: 0,
  };
  assert.throws(
    () => validatePdfFile(emptyFile),
    /Uploaded file is empty/
  );
});

test('ingestionApi - validatePdfFile rejects files exceeding 50 MB limit', () => {
  const oversizedFile = {
    name: 'huge_book.pdf',
    type: 'application/pdf',
    size: MAX_FILE_SIZE_BYTES + 1,
  };
  assert.throws(
    () => validatePdfFile(oversizedFile),
    /File size exceeds the maximum limit of 50 MB/
  );
});

test('ingestionApi - uploadPaper creates FormData with file and posts to /ingestion/upload', async (t) => {
  const originalPost = apiClient.post;
  let interceptedUrl = null;
  let interceptedPayload = null;

  apiClient.post = async (url, payload) => {
    interceptedUrl = url;
    interceptedPayload = payload;
    return {
      data: {
        paper_id: '11111111-1111-1111-1111-111111111111',
        job_id: '22222222-2222-2222-2222-222222222222',
        paper_file_id: '33333333-3333-3333-3333-333333333333',
        chunks_created: 42,
        embeddings_created: 42,
        paper_reused: false,
        file_reused: false,
        job_reused: false,
        status: 'completed',
      },
    };
  };

  t.after(() => {
    apiClient.post = originalPost;
  });

  const mockPdf = new Blob(['%PDF-1.4 mock content'], { type: 'application/pdf' });
  mockPdf.name = 'sample_paper.pdf';

  const result = await uploadPaper(mockPdf);

  assert.equal(interceptedUrl, '/ingestion/upload');
  assert.ok(interceptedPayload instanceof FormData, 'Payload should be a FormData instance');
  assert.ok(interceptedPayload.get('file'), 'FormData should contain file entry');
  assert.equal(interceptedPayload.get('title'), null, 'Title should be absent when omitted');
  assert.equal(interceptedPayload.get('doi'), null, 'DOI should be absent when omitted');
  assert.equal(result.job_id, '22222222-2222-2222-2222-222222222222');
  assert.equal(result.chunks_created, 42);
  assert.equal(result.embeddings_created, 42);
  assert.equal(result.status, 'completed');
});

test('ingestionApi - uploadPaper attaches title and DOI when provided as arguments', async (t) => {
  const originalPost = apiClient.post;
  let interceptedPayload = null;

  apiClient.post = async (url, payload) => {
    interceptedPayload = payload;
    return {
      data: {
        job_id: 'job-title-123',
        title: 'Deep Residual Learning for Image Recognition',
        status: 'completed',
      },
    };
  };

  t.after(() => {
    apiClient.post = originalPost;
  });

  const mockPdf = new Blob(['%PDF-1.4 test'], { type: 'application/pdf' });
  mockPdf.name = 'resnet.pdf';

  const res = await uploadPaper(mockPdf, 'Deep Residual Learning for Image Recognition', '10.1109/CVPR.2016.90');

  assert.equal(interceptedPayload.get('title'), 'Deep Residual Learning for Image Recognition');
  assert.equal(interceptedPayload.get('doi'), '10.1109/CVPR.2016.90');
  assert.equal(res.title, 'Deep Residual Learning for Image Recognition');
});

test('ingestionApi - uploadPaper supports options object with title and doi', async (t) => {
  const originalPost = apiClient.post;
  let interceptedPayload = null;

  apiClient.post = async (url, payload) => {
    interceptedPayload = payload;
    return {
      data: {
        job_id: 'job-opts-123',
        title: 'Attention Is All You Need',
        status: 'completed',
      },
    };
  };

  t.after(() => {
    apiClient.post = originalPost;
  });

  const mockPdf = new Blob(['%PDF-1.4 test'], { type: 'application/pdf' });
  mockPdf.name = 'attention.pdf';

  const res = await uploadPaper(mockPdf, {
    title: 'Attention Is All You Need',
    doi: '10.48550/arXiv.1706.03762',
  });

  assert.equal(interceptedPayload.get('title'), 'Attention Is All You Need');
  assert.equal(interceptedPayload.get('doi'), '10.48550/arXiv.1706.03762');
  assert.equal(res.title, 'Attention Is All You Need');
});

test('ingestionApi - uploadPaper attaches DOI when provided', async (t) => {
  const originalPost = apiClient.post;
  let interceptedPayload = null;

  apiClient.post = async (url, payload) => {
    interceptedPayload = payload;
    return {
      data: {
        job_id: 'job-doi-123',
        status: 'completed',
      },
    };
  };

  t.after(() => {
    apiClient.post = originalPost;
  });

  const mockPdf = new Blob(['%PDF-1.4 test'], { type: 'application/pdf' });
  mockPdf.name = 'paper_with_doi.pdf';

  await uploadPaper(mockPdf, '10.1145/1234567.890123');

  assert.equal(interceptedPayload.get('doi'), '10.1145/1234567.890123');
});

test('ingestionApi - uploadPaper rejects non-PDF without network call', async () => {
  const badFile = {
    name: 'document.txt',
    type: 'text/plain',
    size: 100,
  };

  await assert.rejects(
    async () => uploadPaper(badFile),
    /Only PDF documents \(\.pdf\) are supported/
  );
});

test('ingestionApi - getIngestionJob calls /ingestion/jobs/{job_id}', async (t) => {
  const originalGet = apiClient.get;
  let interceptedUrl = null;

  apiClient.get = async (url) => {
    interceptedUrl = url;
    return {
      data: {
        job_id: 'job-987',
        paper_id: 'paper-987',
        status: 'processing',
        current_stage: 'chunking',
        attempt_count: 1,
        error_message: null,
      },
    };
  };

  t.after(() => {
    apiClient.get = originalGet;
  });

  const result = await getIngestionJob('job-987');

  assert.equal(interceptedUrl, '/ingestion/jobs/job-987');
  assert.equal(result.job_id, 'job-987');
  assert.equal(result.status, 'processing');
  assert.equal(result.current_stage, 'chunking');
});

test('ingestionApi - getIngestionJob rejects empty jobId', async () => {
  await assert.rejects(
    async () => getIngestionJob(''),
    /jobId is required/
  );
  await assert.rejects(
    async () => getIngestionJob(null),
    /jobId is required/
  );
});
