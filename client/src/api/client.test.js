import test from 'node:test';
import assert from 'node:assert/strict';
import { formatApiErrorMessage, apiClient } from './client.js';

test('client - formatApiErrorMessage formats FastAPI 422 validation error array', () => {
  const error422 = {
    response: {
      status: 422,
      data: {
        detail: [
          {
            type: 'missing',
            loc: ['body', 'file'],
            msg: 'Field required',
            input: null,
          },
        ],
      },
    },
  };

  const message = formatApiErrorMessage(error422);
  assert.equal(message, 'file: Field required');
  assert.notEqual(message, '[object Object]');
});

test('client - formatApiErrorMessage formats multiple validation errors cleanly', () => {
  const multiError = {
    response: {
      status: 422,
      data: {
        detail: [
          {
            loc: ['body', 'file'],
            msg: 'Field required',
          },
          {
            loc: ['body', 'doi'],
            msg: 'String should have at least 5 characters',
          },
        ],
      },
    },
  };

  const message = formatApiErrorMessage(multiError);
  assert.equal(message, 'file: Field required; doi: String should have at least 5 characters');
});

test('client - formatApiErrorMessage formats string detail directly', () => {
  const stringError = {
    response: {
      status: 400,
      data: {
        detail: "Invalid PDF format: Content must start with '%PDF-'.",
      },
    },
  };

  const message = formatApiErrorMessage(stringError);
  assert.equal(message, "Invalid PDF format: Content must start with '%PDF-'.");
});

test('client - formatApiErrorMessage handles network and fallback errors', () => {
  const networkError = {
    message: 'Network Error: connection refused',
  };

  const message = formatApiErrorMessage(networkError);
  assert.equal(message, 'Network Error: connection refused');

  const emptyError = formatApiErrorMessage(null);
  assert.equal(emptyError, 'An unexpected error occurred.');
});

test('client - apiClient removes Content-Type for FormData requests', async () => {
  const formData = new FormData();
  formData.append('file', new Blob(['test']), 'test.pdf');

  // Test interceptor behavior directly
  const config = {
    headers: {
      'Content-Type': 'application/json',
    },
    data: formData,
  };

  // Run through request interceptor
  for (const handler of apiClient.interceptors.request.handlers) {
    if (handler && handler.fulfilled) {
      handler.fulfilled(config);
    }
  }

  assert.equal(config.headers['Content-Type'], undefined);
});
