import axios from 'axios';

/**
 * Centralized Axios client configured with base URL from environment variables.
 * Default fallback: http://localhost:8000/api/v1
 */
const baseURL = import.meta?.env?.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

/**
 * Formats API errors into a human-readable message string.
 * Handles FastAPI validation errors (array of loc/msg objects),
 * custom detail strings, HTTP errors, and network errors.
 *
 * @param {any} error - The axios error object
 * @returns {string} Formatted error message
 */
export function formatApiErrorMessage(error) {
  if (!error) {
    return 'An unexpected error occurred.';
  }

  const detail = error.response?.data?.detail;

  if (detail) {
    // FastAPI validation errors (e.g. 422 Unprocessable Entity) return an array of error objects
    if (Array.isArray(detail)) {
      const messages = detail.map((err) => {
        if (typeof err === 'string') return err;
        const location = Array.isArray(err.loc)
          ? err.loc.filter((part) => part !== 'body').join('.')
          : '';
        const msg = err.msg || 'Invalid value';
        return location ? `${location}: ${msg}` : msg;
      });
      return messages.join('; ') || 'Validation error: Invalid request parameters.';
    }

    // Detail is a plain string
    if (typeof detail === 'string') {
      return detail;
    }

    // Detail is an object
    if (typeof detail === 'object') {
      return detail.message || detail.msg || JSON.stringify(detail);
    }
  }

  // Handle standard HTTP status messages
  if (error.response?.statusText) {
    return `Server error (${error.response.status}): ${error.response.statusText}`;
  }

  // Handle network / connection errors
  if (error.message) {
    return error.message;
  }

  return 'An unexpected error occurred while communicating with the server.';
}

// Request interceptor: ensure FormData removes the default application/json header
apiClient.interceptors.request.use(
  (config) => {
    if (config.data instanceof FormData) {
      // Let browser / Axios generate multipart/form-data with the correct boundary
      if (config.headers) {
        delete config.headers['Content-Type'];
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for centralized error formatting
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = formatApiErrorMessage(error);
    const err = new Error(message);
    err.originalError = error;
    err.status = error.response?.status;
    err.data = error.response?.data;
    return Promise.reject(err);
  }
);

/**
 * Derives the server root URL from the API base URL.
 * e.g., 'http://localhost:8000/api/v1' -> 'http://localhost:8000'
 */
export function getServerRootUrl(apiUrl = baseURL) {
  try {
    const url = new URL(apiUrl);
    return url.origin;
  } catch {
    return apiUrl.replace(/\/api\/v1\/?$/, '');
  }
}

/**
 * Dedicated health-check function calling GET /health at the server root.
 * Kept strictly separate from apiClient to avoid appending /api/v1.
 */
export async function checkHealth(timeout = 5000) {
  const rootUrl = getServerRootUrl();
  const healthUrl = `${rootUrl}/health`;

  try {
    const response = await axios.get(healthUrl, {
      timeout,
      headers: {
        Accept: 'application/json',
      },
    });
    return response.data;
  } catch (error) {
    if (error.response) {
      const detail = error.response.data?.detail || error.response.statusText;
      throw new Error(`Health check failed with HTTP ${error.response.status}: ${detail || 'Unknown error'}`);
    } else if (error.request) {
      throw new Error(`Could not connect to backend at ${healthUrl} (${error.message || 'Connection refused'}).`);
    } else {
      throw new Error(`Health check request failed: ${error.message}`);
    }
  }
}

export default apiClient;
