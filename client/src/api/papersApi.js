/**
 * API client methods for research paper catalog and details.
 * Interacts with FastAPI backend paper endpoints (/api/v1/papers).
 */

import apiClient from './client.js';

/**
 * Retrieves paginated research papers with file details and ingestion statistics.
 *
 * @param {object} [options={}] - Query options
 * @param {number} [options.limit=50] - Maximum number of papers to retrieve
 * @param {number} [options.offset=0] - Number of items to skip
 * @param {string|null} [options.search=null] - Optional search term across paper title and DOI
 * @returns {Promise<object>} Response containing items list, total count, limit, offset
 */
export async function listPapers({ limit = 50, offset = 0, search = null } = {}) {
  const params = {
    limit: Math.max(1, Math.min(Number(limit) || 50, 100)),
    offset: Math.max(0, Number(offset) || 0),
  };

  if (search && typeof search === 'string' && search.trim()) {
    params.search = search.trim();
  }

  const response = await apiClient.get('/papers', { params });
  return response.data;
}

/**
 * Retrieves full details for a single research paper by UUID.
 *
 * @param {string} paperId - UUID of the research paper
 * @returns {Promise<object>} Complete paper details including files and metadata
 */
export async function getPaperDetails(paperId) {
  if (!paperId || typeof paperId !== 'string' || !paperId.trim()) {
    throw new Error('paperId is required to retrieve paper details.');
  }

  const response = await apiClient.get(`/papers/${encodeURIComponent(paperId.trim())}`);
  return response.data;
}
