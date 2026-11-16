/**
 * API client methods for document ingestion and background job tracking.
 * Interacts with FastAPI backend ingestion endpoints.
 */

import apiClient from './client.js';

export const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB

/**
 * Validates a file object before upload.
 *
 * @param {File|Blob} file - The file to validate
 * @throws {Error} If the file is missing, empty, invalid type, or exceeds size limit
 */
export function validatePdfFile(file) {
  if (!file) {
    throw new Error('A PDF file must be provided.');
  }

  // Validate filename if available
  if (file.name) {
    const isPdfExt = file.name.trim().toLowerCase().endsWith('.pdf');
    if (!isPdfExt) {
      throw new Error('Invalid file format: Only PDF documents (.pdf) are supported.');
    }
  }

  // Validate MIME type if available
  if (file.type && file.type !== 'application/pdf') {
    // Note: Some browsers may report empty string or generic binary, so only reject if an explicit non-PDF MIME is present
    if (file.type !== 'application/x-pdf' && file.type !== 'application/acrobat') {
      throw new Error('Invalid file format: MIME type must be application/pdf.');
    }
  }

  // Validate file size
  if (typeof file.size === 'number') {
    if (file.size <= 0) {
      throw new Error('Uploaded file is empty.');
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      throw new Error('File size exceeds the maximum limit of 50 MB.');
    }
  }
}

/**
 * Uploads a research paper PDF file to the backend for ingestion, parsing, chunking, and embedding.
 *
 * @param {File|Blob} file - The PDF file to upload
 * @param {string|object|null} [titleOrOptions=null] - Optional paper title string OR options object ({ title, doi })
 * @param {string|null} [doiParam=null] - Optional Digital Object Identifier (DOI)
 * @returns {Promise<object>} Ingestion upload response containing job_id, paper_id, title, chunks_created, etc.
 */
export async function uploadPaper(file, titleOrOptions = null, doiParam = null) {
  validatePdfFile(file);

  let title = null;
  let doi = null;

  if (titleOrOptions && typeof titleOrOptions === 'object' && !(titleOrOptions instanceof String)) {
    title = titleOrOptions.title ?? null;
    doi = titleOrOptions.doi ?? null;
  } else if (
    typeof titleOrOptions === 'string' &&
    doiParam === null &&
    (titleOrOptions.trim().startsWith('10.') || titleOrOptions.trim().toLowerCase().startsWith('doi:'))
  ) {
    // Single string argument formatted as DOI
    doi = titleOrOptions;
    title = null;
  } else {
    title = titleOrOptions;
    doi = doiParam;
  }

  const formData = new FormData();
  formData.append('file', file);

  if (title && typeof title === 'string' && title.trim()) {
    formData.append('title', title.trim());
  }

  if (doi && typeof doi === 'string' && doi.trim()) {
    formData.append('doi', doi.trim());
  }

  const response = await apiClient.post('/ingestion/upload', formData, {
    headers: {
      'Content-Type': undefined,
    },
    timeout: 120000,
  });
  return response.data;
}

/**
 * Retrieves the status and details of an ingestion job by ID.
 *
 * @param {string} jobId - UUID of the ingestion job
 * @returns {Promise<object>} Ingestion job status response (job_id, status, current_stage, etc.)
 */
export async function getIngestionJob(jobId) {
  if (!jobId || typeof jobId !== 'string' || !jobId.trim()) {
    throw new Error('jobId is required to check ingestion status.');
  }

  const response = await apiClient.get(`/ingestion/jobs/${encodeURIComponent(jobId.trim())}`);
  return response.data;
}
