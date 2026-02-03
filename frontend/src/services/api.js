/**
 * API Service for Video Translation Backend
 * Handles all communication with the FastAPI backend at /api/v1
 */

const API_BASE = '/api/v1';

/**
 * Upload a video for translation
 * @param {File} videoFile - The video file to upload
 * @param {string} targetLanguage - Target language (e.g., 'Spanish', 'Japanese')
 * @param {string} sourceLanguage - Source language (default: 'auto')
 * @param {boolean} useRag - Whether to use RAG context (default: true)
 * @returns {Promise<{job_id: string, status: string, message: string}>}
 */
export async function uploadVideo(videoFile, targetLanguage, sourceLanguage = 'auto', useRag = true) {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('target_language', targetLanguage);
    formData.append('source_language', sourceLanguage);
    formData.append('use_rag', useRag.toString());

    const response = await fetch(`${API_BASE}/translate`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(error.detail || 'Failed to upload video');
    }

    return response.json();
}

/**
 * Get the status of a translation job
 * @param {string} jobId - The job ID returned from uploadVideo
 * @returns {Promise<{job_id: string, status: string, progress: string, files: Array, error: string}>}
 */
export async function getJobStatus(jobId) {
    const response = await fetch(`${API_BASE}/status/${jobId}`);

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Status check failed' }));
        throw new Error(error.detail || 'Failed to get job status');
    }

    return response.json();
}

/**
 * Get download URL for a specific file type
 * @param {string} jobId - The job ID
 * @param {string} fileType - Type of file ('audio', 'srt', 'video', or specific type from files list)
 * @returns {string} The download URL
 */
export function getDownloadUrl(jobId, fileType) {
    // Use quick download endpoints for common types
    if (['audio', 'srt', 'video'].includes(fileType)) {
        return `${API_BASE}/jobs/${jobId}/download/${fileType}`;
    }
    // Use generic download for other file types
    return `${API_BASE}/download/${jobId}/${fileType}`;
}

/**
 * List all available files for a completed job
 * @param {string} jobId - The job ID
 * @returns {Promise<{job_id: string, status: string, files: Array, quick_downloads: Object}>}
 */
export async function listJobFiles(jobId) {
    const response = await fetch(`${API_BASE}/jobs/${jobId}/files`);

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Failed to list files' }));
        throw new Error(error.detail || 'Failed to list job files');
    }

    return response.json();
}

/**
 * Poll job status until completion or failure
 * @param {string} jobId - The job ID to poll
 * @param {function} onProgress - Callback for progress updates: (status, progress) => void
 * @param {number} intervalMs - Polling interval in milliseconds (default: 2000)
 * @returns {Promise<Object>} Final status response when completed or failed
 */
export function pollJobStatus(jobId, onProgress, intervalMs = 2000) {
    return new Promise((resolve, reject) => {
        const poll = async () => {
            try {
                const status = await getJobStatus(jobId);

                if (onProgress) {
                    onProgress(status.status, status.progress);
                }

                if (status.status === 'completed') {
                    resolve(status);
                } else if (status.status === 'failed') {
                    reject(new Error(status.error || 'Translation failed'));
                } else {
                    setTimeout(poll, intervalMs);
                }
            } catch (error) {
                reject(error);
            }
        };

        poll();
    });
}

export default {
    uploadVideo,
    getJobStatus,
    getDownloadUrl,
    listJobFiles,
    pollJobStatus,
};
