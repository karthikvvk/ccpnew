/**
 * AuraFlow API Service Layer
 * Connects React frontend to FastAPI backend (hosted on Colab with ngrok)
 */

// API Base URL - will be set from environment or Colab ngrok URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Get the current API base URL
 */
export const getApiBaseUrl = () => API_BASE_URL;

/**
 * Set a custom API URL (for dynamic Colab ngrok URLs)
 */
let dynamicApiUrl = null;
export const setApiBaseUrl = (url) => {
    dynamicApiUrl = url;
};

const getActiveBaseUrl = () => dynamicApiUrl || API_BASE_URL;

/**
 * Translation API endpoints
 */
export const translationApi = {
    /**
     * Start a video translation job
     * @param {File} videoFile - Video file to translate
     * @param {string} targetLanguage - Target language (e.g., "Spanish", "Japanese")
     * @param {string} sourceLanguage - Source language or "auto" for detection
     * @param {boolean} useRag - Enable RAG context enhancement
     * @returns {Promise<{job_id: string, status: string, message: string}>}
     */
    translate: async (videoFile, targetLanguage, sourceLanguage = 'auto', useRag = true) => {
        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('target_language', targetLanguage);
        formData.append('source_language', sourceLanguage);
        formData.append('use_rag', useRag);

        const response = await fetch(`${getActiveBaseUrl()}/api/v1/translate`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Translation request failed: ${response.statusText}`);
        }

        return response.json();
    },

    /**
     * Poll job status
     * @param {string} jobId - Job ID from translate response
     * @returns {Promise<{job_id: string, status: string, progress: string, files: Array}>}
     */
    getStatus: async (jobId) => {
        const response = await fetch(`${getActiveBaseUrl()}/api/v1/status/${jobId}`);

        if (!response.ok) {
            throw new Error(`Status check failed: ${response.statusText}`);
        }

        return response.json();
    },

    /**
     * Get download URL for a processed file
     * @param {string} jobId - Job ID
     * @param {string} fileType - File type (final_video, translated_audio, translation_txt, etc.)
     * @returns {string} Download URL
     */
    getDownloadUrl: (jobId, fileType) => {
        return `${getActiveBaseUrl()}/api/v1/download/${jobId}/${fileType}`;
    },

    /**
     * Health check endpoint
     * @returns {Promise<{status: string, version: string}>}
     */
    health: async () => {
        const response = await fetch(`${getActiveBaseUrl()}/health`);
        return response.json();
    },

    /**
     * Poll until job completes or fails
     * @param {string} jobId - Job ID
     * @param {function} onProgress - Callback for progress updates
     * @param {number} intervalMs - Polling interval in ms (default 2000)
     * @returns {Promise<object>} Final status
     */
    pollUntilComplete: async (jobId, onProgress, intervalMs = 2000) => {
        return new Promise((resolve, reject) => {
            const poll = async () => {
                try {
                    const status = await translationApi.getStatus(jobId);

                    if (onProgress) {
                        onProgress(status);
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
};

/**
 * Streaming API endpoints (for real-time transcription)
 */
export const streamingApi = {
    /**
     * List available audio input devices
     */
    getDevices: async () => {
        const response = await fetch(`${getActiveBaseUrl()}/api/v1/stream/devices`);
        return response.json();
    },

    /**
     * Start audio streaming
     * @param {number|null} device - Device index or null for default
     */
    start: async (device = null) => {
        const url = device !== null
            ? `${getActiveBaseUrl()}/api/v1/stream/start?device=${device}`
            : `${getActiveBaseUrl()}/api/v1/stream/start`;
        const response = await fetch(url, { method: 'POST' });
        return response.json();
    },

    /**
     * Stop audio streaming
     */
    stop: async () => {
        const response = await fetch(`${getActiveBaseUrl()}/api/v1/stream/stop`, { method: 'POST' });
        return response.json();
    },

    /**
     * Get streaming status
     */
    status: async () => {
        const response = await fetch(`${getActiveBaseUrl()}/api/v1/stream/status`);
        return response.json();
    },

    /**
     * Create WebSocket connection for real-time transcription
     * @returns {WebSocket}
     */
    createWebSocket: () => {
        const wsUrl = getActiveBaseUrl().replace('http://', 'ws://').replace('https://', 'wss://');
        return new WebSocket(`${wsUrl}/api/v1/stream/ws`);
    }
};

/**
 * Language utilities
 */
export const languageUtils = {
    // Map UI language codes to full names for API
    codeToName: {
        'ES': 'Spanish',
        'JP': 'Japanese',
        'DE': 'German',
        'EN': 'English',
        'FR': 'French',
        'PT': 'Portuguese',
        'IT': 'Italian',
        'RU': 'Russian',
        'KO': 'Korean',
        'ZH': 'Chinese',
        'AR': 'Arabic',
        'HI': 'Hindi'
    },

    /**
     * Convert UI language code to API language name
     */
    getLanguageName: (code) => {
        return languageUtils.codeToName[code] || code;
    }
};

export default {
    translation: translationApi,
    streaming: streamingApi,
    language: languageUtils,
    setApiBaseUrl,
    getApiBaseUrl: getActiveBaseUrl
};
