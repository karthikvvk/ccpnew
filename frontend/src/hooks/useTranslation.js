import { useState, useCallback, useRef } from 'react';
import { translationApi, languageUtils } from '../services/api';

/**
 * Custom hook for managing video translation workflow
 * Handles file upload, translation job, status polling, and downloads
 */
export function useTranslation() {
    // Job state
    const [jobId, setJobId] = useState(null);
    const [status, setStatus] = useState('idle'); // idle, uploading, processing, completed, failed
    const [progress, setProgress] = useState(null);
    const [error, setError] = useState(null);
    const [files, setFiles] = useState(null);

    // Video state
    const [uploadedVideo, setUploadedVideo] = useState(null);
    const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);

    // Polling reference
    const pollingRef = useRef(null);

    /**
     * Handle video file selection
     */
    const handleVideoSelect = useCallback((file) => {
        if (file) {
            setUploadedVideo(file);
            // Create preview URL
            const url = URL.createObjectURL(file);
            setVideoPreviewUrl(url);
            // Reset previous job state
            setJobId(null);
            setStatus('idle');
            setProgress(null);
            setError(null);
            setFiles(null);
        }
    }, []);

    /**
     * Clear the selected video
     */
    const clearVideo = useCallback(() => {
        if (videoPreviewUrl) {
            URL.revokeObjectURL(videoPreviewUrl);
        }
        setUploadedVideo(null);
        setVideoPreviewUrl(null);
        setJobId(null);
        setStatus('idle');
        setProgress(null);
        setError(null);
        setFiles(null);
    }, [videoPreviewUrl]);

    /**
     * Start translation job
     * @param {string} targetLangCode - Language code (ES, JP, DE, EN, etc.)
     * @param {boolean} useRag - Enable RAG context
     */
    const startTranslation = useCallback(async (targetLangCode, useRag = true) => {
        if (!uploadedVideo) {
            setError('No video selected');
            return;
        }

        try {
            setStatus('uploading');
            setError(null);
            setProgress('Uploading video...');

            // Convert language code to full name
            const targetLanguage = languageUtils.getLanguageName(targetLangCode);

            // Start translation job
            const result = await translationApi.translate(
                uploadedVideo,
                targetLanguage,
                'auto',
                useRag
            );

            setJobId(result.job_id);
            setStatus('processing');
            setProgress('Processing video...');

            // Start polling for status
            pollingRef.current = setInterval(async () => {
                try {
                    const statusResult = await translationApi.getStatus(result.job_id);
                    setProgress(statusResult.progress || 'Processing...');

                    if (statusResult.status === 'completed') {
                        clearInterval(pollingRef.current);
                        pollingRef.current = null;
                        setStatus('completed');
                        setFiles(statusResult.files);
                        setProgress('Translation complete!');
                    } else if (statusResult.status === 'failed') {
                        clearInterval(pollingRef.current);
                        pollingRef.current = null;
                        setStatus('failed');
                        setError(statusResult.error || 'Translation failed');
                    }
                } catch (pollError) {
                    console.error('Polling error:', pollError);
                }
            }, 2000);

        } catch (err) {
            setStatus('failed');
            setError(err.message || 'Failed to start translation');
        }
    }, [uploadedVideo]);

    /**
     * Stop polling (cleanup)
     */
    const stopPolling = useCallback(() => {
        if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
        }
    }, []);

    /**
     * Download a specific file from the completed job
     * @param {string} fileType - Type of file to download
     */
    const downloadFile = useCallback((fileType) => {
        if (!jobId) return;
        const url = translationApi.getDownloadUrl(jobId, fileType);
        window.open(url, '_blank');
    }, [jobId]);

    /**
     * Get download URL for a specific file
     * @param {string} fileType - Type of file
     * @returns {string|null} Download URL or null
     */
    const getDownloadUrl = useCallback((fileType) => {
        if (!jobId) return null;
        return translationApi.getDownloadUrl(jobId, fileType);
    }, [jobId]);

    /**
     * Reset the entire state
     */
    const reset = useCallback(() => {
        stopPolling();
        clearVideo();
    }, [stopPolling, clearVideo]);

    return {
        // State
        jobId,
        status,
        progress,
        error,
        files,
        uploadedVideo,
        videoPreviewUrl,

        // Computed
        isIdle: status === 'idle',
        isUploading: status === 'uploading',
        isProcessing: status === 'processing',
        isCompleted: status === 'completed',
        isFailed: status === 'failed',
        hasVideo: !!uploadedVideo,
        canStart: !!uploadedVideo && status === 'idle',

        // Actions
        handleVideoSelect,
        clearVideo,
        startTranslation,
        stopPolling,
        downloadFile,
        getDownloadUrl,
        reset
    };
}

export default useTranslation;
