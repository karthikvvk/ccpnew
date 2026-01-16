"""
Audio Streaming Module
Captures device audio and provides real-time transcription

Supports:
- Local mode: Direct object access
- Tunnel mode: FastAPI server with endpoints
- Colab GPU offloading for Whisper

Based on streaming/streamer.py and streaming/simple.py
"""

import os
import json
import queue
import threading
import time
import base64
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import numpy as np

# Optional imports
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from utils.logger import setup_logger
from config import settings

logger = setup_logger("streaming")


class AudioStreamer:
    """
    Captures device media audio and pushes to a thread-safe queue.
    
    Usage (Local Mode):
        streamer = AudioStreamer()
        streamer.start(device=device_index)
        
        while streamer.is_streaming():
            chunk = streamer.pop(timeout=1.0)
            if chunk is not None:
                # Process chunk
                pass
        
        streamer.stop()
    """
    
    def __init__(self, 
                 interval_seconds: float = 0.5,
                 buffer_size: int = 10,
                 sample_rate: int = 48000,
                 channels: int = 1):
        """
        Initialize AudioStreamer.
        
        Args:
            interval_seconds: Seconds per audio chunk
            buffer_size: Max chunks in queue
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice not installed. Install with: pip install sounddevice")
        
        self._interval_seconds = interval_seconds
        self._buffer_size = buffer_size
        self._sample_rate = sample_rate
        self._channels = channels
        
        # Thread-safe queue for audio chunks
        self._audio_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        
        # Threading control
        self._streaming = False
        self._lock = threading.Lock()
        
        # Audio stream handle
        self._stream: Optional[sd.InputStream] = None
        
        # Buffer for accumulating audio samples
        self._sample_buffer = np.array([], dtype=np.float32)
        self._samples_per_chunk = int(sample_rate * interval_seconds)
        
        logger.info(f"AudioStreamer initialized: {sample_rate}Hz, {interval_seconds}s chunks")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info: dict, status):
        """Callback for audio stream - accumulates and queues chunks."""
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Convert to mono if needed
        if indata.ndim > 1:
            audio_data = indata.mean(axis=1).astype(np.float32)
        else:
            audio_data = indata.flatten().astype(np.float32)
        
        # Accumulate samples
        self._sample_buffer = np.concatenate([self._sample_buffer, audio_data])
        
        # Check if we have enough samples for a chunk
        while len(self._sample_buffer) >= self._samples_per_chunk:
            chunk = self._sample_buffer[:self._samples_per_chunk]
            self._sample_buffer = self._sample_buffer[self._samples_per_chunk:]
            
            chunk_bytes = chunk.tobytes()
            
            try:
                self._audio_queue.put_nowait(chunk_bytes)
            except queue.Full:
                # Drop oldest if full
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait(chunk_bytes)
                except queue.Empty:
                    pass
    
    def start(self, device: Optional[int] = None):
        """Start audio capture."""
        if self._streaming:
            logger.warning("Streamer already running")
            return
        
        self._streaming = True
        self._sample_buffer = np.array([], dtype=np.float32)
        
        try:
            self._stream = sd.InputStream(
                device=device,
                channels=self._channels,
                samplerate=self._sample_rate,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=int(self._sample_rate * 0.1)
            )
            self._stream.start()
            logger.info(f"Audio streaming started (device: {device}, interval: {self._interval_seconds}s)")
            
        except Exception as e:
            self._streaming = False
            logger.error(f"Failed to start audio stream: {e}")
            raise
    
    def stop(self):
        """Stop audio capture."""
        if not self._streaming:
            return
        
        self._streaming = False
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        logger.info("Audio streaming stopped")
    
    def pop(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Pop next audio chunk from queue."""
        try:
            if timeout is None:
                return self._audio_queue.get_nowait()
            else:
                return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def pop_as_numpy(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Pop next chunk as numpy array."""
        chunk = self.pop(timeout)
        if chunk is not None:
            return np.frombuffer(chunk, dtype=np.float32)
        return None
    
    def is_streaming(self) -> bool:
        """Check if streaming is active."""
        return self._streaming
    
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._audio_queue.qsize()
    
    def clear_queue(self):
        """Clear all items from queue."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_queue(self) -> queue.Queue:
        """Get reference to the queue."""
        return self._audio_queue
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """List available audio input devices."""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': int(device['default_samplerate'])
                })
        return input_devices
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class LiveTranscriber:
    """
    Real-time audio transcription using Whisper.
    
    Usage:
        transcriber = LiveTranscriber()
        
        # Transcribe numpy audio
        text = transcriber.transcribe(audio_array)
        
        # Transcribe file
        text = transcriber.transcribe_file(Path("audio.wav"))
    """
    
    WHISPER_SAMPLE_RATE = 16000
    
    def __init__(self, model_size: str = None, device: str = None):
        """
        Initialize transcriber.
        
        Args:
            model_size: Whisper model size
            device: Device ('cpu' or 'cuda')
        """
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.whisper_device
        self.use_colab = settings.use_colab_gpu and settings.colab_api_url
        
        self.model = None
        
        if self.use_colab:
            logger.info(f"LiveTranscriber will use Colab GPU: {settings.colab_api_url}")
        else:
            logger.info(f"LiveTranscriber initialized: {self.model_size} on {self.device}")
    
    def _ensure_model_loaded(self):
        """Lazy load Whisper model."""
        if self.use_colab or self.model is not None:
            return
        
        import whisper
        logger.info(f"Loading Whisper {self.model_size} on {self.device}...")
        self.model = whisper.load_model(self.model_size, device=self.device)
        logger.info("Whisper model loaded")
    
    def resample(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        """Resample audio to 16kHz for Whisper."""
        if source_rate == self.WHISPER_SAMPLE_RATE:
            return audio
        
        from scipy import signal
        num_samples = int(len(audio) * self.WHISPER_SAMPLE_RATE / source_rate)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)
    
    def transcribe(self, audio: np.ndarray, source_rate: int = 48000, 
                   language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio array.
        
        Args:
            audio: Audio as numpy float32 array
            source_rate: Audio sample rate
            language: Optional language code
            
        Returns:
            Transcription result dict with 'text', 'segments', 'language'
        """
        if len(audio) == 0:
            return {'text': '', 'segments': [], 'language': ''}
        
        # Resample to 16kHz
        audio_16k = self.resample(audio, source_rate)
        
        if self.use_colab:
            return self._transcribe_colab(audio_16k, language)
        
        self._ensure_model_loaded()
        
        result = self.model.transcribe(
            audio_16k,
            language=language if language != 'auto' else None,
            temperature=0.0,
            condition_on_previous_text=False,
            fp16=(self.device == 'cuda')
        )
        
        return {
            'text': result['text'].strip(),
            'segments': result.get('segments', []),
            'language': result.get('language', '')
        }
    
    def _transcribe_colab(self, audio: np.ndarray, language: str = None) -> Dict[str, Any]:
        """Transcribe via Colab GPU."""
        import requests
        
        try:
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio, self.WHISPER_SAMPLE_RATE)
            
            # Load model on Colab if needed
            load_url = f"{settings.colab_api_url}/load_whisper"
            requests.post(load_url, json={'model_size': self.model_size}, timeout=60)
            
            # Transcribe
            url = f"{settings.colab_api_url}/whisper/transcribe"
            with open(temp_path, 'rb') as f:
                response = requests.post(
                    url, 
                    files={'audio': f}, 
                    data={'language': language or 'auto'},
                    timeout=120
                )
            
            os.unlink(temp_path)
            
            if response.status_code == 200:
                result = response.json().get('result', {})
                return {
                    'text': result.get('text', '').strip(),
                    'segments': result.get('segments', []),
                    'language': result.get('language', '')
                }
            else:
                raise Exception(f"Colab API error: {response.text}")
                
        except Exception as e:
            logger.error(f"Colab transcription failed: {e}")
            # Fallback to local
            self.use_colab = False
            self._ensure_model_loaded()
            return self.transcribe(audio, self.WHISPER_SAMPLE_RATE, language)
    
    def transcribe_file(self, filepath: Path) -> Dict[str, Any]:
        """Transcribe audio file."""
        audio, sr = sf.read(filepath, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.transcribe(audio, source_rate=sr)


class StreamingPipeline:
    """
    Combines AudioStreamer + LiveTranscriber for real-time transcription.
    
    Usage:
        pipeline = StreamingPipeline()
        pipeline.start(device=device_index)
        
        for result in pipeline.stream():
            print(result['text'])
        
        pipeline.stop()
    """
    
    def __init__(self, 
                 interval_seconds: float = 2.0,
                 sample_rate: int = 48000,
                 accumulate_chunks: int = 1):
        """
        Initialize streaming pipeline.
        
        Args:
            interval_seconds: Seconds per chunk
            sample_rate: Audio sample rate
            accumulate_chunks: Number of chunks to accumulate before transcribing
        """
        self.streamer = AudioStreamer(
            interval_seconds=interval_seconds,
            sample_rate=sample_rate
        )
        self.transcriber = LiveTranscriber()
        self.sample_rate = sample_rate
        self.accumulate_chunks = accumulate_chunks
        
        self._running = False
        self._callback: Optional[Callable] = None
    
    def start(self, device: Optional[int] = None, callback: Callable = None):
        """
        Start streaming pipeline.
        
        Args:
            device: Audio device index
            callback: Optional callback for transcription results
        """
        self._running = True
        self._callback = callback
        self.streamer.start(device=device)
        logger.info("Streaming pipeline started")
    
    def stop(self):
        """Stop streaming pipeline."""
        self._running = False
        self.streamer.stop()
        logger.info("Streaming pipeline stopped")
    
    def stream(self):
        """Generator that yields transcription results."""
        audio_buffer = np.array([], dtype=np.float32)
        chunks_accumulated = 0
        
        while self._running:
            chunk = self.streamer.pop_as_numpy(timeout=0.5)
            
            if chunk is not None:
                audio_buffer = np.concatenate([audio_buffer, chunk])
                chunks_accumulated += 1
                
                if chunks_accumulated >= self.accumulate_chunks:
                    # Transcribe accumulated audio
                    if len(audio_buffer) > 0:
                        result = self.transcriber.transcribe(
                            audio_buffer, 
                            source_rate=self.sample_rate
                        )
                        
                        if result['text']:
                            yield result
                            
                            if self._callback:
                                self._callback(result)
                    
                    # Reset buffer
                    audio_buffer = np.array([], dtype=np.float32)
                    chunks_accumulated = 0
    
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """List available audio devices."""
        return self.streamer.list_devices()
