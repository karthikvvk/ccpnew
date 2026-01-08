"""
Speech-to-text module using local Whisper
Can run locally (CPU) or in Colab (GPU) for faster processing
"""
import whisper
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from utils.logger import setup_logger
from config import settings

logger = setup_logger("speech_to_text")


class SpeechToText:
    """
    Whisper-based speech recognition with RAG context enhancement
    """
    
    def __init__(self, model_size: str = None, device: str = None):
        """
        Initialize speech-to-text
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v3)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.whisper_device
        self.use_colab = settings.use_colab_gpu and settings.colab_api_url
        self.model = None
        
        # Only load local model if NOT using Colab
        if self.use_colab:
            logger.info(f"Whisper will use Colab GPU: {settings.colab_api_url}")
            logger.info(f"Model: {self.model_size} (loaded on Colab, not locally)")
        else:
            logger.info(f"Loading Whisper model locally: {self.model_size} on {self.device}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
    
    def transcribe(self, 
                   audio_path: Path, 
                   language: str = None,
                   initial_prompt: str = None) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            initial_prompt: Optional prompt to guide transcription
            
        Returns:
            Transcription result dictionary
        """
        # Check if Colab GPU is enabled - use it if available
        if self.use_colab:
            return self._transcribe_colab(audio_path, language)
        
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Determine language
            lang = language if language and language != 'auto' else None
            
            # Optimized transcription parameters (matching Colab precision)
            # These settings significantly improve accuracy, especially for non-English
            result = self.model.transcribe(
                str(audio_path),
                language=lang,
                initial_prompt=initial_prompt,
                verbose=False,
                # Precision settings
                temperature=0.0,              # Deterministic output, no randomness
                word_timestamps=True,         # Get word-level timing
                condition_on_previous_text=False,  # Prevents hallucination/repetition
                fp16=(self.device == "cuda")  # FP16 for GPU only
            )
            
            logger.info(f"Transcription completed. Detected language: {result.get('language', 'unknown')}")
            logger.info(f"Transcribed {len(result.get('segments', []))} segments")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            raise
    
    def _transcribe_colab(self, audio_path: Path, language: str = None) -> Dict[str, Any]:
        """
        Transcribe using Colab GPU API
        
        Args:
            audio_path: Path to audio file
            language: Language code or None
            
        Returns:
            Transcription result
        """
        import requests
        
        try:
            logger.info(f"Transcribing via Colab GPU: {audio_path}")
            
            # First, ensure Whisper model is loaded on Colab
            load_url = f"{settings.colab_api_url}/load_whisper"
            load_response = requests.post(load_url, json={'model_size': self.model_size})
            
            if load_response.status_code == 200:
                logger.info("Whisper model loaded on Colab GPU")
            
            # Send audio for transcription
            url = f"{settings.colab_api_url}/whisper/transcribe"
            
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                data = {'language': language or 'auto'}
                
                response = requests.post(url, files=files, data=data, timeout=600)
            
            if response.status_code == 200:
                result = response.json()['result']
                logger.info(f"Colab transcription completed. Language: {result.get('language', 'unknown')}")
                return result
            else:
                logger.error(f"Colab API error: {response.text}")
                raise Exception(f"Colab API failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to transcribe via Colab: {e}")
            logger.info("Falling back to local transcription...")
            
            # Disable Colab for this instance
            self.use_colab = False
            
            # Load local model if not already loaded
            if self.model is None:
                logger.info(f"Loading Whisper model locally: {self.model_size} on {self.device}")
                self.model = whisper.load_model(self.model_size, device=self.device)
                logger.info("Whisper model loaded successfully")
            
            # Now transcribe locally
            return self.transcribe(audio_path, language)
    
    def transcribe_with_context(self,
                               audio_path: Path,
                               visual_context: str = None,
                               language: str = None) -> Dict[str, Any]:
        """
        Transcribe with visual context from RAG
        
        Args:
            audio_path: Path to audio file
            visual_context: Visual context from RAG query
            language: Language code or None for auto-detect
            
        Returns:
            Transcription result
        """
        # Use visual context as initial prompt
        initial_prompt = None
        if visual_context:
            initial_prompt = f"Visual context: {visual_context[:200]}"  # Limit length
            logger.info(f"Using visual context for transcription")
        
        return self.transcribe(audio_path, language, initial_prompt)
    
    def save_transcription(self, 
                          result: Dict[str, Any], 
                          json_path: Path,
                          txt_path: Path) -> tuple[Path, Path]:
        """
        Save transcription to JSON and TXT files
        
        Args:
            result: Transcription result from Whisper
            json_path: Path to save JSON
            txt_path: Path to save plain text
            
        Returns:
            Tuple of (json_path, txt_path)
        """
        # Save JSON with full details
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved transcription JSON: {json_path}")
        
        # Save plain text
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        logger.info(f"Saved transcription text: {txt_path}")
        
        return json_path, txt_path
    
    def get_segments(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract segments with timing information
        
        Args:
            result: Transcription result
            
        Returns:
            List of segments with start, end, and text
        """
        segments = []
        for seg in result.get('segments', []):
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip()
            })
        
        return segments


# Standalone function for Colab usage
def transcribe_audio_gpu(audio_path: str,
                        output_json: str,
                        output_txt: str,
                        model_size: str = "medium",
                        language: str = None,
                        device: str = "cuda") -> dict:
    """
    Transcribe audio using GPU - Colab compatible
    
    Args:
        audio_path: Path to audio file
        output_json: Path to save JSON
        output_txt: Path to save text
        model_size: Whisper model size
        language: Language code or None
        device: Device ('cuda' for GPU)
        
    Returns:
        Transcription result
    """
    print(f"Loading Whisper {model_size} on {device}...")
    model = whisper.load_model(model_size, device=device)
    
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(
        audio_path,
        language=language if language != 'auto' else None,
        verbose=True
    )
    
    # Save JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save text
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(result['text'])
    
    print(f"Transcription saved to {output_json} and {output_txt}")
    return result
