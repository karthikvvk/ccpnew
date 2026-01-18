"""
Voice cloning module using XTTS-v2 (Coqui TTS)
Falls back to edge-tts for high-quality non-cloned TTS (GPU-only setup)
"""
import os
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from utils.logger import setup_logger
from config import settings

logger = setup_logger("voice_cloner")

# Try to import edge_tts (always available, stable)
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts not installed. Install with: pip install edge-tts")

# Try to import XTTS-v2 (Coqui TTS)
try:
    from TTS.api import TTS
    import torch
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    logger.info("TTS (XTTS-v2) not installed. Install with: pip install TTS")


class VoiceCloner:
    """
    Voice synthesis using XTTS-v2 (voice cloning) or edge-tts (fallback)
    GPU-only setup for maximum performance
    
    Supports:
    - XTTS-v2: Voice cloning from reference audio (GPU)
    - edge-tts: High quality Microsoft TTS (no cloning, but natural voices)
    """
    
    # Edge TTS voice mapping by language
    EDGE_VOICES = {
        'en': 'en-US-AriaNeural',
        'es': 'es-ES-AlvaroNeural',
        'fr': 'fr-FR-DeniseNeural',
        'de': 'de-DE-ConradNeural',
        'it': 'it-IT-DiegoNeural',
        'pt': 'pt-BR-FranciscaNeural',
        'ru': 'ru-RU-DmitryNeural',
        'ja': 'ja-JP-NanamiNeural',
        'ko': 'ko-KR-InJoonNeural',
        'zh-cn': 'zh-CN-XiaoxiaoNeural',
        'ar': 'ar-SA-HamedNeural',
        'hi': 'hi-IN-MadhurNeural',
        'ta': 'ta-IN-PallaviNeural',
        'te': 'te-IN-ShrutiNeural'
    }
    
    # XTTS-v2 supported languages
    XTTS_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 
                      'nl', 'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko', 'hi']
    
    def __init__(self, device: str = None):
        """
        Initialize voice synthesizer
        
        Args:
            device: Device to run on ('cuda')
        """
        self.device = device or settings.tts_device
        self.xtts_model = None
        
        # Initialize XTTS-v2 if voice cloning is enabled and available
        if settings.use_voice_cloning:
            if XTTS_AVAILABLE:
                self._load_xtts_model()
            else:
                logger.warning("XTTS-v2 not available. Will use edge-tts fallback.")
        
        if self.xtts_model:
            logger.info(f"VoiceCloner initialized with XTTS-v2 on {self.device}")
        else:
            logger.info(f"VoiceCloner initialized (edge-tts) on {self.device}")
    
    def _load_xtts_model(self):
        """Load XTTS-v2 model"""
        try:
            logger.info("Loading XTTS-v2 model (this may take a while on first run)...")
            self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Move to GPU if available
            if self.device == "cuda" and torch.cuda.is_available():
                self.xtts_model.to("cuda")
                logger.info("XTTS-v2 loaded on GPU")
            else:
                logger.info("XTTS-v2 loaded on CPU (will be slow)")
        except Exception as e:
            logger.error(f"Failed to load XTTS-v2: {e}")
            self.xtts_model = None
    
    def get_voice_for_language(self, language: str) -> str:
        """Get appropriate edge-tts voice for language"""
        return self.EDGE_VOICES.get(language, 'en-US-AriaNeural')
    
    def _is_xtts_language_supported(self, language: str) -> bool:
        """Check if language is supported by XTTS-v2"""
        return language in self.XTTS_LANGUAGES
    
    async def _generate_edge_tts(self, text: str, output_path: Path, language: str = "en"):
        """Generate speech using edge-tts (async)"""
        voice = self.get_voice_for_language(language)
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
    
    def _generate_xtts(self, text: str, output_path: Path, language: str, reference_audio: Path) -> Path:
        """
        Generate speech using XTTS-v2 voice cloning
        
        Args:
            text: Text to speak
            output_path: Path to save audio
            language: Target language code
            reference_audio: Reference audio for voice cloning
            
        Returns:
            Path to generated audio
        """
        try:
            logger.info(f"Generating XTTS-v2 speech: {text[:50]}...")
            
            # Map language codes for XTTS
            xtts_lang = language
            if language == 'zh-cn':
                xtts_lang = 'zh-cn'
            elif language == 'ta' or language == 'te':
                # Tamil/Telugu not supported by XTTS, fall back to Hindi
                xtts_lang = 'hi'
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate with voice cloning
            self.xtts_model.tts_to_file(
                text=text,
                speaker_wav=str(reference_audio),
                language=xtts_lang,
                file_path=str(output_path)
            )
            
            logger.info(f"XTTS audio saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"XTTS generation failed: {e}")
            raise
    
    def generate_speech(self,
                       text: str,
                       output_path: Path,
                       language: str = "en",
                       reference_audio: Path = None) -> Path:
        """
        Generate speech from text
        
        Args:
            text: Text to speak
            output_path: Path to save audio
            language: Target language code
            reference_audio: Optional reference audio for voice cloning
            
        Returns:
            Path to generated audio
        """
        if not text.strip():
            logger.warning("Empty text, skipping TTS")
            return None
        
        # Use XTTS-v2 for voice cloning if available and reference provided
        if self.xtts_model and reference_audio and settings.use_voice_cloning:
            try:
                return self._generate_xtts(text, output_path, language, reference_audio)
            except Exception as e:
                logger.warning(f"XTTS failed, falling back to edge-tts: {e}")
        
        # Fallback to edge-tts
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts not installed. Install with: pip install edge-tts")
        
        try:
            logger.info(f"Generating edge-tts speech: {text[:50]}...")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run async edge-tts
            asyncio.run(self._generate_edge_tts(text, output_path, language))
            
            logger.info(f"Audio saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            raise
    
    def segments_to_dubbed_audio(self,
                                 segments: List[Dict[str, Any]],
                                 reference_audio: Path,
                                 output_path: Path,
                                 language: str = "en") -> Path:
        """
        Generate dubbed audio from translated segments
        
        Args:
            segments: List of translated segments
            reference_audio: Original audio (for voice cloning)
            output_path: Path to save final audio
            language: Target language code
            
        Returns:
            Path to dubbed audio file
        """
        from pydub import AudioSegment
        
        logger.info(f"Generating dubbed audio for {len(segments)} segments...")
        
        # Log which TTS method will be used
        if self.xtts_model and settings.use_voice_cloning:
            logger.info("Using local XTTS-v2 for voice cloning")
        else:
            logger.info("Using edge-tts (no voice cloning)")
        
        # Create temp directory for segment audio
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            segment_audios = []
            
            for i, segment in enumerate(segments):
                text = segment.get('translated', segment.get('text', ''))
                
                if not text.strip():
                    segment_audios.append(None)
                    continue
                
                audio_path = temp_dir / f"segment_{i:04d}.wav"
                
                try:
                    self.generate_speech(
                        text=text,
                        output_path=audio_path,
                        language=language,
                        reference_audio=reference_audio
                    )
                    segment_audios.append(audio_path)
                except Exception as e:
                    logger.error(f"Segment {i} TTS failed: {e}")
                    segment_audios.append(None)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Generated {i + 1}/{len(segments)} segments")
            
            # Combine segments
            combined = AudioSegment.empty()
            
            for i, (segment, audio_path) in enumerate(zip(segments, segment_audios)):
                if audio_path and audio_path.exists():
                    seg_audio = AudioSegment.from_file(str(audio_path))
                    
                    # Add silence to maintain timing
                    start_ms = int(segment['start'] * 1000)
                    current_duration = len(combined)
                    
                    if start_ms > current_duration:
                        silence = start_ms - current_duration
                        combined += AudioSegment.silent(duration=silence)
                    
                    combined += seg_audio
                else:
                    # Silence for failed segments
                    duration_ms = int((segment['end'] - segment['start']) * 1000)
                    combined += AudioSegment.silent(duration=min(duration_ms, 5000))
            
            # Export
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export as WAV for video reconstruction
            combined.export(str(output_path), format="wav")
            
            logger.info(f"Dubbed audio saved: {output_path}")
            return output_path


# Language code mapping
LANGUAGE_CODES = {
    'english': 'en',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'italian': 'it',
    'portuguese': 'pt',
    'russian': 'ru',
    'japanese': 'ja',
    'korean': 'ko',
    'chinese': 'zh-cn',
    'arabic': 'ar',
    'hindi': 'hi',
    'tamil': 'ta',
    'telugu': 'te'
}


def get_language_code(language: str) -> str:
    """Convert language name to code"""
    return LANGUAGE_CODES.get(language.lower(), 'en')
