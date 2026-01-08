"""
Voice cloning module using OpenVoice (lighter, more stable than XTTS)
Also includes edge-tts for high-quality non-cloned TTS

Supports both local CPU/GPU and Colab GPU via USE_COLAB_GPU setting
"""
import os
import asyncio
import requests
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


class VoiceCloner:
    """
    Voice synthesis using edge-tts (high quality) or OpenVoice (voice cloning)
    
    Supports:
    - edge-tts: High quality Microsoft TTS (no cloning, but natural voices)
    - OpenVoice: Voice cloning (when available)
    - Colab GPU processing (when USE_COLAB_GPU=True)
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
    
    def __init__(self, device: str = None):
        """
        Initialize voice synthesizer
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device or settings.tts_device
        self.use_colab = settings.use_colab_gpu and settings.colab_api_url
        
        if self.use_colab:
            logger.info(f"VoiceCloner initialized with Colab GPU: {settings.colab_api_url}")
        else:
            logger.info(f"VoiceCloner initialized (edge-tts) on {self.device}")
    
    def get_voice_for_language(self, language: str) -> str:
        """Get appropriate edge-tts voice for language"""
        return self.EDGE_VOICES.get(language, 'en-US-AriaNeural')
    
    async def _generate_edge_tts(self, text: str, output_path: Path, language: str = "en"):
        """Generate speech using edge-tts (async)"""
        voice = self.get_voice_for_language(language)
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
    
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
            reference_audio: Optional reference audio for voice cloning (Colab only)
            
        Returns:
            Path to generated audio
        """
        if not text.strip():
            logger.warning("Empty text, skipping TTS")
            return None
        
        # Use Colab for voice cloning if available
        if self.use_colab and reference_audio:
            return self._generate_colab(text, output_path, language, reference_audio)
        
        # Use edge-tts locally
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("edge-tts not installed. Install with: pip install edge-tts")
        
        try:
            logger.info(f"Generating speech: {text[:50]}...")
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
            reference_audio: Original audio (for Colab voice cloning)
            output_path: Path to save final audio
            language: Target language code
            
        Returns:
            Path to dubbed audio file
        """
        from pydub import AudioSegment
        
        logger.info(f"Generating dubbed audio for {len(segments)} segments...")
        
        # Create temp directory for segment audio
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            segment_audios = []
            
            for i, segment in enumerate(segments):
                text = segment.get('translated', segment.get('text', ''))
                
                if not text.strip():
                    segment_audios.append(None)
                    continue
                
                audio_path = temp_dir / f"segment_{i:04d}.mp3"
                
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
    
    def _generate_colab(self,
                       text: str,
                       output_path: Path,
                       language: str,
                       reference_audio: Path) -> Path:
        """Generate speech using Colab GPU (with voice cloning)"""
        try:
            url = f"{settings.colab_api_url}/tts/generate"
            
            files = {'reference_audio': open(reference_audio, 'rb')} if reference_audio else {}
            data = {
                'text': text,
                'language': language
            }
            
            response = requests.post(url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return output_path
            else:
                raise Exception(f"Colab TTS error: {response.text}")
                
        except Exception as e:
            logger.error(f"Colab TTS failed: {e}, falling back to edge-tts")
            self.use_colab = False
            return self.generate_speech(text, output_path, language, None)


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
