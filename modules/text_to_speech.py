"""
Text-to-speech module using gTTS
"""
from gtts import gTTS
from pathlib import Path
from typing import List, Dict, Any
from pydub import AudioSegment
import tempfile
from utils.logger import setup_logger
from config import settings

logger = setup_logger("text_to_speech")


class TextToSpeech:
    """
    Text-to-speech conversion using gTTS
    """
    
    def __init__(self, language: str = None, slow: bool = None):
        """
        Initialize text-to-speech
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
            slow: Whether to use slow speech
        """
        self.language = language or settings.tts_lang
        self.slow = slow if slow is not None else settings.tts_slow
        
        logger.info(f"Initialized TTS with language: {self.language}")
    
    def text_to_speech(self, text: str, output_path: Path) -> Path:
        """
        Convert text to speech
        
        Args:
            text: Text to convert
            output_path: Path to save audio file
            
        Returns:
            Path to audio file
        """
        try:
            logger.info(f"Converting text to speech: {len(text)} characters")
            
            # Create TTS
            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tts.save(str(output_path))
            
            logger.info(f"Saved speech audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert text to speech: {e}")
            raise
    
    def segments_to_speech(self,
                          segments: List[Dict[str, Any]],
                          output_path: Path,
                          text_key: str = 'translated') -> Path:
        """
        Convert segments to speech with timing
        
        Args:
            segments: List of segments with timing and text
            text_key: Key to get text from segment ('translated' or 'text')
            output_path: Path to save combined audio
            
        Returns:
            Path to audio file
        """
        try:
            logger.info(f"Converting {len(segments)} segments to speech")
            
            # Create temporary directory for segment audio files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Generate audio for each segment
                segment_audios = []
                for i, segment in enumerate(segments):
                    text = segment.get(text_key, '')
                    if not text.strip():
                        continue
                    
                    # Generate audio
                    temp_audio_path = temp_dir_path / f"segment_{i:04d}.mp3"
                    tts = gTTS(text=text, lang=self.language, slow=self.slow)
                    tts.save(str(temp_audio_path))
                    
                    # Load audio segment
                    audio_seg = AudioSegment.from_mp3(str(temp_audio_path))
                    segment_audios.append(audio_seg)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(segments)} segments")
                
                # Combine all segments
                if segment_audios:
                    combined_audio = sum(segment_audios)
                    
                    # Save combined audio
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    combined_audio.export(str(output_path), format='mp3')
                    
                    logger.info(f"Saved combined speech audio: {output_path}")
                else:
                    logger.warning("No audio segments were generated")
                    raise ValueError("No valid segments to convert")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert segments to speech: {e}")
            raise
    
    def adjust_audio_duration(self,
                             audio_path: Path,
                             target_duration: float,
                             output_path: Path) -> Path:
        """
        Adjust audio duration to match target
        
        Args:
            audio_path: Path to input audio
            target_duration: Target duration in seconds
            output_path: Path to save adjusted audio
            
        Returns:
            Path to adjusted audio
        """
        try:
            logger.info(f"Adjusting audio duration to {target_duration}s")
            
            # Load audio
            audio = AudioSegment.from_file(str(audio_path))
            
            current_duration = len(audio) / 1000.0  # Convert to seconds
            
            # Calculate speed change needed
            speed_factor = current_duration / target_duration
            
            # Adjust speed (if reasonable)
            if 0.5 <= speed_factor <= 2.0:
                # Change speed
                adjusted_audio = audio._spawn(
                    audio.raw_data,
                    overrides={'frame_rate': int(audio.frame_rate * speed_factor)}
                )
                adjusted_audio = adjusted_audio.set_frame_rate(audio.frame_rate)
            else:
                logger.warning(f"Speed factor {speed_factor} out of range, using original")
                adjusted_audio = audio
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            adjusted_audio.export(str(output_path), format='mp3')
            
            logger.info(f"Saved adjusted audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to adjust audio duration: {e}")
            raise
