"""
Translation module using Google Translate API (simple, reliable)
"""
from googletrans import Translator as GoogleTranslator
from typing import List, Dict, Any
from pathlib import Path
import json
from utils.logger import setup_logger

logger = setup_logger("translator")


class Translator:
    """Simple translator using Google Translate"""
    
    def __init__(self):
        """Initialize translator"""
        self.translator = GoogleTranslator()
        logger.info("Google Translator initialized")
    
    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language (e.g., 'Spanish', 'French')
            
        Returns:
            Translated text
        """
        try:
            # Map language names to codes
            lang_map = {
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
                'english': 'en'
            }
            
            dest_lang = lang_map.get(target_language.lower(), 'en')
            
            result = self.translator.translate(text, dest=dest_lang)
            return result.text
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Return original if translation fails
    
    def translate_segments(self, segments: List[Dict[str, Any]], 
                          target_language: str) -> List[Dict[str, Any]]:
        """
        Translate segments
        
        Args:
            segments: List of segments with 'refined' text
            target_language: Target language
            
        Returns:
            List of translated segments
        """
        logger.info(f"Translating {len(segments)} segments to {target_language}")
        
        translated_segments = []
        
        for i, segment in enumerate(segments):
            # Use refined text if available, otherwise original
            text_to_translate = segment.get('refined', segment.get('text', ''))
            
            translated_text = self.translate_text(text_to_translate, target_language)
            
            translated_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'original': segment.get('original', segment.get('text', '')),
                'refined': segment.get('refined', ''),
                'translated': translated_text
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Translated {i + 1}/{len(segments)} segments")
        
        logger.info(f"Translation completed for all {len(segments)} segments")
        return translated_segments
    
    def save_translation(self, translated_segments: List[Dict[str, Any]],
                        json_path: Path, txt_path: Path) -> tuple[Path, Path]:
        """Save translation to files"""
        # Save JSON
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(translated_segments, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved translation JSON: {json_path}")
        
        # Save text
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        full_text = " ".join([seg['translated'] for seg in translated_segments])
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        logger.info(f"Saved translation text: {txt_path}")
        
        return json_path, txt_path
