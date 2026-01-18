"""
Translation module using Facebook's NLLB-200-1.3B model (GPU-accelerated)
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any
from pathlib import Path
import json
from utils.logger import setup_logger
from config import settings

logger = setup_logger("translator")


# NLLB-200 language code mapping (BCP-47 format)
NLLB_LANG_CODES = {
    # European languages
    'english': 'eng_Latn',
    'spanish': 'spa_Latn',
    'french': 'fra_Latn',
    'german': 'deu_Latn',
    'italian': 'ita_Latn',
    'portuguese': 'por_Latn',
    'dutch': 'nld_Latn',
    'polish': 'pol_Latn',
    'russian': 'rus_Cyrl',
    'ukrainian': 'ukr_Cyrl',
    'greek': 'ell_Grek',
    'turkish': 'tur_Latn',
    
    # Asian languages
    'japanese': 'jpn_Jpan',
    'korean': 'kor_Hang',
    'chinese': 'zho_Hans',
    'chinese_traditional': 'zho_Hant',
    'vietnamese': 'vie_Latn',
    'thai': 'tha_Thai',
    'indonesian': 'ind_Latn',
    'malay': 'zsm_Latn',
    
    # South Asian languages
    'hindi': 'hin_Deva',
    'tamil': 'tam_Taml',
    'telugu': 'tel_Telu',
    'bengali': 'ben_Beng',
    'marathi': 'mar_Deva',
    'gujarati': 'guj_Gujr',
    'kannada': 'kan_Knda',
    'malayalam': 'mal_Mlym',
    'punjabi': 'pan_Guru',
    'urdu': 'urd_Arab',
    
    # Middle Eastern languages
    'arabic': 'arb_Arab',
    'hebrew': 'heb_Hebr',
    'persian': 'pes_Arab',
    
    # African languages
    'swahili': 'swh_Latn',
    'amharic': 'amh_Ethi',
}


class Translator:
    """Translator using Facebook's NLLB-200-1.3B model (GPU-accelerated)"""
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize NLLB translator on GPU"""
        if Translator._model is not None:
            return  # Already initialized
            
        self.model_name = settings.translation_model
        self.device = settings.translation_device
        self.max_length = settings.translation_max_length
        
        logger.info(f"Loading NLLB model: {self.model_name} on {self.device}")
        
        try:
            Translator._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            Translator._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Set model to eval mode
            Translator._model.eval()
            
            logger.info(f"NLLB Translator initialized successfully on {self.device}")
            
            # Warm up the model
            self._warmup()
            
        except Exception as e:
            logger.error(f"Failed to load NLLB model: {e}")
            raise
    
    @property
    def tokenizer(self):
        return Translator._tokenizer
    
    @property
    def model(self):
        return Translator._model
    
    def _warmup(self):
        """Warm up the model with a simple translation"""
        try:
            logger.info("Warming up NLLB model...")
            self.translate_text("Hello", "english", "spanish")
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _get_nllb_code(self, language: str) -> str:
        """Convert language name to NLLB code"""
        lang_lower = language.lower().strip()
        
        if lang_lower in NLLB_LANG_CODES:
            return NLLB_LANG_CODES[lang_lower]
        
        # Check if it's already an NLLB code
        if '_' in lang_lower and len(lang_lower) == 8:
            return lang_lower
        
        # Default to English
        logger.warning(f"Unknown language '{language}', defaulting to English")
        return 'eng_Latn'
    
    def translate_text(self, text: str, source_language: str, target_language: str) -> str:
        """
        Translate text using NLLB-200
        
        Args:
            text: Text to translate
            source_language: Source language (e.g., 'english', 'tamil')
            target_language: Target language (e.g., 'spanish', 'french')
            
        Returns:
            Translated text
        """
        try:
            if not text or not text.strip():
                return text
            
            src_code = self._get_nllb_code(source_language)
            tgt_code = self._get_nllb_code(target_language)
            
            # Set source language for tokenizer
            self.tokenizer.src_lang = src_code
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                    max_length=self.max_length,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True
                )
            
            # Decode output
            translated = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Return original if translation fails
    
    def translate_segments(self, segments: List[Dict[str, Any]], 
                          target_language: str,
                          source_language: str = "auto") -> List[Dict[str, Any]]:
        """
        Translate segments
        
        Args:
            segments: List of segments with 'refined' or 'text' content
            target_language: Target language
            source_language: Source language (default: auto-detect from first segment)
            
        Returns:
            List of translated segments
        """
        logger.info(f"Translating {len(segments)} segments to {target_language}")
        
        # If source is auto, try to detect from content or default to English
        if source_language == "auto":
            source_language = "english"  # Default assumption
            logger.info(f"Source language set to: {source_language}")
        
        translated_segments = []
        
        for i, segment in enumerate(segments):
            # Use refined text if available, otherwise original
            text_to_translate = segment.get('refined', segment.get('text', ''))
            
            translated_text = self.translate_text(
                text_to_translate, 
                source_language, 
                target_language
            )
            
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
        full_text = " ".join([seg['translated'] for seg in translated_segments if seg['translated']])
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        logger.info(f"Saved translation text: {txt_path}")
        
        return json_path, txt_path
