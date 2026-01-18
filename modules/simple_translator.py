"""
Translation module using NLLB-200 (No Language Left Behind)
Replaces Google Translate with Meta's multilingual translation model for better context understanding.

Supports:
- Local CPU/GPU processing
- Colab GPU offloading
- Fallback to Google Translate if LLM fails
"""

import torch
from typing import List, Dict, Any
from pathlib import Path
import json

from utils.logger import setup_logger
from config import settings

logger = setup_logger("translator")

# NLLB Language codes
NLLB_LANG_CODES = {
    'english': 'eng_Latn', 'en': 'eng_Latn',
    'tamil': 'tam_Taml', 'ta': 'tam_Taml',
    'hindi': 'hin_Deva', 'hi': 'hin_Deva',
    'spanish': 'spa_Latn', 'es': 'spa_Latn',
    'french': 'fra_Latn', 'fr': 'fra_Latn',
    'german': 'deu_Latn', 'de': 'deu_Latn',
    'chinese': 'zho_Hans', 'zh': 'zho_Hans',
    'japanese': 'jpn_Jpan', 'ja': 'jpn_Jpan',
    'korean': 'kor_Hang', 'ko': 'kor_Hang',
    'arabic': 'arb_Arab', 'ar': 'arb_Arab',
    'portuguese': 'por_Latn', 'pt': 'por_Latn',
    'russian': 'rus_Cyrl', 'ru': 'rus_Cyrl',
    'telugu': 'tel_Telu', 'te': 'tel_Telu',
    'kannada': 'kan_Knda', 'kn': 'kan_Knda',
    'malayalam': 'mal_Mlym', 'ml': 'mal_Mlym',
    'bengali': 'ben_Beng', 'bn': 'ben_Beng',
}


def get_nllb_code(language: str) -> str:
    """Convert language name/code to NLLB format."""
    return NLLB_LANG_CODES.get(language.lower().strip(), 'eng_Latn')


class Translator:
    """
    Translator using NLLB-200 for high-quality multilingual translation.
    Falls back to Google Translate if LLM fails.
    """
    
    def __init__(self):
        """Initialize translator."""
        self.model_name = settings.llm_model  # Now NLLB-200
        self.device = settings.llm_device
        self.use_colab = settings.use_colab_gpu and settings.colab_api_url
        
        # Lazy load
        self.model = None
        self.tokenizer = None
        
        if self.use_colab:
            logger.info(f"Translator will use Colab GPU: {settings.colab_api_url}")
        else:
            logger.info(f"Translator initialized: {self.model_name} on {self.device}")
    
    def _ensure_model_loaded(self):
        """Lazy load NLLB model."""
        if self.use_colab or self.model is not None:
            return
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        logger.info(f"Loading NLLB model: {self.model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Use float16 for GPU, float32 for CPU
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=None
        )
        
        if self.device == 'cuda':
            self.model = self.model.to('cuda')
        
        logger.info("NLLB model loaded successfully")
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using NLLB-200.
        
        Args:
            text: Text to translate
            source_lang: Source language code/name
            target_lang: Target language code/name
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        src_code = get_nllb_code(source_lang)
        tgt_code = get_nllb_code(target_lang)
        
        if self.use_colab:
            return self._translate_colab(text, src_code, tgt_code)
        
        return self._translate_local(text, src_code, tgt_code)
    
    def _translate_local(self, text: str, src_code: str, tgt_code: str) -> str:
        """Translate locally using NLLB."""
        self._ensure_model_loaded()
        
        try:
            self.tokenizer.src_lang = src_code
            
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            if self.device == 'cuda':
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            tgt_lang_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_token_id,
                    num_beams=8,
                    max_length=256
                )
            
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated.strip()
            
        except Exception as e:
            logger.error(f"NLLB translation failed: {e}")
            return self._fallback_translate(text, tgt_code)
    
    def _translate_colab(self, text: str, src_code: str, tgt_code: str) -> str:
        """Translate via Colab GPU."""
        import requests
        
        try:
            # Ensure model loaded on Colab
            load_url = f"{settings.colab_api_url}/load_llm"
            requests.post(load_url, json={'model_name': self.model_name}, timeout=300)
            
            # Translate
            url = f"{settings.colab_api_url}/llm/translate"
            payload = {
                'text': text,
                'source_lang': src_code,
                'target_lang': tgt_code
            }
            
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code == 200:
                return response.json().get('translated', '')
            else:
                raise Exception(f"Colab API error: {response.text}")
                
        except Exception as e:
            logger.error(f"Colab translation failed: {e}")
            logger.info("Falling back to local translation...")
            self.use_colab = False
            return self._translate_local(text, src_code, tgt_code)
    
    def _fallback_translate(self, text: str, target_code: str) -> str:
        """Fallback to Google Translate."""
        try:
            from deep_translator import GoogleTranslator
            
            # Convert NLLB code to simple code
            simple_code = target_code.split('_')[0]
            if simple_code == 'eng':
                simple_code = 'en'
            
            translator = GoogleTranslator(source='auto', target=simple_code)
            return translator.translate(text)
        except Exception as e:
            logger.error(f"Fallback translation also failed: {e}")
            return text
    
    def translate_segments(self, segments: List[Dict[str, Any]], 
                          source_lang: str,
                          target_lang: str) -> List[Dict[str, Any]]:
        """
        Translate multiple segments.
        
        Args:
            segments: List of segments with 'refined' or 'text'
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Segments with 'translated' field
        """
        logger.info(f"Translating {len(segments)} segments: {source_lang} → {target_lang}")
        
        translated_segments = []
        
        for i, segment in enumerate(segments):
            text = segment.get('refined') or segment.get('text') or segment.get('original', '')
            
            if not text.strip():
                translated_text = ""
            else:
                translated_text = self.translate_text(text, source_lang, target_lang)
            
            translated_segments.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'original': segment.get('original', text),
                'refined': segment.get('refined', text),
                'translated': translated_text
            })
            
            if (i + 1) % 5 == 0:
                logger.info(f"Translated {i + 1}/{len(segments)} segments")
        
        logger.info(f"Translation completed for all {len(segments)} segments")
        return translated_segments
    
    def save_translation(self, translated_segments: List[Dict[str, Any]],
                        json_path: Path, txt_path: Path) -> tuple:
        """Save translation to files."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(translated_segments, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved translation JSON: {json_path}")
        
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        full_text = " ".join([seg['translated'] for seg in translated_segments if seg['translated']])
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        logger.info(f"Saved translation text: {txt_path}")
        
        return json_path, txt_path
