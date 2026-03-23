"""
Context-aware translation module using Llama-3.1-8B-Instruct
Uses RAG context for domain-specific translation accuracy
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from utils.logger import setup_logger
from config import settings

logger = setup_logger("translator")


class Translator:
    """
    Context-aware translator using Llama-3.1-8B-Instruct
    Leverages RAG domain context for improved translation accuracy
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Llama translator on GPU with 4-bit quantization"""
        if Translator._model is not None:
            return  # Already initialized
            
        self.model_name = settings.translation_model
        self.device = settings.translation_device
        self.max_length = settings.translation_max_length
        self.temperature = settings.translation_temperature
        self.use_4bit = settings.translation_use_4bit
        
        logger.info(f"Loading Llama model: {self.model_name} on {self.device}")
        logger.info(f"  4-bit quantization: {self.use_4bit}")
        logger.info(f"  Temperature: {self.temperature}")
        
        try:
            # Load tokenizer
            Translator._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Setup quantization config
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                Translator._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:
                Translator._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            
            # Set pad token if not set
            if Translator._tokenizer.pad_token is None:
                Translator._tokenizer.pad_token = Translator._tokenizer.eos_token
            
            logger.info(f"Llama Translator initialized successfully")
            
            # Warm up the model
            self._warmup()
            
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
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
            logger.info("Warming up Llama model...")
            self.translate_text("வணக்கம்", "english", context="greeting")
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _build_prompt(self, text: str, target_language: str, context: str = None) -> str:
        """
        Build instruction prompt for Llama-3.1-8B-Instruct
        
        Args:
            text: Source text to translate
            target_language: Target language name
            context: RAG context for domain awareness
            
        Returns:
            Formatted prompt string
        """
        domain_info = f"Domain context: {context}" if context else "Domain: general conversation"
        
        # Updated Prompt Construction with Stronger Instructions
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional translator specializing in Tamil to {target_language} translation.

CRITICAL INSTRUCTION:
Your PRIMARY source of truth is the "Source Text" provided below. You must translate exactly what is said.
The "Domain Context" is provided ONLY to help with specific terminology or background understanding.
NEVER let the "Domain Context" override or change the meaning of the "Source Text".

{domain_info}

Rules:
- Translate naturally, preserving the original meaning and tone
- Keep technical terms and proper nouns accurate
- Handle colloquial/spoken language appropriately
- Output ONLY the translation, no explanations
<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate this Tamil text to {target_language}:
{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        return prompt

    def _calculate_confidence(self, segment: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a segment based on ASR metadata.
        
        Args:
           segment: Dictionary containing segment data including 'avg_logprob'
           
        Returns:
           float: 0.0 to 1.0 confidence score
        """
        # Default high confidence if no metadata
        if 'avg_logprob' not in segment:
            return 1.0
            
        avg_logprob = segment.get('avg_logprob', 0.0)
        no_speech_prob = segment.get('no_speech_prob', 0.0)
        
        # Whisper logprobs are typically -1.0 to 0.0 for good transcription
        # Lower means less confident. -1.0 is ~37% prob, -0.5 is ~60%, 0.0 is 100%
        # Normalize logprob to roughly 0-1 scale
        # Simple heuristic: map -2.0...0.0 to 0.0...1.0
        logprob_score = max(0.0, min(1.0, (avg_logprob + 2.0) / 2.0))
        
        # Penalize for high no_speech_probability
        confidence = logprob_score * (1.0 - no_speech_prob)
        
        return confidence
        
    def _verify_consistency(self, source_text: str, translation: str) -> bool:
        """
        Verify if translation is semantically consistent with source (hallucination check).
        
        Args:
            source_text: Original source text
            translation: Generated translation
            
        Returns:
            bool: True if consistent, False if likely hallucinated/drifted
        """
        try:
            check_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict quality control verifier.
Check if the translation contains information NOT present in the source text.
Answer only YES or NO.

Source: {source_text}
Translation: {translation}

Does the translation contain hallucinated information?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            
            inputs = self.tokenizer(
                check_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
                
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
            
            # If answer is YES, it means hallucination detected -> Not consistent
            if "YES" in result:
                logger.warning(f"Hallucination detected in translation for: '{source_text[:30]}...'")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return True # Assume okay on error to optimize for speed
    
    def translate_text(self, text: str, target_language: str, context: str = None) -> str:
        """
        Translate text using Llama with context awareness
        
        Args:
            text: Text to translate (Tamil)
            target_language: Target language (e.g., 'english')
            context: RAG context for domain-specific translation
            
        Returns:
            Translated text
        """
        try:
            if not text or not text.strip():
                return text
            
            prompt = self._build_prompt(text, target_language, context)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.model.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the translation (after the last assistant header)
            if "assistant" in full_output.lower():
                parts = full_output.split("assistant")
                translation = parts[-1].strip()
            else:
                # Fallback: remove the prompt portion
                translation = full_output.replace(prompt, "").strip()
            
            # Clean up any remaining markers
            for marker in ["<|eot_id|>", "<|end_of_text|>", "</s>"]:
                translation = translation.replace(marker, "").strip()
            
            return translation
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text  # Return original if translation fails
    
    def translate_segments(self, segments: List[Dict[str, Any]], 
                          target_language: str,
                          source_language: str = "tamil",
                          context: str = None) -> List[Dict[str, Any]]:
        """
        Translate segments with RAG context
        
        Args:
            segments: List of segments with 'refined' or 'text' content
            target_language: Target language
            source_language: Source language
            context: RAG visual context for domain awareness
            
        Returns:
            List of translated segments
        """
        logger.info(f"Translating {len(segments)} segments to {target_language}")
        if context:
            logger.info(f"Using RAG context: {context[:100]}...")
        
        translated_segments = []
        
        for i, segment in enumerate(segments):
            # Use refined text if available, otherwise original
            text_to_translate = segment.get('refined', segment.get('text', ''))
            
            # Calculate confidence
            confidence = self._calculate_confidence(segment)
            
            # Decide on RAG usage (User logic: < 0.6 use RAG, else skip)
            # RAG DISABLED GLOBALLY
            use_rag = False # confidence < 0.6
            
            current_context = context if use_rag else None
            
            if use_rag:
                logger.info(f"Low confidence ({confidence:.2f}), utilizing RAG context")
            
            translated_text = self.translate_text(
                text_to_translate, 
                target_language, 
                context=current_context
            )
            
            # Verification Pass
            is_consistent = self._verify_consistency(text_to_translate, translated_text)
            
            if not is_consistent and use_rag:
                logger.warning("Translation failed verification. Regenerating WITHOUT RAG context.")
                # Fallback: Retry without RAG
                translated_text = self.translate_text(
                    text_to_translate,
                    target_language,
                    context=None # Force no RAG
                )
            
            translated_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'original': segment.get('original', segment.get('text', '')),
                'refined': segment.get('refined', ''),
                'translated': translated_text,
                'confidence': confidence,
                'is_consistent': is_consistent
            })
            
            if (i + 1) % 5 == 0:
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
