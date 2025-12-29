"""
Translation module using local LLM
Can run locally (CPU) or in Colab (GPU) for faster processing
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from utils.logger import setup_logger
from config import settings

logger = setup_logger("translator")


class Translator:
    """
    Local LLM-based translation with RAG context enhancement
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize translator
        
        Args:
            model_name: LLM model to use
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name or settings.llm_model
        self.device = device or settings.llm_device
        
        # Don't load model yet - lazy load when needed
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Translator initialized with model: {self.model_name}")
    
    def _ensure_model_loaded(self):
        """Lazy load model and tokenizer when needed"""
        if self.model is not None and self.tokenizer is not None:
            return
        
        logger.info(f"Loading translation model: {self.model_name} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Choose model class based on model type
        if 't5' in self.model_name.lower() or 'flan' in self.model_name.lower():
            # T5 models are seq2seq
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        else:
            # Causal LM models (Mistral, Llama, etc.)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                load_in_8bit=True if self.device == "cuda" else False
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Translation model loaded successfully")
    
    def translate_text(self,
                      text: str,
                      target_language: str,
                      visual_context: str = None) -> str:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_language: Target language (e.g., 'Spanish', 'French')
            visual_context: Optional visual context from RAG
            
        Returns:
            Translated text
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        try:
            # Build prompt (simplified for T5)
            if 't5' in self.model_name.lower() or 'flan' in self.model_name.lower():
                # T5 models use simpler prompts
                prompt = f"translate to {target_language}: {text}"
            else:
                # Instruction-tuned models (Mistral, Llama)
                prompt = self._build_translation_prompt(text, target_language, visual_context)
            
            # Generate translation
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.llm_max_length,
                    temperature=settings.llm_temperature,
                    do_sample=True if settings.llm_temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
                )
            
            # Decode output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract translation
            if 't5' in self.model_name.lower() or 'flan' in self.model_name.lower():
                translation = full_output  # T5 outputs translation directly
            else:
                translation = self._extract_translation(full_output, prompt)
            
            return translation.strip()
            
        except Exception as e:
            logger.error(f"Failed to translate text: {e}")
            raise
    
    def translate_segments(self,
                          segments: List[Dict[str, Any]],
                          target_language: str,
                          visual_context: str = None) -> List[Dict[str, Any]]:
        """
        Translate segments with timing information
        
        Args:
            segments: List of segments with 'text', 'start', 'end'
            target_language: Target language
            visual_context: Optional visual context
            
        Returns:
            List of translated segments
        """
        # Check if Colab GPU is enabled
        if settings.use_colab_gpu and settings.colab_api_url:
            return self._translate_segments_colab(segments, target_language, visual_context)
        
        logger.info(f"Translating {len(segments)} segments to {target_language}")
        
        translated_segments = []
        
        for i, segment in enumerate(segments):
            translated_text = self.translate_text(
                segment['text'],
                target_language,
                visual_context
            )
            
            translated_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'original': segment['text'],
                'translated': translated_text
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Translated {i + 1}/{len(segments)} segments")
        
        logger.info(f"Translation completed for all {len(segments)} segments")
        return translated_segments
    
    def _build_translation_prompt(self,
                                 text: str,
                                 target_language: str,
                                 visual_context: str = None) -> str:
        """Build translation prompt for LLM"""
        
        context_info = ""
        if visual_context:
            context_info = f"\n\nVisual Context: {visual_context}"
        
        prompt = f"""[INST] You are a professional translator. Translate the following text to {target_language}.
Only provide the translation, nothing else.{context_info}

Text to translate: {text}

Translation: [/INST]"""
        
        return prompt
    
    def _extract_translation(self, full_output: str, prompt: str) -> str:
        """Extract translation from model output"""
        # Remove the prompt part
        if prompt in full_output:
            translation = full_output.replace(prompt, "").strip()
        else:
            translation = full_output.strip()
        
        return translation
    
    def _translate_segments_colab(self,
                                 segments: List[Dict[str, Any]],
                                 target_language: str,
                                 visual_context: str = None) -> List[Dict[str, Any]]:
        """
        Translate segments using Colab GPU API
        
        Args:
            segments: List of segments
            target_language: Target language
            visual_context: Optional visual context
            
        Returns:
            List of translated segments
        """
        import requests
        
        try:
            logger.info(f"Translating via Colab GPU: {len(segments)} segments to {target_language}")
            
            # First, ensure LLM model is loaded on Colab
            load_url = f"{settings.colab_api_url}/load_llm"
            load_response = requests.post(load_url, json={'model_name': self.model_name})
            
            if load_response.status_code == 200:
                logger.info("LLM model loaded on Colab GPU")
            
            # Send segments for translation
            url = f"{settings.colab_api_url}/llm/translate"
            
            payload = {
                'segments': segments,
                'target_language': target_language,
                'visual_context': visual_context
            }
            
            response = requests.post(url, json=payload, timeout=1800)  # 30 min timeout
            
            if response.status_code == 200:
                translated = response.json()['translated_segments']
                logger.info(f"Colab translation completed for {len(translated)} segments")
                return translated
            else:
                logger.error(f"Colab API error: {response.text}")
                raise Exception(f"Colab API failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to translate via Colab: {e}")
            logger.info("Falling back to local translation...")
            # Fallback to local if Colab fails
            settings.use_colab_gpu = False
            return self.translate_segments(segments, target_language, visual_context)
    
    def save_translation(self,
                        translated_segments: List[Dict[str, Any]],
                        json_path: Path,
                        txt_path: Path) -> tuple[Path, Path]:
        """
        Save translation to JSON and TXT files
        
        Args:
            translated_segments: List of translated segments
            json_path: Path to save JSON
            txt_path: Path to save plain text
            
        Returns:
            Tuple of (json_path, txt_path)
        """
        # Save JSON with full details
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(translated_segments, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved translation JSON: {json_path}")
        
        # Save plain text (concatenated translations)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        full_text = " ".join([seg['translated'] for seg in translated_segments])
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        logger.info(f"Saved translation text: {txt_path}")
        
        return json_path, txt_path


# Standalone function for Colab usage
def translate_text_gpu(input_json: str,
                      output_json: str,
                      output_txt: str,
                      target_language: str,
                      model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                      visual_context: str = None,
                      device: str = "cuda") -> list:
    """
    Translate text using GPU - Colab compatible
    
    Args:
        input_json: Path to transcription JSON
        output_json: Path to save translated JSON
        output_txt: Path to save translated text
        target_language: Target language
        model_name: LLM model to use
        visual_context: Optional visual context
        device: Device ('cuda' for GPU)
        
    Returns:
        List of translated segments
    """
    print(f"Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    # Load transcription
    with open(input_json, 'r', encoding='utf-8') as f:
        transcription = json.load(f)
    
    segments = transcription.get('segments', [])
    print(f"Translating {len(segments)} segments to {target_language}...")
    
    translated_segments = []
    
    for i, segment in enumerate(segments):
        # Build prompt
        context_info = f"\n\nVisual Context: {visual_context}" if visual_context else ""
        prompt = f"""[INST] You are a professional translator. Translate the following text to {target_language}.
Only provide the translation, nothing else.{context_info}

Text to translate: {segment['text']}

Translation: [/INST]"""
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = full_output.replace(prompt, "").strip()
        
        translated_segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'original': segment['text'],
            'translated': translation
        })
        
        if (i + 1) % 10 == 0:
            print(f"Translated {i + 1}/{len(segments)}")
    
    # Save JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(translated_segments, f, indent=2, ensure_ascii=False)
    
    # Save text
    full_text = " ".join([seg['translated'] for seg in translated_segments])
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Translation saved to {output_json} and {output_txt}")
    return translated_segments
