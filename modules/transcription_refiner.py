"""
Transcription refiner using LLM to fix/improve Whisper output
The LLM corrects grammar, fills gaps, and makes sentences complete

Supports both local CPU/GPU and Colab GPU via USE_COLAB_GPU setting
"""
import json
import requests
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from utils.logger import setup_logger
from config import settings

logger = setup_logger("refiner")


class TranscriptionRefiner:
    """
    Uses LLM to refine and improve Whisper transcriptions
    Fixes grammar, completes sentences, fills gaps
    
    Supports:
    - Local CPU/GPU processing
    - Colab GPU processing (when USE_COLAB_GPU=True)
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize refiner
        
        Args:
            model_name: LLM model to use
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name or settings.llm_model
        self.device = device or settings.llm_device
        self.use_colab = settings.use_colab_gpu and settings.colab_api_url
        
        # Lazy load for local processing
        self.model = None
        self.tokenizer = None
        
        if self.use_colab:
            logger.info(f"Refiner initialized with Colab GPU: {settings.colab_api_url}")
        else:
            logger.info(f"Refiner initialized with model: {self.model_name} on {self.device}")
    
    def _ensure_model_loaded(self):
        """Lazy load model for local processing"""
        if self.use_colab:
            return  # Skip local loading when using Colab
            
        if self.model is not None and self.tokenizer is not None:
            return
        
        logger.info(f"Loading refinement model: {self.model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Choose model class
        if 't5' in self.model_name.lower() or 'flan' in self.model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                load_in_8bit=True if self.device == "cuda" else False
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Refinement model loaded successfully")
    
    def refine_text(self, text: str, visual_context: str = None) -> str:
        """
        Refine broken/incomplete text
        
        Args:
            text: Broken transcription from Whisper
            visual_context: Optional visual context from RAG
            
        Returns:
            Refined, complete sentences
        """
        self._ensure_model_loaded()
        
        try:
            # Build refinement prompt
            if 't5' in self.model_name.lower() or 'flan' in self.model_name.lower():
                if visual_context:
                    prompt = f"Fix grammar and complete sentences. Context: {visual_context}. Text: {text}"
                else:
                    prompt = f"Fix grammar and complete sentences: {text}"
            else:
                context_info = f"\n\nVisual context: {visual_context}" if visual_context else ""
                prompt = f"""[INST] Fix the following transcription. Correct grammar errors, complete incomplete sentences, and make it clear and readable.{context_info}

Broken transcription: {text}

Fixed transcription: [/INST]"""
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.llm_max_length,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0
                )
            
            # Decode
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract refined text
            if 't5' in self.model_name.lower() or 'flan' in self.model_name.lower():
                refined = full_output
            else:
                refined = full_output.replace(prompt, "").strip()
            
            logger.info(f"Refined text: {refined[:100]}...")
            return refined.strip()
            
        except Exception as e:
            logger.error(f"Failed to refine text: {e}")
            return text
    
    def refine_segments(self, segments: List[Dict[str, Any]], 
                       visual_context: str = None,
                       source_language: str = "en") -> List[Dict[str, Any]]:
        """
        Refine multiple segments
        
        Args:
            segments: List of segments with 'text', 'start', 'end'
            visual_context: Optional visual context
            source_language: Source language code (refinement only for English)
            
        Returns:
            List of refined segments
        """
        # Flan-T5 only works well with English
        # For non-English, skip refinement and pass through original text
        if source_language and source_language.lower() not in ['en', 'english', 'auto']:
            logger.info(f"Skipping LLM refinement for non-English ({source_language}) - using original Whisper output")
            refined_segments = []
            for segment in segments:
                refined_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'original': segment['text'],
                    'refined': segment['text']  # Pass through original
                })
            return refined_segments
        
        # Use Colab GPU if enabled
        if self.use_colab:
            return self._refine_segments_colab(segments, visual_context)
        
        # Local processing for English
        logger.info(f"Refining {len(segments)} segments locally on {self.device}")
        
        refined_segments = []
        
        for i, segment in enumerate(segments):
            refined_text = self.refine_text(segment['text'], visual_context)
            
            # If refinement returns empty, use original
            if not refined_text.strip():
                refined_text = segment['text']
            
            refined_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'original': segment['text'],
                'refined': refined_text
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Refined {i + 1}/{len(segments)} segments")
        
        logger.info(f"Refinement completed for all {len(segments)} segments")
        return refined_segments
    
    def _refine_segments_colab(self, segments: List[Dict[str, Any]], 
                               visual_context: str = None) -> List[Dict[str, Any]]:
        """
        Refine segments using Colab GPU
        
        Args:
            segments: List of segments
            visual_context: Optional visual context
            
        Returns:
            List of refined segments
        """
        try:
            logger.info(f"Refining {len(segments)} segments via Colab GPU")
            
            # First, ensure refiner model is loaded on Colab
            load_url = f"{settings.colab_api_url}/load_refiner"
            load_response = requests.post(load_url, json={'model_name': self.model_name}, timeout=300)
            
            if load_response.status_code == 200:
                logger.info("Refiner model loaded on Colab GPU")
            else:
                logger.warning(f"Failed to load refiner on Colab: {load_response.text}")
            
            # Send segments for refinement
            url = f"{settings.colab_api_url}/llm/refine"
            
            payload = {
                'segments': segments,
                'visual_context': visual_context
            }
            
            response = requests.post(url, json=payload, timeout=1800)  # 30 min timeout
            
            if response.status_code == 200:
                refined = response.json().get('refined_segments', [])
                logger.info(f"Colab refinement completed for {len(refined)} segments")
                return refined
            else:
                logger.error(f"Colab API error: {response.text}")
                raise Exception(f"Colab API failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to refine via Colab: {e}")
            logger.info("Falling back to local refinement...")
            
            # Disable Colab for this instance and retry locally
            self.use_colab = False
            self._ensure_model_loaded()  # Make sure local model is loaded
            return self.refine_segments(segments, visual_context)
