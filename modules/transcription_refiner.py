"""
Transcription refiner using LLM to fix/improve Whisper output
The LLM corrects grammar, fills gaps, and makes sentences complete (GPU-only)
"""
import json
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
    Fixes grammar, completes sentences, fills gaps (GPU-only)
    """
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize refiner
        
        Args:
            model_name: LLM model to use
            device: Device to run on ('cuda')
        """
        self.model_name = model_name or settings.llm_model
        self.device = device or settings.llm_device
        
        # Lazy load for processing
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Refiner initialized with model: {self.model_name} on {self.device}")
    
    def _ensure_model_loaded(self):
        """Lazy load model for processing"""
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
                    max_new_tokens=settings.translation_max_length,
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
