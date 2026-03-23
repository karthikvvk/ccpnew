"""
Transcription passthrough module
Since we use NLLB for translation only, this module passes through transcriptions
without modification. The refiner step is kept for pipeline compatibility.
"""
from typing import Dict, List, Any
from utils.logger import setup_logger

logger = setup_logger("refiner")


class TranscriptionRefiner:
    """
    Passthrough module - preserves original Whisper transcription
    NLLB handles translation separately, so no refinement is needed.
    """
    
    def __init__(self):
        """Initialize passthrough refiner"""
        logger.info("TranscriptionRefiner initialized (passthrough mode - no LLM refinement)")
    
    def refine_segments(self, segments: List[Dict[str, Any]], 
                       visual_context: str = None,
                       source_language: str = "auto") -> List[Dict[str, Any]]:
        """
        Pass through segments without modification
        
        Args:
            segments: List of segments with 'text', 'start', 'end'
            visual_context: Ignored (kept for compatibility)
            source_language: Ignored (kept for compatibility)
            
        Returns:
            List of segments with original text preserved
        """
        logger.info(f"Passing through {len(segments)} segments (no refinement)")
        
        refined_segments = []
        for segment in segments:
            refined_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'original': segment['text'],
                'refined': segment['text']  # Pass through original
            })
        
        return refined_segments
