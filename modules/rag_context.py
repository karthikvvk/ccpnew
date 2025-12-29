"""
RAG context module for generating visual context from frames
This module can be run locally or adapted for Colab with GPU
"""
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from utils.logger import setup_logger
from config import settings

logger = setup_logger("rag_context")


class RAGContext:
    """
    Generate textual descriptions from frames using vision-language model
    Can run locally (CPU) or in Colab (GPU)
    """
    
    def __init__(self, model_name: str = "microsoft/git-base", device: str = None):
        """
        Initialize RAG context generator
        
        Args:
            model_name: Vision-language model to use
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device or settings.llm_device
        self.model_name = model_name
        
        logger.info(f"Loading vision model: {model_name} on {self.device}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Vision model loaded successfully")
    
    def describe_frame(self, frame_path: Path, max_length: int = 50) -> str:
        """
        Generate description for a single frame
        
        Args:
            frame_path: Path to frame image
            max_length: Maximum length of description
            
        Returns:
            Textual description of the frame
        """
        try:
            # Load image
            image = Image.open(frame_path).convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Generate description
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=max_length
                )
            
            description = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Failed to describe frame {frame_path}: {e}")
            return ""
    
    def describe_frames(self, frame_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Generate descriptions for multiple frames
        
        Args:
            frame_paths: List of frame paths
            
        Returns:
            List of dictionaries with frame info and descriptions
        """
        logger.info(f"Generating descriptions for {len(frame_paths)} frames")
        
        results = []
        for i, frame_path in enumerate(frame_paths):
            description = self.describe_frame(frame_path)
            
            # Extract frame number
            frame_num = int(frame_path.stem.split('_')[-1])
            
            results.append({
                'frame_number': frame_num,
                'frame_path': str(frame_path),
                'description': description
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(frame_paths)} frames")
        
        logger.info(f"Generated {len(results)} frame descriptions")
        return results
    
    def generate_context(self, 
                        similar_frames: List[Dict[str, Any]]) -> str:
        """
        Generate contextual description from similar frames
        
        Args:
            similar_frames: List of similar frame dictionaries
            
        Returns:
            Combined contextual description
        """
        context_parts = []
        
        for frame_info in similar_frames:
            frame_path = Path(frame_info['frame_path'])
            description = self.describe_frame(frame_path)
            
            context_parts.append(
                f"Frame {frame_info['frame_number']}: {description}"
            )
        
        context = " | ".join(context_parts)
        logger.info(f"Generated context from {len(similar_frames)} frames")
        
        return context
    
    def save_context(self, context_data: List[Dict[str, Any]], output_path: Path) -> Path:
        """
        Save context data to JSON file
        
        Args:
            context_data: Context data to save
            output_path: Path to output file
            
        Returns:
            Path to saved file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(context_data, f, indent=2)
        
        logger.info(f"Saved context to: {output_path}")
        return output_path


# Standalone function for Colab usage
def describe_frames_batch(frame_dir: str, 
                         output_json: str,
                         model_name: str = "microsoft/git-base",
                         device: str = "cuda"):
    """
    Batch process frames for descriptions - Colab compatible
    
    Args:
        frame_dir: Directory containing frames
        output_json: Path to save descriptions
        model_name: Vision model to use
        device: Device ('cuda' for GPU)
    """
    rag = RAGContext(model_name=model_name, device=device)
    
    # Get all frames
    frame_paths = sorted(Path(frame_dir).glob('*.jpg'))
    
    # Generate descriptions
    descriptions = rag.describe_frames(frame_paths)
    
    # Save results
    rag.save_context(descriptions, Path(output_json))
    
    print(f"Processed {len(descriptions)} frames")
    return descriptions
