"""
Frame embedding module for generating visual embeddings
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from utils.logger import setup_logger
from config import settings

logger = setup_logger("frame_embedder")


class FrameEmbedder:
    """Generates embeddings for video frames using CLIP model"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize frame embedder
        
        Args:
            model_name: Name of the embedding model (default from settings)
        """
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")
    
    def embed_frame(self, frame_path: Path) -> np.ndarray:
        """
        Generate embedding for a single frame
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Load image
            image = Image.open(frame_path).convert('RGB')
            
            # Generate embedding
            embedding = self.model.encode(image)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed frame {frame_path}: {e}")
            raise
    
    def embed_frames(self, frame_dir: Path) -> List[Tuple[Path, np.ndarray]]:
        """
        Generate embeddings for all frames in a directory
        
        Args:
            frame_dir: Directory containing frame images
            
        Returns:
            List of tuples (frame_path, embedding)
        """
        try:
            # Get all frame files sorted by name
            frame_files = sorted(frame_dir.glob('*.jpg'))
            
            if not frame_files:
                raise ValueError(f"No frames found in {frame_dir}")
            
            logger.info(f"Generating embeddings for {len(frame_files)} frames")
            
            # Load all images
            images = [Image.open(f).convert('RGB') for f in frame_files]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(images, show_progress_bar=True)
            
            # Pair frames with embeddings
            results = list(zip(frame_files, embeddings))
            
            logger.info(f"Generated {len(results)} embeddings successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to embed frames: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get dimension of embedding vectors
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
