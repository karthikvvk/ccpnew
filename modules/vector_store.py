"""
Vector store module using ChromaDB for frame storage and retrieval
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from utils.logger import setup_logger
from config import settings

logger = setup_logger("vector_store")


class VectorStore:
    """ChromaDB vector store for frame embeddings"""
    
    def __init__(self, db_path: Path = None, collection_name: str = "frames"):
        """
        Initialize vector store
        
        Args:
            db_path: Path to database directory (default from settings)
            collection_name: Name of the collection
        """
        self.db_path = db_path or settings.vector_db_path
        self.collection_name = collection_name
        
        logger.info(f"Initializing ChromaDB at: {self.db_path}")
        
        # Create ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Video frame embeddings"}
        )
        
        logger.info(f"Vector store initialized with collection: {collection_name}")
    
    def add_frames(self, frame_embeddings: List[Tuple[Path, np.ndarray]]) -> None:
        """
        Add frame embeddings to the vector store
        
        Args:
            frame_embeddings: List of (frame_path, embedding) tuples
        """
        try:
            logger.info(f"Adding {len(frame_embeddings)} frames to vector store")
            
            ids = []
            embeddings = []
            metadatas = []
            
            for frame_path, embedding in frame_embeddings:
                # Extract frame number from filename (e.g., frame_0001.jpg -> 1)
                frame_num = int(frame_path.stem.split('_')[-1])
                
                ids.append(f"frame_{frame_num:04d}")
                embeddings.append(embedding.tolist())
                metadatas.append({
                    "frame_path": str(frame_path),
                    "frame_number": frame_num
                })
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(frame_embeddings)} frames")
            
        except Exception as e:
            logger.error(f"Failed to add frames: {e}")
            raise
    
    def query_similar_frames(self, 
                            query_embedding: np.ndarray, 
                            n_results: int = 5) -> Dict[str, Any]:
        """
        Query for similar frames
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            Dictionary containing results
        """
        try:
            logger.info(f"Querying for {n_results} similar frames")
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            # Format results
            formatted_results = {
                'frames': [],
                'distances': results['distances'][0] if results['distances'] else []
            }
            
            if results['metadatas'] and results['metadatas'][0]:
                for metadata in results['metadatas'][0]:
                    formatted_results['frames'].append({
                        'frame_path': metadata['frame_path'],
                        'frame_number': metadata['frame_number']
                    })
            
            logger.info(f"Found {len(formatted_results['frames'])} similar frames")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to query frames: {e}")
            raise
    
    def get_all_frames(self) -> List[Dict[str, Any]]:
        """
        Get all frames from the vector store
        
        Returns:
            List of frame metadata
        """
        try:
            # Get all items
            results = self.collection.get()
            
            frames = []
            if results['metadatas']:
                for metadata in results['metadatas']:
                    frames.append({
                        'frame_path': metadata['frame_path'],
                        'frame_number': metadata['frame_number']
                    })
            
            logger.info(f"Retrieved {len(frames)} frames from vector store")
            return frames
            
        except Exception as e:
            logger.error(f"Failed to get all frames: {e}")
            raise
    
    def count(self) -> int:
        """
        Get count of frames in vector store
        
        Returns:
            Number of frames
        """
        return self.collection.count()
    
    def delete_collection(self) -> None:
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
