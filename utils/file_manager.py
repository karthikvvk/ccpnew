"""
File management utilities for job-specific directories and file tracking
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import uuid
from config import settings
from utils.logger import setup_logger

logger = setup_logger("file_manager")


class FileManager:
    """Manages file operations and tracking for video translation jobs"""
    
    def __init__(self, job_id: str = None):
        """
        Initialize file manager for a job
        
        Args:
            job_id: Unique job identifier (generated if not provided)
        """
        self.job_id = job_id or str(uuid.uuid4())
        self.job_dir = settings.output_dir / self.job_id
        self.tracked_files: Dict[str, Path] = {}
        
        # Create job directory
        self.job_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created job directory: {self.job_dir}")
    
    def get_path(self, file_type: str, filename: str = None) -> Path:
        """
        Get path for a specific file type
        
        Args:
            file_type: Type of file (e.g., 'frames', 'audio', 'transcription')
            filename: Optional specific filename
            
        Returns:
            Path object for the file
        """
        path_mapping = {
            'frames': self.job_dir / 'frames',
            'vector_db': self.job_dir / 'vector_db',
            'original_audio': self.job_dir / 'original_audio.wav',
            'transcription_json': self.job_dir / 'transcription.json',
            'transcription_txt': self.job_dir / 'transcription.txt',
            'rag_context': self.job_dir / 'rag_context.json',
            'translation_json': self.job_dir / 'translation.json',
            'translation_txt': self.job_dir / 'translation.txt',
            'translated_audio': self.job_dir / 'translated_audio.mp3',
            'final_video': self.job_dir / 'final_video.mp4',
        }
        
        if filename:
            return self.job_dir / filename
        
        path = path_mapping.get(file_type, self.job_dir / file_type)
        
        # Create parent directory if needed
        if path.suffix:  # It's a file
            path.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory
            path.mkdir(parents=True, exist_ok=True)
        
        return path
    
    def track_file(self, file_type: str, file_path: Path) -> None:
        """
        Track a file for this job
        
        Args:
            file_type: Type/category of file
            file_path: Path to the file
        """
        self.tracked_files[file_type] = file_path
        logger.info(f"Tracked file [{file_type}]: {file_path}")
    
    def save_manifest(self) -> Path:
        """
        Save a manifest of all tracked files
        
        Returns:
            Path to the manifest file
        """
        manifest = {
            'job_id': self.job_id,
            'timestamp': datetime.now().isoformat(),
            'files': {
                file_type: str(path.resolve())
                for file_type, path in self.tracked_files.items()
            }
        }
        
        manifest_path = self.job_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved manifest: {manifest_path}")
        return manifest_path
    
    def cleanup(self, keep_manifest: bool = True) -> None:
        """
        Clean up job directory
        
        Args:
            keep_manifest: Whether to keep manifest.json
        """
        if keep_manifest:
            manifest_path = self.job_dir / 'manifest.json'
            if manifest_path.exists():
                # Move manifest to parent dir temporarily
                temp_manifest = settings.output_dir / f"{self.job_id}_manifest.json"
                shutil.move(str(manifest_path), str(temp_manifest))
        
        # Remove job directory
        shutil.rmtree(self.job_dir, ignore_errors=True)
        logger.info(f"Cleaned up job directory: {self.job_dir}")
        
        if keep_manifest:
            # Recreate job dir and move manifest back
            self.job_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_manifest), str(manifest_path))
    
    def get_all_files(self) -> List[Path]:
        """
        Get list of all tracked files
        
        Returns:
            List of file paths
        """
        return list(self.tracked_files.values())
