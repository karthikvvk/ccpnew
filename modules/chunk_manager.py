"""
Chunk lifecycle management for idempotent video processing
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger("chunk_manager")


class ChunkManager:
    """
    Manages chunk directories and status tracking for robust video processing.
    Supports failure isolation, retry, and resume capabilities.
    """
    
    def __init__(self, job_dir: Path):
        """
        Initialize chunk manager for a job
        
        Args:
            job_dir: Path to job directory
        """
        self.job_dir = Path(job_dir)
        self.chunks_dir = self.job_dir / 'chunks'
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ChunkManager at: {self.chunks_dir}")
    
    def create_chunk(self, chunk_id: int) -> Path:
        """
        Create chunk directory structure
        
        Args:
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Path to chunk directory
        """
        chunk_dir = self.chunks_dir / f"chunk_{chunk_id:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (chunk_dir / 'audio_subchunks').mkdir(exist_ok=True)
        (chunk_dir / 'frames').mkdir(exist_ok=True)
        
        # Initialize status.json if not exists
        status_path = chunk_dir / 'status.json'
        if not status_path.exists():
            self._write_status(chunk_id, {
                'chunk_id': chunk_id,
                'created_at': datetime.now().isoformat(),
                'stage': 'initialized',
                'completed': False,
                'error': None,
                'metadata': {}
            })
        
        logger.info(f"Created chunk directory: {chunk_dir}")
        return chunk_dir
    
    def get_chunk_dir(self, chunk_id: int) -> Path:
        """Get path to chunk directory"""
        return self.chunks_dir / f"chunk_{chunk_id:03d}"
    
    def get_chunk_status(self, chunk_id: int) -> Dict[str, Any]:
        """
        Read status.json for a chunk
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Status dictionary
        """
        status_path = self.get_chunk_dir(chunk_id) / 'status.json'
        
        if not status_path.exists():
            return {
                'chunk_id': chunk_id,
                'stage': 'not_started',
                'completed': False,
                'error': None
            }
        
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read status for chunk {chunk_id}: {e}")
            return {'chunk_id': chunk_id, 'stage': 'unknown', 'error': str(e)}
    
    def update_chunk_status(self, 
                            chunk_id: int, 
                            stage: str, 
                            success: bool, 
                            metadata: Dict[str, Any] = None) -> None:
        """
        Update status.json with stage completion info
        
        Args:
            chunk_id: Chunk identifier
            stage: Current processing stage
            success: Whether the stage completed successfully
            metadata: Additional metadata to store
        """
        status = self.get_chunk_status(chunk_id)
        
        status['stage'] = stage
        status['completed'] = success and stage == 'completed'
        status['updated_at'] = datetime.now().isoformat()
        
        if metadata:
            if 'error' in metadata:
                status['error'] = metadata.pop('error')
            status['metadata'] = {**status.get('metadata', {}), **metadata}
        
        if not success:
            status['failed'] = True
            status['failed_at'] = stage
        
        self._write_status(chunk_id, status)
        logger.info(f"Updated chunk {chunk_id} status: stage={stage}, success={success}")
    
    def _write_status(self, chunk_id: int, status: Dict[str, Any]) -> None:
        """Write status to file"""
        status_path = self.get_chunk_dir(chunk_id) / 'status.json'
        status_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
    
    def get_failed_chunks(self) -> List[int]:
        """
        Return list of chunk IDs that failed
        
        Returns:
            List of failed chunk IDs
        """
        failed = []
        
        for chunk_dir in sorted(self.chunks_dir.glob('chunk_*')):
            try:
                chunk_id = int(chunk_dir.name.split('_')[1])
                status = self.get_chunk_status(chunk_id)
                if status.get('failed') or status.get('error'):
                    failed.append(chunk_id)
            except (ValueError, IndexError):
                continue
        
        return failed
    
    def get_pending_chunks(self) -> List[int]:
        """
        Return list of chunk IDs not yet processed
        
        Returns:
            List of pending chunk IDs
        """
        pending = []
        
        for chunk_dir in sorted(self.chunks_dir.glob('chunk_*')):
            try:
                chunk_id = int(chunk_dir.name.split('_')[1])
                status = self.get_chunk_status(chunk_id)
                if not status.get('completed') and not status.get('failed'):
                    pending.append(chunk_id)
            except (ValueError, IndexError):
                continue
        
        return pending
    
    def get_completed_chunks(self) -> List[int]:
        """
        Return list of completed chunk IDs
        
        Returns:
            List of completed chunk IDs
        """
        completed = []
        
        for chunk_dir in sorted(self.chunks_dir.glob('chunk_*')):
            try:
                chunk_id = int(chunk_dir.name.split('_')[1])
                status = self.get_chunk_status(chunk_id)
                if status.get('completed'):
                    completed.append(chunk_id)
            except (ValueError, IndexError):
                continue
        
        return completed
    
    def get_all_chunk_ids(self) -> List[int]:
        """Get all chunk IDs in order"""
        chunk_ids = []
        
        for chunk_dir in sorted(self.chunks_dir.glob('chunk_*')):
            try:
                chunk_id = int(chunk_dir.name.split('_')[1])
                chunk_ids.append(chunk_id)
            except (ValueError, IndexError):
                continue
        
        return sorted(chunk_ids)
    
    def get_chunk_audio_path(self, chunk_id: int) -> Optional[Path]:
        """Get path to chunk's TTS audio"""
        chunk_dir = self.get_chunk_dir(chunk_id)
        tts_path = chunk_dir / 'tts.wav'
        
        if tts_path.exists():
            return tts_path
        
        # Fallback to metadata
        status = self.get_chunk_status(chunk_id)
        tts_path_str = status.get('metadata', {}).get('tts_path')
        if tts_path_str and Path(tts_path_str).exists():
            return Path(tts_path_str)
        
        return None
    
    def cleanup_chunks(self, keep_failed: bool = True) -> None:
        """
        Remove chunk artifacts after successful merge
        
        Args:
            keep_failed: Whether to preserve failed chunk directories
        """
        import shutil
        
        for chunk_dir in self.chunks_dir.glob('chunk_*'):
            try:
                chunk_id = int(chunk_dir.name.split('_')[1])
                status = self.get_chunk_status(chunk_id)
                
                if keep_failed and (status.get('failed') or status.get('error')):
                    logger.info(f"Keeping failed chunk: {chunk_dir}")
                    continue
                
                shutil.rmtree(chunk_dir)
                logger.info(f"Cleaned up chunk: {chunk_dir}")
                
            except Exception as e:
                logger.warning(f"Failed to cleanup chunk directory {chunk_dir}: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of chunk processing status
        
        Returns:
            Summary dictionary
        """
        all_chunks = self.get_all_chunk_ids()
        completed = self.get_completed_chunks()
        failed = self.get_failed_chunks()
        pending = self.get_pending_chunks()
        
        return {
            'total_chunks': len(all_chunks),
            'completed': len(completed),
            'failed': len(failed),
            'pending': len(pending),
            'completion_rate': len(completed) / len(all_chunks) if all_chunks else 0,
            'failed_chunk_ids': failed,
            'pending_chunk_ids': pending
        }
