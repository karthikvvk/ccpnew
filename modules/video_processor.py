"""
Video processing module for audio/frame extraction and video reconstruction
"""
import ffmpeg
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import cv2
from utils.logger import setup_logger
from config import settings

logger = setup_logger("video_processor")


class VideoProcessor:
    """Handles video input/output operations using ffmpeg"""
    
    def __init__(self, video_path: Path):
        """
        Initialize video processor
        
        Args:
            video_path: Path to input video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Initialized VideoProcessor for: {video_path}")
    
    def extract_audio(self, output_path: Path) -> Path:
        """
        Extract audio from video
        
        Args:
            output_path: Path where audio should be saved
            
        Returns:
            Path to extracted audio file
        """
        try:
            logger.info(f"Extracting audio to: {output_path}")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract audio using ffmpeg
            stream = ffmpeg.input(str(self.video_path))
            stream = ffmpeg.output(stream.audio, str(output_path), 
                                  acodec='pcm_s16le', 
                                  ac=1, 
                                  ar='16000')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"Audio extracted successfully: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"Failed to extract audio: {e}")
            raise
    
    def extract_frames(self, output_dir: Path, fps: int = None) -> Path:
        """
        Extract frames from video at specified FPS
        
        Args:
            output_dir: Directory where frames should be saved
            fps: Frames per second to extract (default from settings)
            
        Returns:
            Path to directory containing frames
        """
        try:
            fps = fps or settings.frame_extract_fps
            logger.info(f"Extracting frames at {fps} fps to: {output_dir}")
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames using ffmpeg
            stream = ffmpeg.input(str(self.video_path))
            stream = ffmpeg.filter(stream, 'fps', fps=fps)
            stream = ffmpeg.output(stream, str(output_dir / 'frame_%04d.jpg'),
                                  **{'qscale:v': 2})
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Count extracted frames
            frame_count = len(list(output_dir.glob('*.jpg')))
            logger.info(f"Extracted {frame_count} frames successfully")
            
            return output_dir
            
        except ffmpeg.Error as e:
            logger.error(f"Failed to extract frames: {e}")
            raise
    
    def get_video_info(self) -> dict:
        """
        Get video metadata
        
        Returns:
            Dictionary containing video information
        """
        try:
            probe = ffmpeg.probe(str(self.video_path))
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            info = {
                'duration': float(probe['format']['duration']),
                'width': int(video_info['width']),
                'height': int(video_info['height']),
                'fps': eval(video_info['r_frame_rate']),
                'has_audio': audio_info is not None
            }
            
            logger.info(f"Video info: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    @staticmethod
    def reconstruct_video(original_video: Path, 
                         new_audio: Path, 
                         output_path: Path) -> Path:
        """
        Reconstruct video with new audio track
        
        Args:
            original_video: Path to original video file
            new_audio: Path to new audio file
            output_path: Path where output video should be saved
            
        Returns:
            Path to output video
        """
        try:
            logger.info(f"Reconstructing video with new audio")
            logger.info(f"  Video: {original_video}")
            logger.info(f"  Audio: {new_audio}")
            logger.info(f"  Output: {output_path}")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Combine video and new audio
            video_stream = ffmpeg.input(str(original_video)).video
            audio_stream = ffmpeg.input(str(new_audio)).audio
            
            stream = ffmpeg.output(video_stream, audio_stream, str(output_path),
                                  vcodec='copy',
                                  acodec=settings.audio_codec,
                                  shortest=None)
            
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"Video reconstructed successfully: {output_path}")
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"Failed to reconstruct video: {e}")
            raise
