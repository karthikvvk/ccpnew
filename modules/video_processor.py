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
    
    @staticmethod
    def adjust_audio_speed(audio_path: Path, 
                          target_duration: float,
                          output_path: Path = None) -> Path:
        """
        Adjust audio speed to match target duration (video length)
        
        Args:
            audio_path: Path to audio file
            target_duration: Target duration in seconds (video length)
            output_path: Path for output file (default: overwrites input)
            
        Returns:
            Path to adjusted audio file
        """
        try:
            # Get current audio duration
            probe = ffmpeg.probe(str(audio_path))
            current_duration = float(probe['format']['duration'])
            
            # Calculate speed factor
            speed_factor = current_duration / target_duration
            
            logger.info(f"Adjusting audio speed:")
            logger.info(f"  Current duration: {current_duration:.2f}s")
            logger.info(f"  Target duration: {target_duration:.2f}s")
            logger.info(f"  Speed factor: {speed_factor:.3f}x")
            
            # If audio is already close to target (within 1%), skip adjustment
            if 0.99 <= speed_factor <= 1.01:
                logger.info("Audio duration already matches video, skipping adjustment")
                return audio_path
            
            # Limit speed factor to reasonable range (0.5x to 2.0x)
            # For extreme cases, we need to chain multiple atempo filters
            if speed_factor < 0.5:
                logger.warning(f"Speed factor {speed_factor:.2f} is very low, audio will be significantly shortened")
                speed_factor = max(0.5, speed_factor)
            elif speed_factor > 2.0:
                logger.warning(f"Speed factor {speed_factor:.2f} is very high, audio will be significantly stretched")
                speed_factor = min(2.0, speed_factor)
            
            # Prepare output path
            if output_path is None:
                output_path = audio_path.parent / f"{audio_path.stem}_synced{audio_path.suffix}"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Apply speed adjustment using atempo filter
            # atempo range is 0.5 to 2.0, chain if needed
            filters = []
            remaining_factor = speed_factor
            
            while remaining_factor > 2.0:
                filters.append('atempo=2.0')
                remaining_factor /= 2.0
            while remaining_factor < 0.5:
                filters.append('atempo=0.5')
                remaining_factor /= 0.5
            
            filters.append(f'atempo={remaining_factor}')
            filter_chain = ','.join(filters)
            
            # Build ffmpeg command
            stream = ffmpeg.input(str(audio_path))
            stream = stream.filter('atempo', remaining_factor)
            stream = ffmpeg.output(stream, str(output_path), acodec='pcm_s16le')
            
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Verify output
            output_probe = ffmpeg.probe(str(output_path))
            output_duration = float(output_probe['format']['duration'])
            
            logger.info(f"Audio speed adjusted successfully")
            logger.info(f"  New duration: {output_duration:.2f}s")
            
            return output_path
            
        except ffmpeg.Error as e:
            logger.error(f"Failed to adjust audio speed: {e}")
            # Return original audio if adjustment fails
            return audio_path
        except Exception as e:
            logger.error(f"Error adjusting audio speed: {e}")
            return audio_path
    
    @staticmethod
    def get_audio_duration(audio_path: Path) -> float:
        """Get audio file duration in seconds"""
        try:
            probe = ffmpeg.probe(str(audio_path))
            return float(probe['format']['duration'])
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0
    
    @staticmethod
    def split_video(video_path: Path, 
                    duration: int = 300,
                    overlap: float = 0,
                    output_dir: Path = None) -> list:
        """
        Split video into chunks of specified duration
        
        Args:
            video_path: Path to input video
            duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            output_dir: Directory for output chunks (default: same as video)
            
        Returns:
            List of paths to video chunks
        """
        try:
            video_path = Path(video_path)
            output_dir = output_dir or video_path.parent / 'video_chunks'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get video duration
            probe = ffmpeg.probe(str(video_path))
            total_duration = float(probe['format']['duration'])
            
            logger.info(f"Splitting video ({total_duration:.1f}s) into {duration}s chunks")
            
            chunks = []
            chunk_id = 0
            start_time = 0
            
            while start_time < total_duration:
                chunk_path = output_dir / f"chunk_{chunk_id:03d}.mp4"
                
                # Calculate chunk duration (may be shorter for last chunk)
                chunk_duration = min(duration, total_duration - start_time)
                
                # Use ffmpeg to extract chunk
                stream = ffmpeg.input(str(video_path), ss=start_time, t=chunk_duration)
                stream = ffmpeg.output(stream, str(chunk_path), 
                                      vcodec='copy', 
                                      acodec='copy',
                                      avoid_negative_ts='make_zero')
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                chunks.append({
                    'path': chunk_path,
                    'start_time': start_time,
                    'duration': chunk_duration,
                    'chunk_id': chunk_id
                })
                
                logger.info(f"  Chunk {chunk_id}: {start_time:.1f}s - {start_time + chunk_duration:.1f}s")
                
                chunk_id += 1
                start_time += duration - overlap  # Apply overlap
                
            logger.info(f"Split video into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to split video: {e}")
            raise
    
    @staticmethod
    def split_audio(audio_path: Path,
                    duration: int = 30,
                    overlap: float = 1.5,
                    output_dir: Path = None) -> list:
        """
        Split audio into sub-chunks for STT processing
        
        Args:
            audio_path: Path to input audio
            duration: Duration of each sub-chunk in seconds
            overlap: Overlap between sub-chunks in seconds
            output_dir: Directory for output sub-chunks
            
        Returns:
            List of tuples: (path, start_time, end_time)
        """
        try:
            audio_path = Path(audio_path)
            output_dir = output_dir or audio_path.parent / 'audio_subchunks'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get audio duration
            probe = ffmpeg.probe(str(audio_path))
            total_duration = float(probe['format']['duration'])
            
            logger.info(f"Splitting audio ({total_duration:.1f}s) into {duration}s sub-chunks with {overlap}s overlap")
            
            subchunks = []
            subchunk_id = 0
            start_time = 0
            
            while start_time < total_duration:
                subchunk_path = output_dir / f"subchunk_{subchunk_id:03d}.wav"
                
                # Calculate sub-chunk duration
                end_time = min(start_time + duration, total_duration)
                subchunk_duration = end_time - start_time
                
                # Use ffmpeg to extract sub-chunk
                stream = ffmpeg.input(str(audio_path), ss=start_time, t=subchunk_duration)
                stream = ffmpeg.output(stream, str(subchunk_path),
                                      acodec='pcm_s16le',
                                      ac=1,
                                      ar='16000')
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                subchunks.append({
                    'path': subchunk_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'subchunk_id': subchunk_id
                })
                
                subchunk_id += 1
                start_time += duration - overlap  # Apply overlap
                
                # Prevent infinite loop if overlap >= duration
                if duration - overlap <= 0:
                    logger.warning("Overlap >= duration, breaking to avoid infinite loop")
                    break
                    
            logger.info(f"Split audio into {len(subchunks)} sub-chunks")
            return subchunks
            
        except Exception as e:
            logger.error(f"Failed to split audio: {e}")
            raise
    
    @staticmethod
    def merge_audios_timeline(audio_chunks: list,
                              output_path: Path,
                              crossfade_ms: int = 50,
                              validate_duration: bool = True) -> Path:
        """
        Merge audio chunks with timeline awareness and optional crossfade
        
        Args:
            audio_chunks: List of dicts with 'path', 'start_time', 'duration'
            output_path: Path for merged audio output
            crossfade_ms: Crossfade duration in milliseconds at joins
            validate_duration: Whether to validate duration continuity
            
        Returns:
            Path to merged audio file
        """
        try:
            if not audio_chunks:
                raise ValueError("No audio chunks to merge")
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sort chunks by start_time
            sorted_chunks = sorted(audio_chunks, key=lambda x: x.get('start_time', 0))
            
            logger.info(f"Merging {len(sorted_chunks)} audio chunks with {crossfade_ms}ms crossfade")
            
            # Validate duration continuity
            if validate_duration:
                for i in range(1, len(sorted_chunks)):
                    prev_end = sorted_chunks[i-1].get('start_time', 0) + sorted_chunks[i-1].get('duration', 0)
                    curr_start = sorted_chunks[i].get('start_time', 0)
                    gap = curr_start - prev_end
                    if abs(gap) > 0.5:  # More than 0.5s gap/overlap
                        logger.warning(f"Timeline gap/overlap between chunks: {gap:.2f}s")
            
            # Create concat file for ffmpeg
            concat_file = output_path.parent / 'concat_list.txt'
            with open(concat_file, 'w') as f:
                for chunk in sorted_chunks:
                    # Escape single quotes in path
                    escaped_path = str(chunk['path']).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
            
            # Use ffmpeg concat demuxer
            if crossfade_ms > 0 and len(sorted_chunks) > 1:
                # Complex filter for crossfade
                inputs = [ffmpeg.input(str(chunk['path'])) for chunk in sorted_chunks]
                
                # Build filter chain for crossfade
                crossfade_sec = crossfade_ms / 1000.0
                
                if len(inputs) == 2:
                    # Simple case: 2 inputs
                    merged = ffmpeg.filter([inputs[0], inputs[1]], 'acrossfade', d=crossfade_sec)
                else:
                    # Chain crossfades for multiple inputs
                    merged = inputs[0]
                    for i in range(1, len(inputs)):
                        merged = ffmpeg.filter([merged, inputs[i]], 'acrossfade', d=crossfade_sec)
                
                stream = ffmpeg.output(merged, str(output_path), acodec='pcm_s16le')
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
            else:
                # Simple concat without crossfade
                stream = ffmpeg.input(str(concat_file), format='concat', safe=0)
                stream = ffmpeg.output(stream, str(output_path), acodec='pcm_s16le')
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            # Cleanup concat file
            concat_file.unlink(missing_ok=True)
            
            # Verify output
            output_probe = ffmpeg.probe(str(output_path))
            output_duration = float(output_probe['format']['duration'])
            
            logger.info(f"Merged audio: {output_duration:.2f}s -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to merge audio chunks: {e}")
            raise

