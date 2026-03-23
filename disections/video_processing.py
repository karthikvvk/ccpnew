"""
Standalone script for Video Processing using FFmpeg.
This script demonstrates common video operations: extracting audio, extracting frames, and merging/replacing audio.

Dependencies:
    pip install ffmpeg-python
    (Requires ffmpeg installed on the system)
"""
import ffmpeg
import argparse
import sys
import os
from pathlib import Path

def extract_audio(video_path, output_path):
    """
    Extract audio from video.
    """
    print(f"Extracting audio from '{video_path}' to '{output_path}'...")
    try:
        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.output(stream.audio, str(output_path), 
                              acodec='pcm_s16le', 
                              ac=1, 
                              ar='16000')
        ffmpeg.run(stream, overwrite_output=True, quiet=False)
        print("Audio extraction complete.")
    except Exception as e:
        print(f"Error extracting audio: {e}")

def extract_frames(video_path, output_dir, fps=1):
    """
    Extract frames from video.
    """
    print(f"Extracting frames from '{video_path}' to '{output_dir}' at {fps} fps...")
    os.makedirs(output_dir, exist_ok=True)
    try:
        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.filter(stream, 'fps', fps=fps)
        stream = ffmpeg.output(stream, str(Path(output_dir) / 'frame_%04d.jpg'),
                              **{'qscale:v': 2})
        ffmpeg.run(stream, overwrite_output=True, quiet=False)
        print("Frame extraction complete.")
    except Exception as e:
        print(f"Error extracting frames: {e}")

def replace_audio(video_path, new_audio_path, output_path):
    """
    Replace the audio track of a video with a new audio file.
    """
    print(f"Replacing audio in '{video_path}' with '{new_audio_path}'...")
    try:
        video_stream = ffmpeg.input(str(video_path)).video
        audio_stream = ffmpeg.input(str(new_audio_path)).audio
        
        stream = ffmpeg.output(video_stream, audio_stream, str(output_path),
                              vcodec='copy',
                              acodec='aac',
                              shortest=None)
        ffmpeg.run(stream, overwrite_output=True, quiet=False)
        print(f"Video saved to: {output_path}")
    except Exception as e:
        print(f"Error replacing audio: {e}")

def main():
    parser = argparse.ArgumentParser(description="Standalone Video Processing Demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Extract Audio
    p_audio = subparsers.add_parser("extract_audio", help="Extract audio from video")
    p_audio.add_argument("video", help="Input video file")
    p_audio.add_argument("output", help="Output audio file")
    
    # Extract Frames
    p_frames = subparsers.add_parser("extract_frames", help="Extract frames from video")
    p_frames.add_argument("video", help="Input video file")
    p_frames.add_argument("output_dir", help="Output directory for frames")
    p_frames.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    
    # Replace Audio
    p_replace = subparsers.add_parser("replace_audio", help="Replace video audio")
    p_replace.add_argument("video", help="Input video file")
    p_replace.add_argument("audio", help="New input audio file")
    p_replace.add_argument("output", help="Output video file")
    
    args = parser.parse_args()
    
    if args.command == "extract_audio":
        extract_audio(args.video, args.output)
    elif args.command == "extract_frames":
        extract_frames(args.video, args.output_dir, args.fps)
    elif args.command == "replace_audio":
        replace_audio(args.video, args.audio, args.output)

if __name__ == "__main__":
    main()
