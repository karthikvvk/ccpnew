"""
Audio Source Separation: Voice vs Music vs Noise
================================================

This script uses Demucs to separate audio into:
- Vocals (speech/singing)
- Music (instruments, background)
- Bass, Drums (optional 4-stem mode)

Usage:
    python audio_separator.py input.mp3 --model htdemucs --vocals-only
    python audio_separator.py input.wav --model mdx_extra --four-stems
"""

import os
import sys
import argparse
import subprocess
import torch
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import demucs
        import torchaudio
        import soundfile
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Install required packages:")
        print("   pip install demucs torchaudio soundfile librosa")
        return False


def separate_audio(input_file, model='htdemucs', vocals_only=True, output_dir='separated', use_gpu=True):
    """
    Separate audio into stems using Demucs
    
    Args:
        input_file: Path to input audio file
        model: Model to use (htdemucs, mdx_extra, etc.)
        vocals_only: If True, only separate vocals/no_vocals
        output_dir: Directory for output files
        use_gpu: Use GPU if available
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return False
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Build command
    cmd = ['demucs', '-n', model, '-o', output_dir]
    
    if vocals_only:
        cmd.extend(['--two-stems', 'vocals'])
    
    # Force CPU if requested or GPU not available
    if not use_gpu or not torch.cuda.is_available():
        cmd.extend(['--device', 'cpu'])
        device_name = 'CPU'
    else:
        device_name = f'GPU ({torch.cuda.get_device_name(0)})'
    
    cmd.append(input_file)
    
    # Print info
    print(f"\n🎵 Audio Source Separation")
    print(f"   Input: {input_file}")
    print(f"   Model: {model}")
    print(f"   Mode: {'2-stem (vocals only)' if vocals_only else '4-stem (full)'}")
    print(f"   Device: {device_name}")
    print(f"   Output: {output_dir}/")
    print("\n⏳ Processing... (this may take a few minutes)\n")
    
    # Run separation
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Separation complete!\n")
            
            # Find output files
            base_name = Path(input_file).stem
            model_output = Path(output_dir) / model / base_name
            
            print("📁 Output files:")
            if model_output.exists():
                for wav_file in sorted(model_output.glob('*.wav')):
                    file_size = wav_file.stat().st_size / (1024*1024)  # MB
                    print(f"   ✓ {wav_file.name} ({file_size:.1f} MB)")
                    
            return True
        else:
            print("❌ Error during separation:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def enhance_vocals(vocals_file, output_file=None, noise_gate_db=-40):
    """
    Apply post-processing to enhance vocals
    
    Args:
        vocals_file: Path to vocals.wav file
        output_file: Output path (default: vocals_enhanced.wav)
        noise_gate_db: Threshold for noise gate in dB
    """
    
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        
        print(f"\n🔧 Enhancing vocals...")
        print(f"   Input: {vocals_file}")
        print(f"   Noise gate: {noise_gate_db} dB")
        
        # Load audio
        y, sr = librosa.load(vocals_file, sr=None)
        
        # Apply noise gate
        y_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        mask = y_db > noise_gate_db
        y_gated = y * mask
        
        # Normalize
        y_normalized = librosa.util.normalize(y_gated)
        
        # Save
        if output_file is None:
            output_file = str(Path(vocals_file).parent / 'vocals_enhanced.wav')
        
        sf.write(output_file, y_normalized, sr)
        
        print(f"   ✅ Enhanced vocals saved: {output_file}")
        return output_file
        
    except ImportError:
        print("   ⚠️  Skipping enhancement (librosa not installed)")
        print("   Install with: pip install librosa")
        return None
    except Exception as e:
        print(f"   ❌ Enhancement failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Separate audio into voice, music, and other stems using Demucs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - extract vocals only
  python audio_separator.py song.mp3
  
  # Use different model
  python audio_separator.py video.mp4 --model mdx_extra
  
  # Get all 4 stems (vocals, bass, drums, other)
  python audio_separator.py music.wav --four-stems
  
  # Force CPU (no GPU)
  python audio_separator.py audio.m4a --no-gpu
  
  # With post-processing
  python audio_separator.py podcast.mp3 --enhance

Available models:
  htdemucs      - Hybrid Transformer (best quality, slower)
  htdemucs_ft   - Fine-tuned (even better quality)
  mdx_extra     - Fast and accurate (recommended for CPU)
  mdx_extra_q   - Quantized (fastest, lower memory)
        """
    )
    
    parser.add_argument('input', help='Input audio file (MP3, WAV, M4A, etc.)')
    parser.add_argument('--model', '-m', default='htdemucs',
                       choices=['htdemucs', 'htdemucs_ft', 'mdx_extra', 'mdx_extra_q'],
                       help='Model to use (default: htdemucs)')
    parser.add_argument('--output', '-o', default='separated',
                       help='Output directory (default: separated)')
    parser.add_argument('--four-stems', action='store_true',
                       help='Separate into 4 stems instead of just vocals/no_vocals')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--enhance', action='store_true',
                       help='Apply post-processing to enhance vocals')
    parser.add_argument('--noise-gate', type=int, default=-40,
                       help='Noise gate threshold in dB (default: -40)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check GPU availability
    if torch.cuda.is_available() and not args.no_gpu:
        print(f"🔥 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU (this will be slower)")
    
    # Run separation
    success = separate_audio(
        input_file=args.input,
        model=args.model,
        vocals_only=not args.four_stems,
        output_dir=args.output,
        use_gpu=not args.no_gpu
    )
    
    if not success:
        sys.exit(1)
    
    # Enhance vocals if requested
    if args.enhance and not args.four_stems:
        base_name = Path(args.input).stem
        vocals_path = Path(args.output) / args.model / base_name / 'vocals.wav'
        
        if vocals_path.exists():
            enhance_vocals(vocals_path, noise_gate_db=args.noise_gate)
    
    print("\n✨ Done! Check the output directory for separated audio files.")
    

if __name__ == '__main__':
    main()