"""
Standalone script for OpenAI Whisper Speech-to-Text.
This script demonstrates how to load the Whisper model and transcribe an audio file.

Dependencies:
    pip install openai-whisper torch
"""
import whisper
import argparse
import sys
import torch
from pathlib import Path

def setup_whisper(model_size="base", device=None):
    """
    Load the Whisper model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Whisper model '{model_size}' on {device}...")
    try:
        model = whisper.load_model(model_size, device=device)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def transcribe_audio(model, audio_path):
    """
    Transcribe the audio file.
    """
    print(f"Transcribing '{audio_path}'...")
    if not Path(audio_path).exists():
        print(f"Error: File '{audio_path}' not found.")
        sys.exit(1)
        
    try:
        # standard transcribe
        result = model.transcribe(audio_path)
        return result
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Standalone Whisper STT Demo")
    parser.add_argument("audio_path", help="Path to the input audio file")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    args = parser.parse_args()

    model = setup_whisper(model_size=args.model)
    result = transcribe_audio(model, args.audio_path)
    
    print("-" * 50)
    print("Detected Language:", result.get('language'))
    print("-" * 50)
    print("Transcription Text:")
    print(result['text'].strip())
    print("-" * 50)

if __name__ == "__main__":
    main()
