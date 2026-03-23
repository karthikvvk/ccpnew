"""
Standalone script for Coqui XTTS-v2 Voice Cloning.
This script demonstrates how to clone a voice from a reference audio file and generate speech.

Dependencies:
    pip install TTS torch
"""
import argparse
import sys
import torch
import os
from pathlib import Path

# Try to import TTS
try:
    from TTS.api import TTS
except ImportError:
    print("Error: 'TTS' library not found. Please install it using 'pip install TTS'.")
    sys.exit(1)

def setup_xtts(device=None):
    """
    Load the XTTS-v2 model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading XTTS-v2 model on {device} (this may take a while)...")
    try:
        # Using the specific model used in the project
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        if device == "cuda":
            model.to("cuda")
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def clone_voice(model, text, reference_audio, output_path="output_cloned.wav", language="en"):
    """
    Generate speech using voice cloning.
    """
    print(f"Generating speech...")
    print(f"Text: '{text}'")
    print(f"Reference Audio: '{reference_audio}'")
    print(f"Language: '{language}'")
    
    if not Path(reference_audio).exists():
        print(f"Error: Reference audio file '{reference_audio}' not found.")
        sys.exit(1)
        
    try:
        model.tts_to_file(
            text=text,
            speaker_wav=str(reference_audio),
            language=language,
            file_path=str(output_path)
        )
        print(f"Cloned audio saved to: {output_path}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Standalone Coqui XTTS Voice Cloning Demo")
    parser.add_argument("--text", type=str, required=True, help="Text to speak")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference audio (wav/mp3) for cloning")
    parser.add_argument("--output", type=str, default="output_cloned.wav", help="Output audio file path")
    parser.add_argument("--lang", type=str, default="en", help="Language code (e.g., en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi)")
    
    args = parser.parse_args()
    
    model = setup_xtts()
    clone_voice(model, args.text, args.ref, args.output, args.lang)

if __name__ == "__main__":
    main()
