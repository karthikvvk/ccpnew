"""
Standalone script for Google Text-to-Speech (gTTS).
This script demonstrates how to convert text to speech using Google's TTS API.

Dependencies:
    pip install gTTS
"""
from gtts import gTTS
import argparse
import os

def generate_speech(text, output_file="output.mp3", lang="en", slow=False):
    """
    Convert text to speech and save to file.
    """
    print(f"Generating speech for: '{text}'")
    print(f"Language: {lang}, Slow: {slow}")
    
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(output_file)
        print(f"Audio saved to: {output_file}")
    except Exception as e:
        print(f"Error generating speech: {e}")

def main():
    parser = argparse.ArgumentParser(description="Standalone gTTS Demo")
    parser.add_argument("--text", type=str, required=True, help="Text to convert to speech")
    parser.add_argument("--output", type=str, default="output.mp3", help="Output audio file path")
    parser.add_argument("--lang", type=str, default="en", help="Language code (e.g., en, ta, es)")
    parser.add_argument("--slow", action="store_true", help="Speak slowly")
    
    args = parser.parse_args()
    
    generate_speech(args.text, args.output, args.lang, args.slow)

if __name__ == "__main__":
    main()
