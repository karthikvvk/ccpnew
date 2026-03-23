# ===================================================
# XTTS-v2 Voice Cloning Test - Colab Scratch File
# ===================================================
# Upload this to Colab and run cell by cell
# Upload a reference audio file (WAV/MP3) to clone the voice

# Cell 1: Install dependencies
# !pip install -q TTS

# Cell 2: Import and load model
from TTS.api import TTS
import torch

print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load XTTS-v2 model
print("Loading XTTS-v2 model (first run downloads ~2GB)...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
print("✅ Model loaded!")

# Cell 3: Upload reference audio
from google.colab import files
print("Upload a reference audio file (the voice to clone):")
uploaded = files.upload()
reference_audio = list(uploaded.keys())[0]
print(f"Reference audio: {reference_audio}")

# Cell 4: Generate cloned speech
text = "Hello! This is a test of voice cloning using XTTS version 2."
language = "en"  # en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

output_file = "cloned_output.wav"

print(f"Cloning voice with text: {text}")
tts.tts_to_file(
    text=text,
    speaker_wav=reference_audio,
    language=language,
    file_path=output_file
)
print(f"✅ Done! Output saved to: {output_file}")

# Cell 5: Play the output
from IPython.display import Audio
Audio(output_file)

# Cell 6: Download the output
files.download(output_file)
