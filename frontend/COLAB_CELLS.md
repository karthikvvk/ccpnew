# ===============================================================
# AuraFlow FastAPI Backend - Colab Setup
# ===============================================================
# 
# INSTRUCTIONS:
# 1. Open Google Colab: https://colab.research.google.com
# 2. Create a new notebook
# 3. Change runtime to GPU: Runtime → Change runtime type → T4 GPU
# 4. Copy-paste each section into separate code cells
# 5. Run cells in order
# 6. Copy the ngrok URL when it appears
# 7. Set VITE_API_URL in frontend to the ngrok URL
#
# ===============================================================

# ===================== CELL 1: Install Dependencies =====================
# Run this cell first to install all required packages

!pip install -q fastapi uvicorn[standard] pydantic-settings ffmpeg-python opencv-python pydub librosa soundfile openai-whisper "numpy<2.0" chromadb sentence-transformers transformers sentencepiece accelerate bitsandbytes Pillow python-dotenv aiofiles googletrans==4.0.0rc1 deep-translator gtts edge-tts python-multipart sounddevice scipy pyngrok nest-asyncio

# Install ffmpeg system package
!apt-get install -y ffmpeg > /dev/null 2>&1

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  No GPU - enable in Runtime → Change runtime type → T4 GPU")


# ===================== CELL 2: Clone Repository =====================
# Clone the AuraFlow backend code

import os

# Clone repo
if not os.path.exists('Aura-Flow-Mutli-Agent-Content-Localiation-Engine'):
    !git clone https://github.com/Vishal-V-D/Aura-Flow-Mutli-Agent-Content-Localiation-Engine.git
    print("✅ Repository cloned")
else:
    print("✅ Repository already exists")

# Change to backend directory
os.chdir('Aura-Flow-Mutli-Agent-Content-Localiation-Engine/ccpnew')
print(f"📁 Working directory: {os.getcwd()}")


# ===================== CELL 3: Configure ngrok =====================
# Set up ngrok for public URL access

from pyngrok import ngrok

# IMPORTANT: Replace with your ngrok authtoken from https://dashboard.ngrok.com/auth
# Free account gives you 1 tunnel
NGROK_AUTHTOKEN = "YOUR_NGROK_AUTHTOKEN_HERE"  # ⚠️ Replace this!

if NGROK_AUTHTOKEN != "YOUR_NGROK_AUTHTOKEN_HERE":
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    print("✅ ngrok configured")
else:
    print("⚠️  Please set NGROK_AUTHTOKEN in the cell above!")
    print("   Get your token from: https://dashboard.ngrok.com/auth")


# ===================== CELL 4: Start FastAPI Server =====================
# This runs the full FastAPI backend with ngrok tunnel

import nest_asyncio
import threading
import uvicorn
import sys

# Allow nested event loops (required for Colab)
nest_asyncio.apply()

# Add backend to path
sys.path.insert(0, os.getcwd())

# Import the FastAPI app
from main import app

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print("\n" + "="*60)
print("🚀 AURAFLOW BACKEND IS RUNNING!")
print("="*60)
print(f"\n📡 Public URL: {public_url.public_url}")
print(f"\n📋 Swagger UI: {public_url.public_url}/docs")
print(f"\n🔧 Add to frontend .env.development:")
print(f"   VITE_API_URL={public_url.public_url}")
print("\n" + "="*60)
print("⚠️  Keep this cell running! Don't close the notebook.")
print("="*60 + "\n")

# Run uvicorn in the main thread (blocking)
# The Colab notebook will keep running as long as this cell is active
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# ===================== ALTERNATIVE: Run in Background =====================
# If you want to run other cells while server is running, use this instead:
# 
# def run_server():
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
# 
# server_thread = threading.Thread(target=run_server, daemon=True)
# server_thread.start()
# print("✅ Server running in background")
#
# Then you can run additional cells for testing, etc.
