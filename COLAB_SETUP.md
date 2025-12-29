# Colab GPU Server Setup Guide

## Quick Start

### 1. Open Colab Notebook
Open `colab_api_server.ipynb` in Google Colab

### 2. Enable GPU
- Runtime → Change runtime type
- Hardware accelerator: **GPU**
- Save

### 3. Run the Server Cell
Run the single cell in the notebook. It will:
- Install dependencies
- Start Flask server
- Create ngrok tunnel
- Display a public URL

### 4. Copy the URL
You'll see output like:
```
============================================================
🚀 Colab GPU Server is running!
============================================================
Public URL: https://xxxx-xx-xxx-xxx-xx.ngrok.io

Add this to your local .env file:
COLAB_API_URL=https://xxxx-xx-xxx-xxx-xx.ngrok.io
USE_COLAB_GPU=True
============================================================
```

### 5. Update Local .env
Add to your `/home/muruga/workspace/ccpnew/.env`:
```env
USE_COLAB_GPU=True
COLAB_API_URL=https://xxxx-xx-xxx-xxx-xx.ngrok.io
```
(Replace with your actual ngrok URL)

### 6. Restart Local Server
```bash
# Stop current server (Ctrl+C)
# Start again
python main.py
```

### 7. Upload Videos as Normal
```bash
curl -X POST http://localhost:8000/api/v1/translate \
  -F "video=@video.mp4" \
  -F "target_language=Spanish" \
  -F "use_rag=true"
```

## How It Works

```
┌─────────────────┐
│  Your Computer  │
│  (FastAPI)      │
└────────┬────────┘
         │
         │ 1. Extract audio/frames locally
         │ 2. Send audio to Colab
         ▼
┌─────────────────┐
│  Colab (GPU)    │
│  Flask Server   │
│                 │
│  Whisper STT ───┤ 3. Transcribe on GPU
│  LLM Translate ─┤ 4. Translate on GPU
└────────┬────────┘
         │
         │ 5. Return results
         ▼
┌─────────────────┐
│  Your Computer  │
│  (FastAPI)      │
│                 │
│  gTTS ──────────┤ 6. Generate speech locally
│  Video Recon ───┤ 7. Create final video
└─────────────────┘
```

## Benefits

- ⚡ **30-50x faster** Whisper transcription
- ⚡ **5-10x faster** LLM translation
- 💰 **Free GPU** from Google Colab
- 🔄 **Automatic fallback** to local if Colab fails
- 📦 **No setup** on your machine needed

## API Endpoints (Colab Server)

### Health Check
```bash
curl https://your-ngrok-url.ngrok.io/health
```

### Load Whisper Model
```bash
curl -X POST https://your-ngrok-url.ngrok.io/load_whisper \
  -H "Content-Type: application/json" \
  -d '{"model_size": "medium"}'
```

### Load LLM Model
```bash
curl -X POST https://your-ngrok-url.ngrok.io/load_llm \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mistralai/Mistral-7B-Instruct-v0.2"}'
```

### Transcribe Audio
```bash
curl -X POST https://your-ngrok-url.ngrok.io/whisper/transcribe \
  -F "audio=@audio.wav" \
  -F "language=auto"
```

### Translate Text
```bash
curl -X POST https://your-ngrok-url.ngrok.io/llm/translate \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [...],
    "target_language": "Spanish"
  }'
```

## Troubleshooting

### Ngrok URL Expired
Ngrok URLs expire after a few hours. When that happens:
1. Restart the Colab cell
2. Copy the new URL
3. Update `.env`
4. Restart local server

### Colab Disconnected
If Colab disconnects:
1. Rerun the cell
2. Get new URL
3. Update `.env`

### Fallback to Local
If Colab fails, the system automatically falls back to local CPU processing. Check logs for:
```
Failed to transcribe via Colab: ...
Falling back to local transcription...
```

### GPU Not Available
Make sure GPU is enabled in Colab:
- Runtime → Change runtime type → GPU

Check with:
```bash
curl https://your-ngrok-url.ngrok.io/health
```
Should show `"gpu_available": true`

## Notes

- Keep Colab notebook open while processing
- Ngrok URLs change each time you restart
- First request loads models (slower), subsequent requests are fast
- Models stay loaded in memory for the session
