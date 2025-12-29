# Video Translation System

A modular FastAPI-based video translation system that uses RAG-enhanced speech recognition to provide context-aware transcription, translation, and audio dubbing.

## Features

- **Video Processing**: Extract audio and frames from videos
- **RAG-Enhanced Transcription**: Use visual context to improve speech recognition accuracy
- **Local Processing**: Run Whisper and LLM models locally (CPU) or in Google Colab (GPU)
- **Multiple Languages**: Support for multiple source and target languages
- **Modular Architecture**: Easy to extend and customize
- **FastAPI Backend**: RESTful API for video translation
- **Comprehensive Logging**: Track all intermediate files and processing stages

## Architecture

```
Video Input → Audio/Frame Extraction → Vector DB (Frame Embeddings) 
    → RAG Context Generation → Whisper STT (with context)
    → LLM Translation (with context) → gTTS → Video Reconstruction
```

## Installation

### Prerequisites

- Python 3.9 or higher
- ffmpeg installed on your system
- (Optional) CUDA-capable GPU for faster processing

### Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` to configure settings:
   - `WHISPER_MODEL`: Whisper model size (tiny, base, small, medium, large)
   - `LLM_MODEL`: Local LLM model for translation
   - `WHISPER_DEVICE`: cpu or cuda
   - `LLM_DEVICE`: cpu or cuda

## Usage

### Option 1: FastAPI Server

1. Start the server:
```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

3. API Documentation: `http://localhost:8000/docs`

4. Upload and translate a video:
```python
import requests

with open('video.mp4', 'rb') as f:
    files = {'video': f}
    data = {
        'target_language': 'Spanish',
        'source_language': 'auto',
        'use_rag': True
    }
    response = requests.post('http://localhost:8000/api/v1/translate', 
                           files=files, data=data)

job_id = response.json()['job_id']
print(f"Job ID: {job_id}")
```

5. Check status:
```python
status = requests.get(f'http://localhost:8000/api/v1/status/{job_id}')
print(status.json())
```

### Option 2: Direct Pipeline Usage

```python
from services.pipeline import TranslationPipeline
from pathlib import Path

pipeline = TranslationPipeline()
result = pipeline.process(
    video_path=Path("video.mp4"),
    target_language="French",
    source_language="auto",
    use_rag=True
)

print(f"Final video: {result['files']['final_video']}")
```

### Option 3: Google Colab (GPU Acceleration)

For faster processing of Whisper and LLM on GPU:

1. **Open the Colab notebook**: `colab_whisper_llm.ipynb`
2. **Upload your files** to Colab
3. **Run the cells** to process with GPU acceleration
4. **Download the results**

The notebook includes standalone functions that can run independently in Colab.

## Output Files

For each translation job, the following files are generated in `outputs/{job_id}/`:

1. `frames/` - Extracted video frames
2. `vector_db/` - ChromaDB vector database
3. `original_audio.wav` - Extracted original audio
4. `transcription.json` - Full Whisper transcription with timing
5. `transcription.txt` - Plain text transcription
6. `rag_context.json` - Visual context from frames
7. `translation.json` - Translated segments with timing
8. `translation.txt` - Plain text translation
9. `translated_audio.mp3` - Generated speech in target language
10. `final_video.mp4` - Final video with dubbed audio
11. `manifest.json` - Index of all generated files

## Module Structure

```
├── modules/
│   ├── video_processor.py      # Video/audio extraction & reconstruction
│   ├── frame_embedder.py       # Frame embedding generation
│   ├── vector_store.py         # ChromaDB integration
│   ├── rag_context.py          # Visual context generation
│   ├── speech_to_text.py       # Whisper STT
│   ├── translator.py           # LLM translation
│   └── text_to_speech.py       # gTTS synthesis
├── api/
│   └── endpoints.py            # FastAPI endpoints
├── models/
│   └── schemas.py              # Pydantic models
├── services/
│   └── pipeline.py             # Main orchestration
├── utils/
│   ├── logger.py               # Logging utilities
│   └── file_manager.py         # File management
├── config.py                   # Configuration
├── main.py                     # FastAPI app
└── requirements.txt            # Dependencies
```

## API Endpoints

### POST /api/v1/translate
Upload and translate a video

**Parameters:**
- `video` (file): Video file
- `target_language` (string): Target language (e.g., "Spanish", "French")
- `source_language` (string): Source language or "auto"
- `use_rag` (boolean): Enable RAG context

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Translation job queued successfully"
}
```

### GET /api/v1/status/{job_id}
Check translation status

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "progress": "Completed",
  "files": [
    {
      "file_type": "final_video",
      "file_path": "/path/to/final_video.mp4",
      "exists": true
    }
  ]
}
```

### GET /api/v1/download/{job_id}/{file_type}
Download a specific file

**Parameters:**
- `job_id`: Job identifier
- `file_type`: Type of file (e.g., "final_video", "transcription_json")

## Customization

### Using Different Models

Edit `.env`:

```env
# Whisper model sizes: tiny, base, small, medium, large
WHISPER_MODEL=large

# LLM models (HuggingFace)
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Or use other models:
# LLM_MODEL=meta-llama/Llama-2-7b-chat-hf
# LLM_MODEL=google/flan-t5-large
```

### Disable RAG

If you want faster processing without visual context:

```python
pipeline.process(
    video_path=video_path,
    target_language="Spanish",
    use_rag=False  # Disable RAG
)
```

## Performance Tips

1. **Use GPU**: Set `WHISPER_DEVICE=cuda` and `LLM_DEVICE=cuda` in `.env`
2. **Use Colab**: For free GPU access, use the provided Colab notebook
3. **Smaller Models**: Use `tiny` or `base` Whisper for faster processing
4. **Reduce FPS**: Lower `FRAME_EXTRACT_FPS` for fewer frames to process

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg using your system package manager.

### Out of Memory
- Use smaller models (tiny/base for Whisper)
- Disable RAG with `use_rag=False`
- Process on CPU instead of GPU
- Use Colab with higher RAM

### Slow Processing
- Use GPU instead of CPU
- Use smaller models
- Reduce frame extraction FPS
- Use Colab for GPU acceleration

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
