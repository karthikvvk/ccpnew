# Video Translation Project Structure

## Overview
This project implements a modular video translation system with RAG-enhanced speech recognition. The system can run locally (CPU) or leverage Google Colab for GPU-accelerated processing.

## Project Layout

```
ccpnew/
├── config.py                   # Configuration management
├── main.py                     # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration
├── README.md                   # Project documentation
├── colab_gpu_modules.py        # Colab-compatible GPU functions
│
├── modules/                    # Core processing modules
│   ├── video_processor.py      # Video/audio extraction & reconstruction
│   ├── frame_embedder.py       # CLIP-based frame embeddings
│   ├── vector_store.py         # ChromaDB vector database
│   ├── rag_context.py          # Visual context generation
│   ├── speech_to_text.py       # Whisper STT (Colab-ready)
│   ├── translator.py           # LLM translation (Colab-ready)
│   └── text_to_speech.py       # gTTS speech synthesis
│
├── api/                        # FastAPI endpoints
│   └── endpoints.py            # REST API routes
│
├── models/                     # Data models
│   └── schemas.py              # Pydantic schemas
│
├── services/                   # Business logic
│   └── pipeline.py             # Main orchestration pipeline
│
├── utils/                      # Utilities
│   ├── logger.py               # Logging system
│   └── file_manager.py         # File tracking and management
│
├── examples/                   # Usage examples
│   └── example_usage.py        # API and pipeline examples
│
└── outputs/                    # Generated files (auto-created)
    └── {job_id}/
        ├── frames/             # Extracted frames
        ├── vector_db/          # ChromaDB database
        ├── original_audio.wav
        ├── transcription.json
        ├── transcription.txt
        ├── rag_context.json
        ├── translation.json
        ├── translation.txt
        ├── translated_audio.mp3
        ├── final_video.mp4
        └── manifest.json
```

## Module Descriptions

### Core Modules

**video_processor.py**
- Extracts audio from video using ffmpeg
- Extracts frames at specified FPS
- Reconstructs video with new audio track
- Provides video metadata

**frame_embedder.py**
- Uses CLIP model to generate visual embeddings
- Batch processes frames efficiently
- Supports both single and multiple frame embedding

**vector_store.py**
- ChromaDB integration for storing frame embeddings
- Similarity search for relevant frames
- Persistent storage with metadata

**rag_context.py**
- Generates textual descriptions from frames
- Uses vision-language model (GIT or similar)
- Provides visual context for transcription
- **Colab-ready**: Can run on GPU for faster processing

**speech_to_text.py**
- Local Whisper model for speech recognition
- Supports RAG context integration
- Handles multiple languages
- **Colab-ready**: Includes GPU-accelerated function

**translator.py**
- Local LLM for translation (Mistral/Llama)
- Context-aware translation using RAG
- Preserves segment timing
- **Colab-ready**: Includes GPU-accelerated function

**text_to_speech.py**
- gTTS for speech synthesis
- Segment-based audio generation
- Audio duration adjustment

### API Layer

**endpoints.py**
- POST `/api/v1/translate` - Upload and translate video
- GET `/api/v1/status/{job_id}` - Check processing status
- GET `/api/v1/download/{job_id}/{file_type}` - Download files
- Background task processing

**schemas.py**
- Request/response models
- Status enumerations
- File information models

### Services

**pipeline.py**
- Orchestrates all modules
- Manages workflow: extract → embed → RAG → STT → translate → TTS → reconstruct
- Tracks all intermediate files
- Error handling and logging

### Utilities

**logger.py**
- Centralized logging
- File and console output
- Configurable log levels

**file_manager.py**
- Job-specific directory creation
- File tracking and manifest generation
- Cleanup utilities

## Workflow

1. **Video Input** → Upload via API or provide path
2. **Audio Extraction** → Extract audio track (WAV format)
3. **Frame Extraction** → Extract frames at configured FPS
4. **Frame Embedding** → Generate CLIP embeddings for all frames
5. **Vector Storage** → Store embeddings in ChromaDB
6. **RAG Context** → Query relevant frames and generate descriptions
7. **Speech-to-Text** → Whisper transcription with visual context
8. **Translation** → LLM translates text using RAG context
9. **Text-to-Speech** → gTTS generates audio in target language
10. **Video Reconstruction** → Combine original video with new audio

## GPU Acceleration (Colab)

For steps that benefit from GPU:
- **Whisper (STT)**: Use `colab_gpu_modules.py::whisper_transcribe_gpu()`
- **LLM (Translation)**: Use `colab_gpu_modules.py::llm_translate_gpu()`

These can run independently in Colab and results can be integrated back into the pipeline.

## Configuration

Key settings in `.env`:

```env
# Models
WHISPER_MODEL=medium          # tiny, base, small, medium, large
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Devices
WHISPER_DEVICE=cpu            # cpu or cuda
LLM_DEVICE=cpu                # cpu or cuda

# Processing
FRAME_EXTRACT_FPS=1           # Frames per second to extract
```

## File Logging

All intermediate files are logged and tracked:
1. Frames directory path
2. Vector DB path
3. Original audio
4. Transcription (JSON + TXT)
5. RAG context (JSON)
6. Translation (JSON + TXT)
7. Translated audio
8. Final video
9. Manifest (lists all files)

Each job gets a unique directory with complete audit trail.
