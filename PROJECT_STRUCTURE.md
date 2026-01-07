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
│   ├── semantic_rag.py         # Semantic RAG with self-pruning
│   ├── rag_context.py          # Visual context generation
│   ├── speech_to_text.py       # Whisper STT (Colab-ready)
│   ├── transcription_refiner.py # LLM sentence completion (Flan-T5)
│   ├── simple_translator.py    # Google Translate wrapper
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
└── outputs/                    # Generated files (auto-created)
    └── {job_id}/
        ├── frames/             # Extracted frames
        ├── vector_db/          # ChromaDB database
        ├── original_audio.wav
        ├── transcription.json
        ├── refined_transcription.json
        ├── rag_analysis.json
        ├── translation.json
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

**semantic_rag.py**
- CLIP-based semantic matching with hierarchical clustering
- Temporal voting and self-pruning (configurable)
- Generates confidence scores for visual context
- Self-prunes when confidence < 0.3 (production mode)

**speech_to_text.py**
- Local Whisper model for speech recognition
- Supports RAG context integration
- Handles multiple languages
- **Colab-ready**: Includes GPU-accelerated function

**transcription_refiner.py**
- Uses Flan-T5 (CPU-optimized) to fix Whisper output
- Corrects grammar, completes broken sentences
- Integrates visual context for better refinement

**simple_translator.py**
- Google Translate wrapper for reliable translation
- Fast, no heavy model loading
- Handles segment-based translation with timing

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
6. **Semantic RAG** → Analyze frames, generate visual context (with self-pruning)
7. **Speech-to-Text** → Whisper transcription with visual context
8. **LLM Refinement** → Fix/complete broken sentences (Flan-T5)
9. **Translation** → Google Translate to target language
10. **Text-to-Speech** → gTTS generates audio in target language
11. **Video Reconstruction** → Combine original video with new audio

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
LLM_MODEL=google/flan-t5-large  # CPU-optimized for sentence completion

# Devices
WHISPER_DEVICE=cpu            # cpu or cuda
LLM_DEVICE=cpu                # cpu or cuda

# RAG Toggle
RAG_ENABLE_SELF_PRUNING=False # False=testing, True=production

# Processing
FRAME_EXTRACT_FPS=2           # Frames per second to extract
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
