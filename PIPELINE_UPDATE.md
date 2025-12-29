# Updated Pipeline Summary

The new video translation pipeline flow is:

1. **Video → Audio/Frames Extraction**
2. **Frame Embeddings → Vector DB**  
3. **RAG → Visual Context Generation**
4. **Whisper → Speech-to-Text** (may have errors/incomplete sentences)
5. **LLM Refiner → Fix/Complete Sentences** (same language, just fixing)
6. **Google Translate → Translate to Target Language**
7. **gTTS → Text-to-Speech**
8. **Video Reconstruction**

## Key Changes

- **LLM is now a REFINER** - fixes broken Whisper output
- **Google Translate handles actual translation** - simple, fast, reliable
- **Separate modules**:
  - `transcription_refiner.py` - uses LLM to fix sentences
  - `simple_translator.py` - uses Google Translate API

## Install Required Package

```bash
pip install googletrans==4.0.0rc1 --break-system-packages
# OR add to requirements.txt
```

## New .env Setting

Keep the LLM model for refinement (not translation):
```env
LLM_MODEL=google/flan-t5-large  # or any model, used for refining only
```

This approach is:
- ✅ **Simpler** - Google Translate is reliable
- ✅ **Faster** - No need for huge translation LLM
- ✅ **Better** - LLM fixes Whisper errors, Google translates properly
