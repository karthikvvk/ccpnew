```
bash
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VIDEO TRANSLATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   INPUT     │    │     RAG     │    │ PROCESSING  │    │   OUTPUT    │   │
│  │   VIDEO     │──▶│   CONTEXT   │──▶│   STAGES    │──▶│   VIDEO     │   │
│  │   (MP4)     │    │   (CLIP)    │    │             │    │   (MP4)     │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1: EXTRACTION              STAGE 2: UNDERSTANDING                    │
│  ┌─────────────────┐              ┌─────────────────┐                       │
│  │    FFmpeg       │              │   CLIP-ViT-B-32 │◀──Frame Embeddings   │
│  │  - Audio (WAV)  │              │      + ChromaDB │◀──Vector Store       │
│  │  - Frames (JPG) │              └─────────────────┘                       │
│  └─────────────────┘              ┌─────────────────┐                       │
│                                   │  Semantic RAG   │◀──Domain Context     │
│                                   └─────────────────┘                       │
│                                                                             │
│  STAGE 3: TRANSCRIPTION           STAGE 4: TRANSLATION                      │
│  ┌─────────────────┐              ┌─────────────────┐                       │
│  │ OpenAI Whisper  │              │   Llama-3.1-8B  │                       │
│  │    (Medium)     │──Text──────▶│    Instruct     │                       │
│  │   GPU (CUDA)    │              │  + RAG Context  │                       │
│  └─────────────────┘              │   4-bit Quant   │                       │
│                                   └─────────────────┘                       │
│                                                                             │
│  STAGE 5: TTS                     STAGE 6: SYNTHESIS                        │
│  ┌─────────────────┐              ┌─────────────────┐                       │
│  │   Edge-TTS /    │              │     FFmpeg      │                       │
│  │   XTTS-v2       │──Audio─────▶│  Audio Sync +   │                       │
│  │ (Voice Cloning) │              │  Video Merge    │                       │
│  └─────────────────┘              └─────────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


```




flowchart TD
    A[📹 Video Upload] --> B[FFmpeg Extract]
    B --> C[Audio WAV]
    B --> D[Frames JPG]
    
    D --> E[CLIP Embeddings]
    E --> F[ChromaDB]
    F --> G[Semantic RAG]
    
    C --> H[Whisper STT]
    H --> I[Tamil Text]
    
    G --> J[Domain Context]
    I --> K[Llama-3.1-8B]
    J --> K
    
    K --> L[English Text]
    L --> M[Edge-TTS]
    M --> N[Dubbed Audio]
    
    N --> O[Audio Speed Sync]
    O --> P[FFmpeg Merge]
    A --> P
    P --> Q[📹 Translated Video]
    
    L --> R[SRT Subtitles]