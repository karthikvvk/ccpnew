"""
Colab-compatible module for GPU-accelerated Whisper and LLM processing

This file contains standalone functions that can be used in Google Colab
for faster GPU processing of speech-to-text and translation.

Copy the relevant functions to your Colab notebook cells.
"""

# ============================================================================
# CELL 1: Install Dependencies (Run first in Colab)
# ============================================================================
"""
!pip install openai-whisper
!pip install transformers accelerate bitsandbytes
!pip install torch
"""

# ============================================================================
# CELL 2: Whisper Speech-to-Text (GPU)
# ============================================================================

def whisper_transcribe_gpu(audio_path, output_json, output_txt, 
                          model_size="medium", language=None):
    """
    Transcribe audio using Whisper on GPU
    
    Args:
        audio_path: Path to audio file (upload to Colab first)
        output_json: Path to save JSON output
        output_txt: Path to save text output
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code (e.g., 'en', 'es') or None for auto-detect
    
    Returns:
        Transcription result dictionary
    """
    import whisper
    import json
    
    print(f"Loading Whisper {model_size} on GPU...")
    model = whisper.load_model(model_size, device="cuda")
    
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(
        audio_path,
        language=language,
        verbose=True
    )
    
    # Save JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save text
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(result['text'])
    
    print(f"\n✓ Transcription saved to {output_json} and {output_txt}")
    print(f"Detected language: {result.get('language', 'unknown')}")
    print(f"Number of segments: {len(result.get('segments', []))}")
    
    return result


# Example usage in Colab:
"""
# Upload your audio file to Colab first
from google.colab import files
uploaded = files.upload()  # Upload original_audio.wav

# Run transcription
result = whisper_transcribe_gpu(
    audio_path="original_audio.wav",
    output_json="transcription.json",
    output_txt="transcription.txt",
    model_size="medium",
    language="en"  # or None for auto-detect
)

# Download results
files.download("transcription.json")
files.download("transcription.txt")
"""


# ============================================================================
# CELL 3: LLM Translation (GPU)
# ============================================================================

def llm_translate_gpu(input_json, output_json, output_txt,
                     target_language, model_name="mistralai/Mistral-7B-Instruct-v0.2",
                     visual_context=None):
    """
    Translate transcription using LLM on GPU
    
    Args:
        input_json: Path to transcription JSON (from Whisper)
        output_json: Path to save translated JSON
        output_txt: Path to save translated text
        target_language: Target language (e.g., 'Spanish', 'French')
        model_name: HuggingFace model name
        visual_context: Optional visual context string
    
    Returns:
        List of translated segments
    """
    import json
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"Loading {model_name} on GPU...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    # Load transcription
    with open(input_json, 'r', encoding='utf-8') as f:
        transcription = json.load(f)
    
    segments = transcription.get('segments', [])
    print(f"Translating {len(segments)} segments to {target_language}...")
    
    translated_segments = []
    
    for i, segment in enumerate(segments):
        # Build prompt
        context_info = f"\n\nVisual Context: {visual_context}" if visual_context else ""
        prompt = f"""[INST] You are a professional translator. Translate the following text to {target_language}.
Only provide the translation, nothing else.{context_info}

Text to translate: {segment['text']}

Translation: [/INST]"""
        
        # Generate translation
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = full_output.replace(prompt, "").strip()
        
        translated_segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'original': segment['text'],
            'translated': translation
        })
        
        if (i + 1) % 5 == 0:
            print(f"  Translated {i + 1}/{len(segments)} segments")
    
    # Save JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(translated_segments, f, indent=2, ensure_ascii=False)
    
    # Save text
    full_text = " ".join([seg['translated'] for seg in translated_segments])
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"\n✓ Translation saved to {output_json} and {output_txt}")
    return translated_segments


# Example usage in Colab:
"""
# Upload your transcription.json from previous step
from google.colab import files
uploaded = files.upload()  # Upload transcription.json

# Run translation
translated = llm_translate_gpu(
    input_json="transcription.json",
    output_json="translation.json",
    output_txt="translation.txt",
    target_language="Spanish",
    model_name="mistralai/Mistral-7B-Instruct-v0.2"
)

# Download results
files.download("translation.json")
files.download("translation.txt")
"""


# ============================================================================
# CELL 4: Complete Pipeline (Whisper + LLM) in Colab
# ============================================================================

def complete_translation_pipeline_gpu(audio_path, target_language,
                                     whisper_model="medium",
                                     llm_model="mistralai/Mistral-7B-Instruct-v0.2",
                                     source_language=None):
    """
    Complete translation pipeline in Colab
    
    Args:
        audio_path: Path to audio file
        target_language: Target language (e.g., 'Spanish')
        whisper_model: Whisper model size
        llm_model: LLM model name
        source_language: Source language or None for auto-detect
    
    Returns:
        Tuple of (transcription_result, translation_result)
    """
    print("=" * 60)
    print("STEP 1: Speech-to-Text with Whisper")
    print("=" * 60)
    
    transcription = whisper_transcribe_gpu(
        audio_path=audio_path,
        output_json="transcription.json",
        output_txt="transcription.txt",
        model_size=whisper_model,
        language=source_language
    )
    
    print("\n" + "=" * 60)
    print("STEP 2: Translation with LLM")
    print("=" * 60)
    
    translation = llm_translate_gpu(
        input_json="transcription.json",
        output_json="translation.json",
        output_txt="translation.txt",
        target_language=target_language,
        model_name=llm_model
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - transcription.json")
    print("  - transcription.txt")
    print("  - translation.json")
    print("  - translation.txt")
    
    return transcription, translation


# Example usage in Colab:
"""
# Upload audio file
from google.colab import files
uploaded = files.upload()  # Upload your audio file

# Run complete pipeline
audio_filename = list(uploaded.keys())[0]  # Get uploaded filename

transcription, translation = complete_translation_pipeline_gpu(
    audio_path=audio_filename,
    target_language="French",
    whisper_model="medium",
    source_language=None  # Auto-detect
)

# Download all results
files.download("transcription.json")
files.download("transcription.txt")
files.download("translation.json")
files.download("translation.txt")
"""


# ============================================================================
# CELL 5: Performance Monitoring
# ============================================================================

def check_gpu_status():
    """Check GPU availability and memory"""
    import torch
    
    if torch.cuda.is_available():
        print("✓ GPU is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("✗ No GPU available - using CPU")
        print("Note: Make sure to enable GPU in Colab:")
        print("  Runtime → Change runtime type → Hardware accelerator: GPU")


# Run this to check GPU status:
"""
check_gpu_status()
"""
