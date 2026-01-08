"""
Flask server for Colab - GPU acceleration for Whisper and LLM
Uses edge-tts for TTS (stable, high quality)

Copy this entire code into a Colab cell and run it
"""

CODE = """
# Install dependencies
!pip install -q flask flask-cors pyngrok
!pip install -q openai-whisper
!pip install -q transformers accelerate bitsandbytes sentencepiece
!pip install -q edge-tts

import os
import asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from pyngrok import ngrok
import tempfile
import edge_tts

app = Flask(__name__)
CORS(app)

# Global model cache
whisper_model = None
refiner_model = None
refiner_tokenizer = None

# Edge TTS voices by language
EDGE_VOICES = {
    'en': 'en-US-AriaNeural',
    'es': 'es-ES-AlvaroNeural',
    'fr': 'fr-FR-DeniseNeural',
    'de': 'de-DE-ConradNeural',
    'it': 'it-IT-DiegoNeural',
    'pt': 'pt-BR-FranciscaNeural',
    'ja': 'ja-JP-NanamiNeural',
    'ko': 'ko-KR-InJoonNeural',
    'zh-cn': 'zh-CN-XiaoxiaoNeural',
    'hi': 'hi-IN-MadhurNeural'
}

@app.route('/health', methods=['GET'])
def health():
    '''Health check'''
    return jsonify({
        'status': 'healthy',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
        'whisper_loaded': whisper_model is not None,
        'refiner_loaded': refiner_model is not None
    })

# ============ WHISPER ============
@app.route('/load_whisper', methods=['POST'])
def load_whisper():
    '''Load Whisper model on GPU'''
    global whisper_model
    
    data = request.json or {}
    model_size = data.get('model_size', 'medium')
    
    print(f"Loading Whisper {model_size} on GPU...")
    whisper_model = whisper.load_model(model_size, device="cuda")
    print("✅ Whisper loaded!")
    
    return jsonify({'status': 'success', 'model': model_size})

@app.route('/whisper/transcribe', methods=['POST'])
def transcribe():
    '''Transcribe audio using Whisper on GPU'''
    global whisper_model
    
    if whisper_model is None:
        return jsonify({'error': 'Call /load_whisper first'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', None)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        audio_file.save(f.name)
        temp_path = f.name
    
    try:
        print("Transcribing with optimized settings...")
        result = whisper_model.transcribe(
            temp_path,
            language=language if language != 'auto' else None,
            verbose=False,
            # Precision settings (critical for accuracy)
            temperature=0.0,
            word_timestamps=True,
            condition_on_previous_text=False,
            fp16=True
        )
        os.unlink(temp_path)
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        os.unlink(temp_path)
        return jsonify({'error': str(e)}), 500

# ============ LLM REFINER ============
@app.route('/load_refiner', methods=['POST'])
def load_refiner():
    '''Load LLM refiner (Flan-T5) on GPU'''
    global refiner_model, refiner_tokenizer
    
    data = request.json or {}
    model_name = data.get('model_name', 'google/flan-t5-large')
    
    print(f"Loading {model_name} on GPU...")
    refiner_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if 't5' in model_name.lower() or 'flan' in model_name.lower():
        refiner_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        refiner_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True
        )
    print("✅ Refiner loaded!")
    
    return jsonify({'status': 'success', 'model': model_name})

@app.route('/load_llm', methods=['POST'])
def load_llm():
    '''Alias for load_refiner'''
    return load_refiner()

@app.route('/llm/refine', methods=['POST'])
def refine():
    '''Refine/fix transcription segments using LLM on GPU'''
    global refiner_model, refiner_tokenizer
    
    if refiner_model is None:
        return jsonify({'error': 'Call /load_refiner first'}), 400
    
    data = request.json
    segments = data.get('segments', [])
    visual_context = data.get('visual_context', None)
    
    print(f"Refining {len(segments)} segments on GPU...")
    refined_segments = []
    
    for i, seg in enumerate(segments):
        text = seg.get('text', '')
        
        if visual_context:
            prompt = f"Fix grammar and complete sentences. Context: {visual_context}. Text: {text}"
        else:
            prompt = f"Fix grammar and complete sentences: {text}"
        
        inputs = refiner_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = refiner_model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
        
        refined = refiner_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        refined_segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'original': text,
            'refined': refined.strip()
        })
        
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(segments)} done")
    
    print("✅ Refinement complete!")
    return jsonify({'status': 'success', 'refined_segments': refined_segments})

# ============ TTS (edge-tts) ============
@app.route('/tts/generate', methods=['POST'])
def generate_tts():
    '''Generate speech using edge-tts'''
    text = request.form.get('text', '')
    language = request.form.get('language', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    voice = EDGE_VOICES.get(language, 'en-US-AriaNeural')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        output_path = f.name
    
    async def generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
    
    try:
        print(f"Generating TTS: {text[:50]}...")
        asyncio.run(generate())
        return send_file(output_path, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start server
print("Starting ngrok tunnel...")
public_url = ngrok.connect(5000)
print(f"\\n{'='*50}")
print(f"🚀 COLAB GPU SERVER READY")
print(f"{'='*50}")
print(f"URL: {public_url}")
print(f"\\nSet in your local .env:")
print(f"  COLAB_API_URL={public_url}")
print(f"  USE_COLAB_GPU=True")
print(f"{'='*50}")
print(f"\\nEndpoints:")
print(f"  POST /load_whisper       - Load Whisper")
print(f"  POST /load_refiner       - Load Flan-T5")
print(f"  POST /whisper/transcribe - Transcribe audio")
print(f"  POST /llm/refine         - Fix sentences")
print(f"  POST /tts/generate       - Generate speech")
print(f"{'='*50}\\n")

app.run(port=5000)
"""

print("="*60)
print("COLAB GPU SERVER - Whisper + LLM + Edge-TTS")
print("="*60)
print("\\nThis server handles:")
print("  1. Whisper transcription (GPU)")
print("  2. LLM sentence refinement (GPU)")
print("  3. Edge-TTS speech synthesis (stable, high quality)")
print("\\nTranslation, video processing run locally.")
print("="*60)
print("\\nCopy below into a Colab cell:")
print("="*60)
print(CODE)
