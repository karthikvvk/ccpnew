"""
Flask server for Colab - Exposes Whisper and LLM as API endpoints
This runs in a single Colab cell and processes requests from your local machine

Copy this entire cell into Colab and run it
"""

# Cell 1: Install dependencies and start Flask server
CODE = """
# Install dependencies
!pip install -q flask flask-cors pyngrok openai-whisper transformers accelerate bitsandbytes

import os
import json
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyngrok import ngrok
import tempfile

app = Flask(__name__)
CORS(app)

# Global model cache
whisper_model = None
llm_model = None
llm_tokenizer = None

@app.route('/health', methods=['GET'])
def health():
    '''Health check endpoint'''
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU"
    
    return jsonify({
        'status': 'healthy',
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'whisper_loaded': whisper_model is not None,
        'llm_loaded': llm_model is not None
    })

@app.route('/load_whisper', methods=['POST'])
def load_whisper_model():
    '''Load Whisper model into memory'''
    global whisper_model
    
    data = request.json
    model_size = data.get('model_size', 'medium')
    
    print(f"Loading Whisper {model_size}...")
    whisper_model = whisper.load_model(model_size, device="cuda")
    
    return jsonify({
        'status': 'success',
        'message': f'Whisper {model_size} loaded on GPU'
    })

@app.route('/load_llm', methods=['POST'])
def load_llm_model():
    '''Load LLM model into memory'''
    global llm_model, llm_tokenizer
    
    data = request.json
    model_name = data.get('model_name', 'mistralai/Mistral-7B-Instruct-v0.2')
    
    print(f"Loading {model_name}...")
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    return jsonify({
        'status': 'success',
        'message': f'{model_name} loaded on GPU'
    })

@app.route('/whisper/transcribe', methods=['POST'])
def transcribe():
    '''Transcribe audio using Whisper'''
    global whisper_model
    
    if whisper_model is None:
        return jsonify({'error': 'Whisper model not loaded. Call /load_whisper first'}), 400
    
    # Get audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', None)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        audio_file.save(temp_audio.name)
        temp_path = temp_audio.name
    
    try:
        print(f"Transcribing {temp_path}...")
        result = whisper_model.transcribe(
            temp_path,
            language=language if language != 'auto' else None,
            verbose=False
        )
        
        # Clean up
        os.unlink(temp_path)
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        os.unlink(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/llm/translate', methods=['POST'])
def translate():
    '''Translate text using LLM'''
    global llm_model, llm_tokenizer
    
    if llm_model is None or llm_tokenizer is None:
        return jsonify({'error': 'LLM model not loaded. Call /load_llm first'}), 400
    
    data = request.json
    segments = data.get('segments', [])
    target_language = data.get('target_language', 'Spanish')
    visual_context = data.get('visual_context', None)
    
    print(f"Translating {len(segments)} segments to {target_language}...")
    
    translated_segments = []
    
    for i, segment in enumerate(segments):
        # Build prompt
        context_info = f"\\n\\nVisual Context: {visual_context}" if visual_context else ""
        prompt = f'''[INST] You are a professional translator. Translate the following text to {target_language}.
Only provide the translation, nothing else.{context_info}

Text to translate: {segment['text']}

Translation: [/INST]'''
        
        # Generate
        inputs = llm_tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=llm_tokenizer.eos_token_id
            )
        
        full_output = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = full_output.replace(prompt, "").strip()
        
        translated_segments.append({
            'start': segment['start'],
            'end': segment['end'],
            'original': segment['text'],
            'translated': translation
        })
        
        if (i + 1) % 5 == 0:
            print(f"  Translated {i + 1}/{len(segments)}")
    
    return jsonify({
        'status': 'success',
        'translated_segments': translated_segments
    })

# Start ngrok tunnel
print("Starting ngrok tunnel...")
public_url = ngrok.connect(5000)
print(f"\\n{'='*60}")
print(f"🚀 Colab GPU Server is running!")
print(f"{'='*60}")
print(f"Public URL: {public_url}")
print(f"\\nAdd this to your local .env file:")
print(f"COLAB_API_URL={public_url}")
print(f"USE_COLAB_GPU=True")
print(f"{'='*60}\\n")

# Run Flask
app.run(port=5000)
"""

print("="*70)
print("COLAB FLASK SERVER - GPU ACCELERATION")
print("="*70)
print("\\nCopy the code below into a Colab cell and run it:")
print("="*70)
print(CODE)
print("="*70)
