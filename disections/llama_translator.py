"""
Standalone script for Llama-3.1-8B-Instruct Translation.
This script demonstrates how to use a local LLM for context-aware translation.

Dependencies:
    pip install torch transformers bitsandbytes accelerate
"""
import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def setup_llama(load_in_4bit=True):
    """
    Load the Llama model and tokenizer.
    """
    print(f"Loading Llama model: {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        if load_in_4bit:
            print("Loading in 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            print("Loading in full precision (fp16)...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
        print("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: You need access to the model on Hugging Face and be logged in via 'huggingface-cli login'.")
        sys.exit(1)

def build_prompt(text, target_language, context=None):
    """
    Construct the translation prompt.
    """
    domain_info = f"Domain context: {context}" if context else "Domain: general conversation"
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional translator specializing in translation to {target_language}.

CRITICAL INSTRUCTION:
Your PRIMARY source of truth is the "Source Text" provided below. You must translate exactly what is said.
The "Domain Context" is provided ONLY to help with specific terminology or background understanding.
NEVER let the "Domain Context" override or change the meaning of the "Source Text".

{domain_info}

Rules:
- Translate naturally, preserving the original meaning and tone
- Keep technical terms and proper nouns accurate
- Output ONLY the translation, no explanations
<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate this text to {target_language}:
{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

def translate(tokenizer, model, text, target_lang, context=None):
    """
    Generate translation.
    """
    prompt = build_prompt(text, target_lang, context)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract translation
    if "assistant" in full_output.lower():
        parts = full_output.split("assistant")
        translation = parts[-1].strip()
    else:
        translation = full_output.replace(prompt, "").strip()
        
    return translation

def main():
    parser = argparse.ArgumentParser(description="Standalone Llama Translation Demo")
    parser.add_argument("text", help="Text to translate")
    parser.add_argument("--target", default="english", help="Target language")
    parser.add_argument("--context", help="Optional context for the translation")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    tokenizer, model = setup_llama(load_in_4bit=not args.no_4bit)
    
    translation = translate(tokenizer, model, args.text, args.target, args.context)
    
    print("-" * 50)
    print("Original:", args.text)
    print("Target:", args.target)
    if args.context:
        print("Context:", args.context)
    print("-" * 50)
    print("Translation:")
    print(translation)
    print("-" * 50)

if __name__ == "__main__":
    main()
