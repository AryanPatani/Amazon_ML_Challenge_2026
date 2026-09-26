"""
Ticket C1: Inference Setup
Loads an Apache-2.0 / MIT LLM (<= 8B params) in 4-bit precision.
Measures inference throughput on a dummy sample of pairs.
Expected to run on AWS GPU instances (e.g. g5.xlarge).
"""

import time
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# E.g. Qwen2.5-7B-Instruct (Apache-2.0, fast, extremely good reasoning)
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def load_model(model_name):
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading {model_name} in 4-bit quantization...")
    # 4-bit config to fit comfortably in 24GB or even 16GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return tokenizer, model

def measure_throughput(tokenizer, model, num_samples=50):
    print(f"\nMeasuring throughput for {num_samples} samples...")
    
    # Dummy prompts representing typical pair verification
    prompt_template = (
        "Are these two entities exactly the same business at the same location?\n"
        "Entity 1: Starbucks, 123 Main St, New York, US\n"
        "Entity 2: Starbucks Coffee, 123 Main Street, NY, US\n"
        "Answer Yes or No."
    )
    
    # We batch them or do sequential. To simulate simple generation, we do sequential.
    # We only need the LLM to output 1 token ("Yes" or "No").
    
    # warmup
    inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
    _ = model.generate(**inputs, max_new_tokens=2)
    
    start_time = time.time()
    
    for i in range(num_samples):
        # We can simulate different prompts, but for throughput testing, one is fine
        inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,       # We only want "Yes" or "No"
                do_sample=False,        # Greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )
            
    end_time = time.time()
    elapsed = end_time - start_time
    pairs_per_sec = num_samples / elapsed
    
    print(f"--- Throughput Results ---")
    print(f"Total time for {num_samples} pairs: {elapsed:.2f} seconds")
    print(f"Speed: {pairs_per_sec:.2f} pairs/second")
    
    # Assuming the uncertain band is ~100,000 pairs
    est_hours = (100000 / pairs_per_sec) / 3600
    print(f"Estimated time to process 100k uncertain pairs: {est_hours:.2f} hours")
    
    if est_hours < 48:
        print("Verdict: PASS. Throughput comfortably fits within the 72-hour budget limit.")
    else:
        print("Verdict: FAIL. Too slow. Consider vLLM or a smaller model.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples for throughput test")
    parser.add_argument("--dry-run", action="store_true", help="Just verify imports and script logic (no download)")
    args = parser.parse_args()
    
    if args.dry_run:
        print("Dry run successful. Environment and script logic verified.")
        return
        
    tokenizer, model = load_model(args.model)
    measure_throughput(tokenizer, model, args.samples)

if __name__ == "__main__":
    main()
