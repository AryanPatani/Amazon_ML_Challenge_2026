"""
Ticket C3: LoRA Fine-Tuning
Fine-tunes the verifier LLM using PEFT/LoRA on the hard negatives/positives
sampled in Path A.
Outputs adapter weights that can be loaded in C4.
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from src.common.data_loader import load_all_sources

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPT_TEMPLATE = """You are an expert entity resolution judge. Your task is to determine if two records refer to the exact same business at the exact same location.
Ignore minor phonetic spelling differences, abbreviations, or missing generic landmarks.
However, if the franchise is the same but the street number/location is explicitly different, it is NOT a match.

Record 1: {s1_name}, {s1_addr}
Record 2: {c_name}, {c_addr}
Are these the exact same business at the exact same location? Answer Yes or No: """

def prepare_dataset(train_features_path, s1_df, cands_df):
    print(f"Loading training pairs from {train_features_path}...")
    train_df = pd.read_parquet(train_features_path)
    
    s1_dict = s1_df.set_index('entity_id')[['business_name', 'business_address']].to_dict('index')
    cands_dict = cands_df.set_index('entity_id')[['business_name', 'business_address']].to_dict('index')
    
    texts = []
    for _, row in train_df.iterrows():
        s1_info = s1_dict.get(row['s1_id'], {})
        c_info = cands_dict.get(row['cand_id'], {})
        
        prompt = PROMPT_TEMPLATE.format(
            s1_name=s1_info.get('business_name', ''),
            s1_addr=s1_info.get('business_address', ''),
            c_name=c_info.get('business_name', ''),
            c_addr=c_info.get('business_address', '')
        )
        
        # The target label
        label = "Yes" if row['is_match'] == 1 else "No"
        
        # Combine prompt and label for causal language modeling
        full_text = prompt + label
        texts.append(full_text)
        
    return Dataset.from_dict({"text": texts})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, default="output/a3_train_sampled.parquet")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=str, default="models/path_c_lora")
    args = parser.parse_args()

    if not args.train_data.exists():
        print(f"Error: {args.train_data} not found. Please run A3 first to generate the sampled training set.")
        return

    print("Loading Raw data for dictionary lookups...")
    s1, s2, s3, _ = load_all_sources()
    cands_df = pd.concat([s2, s3], ignore_index=True)

    dataset = prepare_dataset(args.train_data, s1, cands_df)
    print(f"Prepared {len(dataset)} training examples.")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500, # Cap at 500 steps for fast tuning
        save_strategy="steps",
        save_steps=100,
        optim="paged_adamw_8bit",
        fp16=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )

    print("Starting LoRA Fine-Tuning...")
    trainer.train()
    
    print(f"Saving LoRA adapters to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
