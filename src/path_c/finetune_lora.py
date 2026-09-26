"""
Ticket C3: LoRA Fine-Tuning
Fine-tunes the verifier LLM using PEFT/LoRA on the hard negatives/positives
sampled in Path A (or generated directly from ground truth).
Outputs adapter weights that can be loaded in C4.
"""

import sys
import os
import argparse
from pathlib import Path
import random
import pandas as pd
import numpy as np
import torch
from datasets import Dataset

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.data_loader import load_all_sources

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPT_TEMPLATE = """You are an expert entity resolution judge. Your task is to determine if two records refer to the exact same business at the exact same location.
Ignore minor phonetic spelling differences, abbreviations, or missing generic landmarks.
However, if the franchise is the same but the street number/location is explicitly different, it is NOT a match.

Record 1: {s1_name}, {s1_addr}
Record 2: {c_name}, {c_addr}
Are these the exact same business at the exact same location? Answer Yes or No: """


def generate_pairs_from_ground_truth(s1_df, cands_df, gt_df, max_samples=5000, neg_ratio=3):
    """Fallback generator: build positive and negative pairs directly from ground truth."""
    print("Generating training pairs directly from ground truth...")
    s1_ids = set(s1_df['entity_id'].unique())
    c_ids = list(cands_df['entity_id'].unique())

    pos_pairs = []
    gt_map = {}
    for row in gt_df.itertuples(index=False):
        s1_id = getattr(row, 'source1_entity_id')
        matches = getattr(row, 'matches', [])
        gt_map[s1_id] = set(matches)
        for m in matches:
            if s1_id in s1_ids:
                pos_pairs.append({'s1_id': s1_id, 'cand_id': m, 'is_match': 1})

    # Sample negatives
    rng = random.Random(42)
    neg_pairs = []
    target_negs = int(len(pos_pairs) * neg_ratio)
    all_s1 = list(s1_ids)

    while len(neg_pairs) < target_negs and all_s1:
        s1 = rng.choice(all_s1)
        known_matches = gt_map.get(s1, set())
        cand = rng.choice(c_ids)
        if cand not in known_matches:
            neg_pairs.append({'s1_id': s1, 'cand_id': cand, 'is_match': 0})

    combined = pos_pairs + neg_pairs
    rng.shuffle(combined)
    if max_samples and len(combined) > max_samples:
        combined = combined[:max_samples]
    return pd.DataFrame(combined)


def prepare_dataset(train_features_path, s1_df, cands_df, gt_df=None, tokenizer=None, max_samples=5000):
    """Load training pairs and format them into HuggingFace Dataset for Causal LM training."""
    train_df = None
    path = Path(train_features_path) if train_features_path else None

    # Check if a file or directory was provided
    if path and path.is_file():
        print(f"Loading training pairs from file: {path}...")
        if path.suffix in (".parquet", ".pq"):
            train_df = pd.read_parquet(path)
        elif path.suffix in (".csv", ".tsv"):
            sep = "\t" if path.suffix == ".tsv" else ","
            train_df = pd.read_csv(path, sep=sep)
    elif path and path.is_dir():
        print(f"Provided path '{path}' is a directory. Generating pairs from ground truth...")
        train_df = generate_pairs_from_ground_truth(s1_df, cands_df, gt_df, max_samples=max_samples)

    if train_df is None or len(train_df) == 0:
        if gt_df is not None:
            print("No valid input file found. Automatically constructing dataset from ground truth...")
            train_df = generate_pairs_from_ground_truth(s1_df, cands_df, gt_df, max_samples=max_samples)
        else:
            raise FileNotFoundError(f"Could not load or generate training data from {train_features_path}")

    # Normalize column names
    col_rename = {
        'source1_entity_id': 's1_id',
        'entity_id_1': 's1_id',
        'entity_id_2': 'cand_id',
        'candidate_id': 'cand_id',
        'matched_entity_id': 'cand_id'
    }
    train_df = train_df.rename(columns=col_rename)

    # Derive is_match if missing
    if 'is_match' not in train_df.columns and gt_df is not None:
        print("Deriving 'is_match' from ground truth...")
        gt_pairs = set()
        for row in gt_df.itertuples(index=False):
            s1_id = getattr(row, 'source1_entity_id')
            for m in getattr(row, 'matches', []):
                gt_pairs.add((s1_id, m))
        train_df['is_match'] = [1 if (s1, c) in gt_pairs else 0 for s1, c in zip(train_df['s1_id'], train_df['cand_id'])]

    if max_samples and len(train_df) > max_samples:
        print(f"Subsampling {len(train_df):,} pairs to max_samples={max_samples:,}...")
        train_df = train_df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    # Fast entity lookup dicts
    s1_dict = s1_df.set_index('entity_id')[['business_name', 'business_address']].to_dict('index')
    cands_dict = cands_df.set_index('entity_id')[['business_name', 'business_address']].to_dict('index')

    eos = tokenizer.eos_token if tokenizer and hasattr(tokenizer, "eos_token") and tokenizer.eos_token else "\n"
    texts = []

    for row in train_df.itertuples(index=False):
        s1_id = getattr(row, 's1_id')
        cand_id = getattr(row, 'cand_id')
        is_m = getattr(row, 'is_match', 0)

        s1_info = s1_dict.get(s1_id, {})
        c_info = cands_dict.get(cand_id, {})

        s1_name = str(s1_info.get('business_name') or '')
        s1_addr = str(s1_info.get('business_address') or '')
        c_name = str(c_info.get('business_name') or '')
        c_addr = str(c_info.get('business_address') or '')

        prompt = PROMPT_TEMPLATE.format(
            s1_name=s1_name,
            s1_addr=s1_addr,
            c_name=c_name,
            c_addr=c_addr
        )

        label = "Yes" if is_m in (1, True, "1", "True", "true") else "No"
        full_text = prompt + label + eos
        texts.append(full_text)

    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser(description="Ticket C3: LoRA Fine-Tuning")
    parser.add_argument("--train-data", type=Path, default=Path("output/a3_train_sampled.parquet"), help="Path to sampled training pairs")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model name or path")
    parser.add_argument("--output-dir", type=str, default="models/path_c_lora", help="Output directory for adapters")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional raw dataset path (train/ directory or parent)")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max pairs to use for training")
    parser.add_argument("--batch-size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=500, help="Max training steps")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    print("Loading Raw data for dictionary lookups...")
    s1, s2, s3, gt = load_all_sources(train_dir=args.data_dir)
    cands_df = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=['entity_id'])

    print(f"Loading tokenizer for {args.model}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset(
        args.train_data,
        s1,
        cands_df,
        gt_df=gt,
        tokenizer=tokenizer,
        max_samples=args.max_samples
    )
    print(f"Prepared {len(dataset)} training examples.")

    device_available = torch.cuda.is_available()
    print(f"CUDA available: {device_available}")

    if device_available:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("Notice: Running on CPU (quantization disabled).")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )

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
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=100,
        optim="paged_adamw_8bit" if device_available else "adamw_torch",
        fp16=device_available,
        report_to="none"
    )

    # Note: peft_config is passed as None to SFTTrainer since model is already a PeftModel
    sft_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": None,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_length,
        "args": training_args,
    }

    try:
        trainer = SFTTrainer(**sft_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**sft_kwargs, tokenizer=tokenizer)

    print("Starting LoRA Fine-Tuning...")
    trainer.train()

    print(f"Saving LoRA adapters to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
