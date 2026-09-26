"""
Ticket C2: LLM Verifier (Zero-Shot / Few-Shot Logprob Extraction)
Evaluates uncertain pairs using an LLM, extracting the logprob of 'Yes'
to produce a continuous confidence score for the ensemble.
Designed to be run on Kaggle/AWS GPUs.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.common.data_loader import load_all_sources
from eval.f05 import macro_f05

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

FEW_SHOT_PROMPT = """You are an expert entity resolution judge. Your task is to determine if two records refer to the exact same business at the exact same location.
Ignore minor phonetic spelling differences, abbreviations, or missing generic landmarks.
However, if the franchise is the same but the street number/location is explicitly different, it is NOT a match.

Example 1:
Record 1: Starbucks, 123 Main St
Record 2: Starbucks Coffee, 123 Main Street
Are these the exact same business at the exact same location? Answer Yes or No: Yes

Example 2:
Record 1: Starbucks, 123 Main St
Record 2: Starbucks, 456 Oak St
Are these the exact same business at the exact same location? Answer Yes or No: No

Example 3:
Record 1: Maa Durga Stores, Near Highway
Record 2: Ma Darga Store, Near Highway
Are these the exact same business at the exact same location? Answer Yes or No: Yes

Example 4:
Record 1: M.S.E.B, Mumbai
Record 2: Maharashtra State Electricity Board, Mumbai
Are these the exact same business at the exact same location? Answer Yes or No: Yes

Now verify the following pair:
Record 1: {s1_name}, {s1_addr}
Record 2: {c_name}, {c_addr}
Are these the exact same business at the exact same location? Answer Yes or No: """

def load_llm(model_name):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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

def get_yes_logprob(model, tokenizer, prompt, yes_token_id, no_token_id):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
        
    # Get logits for the very last token generated (the one predicting the answer)
    next_token_logits = outputs.logits[0, -1, :]
    
    # We only care about the relative probability of "Yes" vs "No"
    yes_logit = next_token_logits[yes_token_id].item()
    no_logit = next_token_logits[no_token_id].item()
    
    # Softmax over just these two tokens to get a clean probability [0, 1]
    # p(yes) = exp(yes) / (exp(yes) + exp(no))
    max_logit = max(yes_logit, no_logit)
    exp_yes = math.exp(yes_logit - max_logit)
    exp_no = math.exp(no_logit - max_logit)
    
    prob_yes = exp_yes / (exp_yes + exp_no)
    return prob_yes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default="scores/path_a_val.parquet", help="Base scores from Path A")
    parser.add_argument("--out", type=Path, default="scores/path_c_val.parquet", help="Path to save LLM scores")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--lower-bound", type=float, default=0.3, help="Uncertain band lower bound")
    parser.add_argument("--upper-bound", type=float, default=0.7, help="Uncertain band upper bound")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading the actual LLM (mock scores)")
    args = parser.parse_args()

    if not args.scores.exists():
        print(f"Error: {args.scores} not found.")
        return

    print("Loading pairs and raw data...")
    scores_df = pd.read_parquet(args.scores)
    s1, s2, s3, _ = load_all_sources()
    cands_df = pd.concat([s2, s3], ignore_index=True)
    
    # Dictionaries for fast lookup
    s1_dict = s1.set_index('entity_id')[['business_name', 'business_address']].to_dict('index')
    cands_dict = cands_df.set_index('entity_id')[['business_name', 'business_address']].to_dict('index')

    # Filter to uncertain band
    uncertain_mask = (scores_df['score'] >= args.lower_bound) & (scores_df['score'] <= args.upper_bound)
    hard_pairs = scores_df[uncertain_mask].copy()
    print(f"Identified {len(hard_pairs):,} uncertain pairs out of {len(scores_df):,} total pairs.")
    
    if args.dry_run:
        print("Dry run enabled. Mocking LLM scores...")
        hard_pairs['llm_score'] = np.random.uniform(0.0, 1.0, size=len(hard_pairs))
    else:
        tokenizer, model = load_llm(args.model)
        
        # Token IDs for "Yes" and "No". Note: LLM tokenization is tricky with spaces.
        # "Yes" and " Yes" might be different tokens. We find the exact token ID.
        yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[-1]
        no_token_id = tokenizer.encode("No", add_special_tokens=False)[-1]
        
        print("Running LLM Inference on hard pairs...")
        llm_scores = []
        
        # In a real Kaggle notebook, we'd batch this. For now, sequential for simplicity.
        for _, row in tqdm(hard_pairs.iterrows(), total=len(hard_pairs)):
            s1_info = s1_dict.get(row['s1_id'], {})
            c_info = cands_dict.get(row['cand_id'], {})
            
            prompt = FEW_SHOT_PROMPT.format(
                s1_name=s1_info.get('business_name', ''),
                s1_addr=s1_info.get('business_address', ''),
                c_name=c_info.get('business_name', ''),
                c_addr=c_info.get('business_address', '')
            )
            
            prob = get_yes_logprob(model, tokenizer, prompt, yes_token_id, no_token_id)
            llm_scores.append(prob)
            
        hard_pairs['llm_score'] = llm_scores

    # Merge LLM scores back into the main dataframe
    scores_df = scores_df.merge(
        hard_pairs[['s1_id', 'cand_id', 'llm_score']], 
        on=['s1_id', 'cand_id'], 
        how='left'
    )
    
    args.out.parent.mkdir(exist_ok=True, parents=True)
    scores_df.to_parquet(args.out, index=False)
    print(f"\nSaved combined scores (Path A + Path C) to {args.out}")

if __name__ == "__main__":
    main()
