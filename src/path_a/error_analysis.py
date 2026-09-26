"""
Ticket A6: Error Analysis and LOCO Evaluation
Simulates Leave-One-Country-Out (LOCO) evaluation to test generalization
for unseen countries (e.g., France).
Extracts Top 50 False Positives and False Negatives for Path C (LLM).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from eval.f05 import make_leave_one_country_out_splits, macro_f05
from src.common.data_loader import load_all_sources
from src.path_a.decision import apply_one_to_one_assignment

def analyze_errors(preds, gt_dict, pairs_df, s1_df, cands_df, top_k=50):
    # Map raw strings for reporting
    s1_map = s1_df.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
    cands_map = cands_df.set_index('entity_id')[['business_name', 'business_address', 'country']].to_dict('index')
    
    fp_list = []
    fn_list = []
    
    # scores lookup for sorting
    scores_dict = {}
    for _, row in pairs_df.iterrows():
        scores_dict[(row['s1_id'], row['cand_id'])] = row['score']
        
    for s1_id, true_matches in gt_dict.items():
        pred_matches = set(preds.get(s1_id, []))
        true_matches = set(true_matches)
        
        # False Positives (predicted but not true)
        for p in pred_matches - true_matches:
            fp_list.append({
                's1_id': s1_id, 'cand_id': p,
                'score': scores_dict.get((s1_id, p), 0.0)
            })
            
        # False Negatives (true but not predicted)
        for t in true_matches - pred_matches:
            fn_list.append({
                's1_id': s1_id, 'cand_id': t,
                'score': scores_dict.get((s1_id, t), 0.0)
            })
            
    # Sort FPs by highest confidence score (the model was very sure but wrong)
    fp_list = sorted(fp_list, key=lambda x: x['score'], reverse=True)[:top_k]
    # Sort FNs by lowest score (the model missed it completely)
    fn_list = sorted(fn_list, key=lambda x: x['score'])[:top_k]
    
    print(f"\n--- TOP {top_k} FALSE POSITIVES (High score, wrong match) ---")
    for fp in fp_list[:10]: # Print top 10 to terminal
        s1 = s1_map.get(fp['s1_id'], {})
        cand = cands_map.get(fp['cand_id'], {})
        print(f"Score: {fp['score']:.3f} | S1: {s1.get('business_name')} ({s1.get('country')}) -> Cand: {cand.get('business_name')} ({cand.get('country')})")
        print(f"   S1 Addr: {s1.get('business_address')}")
        print(f"   Cand Addr: {cand.get('business_address')}")
        print()
        
    print(f"\n--- TOP {top_k} FALSE NEGATIVES (Low score, missed match) ---")
    for fn in fn_list[:10]:
        s1 = s1_map.get(fn['s1_id'], {})
        cand = cands_map.get(fn['cand_id'], {})
        print(f"Score: {fn['score']:.3f} | S1: {s1.get('business_name')} ({s1.get('country')}) -> Cand: {cand.get('business_name')} ({cand.get('country')})")
        print(f"   S1 Addr: {s1.get('business_address')}")
        print(f"   Cand Addr: {cand.get('business_address')}")
        print()
        
    return fp_list, fn_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default="scores/path_a_val.parquet")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    if not args.scores.exists():
        print(f"Error: {args.scores} not found.")
        return

    print("Loading data...")
    scores_df = pd.read_parquet(args.scores)
    s1, s2, s3, gt_df = load_all_sources()
    cands_df = pd.concat([s2, s3], ignore_index=True)
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        gt_dict[row['source1_entity_id']] = row['matches']

    print("Creating LOCO splits...")
    loco_splits = make_leave_one_country_out_splits(s1)
    
    # LOCO Evaluation
    print("\n--- Leave-One-Country-Out Evaluation ---")
    baseline_preds = apply_one_to_one_assignment(scores_df, args.threshold)
    
    for held_out_country, (train_ids, test_ids) in loco_splits.items():
        # Subset predictions and GT for the held out country
        test_preds = {k: v for k, v in baseline_preds.items() if k in test_ids}
        test_gt = {k: v for k, v in gt_dict.items() if k in test_ids}
        
        # Fill missing singletons
        for tid in test_ids:
            if tid not in test_preds: test_preds[tid] = []
            
        f05 = macro_f05(test_preds, test_gt)
        print(f"Held out country: {held_out_country.upper()} | F0.5: {f05:.4f}")

    # Global Error Analysis
    print("\nRunning Error Analysis on validation set...")
    fp, fn = analyze_errors(baseline_preds, gt_dict, scores_df, s1, cands_df, top_k=50)

if __name__ == "__main__":
    main()
