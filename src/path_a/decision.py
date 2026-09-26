"""
Ticket A5: Decision Layer
Takes raw pair scores (e.g., from A4) and applies:
1. Global threshold
2. One-to-one assignment (a candidate can only belong to its highest-scoring S1)
3. Singleton rule (relative gap between top-1 and top-2, minimum absolute score)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

from eval.f05 import macro_f05
from src.common.data_loader import load_all_sources

def apply_global_threshold(scores_df, threshold):
    preds = defaultdict(list)
    filtered = scores_df[scores_df['score'] >= threshold]
    for _, row in filtered.iterrows():
        preds[row['s1_id']].append(row['cand_id'])
    return dict(preds)

def apply_one_to_one_assignment(scores_df, threshold):
    filtered = scores_df[scores_df['score'] >= threshold].copy()
    # Sort by score descending so we keep the best match for each candidate
    filtered = filtered.sort_values(by='score', ascending=False)
    # A candidate should only map to exactly ONE S1 entity (its highest scoring one)
    filtered = filtered.drop_duplicates(subset=['cand_id'], keep='first')
    
    preds = defaultdict(list)
    for _, row in filtered.iterrows():
        preds[row['s1_id']].append(row['cand_id'])
    return dict(preds)

def apply_singleton_rule(scores_df, threshold, min_absolute_score, min_margin):
    # Base: one-to-one assignment
    filtered = scores_df[scores_df['score'] >= threshold].copy()
    filtered = filtered.sort_values(by='score', ascending=False)
    filtered = filtered.drop_duplicates(subset=['cand_id'], keep='first')
    
    preds = defaultdict(list)
    
    # Evaluate S1 entities
    # We want to identify singletons. If an S1's best match score < min_absolute_score,
    # OR if the margin between its best match and second best match is < min_margin (confusion),
    # we declare it a singleton (return empty list).
    
    s1_groups = filtered.groupby('s1_id')
    for s1_id, group in s1_groups:
        top_scores = group['score'].values
        
        # If the best score isn't strong enough
        if len(top_scores) > 0 and top_scores[0] < min_absolute_score:
            continue
            
        # If there's a second candidate and the margin is too small
        if len(top_scores) > 1:
            margin = top_scores[0] - top_scores[1]
            if margin < min_margin:
                continue
                
        # Otherwise, keep all valid candidates above threshold (which are already one-to-one filtered)
        preds[s1_id].extend(group['cand_id'].tolist())
        
    return dict(preds)
    
def fill_missing_s1(preds, all_val_s1_ids):
    # Ensure every S1 in the val set is in the dictionary (even as an empty list)
    out = {s1: [] for s1 in all_val_s1_ids}
    out.update(preds)
    return out

def main():
    parser = argparse.ArgumentParser(description="A5 Decision Layer")
    parser.add_argument("--scores", type=Path, default="scores/path_a_val.parquet", help="Path to val scores from A4")
    parser.add_argument("--val-split", type=Path, default="splits/val_s1_ids.txt", help="Path to validation S1 IDs")
    args = parser.parse_args()
    
    if not args.scores.exists():
        print(f"Error: {args.scores} not found. Run A4 first.")
        return

    print(f"Loading scores from {args.scores}...")
    scores_df = pd.read_parquet(args.scores)
    
    with open(args.val_split, "r") as f:
        val_ids = set(line.strip() for line in f if line.strip())
        
    print("Loading Ground Truth...")
    _, _, _, gt_df = load_all_sources()
    gt_dict = {}
    for _, row in gt_df.iterrows():
        if row['source1_entity_id'] in val_ids:
            gt_dict[row['source1_entity_id']] = row['matches']
            
    # Tuning global threshold
    print("\n1. Tuning Base Global Threshold...")
    best_t = 0.5
    best_f05_base = 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        preds = apply_global_threshold(scores_df, t)
        preds = fill_missing_s1(preds, val_ids)
        f05 = macro_f05(preds, gt_dict)
        if f05 > best_f05_base:
            best_f05_base = f05
            best_t = t
            
    print(f"  Best Base F0.5: {best_f05_base:.5f} at threshold {best_t:.2f}")
    
    # Tuning one-to-one assignment
    print("\n2. Evaluating One-to-One Assignment...")
    best_t_1to1 = best_t
    best_f05_1to1 = 0.0
    for t in np.arange(best_t - 0.2, best_t + 0.2, 0.05):
        if t < 0: continue
        preds = apply_one_to_one_assignment(scores_df, t)
        preds = fill_missing_s1(preds, val_ids)
        f05 = macro_f05(preds, gt_dict)
        if f05 > best_f05_1to1:
            best_f05_1to1 = f05
            best_t_1to1 = t
            
    gain_1to1 = best_f05_1to1 - best_f05_base
    print(f"  Best 1-to-1 F0.5: {best_f05_1to1:.5f} at threshold {best_t_1to1:.2f} (Gain: +{gain_1to1:.5f})")

    # Tuning Singleton Rules
    print("\n3. Evaluating Singleton Rules (Absolute Score & Margin)...")
    best_f05_final = best_f05_1to1
    best_abs = 0.0
    best_margin = 0.0
    
    # Grid search for singleton params
    for min_abs in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for min_margin in [0.0, 0.05, 0.1, 0.2]:
            preds = apply_singleton_rule(scores_df, best_t_1to1, min_abs, min_margin)
            preds = fill_missing_s1(preds, val_ids)
            f05 = macro_f05(preds, gt_dict)
            
            if f05 > best_f05_final:
                best_f05_final = f05
                best_abs = min_abs
                best_margin = min_margin

    gain_singleton = best_f05_final - best_f05_1to1
    print(f"  Best Final F0.5: {best_f05_final:.5f} (Gain over 1-to-1: +{gain_singleton:.5f})")
    print(f"  Optimal parameters: threshold={best_t_1to1:.2f}, min_abs={best_abs:.2f}, min_margin={best_margin:.2f}")

    print("\n--- Summary ---")
    print(f"Base Threshold F0.5:      {best_f05_base:.5f}")
    print(f"With 1-to-1 Assignment:   {best_f05_1to1:.5f}  (+{gain_1to1:.5f})")
    print(f"With Singleton Logic:     {best_f05_final:.5f}  (+{gain_singleton:.5f})")
    print(f"Total Gain:               +{best_f05_final - best_f05_base:.5f}")

if __name__ == "__main__":
    main()
