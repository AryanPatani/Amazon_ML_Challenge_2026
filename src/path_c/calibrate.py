"""
Ticket C4: LLM Score Calibration
Calibrates the raw LLM logprobs into true probabilities using Isotonic Regression.
Outputs the calibrated scores to a new parquet file for the final ensemble.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.common.data_loader import load_ground_truth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default="scores/path_c_val.parquet", help="Raw LLM scores")
    parser.add_argument("--gt", type=Path, default="data/dataset/train/train_ground_truth.tsv", help="Ground truth file")
    parser.add_argument("--out", type=Path, default="scores/path_c_calibrated.parquet", help="Output path")
    args = parser.parse_args()

    if not args.scores.exists():
        print(f"Error: {args.scores} not found. Please run C2 first.")
        return

    print("Loading LLM scores and Ground Truth...")
    df = pd.read_parquet(args.scores)
    
    # We only want to calibrate rows that actually have an LLM score
    # (Since we only ran the LLM on the uncertain band, many rows will be NaN for llm_score)
    llm_df = df[df['llm_score'].notnull()].copy()
    
    if len(llm_df) == 0:
        print("Error: No LLM scores found in the dataframe. Did C2 run successfully?")
        return

    # Load ground truth and create a fast lookup dictionary: s1_id -> set(matches)
    gt_df = load_ground_truth(args.gt)
    gt_dict = {row['source1_entity_id']: set(row['matches']) for _, row in gt_df.iterrows()}

    # Assign binary labels (1.0 for match, 0.0 for non-match)
    print("Mapping ground truth to candidates...")
    def check_match(row):
        matches = gt_dict.get(row['s1_id'], set())
        return 1.0 if row['cand_id'] in matches else 0.0

    llm_df['is_match'] = llm_df.apply(check_match, axis=1)

    # Calculate pre-calibration Brier Score (lower is better)
    pre_brier = brier_score_loss(llm_df['is_match'], llm_df['llm_score'])
    print(f"\nPre-Calibration Brier Score:  {pre_brier:.4f}")

    # Fit Isotonic Regression
    print("Fitting Isotonic Regression calibrator...")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    
    # Isotonic regression expects 1D arrays
    X = llm_df['llm_score'].values
    y = llm_df['is_match'].values
    
    llm_df['calibrated_llm_score'] = iso.fit_transform(X, y)

    # Calculate post-calibration Brier Score
    post_brier = brier_score_loss(llm_df['is_match'], llm_df['calibrated_llm_score'])
    print(f"Post-Calibration Brier Score: {post_brier:.4f}")
    print(f"Improvement: {(pre_brier - post_brier) / pre_brier * 100:.2f}%\n")

    # Merge calibrated scores back into the main dataframe
    # Rows that were not run through the LLM will just have NaN for calibrated_llm_score
    df = df.merge(
        llm_df[['s1_id', 'cand_id', 'calibrated_llm_score']], 
        on=['s1_id', 'cand_id'], 
        how='left'
    )
    
    args.out.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved calibrated scores to {args.out}")

if __name__ == "__main__":
    main()
