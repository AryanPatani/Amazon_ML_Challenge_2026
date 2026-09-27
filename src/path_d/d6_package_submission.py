"""
Ticket D6: Package Submission
=========================================================

Reads the candidate pairs from D1 and the final resolved matches from D4,
and formats them into the official submission TSV format.
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def save_tsv(df, out_path, col_name='matches'):
    # Group by s1_id and aggregate cand_ids into comma-separated string
    print(f"  Aggregating {len(df):,} rows...")
    grouped = df.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(map(str, x))).reset_index()
    grouped.rename(columns={'s1_id': 'source1_entity_id', 'cand_id': col_name}, inplace=True)
    
    out_path.parent.mkdir(exist_ok=True, parents=True)
    grouped.to_csv(out_path, sep='\t', index=False)
    print(f"  Saved {len(grouped):,} S1 entities to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1-scores", type=Path, default="scores/path_d1_val.parquet")
    parser.add_argument("--d4-final", type=Path, default="scores/path_d4_final.parquet")
    parser.add_argument("--out-cand", type=Path, default="output/candidate_pairs.tsv")
    parser.add_argument("--out-match", type=Path, default="output/matching_results.tsv")
    args = parser.parse_args()

    print("=" * 60)
    print("D6: PACKAGE SUBMISSION")
    print("=" * 60)

    # 1. Candidate Pairs
    print(f"\n[Candidates] Loading D1 candidate pairs from {args.d1_scores}...")
    if args.d1_scores.exists():
        d1_df = pd.read_parquet(args.d1_scores)
        # Keep only pairs that passed the D1 threshold (used for D2 graph)
        d1_cands = d1_df[d1_df['score'] >= 0.10][['s1_id', 'cand_id']]
        print("  Generating candidate_pairs.tsv...")
        save_tsv(d1_cands, args.out_cand, col_name='matched_entity_ids')
    else:
        print(f"  Warning: {args.d1_scores} not found. Skipping candidates.")

    # 2. Matching Results
    print(f"\n[Matches] Loading D4 final predictions from {args.d4_final}...")
    if args.d4_final.exists():
        d4_df = pd.read_parquet(args.d4_final)
        print("  Generating matching_results.tsv...")
        save_tsv(d4_df, args.out_match, col_name='matches')
    else:
        print(f"  Warning: {args.d4_final} not found. Skipping matches.")
        
    print("\n[Done] Run 'python utils/validate_submission.py' to verify format.")
    print("=" * 60)

if __name__ == "__main__":
    main()
