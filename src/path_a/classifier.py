"""
Ticket A4: Classifier
Trains XGBoost on A3 sampled data, evaluates via GroupKFold by S1 entity,
and writes scores for the validation set in the shared format.
No country feature is used.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import json

from eval.f05 import macro_f05
from src.common.data_loader import load_all_sources

def threshold_predictions(pairs_df, threshold):
    """Convert scores to a dictionary format for F0.5 evaluation."""
    preds = defaultdict(list)
    for _, row in pairs_df.iterrows():
        if row['score'] >= threshold:
            preds[row['s1_id']].append(row['cand_id'])
    return dict(preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default="output/a3_train_sampled.parquet", help="Path to sampled training features")
    parser.add_argument("--all-features", type=Path, default="output/a2_features.parquet", help="Path to full features")
    parser.add_argument("--val-split", type=Path, default="splits/val_s1_ids.txt", help="Path to validation S1 IDs")
    parser.add_argument("--out-val-scores", type=Path, default="scores/path_a_val.parquet", help="Path to save val scores")
    args = parser.parse_args()

    print("Loading datasets...")
    train_df = pd.read_parquet(args.train)
    full_df = pd.read_parquet(args.all_features)
    
    with open(args.val_split, "r") as f:
        val_ids = set(line.strip() for line in f if line.strip())

    # Ensure train_df does not contain validation entities
    train_df = train_df[~train_df['s1_id'].isin(val_ids)].reset_index(drop=True)
    val_df = full_df[full_df['s1_id'].isin(val_ids)].reset_index(drop=True)

    # Feature columns (exclude IDs and target)
    exclude_cols = {'s1_id', 'cand_id', 'is_match', 'country'} # country is explicitly forbidden
    feature_cols = sorted(list(set(train_df.columns) - exclude_cols))

    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols]
    y_train = train_df['is_match']
    groups_train = train_df['s1_id']

    X_val = val_df[feature_cols]
    
    print("\n--- Running GroupKFold CV on Train set ---")
    gkf = GroupKFold(n_splits=5)
    cv_aucs = []
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        X_t, y_t = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_v, y_v = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        preds = model.predict_proba(X_v)[:, 1]
        
        auc = roc_auc_score(y_v, preds)
        cv_aucs.append(auc)
        print(f" Fold {fold+1} AUC: {auc:.4f}")

    print(f"Mean CV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")

    print("\n--- Training Final Model on all train data ---")
    model.fit(X_train, y_train, verbose=False)

    print("\n--- Predicting on Validation Set ---")
    val_preds = model.predict_proba(X_val)[:, 1]
    val_df['score'] = val_preds

    # Evaluate F0.5 on validation set
    print("Loading Ground Truth for F0.5 Evaluation...")
    _, _, _, gt_df = load_all_sources()
    gt_dict = {}
    for _, row in gt_df.iterrows():
        if row['source1_entity_id'] in val_ids:
            gt_dict[row['source1_entity_id']] = row['matches']

    # Grid search for best threshold on val set (this normally goes in A5 but helps to report val macro F0.5 here)
    best_thresh = 0.5
    best_f05 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.1):
        pred_dict = threshold_predictions(val_df, thresh)
        f05 = macro_f05(pred_dict, gt_dict)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"Best Validation Macro F0.5: {best_f05:.4f} (at threshold {best_thresh:.2f})")

    # Export scores
    out_cols = ['s1_id', 'cand_id', 'score']
    out_df = val_df[out_cols]
    args.out_val_scores.parent.mkdir(exist_ok=True, parents=True)
    out_df.to_parquet(args.out_val_scores, index=False)
    print(f"Saved {len(out_df):,} scored validation pairs to {args.out_val_scores}")

if __name__ == "__main__":
    main()
