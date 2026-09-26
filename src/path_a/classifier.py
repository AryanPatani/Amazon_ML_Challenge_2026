"""
Ticket A4: Classifier
Trains XGBoost (or LightGBM / HistGradientBoosting) on A3 sampled data,
evaluates via GroupKFold by S1 entity, and writes scores for the validation
set in the shared format.
No country feature is used.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.f05 import macro_f05
from src.common.data_loader import load_all_sources

# Optional tree boosting backends
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def threshold_predictions(pairs_df, threshold):
    """Convert scores to a dictionary format for F0.5 evaluation."""
    preds = defaultdict(list)
    for row in pairs_df.itertuples(index=False):
        s1 = getattr(row, 's1_id')
        cand = getattr(row, 'cand_id')
        score = getattr(row, 'score')
        if score >= threshold:
            preds[s1].append(cand)
    return dict(preds)


def create_classifier(model_type="auto"):
    """Instantiate classifier based on requested type and environment availability."""
    if model_type == "xgboost" or (model_type == "auto" and HAS_XGB):
        if not HAS_XGB:
            raise ImportError("XGBoost is requested but not installed.")
        return "xgb", xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='auc',
            n_jobs=-1
        )
    elif model_type == "lightgbm" or (model_type == "auto" and HAS_LGB):
        if not HAS_LGB:
            raise ImportError("LightGBM is requested but not installed.")
        return "lgb", lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        if model_type != "auto" and model_type != "histgb":
            print(f"Requested model '{model_type}' not available, falling back to HistGradientBoostingClassifier.")
        return "histgb", HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )


def main():
    parser = argparse.ArgumentParser(description="Ticket A4: Classifier")
    parser.add_argument("--train", type=Path, default="output/a3_train_sampled.parquet", help="Path to sampled training features")
    parser.add_argument("--all-features", type=Path, default="output/a2_features.parquet", help="Path to full features")
    parser.add_argument("--val-split", type=Path, default="splits/val_s1_ids.txt", help="Path to validation S1 IDs")
    parser.add_argument("--out-val-scores", type=Path, default="scores/path_a_val.parquet", help="Path to save val scores")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional raw dataset path (train/ directory or parent)")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "xgboost", "lightgbm", "histgb"], help="Model backend")
    args = parser.parse_args()

    if not args.train.exists():
        print(f"Error: Train file {args.train} not found. Please run A3 sampling first.")
        return

    print("Loading datasets...")
    train_df = pd.read_parquet(args.train)

    if args.all_features.exists():
        full_df = pd.read_parquet(args.all_features)
    else:
        print(f"Notice: {args.all_features} not found. Using train set for candidate lookups.")
        full_df = train_df.copy()

    # Load or generate validation IDs
    val_ids = set()
    if args.val_split.exists():
        with open(args.val_split, "r") as f:
            val_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(val_ids):,} validation S1 IDs from {args.val_split}")
    else:
        print(f"Warning: {args.val_split} not found. Generating deterministic 20% holdout...")
        unique_s1 = sorted(train_df['s1_id'].unique())
        rng = np.random.default_rng(42)
        val_count = max(1, int(len(unique_s1) * 0.2))
        val_ids = set(rng.choice(unique_s1, size=val_count, replace=False))
        args.val_split.parent.mkdir(exist_ok=True, parents=True)
        with open(args.val_split, "w") as f:
            for s1_id in sorted(val_ids):
                f.write(f"{s1_id}\n")
        print(f"Saved generated validation split to {args.val_split}")

    # Ensure train_df does not contain validation entities
    train_df = train_df[~train_df['s1_id'].isin(val_ids)].reset_index(drop=True)
    val_df = full_df[full_df['s1_id'].isin(val_ids)].reset_index(drop=True)

    if len(val_df) == 0:
        print("Notice: No validation pairs found in full_df with val_ids. Splitting 20% of train pairs for validation.")
        s1_pool = train_df['s1_id'].unique()
        v_s1 = set(s1_pool[:max(1, int(len(s1_pool) * 0.2))])
        val_df = train_df[train_df['s1_id'].isin(v_s1)].reset_index(drop=True)
        train_df = train_df[~train_df['s1_id'].isin(v_s1)].reset_index(drop=True)

    # Feature columns (exclude IDs and target)
    exclude_cols = {'s1_id', 'cand_id', 'is_match', 'country'}  # country is explicitly forbidden
    feature_cols = sorted(list((set(train_df.columns) & set(val_df.columns)) - exclude_cols))

    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df['is_match'].astype(int)
    groups_train = train_df['s1_id']

    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    backend_name, _ = create_classifier(args.model)
    print(f"\n--- Running GroupKFold CV on Train set (Backend: {backend_name}) ---")
    gkf = GroupKFold(n_splits=min(5, len(groups_train.unique())))
    cv_aucs = []

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        X_t, y_t = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_v, y_v = X_train.iloc[val_idx], y_train.iloc[val_idx]

        if len(np.unique(y_t)) < 2:
            continue

        _, fold_model = create_classifier(args.model)
        if backend_name == "xgb":
            fold_model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        else:
            fold_model.fit(X_t, y_t)

        preds = fold_model.predict_proba(X_v)[:, 1]

        if len(np.unique(y_v)) > 1:
            auc = roc_auc_score(y_v, preds)
            cv_aucs.append(auc)
            print(f" Fold {fold+1} AUC: {auc:.4f}")
        else:
            print(f" Fold {fold+1} skipped AUC (single class in fold)")

    if cv_aucs:
        print(f"Mean CV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")
    else:
        print("CV completed without multiple classes to report AUC.")

    print("\n--- Training Final Model on all train data ---")
    _, final_model = create_classifier(args.model)
    final_model.fit(X_train, y_train)

    print("\n--- Predicting on Validation Set ---")
    val_preds = final_model.predict_proba(X_val)[:, 1]
    val_df['score'] = val_preds

    # Evaluate F0.5 on validation set if ground truth is available
    print("Loading Ground Truth for F0.5 Evaluation...")
    try:
        _, _, _, gt_df = load_all_sources(train_dir=args.data_dir)
        gt_dict = {}
        for row in gt_df.itertuples(index=False):
            s1_id = getattr(row, 'source1_entity_id')
            if s1_id in val_ids:
                gt_dict[s1_id] = getattr(row, 'matches', [])

        if gt_dict and len(val_df) > 0:
            best_thresh = 0.5
            best_f05 = 0.0
            for thresh in np.arange(0.1, 0.9, 0.05):
                pred_dict = threshold_predictions(val_df, thresh)
                f05 = macro_f05(pred_dict, gt_dict)
                if f05 > best_f05:
                    best_f05 = f05
                    best_thresh = thresh
            print(f"Best Validation Macro F0.5: {best_f05:.4f} (at threshold {best_thresh:.2f})")
        else:
            print("Validation ground truth empty or no matches in validation set; skipping F0.5 grid search.")
    except Exception as e:
        print(f"Notice: Ground truth evaluation skipped ({e})")

    # Export scores
    out_cols = ['s1_id', 'cand_id', 'score']
    out_df = val_df[out_cols]
    args.out_val_scores.parent.mkdir(exist_ok=True, parents=True)
    out_df.to_parquet(args.out_val_scores, index=False)
    print(f"Saved {len(out_df):,} scored validation pairs to {args.out_val_scores}")


if __name__ == "__main__":
    main()
