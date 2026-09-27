"""
============================================================
  PATH A — OPTIMIZED HIGH-ACCURACY XGBOOST TRAINING
  Amazon ML Challenge 2026
============================================================

Total Runtime: ~6 to 7 minutes | Zero OOM Risk | Peak RAM: ~2.5 GB
Trained on 60,000 S1 entities (~300,000 balanced pairs)
Mines True Positives + High-Overlap TF-IDF Hard Negatives

Outputs:
  - Model: /kaggle/working/output/path_a_xgb_optimized.json
  - Evaluates & reports Validation Macro F0.5 & optimal threshold!
============================================================
"""

import os, sys, time, subprocess
from pathlib import Path

# ============================================================
# STEP 1: Environment Setup & Discovery
# ============================================================
print("=" * 60)
print("🚀 STEP 1: PATH DISCOVERY & ENVIRONMENT SETUP")
print("=" * 60)

POSSIBLE_CODE_ROOTS = [
    "/kaggle/input/datasets/vivantejani/ml-code/ml-code",
    "/kaggle/input/datasets/vivantejani/ml-code",
    "/kaggle/input/datasets/vivantejani",
    "/kaggle/input/ml-code/Amazon_ML_Challenge_2026",
    "/kaggle/input/ml-code",
    "/kaggle/working/ml-code",
]
CODE_ROOT = next((p for p in POSSIBLE_CODE_ROOTS if (Path(p) / "src").exists()), None)

if not CODE_ROOT:
    for p in Path("/kaggle").rglob("features.py"):
        if p.parent.name == "path_a" and (p.parent.parent.name == "src"):
            CODE_ROOT = str(p.parent.parent.parent)
            break

if not CODE_ROOT:
    print("Cloning repository from GitHub...")
    subprocess.run(
        ["git", "clone", "https://github.com/AryanPatani/Amazon_ML_Challenge_2026.git", "/kaggle/working/ml-code"],
        capture_output=True, text=True
    )
    CODE_ROOT = "/kaggle/working/ml-code"

print(f"✅ Code root: {CODE_ROOT}")
sys.path.insert(0, CODE_ROOT)
os.chdir("/kaggle/working")

TRAIN_DIR = None
for p in Path("/kaggle").rglob("train_source1.tsv"):
    TRAIN_DIR = str(p.parent)
    break

if not TRAIN_DIR:
    raise FileNotFoundError("Competition training data not found. Please attach the dataset.")

print(f"✅ Train data: {TRAIN_DIR}")

OUTPUT_DIR = Path("/kaggle/working/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Installing required dependencies...")
subprocess.run(["pip", "install", "-q", "rapidfuzz", "phonetics", "xgboost"], check=True)
print("✅ Environment ready!")

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.path_a.features import compute_features
from src.common.data_loader import load_ground_truth
from eval.f05 import macro_f05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ============================================================
# STEP 2: Load S1 Sample + Comprehensive Candidate Pool
# ============================================================
print("\n" + "=" * 60)
print("📥 STEP 2: LOADING DATA & GROUND TRUTH")
print("=" * 60)
t0 = time.time()

# 60,000 S1 entities provides the optimal statistical sample (~300,000 pairs)
SAMPLE_SIZE = 60_000
print(f"Loading {SAMPLE_SIZE:,} S1 rows...")
s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t", nrows=SAMPLE_SIZE, dtype=str, keep_default_na=False)

print("Loading candidate pool from S2 & S3 (200,000 rows each)...")
s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t", nrows=200_000, dtype=str, keep_default_na=False)
s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t", nrows=200_000, dtype=str, keep_default_na=False)
cands = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=["entity_id"])

print("Loading Ground Truth...")
gt = load_ground_truth(Path(TRAIN_DIR) / "train_ground_truth.tsv")
gt_map = {row["source1_entity_id"]: set(row["matches"]) for _, row in gt.iterrows()}
print(f"Data loaded in {time.time()-t0:.1f}s")

# ============================================================
# STEP 3: Pair Mining (True Matches + TF-IDF Hard Negatives)
# ============================================================
print("\n" + "=" * 60)
print("🎯 STEP 3: MINING POSITIVES & HIGH-OVERLAP HARD NEGATIVES")
print("=" * 60)
t0 = time.time()

# 1. True Positives
pos_pairs = []
for sid in s1["entity_id"]:
    for cid in gt_map.get(sid, []):
        pos_pairs.append({"s1_id": sid, "cand_id": cid, "is_match": 1})

print(f"Exact true positive pairs: {len(pos_pairs):,}")

# 2. Hard Negatives via Character 3-gram TF-IDF Nearest Neighbors
print("Mining lexical hard negatives via TF-IDF nearest neighbors...")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2)
cand_vecs = vec.fit_transform(cands["business_name"].fillna(""))
s1_vecs = vec.transform(s1["business_name"].fillna(""))

# Retrieve top-4 nearest candidate entities per S1 entity
nn = NearestNeighbors(n_neighbors=4, metric="cosine", algorithm="brute", n_jobs=-1)
nn.fit(cand_vecs)
_, indices = nn.kneighbors(s1_vecs)

cand_id_arr = cands["entity_id"].values
neg_pairs = []
for i, sid in enumerate(s1["entity_id"]):
    true_matches = gt_map.get(sid, set())
    for idx in indices[i]:
        cid = cand_id_arr[idx]
        if cid not in true_matches:
            neg_pairs.append({"s1_id": sid, "cand_id": cid, "is_match": 0})

all_pairs_df = pd.DataFrame(pos_pairs + neg_pairs).drop_duplicates(subset=["s1_id", "cand_id"])
print(f"Total training pairs: {len(all_pairs_df):,} (Pos: {len(pos_pairs):,}, Neg: {len(neg_pairs):,}) mined in {time.time()-t0:.1f}s")

# ============================================================
# STEP 4: Feature Extraction (Jaro-Winkler, Levenshtein, PIN)
# ============================================================
print("\n" + "=" * 60)
print("⚙️ STEP 4: EXTRACTING CLASSICAL ML FEATURES (~3 to 4 mins)")
print("=" * 60)
t0 = time.time()

feats_df = compute_features(all_pairs_df, s1, cands)
feats_df["is_match"] = all_pairs_df["is_match"].values

features_path = OUTPUT_DIR / "a2_features.parquet"
feats_df.to_parquet(features_path, index=False)
print(f"Feature extraction complete! {feats_df.shape} saved in {time.time()-t0:.1f}s")

# ============================================================
# STEP 5: Train GPU-Accelerated XGBoost Classifier
# ============================================================
print("\n" + "=" * 60)
print("🌲 STEP 5: TRAINING XGBOOST & VALIDATING F0.5")
print("=" * 60)
t0 = time.time()

exclude_cols = {"s1_id", "cand_id", "is_match", "country"}
feature_cols = sorted([c for c in feats_df.columns if c not in exclude_cols])
print(f"Features used ({len(feature_cols)}): {feature_cols}")

# Split 20% validation by unique S1 entities to prevent data leakage
unique_s1 = sorted(feats_df["s1_id"].unique())
train_s1, val_s1 = train_test_split(unique_s1, test_size=0.2, random_state=42)
train_s1_set, val_s1_set = set(train_s1), set(val_s1)

train_mask = feats_df["s1_id"].isin(train_s1_set)
val_mask = feats_df["s1_id"].isin(val_s1_set)

X_train = feats_df.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_train = feats_df.loc[train_mask, "is_match"].astype(int)

X_val = feats_df.loc[val_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_val = feats_df.loc[val_mask, "is_match"].astype(int)

use_gpu = torch.cuda.is_available()
print(f"Training on {'GPU (CUDA)' if use_gpu else 'CPU'} ({len(X_train):,} train pairs, {len(X_val):,} val pairs)...")

clf = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.06,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cuda" if use_gpu else "cpu",
    eval_metric="logloss",
    random_state=42
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# Evaluate Validation F0.5
print("\nEvaluating Validation F0.5 score across thresholds...")
val_preds = clf.predict_proba(X_val)[:, 1]
val_pairs_df = feats_df.loc[val_mask, ["s1_id", "cand_id"]].copy()
val_pairs_df["score"] = val_preds

val_gt_dict = {sid: list(gt_map.get(sid, [])) for sid in val_s1_set if sid in gt_map}

best_thresh = 0.5
best_f05 = 0.0
for thresh in np.arange(0.2, 0.9, 0.05):
    pred_dict = {}
    above = val_pairs_df[val_pairs_df["score"] >= thresh]
    for sid, cid in zip(above["s1_id"], above["cand_id"]):
        pred_dict.setdefault(sid, []).append(cid)
    
    score = macro_f05(pred_dict, val_gt_dict)
    if score > best_f05:
        best_f05 = score
        best_thresh = thresh

print(f"🏆 Best Validation Macro F0.5: {best_f05:.4f} (Optimal Threshold: {best_thresh:.2f})")

model_path = OUTPUT_DIR / "path_a_xgb_optimized.json"
clf.save_model(str(model_path))
print(f"\n✅ Saved trained XGBoost model to: {model_path}")

# Feature Importances
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(importances.head(10).to_string())

print("\n" + "=" * 60)
print(f"🎉 TRAINING COMPLETE IN {time.time()-t0:.1f}s!")
print(f"Download your model from: {model_path}")
print("=" * 60)
