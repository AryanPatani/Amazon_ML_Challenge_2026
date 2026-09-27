"""
============================================================
  PATH A — FAST & ROBUST XGBOOST TRAINING PIPELINE
  Amazon ML Challenge 2026
============================================================

Total runtime: ~3 to 5 minutes!
Zero GPU memory issues (no 80-minute neural encoding of 2.2M texts).
Trains XGBoost on 150k balanced pairs (Ground Truth + TF-IDF Hard Negatives).

Exports: /kaggle/working/output/path_a_xgb_optimized.json
============================================================
"""

import os, sys, time, subprocess
from pathlib import Path

# ============================================================
# STEP 1: Environment Setup & Directory Discovery
# ============================================================
print("=" * 60)
print("🚀 STEP 1: PATH DISCOVERY & SETUP")
print("=" * 60)

POSSIBLE_CODE_ROOTS = [
    "/kaggle/input/datasets/vivantejani/ml-code/ml-code",
    "/kaggle/input/datasets/vivantejani/ml-code",
    "/kaggle/input/datasets/vivantejani",
    "/kaggle/input/ml-code/Amazon_ML_Challenge_2026",
    "/kaggle/input/ml-code",
    "/kaggle/working/ml-code",
    "/kaggle/working/Amazon_ML_Challenge_2026",
]
CODE_ROOT = next((p for p in POSSIBLE_CODE_ROOTS if (Path(p) / "src").exists()), None)

if not CODE_ROOT:
    for p in Path("/kaggle").rglob("features.py"):
        if p.parent.name == "path_a" and (p.parent.parent.name == "src"):
            CODE_ROOT = str(p.parent.parent.parent)
            break

if not CODE_ROOT:
    print("🌐 Cloning repository from GitHub...")
    subprocess.run(
        ["git", "clone", "https://github.com/AryanPatani/Amazon_ML_Challenge_2026.git", "/kaggle/working/ml-code"],
        capture_output=True, text=True
    )
    if (Path("/kaggle/working/ml-code") / "src").exists():
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

# Install fast dependencies
print("📦 Installing dependencies...")
subprocess.run(["pip", "install", "-q", "rapidfuzz", "phonetics", "xgboost"], check=True)
print("✅ Dependencies ready!")

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import xgboost as xgb

from src.path_a.features import compute_features
from src.common.data_loader import load_ground_truth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Accelerator Device: {DEVICE}")

# ============================================================
# STEP 2: Load Data & Ground Truth
# ============================================================
print("\n" + "=" * 60)
print("📥 STEP 2: LOADING DATA & GROUND TRUTH")
print("=" * 60)
t0 = time.time()

# Subsample 25,000 S1 entities for ultra-fast training (~150,000 training pairs)
SAMPLE_SIZE = 25_000
print(f"Loading S1 sample ({SAMPLE_SIZE:,} rows)...")
s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t", nrows=SAMPLE_SIZE, dtype=str, keep_default_na=False)

print("Loading S2 and S3...")
# Load matching pool of S2 and S3 (100k rows each provides ample hard negatives)
s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t", nrows=100_000, dtype=str, keep_default_na=False)
s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t", nrows=100_000, dtype=str, keep_default_na=False)
cands = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=["entity_id"])

print("Loading Ground Truth...")
gt = load_ground_truth(Path(TRAIN_DIR) / "train_ground_truth.tsv")
gt_map = {row["source1_entity_id"]: set(row["matches"]) for _, row in gt.iterrows()}
print(f"Data loaded in {time.time()-t0:.1f}s")

# ============================================================
# STEP 3: Pair Mining (Ground Truth Positives + TF-IDF Hard Negatives)
# ============================================================
print("\n" + "=" * 60)
print("🎯 STEP 3: MINING POSITIVES & HARD NEGATIVES")
print("=" * 60)
t0 = time.time()

# 1. Exact True Positive Pairs from Ground Truth
pos_pairs = []
for sid in s1["entity_id"]:
    for cid in gt_map.get(sid, []):
        pos_pairs.append({"s1_id": sid, "cand_id": cid, "is_match": 1})

print(f"True positive pairs: {len(pos_pairs):,}")

# 2. Hard Negatives via Character 3-gram TF-IDF Cosine Similarity
print("Mining hard negatives using TF-IDF nearest neighbors...")
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2)
cand_vecs = vec.fit_transform(cands["business_name"].fillna(""))
s1_vecs = vec.transform(s1["business_name"].fillna(""))

# Find top-4 nearest candidates per S1 entity
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
# STEP 4: Classical ML Feature Extraction
# ============================================================
print("\n" + "=" * 60)
print("⚙️ STEP 4: EXTRACTING CLASSICAL ML FEATURES")
print("=" * 60)
t0 = time.time()

# Compute all 21 features (Jaro-Winkler, Levenshtein, Jaccard, Token Overlap, PIN match, etc.)
feats_df = compute_features(all_pairs_df, s1, cands)
feats_df["is_match"] = all_pairs_df["is_match"].values

features_path = OUTPUT_DIR / "a2_features.parquet"
feats_df.to_parquet(features_path, index=False)
print(f"Feature extraction complete! {feats_df.shape} saved in {time.time()-t0:.1f}s")

# ============================================================
# STEP 5: Train GPU-Accelerated XGBoost Classifier
# ============================================================
print("\n" + "=" * 60)
print("🌲 STEP 5: TRAINING XGBOOST CLASSIFIER")
print("=" * 60)
t0 = time.time()

exclude_cols = {"s1_id", "cand_id", "is_match", "country"}
feature_cols = sorted([c for c in feats_df.columns if c not in exclude_cols])
print(f"Features ({len(feature_cols)}): {feature_cols}")

X = feats_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
y = feats_df["is_match"].astype(int)

use_gpu = torch.cuda.is_available()
print(f"Fitting XGBoost on {'GPU (CUDA)' if use_gpu else 'CPU'} ({len(X):,} pairs)...")

clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    device="cuda" if use_gpu else "cpu",
    eval_metric="logloss",
    random_state=42
)
clf.fit(X, y)

model_path = OUTPUT_DIR / "path_a_xgb_optimized.json"
clf.save_model(str(model_path))
print(f"Training completed in {time.time()-t0:.1f}s!")
print(f"✅ Saved trained XGBoost model to: {model_path}")

# Display Feature Importances
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(importances.head(10).to_string())

print("\n" + "=" * 60)
print("🎉 ALL DONE! DOWNLOAD YOUR TRAINED MODEL:")
print(f"   {model_path}")
print("=" * 60)
