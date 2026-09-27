"""
============================================================
  PATH A — OPTIMIZED TRAINING USING PATH B CANDIDATES
  Amazon ML Challenge 2026
============================================================

INSTRUCTIONS:
1. Create a Kaggle notebook (Python).
2. Attach datasets: 
   - Competition dataset (train_source1.tsv, etc.)
   - Our repo (ml-code, containing src/ and models/bi_encoder_b2/)
3. Set accelerator to GPU T4 x2.
4. Run all cells. Expected runtime: ~1.5 hours.

Why this is fast:
Instead of running feature extraction (CPU bound) on all 2.2 million S1 entities (44M pairs -> 4.5 hrs), 
we randomly subsample 100,000 S1 entities, get their Path B candidates (2M pairs), extract 
features in ~15 mins, and train XGBoost. This is more than enough data to train a perfect classifier.

At the end, download the `path_a_xgb_optimized.json` model!
============================================================
"""

import os, sys, time, subprocess, random
from pathlib import Path

# ============================================================
# CELL 1: Setup
# ============================================================
# Locate code root
POSSIBLE_CODE_ROOTS = [
    "/kaggle/input/datasets/vivantejani/ml-code/ml-code",
    "/kaggle/input/ml-code/Amazon_ML_Challenge_2026",
    "/kaggle/input/ml-code",
    "/kaggle/working/ml-code",
    "/kaggle/working/Amazon_ML_Challenge_2026",
]
CODE_ROOT = next((p for p in POSSIBLE_CODE_ROOTS if (Path(p) / "src").exists()), None)
if not CODE_ROOT:
    try:
        CODE_ROOT = str(next(Path("/kaggle/input").rglob("src/path_a/features.py")).parent.parent.parent)
    except StopIteration:
        print("\n" + "!" * 80)
        print("ERROR: COULD NOT FIND THE CODE DIRECTORY (src/)")
        print("!" * 80)
        print("It looks like the 'ml-code' dataset is not attached, or the directory structure is wrong.")
        print("Please check the following:")
        print("1. Did you attach the dataset containing our repository code?")
        print("2. Expand the attached datasets in the right sidebar. You should see a 'src' folder somewhere.")
        print("3. If you see it, update the POSSIBLE_CODE_ROOTS list in this cell with that path.")
        print("\nHere are the directories currently found in /kaggle/input/:")
        import glob
        for d in glob.glob("/kaggle/input/*"):
            print(f" - {d}")
            for sub_d in glob.glob(f"{d}/*"):
                print(f"    - {sub_d}")
        print("!" * 80)
        raise FileNotFoundError("Code root not found. Please attach the repo dataset.")

print(f"Code root: {CODE_ROOT}")
sys.path.insert(0, CODE_ROOT)
os.chdir(CODE_ROOT)

# Locate training data
TRAIN_DIR = None
for p in Path("/kaggle").rglob("train_source1.tsv"):
    TRAIN_DIR = str(p.parent)
    break
print(f"Train data: {TRAIN_DIR}")

OUTPUT_DIR = Path("/kaggle/working/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Install dependencies
os.system("pip install -q rapidfuzz phonetics sentence-transformers xgboost")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ============================================================
# CELL 2: Load Data
# ============================================================
print("\n=== LOADING TRAINING DATA ===")
s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

# Subsample 150,000 S1 entities for training to save CPU time on feature extraction
SAMPLE_SIZE = 150_000
np.random.seed(42)
if len(s1) > SAMPLE_SIZE:
    print(f"Subsampling S1 from {len(s1):,} to {SAMPLE_SIZE:,} for fast training...")
    s1 = s1.sample(n=SAMPLE_SIZE, random_state=42).copy()

def serialize(df):
    texts = []
    for _, row in df.iterrows():
        name = str(row.get("business_name", "")).strip()
        addr = str(row.get("business_address", "")).strip()
        texts.append(f"{name} | {addr}" if addr and addr.lower() not in ("nan","n/a","") else name)
    return texts

print("Serializing texts...")
s1_texts = serialize(s1)
corpus_texts = serialize(s2) + serialize(s3)
corpus_ids = list(s2["entity_id"]) + list(s3["entity_id"])
s1_ids = list(s1["entity_id"])

# ============================================================
# CELL 3: Bi-Encoder Candidate Generation
# ============================================================
print("\n=== GENERATING CANDIDATES (BI-ENCODER) ===")
BI_PATH = next((p for p in [f"{CODE_ROOT}/models/bi_encoder_b2", "/kaggle/working/models/bi_encoder_b2"] if Path(p).exists()), "sentence-transformers/all-MiniLM-L6-v2")
print(f"Loading Bi-Encoder: {BI_PATH}")
bi_encoder = SentenceTransformer(BI_PATH, device=DEVICE)

print("Encoding Corpus...")
corpus_emb = bi_encoder.encode(corpus_texts, batch_size=512, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

print("Encoding S1...")
s1_emb = bi_encoder.encode(s1_texts, batch_size=512, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

print("Retrieving Top-20 Candidates...")
TOP_K = 20
RBATCH = 2048
corpus_t = torch.tensor(corpus_emb, device=DEVICE, dtype=torch.float32)

pairs = []
for start in tqdm(range(0, len(s1_emb), RBATCH)):
    batch = torch.tensor(s1_emb[start:start+RBATCH], device=DEVICE, dtype=torch.float32)
    sims = torch.mm(batch, corpus_t.T)
    _, idxs = torch.topk(sims, k=TOP_K, dim=1)
    for i, idx_row in enumerate(idxs.cpu().numpy()):
        sid = s1_ids[start + i]
        for j in idx_row:
            pairs.append({"s1_id": sid, "cand_id": corpus_ids[j]})

del corpus_t, s1_emb, corpus_emb
pairs_df = pd.DataFrame(pairs)
pairs_path = OUTPUT_DIR / "a1_candidate_pairs_optimized.csv"
pairs_df.to_csv(pairs_path, index=False)
print(f"Generated {len(pairs_df):,} pairs for feature extraction.")

# ============================================================
# CELL 4: Path A Feature Extraction
# ============================================================
print("\n=== EXTRACTING CLASSICAL ML FEATURES (CPU) ===")
print("This takes about 10-15 minutes...")

# We can call features.py directly
features_cmd = [
    sys.executable, f"{CODE_ROOT}/src/path_a/features.py",
    "--pairs", str(pairs_path),
    "--out", str(OUTPUT_DIR / "a2_features.parquet"),
    "--data-dir", TRAIN_DIR
]
subprocess.run(features_cmd, check=True)

# ============================================================
# CELL 5: Negative Sampling
# ============================================================
print("\n=== NEGATIVE SAMPLING ===")
sample_cmd = [
    sys.executable, f"{CODE_ROOT}/src/path_a/sample_negatives.py",
    "--features", str(OUTPUT_DIR / "a2_features.parquet"),
    "--out", str(OUTPUT_DIR / "a3_train_sampled.parquet"),
    "--ratio", "3.0"
]
subprocess.run(sample_cmd, check=True)

# ============================================================
# CELL 6: Train XGBoost Classifier
# ============================================================
print("\n=== TRAINING XGBOOST CLASSIFIER ===")
train_cmd = [
    sys.executable, f"{CODE_ROOT}/src/path_a/classifier.py",
    "--train", str(OUTPUT_DIR / "a3_train_sampled.parquet"),
    "--out", str(OUTPUT_DIR / "path_a_xgb_optimized.json")
]
subprocess.run(train_cmd, check=True)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print(f"Download your optimized model from: /kaggle/working/output/path_a_xgb_optimized.json")
print("="*60)
