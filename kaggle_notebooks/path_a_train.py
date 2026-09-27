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
# Locate code root with multi-stage discovery & fallback
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

# Search recursively for features.py if not in default paths
if not CODE_ROOT:
    for p in Path("/kaggle").rglob("features.py"):
        if p.parent.name == "path_a" and (p.parent.parent.name == "src"):
            CODE_ROOT = str(p.parent.parent.parent)
            break

# Auto-extract any zip files in /kaggle/input if code wasn't found
if not CODE_ROOT:
    zip_files = list(Path("/kaggle/input").rglob("*.zip"))
    if zip_files:
        import zipfile
        for zf in zip_files:
            print(f"📦 Found zip archive: {zf.name}. Extracting to /kaggle/working/ml-code...")
            try:
                with zipfile.ZipFile(zf, 'r') as zip_ref:
                    zip_ref.extractall("/kaggle/working/ml-code")
            except Exception as e:
                print(f"Extraction warning: {e}")
        for p in Path("/kaggle/working/ml-code").rglob("features.py"):
            if p.parent.name == "path_a":
                CODE_ROOT = str(p.parent.parent.parent)
                break

# Git clone fallback (if Internet is ON in Kaggle settings)
if not CODE_ROOT:
    print("🌐 Attempting to clone repository from GitHub...")
    clone_res = subprocess.run(
        ["git", "clone", "https://github.com/AryanPatani/Amazon_ML_Challenge_2026.git", "/kaggle/working/ml-code"],
        capture_output=True, text=True
    )
    if (Path("/kaggle/working/ml-code") / "src").exists():
        CODE_ROOT = "/kaggle/working/ml-code"
        print("✅ Cloned successfully from GitHub!")

if not CODE_ROOT:
    print("\n" + "!" * 80)
    print("ERROR: COULD NOT FIND THE CODE DIRECTORY (src/)")
    print("!" * 80)
    print("Here is the FULL directory tree of /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        level = root.replace("/kaggle/input", "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files[:8]:
            print(f"{subindent}{f}")
        if len(files) > 8:
            print(f"{subindent}... and {len(files) - 8} more files")
    print("!" * 80)
    raise FileNotFoundError(
        "Code root not found. Please enable Internet in Kaggle settings (right sidebar -> Internet: ON) "
        "or attach the repository dataset."
    )

print(f"✅ Code root: {CODE_ROOT}")
sys.path.insert(0, CODE_ROOT)
os.chdir("/kaggle/working")

# Locate training data
TRAIN_DIR = None
for p in Path("/kaggle").rglob("train_source1.tsv"):
    TRAIN_DIR = str(p.parent)
    break

if not TRAIN_DIR:
    print("\n" + "!" * 80)
    print("ERROR: COULD NOT FIND COMPETITION DATASET (train_source1.tsv)")
    print("!" * 80)
    print("The competition dataset is NOT currently attached to this notebook.")
    print("To fix this:")
    print("1. In the right sidebar, click '+ Add Data' (or 'Add Input').")
    print("2. Search for the competition dataset (containing train_source1.tsv).")
    print("3. Click the '+' button to add it, then re-run this cell.")
    print("!" * 80)
    raise FileNotFoundError("Competition training data not found. Please attach the dataset.")

print(f"✅ Train data: {TRAIN_DIR}")

OUTPUT_DIR = Path("/kaggle/working/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Install dependencies
print("📦 Installing dependencies (rapidfuzz, phonetics, sentence-transformers, xgboost)...")
subprocess.run(["pip", "install", "-q", "rapidfuzz", "phonetics", "sentence-transformers", "xgboost"], check=True)
print("✅ Environment ready!")

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
# Locate fine-tuned bi-encoder model
BI_PATH = None
for p in [f"{CODE_ROOT}/models/bi_encoder_b2", "/kaggle/working/models/bi_encoder_b2"]:
    if Path(p).exists() and (Path(p) / "modules.json").exists():
        BI_PATH = p
        break

if not BI_PATH:
    for p in Path("/kaggle").rglob("bi_encoder_b2"):
        if (p / "modules.json").exists():
            BI_PATH = str(p)
            break

if not BI_PATH:
    for p in Path("/kaggle").rglob("modules.json"):
        if "cross" not in str(p):
            BI_PATH = str(p.parent)
            break

if not BI_PATH:
    print("⚠️ Fine-tuned bi_encoder_b2 not found. Falling back to all-MiniLM-L6-v2.")
    BI_PATH = "sentence-transformers/all-MiniLM-L6-v2"

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
import xgboost as xgb

train_df = pd.read_parquet(OUTPUT_DIR / "a3_train_sampled.parquet")
exclude_cols = {'s1_id', 'cand_id', 'is_match', 'country'}
feature_cols = sorted([c for c in train_df.columns if c not in exclude_cols])
print(f"Features used ({len(feature_cols)}): {feature_cols}")

X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_train = train_df['is_match'].astype(int)

use_gpu = torch.cuda.is_available()
print(f"Fitting XGBoost on {'GPU (CUDA)' if use_gpu else 'CPU'} ({len(X_train):,} pairs)...")
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
clf.fit(X_train, y_train)

model_path = OUTPUT_DIR / "path_a_xgb_optimized.json"
clf.save_model(str(model_path))
print(f"✅ Saved trained XGBoost model to: {model_path}")

# Feature importances
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(importances.head(10))

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print(f"Download your optimized model from: /kaggle/working/output/path_a_xgb_optimized.json")
print("="*60)
