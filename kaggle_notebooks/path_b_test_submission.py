"""
============================================================
  PATH B — TEST SET INFERENCE & SUBMISSION GENERATOR
  Amazon ML Challenge 2026
============================================================

INSTRUCTIONS FOR THE PERSON RUNNING THIS:
------------------------------------------
1. Create a new Kaggle notebook and paste this entire script.

2. Attach these datasets to the notebook:
   - The competition dataset (containing test_source1/2/3.tsv)
   - The ml-code dataset uploaded from our repo
     (must contain models/bi_encoder_b2/ and models/cross_encoder_b3/)

3. Set accelerator to GPU T4 x2.

4. Run all cells. Expected runtime: ~3-4 hours.

5. Download from /kaggle/working/output/:
   - matching_results.tsv
   - candidate_pairs.tsv

6. Submit those two files to the leaderboard.

Expected validation F0.5: ~97.97%  |  Threshold: 0.95
============================================================
"""

# ============================================================
# CELL 1: Environment Setup & Path Discovery
# ============================================================
import os, sys, time, subprocess
from pathlib import Path

# --- Locate code root (adjust if upload path differs) ---
POSSIBLE_CODE_ROOTS = [
    "/kaggle/input/datasets/vivantejani/ml-code/ml-code",
    "/kaggle/input/ml-code/Amazon_ML_Challenge_2026",
    "/kaggle/input/ml-code",
    "/kaggle/working/ml-code",
    "/kaggle/working/Amazon_ML_Challenge_2026",
]
CODE_ROOT = None
for path in POSSIBLE_CODE_ROOTS:
    if Path(path).exists() and (Path(path) / "src").exists():
        CODE_ROOT = path
        break
if CODE_ROOT is None:
    try:
        CODE_ROOT = str(next(Path("/kaggle/input").rglob("src/path_b/b3_pipeline.py")).parent.parent.parent)
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
if CODE_ROOT is None:
    raise FileNotFoundError("Code root not found. Attach the ml-code dataset containing src/.")

print(f"Code root: {CODE_ROOT}")
sys.path.insert(0, CODE_ROOT)
os.chdir(CODE_ROOT)

# --- Locate test data ---
TEST_DIR = None
for p in Path("/kaggle").rglob("test_source1.tsv"):
    TEST_DIR = str(p.parent)
    break
if TEST_DIR is None:
    raise FileNotFoundError("test_source1.tsv not found. Attach the competition dataset.")
print(f"Test data: {TEST_DIR}")

OUTPUT_DIR = Path("/kaggle/working/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CELL 2: Install Dependencies
# ============================================================
os.system("pip install -q rapidfuzz phonetics sentence-transformers")

# ============================================================
# CELL 3: Imports
# ============================================================
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

print(f"PyTorch: {torch.__version__}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", end="")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  (No GPU — will be slow)")

# ============================================================
# CELL 4: Load Test Data
# ============================================================
print("\n=== LOADING TEST DATA ===")
t0 = time.time()
s1 = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
s2 = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
s3 = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
print(f"S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}  ({time.time()-t0:.1f}s)")

# ============================================================
# CELL 5: Serialize Entity Texts  (name | address)
# ============================================================
def serialize(df):
    texts = []
    for _, row in df.iterrows():
        name = str(row.get("business_name", "")).strip()
        addr = str(row.get("business_address", "")).strip()
        if addr and addr.lower() not in ("", "nan", "n/a", "null"):
            texts.append(f"{name} | {addr}")
        else:
            texts.append(name)
    return texts

print("\n=== SERIALIZING TEXTS ===")
t0 = time.time()
s1_texts     = serialize(s1)
corpus_texts = serialize(s2) + serialize(s3)
corpus_ids   = list(s2["entity_id"]) + list(s3["entity_id"])
s1_ids       = list(s1["entity_id"])
print(f"S1: {len(s1_texts):,}  Corpus: {len(corpus_texts):,}  ({time.time()-t0:.1f}s)")

# ============================================================
# CELL 6: Load Bi-Encoder & Encode Corpus
# ============================================================
print("\n=== ENCODING CORPUS WITH BI-ENCODER ===")
BI_PATH = next(
    (p for p in [
        f"{CODE_ROOT}/models/bi_encoder_b2",
        "/kaggle/working/models/bi_encoder_b2",
    ] if Path(p).exists()),
    "sentence-transformers/all-MiniLM-L6-v2"  # fallback
)
print(f"Model: {BI_PATH}")
bi_encoder = SentenceTransformer(BI_PATH, device=DEVICE)

t0 = time.time()
corpus_emb = bi_encoder.encode(
    corpus_texts, batch_size=512, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True
)
print(f"Corpus encoded: {corpus_emb.shape}  ({time.time()-t0:.1f}s)")

# ============================================================
# CELL 7: Encode S1 & Retrieve Top-20 Candidates
# ============================================================
print("\n=== ENCODING S1 & RETRIEVAL ===")
TOP_K = 20

t0 = time.time()
s1_emb = bi_encoder.encode(
    s1_texts, batch_size=512, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True
)
print(f"S1 encoded: {s1_emb.shape}  ({time.time()-t0:.1f}s)")

print(f"\nTop-{TOP_K} retrieval via batched cosine similarity...")
t0 = time.time()
RBATCH = 2048
corpus_t = torch.tensor(corpus_emb, device=DEVICE, dtype=torch.float32)
candidates = {}

for start in tqdm(range(0, len(s1_emb), RBATCH), desc="Retrieval"):
    batch = torch.tensor(s1_emb[start:start+RBATCH], device=DEVICE, dtype=torch.float32)
    sims = torch.mm(batch, corpus_t.T)
    scores, idxs = torch.topk(sims, k=TOP_K, dim=1)
    for i, (sc_row, idx_row) in enumerate(zip(scores.cpu().numpy(), idxs.cpu().numpy())):
        sid = s1_ids[start + i]
        candidates[sid] = [(corpus_ids[j], float(sc)) for j, sc in zip(idx_row, sc_row)]

del corpus_t, s1_emb, corpus_emb
print(f"Retrieval done ({time.time()-t0:.1f}s)")

# ============================================================
# CELL 8: Cross-Encoder Re-Ranking
# ============================================================
print("\n=== CROSS-ENCODER RE-RANKING ===")
CE_PATH = next(
    (p for p in [
        f"{CODE_ROOT}/models/cross_encoder_b3",
        "/kaggle/working/models/cross_encoder_b3",
    ] if Path(p).exists()),
    None
)

# Build entity text lookup
all_lookup = {}
for df in (s1, s2, s3):
    for _, row in df.iterrows():
        eid  = row["entity_id"]
        name = str(row.get("business_name", "")).strip()
        addr = str(row.get("business_address", "")).strip()
        if addr and addr.lower() not in ("", "nan", "n/a", "null"):
            all_lookup[eid] = f"{name} | {addr}"
        else:
            all_lookup[eid] = name

if CE_PATH:
    print(f"Model: {CE_PATH}")
    cross_encoder = CrossEncoder(CE_PATH, device=DEVICE)

    ce_pairs, ce_meta = [], []
    for sid, cands in candidates.items():
        s1_txt = all_lookup.get(sid, sid)
        for cid, bi_sc in cands:
            ce_pairs.append([s1_txt, all_lookup.get(cid, cid)])
            ce_meta.append((sid, cid, bi_sc))

    print(f"Total CE pairs: {len(ce_pairs):,}")
    t0 = time.time()
    ce_scores = cross_encoder.predict(
        ce_pairs, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, apply_softmax=True
    )
    print(f"Re-ranking done ({time.time()-t0:.1f}s)")

    final = {}
    for (sid, cid, _), ce_sc in zip(ce_meta, ce_scores):
        sc = float(ce_sc[1]) if hasattr(ce_sc, "__len__") else float(ce_sc)
        final.setdefault(sid, []).append((cid, sc))
else:
    print("WARNING: Cross-encoder not found. Using bi-encoder scores only.")
    final = candidates

# ============================================================
# CELL 9: Threshold + 1-to-1 Assignment
# ============================================================
print("\n=== APPLYING THRESHOLD (0.95) + 1-TO-1 CONSTRAINT ===")
THRESHOLD = 0.95

all_pairs = sorted(
    [(sc, sid, cid)
     for sid, cands in final.items()
     for cid, sc in cands
     if sc >= THRESHOLD],
    reverse=True
)
print(f"Pairs above threshold: {len(all_pairs):,}")

assigned: set = set()
matching: dict = {sid: [] for sid in s1_ids}
for sc, sid, cid in all_pairs:
    if cid not in assigned:
        assigned.add(cid)
        matching[sid].append(cid)

n_matched   = sum(1 for v in matching.values() if v)
n_singleton = sum(1 for v in matching.values() if not v)
print(f"Matched entities:    {n_matched:,}")
print(f"Singleton entities:  {n_singleton:,}")

# ============================================================
# CELL 10: Write Submission Files
# ============================================================
print("\n=== WRITING TSV FILES ===")

# matching_results.tsv
match_df = pd.DataFrame([
    {"source1_entity_id": sid, "matched_entity_ids": ",".join(matching[sid])}
    for sid in s1_ids
])
match_path = OUTPUT_DIR / "matching_results.tsv"
match_df.to_csv(match_path, sep="\t", index=False)
print(f"Written: {match_path}  ({len(match_df):,} rows)")

# candidate_pairs.tsv (all above-threshold, pre-1-to-1 constraint)
cand_rows = []
for sid in s1_ids:
    above = sorted(
        [(cid, sc) for cid, sc in final.get(sid, []) if sc >= THRESHOLD],
        key=lambda x: -x[1]
    )
    if not above:  # guarantee every S1 has at least its best candidate
        top = sorted(final.get(sid, []), key=lambda x: -x[1])[:1]
        above = top
    cand_rows.append({
        "source1_entity_id": sid,
        "candidate_entity_ids": ",".join(c for c, _ in above)
    })

cand_df = pd.DataFrame(cand_rows)
cand_path = OUTPUT_DIR / "candidate_pairs.tsv"
cand_df.to_csv(cand_path, sep="\t", index=False)
print(f"Written: {cand_path}  ({len(cand_df):,} rows)")

# ============================================================
# CELL 11: Run Validator
# ============================================================
print("\n=== RUNNING VALIDATOR ===")
validator = next(
    (p for p in [
        f"{CODE_ROOT}/utils/validate_submission.py",
        f"{CODE_ROOT}/data/utils/validate_submission.py",
    ] if Path(p).exists()),
    None
)
if validator:
    r = subprocess.run(
        [sys.executable, validator,
         "--matching", str(match_path),
         "--candidate", str(cand_path),
         "--test-dir", TEST_DIR],
        capture_output=True, text=True
    )
    print(r.stdout)
    if r.returncode == 0:
        print("VALIDATOR: PASS - Ready to submit!")
    else:
        print("VALIDATOR: FAIL - Fix issues before submitting.")
        print(r.stderr)
else:
    print("Validator not found - skipping.")

# ============================================================
# CELL 12: Summary
# ============================================================
print("\n" + "="*60)
print("SUBMISSION READY")
print("="*60)
print(f"  Model:     Path B (Bi-Encoder B2 + Cross-Encoder B3)")
print(f"  Threshold: {THRESHOLD}  (1-to-1 constraint applied)")
print(f"  Val F0.5:  97.97% (1,000-entity validation subset)")
print(f"  Matched:   {n_matched:,} S1 entities")
print(f"  Singletons:{n_singleton:,} S1 entities")
print()
print("  Download from /kaggle/working/output/:")
print("    matching_results.tsv")
print("    candidate_pairs.tsv")
print("="*60)
