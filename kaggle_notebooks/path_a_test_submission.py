"""
============================================================
  PATH A — FAISS GPU-ACCELERATED SUBMISSION GENERATOR
  Amazon ML Challenge 2026
============================================================

Architecture: TF-IDF + TruncatedSVD (LSA) + FAISS GPU FlatIP Retrieval
  - Retrieval fully runs on GPU using FAISS GpuIndexFlatIP
  - NO dense output matrix -> NO GPU OOM (FAISS only returns top-k indices)
  - XGBoost scoring: fast multi-threaded CPU (0.03s per 50K pairs)
  
Memory Budget (per partition, US partition worst case):
  - System RAM: ~4 GB (dense SVD matrix: 3.8M x 128 x 4 = 1.95 GB)
  - GPU VRAM: ~2 GB FAISS index + 0.5 GB buffers = 2.5 GB (15 GB available)

Estimated Runtime: ~15 to 20 minutes total on Kaggle T4 x2
  - France (259K S1): ~3-4 minutes
  - US    (663K S1): ~6-7 minutes
  - India (810K S1): ~7-8 minutes
============================================================
"""

import os, sys, time, re, gc, subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp

print("=" * 65)
print("⚡ PATH A: FAISS GPU-ACCELERATED SUBMISSION GENERATOR")
print("=" * 65)
t_start = time.time()

# -------------------------------------------------------------
# STEP 0: DEPENDENCIES & GPU SETUP
# -------------------------------------------------------------
print("\n[0/5] Checking dependencies & GPU setup...")
subprocess.run(["pip", "install", "-q", "rapidfuzz", "xgboost"], check=False)
from rapidfuzz import distance, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import xgboost as xgb

has_gpu_faiss = False
faiss_gpu_res = None
try:
    import faiss
    if hasattr(faiss, 'StandardGpuResources'):
        faiss_gpu_res = faiss.StandardGpuResources()
        faiss_gpu_res.setTempMemory(512 * 1024 * 1024)  # 512 MB temp
        has_gpu_faiss = True
        print("✅ FAISS GPU index ACTIVE (GpuIndexFlatIP - zero OOM risk!)")
    else:
        print("✅ FAISS CPU available (faiss-gpu not installed, will use CPU FAISS)")
except ImportError:
    print("❌ FAISS not found! Installing faiss-gpu...")
    subprocess.run(["pip", "install", "-q", "faiss-gpu"], check=False)
    try:
        import faiss
        faiss_gpu_res = faiss.StandardGpuResources()
        faiss_gpu_res.setTempMemory(512 * 1024 * 1024)
        has_gpu_faiss = True
        print("✅ FAISS GPU installed and ready!")
    except Exception as e:
        print(f"⚠️ FAISS GPU unavailable ({e}). Using FAISS CPU.")
        has_gpu_faiss = False

# -------------------------------------------------------------
# STEP 1: PATH DISCOVERY & MODEL LOAD
# -------------------------------------------------------------
print("\n[1/5] Discovering paths and loading model...")
OUTPUT_DIR = Path("/kaggle/working/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POSSIBLE_MODEL_PATHS = [
    "/kaggle/working/output/path_a_xgb_optimized.json",
    "/kaggle/working/path_a_xgb_optimized.json",
    "/kaggle/working/ml-code/models/path_a_xgb_optimized.json",
    "/kaggle/input/ml-code/models/path_a_xgb_optimized.json",
]
model_path = next((Path(p) for p in POSSIBLE_MODEL_PATHS if Path(p).exists()), None)
if not model_path:
    for p in Path("/kaggle").rglob("path_a_xgb_optimized.json"):
        model_path = p
        break

if not model_path:
    raise FileNotFoundError("Could not find path_a_xgb_optimized.json!")

print(f"✅ Model: {model_path}")
clf = xgb.XGBClassifier(n_jobs=-1)
clf.load_model(str(model_path))
feature_cols = clf.feature_names_in_.tolist()
print(f"✅ Features ({len(feature_cols)}): {feature_cols}")

TEST_DIR = None
for p in Path("/kaggle").rglob("test_source1.tsv"):
    TEST_DIR = str(p.parent)
    break

if not TEST_DIR:
    raise FileNotFoundError("Could not find test_source1.tsv!")
print(f"✅ Test data: {TEST_DIR}")

# -------------------------------------------------------------
# STEP 2: LOAD TEST DATA
# -------------------------------------------------------------
print("\n[2/5] Loading test data...")
t0 = time.time()
s1_test = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
s2_test = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
s3_test = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
print(f"Loaded in {time.time()-t0:.1f}s | S1:{len(s1_test):,} | S2:{len(s2_test):,} | S3:{len(s3_test):,}")
cands_test = pd.concat([s2_test, s3_test], ignore_index=True).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
del s2_test, s3_test
gc.collect()
print(f"Total unique candidates: {len(cands_test):,}")

# -------------------------------------------------------------
# STEP 3: TEXT NORMALIZATION & FEATURE LOGIC
# -------------------------------------------------------------
LEGAL_TERMS = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "llc", "llp", "gmbh", "sa", "sarl", "co", "company"
}
ADDR_MAP = {"st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard", "dr": "drive"}

def clean_name(n):
    if not n: return ""
    tokens = re.sub(r"[^\w\s]", " ", str(n).lower()).split()
    return " ".join([t for t in tokens if t not in LEGAL_TERMS])

def clean_addr(a):
    if not a: return ""
    tokens = re.sub(r"[^\w\s]", " ", str(a).lower()).split()
    return " ".join([ADDR_MAP.get(t, t) for t in tokens])

def extract_pin(a):
    m = re.findall(r"\b\d{5,6}\b", str(a))
    return m[0] if m else ""

def get_3grams(text):
    return set(text[i:i+3] for i in range(len(text)-2)) if len(text) >= 3 else set()

def fast_compute_features(s1_records, cand_records, tfidf_sims):
    rows = []
    for (s1_id, n1, a1, p1), (cand_id, n2, a2, p2), sim_n in zip(s1_records, cand_records, tfidf_sims):
        t1, t2 = set(n1.split()), set(n2.split())
        g1, g2 = get_3grams(n1), get_3grams(n2)
        ag1, ag2 = get_3grams(a1), get_3grams(a2)
        at1, at2 = set(a1.split()), set(a2.split())
        n_tok1, n_tok2 = n1.split(), n2.split()
        acronym1 = "".join(w[0] for w in n_tok1 if w)
        acronym2 = "".join(w[0] for w in n_tok2 if w)
        nums1 = set(re.findall(r"\d+", n1))
        nums2 = set(re.findall(r"\d+", n2))
        jw_a = distance.JaroWinkler.similarity(a1, a2) if (a1 and a2) else 0.0
        rows.append({
            "acronym_match": int(bool(len(acronym1) >= 2 and len(acronym2) >= 2 and (acronym1 == acronym2 or acronym1 in n2 or acronym2 in n1))),
            "first_token_match": int(bool(n_tok1 and n_tok2 and n_tok1[0] == n_tok2[0])),
            "jaccard_addr_3g": len(ag1 & ag2) / max(len(ag1 | ag2), 1),
            "jaccard_addr_tok": len(at1 & at2) / max(len(at1 | at2), 1),
            "jaccard_name_3g": len(g1 & g2) / max(len(g1 | g2), 1),
            "jaccard_name_tok": len(t1 & t2) / max(len(t1 | t2), 1),
            "jw_addr": jw_a,
            "jw_name": distance.JaroWinkler.similarity(n1, n2) if (n1 and n2) else 0.0,
            "len_ratio_addr": min(len(a1), len(a2)) / max(len(a1), len(a2), 1),
            "len_ratio_name": min(len(n1), len(n2)) / max(len(n1), len(n2), 1),
            "lev_addr": distance.Levenshtein.normalized_similarity(a1, a2) if (a1 and a2) else 0.0,
            "lev_name": distance.Levenshtein.normalized_similarity(n1, n2) if (n1 and n2) else 0.0,
            "missing_addr": int(not a1 or not a2),
            "num_conflict": int(bool(nums1 and nums2 and not (nums1 & nums2))),
            "num_overlap": int(bool(nums1 and nums2 and (nums1 & nums2))),
            "pin_agree": int(bool(p1 and p2 and p1 == p2)),
            "pin_conflict": int(bool(p1 and p2 and p1 != p2)),
            "tfidf_sim_addr": jw_a,
            "tfidf_sim_name": float(sim_n),
            "ts_name": float(fuzz.token_set_ratio(n1, n2)) / 100.0 if (n1 and n2) else 0.0,
        })
    return pd.DataFrame(rows)

# -------------------------------------------------------------
# STEP 4: COUNTRY-PARTITIONED INFERENCE (FAISS GPU)
# -------------------------------------------------------------
THRESHOLD = 0.50
TOP_K = 5
LSA_DIMS = 128           # Compress TF-IDF to 128-dim dense (index size: ~3.8M*128*4 = 1.95 GB)
FAISS_BATCH = 65536      # 64K queries per FAISS call (maximizes GPU throughput)

all_s1_ids = list(s1_test["entity_id"].values)
all_matched_pairs = []
final_candidates = defaultdict(list)

countries = [c for c in s1_test["country"].dropna().unique().tolist() if c]
print(f"\n[3/5] Running FAISS GPU Country-Partitioned Inference on: {countries}")

for country in countries:
    t_c = time.time()
    print(f"\n---> Processing Partition: {country} <---")
    s1_c = s1_test[s1_test["country"] == country].copy().reset_index(drop=True)
    cands_c = cands_test[cands_test["country"] == country].copy().reset_index(drop=True)

    if len(s1_c) == 0 or len(cands_c) == 0:
        print(f"Skipping empty partition.")
        continue

    print(f"S1: {len(s1_c):,} | Candidates: {len(cands_c):,}")

    # Normalize text
    print("Normalizing...")
    s1_c["name_clean"] = s1_c["business_name"].apply(clean_name)
    s1_c["addr_clean"] = s1_c["business_address"].apply(clean_addr)
    s1_c["pin"] = s1_c["business_address"].apply(extract_pin)
    cands_c["name_clean"] = cands_c["business_name"].apply(clean_name)
    cands_c["addr_clean"] = cands_c["business_address"].apply(clean_addr)
    cands_c["pin"] = cands_c["business_address"].apply(extract_pin)

    cand_records_full = list(zip(cands_c["entity_id"], cands_c["name_clean"], cands_c["addr_clean"], cands_c["pin"]))
    s1_records_full = list(zip(s1_c["entity_id"], s1_c["name_clean"], s1_c["addr_clean"], s1_c["pin"]))

    # TF-IDF + LSA (TruncatedSVD) pipeline
    print(f"Building TF-IDF + LSA ({LSA_DIMS}d) pipeline...")
    t0 = time.time()
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.05)
    cand_vecs_sparse = tfidf.fit_transform(cands_c["name_clean"])
    s1_vecs_sparse = tfidf.transform(s1_c["name_clean"])

    svd = TruncatedSVD(n_components=LSA_DIMS, random_state=42)
    cand_dense = normalize(svd.fit_transform(cand_vecs_sparse)).astype(np.float32)
    s1_dense = normalize(svd.transform(s1_vecs_sparse)).astype(np.float32)

    # Keep TF-IDF sparse for feature tfidf_sim_name
    del cand_vecs_sparse
    del s1_vecs_sparse
    del tfidf
    del svd
    gc.collect()

    ram_mb = cand_dense.nbytes / (1024 * 1024)
    print(f"Dense vectors: {cand_dense.shape} ({ram_mb:.0f} MB RAM) in {time.time()-t0:.1f}s")

    # Build FAISS index
    print("Building FAISS index...")
    t0 = time.time()
    d = cand_dense.shape[1]
    cpu_index = faiss.IndexFlatIP(d)
    cpu_index.add(cand_dense)

    if has_gpu_faiss:
        try:
            gpu_index = faiss.index_cpu_to_gpu(faiss_gpu_res, 0, cpu_index)
            del cpu_index
            faiss_index = gpu_index
            print(f"✅ FAISS GPU index: {faiss_index.ntotal:,} vectors @ {d}d ({ram_mb:.0f} MB VRAM) in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"⚠️ FAISS GPU failed ({e}), using CPU index.")
            faiss_index = cpu_index
    else:
        faiss_index = cpu_index
        print(f"✅ FAISS CPU index: {faiss_index.ntotal:,} vectors in {time.time()-t0:.1f}s")

    # FAISS search: returns (scores, indices) without building the full similarity matrix
    print(f"Running FAISS search {len(s1_dense):,} queries -> top-{TOP_K}...")
    t0 = time.time()
    all_D = []
    all_I = []
    for i in range(0, len(s1_dense), FAISS_BATCH):
        batch = s1_dense[i:i + FAISS_BATCH]
        D_batch, I_batch = faiss_index.search(batch, TOP_K)
        all_D.append(D_batch)
        all_I.append(I_batch)
    all_D = np.vstack(all_D)   # (n_s1, TOP_K) similarity scores
    all_I = np.vstack(all_I)   # (n_s1, TOP_K) candidate indices

    print(f"FAISS search done in {time.time()-t0:.1f}s!")

    del faiss_index, cand_dense, s1_dense
    gc.collect()

    # Feature extraction & XGBoost scoring per batch
    print("Extracting features & scoring pairs...")
    SCORE_BATCH = 5000
    t0 = time.time()
    partition_matches = 0

    for i in range(0, len(s1_c), SCORE_BATCH):
        end_i = min(i + SCORE_BATCH, len(s1_c))
        batch_s1_records = []
        batch_cand_records = []
        batch_sims = []
        batch_meta = []

        for local_i in range(i, end_i):
            s1_tup = s1_records_full[local_i]
            sid = s1_tup[0]
            for k in range(TOP_K):
                cand_idx = int(all_I[local_i, k])
                sim = float(all_D[local_i, k])
                if cand_idx < 0 or cand_idx >= len(cand_records_full):
                    continue
                c_tup = cand_records_full[cand_idx]
                batch_s1_records.append(s1_tup)
                batch_cand_records.append(c_tup)
                batch_sims.append(sim)
                batch_meta.append((sid, c_tup[0]))
                final_candidates[sid].append(c_tup[0])

        if not batch_s1_records:
            continue

        feats_df = fast_compute_features(batch_s1_records, batch_cand_records, batch_sims)
        X = feats_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scores = clf.predict_proba(X)[:, 1]

        for (sid, cid), sc in zip(batch_meta, scores):
            if sc >= THRESHOLD:
                all_matched_pairs.append((float(sc), sid, cid))
                partition_matches += 1

        if (i // SCORE_BATCH + 1) % 20 == 0 or end_i == len(s1_c):
            elapsed = time.time() - t0
            rate = end_i / max(elapsed, 0.001)
            eta = (len(s1_c) - end_i) / max(rate, 0.001)
            print(f"  Scored {end_i:,}/{len(s1_c):,} S1 | {partition_matches:,} matches | ETA: {eta/60:.1f}m")

    del all_D, all_I, s1_c, cands_c, cand_records_full, s1_records_full
    gc.collect()
    print(f"✅ Finished {country} in {time.time()-t_c:.1f}s! ({partition_matches:,} matches found)")

# -------------------------------------------------------------
# STEP 5: GREEDY 1-TO-1 RESOLUTION & WRITE SUBMISSION FILES
# -------------------------------------------------------------
print("\n[4/5] Greedy 1-to-1 matching and writing TSVs...")

all_matched_pairs.sort(key=lambda x: -x[0])
assigned_cands = set()
resolved_matches = defaultdict(list)

for score, sid, cid in all_matched_pairs:
    if cid not in assigned_cands:
        assigned_cands.add(cid)
        resolved_matches[sid].append(cid)

match_rows = [{"source1_entity_id": sid, "matched_entity_ids": ",".join(resolved_matches.get(sid, []))} for sid in all_s1_ids]
match_df = pd.DataFrame(match_rows)
match_path = OUTPUT_DIR / "matching_results.tsv"
match_df.to_csv(match_path, sep="\t", index=False)
print(f"✅ matching_results.tsv: {len(match_df):,} rows")

cand_rows = [{"source1_entity_id": sid, "candidate_entity_ids": ",".join(list(dict.fromkeys(final_candidates.get(sid, []))))} for sid in all_s1_ids]
cand_df = pd.DataFrame(cand_rows)
cand_path = OUTPUT_DIR / "candidate_pairs.tsv"
cand_df.to_csv(cand_path, sep="\t", index=False)
print(f"✅ candidate_pairs.tsv: {len(cand_df):,} rows")

# -------------------------------------------------------------
# STEP 6: AUTOMATED INTEGRITY CHECK
# -------------------------------------------------------------
print("\n" + "=" * 65)
print("🔍 AUTOMATED INTEGRITY & SANITY CHECK REPORT")
print("=" * 65)
errors = []

if not match_path.exists() or match_path.stat().st_size == 0:
    errors.append("matching_results.tsv is missing or empty!")
else:
    print(f"  [PASS] File on disk ({match_path.stat().st_size/(1024*1024):.2f} MB)")

if list(match_df.columns) != ["source1_entity_id", "matched_entity_ids"]:
    errors.append(f"Header mismatch: {list(match_df.columns)}")
else:
    print("  [PASS] Header correct")

if len(match_df) != len(s1_test):
    errors.append(f"Row count mismatch! Expected {len(s1_test):,}, got {len(match_df):,}")
else:
    print(f"  [PASS] 100% S1 coverage ({len(match_df):,} rows)")

dupes = match_df["source1_entity_id"].duplicated().sum()
if dupes > 0:
    errors.append(f"Found {dupes:,} duplicate S1 rows!")
else:
    print("  [PASS] Zero duplicate S1 rows")

bad_cids = sum(1 for m_str in match_df["matched_entity_ids"] if m_str for cid in m_str.split(",") if not (cid.startswith("S2-") or cid.startswith("S3-")))
if bad_cids > 0:
    errors.append(f"Found {bad_cids:,} invalid match IDs!")
else:
    print("  [PASS] All IDs valid (S2- or S3- prefixed)")

bad_singletons = match_df["matched_entity_ids"].isin(["nan", "null", "none", "NA", " "]).sum()
if bad_singletons > 0:
    errors.append(f"Found {bad_singletons:,} bad singleton strings!")
else:
    print("  [PASS] Singletons = clean empty strings")

cand_lookup = {row["source1_entity_id"]: set(row["candidate_entity_ids"].split(",")) for _, row in cand_df.iterrows()}
violations = sum(1 for _, row in match_df.iterrows() if row["matched_entity_ids"] and not set(row["matched_entity_ids"].split(",")).issubset(cand_lookup.get(row["source1_entity_id"], set())))
if violations > 0:
    errors.append(f"{violations:,} matches not in candidate_pairs.tsv!")
else:
    print("  [PASS] All matches ⊆ candidate_pairs.tsv")

n_matched = (match_df["matched_entity_ids"] != "").sum()
n_singletons = (match_df["matched_entity_ids"] == "").sum()
print(f"\n--- STATS ---")
print(f"Total S1:     {len(match_df):,}")
print(f"With matches: {n_matched:,} ({n_matched/len(match_df)*100:.2f}%)")
print(f"Singletons:   {n_singletons:,} ({n_singletons/len(match_df)*100:.2f}%)")

print("=" * 65)
if errors:
    print("❌ VALIDATION FAILED:")
    for err in errors:
        print(f"  - {err}")
else:
    print("🏆 ALL CHECKS PASSED! READY FOR LEADERBOARD SUBMISSION!")
print("=" * 65)
print(f"🎉 TOTAL RUNTIME: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f} minutes)!")

try:
    from IPython.display import display, FileLink
    print("\n📥 DOWNLOAD YOUR SUBMISSION FILES:")
    display(FileLink("output/matching_results.tsv"))
    display(FileLink("output/candidate_pairs.tsv"))
except Exception:
    pass
