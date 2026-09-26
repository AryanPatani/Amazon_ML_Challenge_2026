"""
Ticket A1: Blocking (FAST version)
Generates candidate pairs for Path A (Classical ML).
Union of multiple blocking strategies:
  a) Char n-gram TF-IDF top-K by name
  b) Token-overlap on rare name tokens
  c) Address token/PIN/postcode blocks
  d) Phonetic keys (Double Metaphone) on name tokens

Optimizations over original blocking.py:
  - Parallel normalization via joblib threads (n_jobs=-1)
  - Chunked batch processing for TF-IDF NN
"""

import time
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import phonetics
from tqdm import tqdm
from joblib import Parallel, delayed

from src.common.data_loader import load_all_sources
from src.common.normalize import normalize_name, normalize_address


def _norm_name_batch(names):
    return [normalize_name(n) for n in names]

def _norm_addr_batch(addrs):
    return [normalize_address(a) for a in addrs]

def parallel_normalize(series, func, n_jobs=-1, chunk_size=50_000, desc="Normalizing"):
    values = series.tolist()
    n = len(values)
    chunks = [values[i:i+chunk_size] for i in range(0, n, chunk_size)]
    print(f"  {desc}: {n:,} rows in {len(chunks)} chunks")
    results_nested = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(func)(chunk) for chunk in tqdm(chunks, desc=desc)
    )
    return [item for sublist in results_nested for item in sublist]

def build_candidates_df(s2, s3):
    s2 = s2.copy(); s2['source'] = 'S2'
    s3 = s3.copy(); s3['source'] = 'S3'
    return pd.concat([s2, s3], ignore_index=True)

def run_blocking(s1, cands, k=20, n_jobs=-1):
    print("\n[1/4] Normalizing names & addresses (parallel)...")
    t0 = time.time()

    s1_name_norms   = parallel_normalize(s1['business_name'],    _norm_name_batch, n_jobs=n_jobs, desc="S1 names")
    cand_name_norms = parallel_normalize(cands['business_name'],  _norm_name_batch, n_jobs=n_jobs, desc="Cand names")
    s1_addr_norms   = parallel_normalize(s1['business_address'],  _norm_addr_batch, n_jobs=n_jobs, desc="S1 addrs")
    cand_addr_norms = parallel_normalize(cands['business_address'],_norm_addr_batch, n_jobs=n_jobs, desc="Cand addrs")

    s1 = s1.copy()
    s1['name_clean']  = [d['name_clean']  for d in s1_name_norms]
    s1['name_tokens'] = [d['name_tokens'] for d in s1_name_norms]
    s1['pin_code']    = [d['pin_code']    for d in s1_addr_norms]

    cands = cands.copy()
    cands['name_clean']  = [d['name_clean']  for d in cand_name_norms]
    cands['name_tokens'] = [d['name_tokens'] for d in cand_name_norms]
    cands['pin_code']    = [d['pin_code']    for d in cand_addr_norms]

    print(f"  Normalization done in {time.time()-t0:.1f}s")

    pairs = set()
    s1_ids   = s1['entity_id'].values
    cand_ids = cands['entity_id'].values

    # Strategy A: TF-IDF NN
    print("\n[2/4] Strategy A: Char n-gram TF-IDF NN...")
    t0 = time.time()
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,3), min_df=2)
    tfidf.fit(pd.concat([s1['name_clean'], cands['name_clean']]))
    X_s1    = tfidf.transform(s1['name_clean'])
    X_cands = tfidf.transform(cands['name_clean'])
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=n_jobs, algorithm='brute')
    nn.fit(X_cands)
    batch_size = 5_000
    for i in tqdm(range(0, X_s1.shape[0], batch_size), desc="TF-IDF NN"):
        dist, ind = nn.kneighbors(X_s1[i:i+batch_size])
        for row_offset, neighbors in enumerate(ind):
            s1_i = s1_ids[i + row_offset]
            for n_idx in neighbors:
                pairs.add((s1_i, cand_ids[n_idx]))
    print(f"  Strategy A done in {time.time()-t0:.1f}s | pairs: {len(pairs):,}")

    # Strategy B: Rare token overlap
    print("\n[3/4] Strategy B: Rare name tokens...")
    t0 = time.time()
    token_df = defaultdict(int)
    for tokens in s1['name_tokens']:
        for t in set(tokens): token_df[t] += 1
    for tokens in cands['name_tokens']:
        for t in set(tokens): token_df[t] += 1
    rare_tokens = {t for t, df in token_df.items() if df < 50}
    print(f"  Rare tokens: {len(rare_tokens):,}")
    token_to_cands = defaultdict(list)
    for c_id, tokens in zip(cand_ids, cands['name_tokens']):
        for t in set(tokens):
            if t in rare_tokens: token_to_cands[t].append(c_id)
    for s1_i, tokens in tqdm(zip(s1_ids, s1['name_tokens']), total=len(s1_ids), desc="Rare-token"):
        for t in set(tokens):
            if t in rare_tokens:
                for c_id in token_to_cands[t]: pairs.add((s1_i, c_id))
    print(f"  Strategy B done in {time.time()-t0:.1f}s | pairs: {len(pairs):,}")

    # Strategy C: PIN code
    print("\n[4/4] Strategy C: PIN + Strategy D: Phonetics...")
    t0 = time.time()
    pin_to_cands = defaultdict(list)
    for c_id, pin in zip(cand_ids, cands['pin_code']):
        if pin: pin_to_cands[pin].append(c_id)
    for s1_i, pin in tqdm(zip(s1_ids, s1['pin_code']), total=len(s1_ids), desc="PIN"):
        if pin:
            for c_id in pin_to_cands[pin]: pairs.add((s1_i, c_id))
    print(f"  Strategy C done | pairs: {len(pairs):,}")

    # Strategy D: Phonetics
    def get_phonetic_keys(tokens):
        keys = set()
        for t in tokens:
            if len(t) > 3:
                try:
                    for k in phonetics.dmetaphone(t):
                        if k: keys.add(k)
                except Exception:
                    pass
        return keys

    phonetic_to_cands = defaultdict(list)
    for c_id, tokens in tqdm(zip(cand_ids, cands['name_tokens']), total=len(cand_ids), desc="Cand phonetics"):
        for p in get_phonetic_keys(tokens): phonetic_to_cands[p].append(c_id)
    for s1_i, tokens in tqdm(zip(s1_ids, s1['name_tokens']), total=len(s1_ids), desc="S1 phonetics"):
        for p in get_phonetic_keys(tokens):
            for c_id in phonetic_to_cands[p]: pairs.add((s1_i, c_id))
    print(f"  Strategies C+D done in {time.time()-t0:.1f}s | final pairs: {len(pairs):,}")
    return list(pairs)

def evaluate_recall(pairs, gt):
    print("\nEvaluating recall...")
    pairs_set = set(pairs)
    total, found = 0, 0
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        for match in row['matches']:
            total += 1
            if (s1_id, match) in pairs_set: found += 1
    return found / max(1, total), found, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",    type=int,  default=20,  help="Top-K for TF-IDF NN")
    parser.add_argument("--jobs", type=int,  default=-1,  help="n_jobs (-1=all cores)")
    parser.add_argument("--out",  type=Path, default="output/a1_candidate_pairs.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("A1 BLOCKING (fast parallelized)")
    print("=" * 60)
    t_total = time.time()

    print("\nLoading data...")
    s1, s2, s3, gt = load_all_sources()
    cands = build_candidates_df(s2, s3)
    print(f"S1: {len(s1):,}  |  Candidates: {len(cands):,}")

    pairs = run_blocking(s1, cands, k=args.k, n_jobs=args.jobs)
    elapsed = time.time() - t_total

    print(f"\nTotal: {len(pairs):,} pairs in {elapsed:.1f}s")
    print(f"Avg candidates/S1: {len(pairs)/len(s1):.2f}")

    recall, found, total = evaluate_recall(pairs, gt)
    print(f"Recall: {recall*100:.2f}% ({found:,}/{total:,})")

    Path("output").mkdir(exist_ok=True)
    pd.DataFrame(pairs, columns=['s1_id','cand_id']).to_csv(args.out, index=False)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
