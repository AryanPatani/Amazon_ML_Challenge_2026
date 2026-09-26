"""
Ticket A1: Blocking
Generates candidate pairs for Path A (Classical ML).
Union of multiple blocking strategies:
a) Char n-gram TF-IDF top-K by name
b) Token-overlap on rare name tokens
c) Address token/PIN/postcode blocks
d) Phonetic keys (Double Metaphone) on name tokens
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

from src.common.data_loader import load_all_sources
from src.common.normalize import normalize_name, normalize_address

def build_candidates_df(s2, s3):
    """Combine S2 and S3 into a single candidates DataFrame."""
    s2 = s2.copy()
    s2['source'] = 'S2'
    s3 = s3.copy()
    s3['source'] = 'S3'
    return pd.concat([s2, s3], ignore_index=True)

def run_blocking(s1, cands, k=20):
    print("Normalizing S1...")
    s1_names = s1['business_name'].apply(normalize_name)
    s1_addrs = s1['business_address'].apply(normalize_address)
    
    print("Normalizing Candidates...")
    cands_names = cands['business_name'].apply(normalize_name)
    cands_addrs = cands['business_address'].apply(normalize_address)
    
    # Store clean texts
    s1['name_clean'] = [x['name_clean'] for x in s1_names]
    cands['name_clean'] = [x['name_clean'] for x in cands_names]
    
    s1['name_tokens'] = [x['name_tokens'] for x in s1_names]
    cands['name_tokens'] = [x['name_tokens'] for x in cands_names]
    
    s1['pin_code'] = [x['pin_code'] for x in s1_addrs]
    cands['pin_code'] = [x['pin_code'] for x in cands_addrs]
    
    pairs = set()  # (s1_id, cand_id)
    
    s1_ids = s1['entity_id'].values
    cand_ids = cands['entity_id'].values
    
    # --- Strategy A: Char n-gram TF-IDF top-K by name ---
    print("Strategy A: Char n-gram TF-IDF...")
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2)
    # Fit on all names
    all_names = pd.concat([s1['name_clean'], cands['name_clean']])
    tfidf.fit(all_names)
    
    X_s1 = tfidf.transform(s1['name_clean'])
    X_cands = tfidf.transform(cands['name_clean'])
    
    # Use NearestNeighbors (cosine similarity)
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1)
    nn.fit(X_cands)
    
    # batch inference to avoid huge memory
    batch_size = 10000
    for i in tqdm(range(0, X_s1.shape[0], batch_size), desc="TF-IDF NN"):
        dist, ind = nn.kneighbors(X_s1[i:i+batch_size])
        for row_idx, neighbors in enumerate(ind):
            s1_i = s1_ids[i + row_idx]
            for n_idx in neighbors:
                pairs.add((s1_i, cand_ids[n_idx]))

    # --- Strategy B: Token overlap on rare name tokens ---
    print("Strategy B: Rare name tokens...")
    # Compute DF of name tokens
    token_df = defaultdict(int)
    for tokens in s1['name_tokens']:
        for t in set(tokens): token_df[t] += 1
    for tokens in cands['name_tokens']:
        for t in set(tokens): token_df[t] += 1
        
    rare_threshold = 50
    rare_tokens = {t for t, df in token_df.items() if df < rare_threshold}
    
    # Map rare token -> cand_ids
    token_to_cands = defaultdict(list)
    for c_id, tokens in zip(cand_ids, cands['name_tokens']):
        for t in set(tokens):
            if t in rare_tokens:
                token_to_cands[t].append(c_id)
                
    for s1_i, tokens in tqdm(zip(s1_ids, s1['name_tokens']), total=len(s1_ids), desc="Rare token matching"):
        for t in set(tokens):
            if t in rare_tokens:
                for c_id in token_to_cands[t]:
                    pairs.add((s1_i, c_id))

    # --- Strategy C: Address PIN code block ---
    print("Strategy C: PIN code...")
    pin_to_cands = defaultdict(list)
    for c_id, pin in zip(cand_ids, cands['pin_code']):
        if pin:
            pin_to_cands[pin].append(c_id)
            
    for s1_i, pin in tqdm(zip(s1_ids, s1['pin_code']), total=len(s1_ids), desc="PIN matching"):
        if pin:
            for c_id in pin_to_cands[pin]:
                pairs.add((s1_i, c_id))

    # --- Strategy D: Phonetic keys (Double Metaphone) ---
    print("Strategy D: Phonetic keys...")
    def get_phonetic_keys(tokens):
        keys = set()
        for t in tokens:
            if len(t) > 3: # Only for words > 3 chars to avoid noise
                try:
                    res = phonetics.dmetaphone(t)
                    if res:
                        for k in res:
                            if k: keys.add(k)
                except Exception:
                    pass
        return keys

    phonetic_to_cands = defaultdict(list)
    for c_id, tokens in tqdm(zip(cand_ids, cands['name_tokens']), total=len(cand_ids), desc="Cand phonetics"):
        for p in get_phonetic_keys(tokens):
            phonetic_to_cands[p].append(c_id)
            
    for s1_i, tokens in tqdm(zip(s1_ids, s1['name_tokens']), total=len(s1_ids), desc="S1 phonetics"):
        for p in get_phonetic_keys(tokens):
            for c_id in phonetic_to_cands[p]:
                pairs.add((s1_i, c_id))

    return list(pairs)

def evaluate_recall(pairs, gt):
    print("Evaluating recall...")
    pairs_set = set(pairs)
    
    total_true_pairs = 0
    found_true_pairs = 0
    
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        for match in row['matches']:
            total_true_pairs += 1
            if (s1_id, match) in pairs_set:
                found_true_pairs += 1
                
    recall = found_true_pairs / max(1, total_true_pairs)
    return recall, found_true_pairs, total_true_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20, help="Top-K for TF-IDF")
    args = parser.parse_args()
    
    print("Loading data...")
    s1, s2, s3, gt = load_all_sources()
    
    cands = build_candidates_df(s2, s3)
    
    print(f"S1 size: {len(s1)}")
    print(f"Candidates size: {len(cands)}")
    
    start = time.time()
    pairs = run_blocking(s1, cands, k=args.k)
    duration = time.time() - start
    
    print(f"Generated {len(pairs):,} candidate pairs in {duration:.1f}s.")
    print(f"Average candidates per S1: {len(pairs) / len(s1):.2f}")
    
    recall, found, total = evaluate_recall(pairs, gt)
    print(f"Candidate Recall: {recall*100:.2f}% ({found}/{total} true matches covered)")
    
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    df_pairs = pd.DataFrame(pairs, columns=['s1_id', 'cand_id'])
    out_path = out_dir / "a1_candidate_pairs.csv"
    df_pairs.to_csv(out_path, index=False)
    print(f"Saved pairs to {out_path}")

if __name__ == "__main__":
    main()
