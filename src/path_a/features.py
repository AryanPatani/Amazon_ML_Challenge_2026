"""
Ticket A2: Pair Features
Generates a feature table for pairs. Features include:
- String matching: Jaro-Winkler, Levenshtein ratio, token-set/sort ratio
- Set metrics: Jaccard on tokens, Jaccard on char 3-grams
- TF-IDF cosine similarity (name, address)
- Domain specific: first-token match, acronym match, numeric-token overlap
- Flags: PIN agreement, missing-field flags, length ratios
"""

import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common.data_loader import load_all_sources
from src.common.normalize import normalize_name, normalize_address

def compute_jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return inter / union if union > 0 else 0.0

def get_char_ngrams(text, n=3):
    if not text:
        return set()
    text = f" {text} "
    return set(text[i:i+n] for i in range(len(text)-n+1))

def compute_features(pairs_df, s1_df, cands_df):
    print("Normalizing dictionaries...")
    # Pre-normalize and store as dicts for fast lookup
    s1_dict = {}
    for _, row in tqdm(s1_df.iterrows(), total=len(s1_df), desc="Norm S1"):
        n = normalize_name(row['business_name'])
        a = normalize_address(row['business_address'])
        n.update(a)
        s1_dict[row['entity_id']] = n
        
    cands_dict = {}
    for _, row in tqdm(cands_df.iterrows(), total=len(cands_df), desc="Norm Cands"):
        n = normalize_name(row['business_name'])
        a = normalize_address(row['business_address'])
        n.update(a)
        cands_dict[row['entity_id']] = n

    print("Fitting TF-IDF models...")
    all_names = [d['name_clean'] for d in s1_dict.values()] + [d['name_clean'] for d in cands_dict.values()]
    all_addrs = [d['address_clean'] for d in s1_dict.values()] + [d['address_clean'] for d in cands_dict.values()]
    
    tfidf_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    tfidf_name.fit(all_names)
    
    tfidf_addr = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    tfidf_addr.fit(all_addrs)
    
    features = []
    
    print("Extracting features per pair...")
    # Prepare batch inference for tfidf
    s1_names_clean = []
    cand_names_clean = []
    s1_addrs_clean = []
    cand_addrs_clean = []
    
    for _, row in pairs_df.iterrows():
        s1_id = row['s1_id']
        cand_id = row['cand_id']
        
        d1 = s1_dict.get(s1_id, {})
        d2 = cands_dict.get(cand_id, {})
        
        s1_names_clean.append(d1.get('name_clean', ''))
        cand_names_clean.append(d2.get('name_clean', ''))
        s1_addrs_clean.append(d1.get('address_clean', ''))
        cand_addrs_clean.append(d2.get('address_clean', ''))
        
        n1 = d1.get('name_clean', '')
        n2 = d2.get('name_clean', '')
        a1 = d1.get('address_clean', '')
        a2 = d2.get('address_clean', '')
        
        t1_name = set(d1.get('name_tokens', []))
        t2_name = set(d2.get('name_tokens', []))
        t1_addr = set(d1.get('address_tokens', []))
        t2_addr = set(d2.get('address_tokens', []))
        
        # 1. String distances
        jw_name = distance.JaroWinkler.normalized_similarity(n1, n2)
        lev_name = distance.Levenshtein.normalized_similarity(n1, n2)
        ts_name = fuzz.token_set_ratio(n1, n2) / 100.0
        tsort_name = fuzz.token_sort_ratio(n1, n2) / 100.0
        
        jw_addr = distance.JaroWinkler.normalized_similarity(a1, a2)
        lev_addr = distance.Levenshtein.normalized_similarity(a1, a2)
        
        # 2. Set metrics
        jaccard_name_tok = compute_jaccard(t1_name, t2_name)
        jaccard_addr_tok = compute_jaccard(t1_addr, t2_addr)
        
        jaccard_name_3g = compute_jaccard(get_char_ngrams(n1, 3), get_char_ngrams(n2, 3))
        jaccard_addr_3g = compute_jaccard(get_char_ngrams(a1, 3), get_char_ngrams(a2, 3))
        
        # 3. Domain specific
        first_token_match = 1 if (d1.get('name_tokens') and d2.get('name_tokens') and d1['name_tokens'][0] == d2['name_tokens'][0]) else 0
        
        acronym1 = "".join(t[0] for t in d1.get('name_tokens', []) if t)
        acronym2 = "".join(t[0] for t in d2.get('name_tokens', []) if t)
        acronym_match = 1 if (acronym1 and acronym2 and (acronym1 == acronym2 or acronym1 in n2 or acronym2 in n1)) else 0
        
        num1 = set(d1.get('numeric_tokens', []))
        num2 = set(d2.get('numeric_tokens', []))
        num_overlap = 1 if (num1 and num2 and num1.intersection(num2)) else 0
        num_conflict = 1 if (num1 and num2 and not num1.intersection(num2)) else 0
        
        # 4. Flags and Lengths
        pin1 = d1.get('pin_code')
        pin2 = d2.get('pin_code')
        pin_agree = 1 if (pin1 and pin2 and pin1 == pin2) else 0
        pin_conflict = 1 if (pin1 and pin2 and pin1 != pin2) else 0
        
        missing_addr = 1 if (d1.get('is_missing') or d2.get('is_missing')) else 0
        
        len_ratio_name = min(len(n1), len(n2)) / max(len(n1), len(n2)) if max(len(n1), len(n2)) > 0 else 0
        len_ratio_addr = min(len(a1), len(a2)) / max(len(a1), len(a2)) if max(len(a1), len(a2)) > 0 else 0
        
        features.append({
            's1_id': s1_id,
            'cand_id': cand_id,
            'jw_name': jw_name,
            'lev_name': lev_name,
            'ts_name': ts_name,
            'tsort_name': tsort_name,
            'jw_addr': jw_addr,
            'lev_addr': lev_addr,
            'jaccard_name_tok': jaccard_name_tok,
            'jaccard_addr_tok': jaccard_addr_tok,
            'jaccard_name_3g': jaccard_name_3g,
            'jaccard_addr_3g': jaccard_addr_3g,
            'first_token_match': first_token_match,
            'acronym_match': acronym_match,
            'num_overlap': num_overlap,
            'num_conflict': num_conflict,
            'pin_agree': pin_agree,
            'pin_conflict': pin_conflict,
            'missing_addr': missing_addr,
            'len_ratio_name': len_ratio_name,
            'len_ratio_addr': len_ratio_addr
        })

    print("Computing TF-IDF cosine similarities in bulk...")
    # Batch tf-idf transform and sparse dot product
    v1_name = tfidf_name.transform(s1_names_clean)
    v2_name = tfidf_name.transform(cand_names_clean)
    tfidf_sim_name = v1_name.multiply(v2_name).sum(axis=1).A1
    
    v1_addr = tfidf_addr.transform(s1_addrs_clean)
    v2_addr = tfidf_addr.transform(cand_addrs_clean)
    tfidf_sim_addr = v1_addr.multiply(v2_addr).sum(axis=1).A1
    
    for i, f in enumerate(features):
        f['tfidf_sim_name'] = tfidf_sim_name[i]
        f['tfidf_sim_addr'] = tfidf_sim_addr[i]
        
    return pd.DataFrame(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default="output/a1_candidate_pairs.csv", help="Input candidate pairs CSV")
    parser.add_argument("--out", type=Path, default="output/a2_features.parquet", help="Output features parquet")
    args = parser.parse_args()
    
    if not args.pairs.exists():
        print(f"Error: {args.pairs} not found. Please run A1 blocking first.")
        return
        
    pairs_df = pd.read_csv(args.pairs)
    print(f"Loaded {len(pairs_df):,} pairs from {args.pairs}")
    
    print("Loading raw data sources...")
    s1, s2, s3, _ = load_all_sources()
    cands_df = pd.concat([s2, s3], ignore_index=True)
    
    # Filter sources down to just what's needed for the pairs to save memory/time
    needed_s1 = pairs_df['s1_id'].unique()
    needed_cands = pairs_df['cand_id'].unique()
    
    s1 = s1[s1['entity_id'].isin(needed_s1)]
    cands_df = cands_df[cands_df['entity_id'].isin(needed_cands)]
    
    start = time.time()
    feats_df = compute_features(pairs_df, s1, cands_df)
    print(f"Computed features in {time.time() - start:.1f}s")
    
    # Optional: Merge ground truth to create the target column `is_match` for A3/A4
    _, _, _, gt = load_all_sources()
    print("Merging ground truth to create 'is_match' label...")
    
    # Build a fast lookup for ground truth pairs
    gt_pairs = set()
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        for match in row['matches']:
            gt_pairs.add((s1_id, match))
            
    feats_df['is_match'] = feats_df.apply(lambda r: 1 if (r['s1_id'], r['cand_id']) in gt_pairs else 0, axis=1)
    
    args.out.parent.mkdir(exist_ok=True)
    feats_df.to_parquet(args.out, index=False)
    print(f"Saved {len(feats_df):,} feature rows to {args.out}")

if __name__ == "__main__":
    main()
