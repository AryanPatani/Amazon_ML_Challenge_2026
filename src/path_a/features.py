"""
Ticket A2: Pair Features
Generates a feature table for pairs. Features include:
- String matching: Jaro-Winkler, Levenshtein ratio, token-set/sort ratio
- Set metrics: Jaccard on tokens, Jaccard on char 3-grams
- TF-IDF cosine similarity (name, address)
- Domain specific: first-token match, acronym match, numeric-token overlap
- Flags: PIN agreement, missing-field flags, length ratios
"""

import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.data_loader import load_all_sources
from src.common.normalize import normalize_name, normalize_address

# Safe import for rapidfuzz with pure-Python fallback
try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    import difflib

    class _Distance:
        class JaroWinkler:
            @staticmethod
            def normalized_similarity(s1, s2):
                if not s1 and not s2:
                    return 1.0
                if not s1 or not s2:
                    return 0.0
                return difflib.SequenceMatcher(None, str(s1), str(s2)).ratio()

        class Levenshtein:
            @staticmethod
            def normalized_similarity(s1, s2):
                if not s1 and not s2:
                    return 1.0
                if not s1 or not s2:
                    return 0.0
                return difflib.SequenceMatcher(None, str(s1), str(s2)).ratio()

    class _Fuzz:
        @staticmethod
        def token_set_ratio(s1, s2):
            s1, s2 = str(s1), str(s2)
            set1, set2 = set(s1.split()), set(s2.split())
            if not set1 and not set2:
                return 100.0
            inter = ' '.join(sorted(set1 & set2))
            diff1 = ' '.join(sorted(set1 - set2))
            diff2 = ' '.join(sorted(set2 - set1))
            s1_sort = (inter + ' ' + diff1).strip()
            s2_sort = (inter + ' ' + diff2).strip()
            candidates = [difflib.SequenceMatcher(None, s1_sort, s2_sort).ratio()]
            if inter:
                candidates.append(difflib.SequenceMatcher(None, inter, s1_sort).ratio())
                candidates.append(difflib.SequenceMatcher(None, inter, s2_sort).ratio())
            return max(candidates) * 100.0

        @staticmethod
        def token_sort_ratio(s1, s2):
            t1 = ' '.join(sorted(str(s1).split()))
            t2 = ' '.join(sorted(str(s2).split()))
            return difflib.SequenceMatcher(None, t1, t2).ratio() * 100.0

    distance = _Distance()
    fuzz = _Fuzz()


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
    print("Normalizing entity dictionaries...")
    # Pre-normalize and store as dicts for fast lookup
    s1_dict = {}
    for row in tqdm(s1_df.itertuples(index=False), total=len(s1_df), desc="Norm S1"):
        n = normalize_name(getattr(row, 'business_name', ''))
        a = normalize_address(getattr(row, 'business_address', ''))
        n.update(a)
        s1_dict[getattr(row, 'entity_id')] = n

    cands_dict = {}
    for row in tqdm(cands_df.itertuples(index=False), total=len(cands_df), desc="Norm Cands"):
        n = normalize_name(getattr(row, 'business_name', ''))
        a = normalize_address(getattr(row, 'business_address', ''))
        n.update(a)
        cands_dict[getattr(row, 'entity_id')] = n

    print("Fitting TF-IDF models...")
    all_names = [d['name_clean'] for d in s1_dict.values()] + [d['name_clean'] for d in cands_dict.values()]
    all_addrs = [d['address_clean'] for d in s1_dict.values()] + [d['address_clean'] for d in cands_dict.values()]

    tfidf_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    tfidf_name.fit(all_names if all_names else [""])

    tfidf_addr = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
    tfidf_addr.fit(all_addrs if all_addrs else [""])

    features = []

    print("Extracting features per pair...")
    s1_names_clean = []
    cand_names_clean = []
    s1_addrs_clean = []
    cand_addrs_clean = []

    s1_col = pairs_df['s1_id'].values
    cand_col = pairs_df['cand_id'].values

    for s1_id, cand_id in tqdm(zip(s1_col, cand_col), total=len(pairs_df), desc="Pair features"):
        d1 = s1_dict.get(s1_id, {})
        d2 = cands_dict.get(cand_id, {})

        n1 = d1.get('name_clean', '')
        n2 = d2.get('name_clean', '')
        a1 = d1.get('address_clean', '')
        a2 = d2.get('address_clean', '')

        s1_names_clean.append(n1)
        cand_names_clean.append(n2)
        s1_addrs_clean.append(a1)
        cand_addrs_clean.append(a2)

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
        acronym_match = 1 if (len(acronym1) >= 2 and len(acronym2) >= 2 and (acronym1 == acronym2 or acronym1 in n2 or acronym2 in n1)) else 0

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

    print("Computing TF-IDF cosine similarities in bulk (chunked)...")
    batch_size = 50_000
    sim_names = []
    sim_addrs = []
    for b_start in range(0, len(s1_names_clean), batch_size):
        b_end = b_start + batch_size
        v1_n = tfidf_name.transform(s1_names_clean[b_start:b_end])
        v2_n = tfidf_name.transform(cand_names_clean[b_start:b_end])
        sim_n = np.asarray(v1_n.multiply(v2_n).sum(axis=1)).ravel()
        sim_names.append(sim_n)

        v1_a = tfidf_addr.transform(s1_addrs_clean[b_start:b_end])
        v2_a = tfidf_addr.transform(cand_addrs_clean[b_start:b_end])
        sim_a = np.asarray(v1_a.multiply(v2_a).sum(axis=1)).ravel()
        sim_addrs.append(sim_a)

    feats_df = pd.DataFrame(features)
    if len(feats_df) > 0:
        feats_df['tfidf_sim_name'] = np.concatenate(sim_names) if sim_names else np.zeros(len(feats_df))
        feats_df['tfidf_sim_addr'] = np.concatenate(sim_addrs) if sim_addrs else np.zeros(len(feats_df))
    else:
        feats_df['tfidf_sim_name'] = []
        feats_df['tfidf_sim_addr'] = []

    return feats_df


def main():
    parser = argparse.ArgumentParser(description="Ticket A2: Pair Features")
    parser.add_argument("--pairs", type=Path, default="output/a1_candidate_pairs.csv", help="Input candidate pairs CSV")
    parser.add_argument("--out", type=Path, default="output/a2_features.parquet", help="Output features parquet")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional raw dataset path (train/ directory or parent)")
    args = parser.parse_args()

    if not args.pairs.exists():
        print(f"Error: {args.pairs} not found. Please run A1 blocking first.")
        return

    pairs_df = pd.read_csv(args.pairs)
    print(f"Loaded {len(pairs_df):,} pairs from {args.pairs}")

    if len(pairs_df) == 0:
        print("Warning: pairs_df is empty. Writing empty features dataframe.")
        pd.DataFrame().to_parquet(args.out, index=False)
        return

    # Normalize column names if needed
    col_rename = {'source1_entity_id': 's1_id', 'entity_id_1': 's1_id', 'entity_id_2': 'cand_id', 'candidate_id': 'cand_id'}
    pairs_df = pairs_df.rename(columns=col_rename)

    print("Loading raw data sources...")
    s1, s2, s3, gt = load_all_sources(train_dir=args.data_dir)
    cands_df = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=['entity_id'])

    # Filter sources down to just what's needed for the pairs to save memory/time
    needed_s1 = set(pairs_df['s1_id'].unique())
    needed_cands = set(pairs_df['cand_id'].unique())

    s1 = s1[s1['entity_id'].isin(needed_s1)]
    cands_df = cands_df[cands_df['entity_id'].isin(needed_cands)]

    start = time.time()
    feats_df = compute_features(pairs_df, s1, cands_df)
    print(f"Computed features in {time.time() - start:.1f}s")

    print("Merging ground truth to create 'is_match' label...")
    gt_pairs = set()
    for row in gt.itertuples(index=False):
        s1_id = getattr(row, 'source1_entity_id')
        matches = getattr(row, 'matches', [])
        for match in matches:
            gt_pairs.add((s1_id, match))

    feats_df['is_match'] = [1 if (s1_id, cand_id) in gt_pairs else 0 for s1_id, cand_id in zip(feats_df['s1_id'], feats_df['cand_id'])]

    args.out.parent.mkdir(exist_ok=True, parents=True)
    feats_df.to_parquet(args.out, index=False)
    print(f"Saved {len(feats_df):,} feature rows to {args.out}")


if __name__ == "__main__":
    main()
