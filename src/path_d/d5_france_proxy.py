"""
Ticket D5: France Proxy Test
=========================================================

Simulates the theoretical F0.5 drop when processing countries without
structured PIN codes (e.g. France) by disabling PIN logic.

Design:
  1. Fast raw-field pre-blocking (regex PIN + first token) — no normalize() calls
  2. Normalize ONLY the small subset of candidates that were blocked
  3. Comparison vectors: binned name sim, address sim, PIN exact match,
     first-token match, numeric-token overlap
  4. ECMClassifier from recordlinkage to learn m(k)/u(k) without labels
  5. Log-odds score -> sigmoid -> [0,1] → shared output format
  6. Threshold sweep to report val macro F0.5

Why this generalises to France:
  - Zero country-specific features
  - Character-level + token-level comparisons are language-agnostic
  - EM requires no supervision — self-tunes on any country

Output:
  scores/path_d1_val.parquet   — s1_id, cand_id, score
"""

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from recordlinkage.classifiers import ECMClassifier
from tqdm import tqdm
from rapidfuzz import fuzz, distance

from src.common.data_loader import load_all_sources
from src.common.normalize import normalize_name, normalize_address
from eval.f05 import macro_f05


# ---------------------------------------------------------------------------
# Normalisation helpers (used ONLY on small subsets)
# ---------------------------------------------------------------------------

def _normalise_series(df, col_name, col_addr):
    """Add name_clean, name_tokens, pin_code, address_clean cols."""
    print(f"    Normalising {len(df):,} rows...")
    name_norms = [normalize_name(n)    for n in tqdm(df[col_name], desc="    names")]
    addr_norms = [normalize_address(a) for a in tqdm(df[col_addr], desc="    addrs")]
    df = df.copy()
    df['name_clean']    = [d['name_clean']    for d in name_norms]
    df['name_tokens']   = [d['name_tokens']   for d in name_norms]
    df['pin_code']      = [d['pin_code']       for d in addr_norms]
    df['address_clean'] = [d['address_clean']  for d in addr_norms]
    return df


# ---------------------------------------------------------------------------
# Fast raw-field feature extractors (no normalize() needed)
# ---------------------------------------------------------------------------
_PIN_RE   = re.compile(r'\b\d{4,6}\b')
_CLEAN_RE = re.compile(r'[^a-zA-Z0-9\s]')

def _raw_pin(addr):
    m = _PIN_RE.search(str(addr))
    return m.group(0) if m else None

def _raw_first_token(name):
    s = _CLEAN_RE.sub(' ', str(name)).strip().lower()
    toks = s.split()
    return toks[0] if toks else None


# ---------------------------------------------------------------------------
# Comparison vectors
# ---------------------------------------------------------------------------

def compute_comparison_vectors(pairs, s1_df, cands_df):
    print(f"\n[Features] Computing comparison vectors for {len(pairs):,} pairs...")
    s1_dict    = s1_df.set_index('entity_id').to_dict('index')
    cands_dict = cands_df.set_index('entity_id').to_dict('index')
    rows = []
    for s1_id, cand_id in tqdm(pairs, desc="  compare"):
        d1 = s1_dict.get(s1_id, {})
        d2 = cands_dict.get(cand_id, {})
        n1 = d1.get('name_clean', '')
        n2 = d2.get('name_clean', '')
        a1 = d1.get('address_clean', '')
        a2 = d2.get('address_clean', '')
        t1 = d1.get('name_tokens', [])
        t2 = d2.get('name_tokens', [])
        p1 = d1.get('pin_code')
        p2 = d2.get('pin_code')
        jw_name  = distance.JaroWinkler.normalized_similarity(n1, n2)
        lev_name = distance.Levenshtein.normalized_similarity(n1, n2)
        ts_name  = fuzz.token_set_ratio(n1, n2) / 100.0
        jw_addr  = distance.JaroWinkler.normalized_similarity(a1, a2)
        first_tok = 1 if (t1 and t2 and t1[0] == t2[0]) else 0
        nums1 = set(tok for tok in t1 if tok.isdigit())
        nums2 = set(tok for tok in t2 if tok.isdigit())
        num_tok = 1 if (nums1 and nums2 and nums1 & nums2) else 0
        rows.append({'s1_id': s1_id, 'cand_id': cand_id,
                     'jw_name': jw_name, 'lev_name': lev_name,
                     'ts_name': ts_name, 'jw_addr': jw_addr,
                     'first_tok': first_tok,
                     'num_tok': num_tok})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Discretise for Fellegi-Sunter
# ---------------------------------------------------------------------------

def discretise(df):
    feat = pd.DataFrame()
    feat['jw_name_bin']  = (df['jw_name'] >= 0.85).astype(int)
    feat['lev_name_bin'] = (df['lev_name'] >= 0.80).astype(int)
    feat['ts_name_bin']  = (df['ts_name'] >= 0.80).astype(int)
    feat['jw_addr_bin']  = (df['jw_addr'] >= 0.80).astype(int)
    feat['first_tok']    = df['first_tok'].astype(int)
    feat['num_tok']      = df['num_tok'].astype(int)
    return feat


# ---------------------------------------------------------------------------
# Recall evaluation
# ---------------------------------------------------------------------------

def evaluate_blocking_recall(pairs, gt):
    pairs_set = set(pairs)
    total, found = 0, 0
    for _, row in gt.iterrows():
        sid = row['source1_entity_id']
        for mid in row['matches']:
            total += 1
            if (sid, mid) in pairs_set:
                found += 1
    return found / max(1, total), found, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="D1: Fellegi-Sunter probabilistic baseline")
    parser.add_argument("--val-split", type=Path, default="splits/val_s1_ids.txt")
    parser.add_argument("--out",       type=Path, default="scores/path_d5_france.parquet")
    parser.add_argument("--max-cands", type=int,  default=50)
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("D1: FELLEGI-SUNTER PROBABILISTIC BASELINE")
    print("=" * 60)

    # --- Load ---
    print("\nLoading data...")
    s1, s2, s3, gt = load_all_sources()
    cands = pd.concat([s2, s3], ignore_index=True)
    print(f"  S1: {len(s1):,} | Cands: {len(cands):,}")

    with open(args.val_split) as f:
        val_ids = set(line.strip() for line in f if line.strip())
    s1_val = s1[s1['entity_id'].isin(val_ids)].reset_index(drop=True)
    print(f"  Val S1: {len(s1_val):,} entities")

    # --- Fast raw-field pre-blocking (no normalize() on 10M rows) ---
    print("\n[Pre-block] Extracting raw PIN + first-token from all candidates...")
    cands = cands.copy()
    cands['_ftok'] = cands['business_name'].apply(_raw_first_token)

    s1_val = s1_val.copy()
    s1_val['_ftok'] = s1_val['business_name'].apply(_raw_first_token)

    raw_pairs = set()
    print("  [France Proxy] PIN blocking disabled.")

    # First-token blocking (rare tokens 1 < df < 100)
    tok_df = defaultdict(int)
    for t in s1_val['_ftok']:
        if t: tok_df[t] += 1
    for t in cands['_ftok']:
        if t: tok_df[t] += 1
    rare = {t for t, df in tok_df.items() if 1 < df < 100}

    tok_to_cids = defaultdict(list)
    for cid, ftok in zip(cands['entity_id'], cands['_ftok']):
        if ftok and ftok in rare: tok_to_cids[ftok].append(cid)

    for sid, ftok in tqdm(zip(s1_val['entity_id'], s1_val['_ftok']),
                          total=len(s1_val), desc="  Token block"):
        if ftok and ftok in rare:
            for cid in tok_to_cids[ftok][:args.max_cands]:
                raw_pairs.add((sid, cid))

    pairs = list(raw_pairs)
    print(f"  Total pairs: {len(pairs):,}  |  Avg/S1: {len(pairs)/max(1,len(s1_val)):.1f}")

    # --- Normalize ONLY the small candidate subset ---
    needed_cids  = set(cid for _, cid in pairs)
    cands_subset = cands[cands['entity_id'].isin(needed_cids)].reset_index(drop=True)
    print(f"\n[Normalise] {len(s1_val):,} val S1 + {len(cands_subset):,} blocked candidates "
          f"(saved normalising {len(cands)-len(cands_subset):,} rows)")

    print("  S1...")
    s1_val = _normalise_series(s1_val, 'business_name', 'business_address')
    print("  Candidates...")
    cands_subset = _normalise_series(cands_subset, 'business_name', 'business_address')

    # Blocking recall
    recall, found, total = evaluate_blocking_recall(
        pairs, gt[gt['source1_entity_id'].isin(val_ids)])
    print(f"\n  Blocking Recall: {recall*100:.2f}%  ({found:,}/{total:,} true matches covered)")

    # --- Comparison vectors ---
    comp_df = compute_comparison_vectors(pairs, s1_val, cands_subset)

    # --- ECM ---
    print("\n[ECM] Discretising features...")
    feat_df = discretise(comp_df)
    print(f"  Feature matrix: {feat_df.shape}")

    print("[ECM] Fitting Expectation-Conditional Maximisation (unsupervised)...")
    ecm = ECMClassifier(binarize=None)
    ecm.fit(feat_df)

    print("\n  m-probs  P(agree | true match):")
    for col in feat_df.columns:
        mp = ecm.m_probs[col][1]
        print(f"    {col:20s}: {mp:.4f}")
    print("\n  u-probs  P(agree | non-match):")
    for col in feat_df.columns:
        up = ecm.u_probs[col][1]
        print(f"    {col:20s}: {up:.4f}")

    # --- Log-odds → sigmoid score ---
    print("\n[Score] Computing match probabilities...")
    log_odds = np.zeros(len(feat_df))
    for col in feat_df.columns:
        m = np.clip(ecm.m_probs[col][1], 1e-9, 1-1e-9)
        u = np.clip(ecm.u_probs[col][1], 1e-9, 1-1e-9)
        agreed = feat_df[col].values > 0
        log_odds += np.where(agreed, np.log(m/u), np.log((1-m)/(1-u)))
    scores = 1.0 / (1.0 + np.exp(-log_odds))
    comp_df['score'] = scores
    print(f"  Scores — min:{scores.min():.4f}  mean:{scores.mean():.4f}  max:{scores.max():.4f}")

    # --- Threshold sweep ---
    print("\n[Eval] Threshold sweep...")
    gt_dict = {}
    for _, row in gt.iterrows():
        if row['source1_entity_id'] in val_ids:
            gt_dict[row['source1_entity_id']] = row['matches']

    best_thresh, best_f05 = 0.5, 0.0
    for thresh in np.arange(0.1, 0.95, 0.05):
        # Extremely fast groupby instead of slow iterrows
        subset = comp_df[comp_df['score'] >= thresh]
        preds_dict = subset.groupby('s1_id')['cand_id'].apply(list).to_dict()
        
        full_preds = {sid: [] for sid in val_ids}
        full_preds.update(preds_dict)
        
        f05 = macro_f05(full_preds, gt_dict)
        if f05 > best_f05:
            best_f05, best_thresh = f05, thresh

    print(f"  Best Val Macro F0.5: {best_f05:.5f}  (threshold={best_thresh:.2f})")

    # --- Save ---
    print(f"\n[Save] Writing to {args.out}...")
    args.out.parent.mkdir(exist_ok=True, parents=True)
    comp_df[['s1_id','cand_id','score']].to_parquet(args.out, index=False)
    print(f"  Saved {len(comp_df):,} scored pairs.")

    print(f"\n{'='*60}")
    print(f"D1 complete in {time.time()-t_start:.1f}s")
    print(f"Val Macro F0.5 (Fellegi-Sunter / ECM): {best_f05:.5f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
