"""
Ticket D2: Unsupervised Graph Construction
=========================================================

Builds an undirected graph connecting S1 to Candidates (S2/S3) using D1 scores,
and adding Cand-Cand edges (S2-S3) to capture candidate agreement.

Inputs:
  scores/path_d1_val.parquet
Outputs:
  scores/path_d2_edges.parquet (edge list: node1, node2, weight)
"""

import argparse
import itertools
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from rapidfuzz import distance
from tqdm import tqdm

from src.common.data_loader import load_all_sources
from src.common.normalize import normalize_name, normalize_address


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1-scores", type=Path, default="scores/path_d1_val.parquet")
    parser.add_argument("--out-edges", type=Path, default="scores/path_d2_edges.parquet")
    parser.add_argument("--thresh-s1", type=float, default=0.10, help="Min score for S1-Cand edges")
    parser.add_argument("--thresh-cand", type=float, default=0.85, help="Min sim for Cand-Cand edges")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("D2: GRAPH CONSTRUCTION")
    print("=" * 60)

    # 1. Load D1 Edges
    print(f"\n[Load] Reading S1-Cand edges from {args.d1_scores}...")
    d1_df = pd.read_parquet(args.d1_scores)
    
    # Filter weak edges
    s1_edges = d1_df[d1_df['score'] >= args.thresh_s1].copy()
    print(f"  Retained {len(s1_edges):,} / {len(d1_df):,} S1-Cand edges (score >= {args.thresh_s1})")
    
    s1_edges = s1_edges.rename(columns={'s1_id': 'node1', 'cand_id': 'node2', 'score': 'weight'})
    
    # 2. Get unique candidates needed
    unique_cands = set(s1_edges['node2'])
    print(f"\n[Candidates] Extracting {len(unique_cands):,} unique candidates for Cand-Cand comparison...")
    
    _, s2, s3, _ = load_all_sources()
    cands_df = pd.concat([s2, s3], ignore_index=True)
    cands_df = cands_df[cands_df['entity_id'].isin(unique_cands)].reset_index(drop=True)
    
    print("  Normalising candidate strings...")
    # Fast list comprehension normalization
    name_norms = [normalize_name(n)['name_clean'] for n in tqdm(cands_df['business_name'], desc="  names")]
    addr_norms = [normalize_address(a)['address_clean'] for a in tqdm(cands_df['business_address'], desc="  addrs")]
    
    cands_df['name_clean'] = name_norms
    cands_df['addr_clean'] = addr_norms
    
    cand_dict = cands_df.set_index('entity_id')[['name_clean', 'addr_clean']].to_dict('index')
    
    # 3. Cand-Cand Edges (Local cohorts)
    print("\n[Cand-Cand Edges] Comparing candidates within S1 cohorts...")
    
    # Group candidates by S1 parent to avoid N^2 comparisons
    s1_to_cands = s1_edges.groupby('node1')['node2'].apply(list).to_dict()
    
    cand_pairs = set()
    for cands in tqdm(s1_to_cands.values(), desc="  Cohorts"):
        if len(cands) < 2:
            continue
        # Compare all pairs within this S1's cohort
        for c1, c2 in itertools.combinations(cands, 2):
            # Sort to avoid (A,B) and (B,A) duplicates
            if c1 > c2:
                c1, c2 = c2, c1
            cand_pairs.add((c1, c2))
            
    print(f"  Unique Cand-Cand pairs to compare: {len(cand_pairs):,}")
    
    cc_rows = []
    for c1, c2 in tqdm(cand_pairs, desc="  Scoring"):
        d1 = cand_dict.get(c1)
        d2 = cand_dict.get(c2)
        if not d1 or not d2:
            continue
            
        jw_name = distance.JaroWinkler.normalized_similarity(d1['name_clean'], d2['name_clean'])
        
        # Fast fail: if name is completely different, skip address compute
        if jw_name < 0.60:
            continue
            
        jw_addr = distance.JaroWinkler.normalized_similarity(d1['addr_clean'], d2['addr_clean'])
        
        # Simple weighted similarity
        sim = (jw_name * 0.6) + (jw_addr * 0.4)
        
        if sim >= args.thresh_cand:
            cc_rows.append({'node1': c1, 'node2': c2, 'weight': sim})
            
    cc_edges = pd.DataFrame(cc_rows) if cc_rows else pd.DataFrame(columns=['node1', 'node2', 'weight'])
    print(f"  Generated {len(cc_edges):,} Cand-Cand edges (sim >= {args.thresh_cand})")
    
    # 4. Combine and Build Graph
    all_edges = pd.concat([s1_edges, cc_edges], ignore_index=True)
    
    print("\n[Graph] Building NetworkX graph...")
    G = nx.from_pandas_edgelist(all_edges, source='node1', target='node2', edge_attr='weight')
    
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Connected Components: {nx.number_connected_components(G):,}")
    
    # 5. Save
    print(f"\n[Save] Writing edge list to {args.out_edges}...")
    args.out_edges.parent.mkdir(exist_ok=True, parents=True)
    all_edges.to_parquet(args.out_edges, index=False)
    
    print(f"\n{'='*60}")
    print(f"D2 complete in {time.time()-t_start:.1f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
