"""
Ticket D3: Collective Decision & Graph Resolution
=========================================================

Resolves the D2 graph into disjoint clusters containing exactly ONE S1 entity.
Uses Multi-Source Dijkstra (Voronoi partitioning) on edge weights to break hairballs
and assign candidates to the closest S1 node transitively.

Inputs:
  scores/path_d2_edges.parquet
Outputs:
  scores/path_d3_resolved.parquet
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

from src.common.data_loader import load_all_sources
from eval.f05 import macro_f05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2-edges", type=Path, default="scores/path_d2_edges.parquet")
    parser.add_argument("--out", type=Path, default="scores/path_d3_resolved.parquet")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("D3: COLLECTIVE DECISION (GRAPH RESOLUTION)")
    print("=" * 60)

    # 1. Load S1 catalog to know which nodes are S1
    print("\n[Data] Loading S1 catalog to identify S1 nodes...")
    s1, _, _, gt = load_all_sources()
    val_s1_ids = set(gt['source1_entity_id'].unique())
    all_s1_ids = set(s1['entity_id'].unique())
    
    # 2. Load Graph
    print(f"\n[Load] Reading D2 edges from {args.d2_edges}...")
    edges_df = pd.read_parquet(args.d2_edges)
    
    print("  Adding inverse weights for shortest-path distances...")
    # Distance = 1.0 / weight. Strong edges (0.9) have short distance (1.11)
    # Weak edges (0.1) have long distance (10.0)
    edges_df['inv_weight'] = 1.0 / (edges_df['weight'] + 1e-9)
    
    print("  Building NetworkX graph...")
    G = nx.from_pandas_edgelist(edges_df, source='node1', target='node2', edge_attr=['weight', 'inv_weight'])
    print(f"    Nodes: {G.number_of_nodes():,}")
    print(f"    Edges: {G.number_of_edges():,}")
    
    # 3. Resolve Components
    print("\n[Resolve] Partitioning components via Multi-Source Dijkstra (Voronoi)...")
    components = list(nx.connected_components(G))
    print(f"  Found {len(components):,} connected components.")
    
    preds = defaultdict(list)
    
    clean_count = 0
    hairball_count = 0
    dead_count = 0
    
    for c in tqdm(components, desc="  Resolving"):
        c_s1_nodes = [n for n in c if n in all_s1_ids]
        
        if len(c_s1_nodes) == 0:
            dead_count += 1
            continue
            
        elif len(c_s1_nodes) == 1:
            clean_count += 1
            s1_node = c_s1_nodes[0]
            for n in c:
                if n != s1_node:
                    preds[s1_node].append(n)
                    
        else:
            hairball_count += 1
            subgraph = G.subgraph(c)
            # Find the closest S1 node for every node in the hairball
            distances, paths = nx.multi_source_dijkstra(subgraph, c_s1_nodes, weight='inv_weight')
            
            for node, path in paths.items():
                if node not in all_s1_ids:
                    s1_source = path[0]  # The S1 node that claimed this candidate
                    preds[s1_source].append(node)
                    
    print("\n[Stats] Resolution complete:")
    print(f"  Clean clusters (1 S1) : {clean_count:,}")
    print(f"  Hairballs (>1 S1)     : {hairball_count:,} (partitioned successfully)")
    print(f"  Dead clusters (0 S1)  : {dead_count:,} (discarded)")
    
    # 4. Evaluation
    print("\n[Eval] Calculating Macro F0.5 on Validation Set...")
    gt_dict = {}
    for _, row in gt.iterrows():
        if row['source1_entity_id'] in val_s1_ids:
            gt_dict[row['source1_entity_id']] = row['matches']
            
    # Ensure all validation S1s are in preds, even if empty
    full_preds = {sid: [] for sid in val_s1_ids}
    # Only update with preds for validation IDs (ignore test IDs for now)
    for sid, cands in preds.items():
        if sid in val_s1_ids:
            full_preds[sid] = cands
            
    f05 = macro_f05(full_preds, gt_dict)
    print(f"  Graph-Resolved Val Macro F0.5: {f05:.5f}")
    
    # 5. Save
    print(f"\n[Save] Writing resolved predictions to {args.out}...")
    
    # Convert dict to flat dataframe for saving
    out_rows = []
    for sid, cands in preds.items():
        for cand in cands:
            out_rows.append({'s1_id': sid, 'cand_id': cand})
            
    out_df = pd.DataFrame(out_rows) if out_rows else pd.DataFrame(columns=['s1_id', 'cand_id'])
    args.out.parent.mkdir(exist_ok=True, parents=True)
    out_df.to_parquet(args.out, index=False)
    
    print(f"\n{'='*60}")
    print(f"D3 complete in {time.time()-t_start:.1f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
