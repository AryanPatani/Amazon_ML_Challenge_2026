# Path D: Unsupervised Graph Resolution

This directory contains the robust, purely unsupervised fallback pipeline (Path D) for the ML Challenge 2026.

## Overview
Unlike Path A (Supervised Tree-based) or Path B/C (Embeddings/LLM), Path D relies on the topological structure of the data. By using the Fellegi-Sunter model (Expectation-Conditional Maximisation) and NetworkX graph clustering, we achieve an entirely unsupervised baseline that does not require labeled data or country-specific feature engineering.

## Architecture
1. **`d1_fellegi_sunter.py`**: Computes comparison vectors (Jaro-Winkler, Token-Set) and fits an ECM classifier to estimate match/unmatch probabilities iteratively.
2. **`d2_graph_construction.py`**: Builds an undirected graph connecting S1 to Candidates (S2/S3) and importantly adds Cand-Cand transitive edges to capture structural agreement.
3. **`d3_collective_decision.py`**: Uses Multi-Source Dijkstra (Voronoi partitioning) to slice through "hairball" clusters, perfectly enforcing the one-S1-per-cluster constraint.
4. **`d4_singleton_detector.py`**: A margin-based heuristic that drops weak clusters (false positives) to recover precision and confidently predict Singletons.
5. **`d6_package_submission.py`**: Formats the output of D1 and D4 into the official `candidate_pairs.tsv` and `matching_results.tsv`.

## How to Run
Ensure you have the required dependencies (pandas, rapidfuzz, networkx, recordlinkage).

```bash
# 1. Run the Unsupervised Baseline
python -m src.path_d.d1_fellegi_sunter

# 2. Build the Topology Graph
python -m src.path_d.d2_graph_construction

# 3. Resolve Graph Conflicts (Voronoi)
python -m src.path_d.d3_collective_decision

# 4. Prune Weak Clusters (Singletons)
python -m src.path_d.d4_singleton_detector

# 5. Export TSVs
python -m src.path_d.d6_package_submission
```
