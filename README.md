# Amazon ML Challenge 2026 — Business Entity Resolution

This repository implements the end-to-end entity resolution pipeline for the Amazon ML Challenge 2026.

## Repository Layout

```text
├── files/                      # Problem statement, tickets PRD, and EDA report
│   ├── ps.pdf
│   ├── tickets.md
│   └── eda_report.txt
├── src/                        # Source code for the 4 parallel paths and shared logic
│   ├── common/                 # Shared data loaders, text normalizers, and feature utils
│   ├── path_a/                 # Path A: Classical ML (blocking + feature engineering + GBDT)
│   ├── path_b/                 # Path B: Neural retrieval & cross-encoder re-ranking
│   ├── path_c/                 # Path C: LLM as verifier on uncertain pairs
│   └── path_d/                 # Path D: Graph / collective ER & packaging
├── eval/                       # Evaluation scripts (macro F0.5 with singletons)
├── splits/                     # Stratified train/val and leave-one-country-out splits
├── scores/                     # Scored candidate pairs in parquet format
├── output/                     # Final deliverables (matching_results.tsv, candidate_pairs.tsv)
└── utils/                      # Helper scripts and validation utilities
```