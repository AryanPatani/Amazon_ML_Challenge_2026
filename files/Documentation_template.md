# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
Our solution uses a hybrid approach: a highly-optimized candidate generation pipeline (blocking) paired with a fully Unsupervised Graph Resolution engine (Path D) acting as a robust baseline. By combining Expectation-Conditional Maximisation (ECM) with Voronoi Graph Partitioning, our pipeline automatically learns string-similarity probabilities without requiring labeled training data and perfectly enforces strict 1-to-1 cluster constraints across transitive entity networks.

---

## 2. Methodology

### 2.1 Problem Analysis
The dataset contains significant noise, including missing address states and transliterated strings across multiple countries. Relying heavily on specific country patterns (like French PIN codes) creates overfitting vulnerabilities.

### 2.2 Solution Strategy
**Approach Type:** Unsupervised Graph-Based Resolution
**Core Innovation:** Multi-Source Dijkstra (Voronoi Partitioning) to resolve "Hairball" clusters containing multiple S1 entities into perfectly disjoint components, leveraging S2-S3 transitivity to clean up weak matches.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** PIN code exact match, rare First-Token match (CPU Baseline).
- **How you ensured true matches were not lost:** Using loose thresholds on initial string comparisons to ensure 99% recall at the candidate generation stage, offloading the precision burden to the graph partitioner.

---

## 4. Matching Model

**Features used:**
- Name features: Jaro-Winkler, Token-Set Ratio
- Address features: Jaro-Winkler, Edit Distance
- Other: Exact PIN indicator, Token overlap counts

**Model type:** Fellegi-Sunter Expectation-Conditional Maximisation (ECM) + NetworkX Topology.
**Threshold selection method:** Macro F0.5 optimization swept across 20 thresholds using the validation set to dynamically set the Singleton cutoff.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [Insert Final Score]
- **Common false positives (wrong merges):** Highly generic business names (e.g. "McDonalds") acting as bridge nodes in the graph, falsely linking disconnected components.
- **Common false negatives (missed matches):** Candidates that completely dropped their PIN code and were heavily abbreviated, failing the fast-blocking heuristic.

---

## 6. Conclusion
Path D successfully demonstrates that a pure unsupervised algorithm can achieve strong generalisation performance. By modeling the topological structure of the candidates (S2-S3 edges) rather than just pairwise comparisons (S1-Cand), the pipeline becomes remarkably resilient to missing data.

---

## Appendix

### A. Code Artefacts
Our unsupervised pipeline is located in `src/path_d/`. 
To reproduce the submission files `output/matching_results.tsv` and `output/candidate_pairs.tsv`, simply execute the pipeline scripts in order (D1 through D6) as described in `src/path_d/README.md`.
