"""
src/path_b/metrics.py

Ticket B1: Recall@K evaluation for blocking/retrieval.
Measures the candidate recall ceiling (e.g. Recall@20, Recall@50) against
the ground truth matches.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union
import numpy as np
import pandas as pd


def compute_recall_at_k(
    retrieved_candidates: dict[str, list[Union[str, tuple[str, float]]]],
    ground_truth: dict[str, list[str]],
    k_list: Sequence[int] = (10, 20, 50, 100),
) -> dict[str, Union[float, dict[int, float]]]:
    """Compute Recall@K metrics on entity matches.

    Parameters
    ----------
    retrieved_candidates : dict[str, list[str | tuple[str, float]]]
        Mapping: s1_id -> list of candidate entity IDs (or (id, score) tuples),
        ordered by relevance descending.
    ground_truth : dict[str, list[str]]
        Mapping: s1_id -> list of true matched entity IDs.
    k_list : Sequence[int], optional
        List of cutoffs K to evaluate, by default (10, 20, 50, 100).

    Returns
    -------
    dict
        Summary dictionary containing:
        - 'num_queries': Total S1 entities evaluated
        - 'num_non_singletons': Total S1 entities with >= 1 true match
        - 'num_singletons': Total S1 entities with 0 true matches
        - 'total_true_matches': Total individual true match links
        - 'micro_recall_at_k': dict {K: float}
        - 'macro_recall_at_k': dict {K: float}
        - 'all_hits_at_k': dict {K: float} (fraction where 100% of matches retrieved)
    """
    # Deduplicate and sort k_list
    k_list = sorted(set(k_list))
    k_max = max(k_list) if k_list else 0

    # Standardize retrieved_candidates to list of str
    clean_retrieved: dict[str, list[str]] = {}
    for s1_id, cands in retrieved_candidates.items():
        if cands and isinstance(cands[0], tuple):
            clean_retrieved[s1_id] = [c[0] for c in cands]
        else:
            clean_retrieved[s1_id] = list(cands)

    total_true_matches = 0
    non_singleton_count = 0
    singleton_count = 0

    # Per-K accumulators
    micro_hits = {k: 0 for k in k_list}
    macro_recalls = {k: [] for k in k_list}
    all_hits = {k: 0 for k in k_list}

    for s1_id, true_matches in ground_truth.items():
        if s1_id not in clean_retrieved:
            continue

        true_set = set(true_matches)
        n_true = len(true_set)

        if n_true == 0:
            singleton_count += 1
            continue

        non_singleton_count += 1
        total_true_matches += n_true

        cands = clean_retrieved[s1_id]

        for k in k_list:
            top_k_set = set(cands[:k])
            hits = len(true_set.intersection(top_k_set))
            micro_hits[k] += hits
            macro_recalls[k].append(hits / n_true)
            if hits == n_true:
                all_hits[k] += 1

    micro_recall = {
        k: (micro_hits[k] / total_true_matches if total_true_matches > 0 else 0.0)
        for k in k_list
    }
    macro_recall = {
        k: (float(np.mean(macro_recalls[k])) if macro_recalls[k] else 0.0)
        for k in k_list
    }
    all_hits_rate = {
        k: (all_hits[k] / non_singleton_count if non_singleton_count > 0 else 0.0)
        for k in k_list
    }

    return {
        "num_queries": len(clean_retrieved),
        "num_non_singletons": non_singleton_count,
        "num_singletons": singleton_count,
        "total_true_matches": total_true_matches,
        "micro_recall_at_k": micro_recall,
        "macro_recall_at_k": macro_recall,
        "all_hits_at_k": all_hits_rate,
    }


def print_recall_report(metrics: dict) -> None:
    """Print a clean ASCII summary table of Recall@K."""
    print("\n" + "=" * 65)
    print("        PATH B1: DENSE RETRIEVAL CANDIDATE RECALL REPORT       ")
    print("=" * 65)
    print(f"Total Evaluated Queries: {metrics['num_queries']:,}")
    print(f"Non-Singletons:          {metrics['num_non_singletons']:,}")
    print(f"Singletons:              {metrics['num_singletons']:,}")
    print(f"Total True Positive Links: {metrics['total_true_matches']:,}")
    print("-" * 65)
    print(f"{'K':>6} | {'Micro Recall':>14} | {'Macro Recall':>14} | {'100% Match Rate':>16}")
    print("-" * 65)

    micro = metrics["micro_recall_at_k"]
    macro = metrics["macro_recall_at_k"]
    all_h = metrics["all_hits_at_k"]

    for k in sorted(micro.keys()):
        print(
            f"{k:>6} | "
            f"{micro[k]*100:>13.2f}% | "
            f"{macro[k]*100:>13.2f}% | "
            f"{all_h[k]*100:>15.2f}%"
        )
    print("=" * 65 + "\n")
