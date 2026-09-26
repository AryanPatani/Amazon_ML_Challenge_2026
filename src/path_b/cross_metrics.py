"""
src/path_b/cross_metrics.py

Ticket B3: Cross-Encoder evaluation metrics, threshold tuning for Macro F0.5,
1-to-1 global assignment constraint, and shared scores parquet export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from eval.f05 import macro_f05, compute_f05
from src.common.paths import SCORES_DIR, OUTPUT_DIR


def apply_assignment_constraint(
    predictions: dict[str, list[tuple[str, float]]],
    threshold: float,
    min_top_score: Optional[float] = None,
) -> dict[str, list[str]]:
    """Enforce the global 1-to-1 assignment constraint on thresholded predictions.

    Because Source 1 is deduplicated, each S2/S3 candidate record can belong
    to at most one Source 1 entity. If a candidate is above threshold for multiple
    S1 entities, it is assigned exclusively to the S1 entity with the highest score.

    Optionally, if min_top_score is set, an S1 entity must have at least one
    candidate with score >= min_top_score to trigger candidate matching; otherwise
    it is treated as a singleton (abstaining with empty list).

    Parameters
    ----------
    predictions : dict[str, list[tuple[str, float]]]
        Mapping s1_id -> list of (cand_id, score) pairs.
    threshold : float
        Decision threshold for candidate inclusion.
    min_top_score : Optional[float], optional
        Minimum top candidate score required to predict any matches for an S1 entity.

    Returns
    -------
    dict[str, list[str]]
        Mapping s1_id -> list of assigned candidate IDs, sorted descending by score.
    """
    # 1. Filter out anchors whose top candidate score is below min_top_score
    active_s1_set = set(predictions.keys())
    if min_top_score is not None:
        active_s1_set = set()
        for s1_id, cands in predictions.items():
            scores = [float(item[1]) if isinstance(item, tuple) else 1.0 for item in cands]
            max_s = max(scores, default=0.0)
            if max_s >= min_top_score:
                active_s1_set.add(s1_id)

    # 2. Collect all candidate sightings above threshold from active S1 entities
    cand_assignments: dict[str, list[tuple[str, float]]] = {}
    for s1_id in active_s1_set:
        cands = predictions.get(s1_id, [])
        for item in cands:
            cand_id = item[0] if isinstance(item, tuple) else item
            score = float(item[1]) if isinstance(item, tuple) else 1.0
            if score >= threshold:
                if cand_id not in cand_assignments:
                    cand_assignments[cand_id] = []
                cand_assignments[cand_id].append((s1_id, score))

    # 3. For each candidate, find the winning S1 entity (highest score)
    best_assignment: dict[str, tuple[str, float]] = {}
    for cand_id, sightings in cand_assignments.items():
        sightings.sort(key=lambda x: x[1], reverse=True)
        winning_s1, win_score = sightings[0]
        best_assignment[cand_id] = (winning_s1, win_score)

    # 4. Build final predictions dict, sorted descending by score per S1 entity
    s1_assigned: dict[str, list[tuple[str, float]]] = {s1_id: [] for s1_id in predictions}
    for cand_id, (winning_s1, win_score) in best_assignment.items():
        s1_assigned[winning_s1].append((cand_id, win_score))

    final_preds: dict[str, list[str]] = {}
    for s1_id, scored_list in s1_assigned.items():
        scored_list.sort(key=lambda x: x[1], reverse=True)
        final_preds[s1_id] = [cid for cid, _ in scored_list]

    return final_preds


def compute_roc_pr_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> dict[str, float]:
    """Compute ROC-AUC and average precision (PR-AUC) with sklearn / fallback."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"roc_auc": 0.0, "pr_auc": 0.0}

    # Try standard sklearn metrics
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc = float(roc_auc_score(y_true, y_scores))
        pr = float(average_precision_score(y_true, y_scores))
        return {"roc_auc": roc, "pr_auc": pr}
    except Exception:
        pass

    # Vectorized fallback
    order = np.argsort(-y_scores)
    y_true_sorted = y_true[order]
    y_scores_sorted = y_scores[order]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = len(y_true_sorted) - n_pos

    # ROC AUC via Mann-Whitney U statistic
    ranks = np.arange(len(y_scores_sorted), 0, -1)
    sum_ranks_pos = np.sum(ranks[y_true_sorted == 1])
    roc_auc = float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    # PR-AUC / Average Precision via trapezoidal integration
    cum_tp = np.cumsum(y_true_sorted == 1)
    cum_fp = np.cumsum(y_true_sorted == 0)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / n_pos

    pr_auc = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def find_optimal_threshold(
    scored_candidates: dict[str, list[tuple[str, float]]],
    ground_truth: dict[str, list[str]],
    threshold_range: Optional[list[float]] = None,
    min_top_score_offsets: Optional[list[float]] = None,
    enforce_one_to_one: bool = True,
) -> dict:
    """Sweep decision thresholds to maximize macro F0.5 on validation set.

    Parameters
    ----------
    scored_candidates : dict[str, list[tuple[str, float]]]
        Mapping s1_id -> list of (cand_id, probability) pairs.
    ground_truth : dict[str, list[str]]
        Ground truth mapping s1_id -> list of true match IDs.
    threshold_range : Optional[list[float]], optional
        List of thresholds to evaluate, by default [0.05, 0.10, ..., 0.95].
    min_top_score_offsets : Optional[list[float]], optional
        Offsets added to threshold for singleton trigger filter (default: [0.0, 0.05, 0.10, 0.15, 0.20]).
    enforce_one_to_one : bool, optional
        Enforce at most one S1 match per candidate record, by default True.

    Returns
    -------
    dict
        Dictionary containing:
        - "best_threshold": float
        - "best_min_top_score": float | None
        - "best_macro_f05": float
        - "auc_metrics": dict (roc_auc, pr_auc)
        - "sweep_records": list of dicts for each evaluated threshold
        - "best_predictions": dict[str, list[str]]
    """
    if threshold_range is None:
        threshold_range = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]

    if min_top_score_offsets is None:
        min_top_score_offsets = [0.0, 0.05, 0.10, 0.15, 0.20]

    # 1. Flatten all candidate pairs for AUC computation
    y_true_list: list[int] = []
    y_score_list: list[float] = []
    for s1_id, cands in scored_candidates.items():
        true_set = set(ground_truth.get(s1_id, []))
        for item in cands:
            cand_id = item[0] if isinstance(item, tuple) else item
            score = float(item[1]) if isinstance(item, tuple) else 1.0
            y_true_list.append(1 if cand_id in true_set else 0)
            y_score_list.append(score)

    auc_metrics = compute_roc_pr_auc(np.array(y_true_list), np.array(y_score_list))

    # 2. Sweep thresholds & singleton trigger offsets
    # Restrict evaluation to entities present in scored_candidates for massive speedup
    eval_gt_sets: dict[str, set[str]] = {s1_id: set(ground_truth.get(s1_id, [])) for s1_id in scored_candidates}
    total_singletons = sum(1 for s in eval_gt_sets.values() if len(s) == 0)
    total_gt = sum(len(s) for s in eval_gt_sets.values())

    sweep_records = []
    best_thresh = 0.50
    best_min_top = None
    best_score = -1.0
    best_preds: dict[str, list[str]] = {}

    for thresh in threshold_range:
        for offset in min_top_score_offsets:
            min_top = round(min(0.99, thresh + offset), 2) if offset > 0.0 else None

            if enforce_one_to_one:
                preds = apply_assignment_constraint(
                    scored_candidates,
                    threshold=thresh,
                    min_top_score=min_top,
                )
            else:
                preds = {}
                for s1_id, cands in scored_candidates.items():
                    if min_top is not None:
                        scores = [float(item[1]) if isinstance(item, tuple) else 1.0 for item in cands]
                        max_s = max(scores, default=0.0)
                        if max_s < min_top:
                            preds[s1_id] = []
                            continue
                    preds[s1_id] = [
                        (item[0] if isinstance(item, tuple) else item)
                        for item in cands
                        if (float(item[1]) if isinstance(item, tuple) else 1.0) >= thresh
                    ]

            # Fast single-pass evaluation for Macro F0.5 and micro metrics
            entity_f05_scores: list[float] = []
            total_tp = 0
            total_pred = 0
            singleton_correct = 0

            for s1_id, g_set in eval_gt_sets.items():
                p_ids = preds.get(s1_id, [])
                p_set = set(p_ids)
                n_p = len(p_set)
                n_g = len(g_set)
                total_pred += n_p

                if n_g == 0:
                    if n_p == 0:
                        singleton_correct += 1
                        entity_f05_scores.append(1.0)
                    else:
                        entity_f05_scores.append(0.0)
                else:
                    if n_p == 0:
                        entity_f05_scores.append(0.0)
                    else:
                        tp = len(p_set & g_set)
                        total_tp += tp
                        if tp == 0:
                            entity_f05_scores.append(0.0)
                        else:
                            p = tp / n_p
                            r = tp / n_g
                            entity_f05_scores.append((1.25 * p * r) / (0.25 * p + r))

            macro_score = float(np.mean(entity_f05_scores)) if entity_f05_scores else 0.0
            micro_p = total_tp / total_pred if total_pred > 0 else 0.0
            micro_r = total_tp / total_gt if total_gt > 0 else 0.0
            singleton_acc = singleton_correct / total_singletons if total_singletons > 0 else 1.0

            sweep_records.append({
                "threshold": thresh,
                "min_top_score": min_top if min_top is not None else thresh,
                "macro_f05": macro_score,
                "micro_precision": micro_p,
                "micro_recall": micro_r,
                "singleton_accuracy": singleton_acc,
                "predicted_pairs": total_pred,
            })

            if macro_score > best_score:
                best_score = macro_score
                best_thresh = thresh
                best_min_top = min_top
                best_preds = preds

    return {
        "best_threshold": best_thresh,
        "best_min_top_score": best_min_top,
        "best_macro_f05": best_score,
        "auc_metrics": auc_metrics,
        "sweep_records": sweep_records,
        "best_predictions": best_preds,
    }


def format_scored_pairs_dataframe(
    scored_candidates: dict[str, list[tuple[str, float]]],
    bi_encoder_candidates: Optional[dict[str, list[Union[str, tuple[str, float]]]]] = None,
    ground_truth: Optional[dict[str, list[str]]] = None,
    extra_features: bool = False,
) -> pd.DataFrame:
    """Convert candidate score dictionary to standard DataFrame.

    If bi_encoder_candidates is provided or extra_features is True, extra neural features
    (bi_encoder_score, retrieval_rank, margins, etc.) are included for Path A.
    Otherwise, returns the canonical 3-column DataFrame: [s1_id, cand_id, score].
    """
    if extra_features or bi_encoder_candidates is not None or ground_truth is not None:
        from src.path_b.feature_export import build_path_b_feature_dataframe
        return build_path_b_feature_dataframe(
            cross_scored_candidates=scored_candidates,
            bi_encoder_candidates=bi_encoder_candidates,
            ground_truth=ground_truth,
        )

    rows = []
    for s1_id, cands in scored_candidates.items():
        for cand_id, score in cands:
            rows.append({
                "s1_id": str(s1_id),
                "cand_id": str(cand_id),
                "score": float(score),
            })
    return pd.DataFrame(rows)


def export_scores_parquet(
    scored_candidates: dict[str, list[tuple[str, float]]],
    output_path: Union[str, Path] = SCORES_DIR / "path_b_val.parquet",
    bi_encoder_candidates: Optional[dict[str, list[Union[str, tuple[str, float]]]]] = None,
    ground_truth: Optional[dict[str, list[str]]] = None,
    extra_features: bool = False,
) -> Path:
    """Save scored pairs to parquet according to the shared team contract.

    Schema:
        s1_id: string
        cand_id: string
        score: float in [0.0, 1.0]
        plus optional extra features for Path A ensemble when bi_encoder_candidates is given
        or extra_features=True.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = format_scored_pairs_dataframe(
        scored_candidates=scored_candidates,
        bi_encoder_candidates=bi_encoder_candidates,
        ground_truth=ground_truth,
        extra_features=extra_features,
    )
    df.to_parquet(out_file, index=False)
    print(f"[Export] Saved {len(df):,} scored candidate pairs to {out_file}")
    return out_file


def export_matching_results_tsv(
    predictions: dict[str, list[str]],
    output_path: Union[str, Path] = OUTPUT_DIR / "matching_results_path_b.tsv",
) -> Path:
    """Export final matching results TSV in competition submission format.

    Format:
        source1_entity_id \\t matched_entity_ids
    where matched_entity_ids is comma-separated IDs or empty string.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id, matched_ids in predictions.items():
        rows.append({
            "source1_entity_id": str(s1_id),
            "matched_entity_ids": ",".join(matched_ids) if matched_ids else "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_file, sep="\t", index=False)
    print(f"[Export] Saved {len(df):,} prediction rows to {out_file}")
    return out_file
