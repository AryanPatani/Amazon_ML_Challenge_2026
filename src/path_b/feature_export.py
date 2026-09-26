"""
src/path_b/feature_export.py

Ticket B5: Feature export for downstream models (Path A and Path D).
Exports neural retrieval scores (bi-encoder cosine) and cross-encoder probabilities
along with derived ranking features as shared Parquet files:
    scores/path_b_val.parquet
    scores/path_b_test.parquet

Shared Contract (files/tickets.md Section 4):
    - Required columns: s1_id (str), cand_id (str), score (float in [0, 1])
    - Extra features for ensemble:
        - bi_encoder_score: dense embedding cosine similarity
        - retrieval_rank: 1-indexed retrieval rank from bi-encoder
        - retrieval_rr: reciprocal rank (1 / rank)
        - score_margin: gap between candidate score and top candidate score
        - bi_encoder_margin: gap between candidate bi-score and top bi-score
        - score_ratio: ratio to top candidate score
        - is_match: binary ground truth label (val set only)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union, Sequence
import numpy as np
import pandas as pd

from src.common.paths import SCORES_DIR


# ---------------------------------------------------------------------------
# Feature DataFrame construction
# ---------------------------------------------------------------------------

def build_path_b_feature_dataframe(
    cross_scored_candidates: dict[str, list[tuple[str, float]]],
    bi_encoder_candidates: Optional[dict[str, list[Union[str, tuple[str, float]]]]] = None,
    ground_truth: Optional[dict[str, list[str]]] = None,
    include_margins: bool = True,
    include_ratios: bool = True,
) -> pd.DataFrame:
    """Build a rich feature DataFrame combining Cross-Encoder and Bi-Encoder signals.

    Parameters
    ----------
    cross_scored_candidates : dict[str, list[tuple[str, float]]]
        Mapping s1_id -> list of (cand_id, cross_encoder_score) pairs.
    bi_encoder_candidates : Optional[dict[str, list[str | tuple[str, float]]]], optional
        Mapping s1_id -> list of candidate IDs or (cand_id, cosine_sim) pairs from
        the bi-encoder retrieval stage.
    ground_truth : Optional[dict[str, list[str]]], optional
        Mapping s1_id -> list of true matching corpus IDs (for val split).
        If provided, adds an 'is_match' (0 or 1) target column for Path A classifier.
    include_margins : bool, optional
        Compute score gaps relative to the anchor's best candidate, by default True.
    include_ratios : bool, optional
        Compute score ratios relative to the anchor's best candidate, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with schema:
            s1_id : str
            cand_id : str
            score : float in [0.0, 1.0] (shared contract primary score)
            cross_encoder_score : float (alias of score)
            bi_encoder_score : float (cosine similarity)
            retrieval_rank : int (1-indexed rank from retrieval)
            retrieval_rr : float (1.0 / rank)
            score_margin : float (score - top_cross_score <= 0.0)
            score_ratio : float (score / (top_cross_score + 1e-6))
            bi_encoder_margin : float (bi_score - top_bi_score <= 0.0)
            bi_encoder_ratio : float (bi_score / (top_bi_score + 1e-6))
            is_match : int (1 or 0, only if ground_truth is provided)
    """
    # Pre-parse bi-encoder candidates into quick lookups: s1_id -> {cand_id: (rank, score)}
    bi_lookup: dict[str, dict[str, tuple[int, float]]] = {}
    if bi_encoder_candidates is not None:
        for s1_id, cands in bi_encoder_candidates.items():
            s1_str = str(s1_id)
            bi_lookup[s1_str] = {}
            for rank_0, item in enumerate(cands):
                rank = rank_0 + 1  # 1-indexed
                if isinstance(item, tuple):
                    c_id, c_score = item[0], float(item[1])
                else:
                    c_id, c_score = str(item), 0.0
                bi_lookup[s1_str][str(c_id)] = (rank, c_score)

    gt_sets: dict[str, set[str]] = {}
    if ground_truth is not None:
        gt_sets = {str(k): set(v) for k, v in ground_truth.items()}

    rows = []
    for s1_id, scored_list in cross_scored_candidates.items():
        s1_str = str(s1_id)
        if not scored_list:
            continue

        # Extract scores
        clean_pairs: list[tuple[str, float]] = []
        for item in scored_list:
            if isinstance(item, tuple):
                clean_pairs.append((str(item[0]), float(item[1])))
            else:
                clean_pairs.append((str(item), 0.0))

        # Anchor-level anchor top score for margin/ratio calculations
        top_cross_score = max((sc for _, sc in clean_pairs), default=0.0)

        # Bi-encoder candidates for this anchor
        s1_bi_cands = bi_lookup.get(s1_str, {})
        max_bi_rank = max((r for r, _ in s1_bi_cands.values()), default=100)
        top_bi_score = max((sc for _, sc in s1_bi_cands.values()), default=0.0)

        true_matches = gt_sets.get(s1_str, set()) if ground_truth is not None else None

        for cand_id, cross_score in clean_pairs:
            # Bound cross_score to [0.0, 1.0] for contract compliance
            bounded_cross = float(np.clip(cross_score, 0.0, 1.0))

            # Retrieve bi-encoder signals
            bi_info = s1_bi_cands.get(cand_id)
            if bi_info is not None:
                ret_rank, bi_score = bi_info
            else:
                # Candidate was not in bi-encoder top-k (or bi-encoder info not provided)
                ret_rank = max_bi_rank + 1
                bi_score = 0.0

            ret_rr = 1.0 / float(ret_rank)

            row: dict = {
                "s1_id": s1_str,
                "cand_id": cand_id,
                # Shared contract primary score:
                "score": bounded_cross,
                # Explicit feature columns for Path A GBDT:
                "cross_encoder_score": bounded_cross,
                "bi_encoder_score": float(bi_score),
                "retrieval_rank": int(ret_rank),
                "retrieval_rr": float(ret_rr),
            }

            if include_margins:
                row["score_margin"] = float(bounded_cross - top_cross_score)
                row["bi_encoder_margin"] = float(bi_score - top_bi_score)

            if include_ratios:
                row["score_ratio"] = float(bounded_cross / (top_cross_score + 1e-6))
                denom = abs(top_bi_score) if abs(top_bi_score) > 1e-6 else 1.0
                row["bi_encoder_ratio"] = float(bi_score / denom)

            # Rank discounted score (combining bi-rank and cross-score)
            row["rank_discounted_score"] = float(bounded_cross / math.log2(1.0 + ret_rank))

            if true_matches is not None:
                row["is_match"] = 1 if cand_id in true_matches else 0

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_path_b_dataframe(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a DataFrame against the shared contract and Path B export schema.

    Returns
    -------
    (is_valid, error_messages)
    """
    errors: list[str] = []

    required_cols = {"s1_id", "cand_id", "score"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Missing required contract columns: {missing}")

    if not errors:
        # Check nulls
        for col in ["s1_id", "cand_id", "score"]:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} null values.")

        # Check score range
        if df["score"].dtype.kind in "fc":
            min_val = df["score"].min()
            max_val = df["score"].max()
            if min_val < 0.0 - 1e-6 or max_val > 1.0 + 1e-6:
                errors.append(f"'score' values out of [0, 1] range: [{min_val}, {max_val}]")

        # Check types
        if not pd.api.types.is_string_dtype(df["s1_id"]):
            errors.append(f"'s1_id' must be string dtype, got {df['s1_id'].dtype}")
        if not pd.api.types.is_string_dtype(df["cand_id"]):
            errors.append(f"'cand_id' must be string dtype, got {df['cand_id'].dtype}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Parquet Export
# ---------------------------------------------------------------------------

def export_path_b_parquet(
    df: pd.DataFrame,
    output_path: Union[str, Path] = SCORES_DIR / "path_b_val.parquet",
    verbose: bool = True,
) -> Path:
    """Save feature DataFrame to Parquet, validating against the shared contract.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with s1_id, cand_id, score, and extra features.
    output_path : Union[str, Path], optional
        Target parquet file path, by default SCORES_DIR / "path_b_val.parquet".
    verbose : bool, optional
        Whether to print summary upon export, by default True.

    Returns
    -------
    Path
        Absolute path to the exported parquet file.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    is_valid, errors = validate_path_b_dataframe(df)
    if not is_valid:
        raise ValueError(f"DataFrame failed Path B contract validation: {errors}")

    df.to_parquet(out_file, index=False)

    if verbose:
        n_pairs = len(df)
        n_anchors = df["s1_id"].nunique()
        print(f"\n[B5 Export] Successfully saved {n_pairs:,} scored pairs for {n_anchors:,} anchors.")
        print(f"            Target file: {out_file}")
        print(f"            Columns: {list(df.columns)}")
        if "score" in df.columns:
            print(f"            Score range: [{df['score'].min():.4f}, {df['score'].max():.4f}] (mean: {df['score'].mean():.4f})")
        if "is_match" in df.columns:
            n_pos = int(df["is_match"].sum())
            print(f"            Matches in set: {n_pos:,} / {n_pairs:,} ({n_pos / max(1, n_pairs) * 100:.2f}%)")

    return out_file


def load_path_b_features(
    parquet_path: Union[str, Path] = SCORES_DIR / "path_b_val.parquet",
    validate: bool = True,
) -> pd.DataFrame:
    """Load and validate Path B exported features for use in Path A or Path D.

    Parameters
    ----------
    parquet_path : Union[str, Path], optional
        Path to the exported parquet file.
    validate : bool, optional
        Whether to validate the contract schema upon loading.

    Returns
    -------
    pd.DataFrame
        The loaded features DataFrame.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Path B features parquet not found: {path}")

    df = pd.read_parquet(path)

    if validate:
        is_valid, errors = validate_path_b_dataframe(df)
        if not is_valid:
            raise ValueError(f"Loaded parquet {path} failed contract validation: {errors}")

    return df
