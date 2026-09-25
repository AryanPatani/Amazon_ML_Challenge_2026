"""
src/common/data_loader.py

Shared data-loading utilities for the Amazon ML Challenge 2026.
All TSV files are read with explicit sep='\\t' as required by the problem spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.paths import TRAIN_DIR, TEST_DIR, GROUND_TRUTH_FILE


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def load_source(path: str | Path, expected_prefix: Optional[str] = None) -> pd.DataFrame:
    """Load a single source TSV (S1, S2, or S3).

    Parameters
    ----------
    path:
        Absolute or relative path to the source TSV file.
    expected_prefix:
        If given (e.g. ``'S1-'``), assert that every ``entity_id`` starts with
        this prefix.

    Returns
    -------
    pd.DataFrame with columns: entity_id, business_name, business_address, country
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required_cols = {"entity_id", "business_name", "business_address", "country"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Source file {path} is missing columns: {missing}")

    if expected_prefix is not None:
        bad = df[~df["entity_id"].str.startswith(expected_prefix)]
        if len(bad) > 0:
            raise ValueError(
                f"Found {len(bad)} rows in {path} whose entity_id does not start "
                f"with '{expected_prefix}'."
            )
    return df


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Load the ground-truth TSV file.

    Expected columns: ``source1_entity_id``, ``matched_entity_ids`` (or ``matches``) (tab-separated).
    The matches column contains either an empty string (singleton) or a
    comma-separated list of S2/S3 entity IDs.

    Returns
    -------
    pd.DataFrame with columns:
        - source1_entity_id (str)
        - matches           (list[str]): may be empty list for singletons
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    if "source1_entity_id" not in df.columns:
        raise ValueError(f"Ground truth file {path} is missing column: 'source1_entity_id'")

    if "matched_entity_ids" in df.columns:
        match_col = "matched_entity_ids"
    elif "matches" in df.columns:
        match_col = "matches"
    else:
        raise ValueError(
            f"Ground truth file {path} is missing column: expected 'matched_entity_ids' or 'matches'"
        )

    def _parse(cell: str) -> list[str]:
        cell = cell.strip()
        if not cell:
            return []
        return [x.strip() for x in cell.split(",") if x.strip()]

    df["matches"] = df[match_col].apply(_parse)
    return df


def load_all_sources(
    train_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader for the full training set.

    Defaults to the canonical TRAIN_DIR from src.common.paths (resolved via
    __file__, so it works on every teammate's machine without config).

    Returns
    -------
    (source1, source2, source3, ground_truth)
    """
    td = Path(train_dir) if train_dir is not None else TRAIN_DIR

    s1 = load_source(td / "train_source1.tsv", expected_prefix="S1-")
    s2 = load_source(td / "train_source2.tsv", expected_prefix="S2-")
    s3 = load_source(td / "train_source3.tsv", expected_prefix="S3-")

    gt_file = td / "train_ground_truth.tsv"
    if not gt_file.exists():
        gt_file = td / "train_labels.tsv"
    gt = load_ground_truth(gt_file)
    return s1, s2, s3, gt


def load_test_sources(
    test_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience loader for the test set (no labels available).

    Defaults to the canonical TEST_DIR from src.common.paths.

    Returns
    -------
    (source1, source2, source3)
    """
    td = Path(test_dir) if test_dir is not None else TEST_DIR

    s1 = load_source(td / "test_source1.tsv", expected_prefix="S1-")
    s2 = load_source(td / "test_source2.tsv", expected_prefix="S2-")
    s3 = load_source(td / "test_source3.tsv", expected_prefix="S3-")
    return s1, s2, s3
