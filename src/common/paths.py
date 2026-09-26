"""
src/common/paths.py

Central path resolver for the Amazon ML Challenge 2026 repo.

HOW IT WORKS
------------
This file is at  <repo_root>/src/common/paths.py
So  Path(__file__).resolve().parents[2]  always points to the repo root,
regardless of:
  - which OS (Windows / macOS / Linux)
  - which user's machine
  - which directory you cd'd into before running

Everyone on the team puts the dataset under:
    <repo_root>/data/dataset/

That folder is in .gitignore so it is never committed.

USAGE
-----
    from src.common.paths import TRAIN_DIR, TEST_DIR

    s1 = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\\t")
    gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\\t")
    ts1 = pd.read_csv(TEST_DIR  / "test_source1.tsv",  sep="\\t")
"""

import os
from pathlib import Path

# This file: <repo_root>/src/common/paths.py
# parents[0] = src/common
# parents[1] = src
# parents[2] = <repo_root>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _resolve_data_dir() -> Path:
    """Find dataset directory across common teammate layouts.
    
    Checks:
    1. Environment variable AMAZON_ML_DATA_DIR or DATA_DIR
    2. Inside repo:
       - <repo_root>/data/dataset
       - <repo_root>/dataset
       - <repo_root>/data
    3. Outside repo (in same parent folder):
       - <repo_root>/../dataset
       - <repo_root>/../data/dataset
       - <repo_root>/../data
    """
    env_dir = os.environ.get("AMAZON_ML_DATA_DIR") or os.environ.get("DATA_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        if (p / "train").exists() or p.exists():
            return p

    candidates = [
        REPO_ROOT / "data" / "dataset",
        REPO_ROOT / "dataset",
        REPO_ROOT / "data",
        REPO_ROOT.parent / "dataset",
        REPO_ROOT.parent / "data" / "dataset",
        REPO_ROOT.parent / "data",
    ]

    for candidate in candidates:
        if (candidate / "train").exists():
            return candidate

    # Check common Kaggle / Colab input directories
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob("train"):
            if p.is_dir() and (p / "train_source1.tsv").exists():
                return p.parent

    # Fallback to default expected path
    return REPO_ROOT / "data" / "dataset"


DATA_DIR:  Path = _resolve_data_dir()
TRAIN_DIR: Path = DATA_DIR / "train"
TEST_DIR:  Path = DATA_DIR / "test"

# Convenience ground-truth path (dataset uses train_ground_truth.tsv)
GROUND_TRUTH_FILE: Path = TRAIN_DIR / "train_ground_truth.tsv"

# Output dirs (version-controlled but content gitignored via *.tsv / *.parquet)
OUTPUT_DIR: Path = REPO_ROOT / "output"
SCORES_DIR: Path = REPO_ROOT / "scores"
SPLITS_DIR: Path = REPO_ROOT / "splits"


def assert_dataset_exists() -> None:
    """Raise a clear error if the dataset folder is missing."""
    if not (DATA_DIR / "train").exists():
        raise FileNotFoundError(
            f"\n\n[paths.py] Dataset not found! Checked current DATA_DIR:\n  {DATA_DIR}\n\n"
            "Supported locations for teammates:\n"
            f"  1. Inside repo:      {REPO_ROOT}/data/dataset/\n"
            f"  2. Inside repo:      {REPO_ROOT}/dataset/\n"
            f"  3. In parent folder: {REPO_ROOT.parent}/dataset/\n"
            f"  4. In parent folder: {REPO_ROOT.parent}/data/dataset/\n"
            "  5. Set environment variable: export AMAZON_ML_DATA_DIR=/path/to/dataset\n"
        )
