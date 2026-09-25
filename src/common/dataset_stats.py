"""
Ticket 0.1 — Data stats script

Run with:
    python -m src.common.dataset_stats --train-dir data/dataset/train --test-dir data/dataset/test
    # (or simply `python -m src.common.dataset_stats` using defaults)

Prints:
  - Row counts and country distributions for S1, S2, S3
  - Singleton fraction and match-count distribution
  - S2 vs S3 share of total matches
  - KEY CHECK: does any S2/S3 record match >1 S1 entity?

This script does NOT require any GPU or network; it only reads the TSVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd


def _sep() -> None:
    print("=" * 90)


def basic_stats(label: str, df: pd.DataFrame) -> None:
    _sep()
    print(f"BASIC STATS — {label}")
    _sep()
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    bad_ids = df[~df["entity_id"].str.startswith(label.split('(')[0].strip().split()[-1][0:2] + "-")]
    prefix = df["entity_id"].iloc[0][:3] if len(df) else "?"
    prefix = prefix.split('-')[0] + '-'
    bad_ids = df[~df["entity_id"].str.startswith(prefix)]
    print(f"Rows whose entity_id does NOT start with '{prefix}': {len(bad_ids)}")
    print(f"Duplicate entity_id values: {df['entity_id'].duplicated().sum()}")
    print()
    print("Country distribution:")
    print(df["country"].value_counts().to_string())
    print()
    for field in ["business_name", "business_address"]:
        n_empty = (df[field] == "").sum()
        print(f"Empty '{field}': {n_empty:,} ({n_empty / len(df) * 100:.2f}%)")
    dup_rows = df.duplicated(subset=["business_name", "business_address"]).sum()
    src_label = label.split('(')[0].strip()
    print(f"Rows with duplicate (business_name, business_address) within {src_label}: {dup_rows:,}")
    print()
    for field in ["business_name", "business_address"]:
        lengths = df[field].str.len()
        print(
            f"'{field}' length — mean: {lengths.mean():.1f}, "
            f"median: {int(lengths.median())}, "
            f"min: {lengths.min()}, max: {lengths.max()}"
        )
    print()


def _parse_matches(cell: str) -> list[str]:
    cell = str(cell).strip()
    if not cell:
        return []
    return [x.strip() for x in cell.split(",") if x.strip()]


def ground_truth_stats(gt: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> None:
    _sep()
    print("GROUND TRUTH STATS")
    _sep()

    gt = gt.copy()
    gt["match_list"] = gt["matches"].apply(_parse_matches)
    gt["n_matches"] = gt["match_list"].apply(len)

    print(f"Rows in ground truth: {len(gt):,}")
    print(f"Distinct S1 IDs in ground truth: {gt['source1_entity_id'].nunique():,}")
    print(f"Distinct S1 IDs in train_source1.tsv: {s1['entity_id'].nunique():,}")

    s1_ids = set(s1["entity_id"])
    gt_ids = set(gt["source1_entity_id"])
    print(f"S1 entities in source1 but MISSING from ground truth: {len(s1_ids - gt_ids)}")
    print(f"S1 entities in ground truth but NOT in source1 file (should be 0): {len(gt_ids - s1_ids)}")
    print(f"Duplicate source1_entity_id rows in ground truth (should be 0): {gt['source1_entity_id'].duplicated().sum()}")
    print()

    n_singletons = (gt["n_matches"] == 0).sum()
    n_with_match = (gt["n_matches"] >= 1).sum()
    print(f"Singletons (S1 with 0 matches): {n_singletons:,} ({n_singletons / len(gt) * 100:.2f}%)")
    print(f"S1 with >=1 match: {n_with_match:,} ({n_with_match / len(gt) * 100:.2f}%)")
    print(f"Average matches per S1 (all): {gt['n_matches'].mean():.3f}")
    non_singletons = gt[gt["n_matches"] >= 1]
    print(f"Average matches per S1 (non-singletons only): {non_singletons['n_matches'].mean():.3f}")
    print(f"Max matches for a single S1 entity: {gt['n_matches'].max()}")
    print()

    # Distribution of match counts
    print("Distribution of match-count (0,1,2,3,4,5+):")
    cnt = gt["n_matches"].value_counts().sort_index()
    for k, v in cnt.items():
        label = str(k) if k <= 4 else "5+"
        existing = cnt[cnt.index > 4].sum() if k > 4 else v
        if k <= 4:
            print(f"  {label:<4} {v:>10,}")
    fiveplus = cnt[cnt.index >= 5].sum()
    print(f"  5+   {fiveplus:>10,}")
    print()

    # S2 vs S3 split
    all_matched = [m for ml in gt["match_list"] for m in ml]
    total = len(all_matched)
    n_s2 = sum(1 for m in all_matched if m.startswith("S2-"))
    n_s3 = sum(1 for m in all_matched if m.startswith("S3-"))
    print(f"Total matched ID mentions: {total:,}")
    print(f"  from Source 2: {n_s2:,} ({n_s2 / total * 100:.1f}%)")
    print(f"  from Source 3: {n_s3:,} ({n_s3 / total * 100:.1f}%)")
    print()

    s2_ids_in_gt = set(m for m in all_matched if m.startswith("S2-"))
    s3_ids_in_gt = set(m for m in all_matched if m.startswith("S3-"))
    s2_ids_source = set(s2["entity_id"])
    s3_ids_source = set(s3["entity_id"])
    print(f"Matched S2 IDs not found in train_source2.tsv (should be 0): {len(s2_ids_in_gt - s2_ids_source)}")
    print(f"Matched S3 IDs not found in train_source3.tsv (should be 0): {len(s3_ids_in_gt - s3_ids_source)}")
    print()


def check_one_to_one(gt: pd.DataFrame) -> None:
    _sep()
    print("KEY CHECK — does an S2/S3 record ever match MORE THAN ONE S1 entity?")
    _sep()

    gt = gt.copy()
    gt["match_list"] = gt["matches"].apply(_parse_matches)

    match_to_s1: dict[str, list[str]] = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        for m in row["match_list"]:
            match_to_s1.setdefault(m, []).append(s1_id)

    multi = {k: v for k, v in match_to_s1.items() if len(v) > 1}
    total_matched = sum(len(ml) for ml in gt["match_list"])
    print(
        f"S2/S3 records matched to >1 distinct S1 entity: "
        f"{len(multi)} out of {total_matched:,} matched records "
        f"({len(multi) / total_matched * 100:.3f}%)"
    )
    print()
    if multi:
        print("=> WARNING: one-to-one constraint is VIOLATED. Sample conflicts:")
        for k, v in list(multi.items())[:5]:
            print(f"  {k} -> {v}")
    else:
        print(
            "=> Every matched S2/S3 record belongs to exactly one S1 entity in this data.\n"
            "   This CONFIRMS the one-to-one assignment constraint — safe to enforce in Path A5."
        )
    print()


def country_consistency(gt: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> None:
    _sep()
    print("COUNTRY CONSISTENCY CHECK (matched pairs)")
    _sep()

    s1_country = s1.set_index("entity_id")["country"]
    s2_country = s2.set_index("entity_id")["country"]
    s3_country = s3.set_index("entity_id")["country"]
    cand_country = pd.concat([s2_country, s3_country])

    gt2 = gt.copy()
    gt2["match_list"] = gt2["matches"].apply(_parse_matches)

    rows = []
    for _, row in gt2.iterrows():
        s1_c = s1_country.get(row["source1_entity_id"], None)
        for m in row["match_list"]:
            rows.append({"s1_country": s1_c, "cand_country": cand_country.get(m, None)})

    if not rows:
        print("No matched pairs found.")
        return

    pairs = pd.DataFrame(rows)
    same = (pairs["s1_country"] == pairs["cand_country"]).sum()
    diff = len(pairs) - same
    print(f"Matched pairs with SAME country label: {same:,} ({same / len(pairs) * 100:.2f}%)")
    print(f"Matched pairs with DIFFERENT country label: {diff:,} ({diff / len(pairs) * 100:.2f}%)")
    print()
    if diff == 0:
        print(
            "=> Country always matches within a pair in this sample — safe to use as a hard "
            "blocking filter for US/India.\n"
            "   Still avoid one-hot encoding it (France will be unseen)."
        )
    else:
        print("=> WARNING: cross-country matches exist. Do NOT use country as a hard block filter.")
    print()


def test_set_overview(test_dir: Path, s2: pd.DataFrame, s3: pd.DataFrame) -> None:
    _sep()
    print("TEST SET OVERVIEW")
    _sep()

    for fname, prefix in [
        ("test_source1.tsv", "S1-"),
        ("test_source2.tsv", "S2-"),
        ("test_source3.tsv", "S3-"),
    ]:
        fpath = test_dir / fname
        if not fpath.exists():
            print(f"{fname}: NOT FOUND (skip)")
            continue
        df = pd.read_csv(fpath, sep="\t", dtype=str, keep_default_na=False)
        print(f"{fname}: {len(df):,} rows")
        print(df["country"].value_counts().to_string())
        print()

    try:
        ts1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
        print(f"Distinct countries in test_source1: {sorted(ts1['country'].unique())}")
        print("(Compare this against train countries printed earlier — expect an extra unseen country, e.g. France.)")
    except FileNotFoundError:
        pass
    print()


def sample_matched_groups(gt: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, n: int = 8) -> None:
    _sep()
    print(f"SAMPLE MATCHED GROUPS (n={n}) — read these to see real noise patterns")
    _sep()

    gt2 = gt.copy()
    gt2["match_list"] = gt2["matches"].apply(_parse_matches)
    non_singletons = gt2[gt2["match_list"].apply(len) >= 2].sample(min(n, len(gt2)), random_state=42)

    s1_idx = s1.set_index("entity_id")
    s2_idx = s2.set_index("entity_id")
    s3_idx = s3.set_index("entity_id")
    cand_idx = pd.concat([s2_idx, s3_idx])

    for _, row in non_singletons.iterrows():
        s1_id = row["source1_entity_id"]
        if s1_id not in s1_idx.index:
            continue
        s1_row = s1_idx.loc[s1_id]
        country = s1_row["country"]
        print(f"\n--- {s1_id} ({country}) ---")
        print(f"  NAME:    {s1_row['business_name']}")
        print(f"  ADDRESS: {s1_row['business_address']}")
        for m in row["match_list"]:
            if m in cand_idx.index:
                cr = cand_idx.loc[m]
                print(f"    MATCH {m}: {cr['business_name']} | {cr['business_address']}")
            else:
                print(f"    MATCH {m}: [NOT FOUND IN SOURCE]")
    print()


def main(train_dir: Path, test_dir: Path) -> None:
    print(f"EDA report for: train_dir={train_dir}, test_dir={test_dir}\n")

    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)

    gt_file = train_dir / "train_ground_truth.tsv"
    if not gt_file.exists():
        gt_file = train_dir / "train_labels.tsv"
    gt = pd.read_csv(gt_file, sep="\t", dtype=str, keep_default_na=False)
    if "matches" not in gt.columns and "matched_entity_ids" in gt.columns:
        gt["matches"] = gt["matched_entity_ids"]

    basic_stats("train_source1 (S1)", s1)
    basic_stats("train_source2 (S2)", s2)
    basic_stats("train_source3 (S3)", s3)

    ground_truth_stats(gt, s1, s2, s3)
    check_one_to_one(gt)
    country_consistency(gt, s1, s2, s3)
    sample_matched_groups(gt, s1, s2, s3)

    if test_dir.exists():
        test_set_overview(test_dir, s2, s3)

    _sep()
    print("DONE")
    _sep()


if __name__ == "__main__":
    default_train = "data/dataset/train" if Path("data/dataset/train").exists() else "dataset/train"
    default_test = "data/dataset/test" if Path("data/dataset/test").exists() else "dataset/test"

    parser = argparse.ArgumentParser(description="Ticket 0.1 — Dataset EDA and stats")
    parser.add_argument("--train-dir", default=default_train, type=Path)
    parser.add_argument("--test-dir",  default=default_test,  type=Path)
    args = parser.parse_args()
    main(args.train_dir, args.test_dir)
