"""Reproducible exploratory data analysis for the challenge dataset.

Run from any directory with:
    python eda/explore.py

The script only reads data/raw and writes derived Parquet files to
dataset_parquet/. Redirect stdout to retain a complete report if desired.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import gc
import sys

import pandas as pd


# The records include multilingual business names and addresses. Windows may
# otherwise select cp1252 for a redirected console stream and fail mid-report.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PARQUET = ROOT / "dataset_parquet"
READ_KWARGS = {"sep": "\t", "dtype": str, "keep_default_na": False}
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
FILES = {
    "train_source1": (RAW / "train" / "train_source1.tsv", "S1"),
    "train_source2": (RAW / "train" / "train_source2.tsv", "S2"),
    "train_source3": (RAW / "train" / "train_source3.tsv", "S3"),
    "train_ground_truth": (RAW / "train" / "train_ground_truth.tsv", None),
    "test_source1": (RAW / "test" / "test_source1.tsv", "S1"),
    "test_source2": (RAW / "test" / "test_source2.tsv", "S2"),
    "test_source3": (RAW / "test" / "test_source3.tsv", "S3"),
}


def heading(step: int, title: str) -> None:
    print(f"\n{'=' * 100}\nSTEP {step} — {title}\n{'=' * 100}", flush=True)


def load_tsv(path: Path) -> pd.DataFrame:
    """Load a challenge TSV without converting meaningful empty strings to NA."""
    return pd.read_csv(path, **READ_KWARGS)


def bytes_text(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GiB"


def source_check(df: pd.DataFrame, expected_prefix: str) -> dict[str, object]:
    return {
        "empty business_name": int((df["business_name"] == "").sum()),
        "empty business_address": int((df["business_address"] == "").sum()),
        "duplicate entity_id rows": int(df["entity_id"].duplicated().sum()),
        "wrong entity_id prefix": int((~df["entity_id"].str.startswith(f"{expected_prefix}-")).sum()),
        "memory": bytes_text(int(df.memory_usage(deep=True).sum())),
    }


def print_source_record(prefix: str, record: pd.Series) -> None:
    print(
        f"  {prefix}: {record['entity_id']} | name={record['business_name']} | "
        f"address={record['business_address']} | country={record['country']}"
    )


def scan_source_file(path: Path) -> dict[str, object]:
    """Calculate Step-6 field statistics in chunks to keep memory bounded."""
    summary = {
        "rows": 0,
        "name_min": None,
        "name_max": 0,
        "name_sum": 0,
        "address_min": None,
        "address_max": 0,
        "address_sum": 0,
        "zip_like": 0,
        "near": 0,
        "names": Counter(),
    }
    for chunk in pd.read_csv(path, chunksize=200_000, **READ_KWARGS):
        summary["rows"] += len(chunk)
        name_lengths = chunk["business_name"].str.len()
        address_lengths = chunk["business_address"].str.len()
        summary["name_min"] = min(summary["name_min"], int(name_lengths.min())) if summary["name_min"] is not None else int(name_lengths.min())
        summary["name_max"] = max(summary["name_max"], int(name_lengths.max()))
        summary["name_sum"] += int(name_lengths.sum())
        summary["address_min"] = min(summary["address_min"], int(address_lengths.min())) if summary["address_min"] is not None else int(address_lengths.min())
        summary["address_max"] = max(summary["address_max"], int(address_lengths.max()))
        summary["address_sum"] += int(address_lengths.sum())
        summary["zip_like"] += int(chunk["business_address"].str.contains(r"\b\d{5,6}\b", regex=True).sum())
        summary["near"] += int(chunk["business_address"].str.contains(r"\bnear\b", case=False, regex=True).sum())
        summary["names"].update(chunk["business_name"])
    return summary


def main() -> None:
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)
    basic_summary: dict[str, dict[str, object]] = {}
    random_samples: dict[str, pd.DataFrame] = {}

    heading(1, "Basic load and shape")
    for key, (path, expected_prefix) in FILES.items():
        df = load_tsv(path)
        print(f"\n{key}: {len(df):,} rows")
        print(f"columns: {df.columns.tolist()}")
        if expected_prefix is not None:
            country_counts = df["country"].value_counts(dropna=False)
            print("country counts:")
            print(country_counts.to_string())
            checks = source_check(df, expected_prefix)
            print("checks:")
            for label, value in checks.items():
                print(f"  {label}: {value}")
            basic_summary[key] = {"rows": len(df), "countries": country_counts.to_dict(), **checks}
            if key in {"train_source1", "train_source2", "train_source3"}:
                random_samples[key] = df.sample(n=20, random_state=42)[SOURCE_COLUMNS].copy()
        else:
            print(f"memory: {bytes_text(int(df.memory_usage(deep=True).sum()))}")
            basic_summary[key] = {"rows": len(df), "memory": bytes_text(int(df.memory_usage(deep=True).sum()))}
        del df
        gc.collect()

    heading(2, "Raw samples (fixed random_state=42; no string truncation)")
    for key in ("train_source1", "train_source2", "train_source3"):
        print(f"\n{key} — 20 rows")
        print(random_samples[key].to_string(index=False, max_colwidth=None))
    random_samples.clear()
    gc.collect()

    heading(3, "Ground-truth statistics")
    ground_truth = load_tsv(FILES["train_ground_truth"][0])
    train_s1_ids = load_tsv(FILES["train_source1"][0])[["entity_id"]]
    empty_mask = ground_truth["matched_entity_ids"] == ""
    empty_count = int(empty_mask.sum())
    print(f"Source-1 rows in ground truth: {len(ground_truth):,}")
    print(f"Empty matched_entity_ids (singletons): {empty_count:,} / {len(ground_truth):,} ({empty_count / len(ground_truth) * 100:.2f}%)")
    nonempty = ground_truth.loc[~empty_mask, ["source1_entity_id", "matched_entity_ids"]].copy()
    match_counts = nonempty["matched_entity_ids"].str.split(",").str.len()
    distribution = match_counts.value_counts().sort_index()
    print("\nMatch-count distribution for non-singleton Source-1 entities:")
    print(distribution.to_string())
    print(f"min={match_counts.min():.0f}, max={match_counts.max():.0f}, mean={match_counts.mean():.4f}, median={match_counts.median():.1f}")
    pairs = nonempty.assign(matched_id=nonempty["matched_entity_ids"].str.split(",")).explode("matched_id")
    pairs = pairs[["source1_entity_id", "matched_id"]].reset_index(drop=True)
    prefix_counts = pairs["matched_id"].str.extract(r"^(S[23])-", expand=False).value_counts()
    total_matched_ids = len(pairs)
    print("\nMatched-ID source composition:")
    for prefix in ("S2", "S3"):
        count = int(prefix_counts.get(prefix, 0))
        print(f"{prefix}-: {count:,} / {total_matched_ids:,} ({count / total_matched_ids * 100:.2f}%)")
    source1_set = set(train_s1_ids["entity_id"])
    gt_set = set(ground_truth["source1_entity_id"])
    only_s1 = source1_set - gt_set
    only_gt = gt_set - source1_set
    gt_duplicate_s1 = int(ground_truth["source1_entity_id"].duplicated().sum())
    print("\nSource-1 / ground-truth coverage:")
    print(f"duplicate source1_entity_id rows in ground truth: {gt_duplicate_s1:,}")
    print(f"in train_source1 but not ground truth: {len(only_s1):,}")
    print(f"in ground truth but not train_source1: {len(only_gt):,}")
    if only_s1:
        print(f"  examples only in train_source1: {sorted(only_s1)[:10]}")
    if only_gt:
        print(f"  examples only in ground truth: {sorted(only_gt)[:10]}")
    del train_s1_ids
    gc.collect()

    heading(4, "Cross-check of the Source-1 deduplication assumption")
    matched_to_s1_counts = pairs.groupby("matched_id")["source1_entity_id"].nunique()
    shared_ids = matched_to_s1_counts[matched_to_s1_counts > 1].sort_values(ascending=False)
    print(f"Matched IDs associated with more than one distinct Source-1 ID: {len(shared_ids):,} / {len(matched_to_s1_counts):,}")
    if len(shared_ids):
        print("Largest numbers of distinct Source-1 associations:")
        print(shared_ids.head(10).to_string())
    else:
        print("No matched ID is shared by multiple Source-1 entities.")

    # Load Source-1 once for all later record-level joins.
    source1 = load_tsv(FILES["train_source1"][0]).set_index("entity_id", verify_integrity=True)
    duplicate_example_ids = set(shared_ids.head(10).index)
    duplicate_candidate_records: dict[str, pd.Series] = {}
    if duplicate_example_ids:
        for key in ("train_source2", "train_source3"):
            path, _ = FILES[key]
            for chunk in pd.read_csv(path, chunksize=200_000, **READ_KWARGS):
                selected = chunk.loc[chunk["entity_id"].isin(duplicate_example_ids)]
                for row in selected.itertuples(index=False):
                    duplicate_candidate_records[row.entity_id] = pd.Series(row._asdict())
    print("\nTen shared-ID examples with source records:")
    if not duplicate_example_ids:
        print("  None to print.")
    else:
        for matched_id in shared_ids.head(10).index:
            related_s1 = pairs.loc[pairs["matched_id"] == matched_id, "source1_entity_id"].drop_duplicates().tolist()
            print(f"\nmatched_id={matched_id}; associated Source-1 IDs={related_s1}")
            for s1_id in related_s1:
                print_source_record("S1", source1.loc[s1_id])
            candidate = duplicate_candidate_records.get(matched_id)
            if candidate is None:
                print("  Candidate source record: NOT FOUND")
            else:
                print_source_record(matched_id[:2], candidate)
    del duplicate_candidate_records
    gc.collect()

    heading(5, "Noise pattern inventory from matched pairs")
    pair_groups = {
        "S2": pairs.loc[pairs["matched_id"].str.startswith("S2-"), ["source1_entity_id", "matched_id"]],
        "S3": pairs.loc[pairs["matched_id"].str.startswith("S3-"), ["source1_entity_id", "matched_id"]],
    }
    comparison = Counter()
    missing_source1 = 0
    samples: dict[str, list[dict[str, str]]] = {"US": [], "India": []}
    for prefix, key in (("S2", "train_source2"), ("S3", "train_source3")):
        path, _ = FILES[key]
        # Step 4 verifies this lookup is one-to-one. Mapping candidate IDs
        # directly avoids rebuilding a multi-million-row merge table for each
        # input chunk.
        source1_by_matched = pair_groups[prefix].set_index("matched_id")["source1_entity_id"]
        for candidate_chunk in pd.read_csv(path, chunksize=200_000, **READ_KWARGS):
            candidate_chunk["source1_entity_id"] = candidate_chunk["entity_id"].map(source1_by_matched)
            joined = candidate_chunk.loc[candidate_chunk["source1_entity_id"].notna()].copy().reset_index(drop=True)
            if joined.empty:
                continue
            # Use an unnamed label array so the original Source-1 index name
            # remains `entity_id` after reset_index().
            s1_records = source1.reindex(joined["source1_entity_id"].to_numpy()).reset_index()
            missing = s1_records["business_name"].isna()
            missing_source1 += int(missing.sum())
            usable = ~missing
            joined = joined.loc[usable].reset_index(drop=True)
            s1_records = s1_records.loc[usable].reset_index(drop=True)
            candidate_name = joined["business_name"].str.lower().str.strip()
            candidate_address = joined["business_address"].str.lower().str.strip()
            s1_name = s1_records["business_name"].str.lower().str.strip()
            s1_address = s1_records["business_address"].str.lower().str.strip()
            name_equal = candidate_name.eq(s1_name)
            address_equal = candidate_address.eq(s1_address)
            comparison["pairs joined"] += len(joined)
            comparison["identical name"] += int(name_equal.sum())
            comparison["identical address"] += int(address_equal.sum())
            comparison["differ in both"] += int((~name_equal & ~address_equal).sum())
            # Collect only the first 25 examples per country encountered in
            # the deterministic scan. Aggregate counts above still cover every
            # matched pair, but report sampling remains constant-size.
            for country in samples:
                remaining = 25 - len(samples[country])
                if remaining <= 0:
                    continue
                positions = s1_records.index[s1_records["country"].eq(country)][:remaining]
                for idx in positions:
                    samples[country].append(
                        {
                            "s1_id": s1_records.at[idx, "entity_id"],
                            "s1_name": s1_records.at[idx, "business_name"],
                            "s1_address": s1_records.at[idx, "business_address"],
                            "matched_id": joined.at[idx, "entity_id"],
                            "candidate_name": joined.at[idx, "business_name"],
                            "candidate_address": joined.at[idx, "business_address"],
                            "country": country,
                        }
                    )
            del joined, s1_records
            gc.collect()
    print(f"Matched pairs joined to actual source records: {comparison['pairs joined']:,}")
    print(f"Matched pairs with missing Source-1 record: {missing_source1:,}")
    for label in ("identical name", "identical address", "differ in both"):
        print(f"{label}: {comparison[label]:,} / {comparison['pairs joined']:,} ({comparison[label] / comparison['pairs joined'] * 100:.2f}%)")
    print("\nDeterministic scan sample of matched pairs (25 US + 25 India when available):")
    for country in ("US", "India"):
        print(f"\n{country} examples ({len(samples[country])}):")
        for record in samples[country]:
            print(
                f"{record['s1_id']} | {record['s1_name']} | {record['s1_address']} "
                f"|| {record['matched_id']} | {record['candidate_name']} | {record['candidate_address']}"
            )
    del pair_groups, samples
    gc.collect()

    heading(6, "Field length and format check (training source files)")
    combined_names: Counter[str] = Counter()
    for key in ("train_source1", "train_source2", "train_source3"):
        path, _ = FILES[key]
        scanned = scan_source_file(path)
        combined_names.update(scanned["names"])
        rows = scanned["rows"]
        print(f"\n{key} ({rows:,} rows)")
        print(f"business_name length: min={scanned['name_min']}, max={scanned['name_max']}, mean={scanned['name_sum'] / rows:.2f}")
        print(f"business_address length: min={scanned['address_min']}, max={scanned['address_max']}, mean={scanned['address_sum'] / rows:.2f}")
        print(f"addresses with a 5–6 digit number: {scanned['zip_like']:,} ({scanned['zip_like'] / rows * 100:.2f}%); without: {rows - scanned['zip_like']:,} ({(rows - scanned['zip_like']) / rows * 100:.2f}%)")
        print(f"addresses containing 'near': {scanned['near']:,} ({scanned['near'] / rows * 100:.2f}%)")
    print("\nTop 30 business names across the three training sources:")
    for rank, (name, count) in enumerate(combined_names.most_common(30), start=1):
        print(f"{rank:>2}. {name!r}: {count:,}")
    placeholder_values = {"", "n/a", "na", "unknown", "none", "null", "-"}
    observed_placeholders = [(name, count) for name, count in combined_names.items() if name.strip().lower() in placeholder_values]
    print(f"Placeholder/junk-looking exact names observed: {observed_placeholders if observed_placeholders else 'none'}")
    del combined_names
    gc.collect()

    heading(7, "Test-set sanity")
    test_folder_labels = [p.name for p in (RAW / "test").iterdir() if "ground_truth" in p.name.lower() or "label" in p.name.lower()]
    for key in ("test_source1", "test_source2", "test_source3"):
        path, _ = FILES[key]
        test_df = load_tsv(path)
        print(f"\n{key}: {len(test_df):,} rows; training counterpart: {basic_summary[key.replace('test', 'train')]['rows']:,} rows")
        print(test_df["country"].value_counts(dropna=False).to_string())
        print(f"France present: {(test_df['country'] == 'France').any()}")
        del test_df
        gc.collect()
    print(f"\nTest-folder label/ground-truth-like files: {test_folder_labels if test_folder_labels else 'none found'}")

    heading(8, "Parquet copies for fast reload")
    PARQUET.mkdir(exist_ok=True)
    for key, (tsv_path, _) in FILES.items():
        df = load_tsv(tsv_path)
        parquet_path = PARQUET / f"{tsv_path.stem}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"{tsv_path.name}: TSV={bytes_text(tsv_path.stat().st_size)}; Parquet={bytes_text(parquet_path.stat().st_size)}")
        del df
        gc.collect()

    heading(9, "Clean summary")
    print("1. Row counts and country breakdown are printed in Steps 1 and 7.")
    print(f"2. Singletons: {empty_count:,} / {len(ground_truth):,} ({empty_count / len(ground_truth) * 100:.2f}%). The full match-count distribution is printed in Step 3.")
    print(f"3. Shared matched IDs: {len(shared_ids):,} IDs appear under more than one Source-1 entity.")
    print("4. Empirical name/address examples and exact-field comparison rates are printed in Step 5.")
    print("5. Data-quality checks (empty fields, duplicate IDs, prefixes, coverage, test labels) are printed in Steps 1, 3, and 7.")
    print(f"6. Parquet copies were written to: {PARQUET}")


if __name__ == "__main__":
    main()
