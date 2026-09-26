"""
src/path_b/b5_pipeline.py

Ticket B5: Export retrieval scores and embedding-cosine as extra features for Path A's model.
Generates and validates `scores/path_b_{val,test}.parquet` according to the shared team contract.

Outputs
-------
    scores/path_b_val.parquet
    scores/path_b_test.parquet

Columns
-------
    s1_id                  : str (Source 1 entity identifier)
    cand_id                : str (Candidate corpus entity identifier)
    score                  : float in [0.0, 1.0] (primary cross-encoder probability)
    cross_encoder_score    : float (explicit alias of score)
    bi_encoder_score       : float (cosine similarity from bi-encoder)
    retrieval_rank         : int (1-indexed rank from vector retrieval)
    retrieval_rr           : float (reciprocal rank 1.0 / retrieval_rank)
    score_margin           : float (score - top candidate score for this anchor)
    score_ratio            : float (score / top candidate score)
    bi_encoder_margin      : float (bi_encoder_score - top bi_encoder_score)
    bi_encoder_ratio       : float (ratio to top bi_encoder_score)
    rank_discounted_score  : float (score / log2(1 + retrieval_rank))
    is_match               : int in {0, 1} (ground truth label, validation split only)

Usage
-----
    # Demo run (synthetic mock data, verifies contract without GPU/data loading)
    python -m src.path_b.b5_pipeline --demo

    # Fast validation export (sampled)
    python -m src.path_b.b5_pipeline --split val --sample 200

    # Full validation export
    python -m src.path_b.b5_pipeline --split val --full

    # Test set export
    python -m src.path_b.b5_pipeline --split test --sample 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from src.common.paths import (
    TRAIN_DIR,
    TEST_DIR,
    SCORES_DIR,
    OUTPUT_DIR,
    SPLITS_DIR,
)
from src.common.data_loader import load_all_sources, load_test_sources
from src.path_b.serialization import serialize_dataframe
from src.path_b.encoder import DenseEncoder, DEFAULT_MODEL, get_default_device
from src.path_b.retrieval import VectorRetriever
from src.path_b.cross_encoder import (
    CrossEncoderReranker,
    DEFAULT_CROSS_MODEL,
)
from src.path_b.feature_export import (
    build_path_b_feature_dataframe,
    export_path_b_parquet,
    validate_path_b_dataframe,
    load_path_b_features,
)


# ---------------------------------------------------------------------------
# Core Pipeline Execution
# ---------------------------------------------------------------------------

def run_b5_export(
    split: str = "val",
    sample_size: Optional[int] = None,
    bi_encoder_model: str = DEFAULT_MODEL,
    cross_encoder_model: str = DEFAULT_CROSS_MODEL,
    candidates_file: Optional[str] = None,
    top_k_candidates: int = 30,
    encode_batch_size: int = 64,
    device: Optional[str] = None,
    output_path: Optional[str] = None,
    seed: int = 42,
    partition_by_country: bool = True,
    canonicalize: bool = True,
) -> pd.DataFrame:
    """Run neural retrieval + cross-encoder inference and export extra features.

    Parameters
    ----------
    split : str, optional
        'val' or 'test', by default 'val'.
    sample_size : Optional[int], optional
        Number of S1 entities to process (for testing/speed). None = all.
    bi_encoder_model : str, optional
        Bi-encoder checkpoint path or HuggingFace ID.
    cross_encoder_model : str, optional
        Cross-encoder checkpoint path or HuggingFace ID.
    candidates_file : Optional[str], optional
        Precomputed candidate pairs TSV (bypasses bi-encoder retrieval if given).
    top_k_candidates : int, optional
        Number of candidates to retrieve per S1 entity, by default 30.
    encode_batch_size : int, optional
        Inference batch size, by default 64.
    device : Optional[str], optional
        Device ('cpu', 'mps', 'cuda').
    output_path : Optional[str], optional
        Destination parquet file path. Defaults to scores/path_b_{split}.parquet.

    Returns
    -------
    pd.DataFrame
        Exported feature DataFrame.
    """
    start_time = time.time()
    device = device or get_default_device()

    # Determine output path
    if output_path is None:
        target_path = SCORES_DIR / f"path_b_{split}.parquet"
    else:
        target_path = Path(output_path)

    print("=" * 72)
    print(f"   TICKET B5: NEURAL FEATURE EXPORT FOR PATH A ENSEMBLE ({split.upper()})   ")
    print("=" * 72)
    print(f"Target Output:         {target_path}")
    print(f"Split:                 {split}")
    print(f"Bi-Encoder Model:      {bi_encoder_model}")
    print(f"Cross-Encoder Model:   {cross_encoder_model}")
    print(f"Top-K Candidates:      {top_k_candidates}")
    print(f"Device:                {device}")
    print(f"Sample Size:           {'FULL' if sample_size is None else f'{sample_size:,}'}")
    print("-" * 72)

    gt_map: Optional[dict[str, list[str]]] = None

    # ---- 1. Load Data ----------------------------------------------------
    if split == "val":
        print("\n[Step 1/5] Loading training and validation split data...")
        s1, s2, s3, gt = load_all_sources()
        gt_map = dict(zip(gt["source1_entity_id"], gt["matches"]))

        val_split_file = SPLITS_DIR / "val_s1_ids.txt"
        if val_split_file.exists():
            val_ids_list = [line.strip() for line in open(val_split_file) if line.strip()]
            val_ids_set = set(val_ids_list)
            s1_active = s1[s1["entity_id"].isin(val_ids_set)].copy().reset_index(drop=True)
            print(f"Loaded {len(s1_active):,} validation S1 entities from {val_split_file}")
        else:
            print("[Warning] val_s1_ids.txt not found; falling back to 20% split.")
            s1_active = s1.sample(frac=0.2, random_state=seed).reset_index(drop=True)

        if sample_size is not None and sample_size < len(s1_active):
            s1_active = s1_active.head(sample_size).reset_index(drop=True)
            print(f"Sampled down to {len(s1_active):,} S1 entities for export.")

        # Build corpus (S2 + S3)
        # In validation mode with sampling, include needed true matches + random confusers
        if sample_size is not None:
            active_s1_ids = set(s1_active["entity_id"])
            needed_ids: set[str] = set()
            for sid in active_s1_ids:
                needed_ids.update(gt_map.get(sid, []))
            s2_needed = s2[s2["entity_id"].isin(needed_ids)]
            s3_needed = s3[s3["entity_id"].isin(needed_ids)]
            confuser_cap = max(5_000, len(active_s1_ids) * top_k_candidates)
            s2_conf = s2[~s2["entity_id"].isin(needed_ids)].head(confuser_cap)
            s3_conf = s3[~s3["entity_id"].isin(needed_ids)].head(confuser_cap)
            corpus_df = pd.concat([s2_needed, s3_needed, s2_conf, s3_conf], ignore_index=True)
            corpus_df = corpus_df.drop_duplicates("entity_id").reset_index(drop=True)
        else:
            corpus_df = pd.concat([s2, s3], ignore_index=True).drop_duplicates("entity_id").reset_index(drop=True)

    elif split == "test":
        print("\n[Step 1/5] Loading test data sources...")
        s1, s2, s3 = load_test_sources()
        s1_active = s1.copy()
        if sample_size is not None and sample_size < len(s1_active):
            s1_active = s1_active.head(sample_size).reset_index(drop=True)
            print(f"Sampled down to {len(s1_active):,} test S1 entities.")
        corpus_df = pd.concat([s2, s3], ignore_index=True).drop_duplicates("entity_id").reset_index(drop=True)
    else:
        raise ValueError(f"Unknown split: {split!r}. Must be 'val' or 'test'.")

    print(f"Entities to score: S1={len(s1_active):,}, Corpus={len(corpus_df):,}")

    # ---- 2. Serialization ------------------------------------------------
    print("\n[Step 2/5] Serializing S1 and Corpus records...")
    s1_texts = serialize_dataframe(s1_active, drop_country=True, canonicalize=canonicalize)
    corpus_texts = serialize_dataframe(corpus_df, drop_country=True, canonicalize=canonicalize)
    s1_text_map = dict(zip(s1_active["entity_id"], s1_texts))
    corpus_text_map = dict(zip(corpus_df["entity_id"], corpus_texts))

    # ---- 3. Candidate Retrieval / Loading ---------------------------------
    bi_candidates: dict[str, list[tuple[str, float]]] = {}
    if candidates_file and Path(candidates_file).exists():
        print(f"\n[Step 3/5] Loading precomputed candidate pairs from {candidates_file}...")
        cand_df = pd.read_csv(candidates_file, sep="\t", dtype=str)
        # Parse into dict
        for _, row in cand_df.iterrows():
            sid = str(row["source1_entity_id"])
            cands = [c.strip() for c in str(row.get("candidate_entity_ids", "")).split(",") if c.strip()]
            bi_candidates[sid] = [(cid, 0.5) for cid in cands]
    else:
        print("\n[Step 3/5] Running Bi-Encoder retrieval to extract top-K candidates & cosine scores...")
        # Check if fine-tuned checkpoint exists; if not, use base model
        actual_bi_model = bi_encoder_model
        if not Path(actual_bi_model).exists() and Path("models/bi_encoder_b2").exists():
            actual_bi_model = "models/bi_encoder_b2"
            print(f"Found fine-tuned bi-encoder checkpoint at: {actual_bi_model}")

        bi_encoder = DenseEncoder(model_name_or_path=actual_bi_model, device=device)
        print("Encoding corpus...")
        corpus_embeddings = bi_encoder.encode(list(corpus_texts), batch_size=encode_batch_size, is_query=False)
        print("Encoding S1 queries...")
        s1_embeddings = bi_encoder.encode(list(s1_texts), batch_size=encode_batch_size, is_query=True)

        corpus_countries = corpus_df["country"].tolist() if "country" in corpus_df.columns else None
        s1_countries = s1_active["country"].tolist() if "country" in s1_active.columns else None

        retriever = VectorRetriever(
            corpus_embeddings=corpus_embeddings,
            corpus_ids=corpus_df["entity_id"].tolist(),
            corpus_countries=corpus_countries,
            device=device,
        )
        # Retrieve candidates with cosine similarities
        bi_candidates = retriever.search_top_k(
            query_embeddings=s1_embeddings,
            query_ids=s1_active["entity_id"].tolist(),
            query_countries=s1_countries,
            top_k=top_k_candidates,
            partition_by_country=partition_by_country and (corpus_countries is not None),
        )
        del bi_encoder, corpus_embeddings, s1_embeddings, retriever

    # ---- 4. Cross-Encoder Re-Ranking -------------------------------------
    print("\n[Step 4/5] Scoring candidate pairs with Cross-Encoder...")
    actual_cross_model = cross_encoder_model
    # Check for best augmented model from B4 or fine-tuned B3
    if not Path(actual_cross_model).exists():
        if Path("models/cross_encoder_b4_augmented").exists():
            actual_cross_model = "models/cross_encoder_b4_augmented"
            print(f"Using B4 augmented cross-encoder checkpoint: {actual_cross_model}")
        elif Path("models/cross_encoder_b3").exists():
            actual_cross_model = "models/cross_encoder_b3"
            print(f"Using B3 fine-tuned cross-encoder checkpoint: {actual_cross_model}")

    reranker = CrossEncoderReranker(model_name_or_path=actual_cross_model, device=device)
    cross_scored = reranker.rerank_candidates(
        candidate_dict=bi_candidates,
        s1_texts=s1_text_map,
        corpus_texts=corpus_text_map,
        batch_size=encode_batch_size,
        show_progress_bar=True,
    )
    del reranker

    # ---- 5. Feature Engineering and Parquet Export ------------------------
    print("\n[Step 5/5] Compiling feature DataFrame and exporting to Parquet...")
    feature_df = build_path_b_feature_dataframe(
        cross_scored_candidates=cross_scored,
        bi_encoder_candidates=bi_candidates,
        ground_truth=gt_map if split == "val" else None,
        include_margins=True,
        include_ratios=True,
    )

    out_file = export_path_b_parquet(
        df=feature_df,
        output_path=target_path,
        verbose=True,
    )

    # Summary report
    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print(f"             TICKET B5 FEATURE EXPORT SUMMARY ({split.upper()})             ")
    print("=" * 72)
    print(f"Total Candidate Pairs:      {len(feature_df):,}")
    print(f"Unique S1 Entities:         {feature_df['s1_id'].nunique():,}")
    print(f"Exported Parquet:           {out_file}")
    print(f"File Size:                  {out_file.stat().st_size / (1024 * 1024):.2f} MB")
    print("-" * 72)
    print("Feature Statistics:")
    num_cols = ["score", "bi_encoder_score", "retrieval_rank", "retrieval_rr", "score_margin", "bi_encoder_margin"]
    stats_df = feature_df[[c for c in num_cols if c in feature_df.columns]].describe().T[["mean", "std", "min", "max"]]
    print(stats_df.to_string())

    if "is_match" in feature_df.columns:
        print("-" * 72)
        match_mean = feature_df[feature_df["is_match"] == 1]["score"].mean()
        non_match_mean = feature_df[feature_df["is_match"] == 0]["score"].mean()
        print(f"Separation on Validation:")
        print(f"  Mean score (True Matches):    {match_mean:.4f}")
        print(f"  Mean score (Non-Matches):     {non_match_mean:.4f}")
        print(f"  Score Gap:                    {match_mean - non_match_mean:+.4f}")

    print("-" * 72)
    print("Preview of First 3 Exported Rows:")
    print(feature_df.head(3).to_dict(orient="records"))
    print("-" * 72)
    print(f"Total Execution Time:       {elapsed:.1f}s")
    print("=" * 72)

    return feature_df


# ---------------------------------------------------------------------------
# Demo Helper (Synthetic data, no GPU needed)
# ---------------------------------------------------------------------------

def run_demo() -> pd.DataFrame:
    """Run an end-to-end demo on synthetic data to verify contract compliance."""
    print("\n" + "=" * 60)
    print("  TICKET B5 DEMO: SYNTHETIC FEATURE EXPORT")
    print("=" * 60)

    # 1. Synthetic candidate inputs
    mock_cross_scores = {
        "S1-1001": [("S2-2001", 0.92), ("S3-3001", 0.45), ("S2-2002", 0.12)],
        "S1-1002": [("S3-3002", 0.88), ("S2-2003", 0.25)],
        "S1-1003": [("S2-2004", 0.08), ("S3-3003", 0.05)],  # Singleton
    }
    mock_bi_candidates = {
        "S1-1001": [("S2-2001", 0.85), ("S3-3001", 0.72), ("S2-2002", 0.61)],
        "S1-1002": [("S3-3002", 0.79), ("S2-2003", 0.58)],
        "S1-1003": [("S2-2004", 0.65), ("S3-3003", 0.60)],
    }
    mock_gt = {
        "S1-1001": ["S2-2001"],
        "S1-1002": ["S3-3002"],
        "S1-1003": [],  # Singleton
    }

    # 2. Build feature DataFrame
    df = build_path_b_feature_dataframe(
        cross_scored_candidates=mock_cross_scores,
        bi_encoder_candidates=mock_bi_candidates,
        ground_truth=mock_gt,
    )

    demo_path = SCORES_DIR / "path_b_demo.parquet"
    out = export_path_b_parquet(df, output_path=demo_path, verbose=True)

    # 3. Verify load
    loaded = load_path_b_features(out)
    assert len(loaded) == len(df), "Loaded row count mismatch"
    print("\n[Demo] Successfully validated parquet roundtrip.")
    print("Columns in Parquet:")
    for col in loaded.columns:
        print(f"  - {col:<24} ({loaded[col].dtype})")
    print("=" * 60 + "\n")
    return loaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ticket B5: Neural Feature Export for Path A/D Ensemble"
    )
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Data split to score and export (default: 'val')")
    parser.add_argument("--sample", type=int, default=200,
                        help="Sub-sample N S1 entities (default: 200, use 0 for full)")
    parser.add_argument("--full", action="store_true",
                        help="Use full split (overrides --sample)")
    parser.add_argument("--bi-encoder", type=str, default=DEFAULT_MODEL,
                        help=f"Bi-encoder checkpoint or model (default: {DEFAULT_MODEL})")
    parser.add_argument("--cross-encoder", type=str, default=DEFAULT_CROSS_MODEL,
                        help=f"Cross-encoder checkpoint or model (default: {DEFAULT_CROSS_MODEL})")
    parser.add_argument("--candidates-file", type=str, default=None,
                        help="Precomputed candidate pairs TSV to bypass retrieval")
    parser.add_argument("--k", type=int, default=30,
                        help="Top-K candidates per query (default: 30)")
    parser.add_argument("--encode-batch-size", type=int, default=64,
                        help="Inference batch size (default: 64)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device ('cpu', 'mps', 'cuda')")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom destination parquet path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-country-partition", action="store_true",
                        help="Disable country blocking in retrieval")
    parser.add_argument("--demo", action="store_true",
                        help="Run synthetic demo export and exit (fast, no GPU)")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    sample_size = None if args.full else (args.sample if args.sample > 0 else None)
    run_b5_export(
        split=args.split,
        sample_size=sample_size,
        bi_encoder_model=args.bi_encoder,
        cross_encoder_model=args.cross_encoder,
        candidates_file=args.candidates_file,
        top_k_candidates=args.k,
        encode_batch_size=args.encode_batch_size,
        device=args.device,
        output_path=args.output,
        seed=args.seed,
        partition_by_country=not args.no_country_partition,
    )


if __name__ == "__main__":
    main()
