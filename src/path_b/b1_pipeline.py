"""
src/path_b/b1_pipeline.py

Ticket B1: End-to-end Neural Retrieval & Recall Evaluation Pipeline.

Usage:
    # Run fast evaluation on a sample of 1,000 entities
    python -m src.path_b.b1_pipeline --sample 1000 --k 50

    # With canonical text normalization and country partitioning (recommended)
    python -m src.path_b.b1_pipeline --sample 1000 --k 50 --canonicalize --partition-by-country

    # Export official candidate_pairs.tsv
    python -m src.path_b.b1_pipeline --sample 1000 --k 50 --save-candidates output/candidate_pairs.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import pandas as pd

from src.common.data_loader import load_all_sources
from src.common.paths import TRAIN_DIR, OUTPUT_DIR
from src.path_b.serialization import serialize_dataframe
from src.path_b.encoder import DenseEncoder, DEFAULT_MODEL
from src.path_b.retrieval import VectorRetriever
from src.path_b.metrics import compute_recall_at_k, print_recall_report


def run_b1_retrieval(
    sample_size: int | None = 1000,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 50,
    batch_size: int = 256,
    drop_country: bool = True,
    canonicalize: bool = False,
    partition_by_country: bool = False,
    device: str | None = None,
    save_candidates_path: str | Path | None = None,
) -> dict:
    """Execute the full B1 retrieval and recall evaluation workflow."""
    start_time = time.time()
    print("=" * 65)
    print("       STARTING TICKET B1: NEURAL RETRIEVAL & BLOCKING        ")
    print("=" * 65)
    print(f"Model:                {model_name}")
    print(f"Device:               {device or 'auto'}")
    print(f"Top-K:                {top_k}")
    print(f"Drop Country in Text: {drop_country}")
    print(f"Canonicalize Text:    {canonicalize}")
    print(f"Country Partition:    {partition_by_country}")
    print(f"Sample Size:          {'FULL DATASET' if sample_size is None else f'{sample_size:,}'}")
    print("-" * 65)

    # 1. Load data
    print("\n[Step 1/5] Loading datasets from TRAIN_DIR...")
    s1, s2, s3, gt = load_all_sources()
    print(f"Loaded: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}, GT={len(gt):,}")

    # Build ground truth mapping
    gt_map: dict[str, list[str]] = dict(zip(gt["source1_entity_id"], gt["matches"]))

    # If sample_size is set, filter S1 entities and retain relevant corpus subset + confusers
    if sample_size is not None and sample_size < len(s1):
        print(f"\n[Subsampling] Sampling {sample_size:,} S1 entities for evaluation...")
        s1_sample = s1.head(sample_size).copy()
        s1_ids_set = set(s1_sample["entity_id"])

        # Collect all true positive match IDs for this sample
        true_cand_ids = set()
        for s1_id in s1_ids_set:
            true_cand_ids.update(gt_map.get(s1_id, []))

        # Sample additional confusers from S2 and S3 for realistic retrieval pool
        s2_needed = s2[s2["entity_id"].isin(true_cand_ids)]
        s3_needed = s3[s3["entity_id"].isin(true_cand_ids)]

        # Add background confusers
        confuser_size = sample_size * 5
        s2_confusers = s2[~s2["entity_id"].isin(true_cand_ids)].head(confuser_size)
        s3_confusers = s3[~s3["entity_id"].isin(true_cand_ids)].head(confuser_size)

        s2_sample = pd.concat([s2_needed, s2_confusers], ignore_index=True)
        s3_sample = pd.concat([s3_needed, s3_confusers], ignore_index=True)
    else:
        s1_sample = s1
        s2_sample = s2
        s3_sample = s3

    # Combine corpus (S2 + S3)
    corpus = pd.concat([s2_sample, s3_sample], ignore_index=True)
    print(f"Active Query Pool (S1): {len(s1_sample):,}")
    print(f"Active Corpus Pool (S2+S3): {len(corpus):,}")

    # 2. Text Serialization
    print("\n[Step 2/5] Serializing text strings ('name | address')...")
    s1_texts = serialize_dataframe(s1_sample, drop_country=drop_country, canonicalize=canonicalize)
    corpus_texts = serialize_dataframe(corpus, drop_country=drop_country, canonicalize=canonicalize)

    # 3. Dense Embedding
    print("\n[Step 3/5] Encoding text into dense vectors...")
    encoder = DenseEncoder(model_name_or_path=model_name, device=device)
    
    print(f"Encoding {len(corpus_texts):,} corpus records...")
    corpus_embeddings = encoder.encode(corpus_texts, batch_size=batch_size, is_query=False)

    print(f"Encoding {len(s1_texts):,} query records...")
    s1_embeddings = encoder.encode(s1_texts, batch_size=batch_size, is_query=True)

    # 4. Vector Retrieval
    print(f"\n[Step 4/5] Building index & searching Top-{top_k} nearest candidates...")
    corpus_ids = corpus["entity_id"].tolist()
    query_ids = s1_sample["entity_id"].tolist()
    corpus_countries = corpus["country"].tolist() if "country" in corpus.columns else None
    query_countries = s1_sample["country"].tolist() if "country" in s1_sample.columns else None

    retriever = VectorRetriever(
        corpus_embeddings=corpus_embeddings,
        corpus_ids=corpus_ids,
        corpus_countries=corpus_countries,
        device=device,
    )
    results = retriever.search_top_k(
        query_embeddings=s1_embeddings,
        query_ids=query_ids,
        query_countries=query_countries,
        top_k=top_k,
        partition_by_country=partition_by_country,
    )

    # 5. Metrics Evaluation
    print("\n[Step 5/5] Evaluating Recall@K against Ground Truth...")
    eval_k_list = [k for k in [5, 10, 20, 50, 100] if k <= top_k]
    if top_k not in eval_k_list:
        eval_k_list.append(top_k)
    eval_k_list = sorted(set(eval_k_list))

    metrics = compute_recall_at_k(
        retrieved_candidates=results,
        ground_truth=gt_map,
        k_list=eval_k_list,
    )
    print_recall_report(metrics)

    # Optional: Save candidates
    if save_candidates_path:
        out_path = Path(save_candidates_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if str(out_path).endswith(".tsv"):
            retriever.export_candidate_pairs_tsv(results, out_path)
        else:
            cand_df = retriever.to_dataframe(results)
            cand_df.to_parquet(out_path, index=False)
            print(f"Saved {len(cand_df):,} candidate pairs to {out_path}")

    elapsed = time.time() - start_time
    print(f"Ticket B1 completed in {elapsed:.1f}s.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket B1: Dense Multilingual Retrieval")
    parser.add_argument("--sample", type=int, default=1000, help="Number of S1 entities to evaluate (default: 1000)")
    parser.add_argument("--full", action="store_true", help="Run on full dataset instead of sampling")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model identifier")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'mps', 'cuda')")
    parser.add_argument("--k", type=int, default=50, help="Number of candidates to retrieve per entity (default: 50)")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size (default: 256)")
    parser.add_argument("--keep-country", action="store_true", help="Include country in serialized string")
    parser.add_argument("--canonicalize", action="store_true", help="Apply legal suffix and address expansion")
    parser.add_argument("--partition-by-country", action="store_true", help="Perform country-partitioned vector retrieval")
    parser.add_argument("--save-candidates", type=str, default=None, help="Path to save candidate pairs TSV/Parquet")
    args = parser.parse_args()

    sample_size = None if args.full else args.sample
    run_b1_retrieval(
        sample_size=sample_size,
        model_name=args.model,
        top_k=args.k,
        batch_size=args.batch_size,
        drop_country=not args.keep_country,
        canonicalize=args.canonicalize,
        partition_by_country=args.partition_by_country,
        device=args.device,
        save_candidates_path=args.save_candidates,
    )


if __name__ == "__main__":
    main()
