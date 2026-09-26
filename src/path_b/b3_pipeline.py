"""
src/path_b/b3_pipeline.py

Ticket B3: End-to-end Cross-Encoder Re-Ranking Pipeline.

Workflow:
1. Load dataset (S1, S2, S3, ground truth).
2. Load or generate shared stratified validation split (splits/val_s1_ids.txt).
3. Serialize entity texts ('name | address') with canonicalization.
4. Run Dense Retrieval (Ticket B1/B2) to generate top-K candidate pools.
5. Build labeled (S1, candidate) training pairs with true matches and hard negative confusers.
6. Fine-tune Cross-Encoder with binary cross-entropy.
7. Score and re-rank validation candidate pairs to produce match probabilities.
8. Sweep decision thresholds to optimize Macro F0.5 with global 1-to-1 assignment.
9. Export scored pairs to scores/path_b_val.parquet and matching_results_path_b.tsv.
10. Print complete Ticket B3 Acceptance Report.

Usage:
    # Fast test run on 200 entities
    python -m src.path_b.b3_pipeline --sample 200 --epochs 1 --batch-size 16

    # Full training and evaluation
    python -m src.path_b.b3_pipeline --epochs 2 --batch-size 32 --lr 2e-5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd

from src.common.data_loader import load_all_sources
from src.common.paths import TRAIN_DIR, OUTPUT_DIR, SCORES_DIR, SPLITS_DIR
from eval.f05 import make_val_split
from src.path_b.serialization import serialize_dataframe
from src.path_b.encoder import DenseEncoder, DEFAULT_MODEL, get_default_device
from src.path_b.retrieval import VectorRetriever
from src.path_b.metrics import compute_recall_at_k, print_recall_report
from src.path_b.cross_dataset import prepare_cross_encoder_examples
from src.path_b.cross_encoder import (
    CrossEncoderReranker,
    fine_tune_cross_encoder,
    DEFAULT_CROSS_MODEL,
)
from src.path_b.cross_metrics import (
    find_optimal_threshold,
    export_scores_parquet,
    export_matching_results_tsv,
)


def run_b3_pipeline(
    sample_size: int | None = 500,
    val_fraction: float = 0.2,
    bi_encoder_model: str = DEFAULT_MODEL,
    cross_encoder_model: str = DEFAULT_CROSS_MODEL,
    output_model_path: str = "models/cross_encoder_b3",
    epochs: int = 1,
    batch_size: int = 16,
    encode_batch_size: int = 64,
    learning_rate: float = 2e-5,
    top_k_candidates: int = 30,
    max_negs_per_pos: int = 4,
    enforce_one_to_one: bool = True,
    canonicalize: bool = True,
    device: str | None = None,
    seed: int = 42,
    candidates_file: str | None = None,
    scores_parquet_path: str = str(SCORES_DIR / "path_b_val.parquet"),
    results_tsv_path: str = str(OUTPUT_DIR / "matching_results_path_b.tsv"),
    candidates_tsv_path: str = str(OUTPUT_DIR / "candidate_pairs_path_b.tsv"),
    # B4 Augmentation
    augment_positives: bool = False,
    n_augments: int = 2,
    aug_min_ops: int = 1,
    aug_max_ops: int = 3,
) -> dict:
    """Execute end-to-end Cross-Encoder re-ranking and evaluation."""
    start_time = time.time()
    print("=" * 70)
    print("   STARTING TICKET B3: CROSS-ENCODER RE-RANKING PIPELINE     ")
    print("=" * 70)
    print(f"Bi-Encoder Model:      {bi_encoder_model}")
    print(f"Cross-Encoder Model:   {cross_encoder_model}")
    print(f"Device:                {device or 'auto'}")
    print(f"Epochs:                {epochs}")
    print(f"Batch Size:            {batch_size} (Train), {encode_batch_size} (Inference)")
    print(f"Learning Rate:         {learning_rate}")
    print(f"Top-K Candidates:      {top_k_candidates}")
    print(f"Negatives per Pos:     {max_negs_per_pos}")
    print(f"1-to-1 Constraint:     {enforce_one_to_one}")
    print(f"Candidates Source:     {candidates_file or 'Dense Retrieval (on-the-fly)'}")
    print(f"Augment Positives:     {augment_positives}" + (f" (x{n_augments}, ops={aug_min_ops}-{aug_max_ops})" if augment_positives else ""))
    print(f"Seed:                  {seed}")
    print(f"Sample Size:           {'FULL DATASET' if sample_size is None else f'{sample_size:,}'}")
    print("-" * 70)

    # 1. Load data
    print("\n[Step 1/8] Loading dataset...")
    s1, s2, s3, gt = load_all_sources()
    gt_map: dict[str, list[str]] = dict(zip(gt["source1_entity_id"], gt["matches"]))

    # 2. Train / Val split (shared contract)
    split_file = SPLITS_DIR / "val_s1_ids.txt"
    if split_file.exists():
        print(f"\n[Step 2/8] Loading existing shared split from {split_file}...")
        with open(split_file, "r") as f:
            val_ids_all = set(line.strip() for line in f if line.strip())
        val_mask = s1["entity_id"].isin(val_ids_all)
        s1_val_full = s1[val_mask].copy()
        s1_train_full = s1[~val_mask].copy()
    else:
        print(f"\n[Step 2/8] Generating stratified shared split to {split_file}...")
        train_ids_list, val_ids_list = make_val_split(
            s1=s1,
            ground_truth=gt_map,
            val_fraction=val_fraction,
            random_state=seed,
            output_path=split_file,
        )
        val_ids_set = set(val_ids_list)
        val_mask = s1["entity_id"].isin(val_ids_set)
        s1_val_full = s1[val_mask].copy()
        s1_train_full = s1[~val_mask].copy()

    # Subsample if requested
    if sample_size is not None and sample_size < len(s1):
        print(f"\n[Subsampling] Sampling {sample_size:,} entities for Ticket B3 run...")
        val_sample_n = max(10, int(sample_size * val_fraction))
        train_sample_n = sample_size - val_sample_n

        s1_val = s1_val_full.head(val_sample_n).copy().reset_index(drop=True)
        s1_train = s1_train_full.head(train_sample_n).copy().reset_index(drop=True)
        active_s1 = pd.concat([s1_train, s1_val], ignore_index=True)
        active_s1_ids = set(active_s1["entity_id"])

        true_corpus_ids = set()
        for sid in active_s1_ids:
            true_corpus_ids.update(gt_map.get(sid, []))

        s2_needed = s2[s2["entity_id"].isin(true_corpus_ids)]
        s3_needed = s3[s3["entity_id"].isin(true_corpus_ids)]

        confuser_size = sample_size * 5
        s2_confusers = s2[~s2["entity_id"].isin(true_corpus_ids)].head(confuser_size)
        s3_confusers = s3[~s3["entity_id"].isin(true_corpus_ids)].head(confuser_size)

        corpus_active = pd.concat([s2_needed, s3_needed, s2_confusers, s3_confusers], ignore_index=True)
    else:
        s1_val = s1_val_full.reset_index(drop=True)
        s1_train = s1_train_full.reset_index(drop=True)
        corpus_active = pd.concat([s2, s3], ignore_index=True)

    # Deduplicate entity IDs
    s1_train = s1_train.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    s1_val = s1_val_full.drop_duplicates(subset=["entity_id"]).reset_index(drop=True) if sample_size is None else s1_val.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    corpus_active = corpus_active.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

    print(f"Train Entities: {len(s1_train):,}")
    print(f"Val Entities:   {len(s1_val):,}")
    print(f"Corpus Records: {len(corpus_active):,}")

    # 3. Serialize records
    print("\n[Step 3/8] Serializing text records ('name | address')...")
    train_s1_texts = serialize_dataframe(s1_train, drop_country=True, canonicalize=canonicalize)
    val_s1_texts = serialize_dataframe(s1_val, drop_country=True, canonicalize=canonicalize)
    corpus_texts = serialize_dataframe(corpus_active, drop_country=True, canonicalize=canonicalize)

    train_s1_map = dict(zip(s1_train["entity_id"], train_s1_texts))
    val_s1_map = dict(zip(s1_val["entity_id"], val_s1_texts))
    corpus_map = dict(zip(corpus_active["entity_id"], corpus_texts))
    corpus_ids = corpus_active["entity_id"].tolist()
    corpus_countries = corpus_active["country"].tolist() if "country" in corpus_active.columns else None

    # 4. Dense Retrieval for Candidates
    if candidates_file and Path(candidates_file).exists():
        print(f"\n[Step 4/8] Loading precomputed candidates from {candidates_file}...")
        all_candidates = VectorRetriever.load_candidate_pairs_tsv(candidates_file)
        train_s1_set = set(s1_train["entity_id"])
        val_s1_set = set(s1_val["entity_id"])
        train_candidates = {sid: all_candidates.get(sid, []) for sid in train_s1_set}
        val_candidates = {sid: all_candidates.get(sid, []) for sid in val_s1_set}
    else:
        print(f"\n[Step 4/8] Generating Top-{top_k_candidates} candidates via Bi-Encoder...")
        bi_encoder = DenseEncoder(model_name_or_path=bi_encoder_model, device=device)
        corpus_embeddings = bi_encoder.encode(corpus_texts, batch_size=encode_batch_size, is_query=False)

        train_embeddings = bi_encoder.encode(train_s1_texts, batch_size=encode_batch_size, is_query=True)
        val_embeddings = bi_encoder.encode(val_s1_texts, batch_size=encode_batch_size, is_query=True)

        retriever = VectorRetriever(
            corpus_embeddings=corpus_embeddings,
            corpus_ids=corpus_ids,
            corpus_countries=corpus_countries,
            device=device,
        )

        print("[Retrieval] Searching candidates for Train...")
        train_candidates = retriever.search_top_k(
            query_embeddings=train_embeddings,
            query_ids=s1_train["entity_id"].tolist(),
            query_countries=s1_train["country"].tolist() if "country" in s1_train.columns else None,
            top_k=top_k_candidates,
            partition_by_country=True,
        )

        print("[Retrieval] Searching candidates for Validation...")
        val_candidates = retriever.search_top_k(
            query_embeddings=val_embeddings,
            query_ids=s1_val["entity_id"].tolist(),
            query_countries=s1_val["country"].tolist() if "country" in s1_val.columns else None,
            top_k=top_k_candidates,
            partition_by_country=True,
        )

    # Measure Bi-Encoder Recall ceiling
    bi_encoder_metrics = compute_recall_at_k(
        retrieved_candidates=val_candidates,
        ground_truth=gt_map,
        k_list=[k for k in [5, 10, 20, top_k_candidates] if k <= top_k_candidates],
    )
    print("\n--- BI-ENCODER CANDIDATE RECALL REPORT ---")
    print_recall_report(bi_encoder_metrics)

    # 5. Build Cross-Encoder Training Pairs
    print("\n[Step 5/8] Building Cross-Encoder labeled training pairs...")
    train_examples = prepare_cross_encoder_examples(
        s1_ids=s1_train["entity_id"].tolist(),
        s1_texts=train_s1_map,
        corpus_texts=corpus_map,
        ground_truth=gt_map,
        retrieval_candidates=train_candidates,
        max_negatives_per_positive=max_negs_per_pos,
        corpus_ids=corpus_ids,
        random_seed=seed,
        augment_positives=augment_positives,
        n_augments=n_augments,
        aug_min_ops=aug_min_ops,
        aug_max_ops=aug_max_ops,
        aug_seed=seed + 57,
    )

    # 6. Fine-tune Cross-Encoder
    print("\n[Step 6/8] Fine-tuning Cross-Encoder on labeled candidate pairs...")
    fine_tune_cross_encoder(
        train_examples=train_examples,
        output_path=output_model_path,
        base_model_name=cross_encoder_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )

    # 7. Score and Re-rank Validation Pairs
    print("\n[Step 7/8] Scoring and re-ranking Validation candidate pairs with fine-tuned Cross-Encoder...")
    reranker = CrossEncoderReranker(model_name_or_path=output_model_path, device=device)
    val_scored_candidates = reranker.rerank_candidates(
        candidate_dict=val_candidates,
        s1_texts=val_s1_map,
        corpus_texts=corpus_map,
        batch_size=encode_batch_size,
        show_progress_bar=True,
    )

    # 8. Threshold Tuning & Acceptance Evaluation
    print("\n[Step 8/8] Evaluating ROC-AUC and optimizing decision threshold for Macro F0.5...")
    val_gt = {sid: gt_map.get(sid, []) for sid in s1_val["entity_id"]}
    eval_result = find_optimal_threshold(
        scored_candidates=val_scored_candidates,
        ground_truth=val_gt,
        enforce_one_to_one=enforce_one_to_one,
    )

    best_threshold = eval_result["best_threshold"]
    best_min_top_score = eval_result["best_min_top_score"]
    best_macro_f05 = eval_result["best_macro_f05"]
    auc_metrics = eval_result["auc_metrics"]
    best_preds = eval_result["best_predictions"]

    # Export outputs according to shared contract
    print("\n--- EXPORTING ARTIFACTS ACCORDING TO SHARED CONTRACT ---")
    parquet_out = export_scores_parquet(
        scored_candidates=val_scored_candidates,
        output_path=scores_parquet_path,
        bi_encoder_candidates=val_candidates,
        ground_truth=gt_map,
    )
    tsv_out = export_matching_results_tsv(best_preds, results_tsv_path)
    VectorRetriever.export_candidate_pairs_tsv(val_candidates, candidates_tsv_path)
    print(f"[Export] Saved candidate pairs TSV to {candidates_tsv_path}")

    # Summary Report
    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("               TICKET B3 ACCEPTANCE SUMMARY REPORT               ")
    print("=" * 72)
    print(f"Validation Entities:         {len(s1_val):,}")
    print(f"Bi-Encoder Recall@{top_k_candidates}:      {bi_encoder_metrics['macro_recall_at_k'].get(top_k_candidates, 0.0) * 100:.2f}%")
    print(f"Cross-Encoder Pair ROC-AUC:  {auc_metrics['roc_auc'] * 100:.2f}%")
    print(f"Cross-Encoder Pair PR-AUC:   {auc_metrics['pr_auc'] * 100:.2f}%")
    thresh_str = f"{best_threshold:.2f} (Trigger: {best_min_top_score:.2f})" if best_min_top_score is not None else f"{best_threshold:.2f}"
    print(f"Optimal Decision Threshold:  {thresh_str}")
    print(f"Validation Macro F0.5:       {best_macro_f05 * 100:.2f}%")
    print(f"1-to-1 Constraint Applied:   {enforce_one_to_one}")
    print("-" * 72)
    print(f"Scores Parquet File:         {parquet_out}")
    print(f"Predictions TSV File:        {tsv_out}")
    print(f"Candidates TSV File:         {candidates_tsv_path}")
    print(f"Saved Checkpoint:            {output_model_path}")
    print(f"Total Execution Time:        {elapsed:.1f}s")
    print("=" * 72)

    return {
        "best_threshold": best_threshold,
        "best_min_top_score": best_min_top_score,
        "best_macro_f05": best_macro_f05,
        "auc_metrics": auc_metrics,
        "bi_encoder_metrics": bi_encoder_metrics,
        "scores_parquet_path": str(parquet_out),
        "results_tsv_path": str(tsv_out),
        "checkpoint_path": output_model_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket B3: Cross-Encoder Re-Ranking Pipeline")
    parser.add_argument("--sample", type=int, default=300, help="Number of S1 entities (default: 300)")
    parser.add_argument("--full", action="store_true", help="Run on full dataset")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    parser.add_argument("--bi-encoder", type=str, default=DEFAULT_MODEL, help="Bi-Encoder model for candidate retrieval")
    parser.add_argument("--cross-encoder", type=str, default=DEFAULT_CROSS_MODEL, help="Cross-Encoder model for re-ranking")
    parser.add_argument("--output-model", type=str, default="models/cross_encoder_b3", help="Checkpoint save directory")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--encode-batch-size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--k", type=int, default=30, help="Top-K candidates to retrieve for re-ranking")
    parser.add_argument("--max-negs-per-pos", type=int, default=4, help="Negative sampling ratio for training")
    parser.add_argument("--candidates-file", type=str, default=None, help="Path to precomputed candidate pairs TSV")
    parser.add_argument("--no-1to1", action="store_true", help="Disable 1-to-1 global assignment constraint")
    parser.add_argument("--no-canonicalize", action="store_true", help="Disable canonical text normalization")
    parser.add_argument("--augment", action="store_true", help="(B4) Enable synthetic noise augmentation on positive pairs")
    parser.add_argument("--n-augments", type=int, default=2, help="(B4) Number of noisy copies per positive pair (default: 2)")
    parser.add_argument("--aug-min-ops", type=int, default=1, help="(B4) Min augmentation operations per copy (default: 1)")
    parser.add_argument("--aug-max-ops", type=int, default=3, help="(B4) Max augmentation operations per copy (default: 3)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'mps', 'cuda')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    sample_size = None if args.full else args.sample
    run_b3_pipeline(
        sample_size=sample_size,
        val_fraction=args.val_fraction,
        bi_encoder_model=args.bi_encoder,
        cross_encoder_model=args.cross_encoder,
        output_model_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        learning_rate=args.lr,
        top_k_candidates=args.k,
        max_negs_per_pos=args.max_negs_per_pos,
        candidates_file=args.candidates_file,
        enforce_one_to_one=not args.no_1to1,
        canonicalize=not args.no_canonicalize,
        augment_positives=args.augment,
        n_augments=args.n_augments,
        aug_min_ops=args.aug_min_ops,
        aug_max_ops=args.aug_max_ops,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
