"""
src/path_b/b2_pipeline.py

Ticket B2: End-to-end Contrastive Bi-Encoder Fine-Tuning & Acceptance Verification.

Workflow:
1. Load dataset (train S1, S2, S3, ground truth).
2. Split S1 entities into Train and Validation sets (grouped by entity).
3. Run B1 Zero-Shot retrieval on Train to mine hard negative confusers.
4. Measure Zero-Shot Recall@K on the held-out Validation set.
5. Fine-tune bi-encoder with MultipleNegativesRankingLoss.
6. Measure Fine-Tuned Recall@K on the Validation set.
7. Print side-by-side comparison table proving recall improvement.
8. Save fine-tuned model checkpoint.

Usage:
    # Fast verification run on 200 sample entities
    python -m src.path_b.b2_pipeline --sample 200 --epochs 1 --batch-size 16

    # Full training run
    python -m src.path_b.b2_pipeline --epochs 2 --batch-size 32 --lr 2e-5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd

from src.common.data_loader import load_all_sources
from src.common.paths import TRAIN_DIR, OUTPUT_DIR
from src.path_b.serialization import serialize_dataframe
from src.path_b.encoder import DenseEncoder, DEFAULT_MODEL, get_default_device
from src.path_b.retrieval import VectorRetriever
from src.path_b.metrics import compute_recall_at_k, print_recall_report
from src.path_b.hard_negatives import mine_triplets_from_retrieval
from src.path_b.trainer import fine_tune_bi_encoder


def run_b2_pipeline(
    sample_size: int | None = 500,
    val_fraction: float = 0.2,
    base_model_name: str = DEFAULT_MODEL,
    output_model_path: str = "models/bi_encoder_b2",
    epochs: int = 1,
    batch_size: int = 16,
    encode_batch_size: int = 64,
    learning_rate: float = 2e-5,
    top_k_negatives: int = 20,
    max_negs_per_pos: int = 2,
    max_triplets_per_anchor: int = 8,
    top_k_eval: int = 50,
    canonicalize: bool = True,
    device: str | None = None,
    seed: int = 42,
    export_candidates_path: str | None = None,
) -> dict:
    """Execute end-to-end bi-encoder contrastive fine-tuning and evaluation.

    Parameters
    ----------
    sample_size : int | None, optional
        Number of S1 entities to sample (None for full dataset), by default 500.
    val_fraction : float, optional
        Fraction of S1 entities for validation, by default 0.2.
    base_model_name : str, optional
        Base SentenceTransformer identifier, by default DEFAULT_MODEL.
    output_model_path : str, optional
        Save directory for fine-tuned checkpoint, by default "models/bi_encoder_b2".
    epochs : int, optional
        Training epochs, by default 1.
    batch_size : int, optional
        Training mini-batch size, by default 16.
    encode_batch_size : int, optional
        Inference batch size for dense encoding, by default 64.
    learning_rate : float, optional
        AdamW learning rate, by default 2e-5.
    top_k_negatives : int, optional
        Number of vector retrieval confusers to retrieve for mining, by default 20.
    max_negs_per_pos : int, optional
        Max hard negatives paired with each positive, by default 2.
    max_triplets_per_anchor : int, optional
        Max triplets per S1 anchor to maintain balanced batches, by default 8.
    top_k_eval : int, optional
        Max rank evaluated for Recall@K, by default 50.
    canonicalize : bool, optional
        Apply legal suffix / address standardization, by default True.
    device : str | None, optional
        Device override ('cpu', 'mps', 'cuda'), by default auto-detected.
    seed : int, optional
        Random seed for reproducible splits and training, by default 42.
    export_candidates_path : str | None, optional
        Optional path to save official candidate_pairs.tsv for Ticket B3 handoff.

    Returns
    -------
    dict
        Evaluation report dictionary with zero_shot_metrics, tuned_metrics, and deltas.
    """
    start_time = time.time()
    print("=" * 68)
    print("   STARTING TICKET B2: CONTRASTIVE BI-ENCODER FINE-TUNING    ")
    print("=" * 68)
    print(f"Base Model:       {base_model_name}")
    print(f"Device:           {device or 'auto'}")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size} (Train), {encode_batch_size} (Encode)")
    print(f"Learning Rate:    {learning_rate}")
    print(f"Seed:             {seed}")
    print(f"Output Checkpoint: {output_model_path}")
    print(f"Sample Size:      {'FULL DATASET' if sample_size is None else f'{sample_size:,}'}")
    print("-" * 68)

    # 1. Load data
    print("\n[Step 1/6] Loading datasets from TRAIN_DIR...")
    s1, s2, s3, gt = load_all_sources()
    gt_map: dict[str, list[str]] = dict(zip(gt["source1_entity_id"], gt["matches"]))

    # Subsample if requested
    if sample_size is not None and sample_size < len(s1):
        print(f"\n[Subsampling] Sampling {sample_size:,} S1 entities for Ticket B2 run...")
        s1_active = s1.head(sample_size).copy()
        s1_ids_set = set(s1_active["entity_id"])

        true_cand_ids = set()
        for s1_id in s1_ids_set:
            true_cand_ids.update(gt_map.get(s1_id, []))

        s2_needed = s2[s2["entity_id"].isin(true_cand_ids)]
        s3_needed = s3[s3["entity_id"].isin(true_cand_ids)]

        confuser_size = sample_size * 5
        s2_confusers = s2[~s2["entity_id"].isin(true_cand_ids)].head(confuser_size)
        s3_confusers = s3[~s3["entity_id"].isin(true_cand_ids)].head(confuser_size)

        corpus_active = pd.concat([s2_needed, s3_needed, s2_confusers, s3_confusers], ignore_index=True)
    else:
        s1_active = s1
        corpus_active = pd.concat([s2, s3], ignore_index=True)

    # Ensure entity IDs are strictly deduplicated
    s1_active = s1_active.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    corpus_active = corpus_active.drop_duplicates(subset=["entity_id"]).reset_index(drop=True)

    # 2. Train / Validation Split (grouped by S1 entity)
    print(f"\n[Step 2/6] Splitting {len(s1_active):,} S1 entities into Train / Val...")
    n_val = max(10, int(len(s1_active) * val_fraction))
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(s1_active))
    val_indices = shuffled_indices[:n_val]
    train_indices = shuffled_indices[n_val:]

    s1_val = s1_active.iloc[val_indices].reset_index(drop=True)
    s1_train = s1_active.iloc[train_indices].reset_index(drop=True)

    print(f"Train S1 Entities: {len(s1_train):,}")
    print(f"Val S1 Entities:   {len(s1_val):,}")

    # Serialize texts
    print("\n[Step 3/6] Serializing entity records ('name | address')...")
    train_s1_texts = serialize_dataframe(s1_train, drop_country=True, canonicalize=canonicalize)
    val_s1_texts = serialize_dataframe(s1_val, drop_country=True, canonicalize=canonicalize)
    corpus_texts = serialize_dataframe(corpus_active, drop_country=True, canonicalize=canonicalize)

    train_s1_map = dict(zip(s1_train["entity_id"], train_s1_texts))
    val_s1_map = dict(zip(s1_val["entity_id"], val_s1_texts))
    corpus_map = dict(zip(corpus_active["entity_id"], corpus_texts))
    corpus_ids = corpus_active["entity_id"].tolist()
    corpus_countries = corpus_active["country"].tolist() if "country" in corpus_active.columns else None

    # 4. Zero-shot baseline retrieval and evaluation on Val
    print(f"\n[Step 4/6] Measuring ZERO-SHOT baseline recall on Validation set...")
    base_encoder = DenseEncoder(model_name_or_path=base_model_name, device=device)
    corpus_embeddings_zero = base_encoder.encode(corpus_texts, batch_size=encode_batch_size, is_query=False)
    val_embeddings_zero = base_encoder.encode(val_s1_texts, batch_size=encode_batch_size, is_query=True)

    val_retriever_zero = VectorRetriever(
        corpus_embeddings=corpus_embeddings_zero,
        corpus_ids=corpus_ids,
        corpus_countries=corpus_countries,
        device=device,
    )
    val_results_zero = val_retriever_zero.search_top_k(
        query_embeddings=val_embeddings_zero,
        query_ids=s1_val["entity_id"].tolist(),
        query_countries=s1_val["country"].tolist() if "country" in s1_val.columns else None,
        top_k=top_k_eval,
        partition_by_country=True,
    )

    eval_k_list = [k for k in [5, 10, 20, 50] if k <= top_k_eval]
    zero_shot_metrics = compute_recall_at_k(
        retrieved_candidates=val_results_zero,
        ground_truth=gt_map,
        k_list=eval_k_list,
    )
    print("\n--- ZERO-SHOT BASELINE REPORT ---")
    print_recall_report(zero_shot_metrics)

    # 5. Mine Hard Negatives from Train set using Zero-shot Model
    print("\n[Step 5/6] Mining hard negatives on Train entities using vector retrieval...")
    train_embeddings_zero = base_encoder.encode(train_s1_texts, batch_size=encode_batch_size, is_query=True)
    train_retriever = VectorRetriever(
        corpus_embeddings=corpus_embeddings_zero,
        corpus_ids=corpus_ids,
        corpus_countries=corpus_countries,
        device=device,
    )
    train_retrieval_results = train_retriever.search_top_k(
        query_embeddings=train_embeddings_zero,
        query_ids=s1_train["entity_id"].tolist(),
        query_countries=s1_train["country"].tolist() if "country" in s1_train.columns else None,
        top_k=top_k_negatives,
        partition_by_country=True,
    )

    train_triplets = mine_triplets_from_retrieval(
        s1_df=s1_train,
        corpus_df=corpus_active,
        s1_texts=train_s1_map,
        corpus_texts=corpus_map,
        ground_truth=gt_map,
        retrieval_candidates=train_retrieval_results,
        max_negatives_per_positive=max_negs_per_pos,
        max_triplets_per_anchor=max_triplets_per_anchor,
        fallback_to_random_neg=True,
        random_seed=seed,
    )

    # 6. Fine-tune model with MultipleNegativesRankingLoss
    print("\n[Step 6/6] Fine-tuning Bi-Encoder with MultipleNegativesRankingLoss...")
    fine_tuned_st = fine_tune_bi_encoder(
        train_examples=train_triplets,
        output_path=output_model_path,
        base_model_name=base_model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )

    # 7. Evaluate Fine-Tuned Model on Validation Set
    print("\n[Evaluating Fine-Tuned Model] Re-encoding Validation set with fine-tuned bi-encoder...")
    tuned_encoder = DenseEncoder(model_name_or_path=output_model_path, device=device)
    corpus_embeddings_tuned = tuned_encoder.encode(corpus_texts, batch_size=encode_batch_size, is_query=False)
    val_embeddings_tuned = tuned_encoder.encode(val_s1_texts, batch_size=encode_batch_size, is_query=True)

    val_retriever_tuned = VectorRetriever(
        corpus_embeddings=corpus_embeddings_tuned,
        corpus_ids=corpus_ids,
        corpus_countries=corpus_countries,
        device=device,
    )
    val_results_tuned = val_retriever_tuned.search_top_k(
        query_embeddings=val_embeddings_tuned,
        query_ids=s1_val["entity_id"].tolist(),
        query_countries=s1_val["country"].tolist() if "country" in s1_val.columns else None,
        top_k=top_k_eval,
        partition_by_country=True,
    )

    tuned_metrics = compute_recall_at_k(
        retrieved_candidates=val_results_tuned,
        ground_truth=gt_map,
        k_list=eval_k_list,
    )

    print("\n--- FINE-TUNED MODEL REPORT ---")
    print_recall_report(tuned_metrics)

    # 8. Export Candidate Pairs if Requested
    if export_candidates_path:
        print(f"\n[Export] Saving fine-tuned candidate pairs to {export_candidates_path}...")
        val_retriever_tuned.export_candidate_pairs_tsv(val_results_tuned, export_candidates_path)
        print(f"[Export] Saved {len(val_results_tuned):,} candidate lists successfully.")

    # 9. Print Ticket B2 Acceptance Comparison Table
    print("\n" + "=" * 70)
    print("      TICKET B2 ACCEPTANCE: ZERO-SHOT vs FINE-TUNED RECALL@K       ")
    print("=" * 70)
    print(f"{'K':>5} | {'Zero-Shot Macro':>17} | {'Fine-Tuned Macro':>18} | {'Delta':>10}")
    print("-" * 70)

    z_macro = zero_shot_metrics["macro_recall_at_k"]
    t_macro = tuned_metrics["macro_recall_at_k"]
    deltas = {}

    for k in eval_k_list:
        zm = z_macro[k] * 100
        tm = t_macro[k] * 100
        delta = tm - zm
        deltas[k] = delta
        sign = "+" if delta >= 0 else ""
        print(f"{k:>5} | {zm:>16.2f}% | {tm:>17.2f}% | {sign}{delta:>8.2f}%")

    print("=" * 70)
    elapsed = time.time() - start_time
    print(f"\nTicket B2 workflow completed in {elapsed:.1f}s.")

    return {
        "zero_shot_metrics": zero_shot_metrics,
        "tuned_metrics": tuned_metrics,
        "deltas": deltas,
        "checkpoint_path": output_model_path,
        "export_candidates_path": export_candidates_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ticket B2: Contrastive Bi-Encoder Fine-Tuning")
    parser.add_argument("--sample", type=int, default=300, help="Number of S1 entities (default: 300)")
    parser.add_argument("--full", action="store_true", help="Run on full dataset")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    parser.add_argument("--base-model", type=str, default=DEFAULT_MODEL, help="Base SentenceTransformer model")
    parser.add_argument("--output-model", type=str, default="models/bi_encoder_b2", help="Checkpoint save directory")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument("--encode-batch-size", type=int, default=64, help="Embedding encoding batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--top-k-negatives", type=int, default=20, help="Top-K confusers to retrieve for mining (default: 20)")
    parser.add_argument("--max-negs-per-pos", type=int, default=2, help="Max negatives per positive (default: 2)")
    parser.add_argument("--max-triplets-per-anchor", type=int, default=8, help="Max triplets per anchor (default: 8)")
    parser.add_argument("--k", type=int, default=20, help="Top-K candidates to retrieve for evaluation (default: 20)")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'mps', 'cuda')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-canonicalize", action="store_true", help="Disable canonical text normalization")
    parser.add_argument("--export-candidates", type=str, default=None, help="Path to export candidate_pairs.tsv for B3")
    args = parser.parse_args()

    sample_size = None if args.full else args.sample
    run_b2_pipeline(
        sample_size=sample_size,
        val_fraction=args.val_fraction,
        base_model_name=args.base_model,
        output_model_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        learning_rate=args.lr,
        top_k_negatives=args.top_k_negatives,
        max_negs_per_pos=args.max_negs_per_pos,
        max_triplets_per_anchor=args.max_triplets_per_anchor,
        top_k_eval=args.k,
        canonicalize=not args.no_canonicalize,
        device=args.device,
        seed=args.seed,
        export_candidates_path=args.export_candidates,
    )


if __name__ == "__main__":
    main()

