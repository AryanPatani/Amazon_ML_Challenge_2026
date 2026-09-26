"""
src/path_b/b4_pipeline.py

Ticket B4: Augmentation for Robustness — Leave-One-Country-Out Ablation.

Workflow
--------
1. Load data; identify available train countries.
2. For each leave-out country (or a specified subset):
   a. Train cross-encoder WITHOUT augmentation on all other countries.
   b. Train cross-encoder WITH synthetic noise augmentation on all other countries.
   c. Evaluate both models on the left-out country's validation entities.
   d. Record Macro F0.5 delta: augmented vs. baseline.
3. Print a comprehensive ablation table.
4. Persist the best augmented model to `models/cross_encoder_b4_augmented/`.

Acceptance criterion (ticket B4):
  "Ablation: with vs without augmentation on leave-one-country-out."

Usage
-----
    # Quick test (100 entities, 1 epoch)
    python -m src.path_b.b4_pipeline --sample 100 --epochs 1

    # Full ablation leaving out India (India has enough data to be meaningful)
    python -m src.path_b.b4_pipeline --leave-out India --epochs 2

    # Full multi-country ablation
    python -m src.path_b.b4_pipeline --full --epochs 2
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
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
from src.path_b.cross_metrics import find_optimal_threshold
from src.path_b.augmentation import augment_batch


# ---------------------------------------------------------------------------
# Core per-country evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_on_country(
    reranker: CrossEncoderReranker,
    val_candidates: dict,
    val_s1_map: dict,
    corpus_map: dict,
    gt_map: dict,
    val_s1_ids: list,
    encode_batch_size: int = 64,
    enforce_one_to_one: bool = True,
) -> dict:
    """Score + threshold-tune + return metrics for a single held-out split."""
    val_scored = reranker.rerank_candidates(
        candidate_dict=val_candidates,
        s1_texts=val_s1_map,
        corpus_texts=corpus_map,
        batch_size=encode_batch_size,
        show_progress_bar=False,
    )
    val_gt = {sid: gt_map.get(sid, []) for sid in val_s1_ids}
    result = find_optimal_threshold(
        scored_candidates=val_scored,
        ground_truth=val_gt,
        enforce_one_to_one=enforce_one_to_one,
    )
    return result


# ---------------------------------------------------------------------------
# Single ablation run: train on *train_country* data, test on *test_country*
# ---------------------------------------------------------------------------

def run_loco_ablation(
    leave_out_country: str,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt_map: dict,
    bi_encoder_model: str = DEFAULT_MODEL,
    cross_encoder_model: str = DEFAULT_CROSS_MODEL,
    sample_size: Optional[int] = None,
    epochs: int = 1,
    batch_size: int = 16,
    encode_batch_size: int = 64,
    learning_rate: float = 2e-5,
    top_k_candidates: int = 30,
    max_negs_per_pos: int = 4,
    enforce_one_to_one: bool = True,
    canonicalize: bool = True,
    device: Optional[str] = None,
    seed: int = 42,
    # Augmentation knobs
    n_augments: int = 2,
    aug_min_ops: int = 1,
    aug_max_ops: int = 3,
    output_dir: str = "models",
) -> dict:
    """Train and evaluate with/without augmentation, holding out *leave_out_country*.

    Returns a dict with keys:
        leave_out_country, baseline_f05, augmented_f05, delta_f05,
        baseline_auc, augmented_auc, n_train, n_val
    """
    t0 = time.time()
    print(f"\n{'='*68}")
    print(f"  LOCO ABLATION — held-out country: {leave_out_country!r}")
    print(f"{'='*68}")

    # ---- Split by country -----------------------------------------------
    if "country" not in s1.columns:
        raise ValueError("S1 DataFrame must have a 'country' column for LOCO ablation.")

    s1_train_full = s1[s1["country"] != leave_out_country].copy().reset_index(drop=True)
    s1_val_full   = s1[s1["country"] == leave_out_country].copy().reset_index(drop=True)

    if len(s1_val_full) == 0:
        print(f"[WARNING] No S1 entities found for country '{leave_out_country}'. Skipping.")
        return {}

    # Sub-sample for speed if requested
    if sample_size is not None:
        val_n   = max(20, min(int(sample_size * 0.25), len(s1_val_full)))
        train_n = max(20, min(sample_size - val_n, len(s1_train_full)))
        s1_val   = s1_val_full.head(val_n).reset_index(drop=True)
        s1_train = s1_train_full.head(train_n).reset_index(drop=True)
    else:
        s1_val   = s1_val_full.reset_index(drop=True)
        s1_train = s1_train_full.reset_index(drop=True)

    s1_train = s1_train.drop_duplicates("entity_id").reset_index(drop=True)
    s1_val   = s1_val.drop_duplicates("entity_id").reset_index(drop=True)

    train_ids = set(s1_train["entity_id"])
    val_ids   = set(s1_val["entity_id"])
    all_s1_ids = train_ids | val_ids

    # Build minimal corpus (true matches + confusers)
    true_corpus_ids: set = set()
    for sid in all_s1_ids:
        true_corpus_ids.update(gt_map.get(sid, []))

    confuser_cap = max(5_000, len(all_s1_ids) * 5)
    s2_needed    = s2[s2["entity_id"].isin(true_corpus_ids)]
    s3_needed    = s3[s3["entity_id"].isin(true_corpus_ids)]
    s2_conf      = s2[~s2["entity_id"].isin(true_corpus_ids)].head(confuser_cap)
    s3_conf      = s3[~s3["entity_id"].isin(true_corpus_ids)].head(confuser_cap)
    corpus_df    = pd.concat([s2_needed, s3_needed, s2_conf, s3_conf],
                              ignore_index=True).drop_duplicates("entity_id").reset_index(drop=True)

    print(f"  Train: {len(s1_train):,}  Val: {len(s1_val):,}  Corpus: {len(corpus_df):,}")

    # ---- Serialise -------------------------------------------------------
    train_texts  = serialize_dataframe(s1_train,  drop_country=True, canonicalize=canonicalize)
    val_texts    = serialize_dataframe(s1_val,    drop_country=True, canonicalize=canonicalize)
    corpus_texts = serialize_dataframe(corpus_df, drop_country=True, canonicalize=canonicalize)

    train_s1_map = dict(zip(s1_train["entity_id"], train_texts))
    val_s1_map   = dict(zip(s1_val["entity_id"],   val_texts))
    corpus_map   = dict(zip(corpus_df["entity_id"], corpus_texts))
    corpus_ids   = corpus_df["entity_id"].tolist()

    # ---- Bi-Encoder retrieval (shared between baseline and augmented) ----
    print("  [Retrieval] Encoding corpus...")
    bi_encoder = DenseEncoder(model_name_or_path=bi_encoder_model, device=device)
    corpus_embeddings = bi_encoder.encode(list(corpus_texts), batch_size=encode_batch_size, is_query=False)

    train_embeddings = bi_encoder.encode(train_texts, batch_size=encode_batch_size, is_query=True)
    val_embeddings   = bi_encoder.encode(val_texts,   batch_size=encode_batch_size, is_query=True)

    corpus_countries = corpus_df["country"].tolist() if "country" in corpus_df.columns else None
    train_countries  = s1_train["country"].tolist()  if "country" in s1_train.columns else None
    val_countries    = s1_val["country"].tolist()    if "country" in s1_val.columns else None

    retriever = VectorRetriever(
        corpus_embeddings=corpus_embeddings,
        corpus_ids=corpus_ids,
        corpus_countries=corpus_countries,
        device=device,
    )
    train_candidates = retriever.search_top_k(
        query_embeddings=train_embeddings,
        query_ids=s1_train["entity_id"].tolist(),
        query_countries=train_countries,
        top_k=top_k_candidates,
        partition_by_country=True if corpus_countries and train_countries else False,
    )
    val_candidates = retriever.search_top_k(
        query_embeddings=val_embeddings,
        query_ids=s1_val["entity_id"].tolist(),
        query_countries=val_countries,
        top_k=top_k_candidates,
        partition_by_country=True if corpus_countries and val_countries else False,
    )
    # Bi-encoder recall ceiling
    bi_recall = compute_recall_at_k(val_candidates, gt_map,
                                     k_list=[min(top_k_candidates, 20)])
    print(f"  Bi-Encoder Recall@{min(top_k_candidates,20)}: "
          f"{bi_recall['macro_recall_at_k'].get(min(top_k_candidates,20), 0)*100:.1f}%")
    del bi_encoder, corpus_embeddings, train_embeddings, val_embeddings, retriever

    val_s1_ids = s1_val["entity_id"].tolist()

    # ================================================================
    # RUN 1: BASELINE — no augmentation
    # ================================================================
    print("\n  [Run 1/2] BASELINE (no augmentation)...")
    base_examples = prepare_cross_encoder_examples(
        s1_ids=s1_train["entity_id"].tolist(),
        s1_texts=train_s1_map,
        corpus_texts=corpus_map,
        ground_truth=gt_map,
        retrieval_candidates=train_candidates,
        max_negatives_per_positive=max_negs_per_pos,
        corpus_ids=corpus_ids,
        random_seed=seed,
        augment_positives=False,
    )
    base_model_path = f"{output_dir}/cross_encoder_b4_baseline_{leave_out_country.replace(' ','_').lower()}"
    fine_tune_cross_encoder(
        train_examples=base_examples,
        output_path=base_model_path,
        base_model_name=cross_encoder_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )
    base_reranker = CrossEncoderReranker(model_name_or_path=base_model_path, device=device)
    base_result = _evaluate_on_country(
        reranker=base_reranker,
        val_candidates=val_candidates,
        val_s1_map=val_s1_map,
        corpus_map=corpus_map,
        gt_map=gt_map,
        val_s1_ids=val_s1_ids,
        encode_batch_size=encode_batch_size,
        enforce_one_to_one=enforce_one_to_one,
    )
    del base_reranker

    # ================================================================
    # RUN 2: AUGMENTED — synthetic noise on positives
    # ================================================================
    print(f"\n  [Run 2/2] AUGMENTED (n_augments={n_augments}, ops={aug_min_ops}-{aug_max_ops})...")
    aug_examples = prepare_cross_encoder_examples(
        s1_ids=s1_train["entity_id"].tolist(),
        s1_texts=train_s1_map,
        corpus_texts=corpus_map,
        ground_truth=gt_map,
        retrieval_candidates=train_candidates,
        max_negatives_per_positive=max_negs_per_pos,
        corpus_ids=corpus_ids,
        random_seed=seed,
        augment_positives=True,
        n_augments=n_augments,
        aug_min_ops=aug_min_ops,
        aug_max_ops=aug_max_ops,
        aug_seed=seed + 57,
    )
    aug_model_path = f"{output_dir}/cross_encoder_b4_augmented_{leave_out_country.replace(' ','_').lower()}"
    fine_tune_cross_encoder(
        train_examples=aug_examples,
        output_path=aug_model_path,
        base_model_name=cross_encoder_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )
    aug_reranker = CrossEncoderReranker(model_name_or_path=aug_model_path, device=device)
    aug_result = _evaluate_on_country(
        reranker=aug_reranker,
        val_candidates=val_candidates,
        val_s1_map=val_s1_map,
        corpus_map=corpus_map,
        gt_map=gt_map,
        val_s1_ids=val_s1_ids,
        encode_batch_size=encode_batch_size,
        enforce_one_to_one=enforce_one_to_one,
    )
    del aug_reranker

    # Collect metrics
    base_f05 = base_result.get("best_macro_f05", 0.0)
    aug_f05  = aug_result.get("best_macro_f05", 0.0)
    base_auc = base_result.get("auc_metrics", {}).get("roc_auc", 0.0)
    aug_auc  = aug_result.get("auc_metrics", {}).get("roc_auc", 0.0)

    elapsed = time.time() - t0
    print(f"\n  Results for held-out: {leave_out_country!r} (took {elapsed:.0f}s)")
    print(f"    Baseline  — Macro F0.5: {base_f05*100:.2f}%  ROC-AUC: {base_auc*100:.2f}%")
    print(f"    Augmented — Macro F0.5: {aug_f05*100:.2f}%  ROC-AUC: {aug_auc*100:.2f}%")
    delta = aug_f05 - base_f05
    sign  = "+" if delta >= 0 else ""
    print(f"    Delta (aug - base):     {sign}{delta*100:.2f}%  ({'IMPROVED' if delta >= 0 else 'REGRESSED'})")

    return {
        "leave_out_country": leave_out_country,
        "n_train": len(s1_train),
        "n_val": len(s1_val),
        "n_pairs_baseline": len(base_examples),
        "n_pairs_augmented": len(aug_examples),
        "baseline_f05": base_f05,
        "augmented_f05": aug_f05,
        "delta_f05": aug_f05 - base_f05,
        "baseline_auc": base_auc,
        "augmented_auc": aug_auc,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Full pipeline: iterate over all leave-out countries
# ---------------------------------------------------------------------------

def run_b4_pipeline(
    leave_out_countries: Optional[list] = None,
    sample_size: Optional[int] = None,
    bi_encoder_model: str = DEFAULT_MODEL,
    cross_encoder_model: str = DEFAULT_CROSS_MODEL,
    epochs: int = 1,
    batch_size: int = 16,
    encode_batch_size: int = 64,
    learning_rate: float = 2e-5,
    top_k_candidates: int = 30,
    max_negs_per_pos: int = 4,
    enforce_one_to_one: bool = True,
    canonicalize: bool = True,
    device: Optional[str] = None,
    seed: int = 42,
    n_augments: int = 2,
    aug_min_ops: int = 1,
    aug_max_ops: int = 3,
    output_dir: str = "models",
    report_path: str = "outputs/b4_ablation_report.tsv",
) -> list:
    """Run leave-one-country-out ablation for Ticket B4.

    Parameters
    ----------
    leave_out_countries : list[str], optional
        Countries to hold out one at a time. If None, all unique countries
        in S1 that have at least 50 entities are used.
    sample_size : int, optional
        If set, sub-sample S1 entities (for speed in testing).
    ... (see run_loco_ablation for remaining parameters)

    Returns
    -------
    list[dict]
        One result dict per leave-out country.
    """
    global_start = time.time()
    print("=" * 70)
    print("   TICKET B4: AUGMENTATION FOR ROBUSTNESS (LOCO ABLATION)   ")
    print("=" * 70)
    print(f"Bi-Encoder:      {bi_encoder_model}")
    print(f"Cross-Encoder:   {cross_encoder_model}")
    print(f"Epochs:          {epochs}")
    print(f"Batch Size:      {batch_size} train / {encode_batch_size} inference")
    print(f"Top-K:           {top_k_candidates}")
    print(f"Augmentation:    n_augments={n_augments}, ops={aug_min_ops}-{aug_max_ops}")
    print(f"Sample Size:     {sample_size or 'FULL'}")
    print("-" * 70)

    # Load data
    s1, s2, s3, gt = load_all_sources()
    gt_map: dict = dict(zip(gt["source1_entity_id"], gt["matches"]))

    # Determine countries to ablate
    if isinstance(leave_out_countries, str):
        leave_out_countries = [leave_out_countries]

    if leave_out_countries is None:
        if "country" in s1.columns:
            country_counts = s1["country"].value_counts()
            leave_out_countries = country_counts[country_counts >= 50].index.tolist()
        else:
            print("[WARNING] No 'country' column found — cannot run LOCO ablation.")
            return []

    print(f"Countries to ablate: {leave_out_countries}")

    results = []
    for country in leave_out_countries:
        row = run_loco_ablation(
            leave_out_country=country,
            s1=s1, s2=s2, s3=s3, gt_map=gt_map,
            bi_encoder_model=bi_encoder_model,
            cross_encoder_model=cross_encoder_model,
            sample_size=sample_size,
            epochs=epochs,
            batch_size=batch_size,
            encode_batch_size=encode_batch_size,
            learning_rate=learning_rate,
            top_k_candidates=top_k_candidates,
            max_negs_per_pos=max_negs_per_pos,
            enforce_one_to_one=enforce_one_to_one,
            canonicalize=canonicalize,
            device=device,
            seed=seed,
            n_augments=n_augments,
            aug_min_ops=aug_min_ops,
            aug_max_ops=aug_max_ops,
            output_dir=output_dir,
        )
        if row:
            results.append(row)

    # Save best augmented model as canonical b4 checkpoint
    if results:
        best = max(results, key=lambda r: r.get("augmented_f05", 0.0))
        best_country = best["leave_out_country"].replace(" ", "_").lower()
        best_model_path = Path(output_dir) / f"cross_encoder_b4_augmented_{best_country}"
        canonical_path  = Path(output_dir) / "cross_encoder_b4_augmented"
        if best_model_path.exists():
            import shutil
            if canonical_path.exists():
                shutil.rmtree(canonical_path)
            shutil.copytree(str(best_model_path), str(canonical_path))
            print(f"\n[B4] Best augmented model saved to: {canonical_path}")

    # ── Ablation Table ────────────────────────────────────────────────────
    elapsed_total = time.time() - global_start
    print("\n" + "=" * 72)
    print("               TICKET B4 ABLATION REPORT               ")
    print("=" * 72)
    header = (
        f"{'Country':<12} {'N_train':>8} {'N_val':>6} "
        f"{'Base F0.5':>10} {'Aug F0.5':>10} {'Delta':>8} {'AUC base':>9} {'AUC aug':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        delta_str = f"{'+' if r['delta_f05']>=0 else ''}{r['delta_f05']*100:.2f}%"
        row_str = (
            f"{r['leave_out_country']:<12} "
            f"{r['n_train']:>8,} {r['n_val']:>6,} "
            f"{r['baseline_f05']*100:>9.2f}% "
            f"{r['augmented_f05']*100:>9.2f}% "
            f"{delta_str:>8} "
            f"{r['baseline_auc']*100:>8.2f}% "
            f"{r['augmented_auc']*100:>8.2f}%"
        )
        print(row_str)
    print("=" * 72)

    # Compute averages
    if results:
        avg_base = sum(r["baseline_f05"]  for r in results) / len(results)
        avg_aug  = sum(r["augmented_f05"] for r in results) / len(results)
        avg_delta= avg_aug - avg_base
        print(f"{'Average':<12} {'':>8} {'':>6} "
              f"{avg_base*100:>9.2f}% {avg_aug*100:>9.2f}% "
              f"{('+' if avg_delta>=0 else '')}{avg_delta*100:.2f}%")
        print("=" * 72)

    print(f"Total Elapsed: {elapsed_total:.0f}s")

    # Export report
    if results:
        out_path = Path(report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(str(out_path), sep="\t", index=False)
        print(f"[B4] Ablation report saved to: {out_path}")

    return results


# ---------------------------------------------------------------------------
# Demo helper: show augmentation examples (fast, no GPU needed)
# ---------------------------------------------------------------------------

def demo_augmentations(n_examples: int = 5) -> None:
    """Print *n_examples* augmented variants for a few sample entity texts."""
    from src.path_b.augmentation import corrupt_text, augment_batch
    import random

    samples = [
        "Acme Pvt Ltd | 12 Baker Street 110001 New Delhi",
        "Global Tech Corp | 450 Market St San Francisco 94102 US",
        "Reliance Industries Ltd | BKC Bandra East Mumbai 400051",
    ]
    print("\n" + "="*60)
    print("  B4 AUGMENTATION DEMO")
    print("="*60)
    for text in samples:
        print(f"\nOriginal: {text}")
        variants = augment_batch([text], n_augments=n_examples, seed=42)
        for i, v in enumerate(variants, 1):
            print(f"  [{i}] {v}")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ticket B4: Leave-One-Country-Out Augmentation Ablation"
    )
    parser.add_argument("--sample", type=int, default=200,
                        help="Sub-sample N S1 entities per run (default: 200, use 0 for full)")
    parser.add_argument("--full", action="store_true",
                        help="Use full dataset (overrides --sample)")
    parser.add_argument("--leave-out", type=str, default=None, nargs="+",
                        help="Countries to hold out (default: all with >=50 entities)")
    parser.add_argument("--bi-encoder", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cross-encoder", type=str, default=DEFAULT_CROSS_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--max-negs-per-pos", type=int, default=4)
    parser.add_argument("--n-augments", type=int, default=2,
                        help="Noisy copies per positive pair (default: 2)")
    parser.add_argument("--aug-min-ops", type=int, default=1,
                        help="Min augmentation operations per copy (default: 1)")
    parser.add_argument("--aug-max-ops", type=int, default=3,
                        help="Max augmentation operations per copy (default: 3)")
    parser.add_argument("--no-1to1", action="store_true")
    parser.add_argument("--no-canonicalize", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--report-path", type=str, default="outputs/b4_ablation_report.tsv")
    parser.add_argument("--demo", action="store_true",
                        help="Just print augmentation examples and exit (no training)")
    args = parser.parse_args()

    if args.demo:
        demo_augmentations()
        return

    sample_size = None if args.full else (args.sample if args.sample > 0 else None)
    run_b4_pipeline(
        leave_out_countries=args.leave_out,
        sample_size=sample_size,
        bi_encoder_model=args.bi_encoder,
        cross_encoder_model=args.cross_encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        learning_rate=args.lr,
        top_k_candidates=args.k,
        max_negs_per_pos=args.max_negs_per_pos,
        enforce_one_to_one=not args.no_1to1,
        canonicalize=not args.no_canonicalize,
        device=args.device,
        seed=args.seed,
        n_augments=args.n_augments,
        aug_min_ops=args.aug_min_ops,
        aug_max_ops=args.aug_max_ops,
        output_dir=args.output_dir,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
